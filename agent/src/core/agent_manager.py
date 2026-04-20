"""Agent lifecycle manager with layered knowledge-base and RAG support."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from .. import config
from ..models.agent import AgentConfig, AgentState, AgentSummary
from ..models.knowledge import ProductKnowledge, product_json_to_knowledge
from ..models.message import AgentStatus, Message, MessageRole
from .knowledge_base import RichKnowledgeBase
from .knowledge_store import KnowledgeStore
from .llm_provider import LLMProvider, create_llm_provider
from .rag.ingestor import RAGIngestor
from .rag.retriever import RAGRetriever
from .rag.vector_store import VectorStore
from .stt_provider import STTProvider, create_stt_provider
from .web_search import WebSearcher, build_web_context

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages multiple concurrent agents, each bound to a knowledge target."""

    def __init__(self) -> None:
        self._agents: Dict[str, AgentState] = {}
        self.knowledge_store = KnowledgeStore()
        self.rich_knowledge = RichKnowledgeBase()
        self.llm: LLMProvider = create_llm_provider()
        self.stt: STTProvider = create_stt_provider()
        self.web: Optional[WebSearcher] = WebSearcher() if config.WEB_SEARCH_ENABLED else None

        # Label mapping: detection_id → {en, zh, semantic_category_id, custom_product_id, ...}
        self._label_mapping: Dict[str, dict] = {}

        if config.RAG_ENABLED:
            self.vector_store = VectorStore()
            self.retriever = RAGRetriever(self.vector_store)
            self.ingestor = RAGIngestor(self.vector_store)
            logger.info("RAG system initialized")
        else:
            self.vector_store = None
            self.retriever = None
            self.ingestor = None

    def create_agent(
        self,
        product_id: str = "",
        semantic_category_id: str = "",
        object_label: str = "",
        agent_config: Optional[AgentConfig] = None,
    ) -> AgentState:
        if len(self._agents) >= config.MAX_AGENTS:
            raise RuntimeError(f"Max agents ({config.MAX_AGENTS}) reached")
        if not any([product_id, semantic_category_id, object_label]):
            raise RuntimeError("Missing knowledge target")

        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        knowledge = self.knowledge_store.get(product_id) if product_id else None
        resolved = self.rich_knowledge.resolve_target(
            product_id=product_id,
            semantic_category_id=semantic_category_id,
            object_label=object_label,
        )

        system_prompt = (
            agent_config.system_prompt if agent_config and agent_config.system_prompt
            else config.DEFAULT_SYSTEM_PROMPT
        )
        max_history = (
            agent_config.max_history if agent_config else config.MAX_HISTORY_PER_AGENT
        )
        temperature = (
            agent_config.temperature if agent_config else config.DEFAULT_TEMPERATURE
        )

        agent = AgentState(
            agent_id=agent_id,
            product_id=product_id,
            product_name=resolved.product_name
            or (knowledge.product_name if knowledge else "")
            or resolved.display_name,
            semantic_category_id=resolved.semantic_category_id,
            object_label=object_label,
            knowledge=knowledge,
            system_prompt=system_prompt,
            max_history=max_history,
            temperature=temperature,
        )
        self._agents[agent_id] = agent
        logger.info(
            "Created agent %s (product=%s semantic=%s object=%s)",
            agent_id,
            product_id,
            resolved.semantic_category_id,
            object_label,
        )
        return agent

    def destroy_agent(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info("Destroyed agent %s", agent_id)
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[AgentState]:
        return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentSummary]:
        return [
            AgentSummary(
                agent_id=agent.agent_id,
                product_id=agent.product_id,
                product_name=agent.product_name,
                semantic_category_id=agent.semantic_category_id,
                object_label=agent.object_label,
                status=agent.status,
                message_count=len(agent.history),
                created_at=agent.created_at,
            )
            for agent in self._agents.values()
        ]

    def inject_knowledge(self, agent_id: str, knowledge: ProductKnowledge) -> bool:
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        agent.knowledge = knowledge
        agent.product_id = knowledge.product_id
        agent.product_name = knowledge.product_name
        self.knowledge_store.inject(knowledge.product_id, knowledge)
        logger.info(
            "Injected knowledge into agent %s (product=%s entries=%d)",
            agent_id,
            knowledge.product_id,
            len(knowledge.entries),
        )
        return True

    def inject_knowledge_from_dict(self, agent_id: str, data: dict) -> bool:
        knowledge = product_json_to_knowledge(data)
        ok = self.inject_knowledge(agent_id, knowledge)
        if ok and self.ingestor:
            self.ingestor.ingest_product_json(data)
        return ok

    async def ask(self, agent_id: str, query: str) -> str:
        agent = self._agents.get(agent_id)
        if not agent:
            return "Agent not found"

        agent.status = AgentStatus.THINKING
        user_msg = Message(
            role=MessageRole.USER,
            content=query,
            agent_id=agent_id,
            product_id=agent.product_id or None,
        )
        agent.add_message(user_msg)

        resolved = self.rich_knowledge.resolve_target(
            product_id=agent.product_id,
            semantic_category_id=agent.semantic_category_id,
            object_label=agent.object_label or agent.product_name,
        )

        knowledge_ctx = ""
        if self.retriever:
            knowledge_ctx = self.retriever.build_layered_context(
                query=query,
                product_id=agent.product_id,
                product_name=agent.product_name,
                semantic_category_id=resolved.semantic_category_id,
                semantic_category_name=resolved.semantic_category_name,
                object_label=resolved.object_label,
            )

        if not knowledge_ctx and agent.knowledge:
            knowledge_ctx = agent.build_knowledge_context(query)

        if not knowledge_ctx:
            knowledge_ctx = self.rich_knowledge.build_keyword_context(query, resolved)

        # 若本地知识不足，尝试联网补充
        if self.web and self._needs_web_augment(knowledge_ctx):
            web_query = self._build_web_query(query, resolved)
            try:
                web_results = await self.web.search(web_query)
            except Exception as exc:
                logger.warning("Web search failed: %s", exc)
                web_results = []
            if web_results:
                web_block = build_web_context(web_query, web_results)
                knowledge_ctx = (
                    f"{knowledge_ctx}\n\n{web_block}" if knowledge_ctx else web_block
                )
                logger.info(
                    "Augmented context with %d web results for query=%r",
                    len(web_results),
                    web_query,
                )

        effective_system_prompt = agent.system_prompt
        policy_prompt = self.rich_knowledge.build_policy_prompt(resolved)
        if policy_prompt:
            effective_system_prompt = f"{effective_system_prompt}\n\n{policy_prompt}"

        answer = await self.llm.generate(
            system_prompt=effective_system_prompt,
            knowledge_context=knowledge_ctx,
            history=agent.get_context_window()[:-1],
            query=query,
            temperature=agent.temperature,
        )

        if not answer:
            answer = "当前本地知识库没有覆盖这个问题，请换个问法，或补充更具体的物体知识。"

        agent.add_message(
            Message(
                role=MessageRole.AGENT,
                content=answer,
                agent_id=agent_id,
                product_id=agent.product_id or None,
            )
        )
        agent.status = AgentStatus.IDLE
        return answer

    async def multi_ask(self, agent_ids: List[str], query: str) -> Dict[str, str]:
        tasks = {
            agent_id: self.ask(agent_id, query)
            for agent_id in agent_ids
            if agent_id in self._agents
        }
        if not tasks:
            return {}

        results: Dict[str, str] = {}
        done = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for agent_id, result in zip(tasks.keys(), done):
            if isinstance(result, Exception):
                logger.error("Multi-ask error for agent %s: %s", agent_id, result)
                results[agent_id] = f"回答失败：{result}"
            else:
                results[agent_id] = result  # type: ignore[assignment]
        return results

    async def ask_audio(
        self,
        agent_id: str,
        audio_bytes: bytes,
        mime_type: str = "audio/webm",
    ) -> tuple[str, str]:
        text = await self.stt.transcribe(audio_bytes, mime_type)
        if not text:
            return "", "未能识别语音内容，请重试。"
        answer = await self.ask(agent_id, text)
        return text, answer

    # ── Web search helpers ─────────────────────────────────────

    @staticmethod
    def _needs_web_augment(knowledge_ctx: str) -> bool:
        """判断本地知识是否不足 → 需要联网。

        标准：没有任何以 ``- `` 起头的知识条目，说明 RAG/关键字都没召回。
        """
        if not knowledge_ctx:
            return True
        for line in knowledge_ctx.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("- "):
                return False
        return True

    @staticmethod
    def _build_web_query(query: str, resolved) -> str:
        """用物体名 + 用户问题拼一个更贴题的搜索词。"""
        anchors: list[str] = []
        for key in (
            getattr(resolved, "product_name", ""),
            getattr(resolved, "semantic_category_name", ""),
            getattr(resolved, "object_label", ""),
        ):
            if key and key not in anchors:
                anchors.append(key)
        prefix = " ".join(anchors[:2]).strip()
        return f"{prefix} {query}".strip() if prefix else query.strip()

    def ingest_product(self, data: dict) -> int:
        product_id = data.get("product_id", "")
        if product_id:
            self.knowledge_store.inject_from_dict(data)
        if self.ingestor:
            return self.ingestor.ingest_product_json(data)
        return 0

    def ingest_text(
        self,
        product_id: str,
        text: str,
        source: str = "api",
        category: str = "general",
    ) -> int:
        if self.ingestor:
            return self.ingestor.ingest_raw_text(text, product_id, source, category)
        return 0

    def ingest_rich_kb(self) -> dict:
        """Ingest the structured rich knowledge base (products, categories, fallback) into RAG."""
        if self.ingestor:
            return self.ingestor.ingest_rich_kb_from_files()
        return {"products": [], "categories": [], "fallbacks": [], "total_chunks": 0}

    # ── Rich Knowledge Base loading ──────────────────────────────

    def load_rich_knowledge_base(self) -> dict:
        """Load the entire rich knowledge base: core config, categories,
        personas, label mapping, and custom products into RichKnowledgeBase.
        Returns a summary dict."""
        summary: Dict[str, Any] = {"core": False, "categories": 0, "personas": 0, "products": 0, "label_mapping": 0}

        # 1) Core config (design principles, response policy, fallback)
        if config.KB_CORE_FILE.exists():
            try:
                core_data = json.loads(config.KB_CORE_FILE.read_text(encoding="utf-8"))
                self.rich_knowledge.load_from_dict(core_data)
                summary["core"] = True
            except Exception as e:
                logger.error("Failed to load core.json: %s", e)

        # 2) Categories
        if config.KB_CATEGORIES_FILE.exists():
            try:
                cat_data = json.loads(config.KB_CATEGORIES_FILE.read_text(encoding="utf-8"))
                self.rich_knowledge.load_from_dict(cat_data)
                summary["categories"] = len(self.rich_knowledge.semantic_categories)
            except Exception as e:
                logger.error("Failed to load categories.json: %s", e)

        # 3) Personas (stored for reference, not directly used in RichKnowledgeBase yet)
        if config.KB_PERSONAS_FILE.exists():
            try:
                personas_data = json.loads(config.KB_PERSONAS_FILE.read_text(encoding="utf-8"))
                profiles = personas_data.get("persona_profiles", [])
                summary["personas"] = len(profiles)
            except Exception as e:
                logger.error("Failed to load personas.json: %s", e)

        # 4) Custom products
        if config.KB_CUSTOM_PRODUCTS_DIR.exists():
            for f in sorted(config.KB_CUSTOM_PRODUCTS_DIR.glob("*.json")):
                if f.name.startswith("_"):
                    continue
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    pid = data.get("product_id")
                    if pid:
                        self.rich_knowledge.products[pid] = data
                        summary["products"] += 1
                except Exception as e:
                    logger.error("Failed to load product file %s: %s", f, e)

        # 5) Label mapping
        summary["label_mapping"] = self.load_label_mapping()

        logger.info("Rich KB loaded: %s", summary)
        return summary

    # ── Label Mapping ────────────────────────────────────────────

    def load_label_mapping(self) -> int:
        """Load COCO-80 label mapping from config. Returns count of labels loaded."""
        if not config.KB_LABEL_MAPPING_FILE.exists():
            logger.warning("Label mapping file not found: %s", config.KB_LABEL_MAPPING_FILE)
            return 0
        try:
            data = json.loads(config.KB_LABEL_MAPPING_FILE.read_text(encoding="utf-8"))
            self._label_mapping = data.get("labels", {})
            logger.info("Loaded label mapping: %d labels", len(self._label_mapping))
            return len(self._label_mapping)
        except Exception as e:
            logger.error("Failed to load label mapping: %s", e)
            return 0

    def resolve_detection_label(
        self,
        detection_id: Optional[Union[int, str]] = None,
        label_en: str = "",
    ) -> dict:
        """Resolve a YOLO detection ID or English label to knowledge target info.
        Returns dict with keys: product_id, semantic_category_id, object_label, zh, en, baike_query."""
        entry: Optional[dict] = None

        # Try by detection_id first
        if detection_id is not None:
            entry = self._label_mapping.get(str(detection_id))

        # Fallback: search by English label
        if not entry and label_en:
            for _id, mapping in self._label_mapping.items():
                if mapping.get("en", "").lower() == label_en.lower():
                    entry = mapping
                    break

        if not entry:
            return {
                "product_id": "",
                "semantic_category_id": "generic_object",
                "object_label": label_en or str(detection_id or ""),
                "zh": label_en,
                "en": label_en,
                "baike_query": label_en,
            }

        return {
            "product_id": entry.get("custom_product_id") or "",
            "semantic_category_id": entry.get("semantic_category_id", "generic_object"),
            "object_label": entry.get("en", label_en),
            "zh": entry.get("zh", ""),
            "en": entry.get("en", label_en),
            "baike_query": entry.get("baike_query", ""),
        }

    def get_label_mapping(self) -> Dict[str, dict]:
        """Return the full label mapping dict."""
        return dict(self._label_mapping)

    async def health(self) -> dict:
        llm_ok = await self.llm.health_check()
        stt_ok = await self.stt.health_check()
        rag_stats = self.vector_store.global_stats() if self.vector_store else None
        web_ok: Optional[bool] = None
        if self.web is not None:
            try:
                web_ok = await self.web.health_check()
            except Exception:
                web_ok = False
        return {
            "agents": len(self._agents),
            "knowledge_products": self.knowledge_store.count,
            "rich_products": len(self.rich_knowledge.products),
            "semantic_categories": len(self.rich_knowledge.semantic_categories),
            "label_mapping": len(self._label_mapping),
            "rich_files": self.rich_knowledge.loaded_files,
            "llm_provider": config.LLM_PROVIDER,
            "llm_model": self._llm_model_name(),
            "llm_healthy": llm_ok,
            "stt_provider": config.STT_PROVIDER,
            "stt_healthy": stt_ok,
            "rag_enabled": config.RAG_ENABLED,
            "rag_stats": rag_stats,
            "web_search_enabled": bool(self.web),
            "web_search_provider": config.WEB_SEARCH_PROVIDER if self.web else None,
            "web_search_healthy": web_ok,
        }

    def _llm_model_name(self) -> str:
        model = getattr(self.llm, "model", "")
        return model or ""
