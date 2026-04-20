"""LLM provider abstraction — supports Ollama, OpenAI-compatible, and rule-based fallback."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx

from ..models.message import Message, MessageRole
from .. import config

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        knowledge_context: str,
        history: List[Message],
        query: str,
        temperature: float = 0.7,
    ) -> str:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...


class OllamaProvider(LLMProvider):
    """Local Ollama LLM."""

    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model: str = config.OLLAMA_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)

    async def generate(
        self,
        system_prompt: str,
        knowledge_context: str,
        history: List[Message],
        query: str,
        temperature: float = 0.7,
    ) -> str:
        messages = self._build_messages(system_prompt, knowledge_context, history, query)

        try:
            resp = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            return ""

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _build_messages(
        system_prompt: str,
        knowledge_context: str,
        history: List[Message],
        query: str,
    ) -> list[dict]:
        msgs: list[dict] = []

        # System
        system_parts = [system_prompt]
        if knowledge_context:
            system_parts.append(knowledge_context)
        msgs.append({"role": "system", "content": "\n\n".join(system_parts)})

        # History
        for m in history:
            role = "user" if m.role == MessageRole.USER else "assistant"
            msgs.append({"role": role, "content": m.content})

        # Current query
        msgs.append({"role": "user", "content": query})
        return msgs


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible API (vLLM, LM Studio, etc.)."""

    def __init__(
        self,
        base_url: str = config.OPENAI_BASE_URL,
        api_key: str = config.OPENAI_API_KEY,
        model: str = config.OPENAI_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    async def generate(
        self,
        system_prompt: str,
        knowledge_context: str,
        history: List[Message],
        query: str,
        temperature: float = 0.7,
    ) -> str:
        messages = OllamaProvider._build_messages(
            system_prompt, knowledge_context, history, query
        )
        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("OpenAI-compatible generation failed: %s", e)
            return ""

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get(f"{self.base_url}/models")
            return resp.status_code == 200
        except Exception:
            return False


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek Chat / Reasoner —— 与 OpenAI Chat Completions 完全兼容。

    默认走 https://api.deepseek.com/v1 + deepseek-chat；
    API Key 从 env `DEEPSEEK_API_KEY` 读取。
    """

    def __init__(
        self,
        base_url: str = config.DEEPSEEK_BASE_URL,
        api_key: str = config.DEEPSEEK_API_KEY,
        model: str = config.DEEPSEEK_MODEL,
    ):
        if not api_key:
            logger.warning(
                "DEEPSEEK_API_KEY is empty; DeepSeek calls will 401. "
                "Set it via agent/.env"
            )
        super().__init__(base_url=base_url, api_key=api_key, model=model)

    async def health_check(self) -> bool:
        """DeepSeek /models 支持 GET；空 key 也能触发 401，从而快速判断。"""
        try:
            resp = await self.client.get(f"{self.base_url}/models")
            # 200 = OK；401 = key 坏；都说明端点可达
            return resp.status_code in (200, 401)
        except Exception:
            return False


class RuleBasedProvider(LLMProvider):
    """Fallback: no LLM, purely knowledge-retrieval + template answers."""

    async def generate(
        self,
        system_prompt: str,
        knowledge_context: str,
        history: List[Message],
        query: str,
        temperature: float = 0.7,
    ) -> str:
        if knowledge_context:
            # Extract answer lines from knowledge context
            lines = knowledge_context.strip().split("\n")
            answer_parts = [l.lstrip("- ") for l in lines if l.startswith("- ")]
            if answer_parts:
                # If FAQ, extract the answer part
                for part in answer_parts:
                    if "答：" in part:
                        return part.split("答：", 1)[1].strip()
                return "\n".join(answer_parts)
        return "当前本地资料未覆盖该问题，请尝试换一种方式提问。"

    async def health_check(self) -> bool:
        return True


def create_llm_provider(provider_type: Optional[str] = None) -> LLMProvider:
    """Factory to create the configured LLM provider."""
    pt = provider_type or config.LLM_PROVIDER

    if pt == "deepseek":
        return DeepSeekProvider()
    if pt == "ollama":
        return OllamaProvider()
    if pt == "openai_compatible":
        return OpenAICompatibleProvider()
    if pt == "rule":
        return RuleBasedProvider()
    logger.warning("Unknown LLM provider '%s', falling back to rule-based", pt)
    return RuleBasedProvider()
