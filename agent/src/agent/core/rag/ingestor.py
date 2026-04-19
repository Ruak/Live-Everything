"""Knowledge ingestion for product, semantic-category, and fallback scopes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .chunker import Chunker
from .vector_store import VectorStore
from ... import config
from ..knowledge_base import fallback_collection_id, semantic_collection_id

logger = logging.getLogger(__name__)


class RAGIngestor:
    """Ingest various knowledge sources into the vector store."""

    def __init__(self, vector_store: VectorStore, chunker: Optional[Chunker] = None):
        self.store = vector_store
        self.chunker = chunker or Chunker()

    def ingest_product_json(self, data: dict) -> int:
        product_id = data.get("product_id")
        if not product_id:
            logger.warning("Product JSON missing product_id, skipping")
            return 0
        chunks = self.chunker.chunk_product_json(data, product_id)
        return self.store.add_chunks(product_id, chunks)

    def ingest_products_payload(self, data: object) -> list[str]:
        items = data if isinstance(data, list) else [data]
        ingested: list[str] = []

        for item in items:
            if not isinstance(item, dict):
                continue
            count = self.ingest_product_json(item)
            if count > 0:
                ingested.append(item.get("product_id", ""))
        return ingested

    def ingest_products_file(self, path: Path) -> list[str]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)
            return []
        return self.ingest_products_payload(data)

    def ingest_markdown(
        self,
        path: Path,
        product_id: str,
        metadata: Optional[dict] = None,
    ) -> int:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)
            return 0

        chunks = self.chunker.chunk_markdown(
            text,
            source=str(path.name),
            metadata={**(metadata or {}), "product_id": product_id},
        )
        return self.store.add_chunks(product_id, chunks)

    def ingest_text(
        self,
        path: Path,
        product_id: str,
        category: str = "general",
        metadata: Optional[dict] = None,
    ) -> int:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)
            return 0

        chunks = self.chunker.chunk_text(
            text,
            source=str(path.name),
            category=category,
            metadata={**(metadata or {}), "product_id": product_id},
        )
        return self.store.add_chunks(product_id, chunks)

    def ingest_raw_text(
        self,
        text: str,
        product_id: str,
        source: str = "api",
        category: str = "general",
        metadata: Optional[dict] = None,
    ) -> int:
        chunks = self.chunker.chunk_text(
            text,
            source=source,
            category=category,
            metadata={**(metadata or {}), "product_id": product_id},
        )
        return self.store.add_chunks(product_id, chunks)

    def ingest_rich_knowledge_base(self, data: dict) -> dict:
        """Ingest semantic-category and fallback knowledge from a structured KB."""
        total_chunks = 0
        products: list[str] = []
        categories: list[str] = []
        fallbacks: list[str] = []

        for product in data.get("products", []):
            if not isinstance(product, dict):
                continue
            count = self.ingest_rich_product(product)
            if count > 0:
                products.append(product.get("product_id", ""))
                total_chunks += count

        for category in data.get("semantic_categories", []):
            if not isinstance(category, dict):
                continue
            count = self.ingest_semantic_category(category)
            if count > 0:
                categories.append(category.get("semantic_category_id", ""))
                total_chunks += count

        fallback = data.get("generic_fallback")
        if isinstance(fallback, dict):
            count = self.ingest_generic_fallback(fallback)
            if count > 0:
                fallbacks.append(fallback.get("semantic_category_id", "generic_object"))
                total_chunks += count

        return {
            "products": [item for item in products if item],
            "categories": [item for item in categories if item],
            "fallbacks": [item for item in fallbacks if item],
            "total_chunks": total_chunks,
        }

    def ingest_rich_product(self, data: dict) -> int:
        product_id = data.get("product_id")
        if not product_id:
            return 0

        product_name = data.get("product_name", product_id)
        metadata = {
            "product_id": product_id,
            "product_name": product_name,
            "display_name": product_name,
            "scope": "product",
        }
        texts: list[tuple[str, str]] = []

        def add(category: str, text: str) -> None:
            clean = text.strip()
            if clean:
                texts.append((category, clean))

        for key in ("tagline", "one_line_hook", "self_intro_short", "self_intro_medium", "story_monologue_90s"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                add(key, value)

        for item in data.get("selling_points", []):
            if isinstance(item, dict):
                text = "\n".join(
                    part
                    for part in (item.get("title", ""), item.get("detail", ""), item.get("scene_value", ""))
                    if str(part).strip()
                )
                add("selling_point", text)
            elif isinstance(item, str):
                add("selling_point", item)

        specs = data.get("specs", [])
        if isinstance(specs, dict):
            for key, value in specs.items():
                add("spec", f"{key}：{value}")
        else:
            for item in specs:
                if isinstance(item, dict):
                    add("spec", f"{item.get('name', '参数')}：{item.get('value', '')}")

        for category_name in ("audience", "use_cases", "limitations", "care_tips", "common_misunderstandings"):
            for item in data.get(category_name, []):
                if isinstance(item, str):
                    add(category_name, item)

        for step in data.get("guided_demo_script", []):
            if isinstance(step, dict):
                add("guided_demo_script", f"{step.get('step', '步骤')}：{step.get('line', '')}")

        for question_type, answer in data.get("question_type_answers", {}).items():
            add(f"question_type_answer/{question_type}", str(answer))

        for faq in data.get("faq", []):
            if isinstance(faq, dict):
                question = str(faq.get("question", "")).strip()
                answer = str(faq.get("answer", "")).strip()
                if question and answer:
                    add("faq", f"问：{question}\n答：{answer}")

        chunks = []
        for index, (category, text) in enumerate(texts):
            chunk_list = self.chunker.chunk_text(
                text,
                source=product_id,
                category=category,
                metadata=metadata,
            )
            for chunk in chunk_list:
                chunk.chunk_index = len(chunks)
                chunks.append(chunk)

        return self.store.add_chunks(product_id, chunks)

    def ingest_semantic_category(self, data: dict) -> int:
        category_id = data.get("semantic_category_id")
        if not category_id:
            return 0

        display_name = data.get("display_name", category_id)
        collection_id = semantic_collection_id(category_id)
        metadata = {
            "semantic_category_id": category_id,
            "display_name": display_name,
            "scope": "semantic_category",
        }

        texts: list[tuple[str, str]] = []

        def add(category: str, text: str) -> None:
            clean = text.strip()
            if clean:
                texts.append((category, clean))

        for key in ("category_pitch",):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                add(key, value)

        for list_key in (
            "common_roles",
            "common_features",
            "common_scenes",
            "safe_claim_rules",
            "forbidden_claim_rules",
        ):
            for item in data.get(list_key, []):
                if isinstance(item, str):
                    add(list_key, item)

        for key, value in data.get("generic_answer_templates", {}).items():
            add(f"generic_answer_template/{key}", str(value))

        chunks = []
        for category, text in texts:
            chunk_list = self.chunker.chunk_text(
                text,
                source=category_id,
                category=category,
                metadata=metadata,
            )
            for chunk in chunk_list:
                chunk.chunk_index = len(chunks)
                chunks.append(chunk)

        return self.store.add_chunks(collection_id, chunks)

    def ingest_generic_fallback(self, data: dict) -> int:
        category_id = data.get("semantic_category_id", "generic_object")
        collection_id = fallback_collection_id(category_id)
        metadata = {
            "semantic_category_id": category_id,
            "display_name": "通用兜底知识",
            "scope": "fallback",
        }

        texts = [
            (f"generic_fallback/{key}", str(value))
            for key, value in data.get("answer_templates", {}).items()
            if str(value).strip()
        ]

        chunks = []
        for category, text in texts:
            chunk_list = self.chunker.chunk_text(
                text,
                source=category_id,
                category=category,
                metadata=metadata,
            )
            for chunk in chunk_list:
                chunk.chunk_index = len(chunks)
                chunks.append(chunk)

        return self.store.add_chunks(collection_id, chunks)

    def ingest_rich_kb_from_files(self) -> dict:
        """One-shot ingestion of the entire rich knowledge base from config paths.
        Loads core.json, categories.json, and products/custom/*.json into RAG.
        Returns summary dict."""
        result: dict = {"products": [], "categories": [], "fallbacks": [], "total_chunks": 0}

        # 1) Custom products from KB_CUSTOM_PRODUCTS_DIR
        prod_dir = config.KB_CUSTOM_PRODUCTS_DIR
        if prod_dir.exists():
            for f in sorted(prod_dir.glob("*.json")):
                if f.name.startswith("_"):
                    continue
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.error("Failed to read %s: %s", f, e)
                    continue
                count = self.ingest_rich_product(data)
                if count > 0:
                    pid = data.get("product_id", "")
                    result["products"].append(pid)
                    result["total_chunks"] += count

        # 2) Semantic categories from categories.json
        if config.KB_CATEGORIES_FILE.exists():
            try:
                cat_data = json.loads(config.KB_CATEGORIES_FILE.read_text(encoding="utf-8"))
                for cat in cat_data.get("semantic_categories", []):
                    if not isinstance(cat, dict):
                        continue
                    count = self.ingest_semantic_category(cat)
                    if count > 0:
                        result["categories"].append(cat.get("semantic_category_id", ""))
                        result["total_chunks"] += count
            except Exception as e:
                logger.error("Failed to load categories: %s", e)

        # 3) Fallback from core.json
        if config.KB_CORE_FILE.exists():
            try:
                core_data = json.loads(config.KB_CORE_FILE.read_text(encoding="utf-8"))
                fallback = core_data.get("generic_fallback")
                if isinstance(fallback, dict):
                    count = self.ingest_generic_fallback(fallback)
                    if count > 0:
                        result["fallbacks"].append(fallback.get("semantic_category_id", "generic_object"))
                        result["total_chunks"] += count
            except Exception as e:
                logger.error("Failed to load core fallback: %s", e)

        logger.info("Rich KB file ingestion complete: %s", result)
        return result
