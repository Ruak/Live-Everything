"""RAG retriever for product, semantic-category, and fallback knowledge."""

from __future__ import annotations

import logging
from typing import List, Optional

from .vector_store import RetrievalResult, VectorStore
from ... import config
from ..knowledge_base import fallback_collection_id, semantic_collection_id

logger = logging.getLogger(__name__)


class RAGRetriever:
    """High-level retrieval interface for the agent system."""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def retrieve(
        self,
        product_id: str,
        query: str,
        top_k: int = config.RAG_TOP_K,
        score_threshold: float = config.RAG_SCORE_THRESHOLD,
        category_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        return self.store.query(
            product_id=product_id,
            query_text=query,
            top_k=top_k,
            score_threshold=score_threshold,
            category_filter=category_filter,
        )

    def retrieve_global(
        self,
        query: str,
        product_ids: Optional[List[str]] = None,
        top_k: int = config.RAG_TOP_K,
    ) -> List[RetrievalResult]:
        return self.store.query_global(
            query_text=query,
            product_ids=product_ids,
            top_k=top_k,
        )

    def build_context(
        self,
        product_id: str,
        product_name: str,
        query: str,
        top_k: int = config.RAG_TOP_K,
        score_threshold: float = config.RAG_SCORE_THRESHOLD,
    ) -> str:
        results = self.retrieve(
            product_id=product_id,
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        if not results:
            return ""

        parts = [f"以下是关于“{product_name or product_id}”的相关知识："]
        seen_texts: set[str] = set()

        for result in results:
            text_key = result.chunk_text[:100]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            parts.append(f"- [{result.category or 'general'}] {result.chunk_text}")

        return "\n".join(parts)

    def build_layered_context(
        self,
        query: str,
        product_id: str = "",
        product_name: str = "",
        semantic_category_id: str = "",
        semantic_category_name: str = "",
        object_label: str = "",
        top_k: int = config.RAG_TOP_K,
        score_threshold: float = config.RAG_SCORE_THRESHOLD,
    ) -> str:
        """Build context across specific object, semantic category, and generic fallback."""
        sections: list[str] = []
        seen: set[str] = set()

        def append_section(title: str, results: List[RetrievalResult]) -> None:
            filtered: list[RetrievalResult] = []
            for result in results:
                key = result.chunk_text[:120]
                if key in seen:
                    continue
                seen.add(key)
                filtered.append(result)

            if not filtered:
                return

            sections.append(f"[{title}]")
            for result in filtered:
                source_label = (
                    result.metadata.get("display_name")
                    or result.metadata.get("product_name")
                    or result.source
                )
                label = result.category or "general"
                sections.append(f"- ({source_label} / {label}) {result.chunk_text}")

        if product_id:
            append_section(
                f"当前识别物体专属知识：{product_name or product_id}",
                self.retrieve(
                    product_id=product_id,
                    query=query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                ),
            )

        if semantic_category_id:
            append_section(
                f"该类物体的通用知识：{semantic_category_name or semantic_category_id}",
                self.retrieve(
                    product_id=semantic_collection_id(semantic_category_id),
                    query=query,
                    top_k=max(2, min(top_k, 4)),
                    score_threshold=max(score_threshold, 2.2),
                ),
            )

        append_section(
            "通用兜底知识",
            self.retrieve(
                product_id=fallback_collection_id(),
                query=query,
                top_k=2,
                score_threshold=max(score_threshold, 2.5),
            ),
        )

        if not sections:
            return ""

        header = [
            "以下是当前回答可用的本地知识，请严格按层级使用：",
            f"- 当前物体：{product_name or object_label or '未知'}",
        ]
        if semantic_category_id:
            header.append(f"- 物体类别：{semantic_category_name or semantic_category_id}")
        header.append("- 如果只能引用类别知识或兜底知识，必须明确这是通用判断。")

        return "\n".join([*header, *sections])

    def build_multi_product_context(
        self,
        query: str,
        product_ids: List[str],
        top_k: int = config.RAG_TOP_K,
    ) -> str:
        results = self.retrieve_global(
            query=query,
            product_ids=product_ids,
            top_k=top_k,
        )
        if not results:
            return ""

        parts = ["以下是相关对象的知识："]
        for result in results:
            product_name = result.metadata.get("product_name", result.source)
            parts.append(f"- [{product_name} / {result.category}] {result.chunk_text}")
        return "\n".join(parts)
