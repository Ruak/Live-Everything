"""ChromaDB-based vector store for RAG."""

import logging
from typing import List, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .chunker import Chunk
from ... import config

logger = logging.getLogger(__name__)


class RetrievalResult:
    """A single retrieval result with score."""
    __slots__ = ("chunk_text", "source", "category", "metadata", "distance")

    def __init__(
        self,
        chunk_text: str,
        source: str,
        category: str,
        metadata: dict,
        distance: float,
    ):
        self.chunk_text = chunk_text
        self.source = source
        self.category = category
        self.metadata = metadata
        self.distance = distance

    def __repr__(self) -> str:
        return f"RetrievalResult(source={self.source!r}, cat={self.category!r}, dist={self.distance:.3f})"


class VectorStore:
    """Thin wrapper around ChromaDB for chunk storage and retrieval."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_prefix: str = config.RAG_COLLECTION_PREFIX,
    ):
        self.persist_dir = persist_dir or config.RAG_PERSIST_DIR
        self.collection_prefix = collection_prefix

        # Ensure persist directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB initialized at %s", self.persist_dir)

    # ── Collection management ───────────────────────────────────

    def _collection_name(self, product_id: str) -> str:
        """Sanitize collection name for ChromaDB (3-63 chars, alphanumeric + _-)."""
        name = f"{self.collection_prefix}{product_id}"
        # ChromaDB requires 3-63 chars, start/end with alphanum
        name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        name = name.strip("_-") or "default"
        if len(name) < 3:
            name = name + "_col"
        return name[:63]

    def get_or_create_collection(self, product_id: str):
        name = self._collection_name(product_id)
        return self._client.get_or_create_collection(name=name)

    def delete_collection(self, product_id: str) -> bool:
        name = self._collection_name(product_id)
        try:
            self._client.delete_collection(name=name)
            logger.info("Deleted collection: %s", name)
            return True
        except Exception:
            return False

    def list_collections(self) -> List[str]:
        return [c.name for c in self._client.list_collections()]

    # ── Insert ──────────────────────────────────────────────────

    def add_chunks(self, product_id: str, chunks: List[Chunk]) -> int:
        """Add chunks to a product's collection. Returns count added."""
        if not chunks:
            return 0

        collection = self.get_or_create_collection(product_id)

        ids = [c.doc_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "category": c.category,
                "chunk_index": c.chunk_index,
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]

        # Upsert to handle re-ingestion
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Upserted %d chunks into collection for product '%s'",
                     len(chunks), product_id)
        return len(chunks)

    # ── Query ───────────────────────────────────────────────────

    def query(
        self,
        product_id: str,
        query_text: str,
        top_k: int = config.RAG_TOP_K,
        score_threshold: float = config.RAG_SCORE_THRESHOLD,
        category_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Query a product's collection for relevant chunks."""
        collection = self.get_or_create_collection(product_id)

        if collection.count() == 0:
            return []

        where = {"category": category_filter} if category_filter else None

        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=min(top_k, collection.count()),
                where=where,
            )
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return []

        retrieval_results: List[RetrievalResult] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return retrieval_results

        docs = results["documents"][0]
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)

        for doc, dist, meta in zip(docs, dists, metas):
            if dist > score_threshold:
                continue
            retrieval_results.append(RetrievalResult(
                chunk_text=doc,
                source=meta.get("source", ""),
                category=meta.get("category", ""),
                metadata=meta,
                distance=dist,
            ))

        return retrieval_results

    def query_global(
        self,
        query_text: str,
        product_ids: Optional[List[str]] = None,
        top_k: int = config.RAG_TOP_K,
        score_threshold: float = config.RAG_SCORE_THRESHOLD,
    ) -> List[RetrievalResult]:
        """Query across multiple product collections."""
        all_results: List[RetrievalResult] = []
        targets = product_ids or [
            name.removeprefix(self.collection_prefix)
            for name in self.list_collections()
            if name.startswith(self.collection_prefix)
        ]

        for pid in targets:
            results = self.query(pid, query_text, top_k=top_k, score_threshold=score_threshold)
            all_results.extend(results)

        # Sort by distance and return top_k
        all_results.sort(key=lambda r: r.distance)
        return all_results[:top_k]

    # ── Stats ───────────────────────────────────────────────────

    def collection_stats(self, product_id: str) -> dict:
        collection = self.get_or_create_collection(product_id)
        return {
            "product_id": product_id,
            "collection": self._collection_name(product_id),
            "count": collection.count(),
        }

    def global_stats(self) -> dict:
        collections = self.list_collections()
        total = 0
        details = []
        for name in collections:
            col = self._client.get_collection(name)
            cnt = col.count()
            total += cnt
            details.append({"name": name, "count": cnt})
        return {
            "collections": len(collections),
            "total_chunks": total,
            "details": details,
        }
