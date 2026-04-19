"""启动时 RAG ingest：源文件指纹 + 持久化索引存在性，避免无谓全量重跑。"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Tuple

from ... import config

logger = logging.getLogger(__name__)


def _rag_source_paths() -> List[Path]:
    """与 ingest_rich_kb_from_files 一致的 JSON 源（不含 _ 前缀模板）。"""
    paths: List[Path] = []
    if config.KB_CUSTOM_PRODUCTS_DIR.exists():
        for p in sorted(config.KB_CUSTOM_PRODUCTS_DIR.glob("*.json")):
            if p.name.startswith("_"):
                continue
            paths.append(p)
    for p in (config.KB_CATEGORIES_FILE, config.KB_CORE_FILE):
        if p.exists():
            paths.append(p)
    return paths


def compute_rag_source_fingerprint() -> str:
    """对知识库源文件内容做 SHA256，用于判断是否需要重新向量化。"""
    h = hashlib.sha256()
    for p in _rag_source_paths():
        h.update(p.resolve().as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def persisted_rag_has_chunks(persist_dir: Path) -> bool:
    """Chroma 目录存在且至少一个 collection 有文档。"""
    if not persist_dir.is_dir():
        return False
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        for col in client.list_collections():
            if col.count() > 0:
                return True
    except Exception as exc:  # pragma: no cover - 损坏的索引则重建
        logger.warning("Could not read Chroma persist dir %s: %s", persist_dir, exc)
    return False


def should_skip_startup_rag_ingest() -> Tuple[bool, str]:
    """若源未变且磁盘上已有向量数据，则跳过启动 ingest。"""
    if not config.RAG_SKIP_IF_SOURCES_UNCHANGED:
        return False, "RAG_SKIP_IF_SOURCES_UNCHANGED=false"

    fp_path = config.RAG_FINGERPRINT_FILE
    current = compute_rag_source_fingerprint()

    if not fp_path.exists():
        return False, "no fingerprint file yet"

    try:
        stored = fp_path.read_text(encoding="utf-8").strip()
    except OSError:
        return False, "cannot read fingerprint file"

    if stored != current:
        return False, "knowledge source files changed"

    if persisted_rag_has_chunks(config.RAG_PERSIST_DIR):
        return True, "sources unchanged, using persisted index"

    return False, "fingerprint matches but index missing or empty — re-ingesting"


def write_rag_source_fingerprint() -> None:
    """ingest 成功后写入指纹，与当前磁盘上的知识 JSON 对齐。"""
    config.RAG_FINGERPRINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.RAG_FINGERPRINT_FILE.write_text(
        compute_rag_source_fingerprint(),
        encoding="utf-8",
    )
