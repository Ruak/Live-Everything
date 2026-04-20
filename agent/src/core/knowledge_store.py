"""Dynamic knowledge store — load, inject, hot-swap per product."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from ..models.knowledge import ProductKnowledge, product_json_to_knowledge
from ..config import KB_CUSTOM_PRODUCTS_DIR

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """In-memory knowledge store that supports dynamic injection."""

    def __init__(self) -> None:
        self._store: Dict[str, ProductKnowledge] = {}

    # ── Load from file ──────────────────────────────────────────

    def load_from_file(self, path: Path) -> list[str]:
        """Load product knowledge from a JSON file.
        File can be a single product dict or a list of product dicts.
        Returns list of loaded product_ids."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to load knowledge file %s: %s", path, e)
            return []

        items = data if isinstance(data, list) else [data]
        loaded = []
        for item in items:
            pid = item.get("product_id")
            if not pid:
                continue
            knowledge = product_json_to_knowledge(item)
            self._store[pid] = knowledge
            loaded.append(pid)
            logger.info("Loaded knowledge for product %s (%d entries)",
                        pid, len(knowledge.entries))
        return loaded

    def load_all_from_dir(self, directory: Optional[Path] = None) -> list[str]:
        """加载定制商品 JSON（与 knowledge-base/products/custom 一致），供关键字检索兜底。"""
        d = directory or KB_CUSTOM_PRODUCTS_DIR
        if not d.exists():
            logger.warning("Knowledge directory %s does not exist", d)
            return []

        loaded = []
        for f in sorted(d.glob("*.json")):
            loaded.extend(self.load_from_file(f))
        return loaded

    # ── Dynamic injection ───────────────────────────────────────

    def inject(self, product_id: str, knowledge: ProductKnowledge) -> None:
        """Inject or replace knowledge for a product at runtime."""
        self._store[product_id] = knowledge
        logger.info("Injected knowledge for product %s (%d entries)",
                     product_id, len(knowledge.entries))

    def inject_from_dict(self, data: dict) -> Optional[str]:
        """Inject knowledge from a raw product dict. Returns product_id."""
        pid = data.get("product_id")
        if not pid:
            return None
        knowledge = product_json_to_knowledge(data)
        self.inject(pid, knowledge)
        return pid

    # ── Query ───────────────────────────────────────────────────

    def get(self, product_id: str) -> Optional[ProductKnowledge]:
        return self._store.get(product_id)

    def has(self, product_id: str) -> bool:
        return product_id in self._store

    def remove(self, product_id: str) -> bool:
        if product_id in self._store:
            del self._store[product_id]
            logger.info("Removed knowledge for product %s", product_id)
            return True
        return False

    def list_products(self) -> list[str]:
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()

    @property
    def count(self) -> int:
        return len(self._store)
