import json
from typing import Any, List, Dict, Optional

from pydantic import BaseModel


def _plain_text(value: Any) -> str:
    """将 FAQ/卖点等 JSON 中的 str 或结构化对象转为可检索的纯文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        if any(k in value for k in ("title", "detail", "scene_value")):
            lines = []
            if value.get("title"):
                lines.append(str(value["title"]))
            if value.get("detail"):
                lines.append(str(value["detail"]))
            if value.get("scene_value"):
                lines.append(f"场景价值：{value['scene_value']}")
            return "\n".join(lines)
        if "name" in value or "value" in value:
            n = value.get("name", "")
            v = value.get("value", "")
            return f"{n}：{v}".strip("：") if n or v else json.dumps(value, ensure_ascii=False)
        if "myth" in value or "fact" in value:
            parts = []
            if value.get("myth"):
                parts.append(f"误解：{value['myth']}")
            if value.get("fact"):
                parts.append(f"事实：{value['fact']}")
            return "\n".join(parts)
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    if isinstance(value, list):
        return "\n".join(_plain_text(x) for x in value)
    return str(value)


class KnowledgeEntry(BaseModel):
    """A single piece of knowledge about a product."""
    key: str
    content: str
    category: str  # "faq", "spec", "selling_point", "audience", "use_case", "general"
    keywords: List[str] = []


class ProductKnowledge(BaseModel):
    """Full knowledge context for a single product."""
    product_id: str
    product_name: str
    tagline: str = ""
    entries: List[KnowledgeEntry] = []
    raw_data: Dict = {}

    def search(self, query: str, top_k: int = 5) -> List[KnowledgeEntry]:
        """Simple keyword-based search over knowledge entries."""
        q = query.lower()
        query_words = [w for w in q.split() if len(w) > 1]

        scored: list[tuple[float, KnowledgeEntry]] = []
        for entry in self.entries:
            score = 0.0
            entry_text = (entry.content + " " + " ".join(entry.keywords)).lower()

            # Keyword match
            for word in query_words:
                if word in entry_text:
                    score += 1.0
                if word in [k.lower() for k in entry.keywords]:
                    score += 2.0

            # Category boost for FAQ
            if entry.category == "faq" and score > 0:
                score += 1.5

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]


def product_json_to_knowledge(data: Dict) -> ProductKnowledge:
    """Convert a product JSON (from web/public/data/products.json) into a
    ProductKnowledge object with structured entries."""
    entries: List[KnowledgeEntry] = []

    # Tagline
    if data.get("tagline"):
        entries.append(KnowledgeEntry(
            key="tagline",
            content=data["tagline"],
            category="general",
            keywords=["介绍", "一句话", "是什么"],
        ))

    # Selling points（字符串或 { title, detail, scene_value }）
    for i, point in enumerate(data.get("selling_points", [])):
        text = _plain_text(point)
        if not text.strip():
            continue
        entries.append(KnowledgeEntry(
            key=f"selling_point_{i}",
            content=text,
            category="selling_point",
            keywords=["卖点", "优势", "特点", "亮点"],
        ))

    # Specs：dict（k→v）或 [{ name, value }, ...]
    specs_raw = data.get("specs", {})
    if isinstance(specs_raw, dict):
        for k, v in specs_raw.items():
            entries.append(KnowledgeEntry(
                key=f"spec_{k}",
                content=f"{k}：{v}",
                category="spec",
                keywords=["参数", "规格", "配置", str(k)],
            ))
    elif isinstance(specs_raw, list):
        for i, item in enumerate(specs_raw):
            text = _plain_text(item)
            if not text.strip():
                continue
            key = f"spec_{i}"
            if isinstance(item, dict) and item.get("name"):
                key = f"spec_{item.get('name', i)}"
            entries.append(KnowledgeEntry(
                key=key,
                content=text,
                category="spec",
                keywords=["参数", "规格", "配置"],
            ))

    # Audience / use_cases（元素可为 str 或结构化对象）
    for i, aud in enumerate(data.get("audience", [])):
        text = _plain_text(aud)
        if not text.strip():
            continue
        entries.append(KnowledgeEntry(
            key=f"audience_{i}",
            content=text,
            category="audience",
            keywords=["适合", "人群", "谁用", "适用"],
        ))

    for i, uc in enumerate(data.get("use_cases", [])):
        text = _plain_text(uc)
        if not text.strip():
            continue
        entries.append(KnowledgeEntry(
            key=f"use_case_{i}",
            content=text,
            category="use_case",
            keywords=["场景", "用途", "用在", "怎么用"],
        ))

    # FAQ
    for i, faq in enumerate(data.get("faq", [])):
        q = faq.get("question", "")
        a = faq.get("answer", "")
        entries.append(KnowledgeEntry(
            key=f"faq_{i}",
            content=f"问：{q}\n答：{a}",
            category="faq",
            keywords=q.split(),
        ))

    return ProductKnowledge(
        product_id=data.get("product_id", ""),
        product_name=data.get("product_name", ""),
        tagline=data.get("tagline", ""),
        entries=entries,
        raw_data=data,
    )
