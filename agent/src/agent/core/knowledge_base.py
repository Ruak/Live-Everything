"""Rich knowledge base loader and keyword fallback for layered object knowledge."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

SEMANTIC_COLLECTION_PREFIX = "semantic__"
FALLBACK_COLLECTION_PREFIX = "fallback__"


@dataclass
class ResolvedKnowledgeTarget:
    product_id: str = ""
    product_name: str = ""
    semantic_category_id: str = ""
    semantic_category_name: str = ""
    object_label: str = ""

    @property
    def display_name(self) -> str:
        return (
            self.product_name
            or self.object_label
            or self.semantic_category_name
            or self.product_id
            or self.semantic_category_id
            or "当前物体"
        )


def semantic_collection_id(category_id: str) -> str:
    return f"{SEMANTIC_COLLECTION_PREFIX}{category_id}"


def fallback_collection_id(category_id: str = "generic_object") -> str:
    return f"{FALLBACK_COLLECTION_PREFIX}{category_id}"


class RichKnowledgeBase:
    """Structured knowledge registry for specific objects, categories, and fallback."""

    def __init__(self) -> None:
        self.products: dict[str, dict] = {}
        self.semantic_categories: dict[str, dict] = {}
        self.generic_fallback: dict = {}
        self.response_policy: dict = {}
        self.design_principles: list[str] = []
        self.loaded_files: list[str] = []

    def clear(self) -> None:
        self.products.clear()
        self.semantic_categories.clear()
        self.generic_fallback = {}
        self.response_policy = {}
        self.design_principles = []
        self.loaded_files = []

    def is_rich_payload(self, data: object) -> bool:
        if not isinstance(data, dict):
            return False
        return any(
            key in data
            for key in ("semantic_categories", "generic_fallback", "response_policy")
        )

    def load_from_file(self, path: Path) -> bool:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load rich knowledge file %s: %s", path, exc)
            return False

        if not self.is_rich_payload(data):
            return False

        self.load_from_dict(data)
        self.loaded_files.append(str(path))
        logger.info(
            "Loaded rich knowledge file %s (%d products, %d categories)",
            path,
            len(self.products),
            len(self.semantic_categories),
        )
        return True

    def load_from_dict(self, data: dict) -> None:
        for product in data.get("products", []):
            product_id = product.get("product_id")
            if product_id:
                self.products[product_id] = product

        for category in data.get("semantic_categories", []):
            category_id = category.get("semantic_category_id")
            if category_id:
                self.semantic_categories[category_id] = category

        fallback = data.get("generic_fallback")
        if isinstance(fallback, dict) and fallback:
            self.generic_fallback = fallback

        response_policy = data.get("response_policy")
        if isinstance(response_policy, dict) and response_policy:
            self.response_policy = response_policy

        principles = data.get("design_principles")
        if isinstance(principles, list) and principles:
            self.design_principles = [str(item) for item in principles if str(item).strip()]

    def resolve_target(
        self,
        product_id: str = "",
        semantic_category_id: str = "",
        object_label: str = "",
    ) -> ResolvedKnowledgeTarget:
        product = self.products.get(product_id) if product_id else None
        category_id = semantic_category_id or (
            product.get("semantic_category_id", "") if product else ""
        )
        category = self.semantic_categories.get(category_id) if category_id else None

        return ResolvedKnowledgeTarget(
            product_id=product_id or "",
            product_name=(product or {}).get("product_name", ""),
            semantic_category_id=category_id,
            semantic_category_name=(category or {}).get("display_name", ""),
            object_label=object_label,
        )

    def build_policy_prompt(self, target: ResolvedKnowledgeTarget) -> str:
        lines = [
            "回答规则：优先使用当前识别物体的专属知识，其次使用该类物体的通用知识，最后才使用通用兜底知识。",
            "如果回答依赖通用类别知识或兜底知识，必须明确说明这是通用判断，不要伪造品牌、型号、价格、年代或唯一事实。",
        ]

        if target.product_name:
            lines.append(f"当前识别物体：{target.product_name}")
        elif target.object_label:
            lines.append(f"当前识别物体标签：{target.object_label}")

        if target.semantic_category_name:
            lines.append(f"当前物体所属类别：{target.semantic_category_name}")

        safe_phrases = self.response_policy.get("safe_phrases", [])
        if safe_phrases:
            joined = "；".join(str(item) for item in safe_phrases[:4])
            lines.append(f"当只能使用通用知识时，可使用这类限定表达：{joined}")

        if self.design_principles:
            lines.append("知识库原则：" + "；".join(self.design_principles[:3]))

        return "\n".join(lines)

    def build_keyword_context(
        self,
        query: str,
        target: ResolvedKnowledgeTarget,
        top_k: int = 6,
    ) -> str:
        entries: list[tuple[str, str]] = []

        if target.product_id:
            product = self.products.get(target.product_id)
            if product:
                entries.extend(self._product_entries(product))

        if target.semantic_category_id:
            category = self.semantic_categories.get(target.semantic_category_id)
            if category:
                entries.extend(self._semantic_category_entries(category))

        if self.generic_fallback:
            entries.extend(self._fallback_entries(self.generic_fallback))

        ranked = self._rank_entries(query, entries, top_k=top_k)
        if not ranked:
            return ""

        lines = ["以下是可直接引用的本地知识："]
        for label, content in ranked:
            lines.append(f"- [{label}] {content}")
        return "\n".join(lines)

    def _product_entries(self, product: dict) -> list[tuple[str, str]]:
        name = product.get("product_name", product.get("product_id", "当前物体"))
        entries: list[tuple[str, str]] = []

        def add(label: str, value: object) -> None:
            text = self._normalize_text(value)
            if text:
                entries.append((label, text))

        add("商品简介", product.get("tagline"))
        add("一句话亮点", product.get("one_line_hook"))
        add("短介绍", product.get("self_intro_short"))
        add("中介绍", product.get("self_intro_medium"))
        add("讲解词", product.get("story_monologue_90s"))

        for item in product.get("selling_points", []):
            if isinstance(item, dict):
                title = item.get("title", "卖点")
                detail = self._normalize_text(item.get("detail"))
                scene_value = self._normalize_text(item.get("scene_value"))
                add("核心卖点", f"{title}：{detail} {scene_value}".strip())
            else:
                add("核心卖点", item)

        specs = product.get("specs", [])
        if isinstance(specs, dict):
            for key, value in specs.items():
                add("参数", f"{key}：{value}")
        else:
            for spec in specs:
                if isinstance(spec, dict):
                    add("参数", f"{spec.get('name', '参数')}：{spec.get('value', '')}")

        for audience in product.get("audience", []):
            add("适用人群", audience)
        for use_case in product.get("use_cases", []):
            add("使用场景", use_case)
        for limitation in product.get("limitations", []):
            add("限制说明", limitation)
        for care_tip in product.get("care_tips", []):
            add("使用建议", care_tip)
        for misunderstanding in product.get("common_misunderstandings", []):
            add("常见误区", misunderstanding)
        for step in product.get("guided_demo_script", []):
            if isinstance(step, dict):
                add("讲解脚本", f"{step.get('step', '步骤')}：{step.get('line', '')}")

        for question_type, answer in product.get("question_type_answers", {}).items():
            add(f"问题类型回答/{question_type}", answer)

        for faq in product.get("faq", []):
            if isinstance(faq, dict):
                question = self._normalize_text(faq.get("question"))
                answer = self._normalize_text(faq.get("answer"))
                if question and answer:
                    add("FAQ", f"问：{question}\n答：{answer}")

        if not entries and name:
            add("商品名", name)

        return entries

    def _semantic_category_entries(self, category: dict) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []

        def add(label: str, value: object) -> None:
            text = self._normalize_text(value)
            if text:
                entries.append((label, text))

        display_name = category.get("display_name", category.get("semantic_category_id", "通用类别"))
        add("类别名称", display_name)
        add("类别定位", category.get("category_pitch"))
        for role in category.get("common_roles", []):
            add("常见角色", role)
        for feature in category.get("common_features", []):
            add("常见特征", feature)
        for scene in category.get("common_scenes", []):
            add("常见场景", scene)
        for rule in category.get("safe_claim_rules", []):
            add("安全表达", rule)
        for rule in category.get("forbidden_claim_rules", []):
            add("禁止表达", rule)
        for key, value in category.get("generic_answer_templates", {}).items():
            add(f"类别模板/{key}", value)
        return entries

    def _fallback_entries(self, fallback: dict) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for key, value in fallback.get("answer_templates", {}).items():
            text = self._normalize_text(value)
            if text:
                entries.append((f"通用兜底/{key}", text))
        return entries

    def _rank_entries(
        self,
        query: str,
        entries: Iterable[tuple[str, str]],
        top_k: int,
    ) -> list[tuple[str, str]]:
        keywords = self._query_keywords(query)
        scored: list[tuple[float, tuple[str, str]]] = []

        for label, content in entries:
            haystack = f"{label} {content}".lower()
            score = 0.0
            for keyword in keywords:
                if keyword in haystack:
                    score += 1.0
                if keyword in label.lower():
                    score += 0.5
            if not keywords:
                score = 0.1
            if score > 0:
                scored.append((score, (label, content)))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    @staticmethod
    def _query_keywords(query: str) -> list[str]:
        lowered = query.lower()
        parts = [part.strip() for part in re.split(r"[\s,，。！？?]+", lowered)]
        return [part for part in parts if len(part) > 1]

    @staticmethod
    def _normalize_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = [RichKnowledgeBase._normalize_text(item) for item in value]
            return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            parts = [
                f"{key}：{RichKnowledgeBase._normalize_text(item)}"
                for key, item in value.items()
                if RichKnowledgeBase._normalize_text(item)
            ]
            return "\n".join(parts)
        return str(value).strip()
