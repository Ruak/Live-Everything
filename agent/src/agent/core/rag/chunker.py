"""Document chunking — split text into overlapping chunks with metadata."""

import re
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ... import config

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of text with source metadata."""
    text: str
    source: str = ""           # file path or product_id
    category: str = "general"  # faq, spec, selling_point, audience, use_case, general
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return f"{self.source}::{self.category}::{self.chunk_index}"


class Chunker:
    """Split documents into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = config.RAG_CHUNK_SIZE,
        chunk_overlap: int = config.RAG_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        source: str = "",
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split plain text into overlapping chunks."""
        if not text or not text.strip():
            return []

        text = text.strip()
        # If text fits in one chunk, return as-is
        if len(text) <= self.chunk_size:
            return [Chunk(
                text=text,
                source=source,
                category=category,
                chunk_index=0,
                metadata=metadata or {},
            )]

        # Split on paragraph boundaries first, then sentence boundaries
        paragraphs = re.split(r'\n{2,}', text)
        raw_segments: List[str] = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                raw_segments.append(para)
            else:
                # Split long paragraphs on sentence boundaries
                sentences = re.split(r'(?<=[。！？.!?\n])', para)
                raw_segments.extend([s for s in sentences if s.strip()])

        # Merge segments into chunks with overlap
        chunks: List[Chunk] = []
        current = ""
        for seg in raw_segments:
            if current and len(current) + len(seg) > self.chunk_size:
                chunks.append(Chunk(
                    text=current.strip(),
                    source=source,
                    category=category,
                    chunk_index=len(chunks),
                    metadata=metadata or {},
                ))
                # Keep overlap from end of current chunk
                overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current = overlap_text + seg
            else:
                current = current + ("\n" if current else "") + seg

        if current.strip():
            chunks.append(Chunk(
                text=current.strip(),
                source=source,
                category=category,
                chunk_index=len(chunks),
                metadata=metadata or {},
            ))

        logger.debug("Chunked '%s' into %d chunks", source, len(chunks))
        return chunks

    def chunk_markdown(
        self,
        text: str,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split markdown by headings, then chunk each section."""
        # Split on headings (# ## ### etc.)
        sections = re.split(r'(?=^#{1,4}\s)', text, flags=re.MULTILINE)
        all_chunks: List[Chunk] = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Extract heading as category hint
            heading_match = re.match(r'^#{1,4}\s+(.+)', section)
            category = heading_match.group(1).strip() if heading_match else "general"

            chunks = self.chunk_text(
                section,
                source=source,
                category=category,
                metadata={**(metadata or {}), "heading": category},
            )
            # Re-index globally
            for c in chunks:
                c.chunk_index = len(all_chunks)
                all_chunks.append(c)

        return all_chunks

    def chunk_product_json(self, data: dict, product_id: str = "") -> List[Chunk]:
        """Convert a product JSON object into multiple categorized chunks."""
        pid = product_id or data.get("product_id", "unknown")
        pname = data.get("product_name", pid)
        chunks: List[Chunk] = []
        base_meta = {"product_id": pid, "product_name": pname}

        # Tagline
        if data.get("tagline"):
            chunks.extend(self.chunk_text(
                f"商品「{pname}」简介：{data['tagline']}",
                source=pid, category="tagline", metadata=base_meta,
            ))

        # one_line_hook / self_intro_short / self_intro_medium / story_monologue_90s
        for field in ("one_line_hook", "self_intro_short", "self_intro_medium", "story_monologue_90s"):
            val = data.get(field, "")
            if val:
                chunks.extend(self.chunk_text(
                    f"商品「{pname}」{field}：{val}",
                    source=pid, category="intro", metadata=base_meta,
                ))

        # Selling points (support both str list and dict list)
        points = data.get("selling_points", [])
        if points:
            lines: List[str] = []
            for p in points:
                if isinstance(p, dict):
                    title = p.get("title", "卖点")
                    detail = p.get("detail", "")
                    scene = p.get("scene_value", "")
                    lines.append(f"- {title}：{detail} {scene}".strip())
                else:
                    lines.append(f"- {p}")
            text = f"商品「{pname}」的核心卖点：\n" + "\n".join(lines)
            chunks.extend(self.chunk_text(
                text, source=pid, category="selling_point", metadata=base_meta,
            ))

        # Specs (support both dict and list-of-dict)
        specs = data.get("specs", {})
        if specs:
            if isinstance(specs, dict):
                text = f"商品「{pname}」的规格参数：\n" + "\n".join(f"- {k}：{v}" for k, v in specs.items())
            else:
                text = f"商品「{pname}」的规格参数：\n" + "\n".join(
                    f"- {s.get('name', '参数')}：{s.get('value', '')}" if isinstance(s, dict) else f"- {s}"
                    for s in specs
                )
            chunks.extend(self.chunk_text(
                text, source=pid, category="spec", metadata=base_meta,
            ))

        # Audience
        audience = data.get("audience", [])
        if audience:
            text = f"商品「{pname}」适合人群：\n" + "\n".join(f"- {a}" for a in audience)
            chunks.extend(self.chunk_text(
                text, source=pid, category="audience", metadata=base_meta,
            ))

        # Use cases
        use_cases = data.get("use_cases", [])
        if use_cases:
            text = f"商品「{pname}」使用场景：\n" + "\n".join(f"- {u}" for u in use_cases)
            chunks.extend(self.chunk_text(
                text, source=pid, category="use_case", metadata=base_meta,
            ))

        # Limitations
        for i, lim in enumerate(data.get("limitations", [])):
            chunks.extend(self.chunk_text(
                f"商品「{pname}」使用限制：{lim}",
                source=pid, category="limitation", metadata=base_meta,
            ))

        # Care tips
        tips = data.get("care_tips", [])
        if tips:
            text = f"商品「{pname}」使用建议：\n" + "\n".join(f"- {t}" for t in tips)
            chunks.extend(self.chunk_text(
                text, source=pid, category="care_tip", metadata=base_meta,
            ))

        # question_type_answers
        for qtype, answer in data.get("question_type_answers", {}).items():
            if answer:
                chunks.extend(self.chunk_text(
                    f"商品「{pname}」{qtype}类问题回答：{answer}",
                    source=pid, category=f"qtype_{qtype}", metadata=base_meta,
                ))

        # FAQ — each Q&A as a separate chunk for precise retrieval
        for i, faq in enumerate(data.get("faq", [])):
            q = faq.get("question", "")
            a = faq.get("answer", "")
            if q and a:
                chunks.extend(self.chunk_text(
                    f"问：{q}\n答：{a}",
                    source=pid, category="faq",
                    metadata={**base_meta, "faq_index": i},
                ))

        # Description (free-form text if present)
        desc = data.get("description", "")
        if desc:
            chunks.extend(self.chunk_text(
                desc, source=pid, category="description", metadata=base_meta,
            ))

        # Re-index
        for i, c in enumerate(chunks):
            c.chunk_index = i

        logger.info("Chunked product '%s' into %d chunks", pid, len(chunks))
        return chunks

    def chunk_semantic_category(self, data: dict) -> List[Chunk]:
        """Convert a semantic category dict into chunks for RAG retrieval."""
        cat_id = data.get("semantic_category_id", "unknown")
        display = data.get("display_name", cat_id)
        chunks: List[Chunk] = []
        meta = {"semantic_category_id": cat_id, "display_name": display}

        # Category pitch
        pitch = data.get("category_pitch", "")
        if pitch:
            chunks.extend(self.chunk_text(
                f"类别「{display}」定位：{pitch}",
                source=cat_id, category="category_pitch", metadata=meta,
            ))

        # Common roles, features, scenes
        for field, label in [
            ("common_roles", "常见角色"),
            ("common_features", "常见特征"),
            ("common_scenes", "常见场景"),
            ("safe_claim_rules", "安全表达规则"),
        ]:
            items = data.get(field, [])
            if items:
                text = f"类别「{display}」{label}：\n" + "\n".join(f"- {it}" for it in items)
                chunks.extend(self.chunk_text(
                    text, source=cat_id, category=field, metadata=meta,
                ))

        # Generic answer templates
        for qtype, answer in data.get("generic_answer_templates", {}).items():
            if answer:
                chunks.extend(self.chunk_text(
                    f"类别「{display}」{qtype}类回答模板：{answer}",
                    source=cat_id, category=f"template_{qtype}", metadata=meta,
                ))

        for i, c in enumerate(chunks):
            c.chunk_index = i

        logger.info("Chunked semantic category '%s' into %d chunks", cat_id, len(chunks))
        return chunks

    def chunk_fallback(self, data: dict) -> List[Chunk]:
        """Convert fallback / generic_fallback dict into chunks."""
        chunks: List[Chunk] = []
        meta = {"type": "fallback"}

        for qtype, answer in data.get("answer_templates", {}).items():
            if answer:
                chunks.extend(self.chunk_text(
                    f"通用兜底{qtype}类回答：{answer}",
                    source="generic_fallback", category=f"fallback_{qtype}", metadata=meta,
                ))

        for i, c in enumerate(chunks):
            c.chunk_index = i

        logger.info("Chunked fallback into %d chunks", len(chunks))
        return chunks
