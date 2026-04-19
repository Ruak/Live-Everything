"""REST API routes for agent management and knowledge ingestion."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, model_validator

from ..core.agent_manager import AgentManager
from ..core.knowledge_base import fallback_collection_id, semantic_collection_id
from ..core.rag.startup_ingest import write_rag_source_fingerprint
from ..models.agent import AgentConfig

router = APIRouter(prefix="/api", tags=["agents"])

manager: AgentManager = None  # type: ignore


def set_manager(m: AgentManager) -> None:
    global manager
    manager = m


class CreateAgentRequest(BaseModel):
    product_id: str = ""
    semantic_category_id: str = ""
    object_label: str = ""
    detection_id: Optional[int] = None
    system_prompt: Optional[str] = None
    max_history: int = 30
    temperature: float = 0.7

    @model_validator(mode="after")
    def validate_target(self):
        if not any([self.product_id, self.semantic_category_id, self.object_label, self.detection_id is not None]):
            raise ValueError("At least one knowledge target field is required (product_id, semantic_category_id, object_label, or detection_id)")
        return self


class AskRequest(BaseModel):
    query: str


class MultiAskRequest(BaseModel):
    agent_ids: List[str]
    query: str


class InjectKnowledgeRequest(BaseModel):
    data: dict


class AskResponse(BaseModel):
    agent_id: str
    answer: str


class AudioAskResponse(BaseModel):
    agent_id: str
    transcription: str
    answer: str


class IngestProductRequest(BaseModel):
    data: dict


class IngestTextRequest(BaseModel):
    product_id: str
    text: str
    source: str = "api"
    category: str = "general"


class RAGQueryRequest(BaseModel):
    query: str
    product_id: str = ""
    semantic_category_id: str = ""
    use_fallback: bool = False
    top_k: int = 5

    @model_validator(mode="after")
    def validate_target(self):
        if not any([self.product_id, self.semantic_category_id, self.use_fallback]):
            raise ValueError("Need product_id, semantic_category_id, or use_fallback")
        return self


class ResolveLabelRequest(BaseModel):
    detection_id: Optional[int] = None
    label_en: str = ""


@router.get("/health")
async def health():
    return await manager.health()


@router.get("/agents")
async def list_agents():
    return manager.list_agents()


@router.post("/agents")
async def create_agent(req: CreateAgentRequest):
    try:
        product_id = req.product_id
        semantic_category_id = req.semantic_category_id
        object_label = req.object_label

        # Auto-resolve from detection_id or object_label via label mapping
        if req.detection_id is not None or (object_label and not product_id and not semantic_category_id):
            resolved = manager.resolve_detection_label(
                detection_id=req.detection_id,
                label_en=object_label,
            )
            product_id = product_id or resolved.get("product_id", "")
            semantic_category_id = semantic_category_id or resolved.get("semantic_category_id", "")
            object_label = object_label or resolved.get("en", "")

        cfg = AgentConfig(
            product_id=product_id,
            semantic_category_id=semantic_category_id,
            object_label=object_label,
            system_prompt=req.system_prompt or "",
            max_history=req.max_history,
            temperature=req.temperature,
        )
        agent = manager.create_agent(
            product_id=product_id,
            semantic_category_id=semantic_category_id,
            object_label=object_label,
            agent_config=cfg,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    return {
        "agent_id": agent.agent_id,
        "product_id": agent.product_id,
        "product_name": agent.product_name,
        "semantic_category_id": agent.semantic_category_id,
        "object_label": agent.object_label,
    }


@router.delete("/agents/{agent_id}")
async def destroy_agent(agent_id: str):
    if not manager.destroy_agent(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "destroyed", "agent_id": agent_id}


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {
        "agent_id": agent.agent_id,
        "product_id": agent.product_id,
        "product_name": agent.product_name,
        "semantic_category_id": agent.semantic_category_id,
        "object_label": agent.object_label,
        "status": agent.status,
        "message_count": len(agent.history),
        "history": [
            {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}
            for msg in agent.history
        ],
    }


@router.post("/agents/{agent_id}/ask", response_model=AskResponse)
async def ask_agent(agent_id: str, req: AskRequest):
    agent = manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    answer = await manager.ask(agent_id, req.query)
    return AskResponse(agent_id=agent_id, answer=answer)


@router.post("/agents/{agent_id}/audio", response_model=AudioAskResponse)
async def ask_agent_audio(
    agent_id: str,
    audio: UploadFile = File(...),
):
    agent = manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    audio_bytes = await audio.read()
    mime = audio.content_type or "audio/webm"
    transcription, answer = await manager.ask_audio(agent_id, audio_bytes, mime)
    return AudioAskResponse(
        agent_id=agent_id,
        transcription=transcription,
        answer=answer,
    )


@router.post("/agents/multi-ask")
async def multi_ask(req: MultiAskRequest):
    results = await manager.multi_ask(req.agent_ids, req.query)
    return {"query": req.query, "results": results}


@router.post("/agents/{agent_id}/knowledge")
async def inject_knowledge(agent_id: str, req: InjectKnowledgeRequest):
    if not manager.inject_knowledge_from_dict(agent_id, req.data):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "injected", "agent_id": agent_id}


@router.get("/knowledge")
async def list_knowledge():
    return {
        "products": manager.knowledge_store.list_products(),
        "count": manager.knowledge_store.count,
        "rich_products": sorted(manager.rich_knowledge.products.keys()),
        "semantic_categories": sorted(manager.rich_knowledge.semantic_categories.keys()),
        "fallback_loaded": bool(manager.rich_knowledge.generic_fallback),
    }


@router.post("/knowledge/inject")
async def inject_knowledge_global(req: InjectKnowledgeRequest):
    product_id = manager.knowledge_store.inject_from_dict(req.data)
    if not product_id:
        raise HTTPException(status_code=400, detail="Missing product_id in data")
    return {"status": "injected", "product_id": product_id}


@router.post("/rag/ingest/product")
async def rag_ingest_product(req: IngestProductRequest):
    count = manager.ingest_product(req.data)
    return {
        "status": "ingested",
        "chunks": count,
        "product_id": req.data.get("product_id"),
    }


@router.post("/rag/ingest/text")
async def rag_ingest_text(req: IngestTextRequest):
    count = manager.ingest_text(req.product_id, req.text, req.source, req.category)
    return {"status": "ingested", "chunks": count, "product_id": req.product_id}


@router.post("/rag/ingest/reload")
async def rag_ingest_reload():
    """重新加载 JSON 到内存并全量 ingest knowledge-base → 向量库，更新指纹。"""
    rich_summary = manager.load_rich_knowledge_base()
    rich_rag = manager.ingest_rich_kb()
    write_rag_source_fingerprint()
    return {
        "status": "reloaded",
        "rich_kb_loaded": rich_summary,
        "ingest_rich_kb": rich_rag,
    }


@router.post("/rag/query")
async def rag_query(req: RAGQueryRequest):
    if not manager.retriever:
        raise HTTPException(status_code=503, detail="RAG not enabled")

    target_id = req.product_id
    if req.semantic_category_id:
        target_id = semantic_collection_id(req.semantic_category_id)
    if req.use_fallback:
        target_id = fallback_collection_id()

    results = manager.retriever.retrieve(
        product_id=target_id,
        query=req.query,
        top_k=req.top_k,
    )
    return {
        "target_id": target_id,
        "query": req.query,
        "results": [
            {
                "text": result.chunk_text,
                "category": result.category,
                "source": result.source,
                "distance": result.distance,
                "metadata": result.metadata,
            }
            for result in results
        ],
    }


@router.get("/rag/stats")
async def rag_stats():
    """Get RAG vector store statistics."""
    if not manager.vector_store:
        return {"rag_enabled": False}
    return {"rag_enabled": True, **manager.vector_store.global_stats()}


# ── Label mapping routes ────────────────────────────────────


@router.get("/labels")
async def list_labels():
    """Get the full COCO-80 label mapping."""
    mapping = manager.get_label_mapping()
    return {"count": len(mapping), "labels": mapping}


@router.post("/labels/resolve")
async def resolve_label(req: ResolveLabelRequest):
    """Resolve a detection ID or English label to knowledge target info."""
    return manager.resolve_detection_label(
        detection_id=req.detection_id,
        label_en=req.label_en,
    )
