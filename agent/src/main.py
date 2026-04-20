"""AR Product Guide — Agent Service Entry Point."""

import logging
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .core.agent_manager import AgentManager
from .core.rag.startup_ingest import (
    should_skip_startup_rag_ingest,
    write_rag_source_fingerprint,
)
from .api.routes import router as api_router, set_manager
from .api.websocket import websocket_endpoint
from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

agent_manager = AgentManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ─────────────────────────────────────────────
    set_manager(agent_manager)

    # 关键字词条：与 knowledge-base/products/custom 同源
    loaded = agent_manager.knowledge_store.load_all_from_dir()
    logger.info("Loaded keyword knowledge for %d products: %s", len(loaded), loaded)

    # Load rich knowledge base (core, categories, products, label mapping)
    rich_summary = agent_manager.load_rich_knowledge_base()
    logger.info("Rich KB: %s", rich_summary)

    # RAG：向量在仓库 data/.chroma；源未变时可跳过重复 ingest
    if config.RAG_ENABLED and config.RAG_INGEST_ON_STARTUP:
        skip, reason = should_skip_startup_rag_ingest()
        if skip:
            logger.info("RAG startup ingest skipped (%s)", reason)
        else:
            if reason:
                logger.info("RAG startup ingest: %s", reason)
            rich_rag = agent_manager.ingest_rich_kb()
            logger.info("RAG ingestion (knowledge-base): %s", rich_rag)
            write_rag_source_fingerprint()
    elif config.RAG_ENABLED:
        logger.info(
            "RAG startup ingest disabled (RAG_INGEST_ON_STARTUP=false); index at %s",
            config.RAG_PERSIST_DIR,
        )

    health = await agent_manager.health()
    logger.info("System health: %s", health)

    # 异步预热 Whisper，避免第一次 /audio 请求被 10-30s 的模型加载阻塞
    import asyncio as _asyncio

    async def _warmup_stt() -> None:
        try:
            loader = getattr(agent_manager.stt, "_load_model", None)
            if callable(loader):
                await _asyncio.to_thread(loader)
                logger.info("STT provider warmed up")
        except Exception as exc:  # pragma: no cover - 预热失败不应拖死启动
            logger.warning("STT warmup failed: %s", exc)

    _asyncio.create_task(_warmup_stt())

    yield

    # ── Shutdown ────────────────────────────────────────────
    logger.info("Shutting down agent service")


app = FastAPI(
    title="AR Product Guide — Agent Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.websocket("/ws")
async def ws_route(ws: WebSocket):
    await websocket_endpoint(ws, agent_manager)


def start():
    uvicorn.run(
        "src.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
    )


if __name__ == "__main__":
    start()
