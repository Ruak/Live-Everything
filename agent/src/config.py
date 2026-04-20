"""Agent service configuration.

所有敏感或可变配置通过 `.env`（项目根目录 `agent/.env`）注入。
`python-dotenv` 在这里加载；如果没有安装则降级为纯 `os.environ`。
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # agent/ 包根（含 pyproject）
REPO_ROOT = BASE_DIR.parent  # 仓库 live-everything/
# 向量库与知识 JSON 统一放在仓库 data/（与 web/vite 引用的 knowledge-base 同级）
REPO_DATA_DIR = REPO_ROOT / "data"

# ── 加载 .env（在引用任何 env 前做） ────────────────────────────
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(BASE_DIR / ".env", override=False)
except Exception:  # pragma: no cover
    # 没装 python-dotenv 也不致命，部署时可直接用系统 env
    pass


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


# ── LLM ─────────────────────────────────────────────────────────
# 可选: deepseek | ollama | openai_compatible | rule
LLM_PROVIDER = _env("LLM_PROVIDER", "deepseek")

# DeepSeek（OpenAI 兼容）
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = _env("DEEPSEEK_MODEL", "deepseek-chat")

OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen2.5:7b")

OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_MODEL = _env("OPENAI_MODEL", "default")

# ── STT ─────────────────────────────────────────────────────────
STT_PROVIDER = _env("STT_PROVIDER", "whisper")  # whisper | stub
WHISPER_MODEL_SIZE = _env("WHISPER_MODEL_SIZE", "base")
# "auto" → 自动检测，否则按 ISO 代码（zh/en/ja/...）
WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "zh")
# 小于该阈值视为可能是静音或容器损坏；会被保存以便复现
WHISPER_MIN_AUDIO_SECONDS = float(_env("WHISPER_MIN_AUDIO_SECONDS") or 0.25)
# 是否把解析失败的原始音频保存到 .cache/failed_audio 以便人工排查
WHISPER_SAVE_FAILED_AUDIO = _env_bool("WHISPER_SAVE_FAILED_AUDIO", True)

# ── Agent ───────────────────────────────────────────────────────
MAX_AGENTS = _env_int("MAX_AGENTS", 10)
MAX_HISTORY_PER_AGENT = _env_int("MAX_HISTORY_PER_AGENT", 30)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的商品讲解员。"
    "请优先使用【当前物体专属知识】与【同类通用知识】作答，"
    "若本地知识不足，再参考【联网补充知识】；引用联网结果时请明示来源并保持克制。"
    "回答要简洁、准确、有吸引力，不要编造参数、价格或型号。"
    "交互界面侧栏已展示商品名称与简介，请勿在回答开头重复自我介绍或复述完整产品名；直接切入用户问题。"
    "可使用 Markdown（**加粗**、分段、`-` 列表）便于阅读。"
)

# ── RAG ─────────────────────────────────────────────────────────
RAG_ENABLED = _env_bool("RAG_ENABLED", True)
# 是否在启动时执行 ingest（仍会结合指纹尽可能跳过重复计算）
RAG_INGEST_ON_STARTUP = _env_bool("RAG_INGEST_ON_STARTUP", True)
# 源 JSON 未变更且 REPO_DATA_DIR/.chroma 已有数据时跳过 ingest；设 false 则每次启动都跑 ingest（upsert）
RAG_SKIP_IF_SOURCES_UNCHANGED = _env_bool("RAG_SKIP_IF_SOURCES_UNCHANGED", True)
RAG_PERSIST_DIR = REPO_DATA_DIR / ".chroma"
RAG_FINGERPRINT_FILE = REPO_DATA_DIR / ".rag_source_fingerprint"
RAG_COLLECTION_PREFIX = _env("RAG_COLLECTION_PREFIX", "product_")
RAG_CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", 300)
RAG_CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", 50)
RAG_TOP_K = _env_int("RAG_TOP_K", 5)
RAG_SCORE_THRESHOLD = float(_env("RAG_SCORE_THRESHOLD") or 1.5)

# ── Rich Knowledge Base（JSON 源均在此目录下）──────────────────
KNOWLEDGE_BASE_DIR = REPO_DATA_DIR / "knowledge-base"
KB_CONFIG_DIR = KNOWLEDGE_BASE_DIR / "config"
KB_CUSTOM_PRODUCTS_DIR = KNOWLEDGE_BASE_DIR / "products" / "custom"
KB_GENERIC_PRODUCTS_DIR = KNOWLEDGE_BASE_DIR / "products" / "generic"
KB_LABEL_MAPPING_FILE = KB_CONFIG_DIR / "label_mapping.json"
KB_CATEGORIES_FILE = KB_CONFIG_DIR / "categories.json"
KB_PERSONAS_FILE = KB_CONFIG_DIR / "personas.json"
KB_CORE_FILE = KB_CONFIG_DIR / "core.json"

# ── Web Search（RAG 兜底） ─────────────────────────────────────
WEB_SEARCH_ENABLED = _env_bool("WEB_SEARCH_ENABLED", True)
WEB_SEARCH_PROVIDER = _env("WEB_SEARCH_PROVIDER", "duckduckgo")
WEB_SEARCH_TIMEOUT_SECONDS = float(_env("WEB_SEARCH_TIMEOUT_SECONDS") or 6)
WEB_SEARCH_TOP_K = _env_int("WEB_SEARCH_TOP_K", 3)

# ── Server ──────────────────────────────────────────────────────
HOST = _env("HOST", "0.0.0.0")
PORT = _env_int("PORT", 8000)
CORS_ORIGINS = [o for o in _env("CORS_ORIGINS", "*").split(",") if o.strip()]
