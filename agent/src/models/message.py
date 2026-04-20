from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import time


class MessageRole(str, Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class AgentStatus(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    SPEAKING = "speaking"
    LISTENING = "listening"
    ERROR = "error"


class Message(BaseModel):
    role: MessageRole
    content: str
    agent_id: str
    product_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


# ── WebSocket Protocol ──────────────────────────────────────────

class WSMessageType(str, Enum):
    # Client → Server
    CREATE_AGENT = "create_agent"
    DESTROY_AGENT = "destroy_agent"
    ASK = "ask"
    AUDIO = "audio"
    INJECT_KNOWLEDGE = "inject_knowledge"
    MULTI_ASK = "multi_ask"

    # Server → Client
    AGENT_CREATED = "agent_created"
    AGENT_DESTROYED = "agent_destroyed"
    AGENT_THINKING = "agent_thinking"
    AGENT_ANSWER = "agent_answer"
    AGENT_ERROR = "agent_error"
    TRANSCRIPTION = "transcription"
    MULTI_ANSWER = "multi_answer"


class WSMessage(BaseModel):
    type: WSMessageType
    agent_id: Optional[str] = None
    product_id: Optional[str] = None
    text: Optional[str] = None
    data: Optional[dict] = None
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
