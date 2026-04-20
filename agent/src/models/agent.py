from typing import List, Optional
from pydantic import BaseModel, Field
import time

from .message import Message, AgentStatus
from .knowledge import ProductKnowledge


class AgentConfig(BaseModel):
    """Configuration for creating an agent."""
    product_id: str = ""
    semantic_category_id: str = ""
    object_label: str = ""
    system_prompt: str = ""
    max_history: int = 30
    temperature: float = 0.7


class AgentState(BaseModel):
    """Runtime state of a single agent."""
    agent_id: str
    product_id: str = ""
    product_name: str = ""
    semantic_category_id: str = ""
    object_label: str = ""
    status: AgentStatus = AgentStatus.IDLE
    history: List[Message] = []
    created_at: float = Field(default_factory=time.time)

    # Knowledge is not serialized via API by default
    knowledge: Optional[ProductKnowledge] = Field(default=None, exclude=True)
    system_prompt: str = ""
    max_history: int = 30
    temperature: float = 0.7

    def add_message(self, msg: Message) -> None:
        self.history.append(msg)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context_window(self) -> List[Message]:
        """Return conversation history for LLM context."""
        return list(self.history)

    def build_knowledge_context(self, query: str) -> str:
        """Search knowledge and build a context string for the LLM."""
        if not self.knowledge:
            return ""

        results = self.knowledge.search(query, top_k=5)
        if not results:
            return ""

        parts = [f"以下是关于「{self.product_name}」的相关知识："]
        for entry in results:
            parts.append(f"- [{entry.category}] {entry.content}")
        return "\n".join(parts)


class AgentSummary(BaseModel):
    """Lightweight agent info for listing."""
    agent_id: str
    product_id: str
    product_name: str
    semantic_category_id: str = ""
    object_label: str = ""
    status: AgentStatus
    message_count: int
    created_at: float
