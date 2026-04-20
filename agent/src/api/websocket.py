"""WebSocket handler for real-time multi-agent communication."""

import json
import base64
import logging
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect

from ..core.agent_manager import AgentManager
from ..models.message import WSMessage, WSMessageType
from ..models.knowledge import product_json_to_knowledge

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self) -> None:
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)
        logger.info("WS connected (%d active)", len(self.active))

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)
        logger.info("WS disconnected (%d active)", len(self.active))

    async def send(self, ws: WebSocket, msg: WSMessage) -> None:
        try:
            await ws.send_text(msg.model_dump_json())
        except Exception as e:
            logger.error("WS send error: %s", e)

    async def broadcast(self, msg: WSMessage) -> None:
        for ws in list(self.active):
            await self.send(ws, msg)


ws_manager = ConnectionManager()


async def websocket_endpoint(ws: WebSocket, agent_manager: AgentManager):
    """Main WebSocket handler — dispatches messages to agent operations."""
    await ws_manager.connect(ws)

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
                msg_type = data.get("type", "")
            except json.JSONDecodeError:
                await ws_manager.send(ws, WSMessage(
                    type=WSMessageType.AGENT_ERROR,
                    error="Invalid JSON",
                ))
                continue

            # ── CREATE AGENT ────────────────────────────────────
            if msg_type == WSMessageType.CREATE_AGENT:
                product_id = data.get("product_id", "")
                semantic_category_id = data.get("semantic_category_id", "")
                object_label = data.get("object_label", "")
                detection_id = data.get("detection_id")

                # Auto-resolve from detection_id or object_label via label mapping
                if detection_id is not None or (object_label and not product_id and not semantic_category_id):
                    resolved = agent_manager.resolve_detection_label(
                        detection_id=detection_id,
                        label_en=object_label,
                    )
                    product_id = product_id or resolved.get("product_id", "")
                    semantic_category_id = semantic_category_id or resolved.get("semantic_category_id", "")
                    object_label = object_label or resolved.get("en", "")

                if not any([product_id, semantic_category_id, object_label]):
                    await ws_manager.send(ws, WSMessage(
                        type=WSMessageType.AGENT_ERROR,
                        error="Missing knowledge target (product_id, semantic_category_id, object_label, or detection_id)",
                    ))
                    continue
                try:
                    agent = agent_manager.create_agent(
                        product_id=product_id,
                        semantic_category_id=semantic_category_id,
                        object_label=object_label,
                    )
                    await ws_manager.send(ws, WSMessage(
                        type=WSMessageType.AGENT_CREATED,
                        agent_id=agent.agent_id,
                        product_id=agent.product_id,
                        text=agent.product_name,
                        data={
                            "semantic_category_id": agent.semantic_category_id,
                            "object_label": agent.object_label,
                        },
                    ))
                except RuntimeError as e:
                    await ws_manager.send(ws, WSMessage(
                        type=WSMessageType.AGENT_ERROR,
                        error=str(e),
                    ))

            # ── DESTROY AGENT ───────────────────────────────────
            elif msg_type == WSMessageType.DESTROY_AGENT:
                agent_id = data.get("agent_id", "")
                agent_manager.destroy_agent(agent_id)
                await ws_manager.send(ws, WSMessage(
                    type=WSMessageType.AGENT_DESTROYED,
                    agent_id=agent_id,
                ))

            # ── ASK (text) ──────────────────────────────────────
            elif msg_type == WSMessageType.ASK:
                agent_id = data.get("agent_id", "")
                text = data.get("text", "")
                if not agent_id or not text:
                    await ws_manager.send(ws, WSMessage(
                        type=WSMessageType.AGENT_ERROR,
                        agent_id=agent_id,
                        error="Missing agent_id or text",
                    ))
                    continue

                # Signal thinking
                await ws_manager.send(ws, WSMessage(
                    type=WSMessageType.AGENT_THINKING,
                    agent_id=agent_id,
                ))

                answer = await agent_manager.ask(agent_id, text)
                await ws_manager.send(ws, WSMessage(
                    type=WSMessageType.AGENT_ANSWER,
                    agent_id=agent_id,
                    text=answer,
                ))

            # ── AUDIO ───────────────────────────────────────────
            elif msg_type == WSMessageType.AUDIO:
                agent_id = data.get("agent_id", "")
                audio_b64 = data.get("data", "")
                if not agent_id or not audio_b64:
                    await ws_manager.send(ws, WSMessage(
                        type=WSMessageType.AGENT_ERROR,
                        agent_id=agent_id,
                        error="Missing agent_id or audio data",
                    ))
                    continue

                await ws_manager.send(ws, WSMessage(
                    type=WSMessageType.AGENT_THINKING,
                    agent_id=agent_id,
                ))

                audio_bytes = base64.b64decode(audio_b64)
                transcription, answer = await agent_manager.ask_audio(
                    agent_id, audio_bytes
                )

                await ws_manager.send(ws, WSMessage(
                    type=WSMessageType.TRANSCRIPTION,
                    agent_id=agent_id,
                    text=transcription,
                ))
                await ws_manager.send(ws, WSMessage(
                    type=WSMessageType.AGENT_ANSWER,
                    agent_id=agent_id,
                    text=answer,
                ))

            # ── INJECT KNOWLEDGE ────────────────────────────────
            elif msg_type == WSMessageType.INJECT_KNOWLEDGE:
                agent_id = data.get("agent_id", "")
                knowledge_data = data.get("data", {})
                if agent_id and knowledge_data:
                    agent_manager.inject_knowledge_from_dict(agent_id, knowledge_data)
                    await ws_manager.send(ws, WSMessage(
                        type=WSMessageType.AGENT_ANSWER,
                        agent_id=agent_id,
                        text="知识库已更新",
                    ))

            # ── MULTI ASK ───────────────────────────────────────
            elif msg_type == WSMessageType.MULTI_ASK:
                agent_ids = data.get("agent_ids", [])
                text = data.get("text", "")
                if not agent_ids or not text:
                    continue

                for aid in agent_ids:
                    await ws_manager.send(ws, WSMessage(
                        type=WSMessageType.AGENT_THINKING,
                        agent_id=aid,
                    ))

                results = await agent_manager.multi_ask(agent_ids, text)
                await ws_manager.send(ws, WSMessage(
                    type=WSMessageType.MULTI_ANSWER,
                    data={"query": text, "results": results},
                ))

            else:
                await ws_manager.send(ws, WSMessage(
                    type=WSMessageType.AGENT_ERROR,
                    error=f"Unknown message type: {msg_type}",
                ))

    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception as e:
        logger.error("WS error: %s", e)
        ws_manager.disconnect(ws)
