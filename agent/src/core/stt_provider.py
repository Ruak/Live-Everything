"""Speech-to-text provider abstraction."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .. import config

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = float(getattr(config, "WHISPER_MIN_AUDIO_SECONDS", 0.25))
SAVE_FAILED = bool(getattr(config, "WHISPER_SAVE_FAILED_AUDIO", True))
FAILED_AUDIO_DIR = Path(config.BASE_DIR) / ".cache" / "failed_audio"


def _save_failed_sample(audio_bytes: bytes, mime_type: str, reason: str) -> None:
    if not SAVE_FAILED:
        return
    try:
        FAILED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        ext = ".webm" if "webm" in mime_type else ".wav"
        name = f"{int(time.time() * 1000)}_{reason}{ext}"
        path = FAILED_AUDIO_DIR / name
        path.write_bytes(audio_bytes)
        logger.warning("Saved failed audio to %s (%d bytes)", path, len(audio_bytes))
    except Exception as exc:  # pragma: no cover
        logger.debug("Save failed audio ignored: %s", exc)


class STTProvider(ABC):
    """Abstract base for speech-to-text providers."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...


def _resolve_ffmpeg_binary() -> Optional[str]:
    """优先用系统 ffmpeg，其次退回 imageio-ffmpeg 内置的二进制。
    Windows 上多数用户没装 ffmpeg，这里让 openai-whisper 免依赖即可跑。"""

    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:  # pragma: no cover - 仅在未装 ffmpeg 时触发
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        logger.warning(
            "Neither system ffmpeg nor imageio-ffmpeg available: %s. "
            "Install one of them or `pip install imageio-ffmpeg`.",
            exc,
        )
        return None


def _decode_audio_to_pcm(audio_path: str, sample_rate: int = 16000):
    """使用 ffmpeg 将任意容器/编码的音频解码为 Whisper 需要的 mono float32 PCM。

    whisper.transcribe 支持直接传入 numpy array，这样就绕开了它内部对
    PATH 上 ffmpeg 可执行名为 "ffmpeg" 的硬依赖（imageio-ffmpeg 的二进制
    名字通常不是 "ffmpeg.exe"）。
    """

    import numpy as np  # 延迟导入，避免服务启动时的开销

    ffmpeg_bin = _resolve_ffmpeg_binary()
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg not found. 请安装 ffmpeg 或 `pip install imageio-ffmpeg`。"
        )

    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-threads", "0",
        "-i", audio_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-loglevel", "error",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg 解码失败: {stderr.strip()}") from exc

    pcm = np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0
    return pcm


class WhisperProvider(STTProvider):
    """Local Whisper model for offline STT.

    - 需要 `openai-whisper`。
    - 解码使用系统 ffmpeg 或 `imageio-ffmpeg` 提供的内置二进制，不要求用户手动装 ffmpeg。
    - 延迟加载模型，首次转写时才真正加载。
    """

    def __init__(self, model_size: str = config.WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import whisper  # type: ignore
            logger.info("Loading Whisper model: %s", self.model_size)
            self._model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded")
        except ImportError:
            logger.error(
                "openai-whisper not installed. Run: pip install openai-whisper"
            )
            raise
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
            raise

    async def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        await asyncio.to_thread(self._load_model)

        logger.info(
            "STT received: %d bytes, mime=%s", len(audio_bytes or b""), mime_type
        )
        if not audio_bytes or len(audio_bytes) < 512:
            logger.warning("Audio too small (%d bytes), skip", len(audio_bytes or b""))
            _save_failed_sample(audio_bytes or b"", mime_type, "too_small")
            return ""

        ext = ".webm" if "webm" in mime_type else ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        try:
            tmp.write(audio_bytes)
            tmp.close()

            try:
                pcm = await asyncio.to_thread(_decode_audio_to_pcm, tmp.name)
            except Exception as exc:
                logger.exception("ffmpeg decode failed: %s", exc)
                _save_failed_sample(audio_bytes, mime_type, "ffmpeg_error")
                return ""

            duration = float(len(pcm)) / float(SAMPLE_RATE)
            logger.info(
                "Decoded PCM: samples=%d duration=%.2fs", len(pcm), duration
            )
            if duration < MIN_AUDIO_SECONDS:
                logger.warning(
                    "PCM too short (%.2fs < %.2fs), likely silence or empty container",
                    duration,
                    MIN_AUDIO_SECONDS,
                )
                _save_failed_sample(audio_bytes, mime_type, "pcm_short")
                return ""

            language = config.WHISPER_LANGUAGE
            primary_lang: Optional[str] = None if language == "auto" else language

            def _run(lang: Optional[str]) -> str:
                result = self._model.transcribe(  # type: ignore[union-attr]
                    pcm,
                    language=lang,
                    fp16=False,
                )
                return str(result.get("text", "")).strip()

            text = await asyncio.to_thread(_run, primary_lang)
            logger.info(
                "Whisper primary transcription (lang=%s): %r",
                primary_lang, text[:120],
            )

            # 强制某语言识别失败时，再用自动检测兜底一次
            if not text and primary_lang is not None:
                text = await asyncio.to_thread(_run, None)
                logger.info("Whisper fallback auto-detect transcription: %r", text[:120])

            if not text:
                _save_failed_sample(audio_bytes, mime_type, "empty_transcript")

            return text
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    async def health_check(self) -> bool:
        """仅验证 whisper 库可导入 + ffmpeg 可用；不触发重量级模型加载。"""
        if self._model is not None:
            return True
        try:
            import whisper  # type: ignore  # noqa: F401
        except Exception:
            return False
        return _resolve_ffmpeg_binary() is not None


class StubSTTProvider(STTProvider):
    """Stub provider for development — returns empty string."""

    async def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        logger.warning("Stub STT: returning empty transcription")
        return ""

    async def health_check(self) -> bool:
        return True


def create_stt_provider(provider_type: Optional[str] = None) -> STTProvider:
    """Factory to create the configured STT provider."""
    pt = provider_type or config.STT_PROVIDER

    if pt == "whisper":
        return WhisperProvider()
    elif pt == "stub":
        return StubSTTProvider()
    else:
        logger.warning("Unknown STT provider '%s', falling back to stub", pt)
        return StubSTTProvider()
