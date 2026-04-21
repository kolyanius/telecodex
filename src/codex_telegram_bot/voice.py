from __future__ import annotations

from typing import Any, Optional

import structlog
from telegram import Voice

from .config import Settings
from .models import ProcessedVoice

logger = structlog.get_logger(__name__)


class VoiceTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._openai_client: Optional[Any] = None
        self._openai_compatible_client: Optional[Any] = None

    async def transcribe(self, voice: Voice, caption: Optional[str] = None) -> ProcessedVoice:
        initial_file_size = getattr(voice, "file_size", None)
        self._ensure_allowed_file_size(initial_file_size)

        tg_file = await voice.get_file()
        resolved_file_size = getattr(tg_file, "file_size", None)
        self._ensure_allowed_file_size(resolved_file_size)

        if not isinstance(initial_file_size, int) and not isinstance(resolved_file_size, int):
            raise ValueError(
                "Voice message size is unknown. Please retry with a smaller file."
            )

        data = bytes(await tg_file.download_as_bytearray())
        self._ensure_allowed_file_size(len(data))

        logger.info(
            "voice_transcription_started",
            provider=self.settings.voice_provider,
            duration=int(getattr(voice, "duration", 0) or 0),
            file_size=initial_file_size or resolved_file_size or len(data),
        )

        if self.settings.voice_provider == "openai_compatible":
            text = await self._transcribe_openai_compatible(data)
        else:
            text = await self._transcribe_openai(data)

        label = caption or "Voice message transcription:"
        prompt = f"{label}\n\n{text.strip()}"
        return ProcessedVoice(
            prompt=prompt,
            transcription=text.strip(),
            duration_seconds=int(getattr(voice, "duration", 0) or 0),
        )

    def _ensure_allowed_file_size(self, file_size: Optional[int]) -> None:
        if isinstance(file_size, int) and file_size > self.settings.voice_max_file_size_bytes:
            raise ValueError(
                f"Voice file too large: {file_size / 1024 / 1024:.1f}MB. "
                f"Max is {self.settings.voice_max_file_size_mb}MB."
            )

    async def _transcribe_openai(self, data: bytes) -> str:
        client = self._get_openai_client()
        return await self._transcribe_openai_like(
            client=client,
            provider_label="openai",
            data=data,
        )

    async def _transcribe_openai_compatible(self, data: bytes) -> str:
        client = self._get_openai_compatible_client()
        return await self._transcribe_openai_like(
            client=client,
            provider_label="openai_compatible",
            data=data,
        )

    async def _transcribe_openai_like(self, *, client: Any, provider_label: str, data: bytes) -> str:
        try:
            response = await client.audio.transcriptions.create(
                model=self.settings.resolved_voice_model,
                file=("voice.ogg", data),
                temperature=0,
            )
        except Exception as exc:
            logger.warning("voice_transcription_failed", provider=provider_label, error=str(exc))
            if provider_label == "openai":
                message = "OpenAI transcription request failed."
            else:
                message = "OpenAI-compatible transcription request failed."
            raise RuntimeError(message) from exc

        text = (getattr(response, "text", "") or "").strip()
        if not text:
            if provider_label == "openai":
                raise RuntimeError("OpenAI transcription returned empty text")
            raise RuntimeError("OpenAI-compatible transcription returned empty text")
        return text

    def _get_openai_client(self) -> Any:
        if self._openai_client is not None:
            return self._openai_client

        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("Optional dependency 'openai' is not installed.") from exc

        api_key = self.settings.openai_api_key_str
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        self._openai_client = AsyncOpenAI(api_key=api_key)
        return self._openai_client

    def _get_openai_compatible_client(self) -> Any:
        if self._openai_compatible_client is not None:
            return self._openai_compatible_client

        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("Optional dependency 'openai' is not installed.") from exc

        api_key = self.settings.voice_api_key_str
        if not api_key:
            raise RuntimeError("VOICE_API_KEY is not configured")
        base_url = self.settings.voice_api_base_url
        if not base_url:
            raise RuntimeError("VOICE_API_BASE_URL is not configured")

        self._openai_compatible_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        return self._openai_compatible_client
