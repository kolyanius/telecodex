from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

from telegram import Update

from ..config import Settings
from ..models import PreparedCodexRequest, ProcessedDocument, ProcessedImage
from ..services.observability import ObservabilityService
from ..voice import VoiceTranscriber


class MessageInputPreparer:
    def __init__(
        self,
        settings: Settings,
        voice: VoiceTranscriber,
        observability: ObservabilityService,
        logger: Any,
    ):
        self.settings = settings
        self.voice = voice
        self.observability = observability
        self.logger = logger

    def prepare_text(self, message_text: str) -> PreparedCodexRequest:
        return PreparedCodexRequest(prompt=message_text, source="text")

    async def prepare_document(
        self,
        update: Update,
        request_context,
    ) -> Optional[PreparedCodexRequest]:
        document = update.effective_message.document
        if not self.settings.enable_file_uploads:
            await self.observability.record_event(
                "unsupported_input",
                request_context,
                audit_event="unsupported_input",
                event_status="document_disabled",
            )
            await update.effective_message.reply_text(
                "File uploads are disabled.",
            )
            return None

        tg_file = await document.get_file()
        data = bytes(await tg_file.download_as_bytearray())

        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            await self.observability.record_event(
                "unsupported_input",
                request_context,
                audit_event="unsupported_input",
                event_status="document_not_utf8",
            )
            await update.effective_message.reply_text(
                "Поддерживаются только текстовые файлы UTF-8.",
            )
            return None

        if len(content) > 120_000:
            content = content[:120_000] + "\n\n...[truncated]"

        processed = ProcessedDocument(
            prompt=(
                f"User uploaded file `{document.file_name}`.\n"
                "Please review it, explain what it is, and take the user's caption into account if present.\n\n"
                f"Caption: {update.effective_message.caption or ''}\n\n"
                f"```text\n{content}\n```"
            ),
            filename=document.file_name or "",
        )
        request_context.prompt_chars = len(processed.prompt)
        return PreparedCodexRequest(prompt=processed.prompt, source="document")

    async def prepare_voice(
        self,
        update: Update,
        request_context,
    ) -> Optional[PreparedCodexRequest]:
        if not self.settings.enable_voice_messages:
            await self.observability.record_event(
                "unsupported_input",
                request_context,
                audit_event="unsupported_input",
                event_status="voice_disabled",
            )
            await update.effective_message.reply_text(
                "Voice messages are disabled.",
            )
            return None

        progress = await update.effective_message.reply_text("Transcribing...")
        await self.observability.record_event("voice_transcription_started", request_context)
        try:
            processed = await self.voice.transcribe(
                update.effective_message.voice,
                caption=update.effective_message.caption,
            )
            request_context.prompt_chars = len(processed.prompt)
            await progress.delete()
            return PreparedCodexRequest(prompt=processed.prompt, source="voice")
        except Exception as exc:
            await self.observability.record_event(
                "voice_transcription_failed",
                request_context,
                audit_event="voice_transcription_failed",
                event_status="failed",
                error_message=str(exc),
                level="error",
            )
            await progress.edit_text(f"Voice transcription failed: {exc}")
            return None

    async def prepare_photo(
        self,
        update: Update,
        request_context,
    ) -> Optional[PreparedCodexRequest]:
        if not self.settings.codex_enable_images:
            await self.observability.record_event(
                "unsupported_input",
                request_context,
                audit_event="unsupported_input",
                event_status="image_disabled",
            )
            await update.effective_message.reply_text(
                "Image support is disabled. Enable CODEX_ENABLE_IMAGES=true.",
            )
            return None

        photo = update.effective_message.photo[-1]
        tg_file = await photo.get_file()
        data = bytes(await tg_file.download_as_bytearray())

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(data)
            image_path = Path(tmp.name)

        processed = ProcessedImage(
            prompt=update.effective_message.caption or "Please analyze this image.",
            image_path=image_path,
        )
        request_context.prompt_chars = len(processed.prompt)
        return PreparedCodexRequest(
            prompt=processed.prompt,
            source="photo",
            image_paths=[processed.image_path],
            cleanup_paths=[processed.image_path],
        )
