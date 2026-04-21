from __future__ import annotations

from telegram import Update
from telegram.ext import ContextTypes

from ..flows.execution import PromptExecutionFlow
from ..services.observability import ObservabilityService
from ..telegram.inputs import MessageInputPreparer


class MessageHandlers:
    def __init__(
        self,
        preparer: MessageInputPreparer,
        execution: PromptExecutionFlow,
        observability: ObservabilityService,
    ):
        self.preparer = preparer
        self.execution = execution
        self.observability = observability

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        prompt = update.effective_message.text or ""
        request_context = self.observability.make_request_context(
            update,
            context,
            source="text",
            prompt_chars=len(prompt),
        )
        await self.observability.record_event("telegram_update_received", request_context)
        if not await self.observability.ensure_authorized(update, request_context):
            return
        prepared = self.preparer.prepare_text(prompt)
        await self.execution.run_prepared_prompt(
            update=update,
            context=context,
            prepared_request=prepared,
            request_context=request_context,
        )

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        document = update.effective_message.document
        request_context = self.observability.make_request_context(
            update,
            context,
            source="document",
            document_name=document.file_name or "",
            caption_chars=len(update.effective_message.caption or ""),
        )
        await self.observability.record_event("telegram_update_received", request_context)
        await self.observability.record_event("document_received", request_context)
        if not await self.observability.ensure_authorized(update, request_context):
            return
        prepared = await self.preparer.prepare_document(update, request_context)
        if prepared is None:
            return
        await self.execution.run_prepared_prompt(
            update=update,
            context=context,
            prepared_request=prepared,
            request_context=request_context,
        )

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        voice = update.effective_message.voice
        request_context = self.observability.make_request_context(
            update,
            context,
            source="voice",
            caption_chars=len(update.effective_message.caption or ""),
            voice_duration_seconds=int(getattr(voice, "duration", 0) or 0),
        )
        await self.observability.record_event("telegram_update_received", request_context)
        await self.observability.record_event("voice_received", request_context)
        if not await self.observability.ensure_authorized(update, request_context):
            return
        prepared = await self.preparer.prepare_voice(update, request_context)
        if prepared is None:
            return
        await self.execution.run_prepared_prompt(
            update=update,
            context=context,
            prepared_request=prepared,
            request_context=request_context,
        )

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        request_context = self.observability.make_request_context(
            update,
            context,
            source="photo",
            caption_chars=len(update.effective_message.caption or ""),
            image_count=len(update.effective_message.photo or []),
        )
        await self.observability.record_event("telegram_update_received", request_context)
        await self.observability.record_event("photo_received", request_context)
        if not await self.observability.ensure_authorized(update, request_context):
            return
        prepared = await self.preparer.prepare_photo(update, request_context)
        if prepared is None:
            return
        await self.execution.run_prepared_prompt(
            update=update,
            context=context,
            prepared_request=prepared,
            request_context=request_context,
        )
