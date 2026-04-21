from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from telegram import Update
from telegram.ext import ContextTypes

from ..config import Settings
from ..models import CodexResponse, RequestContext
from ..session_store import SessionStore


class ObservabilityService:
    def __init__(self, settings: Settings, session_store: SessionStore, logger: Any):
        self.settings = settings
        self.session_store = session_store
        self.logger = logger

    async def ensure_authorized(self, update: Update, request_context: RequestContext) -> bool:
        if self.is_authorized(update):
            return True
        await self.record_event(
            "auth_denied",
            request_context,
            audit_event="auth_denied",
            event_status="denied",
            level="warning",
        )
        if update.effective_message:
            await update.effective_message.reply_text("Access denied.")
        return False

    def is_authorized(self, update: Update) -> bool:
        if not self.settings.allowed_users:
            return True
        return bool(update.effective_user and update.effective_user.id in self.settings.allowed_users)

    def make_request_context(
        self,
        update: Update,
        context: Optional[ContextTypes.DEFAULT_TYPE],
        *,
        source: str,
        command_name: str = "",
        prompt_chars: int = 0,
        caption_chars: int = 0,
        document_name: str = "",
        image_count: int = 0,
        voice_duration_seconds: int = 0,
    ) -> RequestContext:
        effective_chat = getattr(update, "effective_chat", None)
        effective_message = getattr(update, "effective_message", None)
        effective_user = getattr(update, "effective_user", None)
        return RequestContext(
            source=source,
            user_id=getattr(effective_user, "id", None),
            chat_id=getattr(effective_chat, "id", None),
            message_id=getattr(effective_message, "message_id", None),
            chat_type=str(getattr(effective_chat, "type", "") or ""),
            cwd=str(self.peek_current_directory(context)),
            command_name=command_name,
            prompt_chars=prompt_chars,
            caption_chars=caption_chars,
            document_name=document_name,
            image_count=image_count,
            voice_duration_seconds=voice_duration_seconds,
        )

    def peek_current_directory(self, context: Optional[ContextTypes.DEFAULT_TYPE]) -> Path:
        if context is None:
            return self.settings.approved_directory.resolve()
        current = context.user_data.get("current_directory")
        if current is None:
            return self.settings.approved_directory.resolve()
        try:
            current_path = Path(current).resolve()
            root = self.settings.approved_directory.resolve()
            current_path.relative_to(root)
            return current_path
        except Exception:
            return self.settings.approved_directory.resolve()

    def context_fields(self, request_context: Optional[RequestContext]) -> dict[str, Any]:
        if request_context is None:
            return {}
        fields: dict[str, Any] = {
            "source": request_context.source,
            "user_id": request_context.user_id,
            "chat_id": request_context.chat_id,
            "message_id": request_context.message_id,
            "chat_type": request_context.chat_type,
            "cwd": request_context.cwd,
            "has_previous_thread": request_context.has_previous_thread,
            "prompt_chars": request_context.prompt_chars,
            "caption_chars": request_context.caption_chars,
            "document_name": request_context.document_name,
            "image_count": request_context.image_count,
            "voice_duration_seconds": request_context.voice_duration_seconds,
        }
        if request_context.launch_mode:
            fields["launch_mode"] = request_context.launch_mode
        if request_context.command_name:
            fields["command_name"] = request_context.command_name
        return fields

    async def record_event(
        self,
        event_name: str,
        request_context: Optional[RequestContext],
        *,
        audit_event: Optional[str] = None,
        event_status: str = "",
        level: str = "info",
        **details: Any,
    ) -> None:
        payload = {**self.context_fields(request_context), **details}
        if event_status:
            payload["event_status"] = event_status
        log_method = getattr(self.logger, level)
        log_method(event_name, **payload)

        if audit_event is None or not self.settings.enable_audit_log:
            return

        await self.session_store.log_audit_event(
            user_id=request_context.user_id if request_context else None,
            chat_id=request_context.chat_id if request_context else None,
            project_path=request_context.cwd if request_context else None,
            event_type=audit_event,
            event_status=event_status,
            details=payload,
        )

    @staticmethod
    def response_fields(response: CodexResponse) -> dict[str, Any]:
        return {
            "thread_id": response.thread_id,
            "status": str(response.status),
            "duration_ms": response.duration_ms,
            "input_tokens": response.input_tokens,
            "cached_input_tokens": response.cached_input_tokens,
            "output_tokens": response.output_tokens,
            "fallback_reason": response.fallback_reason,
            "error_message": response.error_message,
        }
