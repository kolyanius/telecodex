from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any, Optional

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from ..codex_runner import CodexRunner
from ..config import Settings
from ..models import (
    CodexLaunchMode,
    CodexResponse,
    CodexResultStatus,
    CodexStreamEvent,
    PreparedCodexRequest,
)
from ..rate_limiter import RateLimiter
from ..services.observability import ObservabilityService
from ..services.projects import ProjectService
from ..session_store import SessionStore
from ..telegram.ui.keyboards import (
    build_full_access_warning_keyboard,
    build_mode_editor_keyboard,
    build_no_project_keyboard,
    build_stop_keyboard,
)
from ..telegram.ui.responder import TelegramResponder
from ..telegram.ui.texts import (
    build_progress_text,
    render_final_text,
    render_full_access_warning_text,
    render_launch_mode_editor_text,
    render_launch_mode_label,
    render_no_projects_text,
)


class PromptExecutionFlow:
    def __init__(
        self,
        settings: Settings,
        session_store: SessionStore,
        codex: CodexRunner,
        rate_limiter: RateLimiter,
        projects: ProjectService,
        observability: ObservabilityService,
        responder: TelegramResponder,
        logger: Any,
    ):
        self.settings = settings
        self.session_store = session_store
        self.codex = codex
        self.rate_limiter = rate_limiter
        self.projects = projects
        self.observability = observability
        self.responder = responder
        self.logger = logger
        self.typing_heartbeat_seconds = 4.0
        self.progress_heartbeat_seconds = 2.0
        self.active_interrupts: dict[int, asyncio.Event] = {}

    @staticmethod
    def _mode_changed_notice(launch_mode: CodexLaunchMode) -> str:
        return (
            f"Режим доступа изменён на `{render_launch_mode_label(launch_mode)}`. "
            "Следующие запросы в этом проекте будут использовать его."
        )

    async def resolve_launch_mode(
        self,
        *,
        user_id: int,
        project_path: Optional[Path],
    ) -> CodexLaunchMode:
        if project_path is None:
            return CodexLaunchMode.from_value(self.settings.codex_default_launch_mode)
        stored = await self.session_store.get_project_launch_mode(user_id, str(project_path))
        if stored is not None:
            return stored
        return CodexLaunchMode.from_value(self.settings.codex_default_launch_mode)

    async def show_mode_editor(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        edit: bool,
        notice: str = "",
    ) -> None:
        request_context = self.observability.make_request_context(update, context, source="command")
        project = await self.projects.resolve_current_project(context, request_context=request_context)
        if project.path is None:
            text = render_no_projects_text()
            reply_markup = build_no_project_keyboard()
        else:
            launch_mode = await self.resolve_launch_mode(
                user_id=update.effective_user.id,
                project_path=project.path,
            )
            text = render_launch_mode_editor_text(
                project_name=project.path.name,
                launch_mode=launch_mode,
                has_active_run=update.effective_user.id in self.active_interrupts,
                notice=notice,
            )
            reply_markup = build_mode_editor_keyboard(
                launch_mode,
                full_access_confirmed=launch_mode == CodexLaunchMode.FULL_ACCESS,
                back_callback="nav:menu",
            )
        if edit:
            await self.responder.edit_callback_message(
                update,
                text,
                reply_markup=reply_markup,
                parse_mode="Markdown",
            )
            return
        await update.effective_message.reply_text(
            text,
            reply_markup=reply_markup,
            parse_mode="Markdown",
        )

    async def set_launch_mode(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        launch_mode: CodexLaunchMode,
    ) -> None:
        request_context = self.observability.make_request_context(update, context, source="command")
        project = await self.projects.resolve_current_project(context, request_context=request_context)
        if project.path is None:
            await self.responder.edit_callback_message(
                update,
                render_no_projects_text(),
                reply_markup=build_no_project_keyboard(),
                parse_mode="Markdown",
            )
            return
        await self.session_store.set_project_launch_mode(
            update.effective_user.id,
            str(project.path),
            launch_mode,
        )
        await self.show_mode_editor(
            update,
            context,
            edit=True,
            notice=self._mode_changed_notice(launch_mode),
        )

    async def confirm_full_access(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        request_context = self.observability.make_request_context(update, context, source="command")
        project = await self.projects.resolve_current_project(context, request_context=request_context)
        if project.path is None:
            await self.responder.edit_callback_message(
                update,
                render_no_projects_text(),
                reply_markup=build_no_project_keyboard(),
                parse_mode="Markdown",
            )
            return
        await self.responder.edit_callback_message(
            update,
            render_full_access_warning_text(project_name=project.path.name),
            reply_markup=build_full_access_warning_keyboard("mode:show"),
            parse_mode="Markdown",
        )

    async def enable_full_access(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        await self.set_launch_mode(update, context, CodexLaunchMode.FULL_ACCESS)

    async def run_prepared_prompt(
        self,
        *,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        prepared_request: PreparedCodexRequest,
        request_context,
    ) -> None:
        user_id = update.effective_user.id
        if not self.rate_limiter.allow(user_id):
            await self.observability.record_event(
                "codex_request_rate_limited",
                request_context,
                audit_event="request_failed",
                event_status="rate_limited",
            )
            await update.effective_message.reply_text("Rate limit exceeded. Please wait a bit.")
            await self._cleanup_paths(prepared_request.cleanup_paths)
            return

        project = await self.projects.resolve_current_project(context, request_context=request_context)
        if project.path is None:
            await self.observability.record_event(
                "project_create_failed",
                request_context,
                audit_event="project_create_failed",
                event_status="no_project_selected",
            )
            await update.effective_message.reply_text(
                render_no_projects_text(),
                reply_markup=build_no_project_keyboard(),
                parse_mode="Markdown",
            )
            await self._cleanup_paths(prepared_request.cleanup_paths)
            return

        cwd = project.path
        launch_mode = await self.resolve_launch_mode(user_id=user_id, project_path=cwd)
        request_context.cwd = str(cwd)
        request_context.launch_mode = launch_mode.value
        if project.auto_created:
            await update.effective_message.reply_text(
                f"Создал и выбрал первый проект: `{cwd.name}`.",
                parse_mode="Markdown",
            )

        previous_thread_id = await self.resolve_previous_thread_id(
            user_id=user_id,
            cwd=cwd,
            request_context=request_context,
        )
        request_context.has_previous_thread = bool(previous_thread_id)

        await self.observability.record_event(
            "codex_request_started",
            request_context,
            audit_event="request_started",
            event_status="started",
            thread_id=previous_thread_id or "",
        )

        stop_markup = build_stop_keyboard(user_id)
        request_started_at = time.monotonic()
        await update.effective_chat.send_action(ChatAction.TYPING)
        progress = await update.effective_message.reply_text(
            build_progress_text(0, [], project_name=cwd.name),
            reply_markup=stop_markup,
        )

        interrupt_event = asyncio.Event()
        self.active_interrupts[user_id] = interrupt_event
        request_finished = asyncio.Event()

        last_progress_lines: list[str] = []
        tool_count = 0
        first_tool = ""
        saw_text_delta = False

        typing_task = asyncio.create_task(
            self.typing_heartbeat(
                chat=update.effective_chat,
                request_finished=request_finished,
                request_context=request_context,
            )
        )
        progress_task = asyncio.create_task(
            self.progress_heartbeat(
                progress=progress,
                stop_markup=stop_markup,
                request_finished=request_finished,
                request_started_at=request_started_at,
                last_progress_lines=last_progress_lines,
                interrupt_event=interrupt_event,
                request_context=request_context,
            )
        )

        async def on_event(event: CodexStreamEvent) -> None:
            nonlocal last_progress_lines, tool_count, first_tool, saw_text_delta
            if event.tool_call:
                tool_count += 1
                if not first_tool:
                    first_tool = event.tool_call.name
                    await self.observability.record_event(
                        "codex_request_progress",
                        request_context,
                        first_tool=first_tool,
                        tool_count=tool_count,
                    )
                line = f"🔧 {event.tool_call.name}"
            elif event.text_delta:
                saw_text_delta = True
                if int(context.user_data.get("verbose_level", self.settings.verbose_level)) >= 2:
                    line = f"💬 {event.text_delta[-100:]}"
                else:
                    return
            elif event.text_snapshot:
                snippet = event.text_snapshot.strip().splitlines()[0][:100]
                line = f"💬 {snippet}" if snippet else ""
            elif event.usage:
                await self.observability.record_event(
                    "codex_request_progress",
                    request_context,
                    tool_count=tool_count,
                    saw_text_delta=saw_text_delta,
                    input_tokens=event.usage.get("input_tokens", 0),
                    cached_input_tokens=event.usage.get("cached_input_tokens", 0),
                    output_tokens=event.usage.get("output_tokens", 0),
                )
                return
            else:
                return

            if not line:
                return

            verbose_level = int(context.user_data.get("verbose_level", self.settings.verbose_level))
            if verbose_level == 0:
                return

            last_progress_lines.append(line)
            last_progress_lines = last_progress_lines[-12:]
            try:
                await progress.edit_text(
                    build_progress_text(
                        int(time.monotonic() - request_started_at),
                        last_progress_lines,
                        project_name=cwd.name,
                    ),
                    reply_markup=stop_markup if not interrupt_event.is_set() else None,
                )
            except Exception:
                self.logger.debug(
                    "telegram_progress_edit_failed",
                    **self.observability.context_fields(request_context),
                )

        try:
            response = await self.codex.run(
                prompt=prepared_request.prompt,
                cwd=cwd,
                launch_mode=launch_mode,
                previous_thread_id=previous_thread_id,
                on_event=on_event,
                interrupt_event=interrupt_event,
                image_paths=prepared_request.image_paths or None,
            )
        except Exception as exc:
            self.active_interrupts.pop(user_id, None)
            self.logger.exception(
                "codex_request_failed_exception",
                **self.observability.context_fields(request_context),
            )
            if previous_thread_id:
                await self.session_store.update_session_result(
                    user_id,
                    str(cwd),
                    last_status="exception",
                    last_error=str(exc),
                )
            await self.observability.record_event(
                "codex_request_failed",
                request_context,
                audit_event="request_failed",
                event_status="exception",
                error_message=str(exc),
                level="error",
            )
            try:
                request_finished.set()
                typing_task.cancel()
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await typing_task
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task
                await progress.delete()
            except Exception:
                self.logger.debug(
                    "telegram_progress_delete_failed",
                    **self.observability.context_fields(request_context),
                )
            await update.effective_message.reply_text(f"Request failed: {exc}")
            await self._cleanup_paths(prepared_request.cleanup_paths)
            return
        finally:
            request_finished.set()
            typing_task.cancel()
            progress_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task
            with contextlib.suppress(asyncio.CancelledError):
                await progress_task
            self.active_interrupts.pop(user_id, None)

        if response.fallback_reason:
            await self.observability.record_event(
                "codex_resume_fallback_used",
                request_context,
                audit_event="resume_fallback",
                event_status="used",
                fallback_reason=response.fallback_reason,
                thread_id=response.thread_id,
            )

        await self.persist_session_result(
            user_id=user_id,
            project_path=str(cwd),
            previous_thread_id=previous_thread_id,
            response=response,
        )

        try:
            await progress.delete()
        except Exception:
            self.logger.debug(
                "telegram_progress_delete_failed",
                **self.observability.context_fields(request_context),
            )

        if response.status == CodexResultStatus.INTERRUPTED:
            await self.observability.record_event(
                "codex_user_interrupt_completed",
                request_context,
                audit_event="request_interrupted",
                event_status=str(response.status),
                **self.observability.response_fields(response),
            )
        elif response.status == CodexResultStatus.SUCCESS:
            await self.observability.record_event(
                "codex_request_finished",
                request_context,
                audit_event="request_finished",
                event_status=str(response.status),
                **self.observability.response_fields(response),
            )
        else:
            await self.observability.record_event(
                "codex_request_failed",
                request_context,
                audit_event="request_failed",
                event_status=str(response.status),
                level="error",
                **self.observability.response_fields(response),
            )

        final_text = render_final_text(response)
        token_line = (
            f"\n\n`[input={response.input_tokens}, cached={response.cached_input_tokens}, "
            f"output={response.output_tokens}]`"
            if int(context.user_data.get("verbose_level", self.settings.verbose_level)) >= 1
            else ""
        )

        await self.responder.send_final_response(
            update=update,
            markdown_text=final_text + token_line,
        )
        await self._cleanup_paths(prepared_request.cleanup_paths)

    async def persist_session_result(
        self,
        *,
        user_id: int,
        project_path: str,
        previous_thread_id: Optional[str],
        response: CodexResponse,
    ) -> None:
        if response.thread_id:
            await self.session_store.upsert_session(
                user_id,
                project_path,
                response.thread_id,
                last_status=str(response.status),
                last_error=response.error_message,
            )
            return
        if previous_thread_id:
            await self.session_store.update_session_result(
                user_id,
                project_path,
                last_status=str(response.status),
                last_error=response.error_message,
            )

    async def resolve_previous_thread_id(
        self,
        *,
        user_id: int,
        cwd: Path,
        request_context,
    ) -> Optional[str]:
        stored_thread_id = await self.session_store.get_thread_id(user_id, str(cwd))
        if stored_thread_id:
            return stored_thread_id

        reset_at_unix = await self.session_store.get_session_reset_at_unix(user_id, str(cwd))
        discovered_thread_id = await asyncio.to_thread(
            self.codex.discover_latest_session_id,
            cwd,
            modified_after=reset_at_unix,
        )
        if not discovered_thread_id:
            return None

        await self.session_store.upsert_session(
            user_id,
            str(cwd),
            discovered_thread_id,
            last_status="discovered",
        )
        await self.observability.record_event(
            "codex_session_discovered",
            request_context,
            audit_event="session_discovered",
            event_status="discovered",
            thread_id=discovered_thread_id,
        )
        return discovered_thread_id

    async def stop_callback(self, update: Update, request_context) -> None:
        query = update.callback_query
        target_user = int(query.data.rsplit(":", 1)[1])
        interrupt = self.active_interrupts.get(target_user)
        await self.observability.record_event(
            "codex_user_interrupt_requested",
            request_context,
            target_user=target_user,
            has_active_interrupt=interrupt is not None,
        )
        if query.from_user.id != target_user:
            await query.answer("You can only stop your own request.", show_alert=True)
            return
        if interrupt is None:
            await query.answer("Already finished.")
            return
        interrupt.set()
        await self.observability.record_event(
            "request_interrupted",
            request_context,
            audit_event="request_interrupted",
            event_status="requested",
            target_user=target_user,
        )
        await query.answer("Stopping...")

    async def typing_heartbeat(
        self,
        *,
        chat: Any,
        request_finished: asyncio.Event,
        request_context,
    ) -> None:
        while not request_finished.is_set():
            try:
                await asyncio.sleep(self.typing_heartbeat_seconds)
                if request_finished.is_set():
                    return
                await chat.send_action(ChatAction.TYPING)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.debug(
                    "telegram_typing_send_failed",
                    **self.observability.context_fields(request_context),
                )

    async def progress_heartbeat(
        self,
        *,
        progress: Any,
        stop_markup,
        request_finished: asyncio.Event,
        request_started_at: float,
        last_progress_lines: list[str],
        interrupt_event: asyncio.Event,
        request_context,
    ) -> None:
        while not request_finished.is_set():
            try:
                await asyncio.sleep(self.progress_heartbeat_seconds)
                if request_finished.is_set():
                    return
                await progress.edit_text(
                    build_progress_text(
                        int(time.monotonic() - request_started_at),
                        last_progress_lines,
                        project_name=Path(request_context.cwd).name if request_context.cwd else "",
                    ),
                    reply_markup=stop_markup if not interrupt_event.is_set() else None,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                message = str(exc)
                if "Message is not modified" in message:
                    self.logger.debug(
                        "telegram_progress_noop",
                        **self.observability.context_fields(request_context),
                    )
                else:
                    self.logger.debug(
                        "telegram_progress_edit_failed",
                        **self.observability.context_fields(request_context),
                    )

    async def _cleanup_paths(self, paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                self.logger.warning("photo_tempfile_cleanup_failed", path=str(path))
