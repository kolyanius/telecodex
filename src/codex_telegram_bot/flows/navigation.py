from __future__ import annotations

from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

from ..config import Settings
from ..models import CodexLaunchMode
from ..services.observability import ObservabilityService
from ..services.projects import ProjectService
from ..session_store import SessionStore
from ..telegram.ui.keyboards import (
    build_navigation_keyboard,
    build_no_project_keyboard,
    build_repo_keyboard,
    build_session_keyboard,
    build_verbose_keyboard,
)
from ..telegram.ui.responder import TelegramResponder
from ..telegram.ui.texts import (
    render_home_text,
    render_no_projects_text,
    render_project_created_text,
    render_project_selected_text,
    render_repo_picker_text,
    render_session_text,
    render_start_chat_text,
    render_status_text,
    render_verbose_text,
)


class NavigationFlow:
    def __init__(
        self,
        settings: Settings,
        session_store: SessionStore,
        projects: ProjectService,
        observability: ObservabilityService,
        responder: TelegramResponder,
        execution: Any,
    ):
        self.settings = settings
        self.session_store = session_store
        self.projects = projects
        self.observability = observability
        self.responder = responder
        self.execution = execution

    async def _resolve_launch_mode(self, user_id: int, cwd) -> CodexLaunchMode:
        if cwd is None:
            return CodexLaunchMode.from_value(self.settings.codex_default_launch_mode)
        stored = await self.session_store.get_project_launch_mode(user_id, str(cwd))
        if stored is not None:
            return stored
        return CodexLaunchMode.from_value(self.settings.codex_default_launch_mode)

    async def show_home(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        request_context,
    ) -> None:
        project = await self.projects.resolve_current_project(context, request_context=request_context)
        llm = await self.execution.resolve_llm_preferences(user_id=update.effective_user.id)
        await update.effective_message.reply_text(
            render_home_text(
                project.path,
                auto_created=project.auto_created,
                model_label=llm.model_label,
                reasoning_label=llm.reasoning_effort.display_label,
            ),
            reply_markup=build_no_project_keyboard() if project.path is None else None,
            parse_mode="Markdown",
        )

    async def show_menu(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        request_context,
        *,
        edit: bool = False,
        notice: str = "",
    ) -> None:
        project = await self.projects.resolve_current_project(context, request_context=request_context)
        if project.path is None:
            text = render_no_projects_text()
            reply_markup = build_no_project_keyboard()
        else:
            session = await self.session_store.get_session(update.effective_user.id, str(project.path))
            launch_mode = await self._resolve_launch_mode(update.effective_user.id, project.path)
            llm = await self.execution.resolve_llm_preferences(user_id=update.effective_user.id)
            text = render_session_text(
                cwd=project.path,
                launch_mode=launch_mode,
                model_label=llm.model_label,
                reasoning_label=llm.reasoning_effort.display_label,
                has_session=session is not None,
                has_active_run=update.effective_user.id in self.execution.active_interrupts,
                auto_created=project.auto_created,
                notice=notice,
            )
            reply_markup = build_session_keyboard()
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

    async def show_start_chat(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        request_context,
    ) -> None:
        project = await self.projects.resolve_current_project(context, request_context=request_context)
        launch_mode = await self._resolve_launch_mode(update.effective_user.id, project.path)
        llm = await self.execution.resolve_llm_preferences(user_id=update.effective_user.id)
        await self.responder.edit_callback_message(
            update,
            render_start_chat_text(
                project.path,
                auto_created=project.auto_created,
                launch_mode=launch_mode,
                model_label=llm.model_label,
                reasoning_label=llm.reasoning_effort.display_label,
            ),
            reply_markup=build_no_project_keyboard() if project.path is None else None,
            parse_mode="Markdown",
        )

    async def start_new_session(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        request_context,
        *,
        edit: bool = False,
    ) -> None:
        project = await self.projects.resolve_current_project(context, request_context=request_context)
        if project.path is None:
            text = "Сначала создай проект."
            if edit:
                await self.responder.edit_callback_message(
                    update,
                    text,
                    reply_markup=build_no_project_keyboard(),
                    parse_mode="Markdown",
                )
            else:
                await update.effective_message.reply_text(
                    text,
                    reply_markup=build_no_project_keyboard(),
                    parse_mode="Markdown",
                )
            return

        cwd = project.path
        await self.session_store.clear_session(update.effective_user.id, str(cwd))
        notice = f"Новая сессия для `{cwd.name}` готова."
        await self.show_menu(update, context, request_context, edit=edit, notice=notice)

    async def show_status(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        request_context,
        *,
        edit: bool = False,
    ) -> None:
        project = await self.projects.resolve_current_project(context, request_context=request_context)
        cwd = project.path
        launch_mode = await self._resolve_launch_mode(update.effective_user.id, cwd)
        session = (
            await self.session_store.get_session(update.effective_user.id, str(cwd))
            if cwd is not None
            else None
        )
        llm = await self.execution.resolve_llm_preferences(user_id=update.effective_user.id)
        verbose = int(context.user_data.get("verbose_level", self.settings.verbose_level))
        text = render_status_text(
            self.settings,
            cwd,
            session,
            verbose,
            auto_created=project.auto_created,
            launch_mode=launch_mode,
            model_label=llm.model_label,
            reasoning_label=llm.reasoning_effort.display_label,
        )
        reply_markup = build_no_project_keyboard() if cwd is None else None
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

    async def show_verbose(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        edit: bool = False,
    ) -> None:
        current = int(context.user_data.get("verbose_level", self.settings.verbose_level))
        text = render_verbose_text(current)
        reply_markup = build_verbose_keyboard(current)
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

    async def set_verbose(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        request_context,
        level: int,
    ) -> None:
        previous_level = int(context.user_data.get("verbose_level", self.settings.verbose_level))
        context.user_data["verbose_level"] = level
        await self.observability.record_event(
            "telegram_verbose_selected",
            request_context,
            audit_event="telegram_verbose_selected",
            previous_verbose_level=previous_level,
            new_verbose_level=level,
        )
        await self.responder.edit_callback_message(
            update,
            render_verbose_text(level),
            reply_markup=build_verbose_keyboard(level),
            parse_mode="Markdown",
        )

    async def handle_repo_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        request_context,
        *,
        command_text: str,
    ) -> None:
        args = command_text.split()[1:]
        base = self.settings.approved_directory.resolve()
        if args and args[0] == "new":
            if len(args) < 2:
                await update.effective_message.reply_text(
                    "Используй `/repo new <name>` или кнопку `➕ Создать проект`.",
                    reply_markup=build_no_project_keyboard(),
                    parse_mode="Markdown",
                )
                return
            try:
                project = await self.projects.create_project(
                    " ".join(args[1:]),
                    context=context,
                    request_context=request_context,
                )
            except Exception as exc:
                await update.effective_message.reply_text(
                    f"Не удалось создать проект: {exc}",
                    reply_markup=build_no_project_keyboard(),
                    parse_mode="Markdown",
                )
                return
            await self.show_menu(
                update,
                context,
                request_context,
                notice=render_project_created_text(project),
            )
            return
        if args:
            candidate = (base / args[0]).resolve()
            self.projects.ensure_in_workspace(candidate)
            if not candidate.exists() or not candidate.is_dir():
                await update.effective_message.reply_text(
                    f"Проект не найден: `{candidate.name}`",
                    reply_markup=build_navigation_keyboard(),
                    parse_mode="Markdown",
                )
                return
            context.user_data["current_directory"] = candidate
            await self.show_menu(
                update,
                context,
                request_context,
                notice=render_project_selected_text(candidate, base),
            )
            return

        await self.show_repo_picker(update, context, request_context, edit=False)

    async def show_repo_picker(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        request_context,
        *,
        edit: bool,
    ) -> None:
        project = await self.projects.resolve_current_project(context, request_context=request_context)
        options, truncated = self.projects.list_repo_options(context)
        await self.observability.record_event(
            "telegram_repo_picker_opened",
            request_context,
            audit_event="telegram_repo_picker_opened",
            project_count=len(options),
            truncated=truncated,
        )
        if not options:
            text = render_no_projects_text()
            reply_markup = build_no_project_keyboard()
        else:
            text = render_repo_picker_text(options, truncated, auto_created=project.auto_created)
            reply_markup = build_repo_keyboard(options)

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

    async def create_project_from_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        request_context,
    ) -> None:
        query = update.callback_query
        try:
            project = await self.projects.create_project(
                None,
                context=context,
                request_context=request_context,
                auto=True,
            )
        except Exception as exc:
            await query.answer("Create failed.", show_alert=True)
            await self.responder.edit_callback_message(
                update,
                f"Не удалось создать проект: `{str(exc)[:160]}`",
                reply_markup=build_no_project_keyboard(),
                parse_mode="Markdown",
            )
            return
        await query.answer(f"Создан {project.name}")
        await self.show_menu(
            update,
            context,
            request_context,
            edit=True,
            notice=render_project_created_text(project),
        )

    async def select_repo_from_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        request_context,
        slug: str,
    ) -> None:
        query = update.callback_query
        current_resolution = await self.projects.resolve_current_project(
            context,
            request_context=request_context,
            create_if_empty=False,
        )
        previous_dir = current_resolution.path or self.settings.approved_directory.resolve()
        try:
            selected_dir = self.projects.resolve_repo_slug(slug)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            await query.answer("Project unavailable.", show_alert=True)
            options, truncated = self.projects.list_repo_options(context)
            await self.responder.edit_callback_message(
                update,
                render_repo_picker_text(options, truncated) if options else render_no_projects_text(),
                reply_markup=build_repo_keyboard(options) if options else build_no_project_keyboard(),
                parse_mode="Markdown",
            )
            return
        context.user_data["current_directory"] = selected_dir
        await self.observability.record_event(
            "telegram_repo_selected",
            request_context,
            audit_event="telegram_repo_selected",
            previous_project=previous_dir.name,
            selected_project=selected_dir.name,
        )
        await query.answer(f"Переключено: {selected_dir.name}")
        await self.show_menu(
            update,
            context,
            request_context,
            edit=True,
            notice=render_project_selected_text(selected_dir, self.settings.approved_directory),
        )
