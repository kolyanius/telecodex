from __future__ import annotations

import structlog
from telegram import BotCommand, MenuButtonCommands, Update
from telegram.ext import (
    AIORateLimiter,
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .codex_runner import CodexRunner
from .config import Settings
from .flows.execution import PromptExecutionFlow
from .flows.navigation import NavigationFlow
from .handlers.callbacks import CallbackHandlers
from .handlers.commands import CommandHandlers
from .handlers.errors import ErrorHandlers
from .handlers.messages import MessageHandlers
from .rate_limiter import RateLimiter
from .services.observability import ObservabilityService
from .services.projects import ProjectService
from .session_store import SessionStore
from .telegram.inputs import MessageInputPreparer
from .telegram.ui.keyboards import build_stop_keyboard
from .telegram.ui.responder import TelegramResponder
from .telegram.ui.texts import build_progress_text
from .voice import VoiceTranscriber


logger = structlog.get_logger(__name__)


class CodexTelegramBot:
    TYPING_HEARTBEAT_SECONDS = 4.0
    PROGRESS_HEARTBEAT_SECONDS = 2.0

    def __init__(self, settings: Settings, session_store: SessionStore):
        self.settings = settings
        self.session_store = session_store
        self.app: Application | None = None

        self._codex = CodexRunner(settings)
        self._voice = VoiceTranscriber(settings)
        self.rate_limiter = RateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        self.observability = ObservabilityService(settings, session_store, logger)
        self.projects = ProjectService(settings, self.observability.record_event)
        self.responder = TelegramResponder(logger)
        self.inputs = MessageInputPreparer(settings, self._voice, self.observability, logger)
        self.execution_flow = PromptExecutionFlow(
            settings,
            session_store,
            self._codex,
            self.rate_limiter,
            self.projects,
            self.observability,
            self.responder,
            logger,
        )
        self.navigation_flow = NavigationFlow(
            settings,
            session_store,
            self.projects,
            self.observability,
            self.responder,
            self.execution_flow,
        )
        self.command_handlers = CommandHandlers(
            self.navigation_flow,
            self.execution_flow,
            self.observability,
        )
        self.message_handlers = MessageHandlers(
            self.inputs,
            self.execution_flow,
            self.observability,
        )
        self.callback_handlers = CallbackHandlers(
            self.navigation_flow,
            self.execution_flow,
            self.observability,
        )
        self.error_handlers = ErrorHandlers(self.observability, logger)

    @property
    def codex(self):
        return self._codex

    @codex.setter
    def codex(self, value) -> None:
        self._codex = value
        self.execution_flow.codex = value

    @property
    def voice(self):
        return self._voice

    @voice.setter
    def voice(self, value) -> None:
        self._voice = value
        self.inputs.voice = value

    @property
    def active_interrupts(self):
        return self.execution_flow.active_interrupts

    async def build(self) -> Application:
        app = (
            Application.builder()
            .token(self.settings.telegram_token_str)
            .rate_limiter(AIORateLimiter())
            .build()
        )
        app.add_handler(CommandHandler("start", self.start_command))
        app.add_handler(CommandHandler("menu", self.menu_command))
        app.add_handler(CommandHandler("controls", self.menu_command))
        app.add_handler(CommandHandler("new", self.new_command))
        app.add_handler(CommandHandler("status", self.status_command))
        app.add_handler(CommandHandler("verbose", self.verbose_command))
        app.add_handler(CommandHandler("repo", self.repo_command))
        app.add_handler(CommandHandler("mode", self.mode_command))
        app.add_handler(CallbackQueryHandler(self.stop_callback, pattern=r"^(?:stop:|action:stop:)"))
        app.add_handler(
            CallbackQueryHandler(
                self.handle_ui_callback,
                pattern=r"^(?:nav:|action:(?:new|create_project)$|verbose:|repo:|mode:|llm:)",
            )
        )
        app.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        app.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        app.add_error_handler(self.error_handler)
        self.app = app
        return app

    async def configure_telegram_ui(self, app: Application | None = None) -> None:
        target_app = app or self.app
        if target_app is None:
            return
        await target_app.bot.set_my_commands(
            [
                BotCommand("start", "Начать работу с ботом"),
                BotCommand("menu", "Открыть текущее меню сессии"),
                BotCommand("repo", "Выбрать или создать проект"),
                BotCommand("mode", "Изменить режим доступа"),
                BotCommand("new", "Начать новую сессию Codex"),
                BotCommand("status", "Показать технический статус"),
            ]
        )
        await target_app.bot.set_chat_menu_button(menu_button=MenuButtonCommands())

    def _sync_runtime_settings(self) -> None:
        self.execution_flow.typing_heartbeat_seconds = self.TYPING_HEARTBEAT_SECONDS
        self.execution_flow.progress_heartbeat_seconds = self.PROGRESS_HEARTBEAT_SECONDS

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.command_handlers.start_command(update, context)

    async def new_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.command_handlers.new_command(update, context)

    async def menu_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.command_handlers.menu_command(update, context)

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.command_handlers.status_command(update, context)

    async def verbose_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.command_handlers.verbose_command(update, context)

    async def repo_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.command_handlers.repo_command(update, context)

    async def mode_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.command_handlers.mode_command(update, context)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._sync_runtime_settings()
        await self.message_handlers.handle_text(update, context)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._sync_runtime_settings()
        await self.message_handlers.handle_document(update, context)

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._sync_runtime_settings()
        await self.message_handlers.handle_voice(update, context)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._sync_runtime_settings()
        await self.message_handlers.handle_photo(update, context)

    async def stop_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.callback_handlers.stop_callback(update, context)

    async def handle_ui_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.callback_handlers.handle_ui_callback(update, context)

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.error_handlers.error_handler(update, context)

    def _is_authorized(self, update: Update) -> bool:
        return self.observability.is_authorized(update)

    def _ensure_in_workspace(self, path) -> None:
        self.projects.ensure_in_workspace(path)

    @staticmethod
    def _chunk_text(text: str, size: int) -> list[str]:
        return TelegramResponder.chunk_text(text, size)

    @staticmethod
    def _build_progress_text(elapsed_seconds: int, last_progress_lines: list[str]) -> str:
        return build_progress_text(elapsed_seconds, last_progress_lines)

    @staticmethod
    def build_stop_keyboard(user_id: int):
        return build_stop_keyboard(user_id)

    async def _typing_heartbeat(self, **kwargs) -> None:
        self._sync_runtime_settings()
        await self.execution_flow.typing_heartbeat(**kwargs)

    async def _progress_heartbeat(self, **kwargs) -> None:
        self._sync_runtime_settings()
        await self.execution_flow.progress_heartbeat(**kwargs)
