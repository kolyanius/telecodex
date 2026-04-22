from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_telegram_bot import main as main_module
from codex_telegram_bot.bot import CodexTelegramBot
from codex_telegram_bot.config import Settings
from codex_telegram_bot.models import (
    CodexLaunchMode,
    CodexResponse,
    CodexResultStatus,
    LocalCodexSession,
    RequestContext,
)
from codex_telegram_bot.session_store import SessionStore


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "telegram_bot_token": "token",
        "telegram_bot_username": "codex_bot",
        "approved_directory": tmp_path,
        "allowed_users": "42",
    }
    values.update(overrides)
    return Settings(**values)


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def info(self, event: str, **kwargs) -> None:
        self.events.append(("info", event, kwargs))

    def warning(self, event: str, **kwargs) -> None:
        self.events.append(("warning", event, kwargs))

    def error(self, event: str, **kwargs) -> None:
        self.events.append(("error", event, kwargs))

    def debug(self, event: str, **kwargs) -> None:
        self.events.append(("debug", event, kwargs))

    def exception(self, event: str, **kwargs) -> None:
        self.events.append(("exception", event, kwargs))


def attach_fake_logger(bot: CodexTelegramBot, fake_logger: FakeLogger) -> None:
    bot.observability.logger = fake_logger
    bot.responder.logger = fake_logger
    bot.execution_flow.logger = fake_logger
    bot.error_handlers.logger = fake_logger
    bot.inputs.logger = fake_logger


class FakeProgressMessage:
    def __init__(self) -> None:
        self.edits: list[tuple[str, dict]] = []
        self.deleted = False

    async def edit_text(self, text: str, **kwargs) -> None:
        self.edits.append((text, kwargs))

    async def delete(self) -> None:
        self.deleted = True


class FakeSentMessage:
    def __init__(self, text: str, progress: FakeProgressMessage, kwargs: dict) -> None:
        self.text = text
        self.progress = progress
        self.kwargs = kwargs


class FakeMessage:
    def __init__(
        self,
        *,
        text: str = "",
        caption: str = "",
        message_id: int = 1,
        document=None,
        voice=None,
        photo=None,
    ) -> None:
        self.text = text
        self.caption = caption
        self.message_id = message_id
        self.document = document
        self.voice = voice
        self.photo = photo or []
        self.replies: list[FakeSentMessage] = []

    async def reply_text(self, text: str, **kwargs):
        progress = FakeProgressMessage()
        self.replies.append(FakeSentMessage(text, progress, kwargs))
        return progress


class HtmlRejectingMessage(FakeMessage):
    async def reply_text(self, text: str, **kwargs):
        if kwargs.get("parse_mode") == "HTML":
            raise RuntimeError("Can't parse entities: unsupported start tag")
        return await super().reply_text(text, **kwargs)


class FakeChat:
    def __init__(self, chat_id: int = 10, chat_type: str = "private") -> None:
        self.id = chat_id
        self.type = chat_type
        self.actions: list[str] = []

    async def send_action(self, action: str) -> None:
        self.actions.append(action)


class FakeUser:
    def __init__(self, user_id: int) -> None:
        self.id = user_id


class FakeUpdate:
    def __init__(
        self,
        *,
        user_id: int = 42,
        text: str = "",
        caption: str = "",
        chat_id: int = 10,
        message_id: int = 1,
        callback_query=None,
    ) -> None:
        self.effective_user = FakeUser(user_id)
        self.effective_chat = FakeChat(chat_id=chat_id)
        self.callback_query = callback_query
        self.effective_message = (
            callback_query.message
            if callback_query is not None
            else FakeMessage(text=text, caption=caption, message_id=message_id)
        )


class FakeCallbackQuery:
    def __init__(self, from_user_id: int, data: str, message: FakeMessage | None = None) -> None:
        self.from_user = FakeUser(from_user_id)
        self.data = data
        self.answers: list[tuple[str, bool]] = []
        self.edits: list[tuple[str, dict]] = []
        self.message = message or FakeMessage()

    async def answer(self, text: str = "", show_alert: bool = False) -> None:
        self.answers.append((text, show_alert))

    async def edit_message_text(self, text: str, **kwargs) -> None:
        self.edits.append((text, kwargs))


class NoopEditCallbackQuery(FakeCallbackQuery):
    async def edit_message_text(self, text: str, **kwargs) -> None:
        raise RuntimeError(
            "Message is not modified: specified new message content and reply markup are exactly the same as a current content and reply markup of the message"
        )


class FakeContext:
    def __init__(self) -> None:
        self.user_data: dict = {}


class FakeCodex:
    def __init__(
        self,
        response: CodexResponse,
        discovered_session_id: str | None = None,
        local_sessions: list[LocalCodexSession] | None = None,
    ) -> None:
        self.response = response
        self.discovered_session_id = discovered_session_id
        self.local_sessions = local_sessions or []
        self.calls: list[dict] = []
        self.discovery_calls: list[dict] = []
        self.local_discovery_calls: list[dict] = []

    async def run(self, **kwargs) -> CodexResponse:
        self.calls.append(kwargs)
        return self.response

    def discover_latest_session_id(
        self,
        cwd: Path,
        *,
        modified_after: float | None = None,
    ) -> str | None:
        self.discovery_calls.append({"cwd": cwd, "modified_after": modified_after})
        return self.discovered_session_id

    def discover_local_sessions(
        self,
        cwd: Path,
        *,
        limit: int | None = None,
    ) -> list[LocalCodexSession]:
        self.local_discovery_calls.append({"cwd": cwd, "limit": limit})
        if limit is None:
            return list(self.local_sessions)
        return list(self.local_sessions[:limit])

    def validate_cli_available(self) -> None:
        return None


class SlowFakeCodex(FakeCodex):
    def __init__(self, response: CodexResponse, delay_seconds: float) -> None:
        super().__init__(response)
        self.delay_seconds = delay_seconds

    async def run(self, **kwargs) -> CodexResponse:
        await asyncio.sleep(self.delay_seconds)
        return self.response


def keyboard_callback_data(markup) -> list[list[str]]:
    return [[button.callback_data for button in row] for row in markup.inline_keyboard]


def make_local_session(session_id: str, project_dir: Path, prompt: str = "Old prompt") -> LocalCodexSession:
    timestamp = datetime(2026, 4, 22, 18, 30)
    return LocalCodexSession(
        session_id=session_id,
        cwd=project_dir,
        created_at=timestamp,
        updated_at=timestamp,
        source_path=project_dir / f"{session_id}.jsonl",
        first_prompt=prompt,
    )


@pytest.mark.asyncio
async def test_auth_denied_creates_audit_and_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path, allowed_users="7")
    bot = CodexTelegramBot(settings, store)
    fake_logger = FakeLogger()
    attach_fake_logger(bot, fake_logger)

    update = FakeUpdate(user_id=42)
    context = FakeContext()

    await bot.start_command(update, context)

    assert update.effective_message.replies[0].text == "Access denied."
    rows = await store.conn.execute_fetchall("SELECT event_type FROM audit_log ORDER BY id DESC")
    assert rows[0][0] == "auth_denied"
    assert any(event == "auth_denied" for _, event, _ in fake_logger.events)
    await store.close()


@pytest.mark.asyncio
async def test_start_command_returns_home_text_without_reply_keyboard(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    update = FakeUpdate(user_id=42)
    context = FakeContext()

    await bot.start_command(update, context)

    reply = update.effective_message.replies[-1]
    assert "Codex Telegram Bot" in reply.text
    assert "/menu" in reply.text
    assert "reply_markup" not in reply.kwargs or reply.kwargs["reply_markup"] is None
    await store.close()


@pytest.mark.asyncio
async def test_start_command_auto_creates_first_project_in_empty_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    monkeypatch.setattr(bot.projects, "default_project_slug", lambda: "2026-04-18-project")

    update = FakeUpdate(user_id=42)
    context = FakeContext()
    await bot.start_command(update, context)

    reply = update.effective_message.replies[-1]
    assert "Автоматически создал первый проект" in reply.text
    assert Path(context.user_data["current_directory"]).name == "2026-04-18-project"
    assert (tmp_path / "2026-04-18-project").is_dir()
    await store.close()


@pytest.mark.asyncio
async def test_status_command_returns_diagnostic_status_without_keyboard(tmp_path: Path) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    await store.upsert_session(42, str(project_dir), "thread-abc", last_status="success")

    update = FakeUpdate(user_id=42)
    context = FakeContext()
    context.user_data["current_directory"] = project_dir
    await bot.status_command(update, context)

    reply = update.effective_message.replies[-1]
    assert "thread-abc" in reply.text
    assert "Модель:" in reply.text
    assert "Контекст:" in reply.text
    assert "Режим доступа: `Песочница`" in reply.text
    assert "reply_markup" not in reply.kwargs or reply.kwargs["reply_markup"] is None
    await store.close()


@pytest.mark.asyncio
async def test_verbose_command_without_arg_returns_selector(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)

    update = FakeUpdate(user_id=42, text="/verbose")
    context = FakeContext()
    await bot.verbose_command(update, context)

    reply = update.effective_message.replies[-1]
    assert "Текущий verbose level" in reply.text
    assert keyboard_callback_data(reply.kwargs["reply_markup"]) == [
        ["verbose:set:0", "verbose:set:1", "verbose:set:2"],
        ["nav:menu"],
    ]
    await store.close()


@pytest.mark.asyncio
async def test_verbose_callback_updates_level_and_audits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    fake_logger = FakeLogger()
    attach_fake_logger(bot, fake_logger)

    callback_query = FakeCallbackQuery(from_user_id=42, data="verbose:set:2")
    update = FakeUpdate(user_id=42, callback_query=callback_query)
    context = FakeContext()
    context.user_data["verbose_level"] = 1

    await bot.handle_ui_callback(update, context)

    assert context.user_data["verbose_level"] == 2
    assert "Текущий verbose level: `2`" in callback_query.edits[-1][0]
    rows = await store.conn.execute_fetchall(
        "SELECT event_type FROM audit_log WHERE event_type = 'telegram_verbose_selected'"
    )
    assert len(rows) == 1
    assert any(event == "telegram_verbose_selected" for _, event, _ in fake_logger.events)
    await store.close()


@pytest.mark.asyncio
async def test_start_chat_callback_opens_chat_ready_screen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    fake_logger = FakeLogger()
    attach_fake_logger(bot, fake_logger)

    callback_query = FakeCallbackQuery(from_user_id=42, data="nav:start")
    update = FakeUpdate(user_id=42, callback_query=callback_query)
    context = FakeContext()

    await bot.handle_ui_callback(update, context)

    assert "Готов к работе" in callback_query.edits[-1][0]
    assert "Режим доступа: `Песочница`" in callback_query.edits[-1][0]
    assert "/menu" in callback_query.edits[-1][0]
    assert callback_query.edits[-1][1]["reply_markup"] is None
    assert any(event == "telegram_nav_start" for _, event, _ in fake_logger.events)
    await store.close()


@pytest.mark.asyncio
async def test_menu_command_opens_session_card(tmp_path: Path) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    update = FakeUpdate(user_id=42, text="/menu")
    context = FakeContext()
    context.user_data["current_directory"] = project_dir

    await bot.menu_command(update, context)

    reply = update.effective_message.replies[-1]
    assert "Текущая сессия." in reply.text
    assert "Режим доступа: `Песочница`" in reply.text
    assert keyboard_callback_data(reply.kwargs["reply_markup"]) == [
        ["nav:repo", "mode:show"],
        ["session:list"],
        ["action:new"],
    ]
    await store.close()


@pytest.mark.asyncio
async def test_sessions_command_lists_project_sessions(tmp_path: Path) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    local_session = make_local_session("old-session", project_dir, "Review the API handlers")
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    bot.codex = FakeCodex(
        CodexResponse(final_text="done", thread_id="old-session", status=CodexResultStatus.SUCCESS),
        local_sessions=[local_session],
    )

    update = FakeUpdate(user_id=42, text="/sessions")
    context = FakeContext()
    context.user_data["current_directory"] = project_dir

    await bot.sessions_command(update, context)

    reply = update.effective_message.replies[-1]
    assert "Сессии проекта." in reply.text
    assert "Проект: `app`" in reply.text
    assert keyboard_callback_data(reply.kwargs["reply_markup"]) == [
        ["session:select:old-session"],
        ["session:refresh", "action:new"],
        ["nav:menu"],
    ]
    assert bot.codex.local_discovery_calls[-1] == {"cwd": project_dir, "limit": 10}
    await store.close()


@pytest.mark.asyncio
async def test_session_select_callback_saves_thread_and_next_prompt_resumes(tmp_path: Path) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    local_session = make_local_session("old-session", project_dir, "Continue this work")
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    bot.codex = FakeCodex(
        CodexResponse(final_text="done", thread_id="old-session", status=CodexResultStatus.SUCCESS),
        local_sessions=[local_session],
    )

    context = FakeContext()
    context.user_data["current_directory"] = project_dir
    callback_query = FakeCallbackQuery(from_user_id=42, data="session:select:old-session")
    update = FakeUpdate(user_id=42, callback_query=callback_query)

    await bot.handle_ui_callback(update, context)

    session = await store.get_session(42, str(project_dir))
    assert session is not None
    assert session.thread_id == "old-session"
    assert session.last_status == "selected"
    assert callback_query.answers[-1] == ("Сессия выбрана", False)
    assert "Выбрана сессия `old-sess`." in callback_query.edits[-1][0]

    status_update = FakeUpdate(user_id=42, text="/status")
    await bot.status_command(status_update, context)
    assert "Thread ID: `old-session`" in status_update.effective_message.replies[-1].text

    text_update = FakeUpdate(user_id=42, text="continue")
    text_update.effective_message = FakeMessage(text="continue", message_id=2)
    await bot.handle_text(text_update, context)

    assert bot.codex.calls[-1]["previous_thread_id"] == "old-session"
    await store.close()


@pytest.mark.asyncio
async def test_manual_session_select_after_new_session_reset_is_allowed(tmp_path: Path) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    local_session = make_local_session("old-session", project_dir, "Return to previous thread")
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    await store.upsert_session(42, str(project_dir), "old-session", last_status="success")
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    bot.codex = FakeCodex(
        CodexResponse(final_text="done", thread_id="fresh-session", status=CodexResultStatus.SUCCESS),
        local_sessions=[local_session],
    )

    context = FakeContext()
    context.user_data["current_directory"] = project_dir
    new_update = FakeUpdate(user_id=42, text="/new")
    await bot.new_command(new_update, context)

    text_update = FakeUpdate(user_id=42, text="start fresh")
    await bot.handle_text(text_update, context)
    assert bot.codex.calls[-1]["previous_thread_id"] is None
    fresh_session = await store.get_session(42, str(project_dir))
    assert fresh_session is not None
    assert fresh_session.thread_id == "fresh-session"

    callback_query = FakeCallbackQuery(from_user_id=42, data="session:select:old-session")
    select_update = FakeUpdate(user_id=42, callback_query=callback_query)
    await bot.handle_ui_callback(select_update, context)

    selected = await store.get_session(42, str(project_dir))
    assert selected is not None
    assert selected.thread_id == "old-session"
    await store.close()


@pytest.mark.asyncio
async def test_session_select_callback_rejects_active_run(tmp_path: Path) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    local_session = make_local_session("old-session", project_dir)
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    bot.codex = FakeCodex(
        CodexResponse(final_text="done", thread_id="old-session", status=CodexResultStatus.SUCCESS),
        local_sessions=[local_session],
    )

    context = FakeContext()
    context.user_data["current_directory"] = project_dir
    bot.active_interrupts[42] = asyncio.Event()
    callback_query = FakeCallbackQuery(from_user_id=42, data="session:select:old-session")
    update = FakeUpdate(user_id=42, callback_query=callback_query)

    await bot.handle_ui_callback(update, context)

    assert callback_query.answers[-1] == ("Дождись завершения текущего запуска.", True)
    assert await store.get_session(42, str(project_dir)) is None
    await store.close()


@pytest.mark.asyncio
async def test_repo_command_renders_inline_picker(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "web").mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)

    update = FakeUpdate(user_id=42, text="/repo")
    context = FakeContext()
    await bot.repo_command(update, context)

    reply = update.effective_message.replies[-1]
    assert "Выбери активный проект" in reply.text
    assert keyboard_callback_data(reply.kwargs["reply_markup"]) == [
        ["repo:select:api"],
        ["repo:select:web"],
        ["action:create_project"],
        ["nav:menu"],
    ]
    await store.close()


@pytest.mark.asyncio
async def test_repo_command_auto_creates_project_in_empty_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    monkeypatch.setattr(bot.projects, "default_project_slug", lambda: "2026-04-18-project")

    update = FakeUpdate(user_id=42, text="/repo")
    context = FakeContext()
    await bot.repo_command(update, context)

    reply = update.effective_message.replies[-1]
    assert "Создал первый проект" in reply.text
    assert Path(context.user_data["current_directory"]).name == "2026-04-18-project"
    assert keyboard_callback_data(reply.kwargs["reply_markup"]) == [
        ["repo:select:2026-04-18-project"],
        ["action:create_project"],
        ["nav:menu"],
    ]
    await store.close()


@pytest.mark.asyncio
async def test_repo_new_command_creates_and_selects_project(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)

    update = FakeUpdate(user_id=42, text="/repo new My API")
    context = FakeContext()
    await bot.repo_command(update, context)

    reply = update.effective_message.replies[-1]
    assert "Создал и выбрал новый проект" in reply.text
    assert Path(context.user_data["current_directory"]).name == "my-api"
    assert (tmp_path / "my-api").is_dir()
    await store.close()


@pytest.mark.asyncio
async def test_repo_new_command_rejects_invalid_name(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)

    update = FakeUpdate(user_id=42, text="/repo new ...")
    context = FakeContext()
    await bot.repo_command(update, context)

    reply = update.effective_message.replies[-1]
    assert "Не удалось создать проект" in reply.text
    assert "current_directory" not in context.user_data
    await store.close()


@pytest.mark.asyncio
async def test_repo_select_callback_switches_directory_and_audits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "web").mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    fake_logger = FakeLogger()
    attach_fake_logger(bot, fake_logger)

    callback_query = FakeCallbackQuery(from_user_id=42, data="repo:select:web")
    update = FakeUpdate(user_id=42, callback_query=callback_query)
    context = FakeContext()

    await bot.handle_ui_callback(update, context)

    assert Path(context.user_data["current_directory"]).name == "web"
    assert "web" in callback_query.edits[-1][0]
    rows = await store.conn.execute_fetchall(
        "SELECT event_type FROM audit_log WHERE event_type = 'telegram_repo_selected'"
    )
    assert len(rows) == 1
    assert any(event == "telegram_repo_selected" for _, event, _ in fake_logger.events)
    await store.close()


@pytest.mark.asyncio
async def test_create_project_callback_uses_suffix_when_name_is_taken(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "2026-04-18-project").mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    monkeypatch.setattr(bot.projects, "default_project_slug", lambda: "2026-04-18-project")

    callback_query = FakeCallbackQuery(from_user_id=42, data="action:create_project")
    update = FakeUpdate(user_id=42, callback_query=callback_query)
    context = FakeContext()

    await bot.handle_ui_callback(update, context)

    assert Path(context.user_data["current_directory"]).name == "2026-04-18-project-2"
    assert "2026-04-18-project-2" in callback_query.edits[-1][0]
    await store.close()


@pytest.mark.asyncio
async def test_invalid_repo_slug_does_not_change_directory(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)

    callback_query = FakeCallbackQuery(from_user_id=42, data="repo:select:missing")
    update = FakeUpdate(user_id=42, callback_query=callback_query)
    context = FakeContext()
    context.user_data["current_directory"] = tmp_path / "api"

    await bot.handle_ui_callback(update, context)

    assert Path(context.user_data["current_directory"]).name == "api"
    assert callback_query.answers[-1] == ("Project unavailable.", True)
    await store.close()


@pytest.mark.asyncio
async def test_repo_refresh_noop_does_not_raise_or_reply_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "api").mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    fake_logger = FakeLogger()
    attach_fake_logger(bot, fake_logger)

    callback_query = NoopEditCallbackQuery(from_user_id=42, data="repo:refresh")
    update = FakeUpdate(user_id=42, callback_query=callback_query)
    context = FakeContext()

    await bot.handle_ui_callback(update, context)

    assert callback_query.answers[-1] == ("Projects", False)
    assert callback_query.message.replies == []
    assert any(event == "telegram_callback_edit_noop" for _, event, _ in fake_logger.events)
    await store.close()


@pytest.mark.asyncio
async def test_handle_text_writes_request_started_and_finished(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "codex_telegram_bot.services.projects.ProjectService.default_project_slug",
        lambda self: "2026-04-18-project",
    )
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    fake_logger = FakeLogger()
    attach_fake_logger(bot, fake_logger)
    bot.codex = FakeCodex(
        CodexResponse(
            final_text="done",
            thread_id="thread-123",
            status=CodexResultStatus.SUCCESS,
            input_tokens=10,
            cached_input_tokens=2,
            output_tokens=5,
            duration_ms=123,
        )
    )

    update = FakeUpdate(user_id=42, text="hello")
    context = FakeContext()
    await bot.handle_text(update, context)

    events = [row[0] for row in await store.conn.execute_fetchall("SELECT event_type FROM audit_log")]
    assert "request_started" in events
    assert "request_finished" in events
    session = await store.get_session(42, str(tmp_path / "2026-04-18-project"))
    assert session is not None
    assert session.thread_id == "thread-123"
    assert session.last_status == "success"
    assert any(event == "codex_request_started" for _, event, _ in fake_logger.events)
    assert any(event == "codex_request_finished" for _, event, _ in fake_logger.events)
    assert "Создал и выбрал первый проект" in update.effective_message.replies[0].text
    assert keyboard_callback_data(update.effective_message.replies[1].kwargs["reply_markup"]) == [
        ["action:stop:42"]
    ]
    assert update.effective_message.replies[1].text == (
        "Проект: 2026-04-18-project\n\nWorking... 0s"
    )
    assert update.effective_message.replies[-1].kwargs["parse_mode"] == "HTML"
    assert "reply_markup" not in update.effective_message.replies[-1].kwargs or update.effective_message.replies[-1].kwargs["reply_markup"] is None
    await store.close()


@pytest.mark.asyncio
async def test_handle_text_discovers_existing_codex_session_when_store_is_empty(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    bot.codex = FakeCodex(
        CodexResponse(final_text="done", thread_id="existing-session", status=CodexResultStatus.SUCCESS),
        discovered_session_id="existing-session",
    )

    update = FakeUpdate(user_id=42, text="continue")
    context = FakeContext()
    context.user_data["current_directory"] = project_dir
    await bot.handle_text(update, context)

    assert bot.codex.calls[0]["previous_thread_id"] == "existing-session"
    session = await store.get_session(42, str(project_dir))
    assert session is not None
    assert session.thread_id == "existing-session"
    events = [row[0] for row in await store.conn.execute_fetchall("SELECT event_type FROM audit_log")]
    assert "session_discovered" in events
    await store.close()


@pytest.mark.asyncio
async def test_handle_text_does_not_resume_discovered_session_after_new_session_reset(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    await store.clear_session(42, str(project_dir))
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    bot.codex = FakeCodex(
        CodexResponse(final_text="done", thread_id="fresh-session", status=CodexResultStatus.SUCCESS)
    )

    update = FakeUpdate(user_id=42, text="start fresh")
    context = FakeContext()
    context.user_data["current_directory"] = project_dir
    await bot.handle_text(update, context)

    assert bot.codex.discovery_calls[0]["modified_after"] is not None
    assert bot.codex.calls[0]["previous_thread_id"] is None
    session = await store.get_session(42, str(project_dir))
    assert session is not None
    assert session.thread_id == "fresh-session"
    await store.close()


@pytest.mark.asyncio
async def test_handle_text_formats_final_markdown_as_html(tmp_path: Path) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    bot.codex = FakeCodex(
        CodexResponse(
            final_text="**Слабые стороны**\n- перегружен\n- нет lock",
            thread_id="thread-123",
            status=CodexResultStatus.SUCCESS,
        )
    )

    update = FakeUpdate(user_id=42, text="hello")
    context = FakeContext()
    context.user_data["current_directory"] = project_dir
    await bot.handle_text(update, context)

    final_reply = update.effective_message.replies[-1]
    assert final_reply.kwargs["parse_mode"] == "HTML"
    assert "<b>Слабые стороны</b>" in final_reply.text
    assert "• перегружен" in final_reply.text
    await store.close()


@pytest.mark.asyncio
async def test_handle_text_falls_back_to_plain_text_when_html_send_fails(tmp_path: Path) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    bot.codex = FakeCodex(
        CodexResponse(
            final_text="**Слабые стороны**\n- перегружен",
            thread_id="thread-123",
            status=CodexResultStatus.SUCCESS,
        )
    )

    update = FakeUpdate(user_id=42, text="hello")
    update.effective_message = HtmlRejectingMessage(text="hello", message_id=1)
    context = FakeContext()
    context.user_data["current_directory"] = project_dir
    await bot.handle_text(update, context)

    final_reply = update.effective_message.replies[-1]
    assert "parse_mode" not in final_reply.kwargs
    assert "Слабые стороны" in final_reply.text
    assert "**" not in final_reply.text
    await store.close()


@pytest.mark.asyncio
async def test_handle_text_sends_typing_heartbeat_and_elapsed_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    monkeypatch.setattr(bot, "TYPING_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(bot, "PROGRESS_HEARTBEAT_SECONDS", 0.01)
    monotonic_state = {"value": 100.0}

    def fake_monotonic() -> float:
        monotonic_state["value"] += 1.0
        return monotonic_state["value"]

    monkeypatch.setattr(
        "codex_telegram_bot.flows.execution.time.monotonic",
        fake_monotonic,
    )
    bot.codex = SlowFakeCodex(
        CodexResponse(
            final_text="done",
            thread_id="thread-123",
            status=CodexResultStatus.SUCCESS,
        ),
        delay_seconds=0.2,
    )

    update = FakeUpdate(user_id=42, text="long task")
    context = FakeContext()
    context.user_data["current_directory"] = project_dir

    await bot.handle_text(update, context)

    progress_reply = update.effective_message.replies[0]
    assert progress_reply.text == "Проект: app\n\nWorking... 0s"
    assert progress_reply.progress.deleted is True
    await store.close()


@pytest.mark.asyncio
async def test_typing_heartbeat_resends_chat_action(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    bot.TYPING_HEARTBEAT_SECONDS = 0.01
    chat = FakeChat()
    request_finished = asyncio.Event()
    request_context = RequestContext(source="text", user_id=42)

    task = asyncio.create_task(
        bot._typing_heartbeat(
            chat=chat,
            request_finished=request_finished,
            request_context=request_context,
        )
    )
    await asyncio.sleep(0.035)
    request_finished.set()
    await task

    assert len(chat.actions) >= 2
    await store.close()


@pytest.mark.asyncio
async def test_progress_heartbeat_updates_elapsed_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    bot.PROGRESS_HEARTBEAT_SECONDS = 0.01
    request_finished = asyncio.Event()
    interrupt_event = asyncio.Event()
    request_context = RequestContext(source="text", user_id=42)
    monotonic_state = {"value": 104.0}
    original_sleep = asyncio.sleep

    class AutoStoppingProgress(FakeProgressMessage):
        async def edit_text(self, text: str, **kwargs) -> None:
            await super().edit_text(text, **kwargs)
            request_finished.set()

    async def immediate_sleep(_: float) -> None:
        await original_sleep(0)

    def fake_monotonic() -> float:
        monotonic_state["value"] += 1.0
        return monotonic_state["value"]

    monkeypatch.setattr("codex_telegram_bot.flows.execution.time.monotonic", fake_monotonic)
    monkeypatch.setattr("codex_telegram_bot.flows.execution.asyncio.sleep", immediate_sleep)
    progress = AutoStoppingProgress()
    await bot._progress_heartbeat(
        progress=progress,
        stop_markup=bot.build_stop_keyboard(42),
        request_finished=request_finished,
        request_started_at=100.0,
        last_progress_lines=[],
        interrupt_event=interrupt_event,
        request_context=request_context,
    )

    assert progress.edits
    assert progress.edits[0][0].startswith("Working... ")
    assert progress.edits[0][0] != "Working... 0s"
    await store.close()


@pytest.mark.asyncio
async def test_handle_text_updates_existing_session_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    fake_logger = FakeLogger()
    attach_fake_logger(bot, fake_logger)
    await store.upsert_session(42, str(project_dir), "existing-thread", last_status="success")
    bot.codex = FakeCodex(
        CodexResponse(
            final_text="",
            thread_id="",
            status=CodexResultStatus.CLI_ERROR,
            error_message="boom",
            duration_ms=55,
        )
    )

    update = FakeUpdate(user_id=42, text="hello")
    context = FakeContext()
    context.user_data["current_directory"] = project_dir
    await bot.handle_text(update, context)

    session = await store.get_session(42, str(project_dir))
    assert session is not None
    assert session.last_status == "cli_error"
    assert session.last_error == "boom"
    rows = await store.conn.execute_fetchall(
        "SELECT event_type, event_status FROM audit_log ORDER BY id DESC LIMIT 2"
    )
    assert ("request_failed", "cli_error") in {(row[0], row[1]) for row in rows}
    assert any(event == "codex_request_failed" for _, event, _ in fake_logger.events)
    assert "reply_markup" not in update.effective_message.replies[-1].kwargs or update.effective_message.replies[-1].kwargs["reply_markup"] is None
    await store.close()


@pytest.mark.asyncio
async def test_voice_disabled_reply_contains_navigation_keyboard(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path, enable_voice_messages=False)
    bot = CodexTelegramBot(settings, store)

    update = FakeUpdate(user_id=42)
    update.effective_message.voice = SimpleNamespace(duration=4)
    context = FakeContext()

    await bot.handle_voice(update, context)

    reply = update.effective_message.replies[-1]
    assert reply.text == "Voice messages are disabled."
    assert "reply_markup" not in reply.kwargs or reply.kwargs["reply_markup"] is None
    await store.close()


@pytest.mark.asyncio
async def test_photo_request_cleans_temp_files_after_run(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path, codex_enable_images=True)
    bot = CodexTelegramBot(settings, store)
    bot.codex = FakeCodex(
        CodexResponse(final_text="done", thread_id="thread-123", status=CodexResultStatus.SUCCESS)
    )

    update = FakeUpdate(user_id=42, caption="Analyze")
    update.effective_message.photo = [SimpleNamespace(get_file=None)]
    context = FakeContext()
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    context.user_data["current_directory"] = project_dir

    photo_message = FakeMessage(caption="Analyze", photo=[SimpleNamespace()])
    update.effective_message = photo_message

    class FakeTelegramFile:
        async def download_as_bytearray(self):
            return bytearray(b"image-bytes")

    async def fake_get_file():
        return FakeTelegramFile()

    photo_message.photo = [SimpleNamespace(get_file=fake_get_file)]

    await bot.handle_photo(update, context)

    assert bot.codex.calls
    image_path = bot.codex.calls[0]["image_paths"][0]
    assert image_path.exists() is False
    await store.close()


@pytest.mark.asyncio
async def test_full_access_requires_confirmation_only_when_mode_changes(tmp_path: Path) -> None:
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    bot.codex = FakeCodex(
        CodexResponse(final_text="done", thread_id="thread-123", status=CodexResultStatus.SUCCESS)
    )

    context = FakeContext()
    context.user_data["current_directory"] = project_dir

    mode_query = FakeCallbackQuery(from_user_id=42, data="mode:show")
    mode_update = FakeUpdate(user_id=42, callback_query=mode_query)
    await bot.handle_ui_callback(mode_update, context)
    assert "Настройка режима доступа." in mode_query.edits[-1][0]

    confirm_query = FakeCallbackQuery(from_user_id=42, data="mode:confirm_full")
    confirm_update = FakeUpdate(user_id=42, callback_query=confirm_query)
    confirm_update.effective_message = mode_query.message
    await bot.handle_ui_callback(confirm_update, context)
    assert "Подтверждение полного доступа." in confirm_query.edits[-1][0]
    assert keyboard_callback_data(confirm_query.edits[-1][1]["reply_markup"]) == [
        ["mode:set:full"],
        ["mode:set:sandbox"],
        ["mode:show"],
    ]

    enable_query = FakeCallbackQuery(from_user_id=42, data="mode:set:full")
    enable_update = FakeUpdate(user_id=42, callback_query=enable_query)
    enable_update.effective_message = confirm_query.message
    await bot.handle_ui_callback(enable_update, context)

    launch_mode = await store.get_project_launch_mode(42, str(project_dir))
    assert launch_mode == CodexLaunchMode.FULL_ACCESS

    update = FakeUpdate(user_id=42, text="hello")
    update.effective_message = FakeMessage(text="hello", message_id=1)
    await bot.handle_text(update, context)
    assert bot.codex.calls[-1]["launch_mode"] == CodexLaunchMode.FULL_ACCESS
    assert update.effective_message.replies[-1].text.startswith("done")
    await store.close()


@pytest.mark.asyncio
async def test_launch_mode_is_restored_per_project(tmp_path: Path) -> None:
    api_dir = tmp_path / "api"
    web_dir = tmp_path / "web"
    api_dir.mkdir()
    web_dir.mkdir()
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)

    context = FakeContext()
    context.user_data["current_directory"] = api_dir

    api_full_query = FakeCallbackQuery(from_user_id=42, data="mode:set:full")
    api_full_update = FakeUpdate(user_id=42, callback_query=api_full_query)
    await bot.execution_flow.set_launch_mode(api_full_update, context, CodexLaunchMode.FULL_ACCESS)

    context.user_data["current_directory"] = web_dir
    web_sandbox_query = FakeCallbackQuery(from_user_id=42, data="mode:set:sandbox")
    web_sandbox_update = FakeUpdate(user_id=42, callback_query=web_sandbox_query)
    await bot.execution_flow.set_launch_mode(web_sandbox_update, context, CodexLaunchMode.SANDBOX)

    context.user_data["current_directory"] = api_dir
    menu_update = FakeUpdate(user_id=42, text="/menu")
    await bot.menu_command(menu_update, context)
    assert "Режим доступа: `Полный доступ`" in menu_update.effective_message.replies[-1].text

    context.user_data["current_directory"] = web_dir
    menu_update = FakeUpdate(user_id=42, text="/menu")
    await bot.menu_command(menu_update, context)
    assert "Режим доступа: `Песочница`" in menu_update.effective_message.replies[-1].text
    await store.close()


@pytest.mark.asyncio
async def test_stop_callback_sets_interrupt_and_audits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path)
    bot = CodexTelegramBot(settings, store)
    fake_logger = FakeLogger()
    attach_fake_logger(bot, fake_logger)

    interrupt_event = asyncio.Event()
    bot.active_interrupts[42] = interrupt_event
    callback_query = FakeCallbackQuery(from_user_id=42, data="action:stop:42")
    update = FakeUpdate(user_id=42, callback_query=callback_query)
    context = FakeContext()

    await bot.stop_callback(update, context)

    assert interrupt_event.is_set() is True
    rows = await store.conn.execute_fetchall("SELECT event_type, event_status FROM audit_log")
    assert ("request_interrupted", "requested") in {(row[0], row[1]) for row in rows}
    assert any(event == "codex_user_interrupt_requested" for _, event, _ in fake_logger.events)
    await store.close()


@pytest.mark.asyncio
async def test_audit_can_be_disabled(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "db.sqlite3")
    await store.initialize()
    settings = make_settings(tmp_path, enable_audit_log=False)
    bot = CodexTelegramBot(settings, store)
    bot.codex = FakeCodex(
        CodexResponse(final_text="done", thread_id="thread-123", status=CodexResultStatus.SUCCESS)
    )

    update = FakeUpdate(user_id=42, text="hello")
    context = FakeContext()
    await bot.handle_text(update, context)

    rows = await store.conn.execute_fetchall("SELECT event_type FROM audit_log")
    assert rows == []
    await store.close()


@pytest.mark.asyncio
async def test_amain_emits_startup_and_shutdown_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[str] = []
    configure_calls: list[str] = []

    class StopLoop(Exception):
        pass

    class FakeStore:
        def __init__(self) -> None:
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def health_check(self) -> bool:
            return True

        async def close(self) -> None:
            self.closed = True

    class FakeUpdater:
        async def start_polling(self, **kwargs) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeApp:
        def __init__(self) -> None:
            self.updater = FakeUpdater()

        async def initialize(self) -> None:
            return None

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def shutdown(self) -> None:
            return None

    class FakeBot:
        def __init__(self, settings, store) -> None:
            self.codex = SimpleNamespace(validate_cli_available=lambda: None)
            self.configure_calls = 0

        async def build(self) -> FakeApp:
            return FakeApp()

        async def configure_telegram_ui(self, app) -> None:
            self.configure_calls += 1

    fake_logger = FakeLogger()
    fake_store = FakeStore()
    fake_settings = SimpleNamespace(
        log_level="INFO",
        approved_directory=tmp_path,
        enable_audit_log=True,
        sqlite_path=tmp_path / "db.sqlite3",
        codex_cli_path="codex",
    )

    def fake_configure_logging(level: str) -> None:
        configure_calls.append(level)

    async def fake_sleep(seconds: float) -> None:
        raise StopLoop()

    monkeypatch.setattr(main_module, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(main_module, "logger", fake_logger)
    monkeypatch.setattr(main_module, "Settings", lambda: fake_settings)
    monkeypatch.setattr(main_module, "SessionStore", lambda path: fake_store)
    monkeypatch.setattr(main_module, "CodexTelegramBot", FakeBot)
    monkeypatch.setattr(main_module.asyncio, "sleep", fake_sleep)

    with pytest.raises(StopLoop):
        await main_module.amain()

    events = [event for _, event, _ in fake_logger.events]
    assert configure_calls == ["INFO", "INFO"]
    assert "app_starting" in events
    assert "settings_loaded" in events
    assert "session_store_initialized" in events
    assert "session_store_healthcheck_ok" in events
    assert "codex_cli_preflight_ok" in events
    assert "telegram_app_initialized" in events
    assert "telegram_ui_configured" in events
    assert "polling_started" in events
    assert "app_shutdown_started" in events
    assert "app_shutdown_completed" in events
