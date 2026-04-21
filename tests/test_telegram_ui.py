from __future__ import annotations

import pytest

from codex_telegram_bot.models import CodexLaunchMode, CodexResponse, CodexResultStatus
from codex_telegram_bot.services.projects import RepoOption
from codex_telegram_bot.telegram.ui.keyboards import (
    build_mode_editor_keyboard,
    build_repo_keyboard,
    build_session_keyboard,
)
from codex_telegram_bot.telegram.ui.responder import TelegramResponder
from codex_telegram_bot.telegram.ui.texts import (
    render_final_text,
    render_full_access_warning_text,
    render_launch_mode_editor_text,
    render_repo_picker_text,
)


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def debug(self, event: str, **kwargs) -> None:
        self.events.append(("debug", event, kwargs))

    def warning(self, event: str, **kwargs) -> None:
        self.events.append(("warning", event, kwargs))


def keyboard_callback_data(markup) -> list[list[str]]:
    return [[button.callback_data for button in row] for row in markup.inline_keyboard]


def test_build_session_keyboard_has_expected_actions() -> None:
    markup = build_session_keyboard()

    assert keyboard_callback_data(markup) == [
        ["nav:repo", "mode:show"],
        ["action:new"],
    ]


def test_build_repo_keyboard_ends_with_back_to_menu() -> None:
    markup = build_repo_keyboard([RepoOption(slug="api", label="api")])

    assert keyboard_callback_data(markup) == [
        ["repo:select:api"],
        ["action:create_project"],
        ["nav:menu"],
    ]


def test_render_repo_picker_text_marks_current_project() -> None:
    text = render_repo_picker_text(
        [RepoOption(slug="api", label="api", is_current=True)],
        truncated=False,
        auto_created=True,
    )

    assert "Создал первый проект `api`." in text
    assert "Текущий проект: `api`" in text


def test_render_final_text_appends_interrupted_marker() -> None:
    text = render_final_text(
        CodexResponse(
            final_text="Stopped",
            thread_id="thread-1",
            status=CodexResultStatus.INTERRUPTED,
        )
    )

    assert "(Interrupted by user)" in text


def test_build_mode_editor_keyboard_marks_selected_mode() -> None:
    markup = build_mode_editor_keyboard(
        CodexLaunchMode.FULL_ACCESS,
        full_access_confirmed=False,
        back_callback="nav:menu",
    )

    assert keyboard_callback_data(markup) == [
        ["mode:set:sandbox", "mode:confirm_full"],
        ["nav:menu"],
    ]
    assert markup.inline_keyboard[0][1].text == "Полный доступ (подтвердить)"


def test_render_mode_editor_text_mentions_next_requests() -> None:
    text = render_launch_mode_editor_text(
        project_name="api",
        launch_mode=CodexLaunchMode.FULL_ACCESS,
        has_active_run=True,
    )

    assert "Проект: `api`" in text
    assert "Полный доступ" in text
    assert "следующему запросу" in text


def test_render_full_access_warning_mentions_project() -> None:
    text = render_full_access_warning_text(project_name="api")

    assert "Подтверждение полного доступа" in text
    assert "Проект: `api`" in text


@pytest.mark.asyncio
async def test_responder_logs_noop_callback_edit() -> None:
    logger = FakeLogger()
    responder = TelegramResponder(logger)

    class FakeQuery:
        async def edit_message_text(self, *args, **kwargs) -> None:
            raise RuntimeError("Message is not modified: same content")

    update = type("Update", (), {"callback_query": FakeQuery(), "effective_message": None})()
    await responder.edit_callback_message(update, "same")

    assert logger.events[0][1] == "telegram_callback_edit_noop"
