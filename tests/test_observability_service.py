from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_telegram_bot.config import Settings
from codex_telegram_bot.models import CodexResponse, CodexResultStatus
from codex_telegram_bot.services.observability import ObservabilityService


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


class FakeStore:
    def __init__(self) -> None:
        self.audit_calls: list[dict] = []

    async def log_audit_event(self, **kwargs) -> None:
        self.audit_calls.append(kwargs)


def make_update() -> SimpleNamespace:
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=10, type="private"),
        effective_message=SimpleNamespace(message_id=11),
        effective_user=SimpleNamespace(id=42),
    )


def test_make_request_context_uses_selected_directory(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = ObservabilityService(settings, FakeStore(), FakeLogger())
    project_dir = tmp_path / "app"
    project_dir.mkdir()
    context = SimpleNamespace(user_data={"current_directory": project_dir})

    request_context = service.make_request_context(
        make_update(),
        context,
        source="command",
        command_name="status",
    )

    assert request_context.cwd == str(project_dir)
    assert request_context.command_name == "status"
    assert request_context.user_id == 42


@pytest.mark.asyncio
async def test_record_event_logs_and_writes_audit_row(tmp_path: Path) -> None:
    logger = FakeLogger()
    store = FakeStore()
    settings = make_settings(tmp_path)
    service = ObservabilityService(settings, store, logger)
    request_context = service.make_request_context(make_update(), SimpleNamespace(user_data={}), source="text")

    await service.record_event(
        "codex_request_started",
        request_context,
        audit_event="request_started",
        event_status="started",
        thread_id="thread-1",
    )

    assert logger.events[0][1] == "codex_request_started"
    assert store.audit_calls[0]["event_type"] == "request_started"
    assert store.audit_calls[0]["event_status"] == "started"


def test_response_fields_returns_tokens_and_status() -> None:
    response = CodexResponse(
        final_text="done",
        thread_id="thread-1",
        status=CodexResultStatus.SUCCESS,
        input_tokens=10,
        cached_input_tokens=3,
        output_tokens=5,
        duration_ms=77,
    )

    fields = ObservabilityService.response_fields(response)

    assert fields["status"] == "success"
    assert fields["input_tokens"] == 10
    assert fields["cached_input_tokens"] == 3
    assert fields["output_tokens"] == 5
