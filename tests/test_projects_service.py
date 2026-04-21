from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_telegram_bot.config import Settings
from codex_telegram_bot.models import RequestContext
from codex_telegram_bot.services.projects import ProjectService


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "telegram_bot_token": "token",
        "telegram_bot_username": "codex_bot",
        "approved_directory": tmp_path,
        "allowed_users": "42",
    }
    values.update(overrides)
    return Settings(**values)


async def noop_record_event(*args, **kwargs) -> None:
    return None


def test_workspace_is_empty_ignores_sqlite_sidecars(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = ProjectService(settings, noop_record_event)

    (tmp_path / "db.sqlite3").write_text("")
    (tmp_path / "db.sqlite3-wal").write_text("")
    (tmp_path / "db.sqlite3-shm").write_text("")

    assert service.workspace_is_empty(tmp_path) is True

    (tmp_path / "README.md").write_text("content")
    assert service.workspace_is_empty(tmp_path) is False


def test_list_repo_options_marks_current_project(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = ProjectService(settings, noop_record_event)
    (tmp_path / "api").mkdir()
    (tmp_path / "web").mkdir()
    context = SimpleNamespace(user_data={"current_directory": tmp_path / "web"})

    options, truncated = service.list_repo_options(context)

    assert truncated is False
    assert [option.slug for option in options] == ["api", "web"]
    assert options[0].is_current is False
    assert options[1].is_current is True
    assert options[1].label == "web • current"


@pytest.mark.asyncio
async def test_create_project_sanitizes_name_and_updates_context(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = ProjectService(settings, noop_record_event)
    context = SimpleNamespace(user_data={})

    project = await service.create_project(
        "My API",
        context=context,
        request_context=RequestContext(source="command"),
    )

    assert project.name == "my-api"
    assert context.user_data["current_directory"] == project
    assert project.is_dir()
