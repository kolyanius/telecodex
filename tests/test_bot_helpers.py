from __future__ import annotations

from pathlib import Path

from codex_telegram_bot.bot import CodexTelegramBot, RateLimiter
from codex_telegram_bot.config import Settings
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


class DummyUser:
    def __init__(self, user_id: int):
        self.id = user_id


class DummyUpdate:
    def __init__(self, user_id: int):
        self.effective_user = DummyUser(user_id)


def test_rate_limiter_blocks_after_limit() -> None:
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    assert limiter.allow(1) is True
    assert limiter.allow(1) is True
    assert limiter.allow(1) is False


def test_bot_chunk_text_splits_long_messages() -> None:
    text = "a" * 11
    chunks = CodexTelegramBot._chunk_text(text, 5)
    assert chunks == ["aaaaa", "aaaaa", "a"]


def test_bot_build_progress_text() -> None:
    assert CodexTelegramBot._build_progress_text(12, []) == "Working... 12s"
    assert CodexTelegramBot._build_progress_text(3, ["🔧 rg", "💬 thinking"]) == (
        "Working... 3s\n\n🔧 rg\n💬 thinking"
    )


def test_bot_authorization_and_workspace_validation(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, allowed_users="42")
    bot = CodexTelegramBot(settings, SessionStore(tmp_path / "db.sqlite3"))

    assert bot._is_authorized(DummyUpdate(42)) is True
    assert bot._is_authorized(DummyUpdate(7)) is False

    outside = tmp_path.parent
    try:
        bot._ensure_in_workspace(outside)
    except PermissionError:
        pass
    else:
        raise AssertionError("Expected PermissionError for path outside workspace")
