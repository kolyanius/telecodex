from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from codex_telegram_bot.models import CodexLaunchMode
from codex_telegram_bot.session_store import SessionStore


@pytest.mark.asyncio
async def test_session_store_crud_and_audit(tmp_path: Path) -> None:
    db_path = tmp_path / "bot.db"
    store = SessionStore(db_path)
    await store.initialize()

    assert await store.health_check() is True

    await store.upsert_session(
        100,
        "/workspace/project",
        "thread-1",
        last_status="success",
        last_error="",
    )
    session = await store.get_session(100, "/workspace/project")
    assert session is not None
    assert session.thread_id == "thread-1"
    assert session.last_status == "success"

    await store.update_session_result(
        100,
        "/workspace/project",
        last_status="timeout",
        last_error="timed out",
    )
    session = await store.get_session(100, "/workspace/project")
    assert session is not None
    assert session.last_status == "timeout"
    assert session.last_error == "timed out"

    assert await store.get_project_launch_mode(100, "/workspace/project") is None
    await store.set_project_launch_mode(100, "/workspace/project", CodexLaunchMode.FULL_ACCESS)
    assert await store.get_project_launch_mode(100, "/workspace/project") == CodexLaunchMode.FULL_ACCESS

    await store.log_audit_event(
        user_id=100,
        chat_id=200,
        project_path="/workspace/project",
        event_type="request_finished",
        event_status="timeout",
        details={"duration_ms": 1234},
    )

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT event_type, event_status, details FROM audit_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == "request_finished"
    assert row[1] == "timeout"
    assert '"duration_ms": 1234' in row[2]

    await store.clear_session(100, "/workspace/project")
    assert await store.get_session(100, "/workspace/project") is None
    await store.close()


@pytest.mark.asyncio
async def test_session_store_migrates_legacy_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE project_sessions (
                user_id INTEGER NOT NULL,
                project_path TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, project_path)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    store = SessionStore(db_path)
    await store.initialize()
    await store.upsert_session(1, "/legacy/project", "thread-legacy", last_status="success")

    migrated = await store.get_session(1, "/legacy/project")
    assert migrated is not None
    assert migrated.last_status == "success"
    assert migrated.last_error == ""
    await store.set_project_launch_mode(1, "/legacy/project", CodexLaunchMode.SANDBOX)
    assert await store.get_project_launch_mode(1, "/legacy/project") == CodexLaunchMode.SANDBOX
    await store.close()
