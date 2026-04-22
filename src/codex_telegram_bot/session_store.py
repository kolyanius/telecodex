from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from .models import CodexLaunchMode, ProjectSession


class SessionStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA foreign_keys = ON")
        await self.conn.execute("PRAGMA journal_mode = WAL")
        await self._initialize_schema_version()
        await self._run_migrations()
        await self.conn.commit()

    async def _initialize_schema_version(self) -> None:
        conn = self._require_conn()
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
            """
        )

    async def _run_migrations(self) -> None:
        current_version = await self._get_schema_version()
        if current_version < 1:
            await self._migration_v1()
            await self._set_schema_version(1)
        if current_version < 2:
            await self._migration_v2()
            await self._set_schema_version(2)
        if current_version < 3:
            await self._migration_v3()
            await self._set_schema_version(3)
        if current_version < 4:
            await self._migration_v4()
            await self._set_schema_version(4)

    async def _get_schema_version(self) -> int:
        conn = self._require_conn()
        cursor = await conn.execute("SELECT MAX(version) FROM schema_version")
        row = await cursor.fetchone()
        return int(row[0] or 0) if row else 0

    async def _set_schema_version(self, version: int) -> None:
        conn = self._require_conn()
        await conn.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))

    async def _migration_v1(self) -> None:
        conn = self._require_conn()
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_sessions (
                user_id INTEGER NOT NULL,
                project_path TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_status TEXT NOT NULL DEFAULT '',
                last_error TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (user_id, project_path)
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                chat_id INTEGER,
                project_path TEXT,
                event_type TEXT NOT NULL,
                event_status TEXT NOT NULL DEFAULT '',
                details TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_project_sessions_user_project
            ON project_sessions(user_id, project_path)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audit_log_created_at
            ON audit_log(created_at)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audit_log_user_event
            ON audit_log(user_id, event_type)
            """
        )

    async def _migration_v2(self) -> None:
        conn = self._require_conn()
        columns = await self._get_table_columns("project_sessions")
        if "last_status" not in columns:
            await conn.execute(
                "ALTER TABLE project_sessions ADD COLUMN last_status TEXT NOT NULL DEFAULT ''"
            )
        if "last_error" not in columns:
            await conn.execute(
                "ALTER TABLE project_sessions ADD COLUMN last_error TEXT NOT NULL DEFAULT ''"
            )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_project_sessions_updated_at
            ON project_sessions(updated_at)
            """
        )

    async def _migration_v3(self) -> None:
        conn = self._require_conn()
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_preferences (
                user_id INTEGER NOT NULL,
                project_path TEXT NOT NULL,
                launch_mode TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, project_path)
            )
            """
        )

    async def _migration_v4(self) -> None:
        conn = self._require_conn()
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_session_resets (
                user_id INTEGER NOT NULL,
                project_path TEXT NOT NULL,
                reset_at_unix REAL NOT NULL,
                PRIMARY KEY (user_id, project_path)
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_project_session_resets_reset_at
            ON project_session_resets(reset_at_unix)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_project_preferences_updated_at
            ON project_preferences(updated_at)
            """
        )

    async def _get_table_columns(self, table_name: str) -> set[str]:
        conn = self._require_conn()
        cursor = await conn.execute("PRAGMA table_info(%s)" % table_name)
        rows = await cursor.fetchall()
        return {str(row["name"]) for row in rows}

    async def upsert_session(
        self,
        user_id: int,
        project_path: str,
        thread_id: str,
        *,
        last_status: str = "",
        last_error: str = "",
    ) -> None:
        conn = self._require_conn()
        await conn.execute(
            """
            INSERT INTO project_sessions (
                user_id, project_path, thread_id, updated_at, last_status, last_error
            )
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
            ON CONFLICT(user_id, project_path)
            DO UPDATE SET
                thread_id=excluded.thread_id,
                updated_at=CURRENT_TIMESTAMP,
                last_status=excluded.last_status,
                last_error=excluded.last_error
            """,
            (user_id, project_path, thread_id, last_status, last_error),
        )
        await conn.execute(
            "DELETE FROM project_session_resets WHERE user_id = ? AND project_path = ?",
            (user_id, project_path),
        )
        await conn.commit()

    async def get_thread_id(self, user_id: int, project_path: str) -> Optional[str]:
        session = await self.get_session(user_id, project_path)
        return session.thread_id if session else None

    async def get_session(self, user_id: int, project_path: str) -> Optional[ProjectSession]:
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT user_id, project_path, thread_id, updated_at, last_status, last_error
            FROM project_sessions
            WHERE user_id = ? AND project_path = ?
            """,
            (user_id, project_path),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return ProjectSession(
            user_id=int(row["user_id"]),
            project_path=str(row["project_path"]),
            thread_id=str(row["thread_id"]),
            updated_at=str(row["updated_at"]),
            last_status=str(row["last_status"] or ""),
            last_error=str(row["last_error"] or ""),
        )

    async def update_session_result(
        self,
        user_id: int,
        project_path: str,
        *,
        last_status: str,
        last_error: str = "",
    ) -> None:
        conn = self._require_conn()
        await conn.execute(
            """
            UPDATE project_sessions
            SET updated_at = CURRENT_TIMESTAMP, last_status = ?, last_error = ?
            WHERE user_id = ? AND project_path = ?
            """,
            (last_status, last_error, user_id, project_path),
        )
        await conn.commit()

    async def clear_session(self, user_id: int, project_path: str) -> None:
        conn = self._require_conn()
        await conn.execute(
            "DELETE FROM project_sessions WHERE user_id = ? AND project_path = ?",
            (user_id, project_path),
        )
        await conn.execute(
            """
            INSERT INTO project_session_resets (user_id, project_path, reset_at_unix)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, project_path)
            DO UPDATE SET reset_at_unix=excluded.reset_at_unix
            """,
            (user_id, project_path, time.time()),
        )
        await conn.commit()

    async def get_session_reset_at_unix(self, user_id: int, project_path: str) -> Optional[float]:
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT reset_at_unix
            FROM project_session_resets
            WHERE user_id = ? AND project_path = ?
            """,
            (user_id, project_path),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return float(row["reset_at_unix"])

    async def set_project_launch_mode(
        self,
        user_id: int,
        project_path: str,
        launch_mode: CodexLaunchMode,
    ) -> None:
        conn = self._require_conn()
        await conn.execute(
            """
            INSERT INTO project_preferences (
                user_id, project_path, launch_mode, updated_at
            )
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, project_path)
            DO UPDATE SET
                launch_mode=excluded.launch_mode,
                updated_at=CURRENT_TIMESTAMP
            """,
            (user_id, project_path, launch_mode.value),
        )
        await conn.commit()

    async def get_project_launch_mode(
        self,
        user_id: int,
        project_path: str,
    ) -> Optional[CodexLaunchMode]:
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT launch_mode
            FROM project_preferences
            WHERE user_id = ? AND project_path = ?
            """,
            (user_id, project_path),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return CodexLaunchMode.from_value(row["launch_mode"])

    async def log_audit_event(
        self,
        *,
        user_id: Optional[int],
        chat_id: Optional[int],
        project_path: Optional[str],
        event_type: str,
        event_status: str = "",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        conn = self._require_conn()
        await conn.execute(
            """
            INSERT INTO audit_log(
                user_id, chat_id, project_path, event_type, event_status, details
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                chat_id,
                project_path,
                event_type,
                event_status,
                json.dumps(details or {}, ensure_ascii=True, sort_keys=True),
            ),
        )
        await conn.commit()

    async def health_check(self) -> bool:
        conn = self._require_conn()
        try:
            cursor = await conn.execute("SELECT 1")
            row = await cursor.fetchone()
            return bool(row and row[0] == 1)
        except Exception:
            return False

    async def close(self) -> None:
        if self.conn is not None:
            await self.conn.close()
            self.conn = None

    def _require_conn(self) -> aiosqlite.Connection:
        if self.conn is None:
            raise RuntimeError("SessionStore is not initialized")
        return self.conn
