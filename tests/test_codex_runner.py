from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from codex_telegram_bot.codex_runner import CodexRunner
from codex_telegram_bot.config import Settings
from codex_telegram_bot.models import CodexLaunchMode, CodexResultStatus, CodexStreamEventKind


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "telegram_bot_token": "token",
        "telegram_bot_username": "codex_bot",
        "approved_directory": tmp_path,
        "codex_cli_path": "codex",
    }
    values.update(overrides)
    return Settings(**values)


class FakeStream:
    def __init__(self, lines: list[str] | None = None, content: str = ""):
        self._lines = [line.encode("utf-8") for line in (lines or [])]
        self._content = content.encode("utf-8")

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""

    async def read(self) -> bytes:
        return self._content


class FakeProcess:
    def __init__(
        self,
        *,
        stdout_lines: list[str] | None = None,
        stderr: str = "",
        returncode: int | None = 0,
        wait_forever: bool = False,
    ):
        self.stdout = FakeStream(lines=stdout_lines or [])
        self.stderr = FakeStream(content=stderr)
        self.returncode = None if wait_forever else returncode
        self._final_returncode = -9 if wait_forever else int(returncode or 0)
        self._done = asyncio.Event()
        self.killed = False
        if not wait_forever:
            self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        return int(self.returncode or 0)

    def kill(self) -> None:
        self.killed = True
        self.returncode = self._final_returncode
        self._done.set()


@pytest.mark.asyncio
async def test_runner_success_and_event_normalization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    events = []
    process = FakeProcess(
        stdout_lines=[
            '{"type":"thread.started","thread_id":"thread-123"}\n',
            '{"type":"agent_message_delta","delta":"Hello"}\n',
            '{"type":"item.completed","item":{"type":"assistant_message","text":"Hello snapshot"}}\n',
            '{"type":"item.completed","item":{"type":"tool_result","name":"Read"}}\n',
            '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":4}}\n',
        ]
    )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(
        prompt="hello",
        cwd=tmp_path,
        launch_mode=CodexLaunchMode.SANDBOX,
        on_event=events.append,
    )

    assert response.status == CodexResultStatus.SUCCESS
    assert response.thread_id == "thread-123"
    assert response.final_text == "Hello snapshot"
    assert response.input_tokens == 10
    assert response.cached_input_tokens == 2
    assert response.output_tokens == 4
    assert [event.kind for event in events] == [
        CodexStreamEventKind.LIFECYCLE,
        CodexStreamEventKind.TEXT_DELTA,
        CodexStreamEventKind.TEXT_SNAPSHOT,
        CodexStreamEventKind.TOOL_CALL,
        CodexStreamEventKind.USAGE,
    ]


@pytest.mark.asyncio
async def test_runner_raises_subprocess_stream_limit_for_large_json_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    large_text = "x" * 70_000
    process = FakeProcess(
        stdout_lines=[
            '{"type":"item.completed","item":{"type":"assistant_message","text":"'
            + large_text
            + '"}}\n'
        ]
    )
    captured: dict[str, int] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["limit"] = kwargs["limit"]
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(prompt="large", cwd=tmp_path, launch_mode=CodexLaunchMode.SANDBOX)

    assert captured["limit"] > len(large_text)
    assert response.status == CodexResultStatus.SUCCESS
    assert response.final_text == large_text


@pytest.mark.asyncio
async def test_runner_prefers_final_snapshot_over_streamed_deltas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    process = FakeProcess(
        stdout_lines=[
            '{"type":"thread.started","thread_id":"thread-123"}\n',
            '{"type":"agent_message_delta","delta":"Принял: сделаю обзор"}\n',
            '{"type":"agent_message_delta","delta":" проекта."}\n',
            '{"type":"item.completed","item":{"type":"assistant_message","text":"Полный обзор проекта:\\n- архитектура\\n- риски\\n- рекомендации"}}\n',
        ]
    )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(prompt="review", cwd=tmp_path, launch_mode=CodexLaunchMode.SANDBOX)

    assert response.status == CodexResultStatus.SUCCESS
    assert response.final_text == "Полный обзор проекта:\n- архитектура\n- риски\n- рекомендации"


@pytest.mark.asyncio
async def test_runner_uses_last_non_empty_snapshot_when_multiple_arrive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    process = FakeProcess(
        stdout_lines=[
            '{"type":"agent_message_delta","delta":"Черновик"}\n',
            '{"type":"item.completed","item":{"type":"assistant_message","text":"Короткий ответ"}}\n',
            '{"type":"item.completed","item":{"type":"assistant_message","text":""}}\n',
            '{"type":"item.completed","item":{"type":"assistant_message","text":"Финальный развёрнутый ответ"}}\n',
        ]
    )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(prompt="review", cwd=tmp_path, launch_mode=CodexLaunchMode.SANDBOX)

    assert response.status == CodexResultStatus.SUCCESS
    assert response.final_text == "Финальный развёрнутый ответ"


@pytest.mark.asyncio
async def test_runner_retries_only_recoverable_resume_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    calls = {"count": 0}

    async def fake_create_subprocess_exec(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeProcess(
                stderr="No conversation found for requested thread",
                returncode=1,
            )
        return FakeProcess(
            stdout_lines=[
                '{"type":"thread.started","thread_id":"fresh-thread"}\n',
                '{"type":"agent_message_delta","delta":"Recovered"}\n',
            ],
            returncode=0,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(
        prompt="hello",
        cwd=tmp_path,
        launch_mode=CodexLaunchMode.SANDBOX,
        previous_thread_id="bad-thread",
    )

    assert calls["count"] == 2
    assert response.status == CodexResultStatus.SUCCESS
    assert response.thread_id == "fresh-thread"
    assert response.final_text == "Recovered"
    assert "No conversation found" in response.fallback_reason


@pytest.mark.asyncio
async def test_runner_returns_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.codex_timeout_seconds = 0.01
    runner = CodexRunner(settings)

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess(wait_forever=True)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(prompt="slow", cwd=tmp_path, launch_mode=CodexLaunchMode.SANDBOX)

    assert response.status == CodexResultStatus.TIMEOUT
    assert "timed out" in response.error_message.lower()


@pytest.mark.asyncio
async def test_runner_returns_interrupted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.codex_timeout_seconds = 10
    runner = CodexRunner(settings)
    interrupt_event = asyncio.Event()

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess(wait_forever=True)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def trigger_interrupt() -> None:
        await asyncio.sleep(0.01)
        interrupt_event.set()

    task = asyncio.create_task(trigger_interrupt())
    response = await runner.run(
        prompt="stop",
        cwd=tmp_path,
        launch_mode=CodexLaunchMode.SANDBOX,
        interrupt_event=interrupt_event,
    )
    await task

    assert response.status == CodexResultStatus.INTERRUPTED
    assert response.interrupted is True


def test_validate_cli_available_with_explicit_binary(tmp_path: Path) -> None:
    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | 0o111)
    settings = make_settings(tmp_path, codex_cli_path=str(binary))

    runner = CodexRunner(settings)
    runner.validate_cli_available()


def test_validate_cli_available_missing_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = make_settings(tmp_path, codex_cli_path="missing-codex")
    runner = CodexRunner(settings)
    monkeypatch.setattr("shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="not found"):
        runner.validate_cli_available()


def test_discover_latest_session_id_matches_exact_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / "codex-home"
    sessions_dir = codex_home / "sessions" / "2026" / "04" / "22"
    sessions_dir.mkdir(parents=True)
    project_dir = tmp_path / "project"
    other_dir = tmp_path / "other"
    project_dir.mkdir()
    other_dir.mkdir()

    older = sessions_dir / "older.jsonl"
    newer = sessions_dir / "newer.jsonl"
    other = sessions_dir / "other.jsonl"
    older.write_text(
        '{"type":"session_meta","payload":{"id":"older-session","cwd":"'
        + str(project_dir)
        + '"}}\n',
        encoding="utf-8",
    )
    newer.write_text(
        '{"type":"session_meta","payload":{"id":"newer-session","cwd":"'
        + str(project_dir)
        + '"}}\n',
        encoding="utf-8",
    )
    other.write_text(
        '{"type":"session_meta","payload":{"id":"other-session","cwd":"'
        + str(other_dir)
        + '"}}\n',
        encoding="utf-8",
    )
    os.utime(older, (1, 1))
    os.utime(newer, (3, 3))
    os.utime(other, (5, 5))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    runner = CodexRunner(make_settings(tmp_path))

    assert runner.discover_latest_session_id(project_dir) == "newer-session"
    assert runner.discover_latest_session_id(project_dir, modified_after=4) is None


def test_discover_local_sessions_lists_matching_cwd_sorted_and_ignores_bad_jsonl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / "codex-home"
    sessions_dir = codex_home / "sessions" / "2026" / "04" / "22"
    sessions_dir.mkdir(parents=True)
    project_dir = tmp_path / "project"
    other_dir = tmp_path / "other"
    project_dir.mkdir()
    other_dir.mkdir()

    def write_session(path: Path, session_id: str, cwd: Path, lines: list[dict | str]) -> None:
        encoded = [
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {
                        "id": session_id,
                        "cwd": str(cwd),
                        "timestamp": "2026-04-22T12:00:00Z",
                    },
                }
            )
        ]
        for line in lines:
            encoded.append(line if isinstance(line, str) else json.dumps(line, ensure_ascii=False))
        path.write_text("\n".join(encoded) + "\n", encoding="utf-8")

    older = sessions_dir / "older.jsonl"
    newer = sessions_dir / "newer.jsonl"
    other = sessions_dir / "other.jsonl"
    broken = sessions_dir / "broken.jsonl"
    write_session(
        older,
        "older-session",
        project_dir,
        [
            "{broken",
            {
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "Implement manual session picker",
                },
            },
        ],
    )
    write_session(
        newer,
        "newer-session",
        project_dir,
        [
            {
                "type": "response_item",
                "payload": {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "<environment_context>...</environment_context>"},
                        {"type": "input_text", "text": "Continue old Codex chat"},
                    ],
                },
            }
        ],
    )
    write_session(
        other,
        "other-session",
        other_dir,
        [
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "Wrong project"},
            }
        ],
    )
    broken.write_text("{not-json\n", encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (3, 3))
    os.utime(other, (5, 5))
    os.utime(broken, (7, 7))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    runner = CodexRunner(make_settings(tmp_path))

    sessions = runner.discover_local_sessions(project_dir)

    assert [session.session_id for session in sessions] == ["newer-session", "older-session"]
    assert sessions[0].first_prompt == "Continue old Codex chat"
    assert sessions[1].first_prompt == "Implement manual session picker"
    assert sessions[0].source_path == newer
    assert runner.discover_local_sessions(project_dir, limit=1)[0].session_id == "newer-session"


@pytest.mark.asyncio
async def test_runner_rejects_images_when_disabled(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, codex_enable_images=False)
    runner = CodexRunner(settings)
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"jpg")

    response = await runner.run(
        prompt="image",
        cwd=tmp_path,
        launch_mode=CodexLaunchMode.SANDBOX,
        image_paths=[image_path],
    )

    assert response.status == CodexResultStatus.CLI_ERROR
    assert "disabled" in response.error_message.lower()


@pytest.mark.asyncio
async def test_runner_builds_read_only_sandbox_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    process = FakeProcess(stdout_lines=['{"type":"item.completed","item":{"type":"assistant_message","text":"ok"}}\n'])
    captured: dict[str, tuple] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(prompt="hello", cwd=tmp_path, launch_mode=CodexLaunchMode.SANDBOX)

    assert response.status == CodexResultStatus.SUCCESS
    assert "--sandbox" in captured["args"]
    assert "read-only" in captured["args"]
    assert 'web_search="disabled"' in captured["args"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in captured["args"]
    assert "--auto-edit" not in captured["args"]
    assert "--suggest" not in captured["args"]


@pytest.mark.asyncio
async def test_runner_passes_reasoning_effort_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path, codex_model="gpt-5.4", codex_reasoning_effort="high")
    runner = CodexRunner(settings)
    process = FakeProcess(stdout_lines=['{"type":"item.completed","item":{"type":"assistant_message","text":"ok"}}\n'])
    captured: dict[str, tuple] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(prompt="hello", cwd=tmp_path, launch_mode=CodexLaunchMode.SANDBOX)

    assert response.status == CodexResultStatus.SUCCESS
    assert "--model" in captured["args"]
    assert "gpt-5.4" in captured["args"]
    assert "--config" in captured["args"]
    assert 'model_reasoning_effort="high"' in captured["args"]


@pytest.mark.asyncio
async def test_runner_builds_full_access_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    process = FakeProcess(stdout_lines=['{"type":"item.completed","item":{"type":"assistant_message","text":"ok"}}\n'])
    captured: dict[str, tuple] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(prompt="hello", cwd=tmp_path, launch_mode=CodexLaunchMode.FULL_ACCESS)

    assert response.status == CodexResultStatus.SUCCESS
    assert "--dangerously-bypass-approvals-and-sandbox" in captured["args"]
    assert "--sandbox" not in captured["args"]
    assert 'web_search="disabled"' not in captured["args"]
