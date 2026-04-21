from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from codex_telegram_bot.codex_runner import CodexRunner, MAX_JSONL_EVENT_BYTES
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
    def __init__(self, *, chunks: list[bytes] | None = None, content: str = ""):
        self._chunks = list(chunks or [])
        self._content = content.encode("utf-8")

    async def read(self, _size: int = -1) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        if self._content:
            content = self._content
            self._content = b""
            return content
        return b""


class FakeStdin:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class FakeProcess:
    def __init__(
        self,
        *,
        stdout_chunks: list[bytes] | None = None,
        stderr: str = "",
        returncode: int | None = 0,
        wait_forever: bool = False,
    ):
        self.stdout = FakeStream(chunks=stdout_chunks or [])
        self.stderr = FakeStream(content=stderr)
        self.stdin = FakeStdin()
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
    payload = (
        '{"type":"thread.started","thread_id":"thread-123"}\n'
        '{"type":"agent_message_delta","delta":"Hello"}\n'
        '{"type":"item.completed","item":{"type":"assistant_message","text":"Hello snapshot"}}\n'
        '{"type":"item.completed","item":{"type":"tool_result","name":"Read"}}\n'
        '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":4}}\n'
    ).encode("utf-8")
    process = FakeProcess(
        stdout_chunks=[payload[:35], payload[35:87], payload[87:160], payload[160:]]
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
    assert bytes(process.stdin.buffer) == b"hello"
    assert process.stdin.closed is True
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
async def test_runner_prefers_final_snapshot_over_streamed_deltas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    process = FakeProcess(
        stdout_chunks=[
            '{"type":"thread.started","thread_id":"thread-123"}\n'.encode("utf-8"),
            '{"type":"agent_message_delta","delta":"Принял: сделаю обзор"}\n'.encode("utf-8"),
            '{"type":"agent_message_delta","delta":" проекта."}\n'.encode("utf-8"),
            '{"type":"item.completed","item":{"type":"assistant_message","text":"Полный обзор проекта:\\n- архитектура\\n- риски\\n- рекомендации"}}\n'.encode("utf-8"),
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
        stdout_chunks=[
            '{"type":"agent_message_delta","delta":"Черновик"}\n'.encode("utf-8"),
            '{"type":"item.completed","item":{"type":"assistant_message","text":"Короткий ответ"}}\n'.encode("utf-8"),
            '{"type":"item.completed","item":{"type":"assistant_message","text":""}}\n'.encode("utf-8"),
            '{"type":"item.completed","item":{"type":"assistant_message","text":"Финальный развёрнутый ответ"}}\n'.encode("utf-8"),
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
            stdout_chunks=[
                '{"type":"thread.started","thread_id":"fresh-thread"}\n'.encode("utf-8"),
                '{"type":"agent_message_delta","delta":"Recovered"}\n'.encode("utf-8"),
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
    process = FakeProcess(
        stdout_chunks=[
            '{"type":"item.completed","item":{"type":"assistant_message","text":"ok"}}\n'.encode(
                "utf-8"
            )
        ]
    )
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
    assert 'model_reasoning_effort="medium"' in captured["args"]
    assert captured["args"][-1] == "-"
    assert "--dangerously-bypass-approvals-and-sandbox" not in captured["args"]
    assert "--auto-edit" not in captured["args"]
    assert "--suggest" not in captured["args"]
    assert bytes(process.stdin.buffer) == b"hello"


@pytest.mark.asyncio
async def test_runner_builds_full_access_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    process = FakeProcess(
        stdout_chunks=[
            '{"type":"item.completed","item":{"type":"assistant_message","text":"ok"}}\n'.encode(
                "utf-8"
            )
        ]
    )
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
    assert captured["args"][-1] == "-"


@pytest.mark.asyncio
async def test_runner_overrides_model_and_reasoning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path, codex_model="gpt-5.3-codex")
    runner = CodexRunner(settings)
    process = FakeProcess(
        stdout_chunks=[
            '{"type":"item.completed","item":{"type":"assistant_message","text":"ok"}}\n'.encode(
                "utf-8"
            )
        ]
    )
    captured: dict[str, tuple] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(
        prompt="hello",
        cwd=tmp_path,
        launch_mode=CodexLaunchMode.SANDBOX,
        model_id="gpt-5.4-mini",
        reasoning_effort="high",
    )

    assert response.status == CodexResultStatus.SUCCESS
    assert ("--model", "gpt-5.4-mini") in list(zip(captured["args"], captured["args"][1:]))
    assert 'model_reasoning_effort="high"' in captured["args"]


@pytest.mark.asyncio
async def test_runner_writes_long_prompt_to_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    long_prompt = "prompt-" + ("x" * 100000)
    process = FakeProcess(
        stdout_chunks=[
            '{"type":"item.completed","item":{"type":"assistant_message","text":"ok"}}\n'.encode(
                "utf-8"
            )
        ]
    )
    captured: dict[str, tuple] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(
        prompt=long_prompt,
        cwd=tmp_path,
        launch_mode=CodexLaunchMode.SANDBOX,
    )

    assert response.status == CodexResultStatus.SUCCESS
    assert captured["args"][-1] == "-"
    assert bytes(process.stdin.buffer) == long_prompt.encode("utf-8")


@pytest.mark.asyncio
async def test_runner_handles_long_jsonl_event_without_readline_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    large_text = "x" * 70000
    line = (
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "assistant_message", "text": large_text},
            }
        )
        + "\n"
    ).encode("utf-8")
    process = FakeProcess(
        stdout_chunks=[line[:1024], line[1024:65550], line[65550:70010], line[70010:]]
    )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(prompt="large", cwd=tmp_path, launch_mode=CodexLaunchMode.SANDBOX)

    assert response.status == CodexResultStatus.SUCCESS
    assert response.final_text == large_text


@pytest.mark.asyncio
async def test_runner_uses_stdin_for_resume_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    process = FakeProcess(
        stdout_chunks=[
            '{"type":"item.completed","item":{"type":"assistant_message","text":"ok"}}\n'.encode(
                "utf-8"
            )
        ]
    )
    captured: dict[str, tuple] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(
        prompt="resume prompt",
        cwd=tmp_path,
        launch_mode=CodexLaunchMode.SANDBOX,
        previous_thread_id="thread-123",
    )

    assert response.status == CodexResultStatus.SUCCESS
    assert ("resume", "thread-123", "-") == captured["args"][-3:]
    assert bytes(process.stdin.buffer) == b"resume prompt"


@pytest.mark.asyncio
async def test_runner_returns_protocol_error_for_oversized_unterminated_jsonl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    runner = CodexRunner(settings)
    process = FakeProcess(
        stdout_chunks=[b"x" * (MAX_JSONL_EVENT_BYTES // 2), b"y" * (MAX_JSONL_EVENT_BYTES // 2 + 1)],
        wait_forever=True,
    )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    response = await runner.run(prompt="bad", cwd=tmp_path, launch_mode=CodexLaunchMode.SANDBOX)

    assert response.status == CodexResultStatus.PROTOCOL_ERROR
    assert "larger than the supported buffer" in response.error_message
    assert process.killed is True
