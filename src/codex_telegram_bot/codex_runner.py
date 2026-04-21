from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import shutil
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

import structlog

from .config import Settings
from .models import (
    CodexLaunchMode,
    CodexResponse,
    CodexResultStatus,
    CodexStreamEvent,
    CodexStreamEventKind,
    CodexToolCall,
)

logger = structlog.get_logger(__name__)

StreamCallback = Callable[[CodexStreamEvent], Awaitable[None]]
STDOUT_READ_CHUNK_SIZE = 16 * 1024
MAX_JSONL_EVENT_BYTES = 1024 * 1024

RECOVERABLE_RESUME_PATTERNS = (
    "no conversation found",
    "session not found",
    "thread not found",
    "could not resume",
    "cannot resume",
    "resume failed",
    "invalid thread",
)


class CodexRunner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate_cli_available(self) -> None:
        cli_path = self.settings.codex_cli_path
        if os.sep in cli_path:
            candidate = Path(cli_path).expanduser()
            if not candidate.exists():
                raise RuntimeError(f"Codex CLI not found at configured path: {candidate}")
            if not os.access(candidate, os.X_OK):
                raise RuntimeError(f"Codex CLI is not executable: {candidate}")
            return

        resolved = shutil.which(cli_path)
        if resolved is None:
            raise RuntimeError(
                f"Codex CLI '{cli_path}' was not found in PATH. "
                "Install Codex or set CODEX_CLI_PATH explicitly."
            )

    async def run(
        self,
        *,
        prompt: str,
        cwd: Path,
        launch_mode: CodexLaunchMode,
        previous_thread_id: Optional[str] = None,
        on_event: Optional[StreamCallback] = None,
        interrupt_event: Optional[asyncio.Event] = None,
        image_paths: Optional[list[Path]] = None,
    ) -> CodexResponse:
        cwd = cwd.resolve()
        self._ensure_in_workspace(cwd)

        response = await self._run_once(
            prompt=prompt,
            cwd=cwd,
            launch_mode=launch_mode,
            previous_thread_id=previous_thread_id,
            on_event=on_event,
            interrupt_event=interrupt_event,
            image_paths=image_paths,
        )
        if previous_thread_id and response.status == CodexResultStatus.RESUME_FAILED:
            logger.warning(
                "codex_resume_fallback",
                cwd=str(cwd),
                thread_id=previous_thread_id,
                reason=response.error_message,
            )
            retry_response = await self._run_once(
                prompt=prompt,
                cwd=cwd,
                launch_mode=launch_mode,
                previous_thread_id=None,
                on_event=on_event,
                interrupt_event=interrupt_event,
                image_paths=image_paths,
            )
            retry_response.fallback_reason = response.error_message
            return retry_response
        return response

    async def _run_once(
        self,
        *,
        prompt: str,
        cwd: Path,
        launch_mode: CodexLaunchMode,
        previous_thread_id: Optional[str],
        on_event: Optional[StreamCallback],
        interrupt_event: Optional[asyncio.Event],
        image_paths: Optional[list[Path]],
    ) -> CodexResponse:
        cmd = [self.settings.codex_cli_path, "exec", "--json"]
        if self.settings.codex_skip_git_repo_check:
            cmd.append("--skip-git-repo-check")
        if launch_mode == CodexLaunchMode.SANDBOX:
            cmd.extend(["--sandbox", "read-only", "--config", 'web_search="disabled"'])
        else:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        if self.settings.codex_model:
            cmd.extend(["--model", self.settings.codex_model])
        if previous_thread_id:
            cmd.extend(["resume", previous_thread_id])
        if image_paths:
            if not self.settings.codex_enable_images:
                return CodexResponse(
                    final_text="",
                    thread_id=previous_thread_id or "",
                    status=CodexResultStatus.CLI_ERROR,
                    error_message="Image support is disabled by config",
                )
            for image_path in image_paths:
                cmd.extend(["--image", str(image_path)])
        cmd.append("-")

        env = os.environ.copy()
        env["PWD"] = str(cwd)

        start_time = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return self._build_response(
                status=CodexResultStatus.CLI_ERROR,
                start_time=start_time,
                thread_id=previous_thread_id or "",
                error_message=(
                    f"Codex CLI '{self.settings.codex_cli_path}' is not available. "
                    "Check CODEX_CLI_PATH or PATH."
                ),
            )
        except Exception as exc:
            return self._build_response(
                status=CodexResultStatus.CLI_ERROR,
                start_time=start_time,
                thread_id=previous_thread_id or "",
                error_message=f"Failed to launch Codex CLI: {exc}",
            )

        state = _RunState(thread_id=previous_thread_id or "", start_time=start_time)

        interrupt_task = asyncio.create_task(
            self._watch_interrupt(
                process=process,
                interrupt_event=interrupt_event,
                state=state,
            )
        )
        timeout_task = asyncio.create_task(self._watch_timeout(process=process, state=state))

        try:
            await self._write_prompt(process=process, prompt=prompt)
            assert process.stdout is not None
            await self._consume_stdout(process=process, state=state, on_event=on_event)

            stderr = ""
            if process.stderr is not None:
                stderr = (await process.stderr.read()).decode("utf-8", errors="replace").strip()

            rc = await process.wait()
        finally:
            interrupt_task.cancel()
            timeout_task.cancel()

        if state.interrupted:
            return self._build_response(
                status=CodexResultStatus.INTERRUPTED,
                start_time=start_time,
                thread_id=state.thread_id,
                final_text=state.resolved_final_text,
                input_tokens=state.input_tokens,
                cached_input_tokens=state.cached_input_tokens,
                output_tokens=state.output_tokens,
                raw_events=state.raw_events,
                interrupted=True,
            )

        if state.timed_out:
            return self._build_response(
                status=CodexResultStatus.TIMEOUT,
                start_time=start_time,
                thread_id=state.thread_id,
                final_text=state.resolved_final_text,
                input_tokens=state.input_tokens,
                cached_input_tokens=state.cached_input_tokens,
                output_tokens=state.output_tokens,
                raw_events=state.raw_events,
                error_message=(
                    f"Codex timed out after {self.settings.codex_timeout_seconds} seconds."
                ),
            )

        if state.protocol_error_message:
            return self._build_response(
                status=CodexResultStatus.PROTOCOL_ERROR,
                start_time=start_time,
                thread_id=state.thread_id,
                final_text=state.resolved_final_text,
                input_tokens=state.input_tokens,
                cached_input_tokens=state.cached_input_tokens,
                output_tokens=state.output_tokens,
                raw_events=state.raw_events,
                error_message=state.protocol_error_message,
            )

        if rc != 0:
            status = CodexResultStatus.CLI_ERROR
            if previous_thread_id and self._is_recoverable_resume_failure(stderr):
                status = CodexResultStatus.RESUME_FAILED
            return self._build_response(
                status=status,
                start_time=start_time,
                thread_id=state.thread_id,
                final_text=state.resolved_final_text,
                input_tokens=state.input_tokens,
                cached_input_tokens=state.cached_input_tokens,
                output_tokens=state.output_tokens,
                raw_events=state.raw_events,
                error_message=(
                    f"Codex CLI exited with code {rc}. "
                    f"Command: {' '.join(shlex.quote(x) for x in cmd)}. "
                    f"stderr: {stderr[:1000]}"
                ),
            )

        if not state.raw_events and state.invalid_json_lines:
            return self._build_response(
                status=CodexResultStatus.PROTOCOL_ERROR,
                start_time=start_time,
                thread_id=state.thread_id,
                error_message="Codex returned no valid JSON events.",
            )

        return self._build_response(
            status=CodexResultStatus.SUCCESS,
            start_time=start_time,
            thread_id=state.thread_id,
            final_text=state.resolved_final_text,
            input_tokens=state.input_tokens,
            cached_input_tokens=state.cached_input_tokens,
            output_tokens=state.output_tokens,
            raw_events=state.raw_events,
        )

    async def _write_prompt(
        self,
        *,
        process: asyncio.subprocess.Process,
        prompt: str,
    ) -> None:
        if process.stdin is None:
            return

        try:
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            process.stdin.close()
            wait_closed = getattr(process.stdin, "wait_closed", None)
            if callable(wait_closed):
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    await wait_closed()

    async def _consume_stdout(
        self,
        *,
        process: asyncio.subprocess.Process,
        state: "_RunState",
        on_event: Optional[StreamCallback],
    ) -> None:
        assert process.stdout is not None
        buffer = bytearray()

        while True:
            chunk = await process.stdout.read(STDOUT_READ_CHUNK_SIZE)
            if not chunk:
                break
            buffer.extend(chunk)

            while True:
                newline_index = buffer.find(b"\n")
                if newline_index < 0:
                    break
                raw_line = bytes(buffer[:newline_index])
                del buffer[: newline_index + 1]
                await self._process_jsonl_line(raw_line=raw_line, state=state, on_event=on_event)

            if len(buffer) > MAX_JSONL_EVENT_BYTES:
                state.protocol_error_message = (
                    "Codex emitted a JSONL event larger than the supported buffer "
                    f"({MAX_JSONL_EVENT_BYTES} bytes) without a newline separator."
                )
                if process.returncode is None:
                    process.kill()
                return

        if buffer:
            await self._process_jsonl_line(raw_line=bytes(buffer), state=state, on_event=on_event)

    async def _process_jsonl_line(
        self,
        *,
        raw_line: bytes,
        state: "_RunState",
        on_event: Optional[StreamCallback],
    ) -> None:
        line_str = raw_line.decode("utf-8", errors="replace").strip()
        if not line_str:
            return

        try:
            event = json.loads(line_str)
        except json.JSONDecodeError:
            state.invalid_json_lines += 1
            logger.warning("codex_invalid_json_event", line=line_str[:300])
            return

        state.raw_events.append(event)
        normalized = self._normalize_event(event)
        if normalized is None:
            return

        if normalized.kind == CodexStreamEventKind.TEXT_DELTA and normalized.text_delta:
            state.streamed_text_parts.append(normalized.text_delta)
        elif normalized.kind == CodexStreamEventKind.TEXT_SNAPSHOT and normalized.text_snapshot:
            state.final_snapshot_text = normalized.text_snapshot
        elif normalized.kind == CodexStreamEventKind.USAGE:
            state.input_tokens = int(normalized.usage.get("input_tokens", 0))
            state.cached_input_tokens = int(normalized.usage.get("cached_input_tokens", 0))
            state.output_tokens = int(normalized.usage.get("output_tokens", 0))
        elif normalized.lifecycle_name == "thread.started":
            state.thread_id = event.get("thread_id", state.thread_id)

        if on_event:
            maybe_awaitable = on_event(normalized)
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable

    async def _watch_interrupt(
        self,
        *,
        process: asyncio.subprocess.Process,
        interrupt_event: Optional[asyncio.Event],
        state: "_RunState",
    ) -> None:
        if interrupt_event is None:
            return
        await interrupt_event.wait()
        state.interrupted = True
        if process.returncode is None:
            process.kill()

    async def _watch_timeout(
        self,
        *,
        process: asyncio.subprocess.Process,
        state: "_RunState",
    ) -> None:
        await asyncio.sleep(self.settings.codex_timeout_seconds)
        if process.returncode is None and not state.interrupted:
            state.timed_out = True
            process.kill()

    def _normalize_event(self, event: dict) -> Optional[CodexStreamEvent]:
        event_type = str(event.get("type", "") or "")
        if event_type == "thread.started":
            return CodexStreamEvent(
                type=event_type,
                kind=CodexStreamEventKind.LIFECYCLE,
                lifecycle_name=event_type,
                raw=event,
            )

        if event_type == "agent_message_delta":
            delta = str(event.get("delta") or event.get("text") or "")
            return CodexStreamEvent(
                type=event_type,
                kind=CodexStreamEventKind.TEXT_DELTA,
                text_delta=delta,
                raw=event,
            )

        if event_type == "item.completed":
            item = event.get("item", {}) or {}
            item_type = str(item.get("type", "") or "")
            if item_type in {"agent_message", "assistant_message"}:
                return CodexStreamEvent(
                    type=event_type,
                    kind=CodexStreamEventKind.TEXT_SNAPSHOT,
                    text_snapshot=str(item.get("text") or ""),
                    raw=event,
                )
            tool_name = (
                item.get("name")
                or item.get("tool_name")
                or item.get("command")
                or item_type
                or "tool"
            )
            return CodexStreamEvent(
                type=event_type,
                kind=CodexStreamEventKind.TOOL_CALL,
                tool_call=CodexToolCall(name=str(tool_name), raw=item),
                raw=event,
            )

        if event_type == "turn.completed":
            usage = event.get("usage", {}) or {}
            return CodexStreamEvent(
                type=event_type,
                kind=CodexStreamEventKind.USAGE,
                usage={
                    "input_tokens": int(usage.get("input_tokens", 0) or 0),
                    "cached_input_tokens": int(usage.get("cached_input_tokens", 0) or 0),
                    "output_tokens": int(usage.get("output_tokens", 0) or 0),
                },
                raw=event,
            )

        if event_type:
            return CodexStreamEvent(
                type=event_type,
                kind=CodexStreamEventKind.LIFECYCLE,
                lifecycle_name=event_type,
                raw=event,
            )
        return None

    def _is_recoverable_resume_failure(self, stderr: str) -> bool:
        lowered = stderr.lower()
        return any(pattern in lowered for pattern in RECOVERABLE_RESUME_PATTERNS)

    def _build_response(
        self,
        *,
        status: CodexResultStatus,
        start_time: float,
        thread_id: str,
        final_text: str = "",
        input_tokens: int = 0,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
        error_message: str = "",
        raw_events: Optional[list[dict]] = None,
        interrupted: bool = False,
    ) -> CodexResponse:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CodexResponse(
            final_text=final_text,
            thread_id=thread_id,
            status=status,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            error_message=error_message,
            raw_events=raw_events or [],
            interrupted=interrupted,
        )

    def _ensure_in_workspace(self, candidate: Path) -> None:
        root = self.settings.approved_directory.resolve()
        try:
            candidate.resolve().relative_to(root)
        except ValueError as exc:
            raise PermissionError(f"Path outside approved directory: {candidate}") from exc


class _RunState:
    def __init__(self, *, thread_id: str, start_time: float):
        self.thread_id = thread_id
        self.start_time = start_time
        self.streamed_text_parts: list[str] = []
        self.final_snapshot_text = ""
        self.raw_events: list[dict] = []
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.invalid_json_lines = 0
        self.interrupted = False
        self.timed_out = False
        self.protocol_error_message = ""

    @property
    def resolved_final_text(self) -> str:
        if self.final_snapshot_text.strip():
            return self.final_snapshot_text.strip()
        return "".join(self.streamed_text_parts).strip()
