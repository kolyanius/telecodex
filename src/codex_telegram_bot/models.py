from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class CodexResultStatus(StringEnum):
    SUCCESS = "success"
    INTERRUPTED = "interrupted"
    RESUME_FAILED = "resume_failed"
    TIMEOUT = "timeout"
    CLI_ERROR = "cli_error"
    PROTOCOL_ERROR = "protocol_error"


class CodexLaunchMode(StringEnum):
    SANDBOX = "sandbox"
    FULL_ACCESS = "full_access"

    @classmethod
    def from_value(cls, value: Any) -> "CodexLaunchMode":
        if isinstance(value, cls):
            return value
        normalized = str(value or cls.SANDBOX.value).strip().lower()
        if normalized == cls.FULL_ACCESS.value:
            return cls.FULL_ACCESS
        return cls.SANDBOX


class CodexStreamEventKind(StringEnum):
    TEXT_DELTA = "text_delta"
    TEXT_SNAPSHOT = "text_snapshot"
    TOOL_CALL = "tool_call"
    LIFECYCLE = "lifecycle"
    USAGE = "usage"
    UNKNOWN = "unknown"


@dataclass
class CodexToolCall:
    name: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodexStreamEvent:
    type: str
    kind: CodexStreamEventKind = CodexStreamEventKind.UNKNOWN
    text_delta: str = ""
    text_snapshot: str = ""
    tool_call: Optional[CodexToolCall] = None
    lifecycle_name: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodexResponse:
    final_text: str
    thread_id: str
    status: CodexResultStatus = CodexResultStatus.SUCCESS
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    error_message: str = ""
    fallback_reason: str = ""
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    interrupted: bool = False


@dataclass
class ProjectSession:
    user_id: int
    project_path: str
    thread_id: str
    updated_at: str
    last_status: str = ""
    last_error: str = ""


@dataclass
class PreparedCodexRequest:
    prompt: str
    source: str
    image_paths: list[Path] = field(default_factory=list)
    cleanup_paths: list[Path] = field(default_factory=list)


@dataclass
class RequestContext:
    source: str
    user_id: Optional[int] = None
    chat_id: Optional[int] = None
    message_id: Optional[int] = None
    chat_type: str = ""
    cwd: str = ""
    command_name: str = ""
    has_previous_thread: bool = False
    prompt_chars: int = 0
    caption_chars: int = 0
    document_name: str = ""
    image_count: int = 0
    voice_duration_seconds: int = 0
    launch_mode: str = ""


@dataclass
class ProcessedVoice:
    prompt: str
    transcription: str
    duration_seconds: int


@dataclass
class ProcessedDocument:
    prompt: str
    filename: str


@dataclass
class ProcessedImage:
    prompt: str
    image_path: Path
