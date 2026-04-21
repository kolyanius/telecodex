from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .models import CodexLaunchMode


class Settings(BaseSettings):
    telegram_bot_token: SecretStr
    telegram_bot_username: str

    approved_directory: Path
    allowed_users: Annotated[list[int], NoDecode] = Field(default_factory=list)

    database_url: str = "sqlite:///./codex_telegram_bot.db"

    codex_cli_path: str = "codex"
    codex_model: Optional[str] = "gpt-5.3-codex"
    codex_full_auto: bool = True
    codex_auto_edit: bool = False
    codex_default_launch_mode: CodexLaunchMode = CodexLaunchMode.SANDBOX
    codex_skip_git_repo_check: bool = True
    codex_timeout_seconds: int = 900
    codex_enable_images: bool = False

    enable_file_uploads: bool = True
    enable_voice_messages: bool = True
    voice_provider: str = "openai"  # openai | openai_compatible
    openai_api_key: Optional[SecretStr] = None
    voice_api_key: Optional[SecretStr] = None
    voice_api_base_url: Optional[str] = None
    voice_transcription_model: Optional[str] = None
    voice_max_file_size_mb: int = 20

    verbose_level: int = 1
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60
    max_active_runs_per_user: int = 1
    enable_audit_log: bool = True
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("allowed_users", mode="before")
    @classmethod
    def parse_allowed_users(cls, value):
        if value is None or value == "":
            return []
        if isinstance(value, int):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [int(x) for x in value]
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return []
            if normalized.startswith("[") and normalized.endswith("]"):
                parsed = json.loads(normalized)
                if isinstance(parsed, list):
                    return [int(x) for x in parsed]
                return [int(parsed)]
            return [int(x.strip()) for x in normalized.split(",") if x.strip()]
        return value

    @field_validator("approved_directory")
    @classmethod
    def validate_approved_directory(cls, value: Path) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise ValueError(f"APPROVED_DIRECTORY must exist and be a directory: {path}")
        return path

    @field_validator("voice_provider", mode="before")
    @classmethod
    def validate_voice_provider(cls, value: str) -> str:
        provider = str(value or "openai").strip().lower()
        if provider not in {"openai", "openai_compatible"}:
            raise ValueError("VOICE_PROVIDER must be one of: openai, openai_compatible")
        return provider

    @field_validator("voice_api_base_url", mode="before")
    @classmethod
    def normalize_voice_api_base_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        base_url = str(value).strip()
        if not base_url:
            return None
        return base_url.rstrip("/")

    @field_validator("verbose_level")
    @classmethod
    def validate_verbose_level(cls, value: int) -> int:
        if value not in (0, 1, 2):
            raise ValueError("VERBOSE_LEVEL must be 0, 1, or 2")
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        level = str(value or "INFO").strip().upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(sorted(valid_levels))}")
        return level

    @field_validator("codex_timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("CODEX_TIMEOUT_SECONDS must be > 0")
        return value

    @field_validator("codex_default_launch_mode", mode="before")
    @classmethod
    def validate_codex_default_launch_mode(cls, value) -> CodexLaunchMode:
        normalized = str(value or CodexLaunchMode.SANDBOX.value).strip().lower()
        if normalized not in {CodexLaunchMode.SANDBOX.value, CodexLaunchMode.FULL_ACCESS.value}:
            raise ValueError("CODEX_DEFAULT_LAUNCH_MODE must be sandbox or full_access")
        return CodexLaunchMode.from_value(normalized)

    @field_validator("max_active_runs_per_user")
    @classmethod
    def validate_max_active_runs_per_user(cls, value: int) -> int:
        if value < 1:
            raise ValueError("MAX_ACTIVE_RUNS_PER_USER must be >= 1")
        return value

    @property
    def telegram_token_str(self) -> str:
        return self.telegram_bot_token.get_secret_value()

    @property
    def sqlite_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("Only sqlite:/// URLs are supported in this implementation")
        return Path(self.database_url[len(prefix):]).expanduser().resolve()

    @property
    def openai_api_key_str(self) -> Optional[str]:
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None

    @property
    def voice_api_key_str(self) -> Optional[str]:
        return self.voice_api_key.get_secret_value() if self.voice_api_key else None

    @property
    def voice_max_file_size_bytes(self) -> int:
        return self.voice_max_file_size_mb * 1024 * 1024

    @property
    def resolved_voice_model(self) -> str:
        if self.voice_transcription_model:
            return self.voice_transcription_model
        if self.voice_provider == "openai":
            return "whisper-1"
        if self.voice_provider == "openai_compatible":
            raise ValueError(
                "VOICE_TRANSCRIPTION_MODEL must be set when VOICE_PROVIDER=openai_compatible"
            )
        return "whisper-1"

    @field_validator("openai_api_key", "voice_api_key", mode="before")
    @classmethod
    def empty_secret_to_none(cls, value):
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("voice_transcription_model", mode="before")
    @classmethod
    def empty_voice_model_to_none(cls, value):
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def model_post_init(self, __context) -> None:
        if self.voice_provider == "openai_compatible":
            if not self.voice_api_key_str:
                raise ValueError(
                    "VOICE_API_KEY must be configured when VOICE_PROVIDER=openai_compatible"
                )
            if not self.voice_api_base_url:
                raise ValueError(
                    "VOICE_API_BASE_URL must be configured when VOICE_PROVIDER=openai_compatible"
                )
            if not self.voice_transcription_model:
                raise ValueError(
                    "VOICE_TRANSCRIPTION_MODEL must be configured when "
                    "VOICE_PROVIDER=openai_compatible"
                )
