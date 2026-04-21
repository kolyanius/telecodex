from __future__ import annotations

from pathlib import Path

import pytest

from codex_telegram_bot.config import Settings


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "telegram_bot_token": "token",
        "telegram_bot_username": "codex_bot",
        "approved_directory": tmp_path,
        "allowed_users": "",
        "voice_provider": "openai",
        "openai_api_key": "",
        "voice_api_key": "",
        "voice_api_base_url": "",
        "voice_transcription_model": "",
    }
    values.update(overrides)
    return Settings(**values)


def test_settings_parse_allowed_users_and_defaults(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        allowed_users="1, 2,3",
        voice_provider="openai",
    )

    assert settings.allowed_users == [1, 2, 3]
    assert settings.voice_provider == "openai"
    assert settings.resolved_voice_model == "whisper-1"


def test_settings_parse_allowed_users_json_string_directly(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        allowed_users="[1,2,3]",
        voice_provider="openai",
    )

    assert settings.allowed_users == [1, 2, 3]


def test_settings_parse_allowed_users_from_env_file_json_list(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=token",
                "TELEGRAM_BOT_USERNAME=codex_bot",
                f"APPROVED_DIRECTORY={tmp_path}",
                "ALLOWED_USERS=[1,2,3]",
                "VOICE_PROVIDER=openai",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.allowed_users == [1, 2, 3]


def test_settings_parse_scalar_allowed_users_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=token",
                "TELEGRAM_BOT_USERNAME=codex_bot",
                f"APPROVED_DIRECTORY={tmp_path}",
                "ALLOWED_USERS=1",
                "VOICE_PROVIDER=openai",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.allowed_users == [1]


def test_settings_parse_csv_allowed_users_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=token",
                "TELEGRAM_BOT_USERNAME=codex_bot",
                f"APPROVED_DIRECTORY={tmp_path}",
                "ALLOWED_USERS=1, 2,3",
                "VOICE_PROVIDER=openai",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.allowed_users == [1, 2, 3]


def test_settings_validate_log_level(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, log_level="debug")
    assert settings.log_level == "DEBUG"

    with pytest.raises(ValueError, match="LOG_LEVEL"):
        make_settings(tmp_path, log_level="TRACE")


def test_settings_validate_timeout_and_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CODEX_TIMEOUT_SECONDS"):
        make_settings(tmp_path, codex_timeout_seconds=0)

    with pytest.raises(ValueError, match="MAX_ACTIVE_RUNS_PER_USER"):
        make_settings(tmp_path, max_active_runs_per_user=0)


def test_settings_validate_approved_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="APPROVED_DIRECTORY"):
        make_settings(tmp_path, approved_directory=missing)


def test_settings_validate_openai_compatible_voice_provider(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        voice_provider="openai_compatible",
        voice_api_key="key",
        voice_api_base_url="https://api.groq.com/openai/v1/",
        voice_transcription_model="whisper-large-v3-turbo",
    )

    assert settings.voice_provider == "openai_compatible"
    assert settings.voice_api_base_url == "https://api.groq.com/openai/v1"
    assert settings.resolved_voice_model == "whisper-large-v3-turbo"


def test_settings_require_openai_compatible_voice_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="VOICE_API_KEY"):
        make_settings(
            tmp_path,
            voice_provider="openai_compatible",
            voice_api_base_url="https://api.groq.com/openai/v1",
            voice_transcription_model="whisper-large-v3-turbo",
        )

    with pytest.raises(ValueError, match="VOICE_API_BASE_URL"):
        make_settings(
            tmp_path,
            voice_provider="openai_compatible",
            voice_api_key="key",
            voice_transcription_model="whisper-large-v3-turbo",
        )

    with pytest.raises(ValueError, match="VOICE_TRANSCRIPTION_MODEL"):
        make_settings(
            tmp_path,
            voice_provider="openai_compatible",
            voice_api_key="key",
            voice_api_base_url="https://api.groq.com/openai/v1",
        )


def test_settings_reject_removed_voice_providers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="VOICE_PROVIDER"):
        make_settings(tmp_path, voice_provider="mistral")

    with pytest.raises(ValueError, match="VOICE_PROVIDER"):
        make_settings(tmp_path, voice_provider="local")


def test_settings_validate_default_launch_mode(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, codex_default_launch_mode="full_access")

    assert str(settings.codex_default_launch_mode) == "full_access"

    with pytest.raises(ValueError, match="CODEX_DEFAULT_LAUNCH_MODE"):
        make_settings(tmp_path, codex_default_launch_mode="unsafe")
