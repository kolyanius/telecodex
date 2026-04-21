from __future__ import annotations

from pathlib import Path

import pytest

from codex_telegram_bot.config import Settings
from codex_telegram_bot.voice import VoiceTranscriber


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "telegram_bot_token": "token",
        "telegram_bot_username": "codex_bot",
        "approved_directory": tmp_path,
    }
    values.update(overrides)
    return Settings(**values)


class FakeTelegramFile:
    def __init__(self, data: bytes, file_size: int | None):
        self._data = data
        self.file_size = file_size

    async def download_as_bytearray(self) -> bytearray:
        return bytearray(self._data)


class FakeVoice:
    def __init__(self, *, data: bytes, file_size: int | None, resolved_file_size: int | None):
        self.file_size = file_size
        self.duration = 5
        self._file = FakeTelegramFile(data=data, file_size=resolved_file_size)

    async def get_file(self) -> FakeTelegramFile:
        return self._file


@pytest.mark.asyncio
async def test_voice_rejects_unknown_size(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, voice_provider="openai")
    transcriber = VoiceTranscriber(settings)
    voice = FakeVoice(data=b"voice-bytes", file_size=None, resolved_file_size=None)

    with pytest.raises(ValueError, match="size is unknown"):
        await transcriber.transcribe(voice)


@pytest.mark.asyncio
async def test_voice_rejects_large_files_before_download(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, voice_provider="openai", voice_max_file_size_mb=1)
    transcriber = VoiceTranscriber(settings)
    voice = FakeVoice(
        data=b"a" * 10,
        file_size=2 * 1024 * 1024,
        resolved_file_size=2 * 1024 * 1024,
    )

    with pytest.raises(ValueError, match="Voice file too large"):
        await transcriber.transcribe(voice)


@pytest.mark.asyncio
async def test_voice_uses_openai_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        voice_provider="openai",
        openai_api_key="key",
    )
    transcriber = VoiceTranscriber(settings)
    voice = FakeVoice(data=b"voice", file_size=5, resolved_file_size=5)

    async def fake_transcribe_openai(data: bytes) -> str:
        assert data == b"voice"
        return "hello from openai"

    monkeypatch.setattr(transcriber, "_transcribe_openai", fake_transcribe_openai)

    processed = await transcriber.transcribe(voice, caption="Caption")
    assert processed.transcription == "hello from openai"
    assert processed.prompt == "Caption\n\nhello from openai"


@pytest.mark.asyncio
async def test_voice_uses_openai_compatible_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(
        tmp_path,
        voice_provider="openai_compatible",
        voice_api_key="key",
        voice_api_base_url="https://api.groq.com/openai/v1",
        voice_transcription_model="whisper-large-v3-turbo",
    )
    transcriber = VoiceTranscriber(settings)
    voice = FakeVoice(data=b"voice", file_size=5, resolved_file_size=5)

    async def fake_transcribe_openai_compatible(data: bytes) -> str:
        assert data == b"voice"
        return "hello from groq"

    monkeypatch.setattr(
        transcriber, "_transcribe_openai_compatible", fake_transcribe_openai_compatible
    )

    processed = await transcriber.transcribe(voice)
    assert processed.transcription == "hello from groq"
    assert processed.prompt == "Voice message transcription:\n\nhello from groq"


def test_openai_compatible_client_uses_base_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        voice_provider="openai_compatible",
        voice_api_key="key",
        voice_api_base_url="https://api.groq.com/openai/v1",
        voice_transcription_model="whisper-large-v3-turbo",
    )
    transcriber = VoiceTranscriber(settings)
    captured: dict[str, str] = {}

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)

    transcriber._get_openai_compatible_client()

    assert captured == {
        "api_key": "key",
        "base_url": "https://api.groq.com/openai/v1",
    }


@pytest.mark.asyncio
async def test_openai_compatible_transcription_uses_configured_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(
        tmp_path,
        voice_provider="openai_compatible",
        voice_api_key="key",
        voice_api_base_url="https://api.groq.com/openai/v1",
        voice_transcription_model="whisper-large-v3-turbo",
    )
    transcriber = VoiceTranscriber(settings)
    captured: dict[str, object] = {}

    class FakeTranscriptions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"text": "hello"})()

    client = type("Client", (), {"audio": type("Audio", (), {"transcriptions": FakeTranscriptions()})()})()

    monkeypatch.setattr(transcriber, "_get_openai_compatible_client", lambda: client)

    text = await transcriber._transcribe_openai_compatible(b"voice")

    assert text == "hello"
    assert captured["model"] == "whisper-large-v3-turbo"
    assert captured["file"] == ("voice.ogg", b"voice")
    assert captured["temperature"] == 0


@pytest.mark.asyncio
async def test_openai_compatible_transcription_normalizes_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(
        tmp_path,
        voice_provider="openai_compatible",
        voice_api_key="key",
        voice_api_base_url="https://api.groq.com/openai/v1",
        voice_transcription_model="whisper-large-v3-turbo",
    )
    transcriber = VoiceTranscriber(settings)

    class FakeTranscriptions:
        async def create(self, **kwargs):
            raise RuntimeError("boom")

    client = type("Client", (), {"audio": type("Audio", (), {"transcriptions": FakeTranscriptions()})()})()
    monkeypatch.setattr(transcriber, "_get_openai_compatible_client", lambda: client)

    with pytest.raises(RuntimeError, match="OpenAI-compatible transcription request failed."):
        await transcriber._transcribe_openai_compatible(b"voice")
