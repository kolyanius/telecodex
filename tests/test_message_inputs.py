from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_telegram_bot.config import Settings
from codex_telegram_bot.models import RequestContext
from codex_telegram_bot.telegram.inputs import MessageInputPreparer


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "telegram_bot_token": "token",
        "telegram_bot_username": "codex_bot",
        "approved_directory": tmp_path,
        "allowed_users": "42",
    }
    values.update(overrides)
    return Settings(**values)


class FakeLogger:
    def warning(self, *args, **kwargs) -> None:
        return None


class FakeObservability:
    def __init__(self) -> None:
        self.events: list[tuple[tuple, dict]] = []

    async def record_event(self, *args, **kwargs) -> None:
        self.events.append((args, kwargs))


class FakeTelegramFile:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.file_size = len(data)

    async def download_as_bytearray(self):
        return bytearray(self.data)


class FakePhoto:
    def __init__(self, data: bytes) -> None:
        self._file = FakeTelegramFile(data)

    async def get_file(self) -> FakeTelegramFile:
        return self._file


class FakeDocument:
    def __init__(self, filename: str, data: bytes) -> None:
        self.file_name = filename
        self._file = FakeTelegramFile(data)

    async def get_file(self) -> FakeTelegramFile:
        return self._file


class FakeProgressMessage:
    def __init__(self) -> None:
        self.deleted = False
        self.edited: list[str] = []

    async def delete(self) -> None:
        self.deleted = True

    async def edit_text(self, text: str) -> None:
        self.edited.append(text)


class FakeMessage:
    def __init__(self, *, caption: str = "", document=None, photo=None) -> None:
        self.caption = caption
        self.document = document
        self.photo = photo or []
        self.replies: list[tuple[str, dict]] = []

    async def reply_text(self, text: str, **kwargs):
        self.replies.append((text, kwargs))
        return FakeProgressMessage()


@pytest.mark.asyncio
async def test_prepare_document_builds_prompt_from_caption_and_text(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, enable_file_uploads=True)
    preparer = MessageInputPreparer(
        settings,
        voice=SimpleNamespace(),
        observability=FakeObservability(),
        logger=FakeLogger(),
    )
    message = FakeMessage(
        caption="Please review",
        document=FakeDocument("notes.txt", b"hello"),
    )
    update = SimpleNamespace(effective_message=message)

    prepared = await preparer.prepare_document(update, RequestContext(source="document"))

    assert prepared is not None
    assert prepared.source == "document"
    assert "User uploaded file `notes.txt`." in prepared.prompt
    assert "Caption: Please review" in prepared.prompt


@pytest.mark.asyncio
async def test_prepare_photo_returns_cleanup_paths(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, codex_enable_images=True)
    preparer = MessageInputPreparer(
        settings,
        voice=SimpleNamespace(),
        observability=FakeObservability(),
        logger=FakeLogger(),
    )
    message = FakeMessage(caption="Analyze this", photo=[FakePhoto(b"image-bytes")])
    update = SimpleNamespace(effective_message=message)

    prepared = await preparer.prepare_photo(update, RequestContext(source="photo"))

    assert prepared is not None
    assert prepared.prompt == "Analyze this"
    assert len(prepared.image_paths) == 1
    assert prepared.image_paths == prepared.cleanup_paths
    assert prepared.image_paths[0].exists()
    prepared.image_paths[0].unlink()


@pytest.mark.asyncio
async def test_prepare_document_replies_when_uploads_are_disabled(tmp_path: Path) -> None:
    observability = FakeObservability()
    settings = make_settings(tmp_path, enable_file_uploads=False)
    preparer = MessageInputPreparer(
        settings,
        voice=SimpleNamespace(),
        observability=observability,
        logger=FakeLogger(),
    )
    message = FakeMessage(document=FakeDocument("notes.txt", b"hello"))
    update = SimpleNamespace(effective_message=message)

    prepared = await preparer.prepare_document(update, RequestContext(source="document"))

    assert prepared is None
    assert message.replies[-1][0] == "File uploads are disabled."
    assert observability.events[-1][0][0] == "unsupported_input"
