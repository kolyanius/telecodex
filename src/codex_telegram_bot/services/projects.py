from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional

from telegram.ext import ContextTypes

from ..config import Settings
from ..models import RequestContext

RecordEvent = Callable[..., Awaitable[None]]


@dataclass(frozen=True)
class RepoOption:
    slug: str
    label: str
    is_current: bool = False


@dataclass(frozen=True)
class ProjectResolution:
    path: Optional[Path]
    auto_created: bool = False


class ProjectService:
    def __init__(self, settings: Settings, record_event: RecordEvent):
        self.settings = settings
        self._record_event = record_event

    async def resolve_current_project(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        request_context: Optional[RequestContext] = None,
        create_if_empty: bool = True,
    ) -> ProjectResolution:
        root = self.settings.approved_directory.resolve()
        current = context.user_data.get("current_directory")
        if current is not None:
            try:
                current_path = Path(current).resolve()
                self.ensure_in_workspace(current_path)
                if current_path.exists() and current_path.is_dir() and current_path.parent == root:
                    return ProjectResolution(path=current_path)
            except Exception:
                pass

        projects = self.list_project_paths()
        if projects:
            context.user_data["current_directory"] = projects[0]
            return ProjectResolution(path=projects[0])

        if create_if_empty and self.workspace_is_empty(root):
            project = await self.create_project(None, context=context, request_context=request_context, auto=True)
            return ProjectResolution(path=project, auto_created=True)

        return ProjectResolution(path=None)

    def ensure_in_workspace(self, path: Path) -> None:
        root = self.settings.approved_directory.resolve()
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise PermissionError(f"Path outside approved directory: {path}") from exc

    def list_repo_options(self, context: ContextTypes.DEFAULT_TYPE) -> tuple[list[RepoOption], bool]:
        current = context.user_data.get("current_directory")
        current_path = Path(current).resolve() if current is not None else None
        entries = self.list_project_paths()
        truncated = len(entries) > 20
        options = [
            RepoOption(
                slug=entry.name,
                label=f"{entry.name} • current" if current_path is not None and entry == current_path else entry.name,
                is_current=current_path is not None and entry == current_path,
            )
            for entry in entries[:20]
        ]
        return options, truncated

    def resolve_repo_slug(self, slug: str) -> Path:
        base = self.settings.approved_directory.resolve()
        candidate = (base / slug).resolve()
        self.ensure_in_workspace(candidate)
        if not candidate.exists():
            raise FileNotFoundError(slug)
        if not candidate.is_dir():
            raise NotADirectoryError(slug)
        if candidate.parent != base:
            raise PermissionError(slug)
        return candidate

    def list_project_paths(self) -> list[Path]:
        base = self.settings.approved_directory.resolve()
        return [p for p in sorted(base.iterdir()) if p.is_dir() and not p.name.startswith(".")]

    def workspace_is_empty(self, root: Path) -> bool:
        for entry in root.iterdir():
            if entry.name == ".DS_Store":
                continue
            if entry.suffix in {".sqlite", ".sqlite3"}:
                continue
            if entry.name.endswith((".sqlite-wal", ".sqlite-shm", ".sqlite3-wal", ".sqlite3-shm")):
                continue
            if entry.name in {
                self.settings.sqlite_path.name,
                f"{self.settings.sqlite_path.name}-wal",
                f"{self.settings.sqlite_path.name}-shm",
            }:
                continue
            return False
        return True

    @staticmethod
    def sanitize_project_name(name: str) -> str:
        candidate = name.strip().lower()
        candidate = re.sub(r"[\\/]+", "-", candidate)
        candidate = re.sub(r"[^a-z0-9._-]+", "-", candidate)
        candidate = re.sub(r"-{2,}", "-", candidate).strip(" .-_")
        if not candidate or candidate in {".", ".."}:
            raise ValueError("Invalid project name")
        return candidate

    def default_project_slug(self) -> str:
        return f"{datetime.now().astimezone().date().isoformat()}-project"

    def next_available_project_path(self, base_slug: str) -> Path:
        base = self.settings.approved_directory.resolve()
        candidate = base / base_slug
        if not candidate.exists():
            return candidate
        suffix = 2
        while True:
            candidate = base / f"{base_slug}-{suffix}"
            if not candidate.exists():
                return candidate
            suffix += 1

    async def create_project(
        self,
        name: Optional[str],
        *,
        context: ContextTypes.DEFAULT_TYPE,
        request_context: Optional[RequestContext],
        auto: bool = False,
    ) -> Path:
        requested_name = name.strip() if name else ""
        await self._record_event(
            "project_create_requested",
            request_context,
            audit_event="project_create_requested",
            requested_name=requested_name,
            auto=auto,
        )
        try:
            base_slug = self.default_project_slug() if auto else self.sanitize_project_name(requested_name)
            project_path = self.next_available_project_path(base_slug)
            project_path.mkdir(parents=False, exist_ok=False)
        except Exception as exc:
            await self._record_event(
                "project_create_failed",
                request_context,
                audit_event="project_create_failed",
                event_status="failed",
                requested_name=requested_name,
                auto=auto,
                error_message=str(exc),
                level="warning",
            )
            raise

        context.user_data["current_directory"] = project_path
        await self._record_event(
            "project_auto_created" if auto else "project_created",
            request_context,
            audit_event="project_auto_created" if auto else "project_created",
            event_status="created",
            selected_project=project_path.name,
            requested_name=requested_name,
            auto=auto,
        )
        return project_path
