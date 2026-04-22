from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ...config import Settings
from ...models import CodexLaunchMode, CodexResponse, CodexResultStatus, LocalCodexSession
from ...services.projects import RepoOption


def render_launch_mode_label(launch_mode: CodexLaunchMode) -> str:
    if launch_mode == CodexLaunchMode.FULL_ACCESS:
        return "Полный доступ"
    return "Песочница"


def render_home_text(cwd: Optional[Path], *, auto_created: bool = False) -> str:
    if cwd is None:
        return (
            "Codex Telegram Bot для работы с кодом прямо из чата.\n\n"
            "Проект ещё не выбран.\n"
            "Открой `/menu` или используй `➕ Создать проект`, чтобы начать."
        )
    created_line = f"Автоматически создал первый проект: `{cwd.name}`.\n\n" if auto_created else ""
    return (
        "Codex Telegram Bot для работы с кодом прямо из чата.\n\n"
        f"{created_line}"
        f"Текущий проект: `{cwd.name}`\n\n"
        "Отправь задачу сообщением или открой `/menu`, чтобы сменить проект, режим доступа или начать новую сессию."
    )


def render_start_chat_text(
    cwd: Optional[Path],
    *,
    auto_created: bool = False,
    launch_mode: CodexLaunchMode = CodexLaunchMode.SANDBOX,
) -> str:
    if cwd is None:
        return (
            "Сейчас проект не выбран.\n\n"
            "Сначала создай новую рабочую папку кнопкой `➕ Создать проект`."
        )
    created_line = f"Автоматически создал первый проект: `{cwd.name}`.\n\n" if auto_created else ""
    return (
        f"{created_line}"
        f"Готов к работе в проекте `{cwd.name}`.\n\n"
        f"Режим доступа: `{render_launch_mode_label(launch_mode)}`.\n\n"
        "Отправь задачу сообщением.\n"
        "Если нужно изменить проект или режим, открой `/menu`."
    )


def render_status_text(
    settings: Settings,
    cwd: Optional[Path],
    session: Any,
    verbose_level: int,
    *,
    auto_created: bool = False,
    launch_mode: CodexLaunchMode = CodexLaunchMode.SANDBOX,
) -> str:
    lines = []
    if auto_created and cwd is not None:
        lines.append(f"Автоматически созданный проект: `{cwd.name}`")
    lines.extend(
        [
            f"Проект: `{cwd.name if cwd is not None else 'не выбран'}`",
            f"Путь: `{cwd if cwd is not None else settings.approved_directory.resolve()}`",
            f"Thread ID: `{session.thread_id if session else 'none'}`",
            f"Режим доступа: `{render_launch_mode_label(launch_mode)}`",
            f"Verbose: `{verbose_level}`",
        ]
    )
    if session and session.last_status:
        lines.append(f"Последний статус: `{session.last_status}`")
    if session and session.last_error:
        lines.append(f"Последняя ошибка: `{session.last_error[:160]}`")
    if cwd is None:
        lines.append("Сначала выбери или создай проект.")
    return "\n".join(lines)


def render_session_text(
    *,
    cwd: Path,
    launch_mode: CodexLaunchMode,
    has_session: bool,
    has_active_run: bool,
    auto_created: bool = False,
    notice: str = "",
) -> str:
    lines = []
    if notice:
        lines.append(notice)
        lines.append("")
    if auto_created:
        lines.append(f"Автоматически создал первый проект: `{cwd.name}`.")
        lines.append("")
    lines.extend(
        [
            "Текущая сессия.",
            "",
            f"Проект: `{cwd.name}`",
            f"Режим доступа: `{render_launch_mode_label(launch_mode)}`",
            f"Сессия: `{'текущая' if has_session else 'новая'}`",
        ]
    )
    if has_active_run:
        lines.append("Запуск: `выполняется`")
    return "\n".join(lines)


def render_local_sessions_text(
    *,
    cwd: Path,
    sessions: list[LocalCodexSession],
    current_thread_id: str = "",
    has_active_run: bool = False,
    notice: str = "",
) -> str:
    lines = []
    if notice:
        lines.extend([notice, ""])
    lines.extend(
        [
            "Сессии проекта.",
            "",
            f"Проект: `{cwd.name}`",
            f"Текущая: `{current_thread_id or 'none'}`",
            "",
        ]
    )
    if sessions:
        lines.append("Выбери локальную сессию Codex для продолжения.")
        lines.append(f"Показаны последние `{len(sessions)}`.")
    else:
        lines.append("Для этого проекта локальные сессии Codex не найдены.")
    if has_active_run:
        lines.extend(["", "Выбор сессии недоступен, пока Codex выполняет запрос."])
    return "\n".join(lines)


def render_verbose_text(current_level: int) -> str:
    return (
        f"Текущий verbose level: `{current_level}`\n\n"
        "0: только итог\n"
        "1: итог и token summary\n"
        "2: больше промежуточного прогресса"
    )


def render_repo_picker_text(
    entries: list[RepoOption],
    truncated: bool,
    *,
    auto_created: bool = False,
) -> str:
    text = "Выбери активный проект. Codex будет запускаться в этой директории."
    if auto_created:
        current_created = next((entry.slug for entry in entries if entry.is_current), "")
        if current_created:
            text = f"Создал первый проект `{current_created}`.\n\n" + text
    if truncated:
        text += "\n\nПоказаны первые 20 подпроектов."
    current = next((entry.slug for entry in entries if entry.is_current), "")
    if current:
        text += f"\n\nТекущий проект: `{current}`"
    return text


def render_project_selected_text(selected_dir: Path, base_dir: Path) -> str:
    relative = selected_dir.resolve().relative_to(base_dir.resolve())
    return f"Активный проект переключён на `{relative}`."


def render_project_created_text(project: Path) -> str:
    return f"Создал и выбрал новый проект: `{project.name}`."


def render_no_projects_text() -> str:
    return (
        "В рабочем каталоге пока нет проектов.\n\n"
        "Нажми `➕ Создать проект`, чтобы создать новую подпапку-проект."
    )


def render_final_text(response: CodexResponse) -> str:
    if response.status == CodexResultStatus.SUCCESS:
        return response.final_text or "Готово, но Codex не вернул финальный текст."
    if response.status == CodexResultStatus.INTERRUPTED:
        base = response.final_text or "Запрос остановлен."
        if "(Interrupted by user)" not in base:
            base += "\n\n(Interrupted by user)"
        return base
    if response.status == CodexResultStatus.TIMEOUT:
        return response.final_text or "Превышено время ожидания запроса."
    if response.status == CodexResultStatus.RESUME_FAILED:
        return response.final_text or "Не удалось продолжить прошлую сессию. Попробуй ещё раз."
    if response.status == CodexResultStatus.PROTOCOL_ERROR:
        return response.final_text or "Codex вернул неожиданный ответ."
    if response.status == CodexResultStatus.CLI_ERROR:
        return response.final_text or f"Codex CLI завершился с ошибкой: {response.error_message}"
    return response.final_text or f"Request failed: {response.error_message}"


def build_progress_text(elapsed_seconds: int, last_progress_lines: list[str]) -> str:
    header = f"Working... {elapsed_seconds}s"
    if not last_progress_lines:
        return header
    return header + "\n\n" + "\n".join(last_progress_lines)


def render_launch_mode_editor_text(
    *,
    project_name: str,
    launch_mode: CodexLaunchMode,
    has_active_run: bool,
    notice: str = "",
) -> str:
    lines = []
    if notice:
        lines.extend([notice, ""])
    lines.extend(
        [
            "Настройка режима доступа.",
            "",
            f"Проект: `{project_name}`",
            f"Текущий режим: `{render_launch_mode_label(launch_mode)}`",
            "",
            "Новый режим будет применяться ко всем следующим запросам в этом проекте.",
        ]
    )
    if has_active_run:
        lines.append("Текущий запуск не изменится. Новый режим применится к следующему запросу.")
    return "\n".join(lines)


def render_full_access_warning_text(*, project_name: str) -> str:
    return (
        "Подтверждение полного доступа.\n\n"
        f"Проект: `{project_name}`\n\n"
        "Полный доступ отключает sandbox для следующих запросов в этом проекте.\n"
        "Codex сможет выполнять команды и работать с файлами без ограничений песочницы."
    )
