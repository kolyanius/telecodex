from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ...models import CodexLaunchMode
from ...services.projects import RepoOption


def build_session_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📁 Проект", callback_data="nav:repo"),
                InlineKeyboardButton("⚙️ Режим", callback_data="mode:show"),
            ],
            [InlineKeyboardButton("🆕 Новая сессия", callback_data="action:new")],
        ]
    )


def build_navigation_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("⬅️ В меню", callback_data="nav:menu")]]
    )


def build_no_project_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("➕ Создать проект", callback_data="action:create_project")],
            [InlineKeyboardButton("📁 Проекты", callback_data="nav:repo")],
        ]
    )


def build_verbose_keyboard(current_level: int) -> InlineKeyboardMarkup:
    buttons = []
    for level in (0, 1, 2):
        label = f"• Verbose {level}" if level == current_level else f"Verbose {level}"
        buttons.append(InlineKeyboardButton(label, callback_data=f"verbose:set:{level}"))
    return InlineKeyboardMarkup(
        [
            buttons,
            [InlineKeyboardButton("⬅️ В меню", callback_data="nav:menu")],
        ]
    )


def build_repo_keyboard(entries: list[RepoOption]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(entry.label, callback_data=f"repo:select:{entry.slug}")]
        for entry in entries
    ]
    rows.append([InlineKeyboardButton("➕ Создать проект", callback_data="action:create_project")])
    rows.append([InlineKeyboardButton("⬅️ В меню", callback_data="nav:menu")])
    return InlineKeyboardMarkup(rows)


def build_stop_keyboard(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("⏹ Остановить", callback_data=f"action:stop:{user_id}")]]
    )


def build_mode_editor_keyboard(
    launch_mode: CodexLaunchMode,
    *,
    full_access_confirmed: bool,
    back_callback: str,
) -> InlineKeyboardMarkup:
    sandbox_label = "• Песочница" if launch_mode == CodexLaunchMode.SANDBOX else "Песочница"
    if launch_mode == CodexLaunchMode.FULL_ACCESS and full_access_confirmed:
        full_access_label = "• Полный доступ"
    elif launch_mode == CodexLaunchMode.FULL_ACCESS:
        full_access_label = "Полный доступ (подтвердить)"
    else:
        full_access_label = "Полный доступ"
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(sandbox_label, callback_data="mode:set:sandbox"),
                InlineKeyboardButton(full_access_label, callback_data="mode:confirm_full"),
            ],
            [InlineKeyboardButton("⬅️ Назад", callback_data=back_callback)],
        ]
    )


def build_full_access_warning_keyboard(back_callback: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("⚠️ Подтвердить полный доступ", callback_data="mode:set:full")],
            [InlineKeyboardButton("🔒 Оставить песочницу", callback_data="mode:set:sandbox")],
            [InlineKeyboardButton("⬅️ Назад", callback_data=back_callback)],
        ]
    )
