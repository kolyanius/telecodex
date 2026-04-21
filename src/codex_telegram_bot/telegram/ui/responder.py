from __future__ import annotations

from typing import Any, Optional

from telegram import InlineKeyboardMarkup, Update
from telegram.constants import ParseMode

from ...telegram_formatting import (
    chunk_telegram_html,
    plain_text_from_markdown,
    render_markdown_to_telegram_html,
)


class TelegramResponder:
    def __init__(self, logger: Any):
        self.logger = logger

    @staticmethod
    def chunk_text(text: str, size: int) -> list[str]:
        if len(text) <= size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            start = end
        return chunks

    async def edit_callback_message(
        self,
        update: Update,
        text: str,
        *,
        reply_markup: Optional[InlineKeyboardMarkup] = None,
        parse_mode: Optional[str] = None,
    ) -> None:
        query = update.callback_query
        if hasattr(query, "edit_message_text"):
            try:
                await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
            except Exception as exc:
                if "message is not modified" in str(exc).lower():
                    self.logger.debug("telegram_callback_edit_noop")
                    return
                raise
            return
        if update.effective_message:
            await update.effective_message.reply_text(
                text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
            )

    async def send_final_response(
        self,
        *,
        update: Update,
        markdown_text: str,
        followup_keyboard: Optional[InlineKeyboardMarkup] = None,
    ) -> None:
        try:
            html_text = render_markdown_to_telegram_html(markdown_text)
            chunks = chunk_telegram_html(html_text, 3800)
            for index, chunk in enumerate(chunks):
                await update.effective_message.reply_text(
                    chunk,
                    reply_markup=followup_keyboard if index == len(chunks) - 1 else None,
                    parse_mode=ParseMode.HTML,
                )
        except Exception as exc:
            self.logger.warning("telegram_html_render_failed", error=str(exc))
            plain_text = plain_text_from_markdown(markdown_text)
            chunks = self.chunk_text(plain_text, 3800)
            for index, chunk in enumerate(chunks):
                await update.effective_message.reply_text(
                    chunk,
                    reply_markup=followup_keyboard if index == len(chunks) - 1 else None,
                )
