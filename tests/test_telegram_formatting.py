from __future__ import annotations

from codex_telegram_bot.telegram_formatting import (
    chunk_telegram_html,
    plain_text_from_markdown,
    render_markdown_to_telegram_html,
    telegram_html_visible_length,
)


def test_render_markdown_to_telegram_html_formats_common_blocks() -> None:
    rendered = render_markdown_to_telegram_html(
        "**Слабые стороны**\n\n- item one\n- item two\n\n`code`\n\n> quote"
    )

    assert "<b>Слабые стороны</b>" in rendered
    assert "• item one" in rendered
    assert "• item two" in rendered
    assert "<code>code</code>" in rendered
    assert "<blockquote>quote</blockquote>" in rendered


def test_render_markdown_to_telegram_html_handles_links() -> None:
    rendered = render_markdown_to_telegram_html(
        "[docs](https://example.com) and [bot.py#L57](/abs/path/bot.py#L57)"
    )

    assert '<a href="https://example.com">docs</a>' in rendered
    assert "<code>bot.py#L57</code>" in rendered
    assert "/abs/path/bot.py#L57" not in rendered


def test_chunk_telegram_html_preserves_pre_blocks() -> None:
    html = "<pre><code>line1\nline2\nline3\nline4\nline5\nline6</code></pre>"
    chunks = chunk_telegram_html(html, max_length=30)

    assert len(chunks) > 1
    assert all(chunk.startswith("<pre><code>") for chunk in chunks)
    assert all(chunk.endswith("</code></pre>") for chunk in chunks)


def test_plain_text_from_markdown_strips_tags() -> None:
    plain = plain_text_from_markdown("**Bold** and [link](https://example.com)")

    assert plain == "Bold and link"


def test_chunk_telegram_html_uses_visible_length_not_raw_html_length() -> None:
    html = "<b>" + ("x" * 20) + "</b>" + "\n\n" + "<code>" + ("y" * 20) + "</code>"
    chunks = chunk_telegram_html(html, max_length=25)

    assert len(chunks) == 2
    assert all(telegram_html_visible_length(chunk) <= 25 for chunk in chunks)


def test_telegram_html_visible_length_ignores_tags_and_decodes_entities() -> None:
    html = "<b>abc</b> &amp; <code>x</code>"

    assert telegram_html_visible_length(html) == len("abc & x")
