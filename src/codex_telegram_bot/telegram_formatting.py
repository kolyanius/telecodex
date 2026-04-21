from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urlparse

import markdown


TOKEN_RE = re.compile(r"(<[^>]+?>|&(?:#\d+|#x[0-9a-fA-F]+|\w+);)")
PRE_RE = re.compile(
    r"^<pre>(?P<code_open><code(?: class=\"language-[^\"]+\")?>)?(?P<content>.*?)(?P<code_close></code>)?</pre>$",
    re.DOTALL,
)
LIST_OR_QUOTE_RE = re.compile(r"^(?:[-*+] |\d+\. |> )")


def render_markdown_to_telegram_html(markdown_text: str) -> str:
    normalized_markdown = _normalize_markdown(markdown_text)
    rendered = markdown.markdown(
        normalized_markdown,
        extensions=["fenced_code", "sane_lists"],
        output_format="html",
    )
    sanitizer = _TelegramHTMLSanitizer()
    sanitizer.feed(rendered)
    sanitizer.close()
    return sanitizer.get_html()


def chunk_telegram_html(html_text: str, max_length: int = 3800) -> list[str]:
    if telegram_html_visible_length(html_text) <= max_length:
        return [html_text]

    blocks = _split_top_level_blocks(html_text)
    chunks: list[str] = []
    current = ""
    for block in blocks:
        for piece in _split_oversized_block(block, max_length):
            if not current:
                current = piece
                continue
            candidate = f"{current}\n\n{piece}"
            if telegram_html_visible_length(candidate) <= max_length:
                current = candidate
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks or [html_text[:max_length]]


def plain_text_from_markdown(markdown_text: str) -> str:
    html_text = render_markdown_to_telegram_html(markdown_text)
    return _strip_html_tags(html_text)


def telegram_html_visible_length(html_text: str) -> int:
    return len(_strip_html_tags(html_text))


def _normalize_markdown(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    normalized: list[str] = []
    previous = ""
    for line in lines:
        stripped = line.lstrip()
        if (
            normalized
            and line
            and previous
            and not previous.isspace()
            and LIST_OR_QUOTE_RE.match(stripped)
            and not LIST_OR_QUOTE_RE.match(previous.lstrip())
            and not previous.rstrip().endswith(("```", ":", "</code>"))
        ):
            normalized.append("")
        normalized.append(line)
        previous = line
    return "\n".join(normalized)


def _split_top_level_blocks(html_text: str) -> list[str]:
    parts = TOKEN_RE.split(html_text)
    blocks: list[str] = []
    current: list[str] = []
    stack: list[tuple[str, str]] = []

    for part in parts:
        if not part:
            continue
        if part.startswith("<") and part.endswith(">"):
            current.append(part)
            _update_stack(stack, part)
            continue

        if stack:
            current.append(part)
            continue

        remaining = part
        while True:
            idx = remaining.find("\n\n")
            if idx == -1:
                if remaining:
                    current.append(remaining)
                break
            prefix = remaining[:idx]
            if prefix:
                current.append(prefix)
            block = "".join(current).strip()
            if block:
                blocks.append(block)
            current = []
            remaining = remaining[idx + 2 :]

    tail = "".join(current).strip()
    if tail:
        blocks.append(tail)
    return blocks or ([html_text] if html_text else [])


def _split_oversized_block(block: str, max_length: int) -> list[str]:
    if telegram_html_visible_length(block) <= max_length:
        return [block]

    pre_match = PRE_RE.match(block)
    if pre_match:
        code_open = pre_match.group("code_open") or ""
        code_close = pre_match.group("code_close") or ""
        prefix = f"<pre>{code_open}"
        suffix = f"{code_close}</pre>"
        return _split_wrapped_text_block(
            prefix=prefix,
            content=pre_match.group("content"),
            suffix=suffix,
            max_length=max_length,
        )

    if block.startswith("<blockquote>") and block.endswith("</blockquote>"):
        prefix = "<blockquote>"
        suffix = "</blockquote>"
        content = block[len(prefix) : -len(suffix)]
        return _split_wrapped_text_block(
            prefix=prefix,
            content=content,
            suffix=suffix,
            max_length=max_length,
        )

    return _split_html_fragment(block, max_length)


def _split_wrapped_text_block(*, prefix: str, content: str, suffix: str, max_length: int) -> list[str]:
    budget = max_length
    if budget <= 0:
        return [prefix + suffix]

    chunks: list[str] = []
    current = ""
    for line in content.splitlines(keepends=True):
        if len(html.unescape(current + line)) <= budget:
            current += line
            continue
        if current:
            chunks.append(f"{prefix}{current}{suffix}")
            current = ""
        while len(html.unescape(line)) > budget:
            split_at = _find_split_position_by_visible_length(line, budget)
            chunks.append(f"{prefix}{line[:split_at]}{suffix}")
            line = line[split_at:]
        current = line
    if current:
        chunks.append(f"{prefix}{current}{suffix}")
    if chunks:
        return chunks
    split_at = _find_split_position_by_visible_length(content, budget)
    return [f"{prefix}{content[:split_at]}{suffix}"]


def _split_html_fragment(fragment: str, max_length: int) -> list[str]:
    tokens = [token for token in TOKEN_RE.split(fragment) if token]
    chunks: list[str] = []
    current: list[str] = []
    stack: list[tuple[str, str]] = []

    def close_tags() -> list[str]:
        return [f"</{tag}>" for tag, _ in reversed(stack)]

    def reopen_tags() -> list[str]:
        return [token for _, token in stack]

    def flush() -> None:
        if not current:
            return
        chunks.append("".join(current + close_tags()))

    for token in tokens:
        pieces = [token]
        if not token.startswith("<") and not token.startswith("&"):
            pieces = _split_text_token(token, max_length)

        for piece in pieces:
            if telegram_html_visible_length("".join(current + [piece] + close_tags())) > max_length and current:
                flush()
                current = reopen_tags()
            current.append(piece)
            if piece.startswith("<") and piece.endswith(">"):
                _update_stack(stack, piece)

    if current:
        flush()
    return chunks or [fragment]


def _split_text_token(token: str, max_length: int) -> list[str]:
    if len(html.unescape(token)) <= max_length:
        return [token]
    pieces: list[str] = []
    remaining = token
    while len(html.unescape(remaining)) > max_length:
        rough_split = _find_split_position_by_visible_length(remaining, max_length)
        split_at = remaining.rfind(" ", 0, rough_split)
        if split_at <= 0:
            split_at = rough_split
        pieces.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _find_split_position_by_visible_length(text: str, max_visible_length: int) -> int:
    visible = 0
    index = 0
    while index < len(text):
        if text[index] == "&":
            match = re.match(r"&(?:#\d+|#x[0-9a-fA-F]+|\w+);", text[index:])
            if match:
                entity = match.group(0)
                entity_visible = len(html.unescape(entity))
                if visible + entity_visible > max_visible_length:
                    break
                visible += entity_visible
                index += len(entity)
                continue
        if visible + 1 > max_visible_length:
            break
        visible += 1
        index += 1
    return max(index, 1)


def _update_stack(stack: list[tuple[str, str]], token: str) -> None:
    match = re.match(r"<\s*(/)?\s*([a-zA-Z0-9-]+)", token)
    if not match:
        return
    closing = bool(match.group(1))
    tag = match.group(2).lower()
    if token.endswith("/>"):
        return
    if closing:
        for index in range(len(stack) - 1, -1, -1):
            if stack[index][0] == tag:
                del stack[index]
                break
        return
    if tag in {"b", "i", "u", "s", "code", "pre", "blockquote", "a"}:
        stack.append((tag, token))


def _is_local_href(href: str) -> bool:
    if not href:
        return True
    if href.startswith(("/", "./", "../", "file://", "app://")):
        return True
    parsed = urlparse(href)
    if parsed.scheme in {"http", "https", "tg", "mailto"}:
        return False
    if re.match(r"^[A-Za-z]:[\\\\/]", href):
        return True
    return not bool(parsed.scheme)


def _normalize_visible_link_text(label: str) -> str:
    normalized = label.strip() or "link"
    normalized = normalized.replace("](", " ")
    return normalized


def _strip_html_tags(html_text: str) -> str:
    text = re.sub(r"<[^>]+>", "", html_text)
    return html.unescape(text)


class _TelegramHTMLSanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.list_stack: list[dict[str, int | str]] = []
        self.link_mode_stack: list[str] = []
        self.in_pre = False
        self.blockquote_depth = 0

    def get_html(self) -> str:
        html_text = "".join(self.parts).strip()
        html_text = re.sub(r"\n{3,}", "\n\n", html_text)
        return html_text

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attrs_dict = {name: value or "" for name, value in attrs}
        tag = tag.lower()

        if tag in {"p", "div"}:
            if self.blockquote_depth == 0:
                self._ensure_block_gap()
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._ensure_block_gap()
            self.parts.append("<b>")
            return
        if tag in {"strong", "b"}:
            self.parts.append("<b>")
            return
        if tag in {"em", "i"}:
            self.parts.append("<i>")
            return
        if tag in {"u", "ins"}:
            self.parts.append("<u>")
            return
        if tag in {"s", "strike", "del"}:
            self.parts.append("<s>")
            return
        if tag == "br":
            self.parts.append("\n")
            return
        if tag == "pre":
            self._ensure_block_gap()
            self.parts.append("<pre>")
            self.in_pre = True
            return
        if tag == "code":
            if self.in_pre:
                language = attrs_dict.get("class", "").strip()
                if language.startswith("language-"):
                    self.parts.append(f'<code class="{html.escape(language, quote=True)}">')
                else:
                    self.parts.append("<code>")
            else:
                self.parts.append("<code>")
            return
        if tag == "blockquote":
            self._ensure_block_gap()
            self.parts.append("<blockquote>")
            self.blockquote_depth += 1
            return
        if tag == "ul":
            self._ensure_block_gap()
            self.list_stack.append({"type": "ul", "index": 0})
            return
        if tag == "ol":
            self._ensure_block_gap()
            self.list_stack.append({"type": "ol", "index": 0})
            return
        if tag == "li":
            if self.parts and not self.parts[-1].endswith("\n"):
                self.parts.append("\n")
            if not self.list_stack:
                self.parts.append("• ")
                return
            current = self.list_stack[-1]
            if current["type"] == "ol":
                current["index"] = int(current["index"]) + 1
                self.parts.append(f"{current['index']}. ")
            else:
                self.parts.append("• ")
            return
        if tag == "a":
            href = attrs_dict.get("href", "").strip()
            if _is_local_href(href):
                self.link_mode_stack.append("code")
                self.parts.append("<code>")
            elif href:
                safe_href = html.escape(href, quote=True)
                self.link_mode_stack.append("a")
                self.parts.append(f'<a href="{safe_href}">')
            else:
                self.link_mode_stack.append("text")
            return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"p", "div"}:
            if self.blockquote_depth == 0:
                self._ensure_block_gap()
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("</b>")
            self._ensure_block_gap()
            return
        if tag in {"strong", "b"}:
            self.parts.append("</b>")
            return
        if tag in {"em", "i"}:
            self.parts.append("</i>")
            return
        if tag in {"u", "ins"}:
            self.parts.append("</u>")
            return
        if tag in {"s", "strike", "del"}:
            self.parts.append("</s>")
            return
        if tag == "pre":
            self.parts.append("</pre>")
            self.in_pre = False
            self._ensure_block_gap()
            return
        if tag == "code":
            self.parts.append("</code>")
            return
        if tag == "blockquote":
            self.parts.append("</blockquote>")
            self.blockquote_depth = max(0, self.blockquote_depth - 1)
            self._ensure_block_gap()
            return
        if tag in {"ul", "ol"}:
            if self.list_stack:
                self.list_stack.pop()
            self._ensure_block_gap()
            return
        if tag == "a":
            mode = self.link_mode_stack.pop() if self.link_mode_stack else "text"
            if mode == "code":
                self.parts.append("</code>")
            elif mode == "a":
                self.parts.append("</a>")
            return

    def handle_data(self, data: str) -> None:
        if not data:
            return
        if not self.in_pre and data.isspace():
            return
        escaped = html.escape(_normalize_visible_link_text(data) if self._in_local_link() else data)
        self.parts.append(escaped)

    def handle_entityref(self, name: str) -> None:
        self.parts.append(html.escape(html.unescape(f"&{name};")))

    def handle_charref(self, name: str) -> None:
        self.parts.append(html.escape(html.unescape(f"&#{name};")))

    def _ensure_block_gap(self) -> None:
        current = "".join(self.parts).rstrip()
        self.parts = [current] if current else []
        if self.parts:
            self.parts.append("\n\n")

    def _in_local_link(self) -> bool:
        return bool(self.link_mode_stack and self.link_mode_stack[-1] == "code")
