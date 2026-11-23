"""R03: purely presentational safeguards for Gradio HTML outputs.

Does not import or modify model artifacts, preprocessing, scoring, or thresholds.
"""
from __future__ import annotations

from functools import wraps
from html import escape
from html.parser import HTMLParser
import re
from typing import Any, Callable

# Permit the elements required by the compact analysis/result cards.
_ALLOWED_TAGS = {
    "div", "span", "p", "section", "article", "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
    "small", "strong", "em", "b", "i", "br", "hr", "ul", "ol", "li", "code", "pre", "sup", "sub",
    "table", "thead", "tbody", "tr", "th", "td", "label", "meter", "progress"
}
_ALLOWED_ATTRS = {"class", "id", "title", "role", "style", "dir", "lang", "tabindex", "aria-label", "aria-hidden"}
_SAFE_CSS_PROPERTIES = {
    "display", "width", "max-width", "min-width", "height", "max-height", "min-height", "background", "background-color", "background-image",
    "color", "border", "border-left", "border-right", "border-top", "border-bottom", "border-color", "border-radius",
    "padding", "padding-left", "padding-right", "padding-top", "padding-bottom", "margin", "margin-left", "margin-right",
    "margin-top", "margin-bottom", "font-size", "font-weight", "font-family", "line-height", "letter-spacing",
    "text-align", "white-space", "overflow", "overflow-wrap", "word-break", "flex", "flex-direction", "flex-wrap",
    "align-items", "justify-content", "gap", "grid-template-columns", "grid-column", "opacity", "vertical-align",
    "position", "top", "left", "right", "bottom", "box-shadow", "text-overflow",
    "flex-grow", "flex-shrink", "flex-basis", "justify-self", "align-self", "box-sizing", "border-width",
    "border-style", "outline", "outline-offset", "font-variant-numeric", "transform", "transition", "z-index"
}
_UNSAFE_CSS = re.compile(r"url\s*\(|expression\s*\(|@import|javascript\s*:|vbscript\s*:|data\s*:|[<>\\]", re.I)
_LONG_WORD = re.compile(r"\S{80,}")


def shorten_visible_token(text: str, max_chars: int = 32) -> str:
    """Display-only truncation; never feed its result into model inference."""
    value = str(text)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "…"


def safe_token_label(token: str, max_chars: int = 32) -> str:
    """HTML-escaped, bounded display label for untrusted SMS tokens."""
    return escape(shorten_visible_token(token, max_chars=max_chars), quote=True)


def _sanitize_style(style: str) -> str:
    clean = []
    for declaration in style.split(";"):
        if ":" not in declaration:
            continue
        prop, value = declaration.split(":", 1)
        prop = prop.strip().lower()
        value = value.strip()
        if not prop or (prop not in _SAFE_CSS_PROPERTIES and not re.fullmatch(r"--[a-z0-9_-]+", prop)) or _UNSAFE_CSS.search(value):
            continue
        # Keep CSS simple: no control characters and no unexpectedly large declarations.
        if len(value) > 250 or any(ord(c) < 32 and c not in "\t" for c in value):
            continue
        clean.append(prop + ": " + value)
    return "; ".join(clean)


class _SafeHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.out: list[str] = []
        self._blocked_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._blocked_depth:
            if tag in {"script", "style", "iframe", "object", "embed", "svg", "math"}:
                self._blocked_depth += 1
            return
        if tag in {"script", "style", "iframe", "object", "embed", "svg", "math"}:
            self._blocked_depth = 1
            return
        if tag not in _ALLOWED_TAGS:
            return
        safe_attrs = []
        for name, value in attrs:
            name = name.lower()
            if value is None or name.startswith("on"):
                continue
            if name not in _ALLOWED_ATTRS and not name.startswith("aria-"):
                continue
            if name == "style":
                value = _sanitize_style(value)
                if not value:
                    continue
            elif name in {"id", "class", "role", "dir", "lang", "tabindex"}:
                if len(value) > 512 or re.search(r"[<>\x00-\x1f]", value):
                    continue
            safe_attrs.append(" " + name + '="' + escape(value, quote=True) + '"')
        self.out.append("<" + tag + "".join(safe_attrs) + ">")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() in _ALLOWED_TAGS and tag.lower() not in {"br", "hr"}:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._blocked_depth:
            if tag in {"script", "style", "iframe", "object", "embed", "svg", "math"}:
                self._blocked_depth -= 1
            return
        if tag in _ALLOWED_TAGS and tag not in {"br", "hr"}:
            self.out.append("</" + tag + ">")

    def handle_data(self, data: str) -> None:
        if self._blocked_depth:
            return
        # R03 C05: prevent a single 2,001-character OOV token from creating a 100-line chart.
        compact = _LONG_WORD.sub(lambda m: shorten_visible_token(m.group(0)), data)
        self.out.append(escape(compact, quote=False))

    def handle_entityref(self, name: str) -> None:
        if not self._blocked_depth:
            self.out.append("&" + name + ";")

    def handle_charref(self, name: str) -> None:
        if not self._blocked_depth:
            self.out.append("&#" + name + ";")

    def handle_comment(self, data: str) -> None:
        # Remove comments from generated results; they are not needed by users.
        pass


def sanitize_generated_html(value: str) -> str:
    """Remove executable HTML and abbreviate excessively long text nodes.

    Trust neither user SMS nor user-derived token labels in generated HTML.
    This is a defense-in-depth output sanitizer; source-level escaping remains preferred.
    """
    parser = _SafeHTML()
    parser.feed(str(value))
    parser.close()
    return "".join(parser.out)


def sanitize_callback_result(result: Any, html_positions: tuple[int, ...]) -> Any:
    if isinstance(result, str):
        return sanitize_generated_html(result) if 0 in html_positions else result
    if isinstance(result, tuple):
        return tuple(sanitize_generated_html(x) if i in html_positions and isinstance(x, str) else x
                     for i, x in enumerate(result))
    if isinstance(result, list):
        return [sanitize_generated_html(x) if i in html_positions and isinstance(x, str) else x
                for i, x in enumerate(result)]
    # Other output shapes are not silently modified.
    return result


def guard_html(fn: Callable, html_positions: tuple[int, ...]) -> Callable:
    @wraps(fn)
    def _safe(*args, **kwargs):
        value = fn(*args, **kwargs)
        return sanitize_callback_result(value, html_positions)
    return _safe


def invalidated_html() -> str:
    return (
        '<section class="r03-pending" role="status" aria-live="polite">'
        '<div class="r03-pending-eyebrow">Perubahan belum dianalisis</div>'
        '<h3>Hasil perlu diperbarui</h3>'
        '<p>Isi pesan atau pengaturan berubah. Klik <strong>Analisis pesan</strong> '
        'untuk menampilkan prediksi terbaru.</p>'
        '</section>'
    )
