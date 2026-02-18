"""Helpers for read-only internet lookup/search."""

from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import ipaddress
import re
from typing import Any
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

_BLOCKED_HOSTS = {
    "localhost",
    "homeassistant",
}
_BLOCKED_SUFFIXES = (
    ".local",
    ".localdomain",
    ".lan",
    ".internal",
    ".home.arpa",
)


def collapse_whitespace(value: str) -> str:
    """Collapse repeated whitespace into single spaces."""
    return " ".join(value.split())


def validate_public_http_url(url: str) -> str | None:
    """Return an error string when URL is not a safe public HTTP(S) URL."""
    try:
        parsed = urlparse(url)
    except Exception:
        return "Invalid URL"

    if parsed.scheme not in {"http", "https"}:
        return "Only http/https URLs are allowed"

    host = (parsed.hostname or "").strip().lower()
    if not host:
        return "URL host is required"

    if host in _BLOCKED_HOSTS or any(host.endswith(suffix) for suffix in _BLOCKED_SUFFIXES):
        return f"Host '{host}' is not allowed"

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return f"Host IP '{host}' is not public"
    return None


def parse_bing_rss(xml_text: str, limit: int) -> list[dict[str, Any]]:
    """Parse Bing RSS search results into compact dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    channel = root.find("channel")
    if channel is None:
        return []

    out: list[dict[str, Any]] = []
    for item in channel.findall("item"):
        title = collapse_whitespace(unescape(item.findtext("title", "")))
        url = collapse_whitespace(item.findtext("link", ""))
        snippet_raw = item.findtext("description", "")
        snippet = collapse_whitespace(unescape(re.sub(r"<[^>]+>", " ", snippet_raw)))
        published = collapse_whitespace(item.findtext("pubDate", ""))
        if not url:
            continue
        out.append(
            {
                "title": title or url,
                "url": url,
                "snippet": snippet,
                "published": published,
            }
        )
        if len(out) >= limit:
            break
    return out


class _HTMLTextExtractor(HTMLParser):
    """Extract visible text and title from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._chunks: list[str] = []
        self._title_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self._title_chunks.append(data)
        self._chunks.append(data)

    @property
    def title(self) -> str:
        return collapse_whitespace("".join(self._title_chunks))

    @property
    def text(self) -> str:
        return collapse_whitespace(" ".join(self._chunks))


def extract_page_text(html_text: str, max_chars: int) -> dict[str, Any]:
    """Extract readable text/title from HTML content."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html_text)
    except Exception:
        # Keep best-effort extraction robust for malformed HTML.
        pass
    text = parser.text
    truncated = len(text) > max_chars
    return {
        "title": parser.title,
        "text": text[:max_chars],
        "truncated": truncated,
        "total_chars": len(text),
    }
