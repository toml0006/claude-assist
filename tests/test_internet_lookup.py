"""Unit tests for internet lookup helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "ai_subscription_assist"
    / "internet_lookup.py"
)
SPEC = importlib.util.spec_from_file_location("ai_subscription_assist_internet_lookup", MODULE_PATH)
assert SPEC and SPEC.loader
internet_lookup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(internet_lookup)

extract_page_text = internet_lookup.extract_page_text
parse_bing_rss = internet_lookup.parse_bing_rss
validate_public_http_url = internet_lookup.validate_public_http_url


def test_validate_public_http_url_accepts_public() -> None:
    assert validate_public_http_url("https://example.com/path") is None


def test_validate_public_http_url_rejects_localhost() -> None:
    error = validate_public_http_url("http://localhost:8123")
    assert error and "not allowed" in error


def test_validate_public_http_url_rejects_private_ip() -> None:
    error = validate_public_http_url("http://192.168.1.10/status")
    assert error and "not public" in error


def test_parse_bing_rss_extracts_items() -> None:
    xml_text = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Result One</title>
      <link>https://example.com/one</link>
      <description>First snippet</description>
      <pubDate>Mon, 16 Feb 2026 00:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Result Two</title>
      <link>https://example.com/two</link>
      <description>Second snippet</description>
      <pubDate>Mon, 16 Feb 2026 00:00:01 GMT</pubDate>
    </item>
  </channel>
</rss>
"""
    items = parse_bing_rss(xml_text, limit=1)
    assert len(items) == 1
    assert items[0]["title"] == "Result One"
    assert items[0]["url"] == "https://example.com/one"


def test_extract_page_text_removes_script_and_title() -> None:
    html = """
<html>
  <head><title>Sample Page</title><script>var x = 1;</script></head>
  <body>
    <h1>Header</h1>
    <p>Hello world</p>
  </body>
</html>
"""
    page = extract_page_text(html, max_chars=1000)
    assert page["title"] == "Sample Page"
    assert "Hello world" in page["text"]
    assert "var x = 1" not in page["text"]

