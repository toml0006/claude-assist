"""Unit tests for provider error helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "claude_assist"
    / "provider_errors.py"
)
SPEC = importlib.util.spec_from_file_location("claude_assist_provider_errors", MODULE_PATH)
assert SPEC and SPEC.loader
provider_errors = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provider_errors)

format_http_error_message = provider_errors.format_http_error_message
format_rate_limited_message = provider_errors.format_rate_limited_message
extract_provider_error_detail = provider_errors.extract_provider_error_detail
parse_retry_after_seconds = provider_errors.parse_retry_after_seconds


def test_parse_retry_after_seconds_accepts_integer_seconds() -> None:
    assert parse_retry_after_seconds("12") == 12


def test_parse_retry_after_seconds_accepts_http_date() -> None:
    now = datetime(2026, 2, 18, 17, 0, 0, tzinfo=timezone.utc)
    retry_at = now + timedelta(seconds=45)
    header = retry_at.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after_seconds(header, now=now) == 45


def test_parse_retry_after_seconds_rejects_invalid() -> None:
    assert parse_retry_after_seconds("n/a") is None


def test_format_rate_limited_message_without_retry_after() -> None:
    msg = format_rate_limited_message("Gemini")
    assert "rate limited" in msg
    assert "wait a moment" in msg


def test_format_rate_limited_message_with_retry_after() -> None:
    msg = format_rate_limited_message("Gemini", retry_after="30")
    assert "30 seconds" in msg


def test_format_rate_limited_message_includes_provider_detail() -> None:
    msg = format_rate_limited_message(
        "Gemini",
        retry_after="30",
        detail="Quota exceeded for project test-project",
    )
    assert "Provider message" in msg
    assert "Quota exceeded" in msg


def test_format_http_error_message_authentication() -> None:
    msg = format_http_error_message("Gemini", 401)
    assert "authentication failed" in msg


def test_extract_provider_error_detail_from_google_error_payload() -> None:
    payload = """
{
  "error": {
    "code": 429,
    "message": "Quota exceeded for quota metric.",
    "status": "RESOURCE_EXHAUSTED"
  }
}
"""
    detail = extract_provider_error_detail(payload, content_type="application/json")
    assert detail == "Quota exceeded for quota metric."


def test_extract_provider_error_detail_falls_back_to_text() -> None:
    detail = extract_provider_error_detail("Too many requests for this subscription")
    assert detail == "Too many requests for this subscription"
