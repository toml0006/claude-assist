"""Helpers to format provider transport errors for user-facing responses."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import re

MAX_PROVIDER_DETAIL_LEN = 280


def _normalize_provider_detail(value: str | None) -> str | None:
    """Normalize provider detail for user display."""
    if not value:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        return None
    if len(text) > MAX_PROVIDER_DETAIL_LEN:
        return f"{text[: MAX_PROVIDER_DETAIL_LEN - 3]}..."
    return text


def _extract_dict_message(payload: dict[str, object]) -> str | None:
    """Extract a concise error message from a JSON object."""
    error_obj = payload.get("error")
    if isinstance(error_obj, dict):
        payload = error_obj

    for key in ("message", "error_description", "description", "detail"):
        value = payload.get(key)
        if isinstance(value, str):
            return _normalize_provider_detail(value)

    details = payload.get("details")
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            for key in ("message", "detail", "description"):
                value = detail.get(key)
                if isinstance(value, str):
                    return _normalize_provider_detail(value)

    return None


def extract_provider_error_detail(
    payload: bytes | str | None,
    *,
    content_type: str | None = None,
) -> str | None:
    """Extract a concise provider detail from HTTP error payload."""
    if payload is None:
        return None

    text = (
        payload.decode("utf-8", errors="replace")
        if isinstance(payload, bytes)
        else str(payload)
    )
    text = text.strip()
    if not text:
        return None

    lower_content_type = (content_type or "").lower()
    parse_as_json = "json" in lower_content_type or text.startswith("{")
    if parse_as_json:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            message = _extract_dict_message(parsed)
            if message:
                return message

    if text.startswith("<") and "html" in text[:120].lower():
        return None

    return _normalize_provider_detail(text)


def parse_retry_after_seconds(
    retry_after: str | None,
    *,
    now: datetime | None = None,
) -> int | None:
    """Parse Retry-After header value into seconds."""
    if not retry_after:
        return None

    value = retry_after.strip()
    if not value:
        return None

    try:
        seconds = int(float(value))
    except ValueError:
        seconds = None

    if seconds is not None:
        return max(seconds, 0)

    try:
        retry_time = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None

    if retry_time.tzinfo is None:
        retry_time = retry_time.replace(tzinfo=timezone.utc)

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)

    return max(int((retry_time - current).total_seconds()), 0)


def format_rate_limited_message(
    provider_name: str,
    *,
    retry_after: str | None = None,
    detail: str | None = None,
) -> str:
    """Build a user-facing rate limit message."""
    wait_seconds = parse_retry_after_seconds(retry_after)
    if wait_seconds is None:
        base_message = (
            f"{provider_name} is rate limited right now. "
            "Please wait a moment and try again."
        )
    elif wait_seconds <= 1:
        base_message = (
            f"{provider_name} is rate limited right now. "
            "Please retry in a few seconds."
        )
    else:
        base_message = (
            f"{provider_name} is rate limited right now. "
            f"Please wait about {wait_seconds} seconds and try again."
        )

    detail_text = _normalize_provider_detail(detail)
    if detail_text:
        return f"{base_message} Provider message: {detail_text}"
    return base_message


def format_http_error_message(provider_name: str, status_code: int) -> str:
    """Build a user-facing generic HTTP status error message."""
    if status_code in (401, 403):
        return (
            f"{provider_name} authentication failed. "
            "Please re-authenticate this service and try again."
        )
    return (
        f"Sorry, I had a problem talking to {provider_name} "
        f"(HTTP {status_code}). Please try again."
    )
