"""Policy helpers for custom tool enablement and yolo mode."""

from __future__ import annotations

from collections.abc import Iterable

# Tools that are intentionally gated behind explicit "yolo mode".
PRIVILEGED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "toggle_automation",
        "add_automation",
        "modify_dashboard",
        "call_service",
        "send_notification",
        "get_error_log",
        "manage_list",
    }
)


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    """Return unique values while preserving order."""
    return list(dict.fromkeys(values))


def default_enabled_tool_names(
    all_tool_names: Iterable[str], yolo_mode: bool
) -> list[str]:
    """Return the default tool allowlist for the given mode."""
    all_names = _dedupe_preserve_order(all_tool_names)
    if yolo_mode:
        return all_names
    return [name for name in all_names if name not in PRIVILEGED_TOOL_NAMES]


def normalize_enabled_tool_names(
    enabled: Iterable[str] | str | None,
    all_tool_names: Iterable[str],
    yolo_mode: bool,
) -> list[str]:
    """Normalize and policy-filter tool names."""
    all_names = _dedupe_preserve_order(all_tool_names)
    valid = set(all_names)

    if enabled is None:
        requested = default_enabled_tool_names(all_names, yolo_mode)
    elif isinstance(enabled, str):
        requested = [enabled]
    else:
        requested = list(enabled)

    normalized = _dedupe_preserve_order(name for name in requested if name in valid)
    if yolo_mode:
        return normalized

    return [name for name in normalized if name not in PRIVILEGED_TOOL_NAMES]
