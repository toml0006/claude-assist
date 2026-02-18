"""Unit tests for tool enablement policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "ai_subscription_assist"
    / "tool_policy.py"
)
SPEC = importlib.util.spec_from_file_location("ai_subscription_assist_tool_policy", MODULE_PATH)
assert SPEC and SPEC.loader
tool_policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool_policy)

PRIVILEGED_TOOL_NAMES = tool_policy.PRIVILEGED_TOOL_NAMES
default_enabled_tool_names = tool_policy.default_enabled_tool_names
normalize_enabled_tool_names = tool_policy.normalize_enabled_tool_names


ALL_TOOLS = [
    "set_model",
    "get_history",
    "call_service",
    "toggle_automation",
    "add_automation",
]


def test_default_tools_non_yolo_excludes_privileged() -> None:
    defaults = default_enabled_tool_names(ALL_TOOLS, yolo_mode=False)
    assert defaults == ["set_model", "get_history"]
    assert not (set(defaults) & PRIVILEGED_TOOL_NAMES)


def test_default_tools_yolo_includes_all() -> None:
    defaults = default_enabled_tool_names(ALL_TOOLS, yolo_mode=True)
    assert defaults == ALL_TOOLS


def test_normalize_filters_unknown_and_dedupes() -> None:
    normalized = normalize_enabled_tool_names(
        ["set_model", "bogus", "set_model", "get_history"],
        ALL_TOOLS,
        yolo_mode=True,
    )
    assert normalized == ["set_model", "get_history"]


def test_normalize_non_yolo_strips_privileged() -> None:
    normalized = normalize_enabled_tool_names(
        ["set_model", "toggle_automation", "add_automation"],
        ALL_TOOLS,
        yolo_mode=False,
    )
    assert normalized == ["set_model"]


def test_normalize_none_uses_default_policy() -> None:
    normalized = normalize_enabled_tool_names(None, ALL_TOOLS, yolo_mode=False)
    assert normalized == ["set_model", "get_history"]
