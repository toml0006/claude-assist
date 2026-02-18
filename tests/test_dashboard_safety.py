"""Unit tests for dashboard structure safety checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "ai_subscription_assist"
    / "dashboard_safety.py"
)
SPEC = importlib.util.spec_from_file_location("ai_subscription_assist_dashboard_safety", MODULE_PATH)
assert SPEC and SPEC.loader
dashboard_safety = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard_safety)

has_explicit_view_change_request = dashboard_safety.has_explicit_view_change_request
has_negated_view_change_request = dashboard_safety.has_negated_view_change_request
validate_view_change_request = dashboard_safety.validate_view_change_request


def test_detects_explicit_add_view_request() -> None:
    assert has_explicit_view_change_request(
        "add_view", "Please create a new view named Patio"
    )


def test_detects_explicit_remove_view_request() -> None:
    assert has_explicit_view_change_request(
        "remove_view", "Delete the old maintenance tab"
    )


def test_detects_negated_add_view_request() -> None:
    assert has_negated_view_change_request(
        "add_view", "You just created a 3rd view, please stop creating new views"
    )


def test_validate_requires_user_request_quote() -> None:
    error = validate_view_change_request(
        action="add_view",
        user_request=None,
        context_text="Create a new dashboard tab",
    )
    assert error and "requires user_request" in error


def test_validate_blocks_when_context_disagrees() -> None:
    error = validate_view_change_request(
        action="add_view",
        user_request="Create a new view called Energy",
        context_text="Please add a card to the existing main view",
    )
    assert error and "does not explicitly request" in error


def test_validate_blocks_when_context_is_negative() -> None:
    error = validate_view_change_request(
        action="add_view",
        user_request="Create a new view called Energy",
        context_text="Stop creating new views",
    )
    assert error and "says not to change" in error


def test_validate_allows_explicit_matching_request() -> None:
    error = validate_view_change_request(
        action="add_view",
        user_request="Create a new view called Energy",
        context_text="Please create a new dashboard view named Energy",
    )
    assert error is None

