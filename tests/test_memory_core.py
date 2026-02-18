"""Unit tests for pure memory helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "ai_subscription_assist"
    / "memory_core.py"
)
SPEC = importlib.util.spec_from_file_location("ai_subscription_assist_memory_core", MODULE_PATH)
assert SPEC and SPEC.loader
memory_core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory_core)

extract_heuristic_memory = memory_core.extract_heuristic_memory
format_memory_prompt = memory_core.format_memory_prompt
is_duplicate_memory = memory_core.is_duplicate_memory
parse_slash_command = memory_core.parse_slash_command
prune_memory_items = memory_core.prune_memory_items
rank_memory_items = memory_core.rank_memory_items
utcnow_iso = memory_core.utcnow_iso


def test_parse_slash_command_memory_add() -> None:
    parsed = parse_slash_command("/memory add --shared remember that we eat at 6pm")
    assert parsed == {
        "kind": "memory_add",
        "text": "remember that we eat at 6pm",
        "shared": True,
    }


def test_parse_slash_command_aliases() -> None:
    assert parse_slash_command("/new") == {"kind": "reset_context"}
    assert parse_slash_command("/reset") == {"kind": "reset_context"}
    assert parse_slash_command("/forget abc123") == {"kind": "memory_delete", "id": "abc123"}
    assert parse_slash_command("/memories weather forecast --limit 3") == {
        "kind": "memory_search",
        "query": "weather forecast",
        "limit": 3,
    }


def test_parse_slash_command_clear_requires_confirm_flag_presence() -> None:
    parsed = parse_slash_command("/memory clear shared")
    assert parsed == {"kind": "memory_clear", "scope": "shared", "confirm": False}

    parsed_confirmed = parse_slash_command("/memory clear shared --confirm")
    assert parsed_confirmed == {
        "kind": "memory_clear",
        "scope": "shared",
        "confirm": True,
    }


def test_parse_slash_command_sessions_commands() -> None:
    assert parse_slash_command("/memory sessions") == {
        "kind": "session_list",
        "scope": "mine",
        "limit": 20,
    }
    assert parse_slash_command("/sessions all --limit 5") == {
        "kind": "session_list",
        "scope": "all",
        "limit": 5,
    }
    assert parse_slash_command("/memory sessions show abc123 --limit 7") == {
        "kind": "session_show",
        "id": "abc123",
        "limit": 7,
    }
    assert parse_slash_command("/memory sessions clear all --confirm") == {
        "kind": "session_clear",
        "target": "all",
        "confirm": True,
    }


def test_extract_heuristic_memory() -> None:
    assert extract_heuristic_memory("remember that my favorite color is green") == (
        "my favorite color is green"
    )
    assert extract_heuristic_memory("I prefer metric units") == "metric units"
    assert extract_heuristic_memory("hello there") is None


def test_extract_heuristic_memory_ignores_sensitive() -> None:
    assert extract_heuristic_memory("remember that my api key is sk-secretsecretsecret") is None


def test_duplicate_detection() -> None:
    existing = ["Use metric units for weather", "lights should be warm white"]
    assert is_duplicate_memory("use metric units for weather", existing)
    assert not is_duplicate_memory("announce trash day on Tuesdays", existing)


def test_rank_memory_items_prefers_relevant_text() -> None:
    items = [
        {
            "id": "a",
            "text": "Use metric units for temperature and weather",
            "updated_at": utcnow_iso(),
            "scope": "user",
        },
        {
            "id": "b",
            "text": "Outdoor lights should be warm white in evening",
            "updated_at": utcnow_iso(),
            "scope": "user",
        },
    ]
    ranked = rank_memory_items(items, query="weather temperature units", top_k=1)
    assert len(ranked) == 1
    assert ranked[0]["id"] == "a"


def test_prune_memory_items_respects_max_items() -> None:
    items = [
        {"id": "a", "text": "one", "updated_at": "2026-02-10T00:00:00+00:00"},
        {"id": "b", "text": "two", "updated_at": "2026-02-11T00:00:00+00:00"},
        {"id": "c", "text": "three", "updated_at": "2026-02-12T00:00:00+00:00"},
    ]
    pruned = prune_memory_items(items, ttl_days=999, max_items=2)
    assert [item["id"] for item in pruned] == ["c", "b"]


def test_format_memory_prompt() -> None:
    prompt = format_memory_prompt(
        [
            {"id": "a", "scope": "user", "text": "I prefer concise responses"},
            {"id": "b", "scope": "shared", "text": "Home location is Seattle"},
        ]
    )
    assert prompt is not None
    assert "Long-term memory context" in prompt
    assert "[user] I prefer concise responses" in prompt
    assert "[shared] Home location is Seattle" in prompt
