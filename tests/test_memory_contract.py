"""Unit tests for websocket memory/session command contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "ai_subscription_assist"
    / "memory_contract.py"
)
SPEC = importlib.util.spec_from_file_location("ai_subscription_assist_memory_contract", CONTRACT_PATH)
assert SPEC and SPEC.loader
memory_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory_contract)


def test_ws_command_type_set_is_complete() -> None:
    assert len(memory_contract.WS_COMMAND_TYPES) == 8
    assert memory_contract.WS_TYPE_MEMORY_STATUS in memory_contract.WS_COMMAND_TYPES
    assert memory_contract.WS_TYPE_SESSION_CLEAR in memory_contract.WS_COMMAND_TYPES


def test_ws_result_keys_cover_all_commands() -> None:
    assert set(memory_contract.WS_RESULT_KEYS) == memory_contract.WS_COMMAND_TYPES


def test_ws_memory_status_result_has_required_keys() -> None:
    keys = memory_contract.WS_RESULT_KEYS[memory_contract.WS_TYPE_MEMORY_STATUS]
    assert "entry_id" in keys
    assert "memory_enabled" in keys
    assert "counts" in keys


def test_ws_session_list_result_has_filter_context_keys() -> None:
    keys = memory_contract.WS_RESULT_KEYS[memory_contract.WS_TYPE_SESSION_LIST]
    assert "entry_id" in keys
    assert "scope" in keys
    assert "subentry_id" in keys
    assert "target_user_id" in keys
    assert "sessions" in keys
