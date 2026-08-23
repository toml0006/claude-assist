"""Unit tests for the Anthropic tool schema sanitizer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "ai_subscription_assist"
    / "schema_sanitizer.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ai_subscription_assist_schema_sanitizer", MODULE_PATH
)
assert SPEC and SPEC.loader
schema_sanitizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(schema_sanitizer)

sanitize_schema = schema_sanitizer.sanitize_schema

COMBINATORS = ("anyOf", "oneOf", "allOf")


def _assert_no_combinators(schema: Any) -> None:
    if isinstance(schema, dict):
        for combinator in COMBINATORS:
            assert combinator not in schema
        for value in schema.values():
            _assert_no_combinators(value)
    elif isinstance(schema, list):
        for entry in schema:
            _assert_no_combinators(entry)


def test_top_level_anyof_is_flattened() -> None:
    # Shape produced by voluptuous_openapi.convert for HassStartTimer-style
    # vol.Any(...) parameters (home-assistant/core#160565).
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"hours": {"type": "integer"}},
                "required": ["hours"],
            },
            {
                "type": "object",
                "properties": {"minutes": {"type": "integer"}},
                "required": ["minutes"],
            },
        ]
    }
    result = sanitize_schema(schema)
    _assert_no_combinators(result)
    assert result["type"] == "object"
    assert set(result["properties"]) == {"hours", "minutes"}
    # Each field is required by only one alternative, so none stay required.
    assert "required" not in result


def test_top_level_oneof_is_flattened() -> None:
    schema = {
        "oneOf": [
            {"type": "object", "properties": {"name": {"type": "string"}}},
            {"type": "object", "properties": {"area": {"type": "string"}}},
        ]
    }
    result = sanitize_schema(schema)
    _assert_no_combinators(result)
    assert result["type"] == "object"
    assert set(result["properties"]) == {"name", "area"}


def test_top_level_allof_merges_and_unions_required() -> None:
    schema = {
        "allOf": [
            {
                "type": "object",
                "properties": {"entity_id": {"type": "string"}},
                "required": ["entity_id"],
            },
            {
                "type": "object",
                "properties": {"brightness": {"type": "integer"}},
                "required": ["brightness"],
            },
        ]
    }
    result = sanitize_schema(schema)
    _assert_no_combinators(result)
    assert result["type"] == "object"
    assert set(result["properties"]) == {"entity_id", "brightness"}
    assert result["required"] == ["entity_id", "brightness"]


def test_anyof_common_required_is_kept() -> None:
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "area": {"type": "string"}},
                "required": ["name", "area"],
            },
        ]
    }
    result = sanitize_schema(schema)
    _assert_no_combinators(result)
    assert result["required"] == ["name"]


def test_nested_combinators_in_properties_and_items() -> None:
    schema = {
        "type": "object",
        "properties": {
            "duration": {
                "anyOf": [{"type": "string"}, {"type": "number"}],
            },
            "targets": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"type": "object", "properties": {"id": {"type": "string"}}},
                        {"type": "object", "properties": {"name": {"type": "string"}}},
                    ]
                },
            },
        },
    }
    result = sanitize_schema(schema)
    _assert_no_combinators(result)
    assert set(result["properties"]["targets"]["items"]["properties"]) == {
        "id",
        "name",
    }
    # Conflicting scalar types stay permissive: no bogus "type" is invented.
    assert "type" not in result["properties"]["duration"]


def test_sibling_keys_take_precedence_over_merged_options() -> None:
    schema = {
        "type": "object",
        "description": "outer",
        "properties": {"kept": {"type": "string"}},
        "anyOf": [
            {
                "type": "string",
                "description": "inner",
                "properties": {"added": {"type": "integer"}},
            }
        ],
    }
    result = sanitize_schema(schema)
    _assert_no_combinators(result)
    assert result["type"] == "object"
    assert result["description"] == "outer"
    assert set(result["properties"]) == {"kept", "added"}


def test_root_always_has_a_type() -> None:
    assert sanitize_schema({}) == {"type": "object"}
    assert sanitize_schema(None) == {"type": "object", "properties": {}}
    plain = sanitize_schema({"properties": {"x": {"type": "string"}}})
    assert plain["type"] == "object"


def test_schema_without_combinators_is_unchanged() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    assert sanitize_schema(schema) == schema
