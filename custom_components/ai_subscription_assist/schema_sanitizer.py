"""Sanitize JSON schemas for the Anthropic Messages API.

The Anthropic API rejects tool input schemas that use ``anyOf``, ``oneOf`` or
``allOf`` at the top level of ``input_schema``. Home Assistant tools whose
parameters use ``vol.Any(...)`` (e.g. ``HassStartTimer``) produce exactly that
shape via ``voluptuous_openapi.convert``, which breaks every request that
includes such a tool. This module flattens those combinators into a single
permissive schema that the API accepts.
"""

from __future__ import annotations

from typing import Any

_COMBINATOR_KEYS = ("allOf", "anyOf", "oneOf")


def _merge_options(
    options: list[dict[str, Any]], *, union_required: bool
) -> dict[str, Any]:
    """Merge combinator options into one permissive schema fragment.

    For ``allOf`` every option applies, so ``required`` is the union of the
    options' requirements. For ``anyOf``/``oneOf`` only one option needs to
    match, so only fields required by every option stay required.
    """
    merged: dict[str, Any] = {}
    types: list[Any] = []
    required_sets: list[list[str]] = []

    for option in options:
        for key, value in option.items():
            if key == "type":
                types.append(value)
            elif key == "required":
                required_sets.append(list(value))
            elif key == "properties" and isinstance(value, dict):
                merged.setdefault("properties", {}).update(value)
            elif key not in merged:
                merged[key] = value

    if types:
        if all(entry == types[0] for entry in types):
            merged["type"] = types[0]
        elif "object" in types:
            merged["type"] = "object"

    required: list[str] = []
    if union_required:
        for entry in required_sets:
            for name in entry:
                if name not in required:
                    required.append(name)
    elif required_sets and len(required_sets) == len(options):
        common = set(required_sets[0]).intersection(*map(set, required_sets[1:]))
        required = [name for name in required_sets[0] if name in common]
    if required:
        merged["required"] = required

    return merged


def _sanitize_node(schema: Any) -> Any:
    """Recursively flatten combinators in a schema fragment."""
    if not isinstance(schema, dict):
        return schema

    result = dict(schema)

    for combinator in _COMBINATOR_KEYS:
        options = result.pop(combinator, None)
        if not isinstance(options, list):
            continue
        merged = _merge_options(
            [
                sanitized
                for option in options
                if isinstance(sanitized := _sanitize_node(option), dict)
            ],
            union_required=combinator == "allOf",
        )
        for key, value in merged.items():
            if key == "properties" and isinstance(result.get("properties"), dict):
                result["properties"] = {**value, **result["properties"]}
            elif key not in result:
                result[key] = value

    if isinstance(result.get("properties"), dict):
        result["properties"] = {
            name: _sanitize_node(value)
            for name, value in result["properties"].items()
        }

    items = result.get("items")
    if isinstance(items, list):
        result["items"] = [_sanitize_node(entry) for entry in items]
    elif isinstance(items, dict):
        result["items"] = _sanitize_node(items)

    return result


def sanitize_schema(schema: Any) -> dict[str, Any]:
    """Return a schema safe to use as an Anthropic tool ``input_schema``.

    Flattens ``anyOf``/``oneOf``/``allOf`` at every level and guarantees the
    root schema is a typed object, as required by the API.
    """
    result = _sanitize_node(schema)
    if not isinstance(result, dict):
        return {"type": "object", "properties": {}}
    result.setdefault("type", "object")
    return result
