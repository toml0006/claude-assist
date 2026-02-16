"""Safety helpers for destructive dashboard structure edits."""

from __future__ import annotations

import re

_VIEW_WORD = r"(?:view|views|tab|tabs|page|pages)"
_ADD_VERB = r"(?:add|create|make)"
_REMOVE_VERB = r"(?:remove|delete|drop)"

_ADD_REQUEST_NEW_PATTERN = re.compile(rf"\b(?:new|another|extra)\b.{{0,15}}\b{_VIEW_WORD}\b")
_ADD_REQUEST_VERB_PATTERN = re.compile(rf"\b{_ADD_VERB}\b(?P<tail>.{{0,35}})\b{_VIEW_WORD}\b")
_REMOVE_REQUEST_PATTERNS = (
    re.compile(rf"\b{_REMOVE_VERB}\b.{{0,30}}\b{_VIEW_WORD}\b"),
)

_ADD_NEGATION_PATTERNS = (
    re.compile(
        rf"\b(?:do not|don't|dont|stop|avoid|without|never|no)\b.{{0,35}}\b{_ADD_VERB}\b.{{0,35}}\b{_VIEW_WORD}\b"
    ),
    re.compile(rf"\b(?:stop|avoid)\b.{{0,20}}\b{_VIEW_WORD}\b"),
)
_REMOVE_NEGATION_PATTERNS = (
    re.compile(
        rf"\b(?:do not|don't|dont|stop|avoid|without|never)\b.{{0,35}}\b{_REMOVE_VERB}\b.{{0,35}}\b{_VIEW_WORD}\b"
    ),
)


def _normalize(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def has_explicit_view_change_request(action: str, text: str | None) -> bool:
    """Return True when text clearly requests creating/removing a dashboard view."""
    normalized = _normalize(text)
    if not normalized:
        return False
    if action == "add_view":
        if _ADD_REQUEST_NEW_PATTERN.search(normalized):
            return True
        match = _ADD_REQUEST_VERB_PATTERN.search(normalized)
        if not match:
            return False
        tail = match.group("tail")
        if "card" in tail and not any(token in tail for token in ("new", "another", "extra")):
            return False
        return True
    if action == "remove_view":
        return _matches_any(normalized, _REMOVE_REQUEST_PATTERNS)
    return False


def has_negated_view_change_request(action: str, text: str | None) -> bool:
    """Return True when text explicitly tells us not to perform a structure edit."""
    normalized = _normalize(text)
    if not normalized:
        return False
    if action == "add_view":
        return _matches_any(normalized, _ADD_NEGATION_PATTERNS)
    if action == "remove_view":
        return _matches_any(normalized, _REMOVE_NEGATION_PATTERNS)
    return False


def validate_view_change_request(
    action: str,
    user_request: str | None,
    context_text: str | None,
) -> str | None:
    """Validate explicit intent for add/remove view actions."""
    quoted = _normalize(user_request)
    if not quoted:
        return (
            f"{action} requires user_request with the user's explicit ask "
            "to create/delete a view or tab"
        )
    if has_negated_view_change_request(action, quoted):
        return (
            f"{action} blocked because user_request indicates not to change "
            "dashboard structure"
        )
    if not has_explicit_view_change_request(action, quoted):
        return (
            f"{action} blocked: user_request must explicitly mention creating/deleting "
            "a dashboard view/tab"
        )

    context = _normalize(context_text)
    if context:
        if has_negated_view_change_request(action, context):
            return (
                f"{action} blocked because the latest user instruction says "
                "not to change dashboard structure"
            )
        if not has_explicit_view_change_request(action, context):
            return (
                f"{action} blocked because the latest user instruction does not "
                "explicitly request creating/deleting a dashboard view/tab"
            )
    return None
