"""Pure contracts for memory/session websocket commands and result shapes."""

from __future__ import annotations

# Keep in sync with const.DOMAIN, duplicated so this module stays pure/importable
# without Home Assistant package context.
DOMAIN = "ai_subscription_assist"

WS_TYPE_ENTRY_LIST = f"{DOMAIN}/entry_list"
WS_TYPE_MEMORY_STATUS = f"{DOMAIN}/memory_status"
WS_TYPE_MEMORY_LIST = f"{DOMAIN}/memory_list"
WS_TYPE_MEMORY_DELETE = f"{DOMAIN}/memory_delete"
WS_TYPE_MEMORY_CLEAR = f"{DOMAIN}/memory_clear"
WS_TYPE_SESSION_LIST = f"{DOMAIN}/session_list"
WS_TYPE_SESSION_GET = f"{DOMAIN}/session_get"
WS_TYPE_SESSION_CLEAR = f"{DOMAIN}/session_clear"

WS_COMMAND_TYPES = {
    WS_TYPE_ENTRY_LIST,
    WS_TYPE_MEMORY_STATUS,
    WS_TYPE_MEMORY_LIST,
    WS_TYPE_MEMORY_DELETE,
    WS_TYPE_MEMORY_CLEAR,
    WS_TYPE_SESSION_LIST,
    WS_TYPE_SESSION_GET,
    WS_TYPE_SESSION_CLEAR,
}

# Key-level contract used by panel integration tests and docs.
WS_RESULT_KEYS = {
    WS_TYPE_ENTRY_LIST: {"entries", "count"},
    WS_TYPE_MEMORY_STATUS: {
        "entry_id",
        "memory_enabled",
        "auto_write",
        "auto_recall",
        "resume_enabled",
        "ttl_days",
        "max_items_per_scope",
        "recall_top_k",
        "resume_max_messages",
        "counts",
    },
    WS_TYPE_MEMORY_LIST: {"entry_id", "scope", "target_user_id", "count", "items"},
    WS_TYPE_MEMORY_DELETE: {"entry_id", "memory_id", "deleted", "scope"},
    WS_TYPE_MEMORY_CLEAR: {"entry_id", "scope", "target_user_id", "removed"},
    WS_TYPE_SESSION_LIST: {
        "entry_id",
        "scope",
        "subentry_id",
        "target_user_id",
        "count",
        "sessions",
    },
    WS_TYPE_SESSION_GET: {"entry_id", "session"},
    WS_TYPE_SESSION_CLEAR: {
        "entry_id",
        "scope",
        "subentry_id",
        "target_user_id",
        "session_id",
        "removed_sessions",
        "removed_messages",
    },
}
