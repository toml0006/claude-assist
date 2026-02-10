"""Constants for the Claude Assist integration."""

import logging

DOMAIN = "claude_assist"
LOGGER = logging.getLogger(__package__)

DEFAULT_CONVERSATION_NAME = "Claude Assist conversation"
DEFAULT_AI_TASK_NAME = "Claude Assist AI Task"

# OAuth constants
OAUTH_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
OAUTH_TOKEN_URL = "https://console.anthropic.com/api/oauth/token"
OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_SCOPES = "user:inference user:profile"
OAUTH_REDIRECT_PATH = "/auth/external/callback"

CONF_ACCESS_TOKEN = "access_token"
CONF_REFRESH_TOKEN = "refresh_token"
CONF_EXPIRES_AT = "expires_at"

CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"
CONF_THINKING_BUDGET = "thinking_budget"
CONF_THINKING_EFFORT = "thinking_effort"
CONF_WEB_SEARCH = "web_search"
CONF_WEB_SEARCH_USER_LOCATION = "user_location"
CONF_WEB_SEARCH_MAX_USES = "web_search_max_uses"
CONF_WEB_SEARCH_CITY = "city"
CONF_WEB_SEARCH_REGION = "region"
CONF_WEB_SEARCH_COUNTRY = "country"
CONF_WEB_SEARCH_TIMEZONE = "timezone"

DATA_REPAIR_DEFER_RELOAD = "repair_defer_reload"

DEFAULT = {
    CONF_CHAT_MODEL: "claude-haiku-4-5",
    CONF_MAX_TOKENS: 3000,
    CONF_TEMPERATURE: 1.0,
    CONF_THINKING_BUDGET: 0,
    CONF_THINKING_EFFORT: "low",
    CONF_WEB_SEARCH: False,
    CONF_WEB_SEARCH_USER_LOCATION: False,
    CONF_WEB_SEARCH_MAX_USES: 5,
}

MIN_THINKING_BUDGET = 1024

NON_THINKING_MODELS = [
    "claude-3-5",  # Both sonnet and haiku
    "claude-3-opus",
    "claude-3-haiku",
]

NON_ADAPTIVE_THINKING_MODELS = [
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-1",
    "claude-opus-4-0",
    "claude-opus-4-20250514",
    "claude-sonnet-4-0",
    "claude-sonnet-4-20250514",
    "claude-3",
]

WEB_SEARCH_UNSUPPORTED_MODELS = [
    "claude-3-haiku",
    "claude-3-opus",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
]

DEPRECATED_MODELS = [
    "claude-3-5-haiku",
    "claude-3-7-sonnet",
    "claude-3-5-sonnet",
    "claude-3-opus",
]

# Token refresh interval (7 hours, tokens expire after 8)
TOKEN_REFRESH_INTERVAL = 7 * 60 * 60
