"""Config flow for Claude Assist integration."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import time
from typing import Any, cast
from urllib.parse import urlencode

import anthropic
import httpx
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components.zone import ENTITY_ID_HOME
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import (
    ATTR_LATITUDE,
    ATTR_LONGITUDE,
    CONF_LLM_HASS_API,
    CONF_NAME,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.network import get_url
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)
from homeassistant.helpers.typing import VolDictType

from .const import (
    CONF_ACCESS_TOKEN,
    CONF_CHAT_MODEL,
    CONF_EXPIRES_AT,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_REFRESH_TOKEN,
    CONF_TEMPERATURE,
    CONF_THINKING_BUDGET,
    CONF_THINKING_EFFORT,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_CITY,
    CONF_WEB_SEARCH_COUNTRY,
    CONF_WEB_SEARCH_MAX_USES,
    CONF_WEB_SEARCH_REGION,
    CONF_WEB_SEARCH_TIMEZONE,
    CONF_WEB_SEARCH_USER_LOCATION,
    DEFAULT,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
    NON_ADAPTIVE_THINKING_MODELS,
    NON_THINKING_MODELS,
    OAUTH_AUTHORIZE_URL,
    OAUTH_CLIENT_ID,
    OAUTH_REDIRECT_PATH,
    OAUTH_SCOPES,
    OAUTH_TOKEN_URL,
    WEB_SEARCH_UNSUPPORTED_MODELS,
)

_LOGGER = logging.getLogger(__name__)


DEFAULT_CONVERSATION_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}

DEFAULT_AI_TASK_OPTIONS = {
    CONF_RECOMMENDED: True,
}


def _generate_pkce_pair() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge."""
    code_verifier = secrets.token_urlsafe(64)[:128]
    code_challenge_digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = (
        base64.urlsafe_b64encode(code_challenge_digest).rstrip(b"=").decode("ascii")
    )
    return code_verifier, code_challenge


async def get_model_list(client: anthropic.AsyncAnthropic) -> list[SelectOptionDict]:
    """Get list of available models."""
    try:
        models = (await client.models.list()).data
    except (anthropic.AnthropicError, Exception):
        models = []
    _LOGGER.debug("Available models: %s", models)
    model_options: list[SelectOptionDict] = []
    short_form = re.compile(r"[^\d]-\d$")
    for model_info in models:
        # Resolve alias from versioned model name
        model_alias = (
            model_info.id[:-9]
            if model_info.id
            not in (
                "claude-3-haiku-20240307",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
            )
            and model_info.id[-2:-1] != "-"
            else model_info.id
        )
        if short_form.search(model_alias):
            model_alias += "-0"
        if model_alias.endswith(("haiku", "opus", "sonnet")):
            model_alias += "-latest"
        model_options.append(
            SelectOptionDict(
                label=model_info.display_name,
                value=model_alias,
            )
        )
    return model_options


class ClaudeAssistConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Claude Assist."""

    VERSION = 1
    MINOR_VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._code_verifier: str | None = None
        self._state: str | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step — show OAuth link."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # User has submitted the authorization code
            auth_code = user_input.get("auth_code", "").strip()
            # Handle case where user pastes the full redirect URL
            if "code=" in auth_code:
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(auth_code)
                code_values = parse_qs(parsed.query).get("code", [])
                if code_values:
                    auth_code = code_values[0]
            if not auth_code:
                errors["base"] = "no_auth_code"
            else:
                # Exchange the auth code for tokens
                try:
                    redirect_uri = self._get_redirect_uri()
                    async_client = get_async_client(self.hass)
                    response = await async_client.post(
                        OAUTH_TOKEN_URL,
                        json={
                            "grant_type": "authorization_code",
                            "code": auth_code,
                            "redirect_uri": redirect_uri,
                            "client_id": OAUTH_CLIENT_ID,
                            "code_verifier": self._code_verifier,
                            "state": self._state,
                        },
                        headers={
                            "Content-Type": "application/json",
                        },
                    )
                    response.raise_for_status()
                    token_data = response.json()
                except httpx.HTTPStatusError as err:
                    _LOGGER.error(
                        "Token exchange failed: %s - %s",
                        err.response.status_code,
                        err.response.text,
                    )
                    errors["base"] = "token_exchange_failed"
                except httpx.HTTPError as err:
                    _LOGGER.error("Token exchange HTTP error: %s", err)
                    errors["base"] = "cannot_connect"
                except Exception:
                    _LOGGER.exception("Unexpected error during token exchange")
                    errors["base"] = "unknown"

                if not errors:
                    access_token = token_data["access_token"]
                    refresh_token = token_data.get("refresh_token", "")
                    expires_in = token_data.get("expires_in", 28800)
                    expires_at = time.time() + expires_in

                    # Validate the token works with a simple messages call
                    # OAuth tokens can't use models.list (API-key only)
                    try:
                        client = anthropic.AsyncAnthropic(
                            api_key=access_token,
                            http_client=get_async_client(self.hass),
                        )
                        await client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=10,
                            messages=[{"role": "user", "content": "hi"}],
                            timeout=15.0,
                        )
                    except anthropic.AuthenticationError:
                        errors["base"] = "authentication_error"
                    except anthropic.AnthropicError as err:
                        _LOGGER.error("API validation error: %s", err)
                        # If we got tokens, don't block on validation failure
                        # The token may have limited scopes
                    except Exception:
                        _LOGGER.exception("Unexpected validation error")
                        # Same — don't block, tokens were obtained

                if not errors:
                    return self.async_create_entry(
                        title="Claude Assist",
                        data={
                            CONF_ACCESS_TOKEN: access_token,
                            CONF_REFRESH_TOKEN: refresh_token,
                            CONF_EXPIRES_AT: expires_at,
                        },
                        subentries=[
                            {
                                "subentry_type": "conversation",
                                "data": DEFAULT_CONVERSATION_OPTIONS,
                                "title": DEFAULT_CONVERSATION_NAME,
                                "unique_id": None,
                            },
                        ],
                    )

        # Generate PKCE pair and state for the OAuth flow
        self._code_verifier, code_challenge = _generate_pkce_pair()
        self._state = secrets.token_urlsafe(32)
        redirect_uri = self._get_redirect_uri()

        auth_params = urlencode(
            {
                "response_type": "code",
                "client_id": OAUTH_CLIENT_ID,
                "redirect_uri": redirect_uri,
                "scope": OAUTH_SCOPES,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
                "state": self._state,
            }
        )
        auth_url = f"{OAUTH_AUTHORIZE_URL}?{auth_params}"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required("auth_code"): str,
                }
            ),
            description_placeholders={
                "auth_url": auth_url,
                "redirect_uri": redirect_uri,
            },
            errors=errors or None,
        )

    def _get_redirect_uri(self) -> str:
        """Get the redirect URI for OAuth."""
        # Use Anthropic's manual redirect page which displays the code for copy/paste
        return "https://platform.claude.com/oauth/code/callback"

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": ConversationSubentryFlowHandler,
        }


class ConversationSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing conversation subentries."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a subentry."""
        self.options = DEFAULT_CONVERSATION_OPTIONS.copy()
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of a subentry."""
        self.options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Set initial options."""
        # abort if entry is not loaded
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(
                label=api.name,
                value=api.id,
            )
            for api in llm.async_get_apis(self.hass)
        ]
        if suggested_llm_apis := self.options.get(CONF_LLM_HASS_API):
            if isinstance(suggested_llm_apis, str):
                suggested_llm_apis = [suggested_llm_apis]
            known_apis = {api.id for api in llm.async_get_apis(self.hass)}
            self.options[CONF_LLM_HASS_API] = [
                api for api in suggested_llm_apis if api in known_apis
            ]

        step_schema: VolDictType = {}
        errors: dict[str, str] = {}

        if self._is_new:
            default_name = DEFAULT_CONVERSATION_NAME
            step_schema[vol.Required(CONF_NAME, default=default_name)] = str

        step_schema.update(
            {
                vol.Optional(CONF_PROMPT): TemplateSelector(),
                vol.Optional(
                    CONF_LLM_HASS_API,
                ): SelectSelector(
                    SelectSelectorConfig(options=hass_apis, multiple=True)
                ),
            }
        )

        step_schema[
            vol.Required(
                CONF_RECOMMENDED, default=self.options.get(CONF_RECOMMENDED, False)
            )
        ] = bool

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)

            if user_input[CONF_RECOMMENDED]:
                if not errors:
                    if self._is_new:
                        return self.async_create_entry(
                            title=user_input.pop(CONF_NAME),
                            data=user_input,
                        )
                    return self.async_update_and_abort(
                        self._get_entry(),
                        self._get_reconfigure_subentry(),
                        data=user_input,
                    )
            else:
                self.options.update(user_input)
                if (
                    CONF_LLM_HASS_API in self.options
                    and CONF_LLM_HASS_API not in user_input
                ):
                    self.options.pop(CONF_LLM_HASS_API)
                if not errors:
                    return await self.async_step_advanced()

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), self.options
            ),
            errors=errors or None,
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage advanced options."""
        errors: dict[str, str] = {}

        step_schema: VolDictType = {
            vol.Optional(
                CONF_CHAT_MODEL,
                default=DEFAULT[CONF_CHAT_MODEL],
            ): SelectSelector(
                SelectSelectorConfig(
                    options=await self._get_model_list(), custom_value=True
                )
            ),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=DEFAULT[CONF_MAX_TOKENS],
            ): int,
            vol.Optional(
                CONF_TEMPERATURE,
                default=DEFAULT[CONF_TEMPERATURE],
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        }

        if user_input is not None:
            self.options.update(user_input)

            if not errors:
                return await self.async_step_model()

        return self.async_show_form(
            step_id="advanced",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), self.options
            ),
            errors=errors,
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage model-specific options."""
        errors: dict[str, str] = {}

        step_schema: VolDictType = {}

        model = self.options[CONF_CHAT_MODEL]

        if not model.startswith(tuple(NON_THINKING_MODELS)) and model.startswith(
            tuple(NON_ADAPTIVE_THINKING_MODELS)
        ):
            step_schema[
                vol.Optional(
                    CONF_THINKING_BUDGET, default=DEFAULT[CONF_THINKING_BUDGET]
                )
            ] = vol.All(
                NumberSelector(
                    NumberSelectorConfig(
                        min=0,
                        max=self.options.get(CONF_MAX_TOKENS, DEFAULT[CONF_MAX_TOKENS]),
                    )
                ),
                vol.Coerce(int),
            )
        else:
            self.options.pop(CONF_THINKING_BUDGET, None)

        if not model.startswith(tuple(NON_ADAPTIVE_THINKING_MODELS)):
            step_schema[
                vol.Optional(
                    CONF_THINKING_EFFORT,
                    default=DEFAULT[CONF_THINKING_EFFORT],
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=["none", "low", "medium", "high", "max"],
                    translation_key=CONF_THINKING_EFFORT,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )
        else:
            self.options.pop(CONF_THINKING_EFFORT, None)

        if not model.startswith(tuple(WEB_SEARCH_UNSUPPORTED_MODELS)):
            step_schema.update(
                {
                    vol.Optional(
                        CONF_WEB_SEARCH,
                        default=DEFAULT[CONF_WEB_SEARCH],
                    ): bool,
                    vol.Optional(
                        CONF_WEB_SEARCH_MAX_USES,
                        default=DEFAULT[CONF_WEB_SEARCH_MAX_USES],
                    ): int,
                    vol.Optional(
                        CONF_WEB_SEARCH_USER_LOCATION,
                        default=DEFAULT[CONF_WEB_SEARCH_USER_LOCATION],
                    ): bool,
                }
            )
        else:
            self.options.pop(CONF_WEB_SEARCH, None)
            self.options.pop(CONF_WEB_SEARCH_MAX_USES, None)
            self.options.pop(CONF_WEB_SEARCH_USER_LOCATION, None)

        self.options.pop(CONF_WEB_SEARCH_CITY, None)
        self.options.pop(CONF_WEB_SEARCH_REGION, None)
        self.options.pop(CONF_WEB_SEARCH_COUNTRY, None)
        self.options.pop(CONF_WEB_SEARCH_TIMEZONE, None)

        if not step_schema:
            user_input = {}

        if user_input is not None:
            if user_input.get(CONF_WEB_SEARCH, DEFAULT[CONF_WEB_SEARCH]) and not errors:
                if user_input.get(
                    CONF_WEB_SEARCH_USER_LOCATION,
                    DEFAULT[CONF_WEB_SEARCH_USER_LOCATION],
                ):
                    user_input.update(await self._get_location_data())

            self.options.update(user_input)

            if not errors:
                if self._is_new:
                    return self.async_create_entry(
                        title=self.options.pop(CONF_NAME),
                        data=self.options,
                    )

                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=self.options,
                )

        return self.async_show_form(
            step_id="model",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), self.options
            ),
            errors=errors or None,
            last_step=True,
        )

    async def _get_model_list(self) -> list[SelectOptionDict]:
        """Get list of available models."""
        entry = self._get_entry()
        client = anthropic.AsyncAnthropic(
            api_key=entry.data[CONF_ACCESS_TOKEN],
            http_client=get_async_client(self.hass),
        )
        return await get_model_list(client)

    async def _get_location_data(self) -> dict[str, str]:
        """Get approximate location data of the user."""
        location_data: dict[str, str] = {}
        zone_home = self.hass.states.get(ENTITY_ID_HOME)
        if zone_home is not None:
            entry = self._get_entry()
            client = anthropic.AsyncAnthropic(
                api_key=entry.data[CONF_ACCESS_TOKEN],
                http_client=get_async_client(self.hass),
            )
            location_schema = vol.Schema(
                {
                    vol.Optional(
                        CONF_WEB_SEARCH_CITY,
                        description="Free text input for the city, e.g. `San Francisco`",
                    ): str,
                    vol.Optional(
                        CONF_WEB_SEARCH_REGION,
                        description="Free text input for the region, e.g. `California`",
                    ): str,
                }
            )
            response = await client.messages.create(
                model=cast(str, DEFAULT[CONF_CHAT_MODEL]),
                messages=[
                    {
                        "role": "user",
                        "content": "Where are the following coordinates located: "
                        f"({zone_home.attributes[ATTR_LATITUDE]},"
                        f" {zone_home.attributes[ATTR_LONGITUDE]})? Please respond "
                        "only with a JSON object using the following schema:\n"
                        f"{convert(location_schema)}",
                    },
                    {
                        "role": "assistant",
                        "content": "{",  # hints the model to skip any preamble
                    },
                ],
                max_tokens=cast(int, DEFAULT[CONF_MAX_TOKENS]),
            )
            _LOGGER.debug("Model response: %s", response.content)
            location_data = location_schema(
                json.loads(
                    "{"
                    + "".join(
                        block.text
                        for block in response.content
                        if isinstance(block, anthropic.types.TextBlock)
                    )
                )
                or {}
            )

        if self.hass.config.country:
            location_data[CONF_WEB_SEARCH_COUNTRY] = self.hass.config.country
        location_data[CONF_WEB_SEARCH_TIMEZONE] = self.hass.config.time_zone

        _LOGGER.debug("Location data: %s", location_data)

        return location_data
