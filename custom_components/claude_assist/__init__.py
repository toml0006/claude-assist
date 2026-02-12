"""The Claude Assist integration."""

from __future__ import annotations

import datetime
import time

import anthropic
import httpx

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import (
    config_validation as cv,
    device_registry as dr,
    entity_registry as er,
)
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_ACCESS_TOKEN,
    CONF_CHAT_MODEL,
    CONF_EXPIRES_AT,
    CONF_REFRESH_TOKEN,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
    LOGGER,
    OAUTH_CLIENT_ID,
    OAUTH_TOKEN_URL,
    TOKEN_REFRESH_INTERVAL,
)

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

# Endpoint to convert OAuth token to API key (same as Claude Code CLI)
API_KEY_URL = "https://api.anthropic.com/api/oauth/claude_cli/create_api_key"

type ClaudeAssistConfigEntry = ConfigEntry[anthropic.AsyncClient]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Claude Assist."""
    return True


async def _async_create_api_key(hass: HomeAssistant, oauth_token: str) -> str | None:
    """Convert an OAuth access token to an API key.

    Claude Code does this same conversion — OAuth tokens can't be used
    directly with /v1/messages, they must be exchanged for an API key first.
    """
    try:
        async_client = get_async_client(hass)
        response = await async_client.post(
            API_KEY_URL,
            content=None,
            headers={"Authorization": f"Bearer {oauth_token}"},
        )
        if response.status_code != 200:
            LOGGER.error(
                "Failed to create API key: %s %s - %s",
                response.status_code,
                response.reason_phrase,
                response.text,
            )
            return None
        data = response.json()
        api_key = data.get("raw_key")
        if api_key:
            LOGGER.debug("Successfully created API key from OAuth token")
            return api_key
        LOGGER.error("No raw_key in API key response: %s", data)
        return None
    except httpx.HTTPError as err:
        LOGGER.error("Failed to create API key from OAuth token: %s", err)
        return None


async def _async_refresh_token(hass: HomeAssistant, entry: ConfigEntry) -> str | None:
    """Refresh the OAuth access token and convert to API key.

    Returns a usable API key, or None on failure.
    """
    refresh_token = entry.data.get(CONF_REFRESH_TOKEN)
    if not refresh_token:
        LOGGER.error("No refresh token available for token refresh")
        return None

    try:
        async_client = get_async_client(hass)
        response = await async_client.post(
            OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": OAUTH_CLIENT_ID,
                "scope": "user:inference user:profile",
            },
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        token_data = response.json()
    except httpx.HTTPError as err:
        LOGGER.error("Failed to refresh token: %s", err)
        return None

    access_token = token_data["access_token"]
    new_data = {**entry.data}
    new_data[CONF_ACCESS_TOKEN] = access_token
    if "refresh_token" in token_data:
        new_data[CONF_REFRESH_TOKEN] = token_data["refresh_token"]
    expires_in = token_data.get("expires_in", 28800)
    new_data[CONF_EXPIRES_AT] = time.time() + expires_in

    hass.config_entries.async_update_entry(entry, data=new_data)
    LOGGER.debug("Successfully refreshed OAuth token, expires in %s seconds", expires_in)

    # Convert the new OAuth token to an API key
    api_key = await _async_create_api_key(hass, access_token)
    return api_key


def _create_client(hass: HomeAssistant, api_key: str) -> anthropic.AsyncClient:
    """Create an Anthropic async client using an API key."""
    return anthropic.AsyncAnthropic(
        api_key=api_key,
        http_client=get_async_client(hass),
    )


async def async_setup_entry(
    hass: HomeAssistant, entry: ClaudeAssistConfigEntry
) -> bool:
    """Set up Claude Assist from a config entry."""
    access_token = entry.data.get(CONF_ACCESS_TOKEN)
    expires_at = entry.data.get(CONF_EXPIRES_AT, 0)

    # Refresh token if expired or close to expiry (within 10 minutes)
    if time.time() > (expires_at - 600):
        LOGGER.debug("Access token expired or near expiry, refreshing...")
        api_key = await _async_refresh_token(hass, entry)
        if not api_key:
            raise ConfigEntryNotReady("Failed to refresh OAuth token")
    else:
        # Convert current OAuth token to API key
        api_key = await _async_create_api_key(hass, access_token)
        if not api_key:
            # Token might be stale, try refreshing
            api_key = await _async_refresh_token(hass, entry)
            if not api_key:
                raise ConfigEntryNotReady("Failed to create API key from OAuth token")

    client = _create_client(hass, api_key)

    # Validate the API key works
    try:
        await client.models.list(timeout=10.0)
    except anthropic.AuthenticationError as err:
        LOGGER.error("API key validation failed: %s", err)
        # Try refreshing once more
        api_key = await _async_refresh_token(hass, entry)
        if not api_key:
            return False
        client = _create_client(hass, api_key)
        try:
            await client.models.list(timeout=10.0)
        except anthropic.AuthenticationError as err2:
            LOGGER.error("Token refresh did not resolve auth error: %s", err2)
            return False
    except anthropic.AnthropicError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    # Set up periodic token refresh (get new OAuth token + convert to API key)
    async def _periodic_refresh(_now: datetime.datetime) -> None:
        """Periodically refresh the OAuth token and API key."""
        new_api_key = await _async_refresh_token(hass, entry)
        if new_api_key:
            entry.runtime_data = _create_client(hass, new_api_key)

    entry.async_on_unload(
        async_track_time_interval(
            hass,
            _periodic_refresh,
            datetime.timedelta(seconds=TOKEN_REFRESH_INTERVAL),
        )
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Claude Assist."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_options(
    hass: HomeAssistant, entry: ClaudeAssistConfigEntry
) -> None:
    """Update options."""
    await hass.config_entries.async_reload(entry.entry_id)
