"""The Claude Assist integration."""

from __future__ import annotations

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
    OAUTH_TOKEN_URL,
    TOKEN_REFRESH_INTERVAL,
)

import datetime

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type ClaudeAssistConfigEntry = ConfigEntry[anthropic.AsyncClient]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Claude Assist."""
    return True


async def _async_refresh_token(hass: HomeAssistant, entry: ConfigEntry) -> str | None:
    """Refresh the OAuth access token.

    Returns the new access token, or None on failure.
    """
    refresh_token = entry.data.get(CONF_REFRESH_TOKEN)
    if not refresh_token:
        LOGGER.error("No refresh token available for token refresh")
        return None

    try:
        async_client = get_async_client(hass)
        response = await async_client.post(
            OAUTH_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        token_data = response.json()
    except httpx.HTTPError as err:
        LOGGER.error("Failed to refresh token: %s", err)
        return None

    new_data = {**entry.data}
    new_data[CONF_ACCESS_TOKEN] = token_data["access_token"]
    if "refresh_token" in token_data:
        new_data[CONF_REFRESH_TOKEN] = token_data["refresh_token"]
    # expires_in is typically in seconds
    expires_in = token_data.get("expires_in", 28800)
    new_data[CONF_EXPIRES_AT] = time.time() + expires_in

    hass.config_entries.async_update_entry(entry, data=new_data)
    LOGGER.debug("Successfully refreshed OAuth token, expires in %s seconds", expires_in)
    return token_data["access_token"]


def _create_client(hass: HomeAssistant, access_token: str) -> anthropic.AsyncClient:
    """Create an Anthropic async client using OAuth access token.

    OAuth access tokens (sk-ant-oat01-...) work directly as the api_key
    since the Anthropic SDK sends them as Bearer tokens.
    """
    return anthropic.AsyncAnthropic(
        api_key=access_token,
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
        access_token = await _async_refresh_token(hass, entry)
        if not access_token:
            raise ConfigEntryNotReady("Failed to refresh OAuth token")

    client = _create_client(hass, access_token)

    # Validate the token works
    try:
        await client.models.list(timeout=10.0)
    except anthropic.AuthenticationError as err:
        LOGGER.error("Invalid OAuth token: %s", err)
        # Try refreshing once more
        access_token = await _async_refresh_token(hass, entry)
        if not access_token:
            return False
        client = _create_client(hass, access_token)
        try:
            await client.models.list(timeout=10.0)
        except anthropic.AuthenticationError as err2:
            LOGGER.error("Token refresh did not resolve auth error: %s", err2)
            return False
    except anthropic.AnthropicError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    # Set up periodic token refresh
    async def _periodic_refresh(_now: datetime.datetime) -> None:
        """Periodically refresh the OAuth token."""
        new_token = await _async_refresh_token(hass, entry)
        if new_token:
            entry.runtime_data = _create_client(hass, new_token)

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
