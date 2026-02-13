"""Custom LLM API for Claude Assist with extended tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from homeassistant.components.homeassistant import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm

from .const import DOMAIN, LOGGER
from .tools import get_custom_tools

API_ID = "claude_assist"


class ClaudeAssistAPI(llm.API):
    """Claude Assist LLM API — wraps Assist API + adds custom tools."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize."""
        super().__init__(
            hass=hass,
            id=API_ID,
            name="Claude Assist",
        )
        self._entry = entry

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        """Get the API instance."""
        # Get the standard Assist API tools
        assist_api = llm.async_get_api(self.hass, llm.LLM_API_ASSIST, llm_context)

        # Combine with our custom tools
        all_tools = list(assist_api.tools) + get_custom_tools(self.hass, self._entry)

        return llm.APIInstance(
            api=self,
            api_prompt=assist_api.api_prompt,
            llm_context=llm_context,
            tools=all_tools,
            custom_serializer=assist_api.custom_serializer,
        )


@callback
def async_register_claude_assist_api(
    hass: HomeAssistant, entry: ConfigEntry
) -> Callable[[], None]:
    """Register the Claude Assist LLM API."""
    api = ClaudeAssistAPI(hass, entry)
    try:
        return llm.async_register_api(hass, api)
    except Exception:
        LOGGER.debug("Claude Assist API already registered or registration failed")
        return lambda: None
