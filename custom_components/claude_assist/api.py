"""Custom LLM API for Claude Assist with extended tools."""

from __future__ import annotations

from collections.abc import Callable

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm

from .const import LOGGER
from .tools import get_custom_tools

API_ID = "claude_assist"


class ClaudeAssistAPI(llm.API):
    """Claude Assist LLM API — wraps Assist API + adds custom tools."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass=hass, id=API_ID, name="Claude Assist")
        self._entry = entry

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        """Return an API instance merging Assist tools + our tools."""
        assist = await llm.async_get_api(self.hass, llm.LLM_API_ASSIST, llm_context)
        tools = list(assist.tools) + get_custom_tools(self.hass, self._entry)
        return llm.APIInstance(
            api=self,
            api_prompt=assist.api_prompt,
            llm_context=llm_context,
            tools=tools,
            custom_serializer=assist.custom_serializer,
        )


@callback
def async_register_claude_assist_api(hass: HomeAssistant, entry: ConfigEntry) -> Callable[[], None]:
    """Register the Claude Assist LLM API (id=claude_assist)."""
    api = ClaudeAssistAPI(hass, entry)
    try:
        return llm.async_register_api(hass, api)
    except Exception as err:
        # Most commonly: already registered
        LOGGER.debug("Claude Assist API registration skipped: %s", err)
        return lambda: None
