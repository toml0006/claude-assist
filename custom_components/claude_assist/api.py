"""Custom LLM API for Claude Assist with extended tools."""

from __future__ import annotations

from collections.abc import Callable

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm

from .const import CONF_ENABLED_TOOLS, LOGGER
from .tools import get_custom_tools


def _api_id_for_subentry(subentry_id: str) -> str:
    return f"claude_assist.{subentry_id}"


class ClaudeAssistSubentryAPI(llm.API):
    """LLM API bound to a specific conversation subentry.

    This allows per-agent tool enable/disable.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str) -> None:
        self._entry = entry
        self._subentry_id = subentry_id
        # Name shown in HA UI (LLM API selector)
        name = f"Claude Assist ({subentry_id})"
        sub = entry.subentries.get(subentry_id)
        if sub is not None:
            name = f"Claude Assist — {sub.title}"
        super().__init__(hass=hass, id=_api_id_for_subentry(subentry_id), name=name)

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        assist = await llm.async_get_api(self.hass, llm.LLM_API_ASSIST, llm_context)

        sub = self._entry.subentries.get(self._subentry_id)
        enabled = None
        if sub is not None:
            enabled = sub.data.get(CONF_ENABLED_TOOLS)
            if isinstance(enabled, str):
                enabled = [enabled]

        tools = list(assist.tools) + get_custom_tools(self.hass, self._entry, enabled=enabled)

        return llm.APIInstance(
            api=self,
            api_prompt=assist.api_prompt,
            llm_context=llm_context,
            tools=tools,
            custom_serializer=assist.custom_serializer,
        )


@callback
def async_register_claude_assist_apis(hass: HomeAssistant, entry: ConfigEntry) -> Callable[[], None]:
    """Register per-subentry Claude Assist APIs.

    Returns an unregister callable.
    """
    unregisters: list[Callable[[], None]] = []
    for subentry in entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue
        api = ClaudeAssistSubentryAPI(hass, entry, subentry.subentry_id)
        try:
            unregisters.append(llm.async_register_api(hass, api))
        except Exception as err:
            LOGGER.debug("Claude Assist API registration skipped (%s): %s", api.id, err)

    def _unregister_all() -> None:
        for unreg in unregisters:
            try:
                unreg()
            except Exception:
                pass

    return _unregister_all
