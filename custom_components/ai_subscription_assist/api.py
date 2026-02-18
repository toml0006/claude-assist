"""Custom LLM API for AI Subscription Assist with extended tools."""

from __future__ import annotations

from collections.abc import Callable

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm

from .const import CONF_ENABLED_TOOLS, LOGGER
from .tools import get_custom_tools


def _api_id_for_subentry(subentry_id: str) -> str:
    return f"ai_subscription_assist.{subentry_id}"


class AiSubscriptionAssistSubentryAPI(llm.API):
    """LLM API bound to a specific conversation subentry.

    This allows per-agent tool enable/disable.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str) -> None:
        self._entry = entry
        self._subentry_id = subentry_id
        # Name shown in HA UI (LLM API selector)
        name = f"AI Subscription Assist ({entry.title}) ({subentry_id})"
        sub = entry.subentries.get(subentry_id)
        if sub is not None:
            name = f"AI Subscription Assist — {entry.title} — {sub.title}"
        super().__init__(hass=hass, id=_api_id_for_subentry(subentry_id), name=name)

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        assist = await llm.async_get_api(self.hass, llm.LLM_API_ASSIST, llm_context)

        sub = self._entry.subentries.get(self._subentry_id)
        enabled = None
        if sub is not None:
            enabled = sub.data.get(CONF_ENABLED_TOOLS)
            if isinstance(enabled, str):
                enabled = [enabled]

        custom_tools = get_custom_tools(
            self.hass,
            self._entry,
            enabled=enabled,
            subentry_id=self._subentry_id,
        )
        tools = list(assist.tools) + custom_tools

        # Extend Assist prompt so the model knows these tools exist.
        tool_names = {t.name for t in custom_tools}
        extra_lines: list[str] = []
        if "internet_lookup" in tool_names:
            extra_lines.append(
                "- Use `internet_lookup` for read-only web search/fetch when you need up-to-date external information."
            )
        if "get_history" in tool_names:
            extra_lines.append(
                "- Use `get_history` to answer questions about *when* something turned on/off or how long it has been in a state."
            )
        if "get_logbook" in tool_names:
            extra_lines.append("- Use `get_logbook` to answer questions about recent events.")
        if "get_statistics" in tool_names:
            extra_lines.append("- Use `get_statistics` for aggregated sensor/energy data over time.")
        if "render_template" in tool_names:
            extra_lines.append("- Use `render_template` to compute answers with Jinja templates.")
        if "modify_dashboard" in tool_names:
            extra_lines.append(
                "- For `modify_dashboard`, always inspect first with `list`/`get`. "
                "Prefer editing existing views via `add_card`/`remove_card`, and only use "
                "`add_view`/`remove_view` when the user explicitly asks to create/delete a view or tab."
            )
        if extra_lines:
            api_prompt = assist.api_prompt + "\n\nYou ALSO have access to these extra tools:\n" + "\n".join(extra_lines) + "\n"
        else:
            api_prompt = assist.api_prompt

        return llm.APIInstance(
            api=self,
            api_prompt=api_prompt,
            llm_context=llm_context,
            tools=tools,
            custom_serializer=assist.custom_serializer,
        )


@callback
def async_register_ai_subscription_assist_apis(hass: HomeAssistant, entry: ConfigEntry) -> Callable[[], None]:
    """Register per-subentry AI Subscription Assist APIs.

    Returns an unregister callable.
    """
    unregisters: list[Callable[[], None]] = []
    for subentry in entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue
        api = AiSubscriptionAssistSubentryAPI(hass, entry, subentry.subentry_id)
        try:
            unregisters.append(llm.async_register_api(hass, api))
        except Exception as err:
            LOGGER.debug(
                "AI Subscription Assist API registration skipped (%s): %s", api.id, err
            )

    def _unregister_all() -> None:
        for unreg in unregisters:
            try:
                unreg()
            except Exception:
                pass

    return _unregister_all
