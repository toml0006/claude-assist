"""Custom tools for Claude Assist."""

from __future__ import annotations

import json
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from .const import CONF_CHAT_MODEL, LOGGER


class SetModelTool(llm.Tool):
    """Tool to change the active Claude model."""

    name = "set_model"
    description = (
        "Change the Claude model used for this conversation. "
        "Available models: "
        "claude-haiku-4-5 (fastest, cheapest), "
        "claude-sonnet-4-5-20250514 (balanced, good default), "
        "claude-opus-4-5-20250514 (most capable, slowest). "
        "Call this when the user asks to switch models or wants a smarter/faster response."
    )

    parameters = vol.Schema(
        {
            vol.Required(
                "model",
                description="The model ID to switch to",
            ): str,
        }
    )

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize."""
        self._hass = hass
        self._entry = entry

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> str:
        """Change the model."""
        model = tool_input.tool_args["model"]

        for subentry in self._entry.subentries.values():
            if subentry.subentry_type != "conversation":
                continue

            new_data = {**subentry.data, CONF_CHAT_MODEL: model}
            hass.config_entries.async_update_subentry(
                self._entry, subentry, data=new_data
            )
            LOGGER.info("Model changed to %s via tool call", model)
            return json.dumps({
                "success": True,
                "model": model,
                "note": "Model changed. The next message will use the new model.",
            })

        return json.dumps({"success": False, "error": "No conversation subentry found"})


def get_custom_tools(
    hass: HomeAssistant, entry: ConfigEntry
) -> list[llm.Tool]:
    """Get all custom tools for the integration."""
    return [
        SetModelTool(hass, entry),
    ]
