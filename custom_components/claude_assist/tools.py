"""Custom tools for AI Subscription Assist."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers import llm

from .const import CONF_CHAT_MODEL, CONF_YOLO_MODE, LOGGER
from .dashboard_safety import validate_view_change_request
from .internet_lookup import (
    collapse_whitespace,
    extract_page_text,
    parse_bing_rss,
    validate_public_http_url,
)
from .tool_policy import (
    default_enabled_tool_names,
    normalize_enabled_tool_names,
)


def _is_yolo_mode(entry: ConfigEntry, subentry_id: str | None) -> bool:
    """Return whether yolo mode is enabled for this subentry."""
    if subentry_id is None:
        return False
    subentry = entry.subentries.get(subentry_id)
    return bool(subentry and subentry.data.get(CONF_YOLO_MODE, False))


def _is_entity_allowed(
    hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None, entity_id: str
) -> bool:
    """Check if an entity can be accessed by this tool call."""
    return _is_yolo_mode(entry, subentry_id) or _should_expose(hass, entity_id)


def _validate_exposed_entities(
    hass: HomeAssistant,
    entry: ConfigEntry,
    subentry_id: str | None,
    entity_ids: list[str],
) -> list[str]:
    """Return disallowed entity ids for this tool call."""
    if _is_yolo_mode(entry, subentry_id):
        return []
    return [eid for eid in entity_ids if not _should_expose(hass, eid)]


class ClaudeAssistTool(llm.Tool):
    """Base class for custom tools with yolo helpers."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        subentry_id: str | None = None,
    ) -> None:
        self._hass = hass
        self._entry = entry
        self._subentry_id = subentry_id

    def _is_yolo_mode(self) -> bool:
        return _is_yolo_mode(self._entry, self._subentry_id)

    def _is_entity_allowed(self, hass: HomeAssistant, entity_id: str) -> bool:
        return _is_entity_allowed(hass, self._entry, self._subentry_id, entity_id)

    def _validate_exposed_entities(
        self, hass: HomeAssistant, entity_ids: list[str]
    ) -> list[str]:
        return _validate_exposed_entities(hass, self._entry, self._subentry_id, entity_ids)


class SetModelTool(ClaudeAssistTool):
    """Tool to change the active model for the conversation agent."""

    name = "set_model"
    description = (
        "Change the model used for this conversation agent. "
        "Model IDs are provider-specific; you can provide any valid model string. "
        "Examples: claude-sonnet-4-5-20250514, gpt-4o-mini, gpt-5.2-codex, gemini-2.5-pro. "
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

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Change the model."""
        model = tool_input.tool_args["model"]

        subentries = list(self._entry.subentries.values())
        if self._subentry_id:
            scoped = self._entry.subentries.get(self._subentry_id)
            if scoped and scoped.subentry_type == "conversation":
                subentries = [scoped]

        for subentry in subentries:
            if subentry.subentry_type != "conversation":
                continue

            new_data = {**subentry.data, CONF_CHAT_MODEL: model}
            hass.config_entries.async_update_subentry(
                self._entry, subentry, data=new_data
            )
            LOGGER.info("Model changed to %s via tool call", model)
            return {
                "success": True,
                "model": model,
                "note": "Model changed. The next message will use the new model.",
            }

        return {"success": False, "error": "No conversation subentry found"}


def _should_expose(hass: HomeAssistant, entity_id: str) -> bool:
    """Check if an entity is exposed to conversation."""
    try:
        from homeassistant.components.homeassistant import async_should_expose
        from homeassistant.components import conversation
        return async_should_expose(hass, conversation.DOMAIN, entity_id)
    except Exception:
        return True


class GetHistoryTool(ClaudeAssistTool):
    """Tool to retrieve entity state history."""

    name = "get_history"
    description = (
        "Get the state history of one or more Home Assistant entities. "
        "Returns historical state changes with timestamps. "
        "Useful for understanding how entity states changed over time."
    )

    parameters = vol.Schema(
        {
            vol.Required(
                "entity_ids",
                description="List of entity IDs to get history for",
            ): [str],
            vol.Optional(
                "start_time",
                description="Start time in ISO format (default: 24 hours ago)",
            ): str,
            vol.Optional(
                "end_time",
                description="End time in ISO format (default: now)",
            ): str,
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Get entity history."""
        try:
            from homeassistant.components.recorder import get_instance, history as recorder_history
            from homeassistant.util import dt as dt_util

            entity_ids = tool_input.tool_args["entity_ids"]

            not_exposed = self._validate_exposed_entities(hass, entity_ids)
            if not_exposed:
                return {"error": f"Entities not exposed: {not_exposed}"}

            now = dt_util.utcnow()
            start_time_str = tool_input.tool_args.get("start_time")
            end_time_str = tool_input.tool_args.get("end_time")

            if start_time_str:
                start_time = dt_util.parse_datetime(start_time_str)
                if start_time is None:
                    return {"error": f"Invalid start_time: {start_time_str}"}
                start_time = dt_util.as_utc(start_time)
            else:
                start_time = now - timedelta(hours=24)

            if end_time_str:
                end_time = dt_util.parse_datetime(end_time_str)
                if end_time is None:
                    return {"error": f"Invalid end_time: {end_time_str}"}
                end_time = dt_util.as_utc(end_time)
            else:
                end_time = now

            # Clamp future end_time (LLMs sometimes pass future dates)
            if end_time > now:
                end_time = now
            if start_time > end_time:
                start_time = end_time - timedelta(hours=24)

            from homeassistant.components.recorder import util as recorder_util

            # Use the recorder session API (more stable signature across HA versions)
            def _query() -> dict[str, list[Any]]:
                with recorder_util.session_scope(hass=hass, read_only=True) as session:
                    return recorder_history.get_significant_states_with_session(
                        hass=hass,
                        session=session,
                        start_time=start_time,
                        end_time=end_time,
                        entity_ids=entity_ids,
                        filters=None,
                        include_start_time_state=True,
                        significant_changes_only=False,
                        minimal_response=True,
                        no_attributes=False,
                    )

            result = await get_instance(hass).async_add_executor_job(_query)

            formatted: list[dict[str, Any]] = []
            for entity_id, states in result.items():
                formatted.append(
                    {
                        "entity_id": entity_id,
                        "states": [
                            {
                                "state": getattr(s, "state", None),
                                "last_changed": s.last_changed.isoformat() if getattr(s, "last_changed", None) else None,
                                "last_updated": s.last_updated.isoformat() if getattr(s, "last_updated", None) else None,
                                "attributes": dict(getattr(s, "attributes", {}) or {}),
                            }
                            for s in states
                        ],
                    }
                )

            return {"start_time": start_time.isoformat(), "end_time": end_time.isoformat(), "results": formatted}
        except Exception as e:
            LOGGER.error("get_history error: %s", e)
            return {"error": str(e)}


class GetLogbookTool(ClaudeAssistTool):
    """Tool to retrieve logbook entries."""

    name = "get_logbook"
    description = (
        "Get recent logbook entries from Home Assistant. "
        "Returns a log of events including state changes, automations triggered, etc. "
        "Useful for understanding what happened recently."
    )

    parameters = vol.Schema(
        {
            vol.Optional(
                "hours_ago",
                description="How many hours of history to retrieve (default: 24)",
            ): vol.Coerce(float),
            vol.Optional(
                "entity_ids",
                description="Optional list of entity IDs to filter by",
            ): [str],
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Get logbook entries."""
        try:
            from homeassistant.util import dt as dt_util
            from homeassistant.components.logbook.helpers import async_determine_event_types
            from homeassistant.components.logbook.processor import EventProcessor

            hours_ago = tool_input.tool_args.get("hours_ago", 24)
            entity_ids = tool_input.tool_args.get("entity_ids")

            if entity_ids:
                not_exposed = self._validate_exposed_entities(hass, entity_ids)
                if not_exposed:
                    return {"error": f"Entities not exposed: {not_exposed}"}

            now = dt_util.utcnow()
            start_time = now - timedelta(hours=hours_ago)

            from homeassistant.components.recorder import get_instance

            # Use logbook processor (same path as /api/logbook)
            event_types = async_determine_event_types(hass, entity_ids, None)
            processor = EventProcessor(
                hass,
                event_types,
                entity_ids,
                None,
                None,
                timestamp=True,
                include_entity_name=True,
            )

            events = await get_instance(hass).async_add_executor_job(
                processor.get_events,
                start_time,
                now,
            )

            # If logbook has nothing (or logbook is filtered), fall back to recorder states
            if not events:
                from homeassistant.components.recorder import history as recorder_history
                from homeassistant.components.recorder import util as recorder_util

                def _query() -> dict[str, list[Any]]:
                    with recorder_util.session_scope(hass=hass, read_only=True) as session:
                        return recorder_history.get_significant_states_with_session(
                            hass=hass,
                            session=session,
                            start_time=start_time,
                            end_time=now,
                            entity_ids=entity_ids,
                            filters=None,
                            include_start_time_state=True,
                            significant_changes_only=False,
                            minimal_response=True,
                            no_attributes=True,
                        )

                result = await get_instance(hass).async_add_executor_job(_query)

                entries: list[dict[str, Any]] = []
                for eid, states in result.items():
                    for s in states[-50:]:
                        entries.append(
                            {
                                "entity_id": eid,
                                "state": getattr(s, "state", None),
                                "when": s.last_changed.isoformat() if getattr(s, "last_changed", None) else None,
                            }
                        )

                entries.sort(key=lambda x: x.get("when") or "", reverse=True)
                return {"entries": entries[:100], "note": "logbook empty; returned recorder history"}

            return {"events": events[:200]}
        except Exception as e:
            LOGGER.error("get_logbook error: %s", e)
            return {"error": str(e)}


class RenderTemplateTool(ClaudeAssistTool):
    """Tool to evaluate Jinja2 templates."""

    name = "render_template"
    description = (
        "Evaluate a Home Assistant Jinja2 template and return the result. "
        "Useful for complex queries, calculations, or accessing state attributes. "
        "Example: '{{ states(\"sensor.temperature\") }}' or "
        "'{{ state_attr(\"climate.living_room\", \"current_temperature\") }}'"
    )

    parameters = vol.Schema(
        {
            vol.Required(
                "template",
                description="Jinja2 template string to evaluate",
            ): str,
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Render a template."""
        try:
            from homeassistant.helpers.template import Template

            template_str = tool_input.tool_args["template"]
            tpl = Template(template_str, hass)
            tpl.hass = hass
            result = tpl.async_render(parse_result=False)
            return {"result": str(result)}
        except Exception as e:
            LOGGER.error("render_template error: %s", e)
            return {"error": str(e)}


class InternetLookupTool(ClaudeAssistTool):
    """Tool to search/fetch public internet content in read-only mode."""

    name = "internet_lookup"
    description = (
        "Read-only internet lookup. "
        "Use action 'search' to find public web results for a query, "
        "or action 'fetch' to read a specific public URL. "
        "This tool cannot write or execute anything."
    )

    parameters = vol.Schema(
        {
            vol.Required(
                "action",
                description="Action: search or fetch",
            ): vol.In(["search", "fetch"]),
            vol.Optional(
                "query",
                description="Search query (required for search action)",
            ): str,
            vol.Optional(
                "url",
                description="Public http/https URL to fetch (required for fetch action)",
            ): str,
            vol.Optional(
                "limit",
                description="Maximum number of search results (1-10, default: 5)",
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
            vol.Optional(
                "max_chars",
                description="Maximum characters of fetched page text (500-20000, default: 6000)",
            ): vol.All(vol.Coerce(int), vol.Range(min=500, max=20000)),
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Search/fetch internet content."""
        try:
            args = tool_input.tool_args
            action = args["action"]
            client = get_async_client(hass)
            headers = {
                "User-Agent": "ai-subscription-assist/1.0",
                "Accept": "text/html,application/xhtml+xml,text/plain,application/json,application/xml;q=0.9,*/*;q=0.1",
            }

            if action == "search":
                query = str(args.get("query", "")).strip()
                if not query:
                    return {"error": "query is required for search"}
                limit = int(args.get("limit", 5))
                # Keep queries bounded and deterministic.
                query = query[:256]

                resp = await client.get(
                    "https://www.bing.com/search",
                    params={"q": query, "format": "rss"},
                    headers=headers,
                    timeout=15.0,
                )
                resp.raise_for_status()
                results = parse_bing_rss(resp.text, limit=limit)
                return {
                    "query": query,
                    "source": "bing_rss",
                    "result_count": len(results),
                    "results": results,
                }

            if action == "fetch":
                url = str(args.get("url", "")).strip()
                if not url:
                    return {"error": "url is required for fetch"}
                max_chars = int(args.get("max_chars", 6000))
                url_error = validate_public_http_url(url)
                if url_error:
                    return {"error": url_error}

                resp = await client.get(
                    url,
                    headers=headers,
                    follow_redirects=True,
                    timeout=20.0,
                )
                resp.raise_for_status()

                final_url = str(resp.url)
                final_url_error = validate_public_http_url(final_url)
                if final_url_error:
                    return {"error": f"Redirect target rejected: {final_url_error}"}

                content_type = str(resp.headers.get("content-type", "")).lower()
                if "text/html" in content_type or "application/xhtml+xml" in content_type:
                    page = extract_page_text(resp.text, max_chars=max_chars)
                    return {
                        "url": url,
                        "final_url": final_url,
                        "content_type": content_type,
                        "title": page["title"],
                        "content": page["text"],
                        "truncated": page["truncated"],
                        "total_chars": page["total_chars"],
                    }

                if "json" in content_type or content_type.startswith("text/") or not content_type:
                    text = resp.text if "json" in content_type else collapse_whitespace(resp.text)
                    truncated = len(text) > max_chars
                    return {
                        "url": url,
                        "final_url": final_url,
                        "content_type": content_type,
                        "content": text[:max_chars],
                        "truncated": truncated,
                        "total_chars": len(text),
                    }

                return {
                    "error": (
                        "Unsupported content type for fetch. "
                        "Only text/html, text/*, and json responses are supported."
                    ),
                    "content_type": content_type,
                    "url": final_url,
                }

            return {"error": f"Unknown action: {action}"}
        except Exception as e:
            LOGGER.error("internet_lookup error: %s", e)
            return {"error": str(e)}


class GetStatisticsTool(ClaudeAssistTool):
    """Tool to get recorder statistics."""

    name = "get_statistics"
    description = (
        "Get statistical data for sensors over time (mean, min, max, sum, etc.). "
        "Useful for energy consumption, temperature trends, and other numeric sensor data. "
        "statistic_ids are usually the same as entity_ids for recorder-tracked entities."
    )

    parameters = vol.Schema(
        {
            vol.Required(
                "statistic_ids",
                description="List of statistic IDs (usually entity IDs) to query",
            ): [str],
            vol.Optional(
                "start_time",
                description="Start time in ISO format (default: 24 hours ago)",
            ): str,
            vol.Optional(
                "end_time",
                description="End time in ISO format (default: now)",
            ): str,
            vol.Optional(
                "period",
                description="Aggregation period: 5minute, hour, day, week, or month (default: hour)",
            ): vol.In(["5minute", "hour", "day", "week", "month"]),
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Get statistics."""
        try:
            from homeassistant.components.recorder import get_instance
            from homeassistant.components.recorder.statistics import statistics_during_period
            from homeassistant.util import dt as dt_util

            args = tool_input.tool_args
            statistic_ids = args["statistic_ids"]
            period = args.get("period", "hour")
            entity_like_stat_ids = [sid for sid in statistic_ids if "." in sid]
            not_exposed = self._validate_exposed_entities(hass, entity_like_stat_ids)
            if not_exposed:
                return {"error": f"Entities not exposed: {not_exposed}"}

            now = dt_util.utcnow()
            start_time_str = args.get("start_time")
            end_time_str = args.get("end_time")

            if start_time_str:
                start_time = dt_util.parse_datetime(start_time_str)
                if start_time is None:
                    return {"error": f"Invalid start_time: {start_time_str}"}
                start_time = dt_util.as_utc(start_time)
            else:
                start_time = now - timedelta(hours=24)

            if end_time_str:
                end_time = dt_util.parse_datetime(end_time_str)
                if end_time is None:
                    return {"error": f"Invalid end_time: {end_time_str}"}
                end_time = dt_util.as_utc(end_time)
            else:
                end_time = now

            result = await get_instance(hass).async_add_executor_job(
                statistics_during_period,
                hass,
                start_time,
                end_time,
                statistic_ids,
                period,
                None,  # units
                {"change", "last_reset", "max", "mean", "min", "state", "sum"},
            )

            return {"result": result}
        except Exception as e:
            LOGGER.error("get_statistics error: %s", e)
            return {"error": str(e)}


class ListAutomationsTool(ClaudeAssistTool):
    """Tool to list all automations."""

    name = "list_automations"
    description = (
        "List all automations in Home Assistant with their current state (on/off) "
        "and last triggered time."
    )

    parameters = vol.Schema({})

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """List automations."""
        try:
            automations = []
            for state in hass.states.async_all("automation"):
                if not self._is_entity_allowed(hass, state.entity_id):
                    continue
                automations.append({
                    "entity_id": state.entity_id,
                    "friendly_name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "last_triggered": state.attributes.get("last_triggered"),
                    "id": state.attributes.get("id"),
                })
            return {"automations": automations}
        except Exception as e:
            LOGGER.error("list_automations error: %s", e)
            return {"error": str(e)}


class ToggleAutomationTool(ClaudeAssistTool):
    """Tool to enable/disable an automation."""

    name = "toggle_automation"
    description = (
        "Enable or disable a Home Assistant automation by entity_id. "
        "Use action 'turn_on' to enable, 'turn_off' to disable, or 'toggle' to switch."
    )

    parameters = vol.Schema(
        {
            vol.Required(
                "entity_id",
                description="The automation entity_id (e.g., automation.my_automation)",
            ): str,
            vol.Optional(
                "action",
                description="Action: turn_on, turn_off, or toggle (default: toggle)",
            ): vol.In(["turn_on", "turn_off", "toggle"]),
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Toggle automation."""
        try:
            entity_id = tool_input.tool_args["entity_id"]
            action = tool_input.tool_args.get("action", "toggle")

            if not entity_id.startswith("automation."):
                return {"error": "entity_id must be an automation entity"}

            state = hass.states.get(entity_id)
            if state is None:
                return {"error": f"Automation {entity_id} not found"}
            if not self._is_entity_allowed(hass, entity_id):
                return {"error": f"Entity {entity_id} is not exposed"}

            await hass.services.async_call(
                "automation", action, {"entity_id": entity_id}, blocking=True
            )

            new_state = hass.states.get(entity_id)
            return {
                "success": True,
                "entity_id": entity_id,
                "previous_state": state.state,
                "new_state": new_state.state if new_state else "unknown",
            }
        except Exception as e:
            LOGGER.error("toggle_automation error: %s", e)
            return {"error": str(e)}


class AddAutomationTool(ClaudeAssistTool):
    """Tool to create a new automation."""

    name = "add_automation"
    description = (
        "Create a new Home Assistant automation from a YAML configuration string. "
        "The automation will be written to automations.yaml and reloaded. "
        "Provide the full automation config as a YAML string."
    )

    parameters = vol.Schema(
        {
            vol.Required(
                "automation_config",
                description="YAML string defining the automation (alias, trigger, condition, action)",
            ): str,
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Add automation."""
        try:
            import os
            import time
            import yaml as pyyaml

            if not self._is_yolo_mode():
                return {
                    "error": "add_automation requires yolo mode enabled for this agent"
                }

            automation_config_str = tool_input.tool_args["automation_config"]
            parsed = pyyaml.safe_load(automation_config_str)

            if isinstance(parsed, list):
                config = parsed[0] if parsed else {}
            elif isinstance(parsed, dict):
                config = parsed
            else:
                return {"error": "Invalid automation config format"}

            if "id" not in config:
                config["id"] = str(round(time.time() * 1000))

            required_fields = {"alias", "trigger", "action"}
            missing = sorted(field for field in required_fields if field not in config)
            if missing:
                return {
                    "error": f"Missing required automation fields: {', '.join(missing)}"
                }

            automations_path = os.path.join(hass.config.config_dir, "automations.yaml")

            def _write_automation() -> str:
                existing_text = ""
                existing_items: list[dict[str, Any]] = []
                if os.path.exists(automations_path):
                    with open(automations_path, "r", encoding="utf-8") as f:
                        existing_text = f.read()
                    loaded = pyyaml.safe_load(existing_text) if existing_text.strip() else []
                    if loaded is None:
                        loaded = []
                    if not isinstance(loaded, list):
                        raise ValueError("automations.yaml must contain a list of automations")
                    existing_items = loaded

                updated = [*existing_items, config]
                temp_path = f"{automations_path}.tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    pyyaml.dump(updated, f, allow_unicode=True, sort_keys=False)
                os.replace(temp_path, automations_path)
                return existing_text

            previous_text = await hass.async_add_executor_job(_write_automation)
            try:
                await hass.services.async_call("automation", "reload", blocking=True)
            except Exception:
                def _restore() -> None:
                    with open(automations_path, "w", encoding="utf-8") as f:
                        f.write(previous_text)

                await hass.async_add_executor_job(_restore)
                raise

            return {
                "success": True,
                "automation_id": config["id"],
                "alias": config.get("alias", "unnamed"),
            }
        except Exception as e:
            LOGGER.error("add_automation error: %s", e)
            return {"error": str(e)}


class ModifyDashboardTool(ClaudeAssistTool):
    """Tool to read/modify Lovelace dashboard config via official APIs."""

    name = "modify_dashboard"
    description = (
        "Read or modify a Lovelace dashboard configuration. "
        "Actions: 'list' to list all dashboards, 'get' to read current config, "
        "'add_card' to add a card to an existing view (or section), "
        "'remove_card' to remove a card from a view/section, "
        "'add_view' to add a new view (requires explicit confirm), "
        "'remove_view' to delete an existing view (requires explicit confirm). "
        "Prefer editing existing views first; call 'get' before write actions. "
        "Only storage-mode dashboards can be modified; YAML-mode dashboards are read-only."
    )

    parameters = vol.Schema(
        {
            vol.Required(
                "action",
                description="Action: list, get, add_card, remove_card, add_view, remove_view",
            ): vol.In(["list", "get", "add_card", "remove_card", "add_view", "remove_view"]),
            vol.Optional(
                "url_path",
                description="Dashboard url_path (default: None for the default dashboard). "
                "Use the url_path from 'list' action results.",
            ): vol.Any(str, None),
            vol.Optional(
                "view_index",
                description="View index (0-based) for operations. Prefer this or view_title.",
            ): int,
            vol.Optional(
                "view_title",
                description="View title (case-insensitive) to target. Used when view_index is omitted.",
            ): str,
            vol.Optional(
                "section_index",
                description="Section index (0-based) for section-based dashboards.",
            ): int,
            vol.Optional(
                "section_title",
                description="Section title (case-insensitive) for section-based dashboards.",
            ): str,
            vol.Optional(
                "card_config",
                description="JSON string of the card configuration to add",
            ): str,
            vol.Optional(
                "card_index",
                description="Card index to remove (0-based)",
            ): int,
            vol.Optional(
                "view_config",
                description="JSON string of the view configuration to add",
            ): str,
            vol.Optional(
                "confirm",
                description="Required for add_view/remove_view to prevent accidental structure changes.",
            ): bool,
            vol.Optional(
                "user_request",
                description=(
                    "Required for add_view/remove_view. Quote the user's explicit "
                    "request to create/delete a view or tab."
                ),
            ): str,
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    def _get_lovelace_data(self, hass: HomeAssistant) -> Any:
        """Get the LovelaceData object from hass.data."""
        try:
            from homeassistant.components.lovelace.const import LOVELACE_DATA
            return hass.data.get(LOVELACE_DATA)
        except (ImportError, KeyError):
            return None

    def _get_dashboard(self, hass: HomeAssistant, url_path: str | None) -> Any:
        """Look up a dashboard config object."""
        ll_data = self._get_lovelace_data(hass)
        if ll_data is None:
            return None
        dashboards = ll_data.dashboards
        # For default dashboard, try 'lovelace' key then None key
        if url_path is None:
            return dashboards.get("lovelace") or dashboards.get(None)
        return dashboards.get(url_path)

    def _summarize_views(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Return compact view metadata for LLM guidance."""
        views = config.get("views", [])
        if not isinstance(views, list):
            return []
        summary: list[dict[str, Any]] = []
        for idx, view in enumerate(views):
            if not isinstance(view, dict):
                continue
            cards = view.get("cards", [])
            sections = view.get("sections", [])
            section_summary: list[dict[str, Any]] = []
            if isinstance(sections, list):
                for s_idx, section in enumerate(sections):
                    if not isinstance(section, dict):
                        continue
                    s_cards = section.get("cards", [])
                    section_summary.append(
                        {
                            "index": s_idx,
                            "title": section.get("title", ""),
                            "card_count": len(s_cards) if isinstance(s_cards, list) else 0,
                        }
                    )
            summary.append(
                {
                    "index": idx,
                    "title": view.get("title", f"View {idx + 1}"),
                    "path": view.get("path", ""),
                    "card_count": len(cards) if isinstance(cards, list) else 0,
                    "section_count": len(sections) if isinstance(sections, list) else 0,
                    "sections": section_summary,
                }
            )
        return summary

    def _resolve_view_index(
        self, views: list[Any], args: dict[str, Any]
    ) -> tuple[int | None, str | None]:
        """Resolve target view index using index or title."""
        if not views:
            return None, "Dashboard has no views"

        if isinstance(args.get("view_index"), int):
            view_index = args["view_index"]
            if view_index < 0 or view_index >= len(views):
                return None, f"View index {view_index} out of range (have {len(views)} views)"
            return view_index, None

        view_title = args.get("view_title")
        if isinstance(view_title, str) and view_title.strip():
            needle = view_title.strip().lower()
            for idx, view in enumerate(views):
                if not isinstance(view, dict):
                    continue
                title = str(view.get("title", "")).strip().lower()
                if title == needle:
                    return idx, None
            return None, f"View title '{view_title}' not found"

        return 0, None

    def _resolve_section_index(
        self, view: dict[str, Any], args: dict[str, Any]
    ) -> tuple[int | None, str | None]:
        """Resolve section index for a section-based view."""
        sections = view.get("sections", [])
        if not isinstance(sections, list) or not sections:
            return None, "View has no sections"

        if isinstance(args.get("section_index"), int):
            section_index = args["section_index"]
            if section_index < 0 or section_index >= len(sections):
                return None, (
                    f"Section index {section_index} out of range "
                    f"(have {len(sections)} sections)"
                )
            return section_index, None

        section_title = args.get("section_title")
        if isinstance(section_title, str) and section_title.strip():
            needle = section_title.strip().lower()
            for idx, section in enumerate(sections):
                if not isinstance(section, dict):
                    continue
                title = str(section.get("title", "")).strip().lower()
                if title == needle:
                    return idx, None
            return None, f"Section title '{section_title}' not found"

        return 0, None

    def _extract_user_request_text(self, llm_context: llm.LLMContext) -> str:
        """Best-effort extraction of user-authored text from llm context."""
        fields = ("user_prompt", "prompt", "input", "text", "query")
        parts: list[str] = []

        def _append(value: Any) -> None:
            if isinstance(value, str):
                text = value.strip()
                if text and text not in parts:
                    parts.append(text)

        for field in fields:
            _append(getattr(llm_context, field, None))

        context_obj = getattr(llm_context, "context", None)
        if isinstance(context_obj, dict):
            for field in fields:
                _append(context_obj.get(field))
        elif context_obj is not None:
            for field in fields:
                _append(getattr(context_obj, field, None))

        return "\n".join(parts)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Modify dashboard."""
        try:
            from homeassistant.components.lovelace.const import (
                MODE_STORAGE,
                ConfigNotFound,
            )

            args = tool_input.tool_args
            action = args["action"]
            url_path = args.get("url_path")
            write_actions = {"add_card", "remove_card", "add_view", "remove_view"}
            if action in write_actions and not self._is_yolo_mode():
                return {
                    "error": "modify_dashboard write actions require yolo mode enabled for this agent"
                }

            if action == "list":
                ll_data = self._get_lovelace_data(hass)
                if ll_data is None:
                    return {"error": "Lovelace integration not loaded"}
                result = []
                for key, dash in ll_data.dashboards.items():
                    info: dict[str, Any] = {
                        "url_path": key,
                        "mode": dash.mode,
                    }
                    if dash.config:
                        info["title"] = dash.config.get("title", "")
                        info["icon"] = dash.config.get("icon", "")
                    result.append(info)
                return {"dashboards": result}

            dashboard = self._get_dashboard(hass, url_path)
            if dashboard is None:
                return {"error": f"Dashboard '{url_path}' not found"}

            if action == "get":
                try:
                    config = await dashboard.async_load(False)
                except ConfigNotFound:
                    return {"result": None, "note": "No config found (auto-generated dashboard)"}
                return {
                    "result": config,
                    "mode": dashboard.mode,
                    "views": self._summarize_views(config),
                }

            # Write operations require storage mode
            if dashboard.mode != MODE_STORAGE:
                return {
                    "error": f"Dashboard '{url_path}' is in {dashboard.mode} mode and cannot be modified. "
                    "Only storage-mode dashboards support edits."
                }

            try:
                config = await dashboard.async_load(False)
            except ConfigNotFound:
                config = {"views": []}

            if action == "add_card":
                card_config_str = args.get("card_config")
                if not card_config_str:
                    return {"error": "card_config is required"}

                card = json.loads(card_config_str)
                views = config.get("views", [])
                if not isinstance(views, list):
                    return {"error": "Dashboard views format is invalid"}
                view_index, view_error = self._resolve_view_index(views, args)
                if view_error:
                    return {"error": view_error, "views": self._summarize_views(config)}
                assert view_index is not None
                view = views[view_index]
                if not isinstance(view, dict):
                    return {"error": f"View at index {view_index} is invalid"}

                if isinstance(view.get("sections"), list) and view["sections"]:
                    section_index, section_error = self._resolve_section_index(view, args)
                    if section_error:
                        return {
                            "error": section_error,
                            "views": self._summarize_views(config),
                        }
                    assert section_index is not None
                    section = view["sections"][section_index]
                    if not isinstance(section, dict):
                        return {"error": f"Section at index {section_index} is invalid"}
                    if "cards" not in section or not isinstance(section.get("cards"), list):
                        section["cards"] = []
                    section["cards"].append(card)
                    target: dict[str, Any] = {
                        "view_index": view_index,
                        "section_index": section_index,
                    }
                else:
                    if "cards" not in view or not isinstance(view.get("cards"), list):
                        view["cards"] = []
                    view["cards"].append(card)
                    target = {"view_index": view_index}

                await dashboard.async_save(config)
                return {
                    "success": True,
                    "action": "card_added",
                    **target,
                    "views": self._summarize_views(config),
                }

            elif action == "remove_card":
                card_index = args.get("card_index")
                if card_index is None:
                    return {"error": "card_index is required"}

                views = config.get("views", [])
                if not isinstance(views, list):
                    return {"error": "Dashboard views format is invalid"}
                view_index, view_error = self._resolve_view_index(views, args)
                if view_error:
                    return {"error": view_error, "views": self._summarize_views(config)}
                assert view_index is not None
                view = views[view_index]
                if not isinstance(view, dict):
                    return {"error": f"View at index {view_index} is invalid"}

                if isinstance(view.get("sections"), list) and view["sections"]:
                    section_index, section_error = self._resolve_section_index(view, args)
                    if section_error:
                        return {
                            "error": section_error,
                            "views": self._summarize_views(config),
                        }
                    assert section_index is not None
                    section = view["sections"][section_index]
                    if not isinstance(section, dict):
                        return {"error": f"Section at index {section_index} is invalid"}
                    cards = section.get("cards", [])
                    if not isinstance(cards, list):
                        return {"error": f"Section {section_index} has no cards list"}
                    if card_index < 0 or card_index >= len(cards):
                        return {
                            "error": (
                                f"Card index {card_index} out of range "
                                f"(have {len(cards)} cards in section {section_index})"
                            )
                        }
                    removed = cards.pop(card_index)
                    target = {"view_index": view_index, "section_index": section_index}
                else:
                    cards = view.get("cards", [])
                    if not isinstance(cards, list):
                        return {"error": f"View {view_index} has no cards list"}
                    if card_index < 0 or card_index >= len(cards):
                        return {
                            "error": (
                                f"Card index {card_index} out of range "
                                f"(have {len(cards)} cards)"
                            )
                        }
                    removed = cards.pop(card_index)
                    target = {"view_index": view_index}

                await dashboard.async_save(config)
                return {
                    "success": True,
                    "action": "card_removed",
                    **target,
                    "removed_card": removed,
                    "views": self._summarize_views(config),
                }

            elif action == "add_view":
                if not bool(args.get("confirm", False)):
                    return {
                        "error": (
                            "add_view is blocked unless confirm=true. "
                            "Only add a view when the user explicitly asks for a new view."
                        ),
                        "views": self._summarize_views(config),
                    }
                intent_error = validate_view_change_request(
                    action="add_view",
                    user_request=args.get("user_request"),
                    context_text=self._extract_user_request_text(llm_context),
                )
                if intent_error:
                    return {"error": intent_error, "views": self._summarize_views(config)}
                view_config_str = args.get("view_config")
                if not view_config_str:
                    return {"error": "view_config is required"}

                view = json.loads(view_config_str)
                if "views" not in config:
                    config["views"] = []
                config["views"].append(view)

                await dashboard.async_save(config)
                return {
                    "success": True,
                    "action": "view_added",
                    "view_count": len(config["views"]),
                    "views": self._summarize_views(config),
                }

            elif action == "remove_view":
                if not bool(args.get("confirm", False)):
                    return {
                        "error": (
                            "remove_view is blocked unless confirm=true."
                        ),
                        "views": self._summarize_views(config),
                    }
                intent_error = validate_view_change_request(
                    action="remove_view",
                    user_request=args.get("user_request"),
                    context_text=self._extract_user_request_text(llm_context),
                )
                if intent_error:
                    return {"error": intent_error, "views": self._summarize_views(config)}
                views = config.get("views", [])
                if not isinstance(views, list):
                    return {"error": "Dashboard views format is invalid"}
                view_index, view_error = self._resolve_view_index(views, args)
                if view_error:
                    return {"error": view_error, "views": self._summarize_views(config)}
                assert view_index is not None
                removed_view = views.pop(view_index)
                await dashboard.async_save(config)
                return {
                    "success": True,
                    "action": "view_removed",
                    "view_index": view_index,
                    "removed_view": removed_view,
                    "view_count": len(views),
                    "views": self._summarize_views(config),
                }

            return {"error": f"Unknown action: {action}"}
        except Exception as e:
            LOGGER.error("modify_dashboard error: %s", e)
            return {"error": str(e)}


class CallServiceTool(ClaudeAssistTool):
    """Tool to call arbitrary Home Assistant services."""

    name = "call_service"
    description = (
        "Call any Home Assistant service with JSON service_data. "
        "This is powerful and potentially destructive; intended for YOLO mode only "
        "(for advanced entity CRUD and helper management)."
    )

    parameters = vol.Schema(
        {
            vol.Required("domain", description="Service domain, e.g. 'input_boolean'"): str,
            vol.Required("service", description="Service name, e.g. 'create'"): str,
            vol.Optional(
                "service_data",
                description="Service data JSON object or plain object",
            ): vol.Any(str, dict),
            vol.Optional(
                "return_response",
                description="Whether to return the service response payload",
            ): bool,
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Call a Home Assistant service."""
        try:
            if not self._is_yolo_mode():
                return {"error": "call_service requires yolo mode enabled for this agent"}

            args = tool_input.tool_args
            domain = args["domain"]
            service = args["service"]
            service_data_input = args.get("service_data", {})
            return_response = bool(args.get("return_response", False))

            if isinstance(service_data_input, str):
                try:
                    service_data = json.loads(service_data_input)
                except json.JSONDecodeError as err:
                    return {"error": f"Invalid service_data JSON: {err}"}
            else:
                service_data = service_data_input

            if not isinstance(service_data, dict):
                return {"error": "service_data must be a JSON object"}

            if not hass.services.has_service(domain, service):
                return {"error": f"Service '{domain}.{service}' not found"}

            response = await hass.services.async_call(
                domain,
                service,
                service_data,
                blocking=True,
                return_response=return_response,
            )
            result: dict[str, Any] = {
                "success": True,
                "service": f"{domain}.{service}",
            }
            if return_response:
                result["response"] = response
            return result
        except Exception as e:
            LOGGER.error("call_service error: %s", e)
            return {"error": str(e)}


class SendNotificationTool(ClaudeAssistTool):
    """Tool to send notifications."""

    name = "send_notification"
    description = (
        "Send a notification via Home Assistant's notify service. "
        "Can send to mobile devices, persistent notifications, etc."
    )

    parameters = vol.Schema(
        {
            vol.Required(
                "message",
                description="Notification message text",
            ): str,
            vol.Optional(
                "title",
                description="Notification title",
            ): str,
            vol.Optional(
                "target",
                description="Notify service name (e.g., 'notify.mobile_app_phone'). Default: 'notify.notify'",
            ): str,
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Send notification."""
        try:
            args = tool_input.tool_args
            message = args["message"]
            title = args.get("title")
            target = args.get("target", "notify.notify")

            # Parse domain.service from target
            parts = target.split(".", 1)
            if len(parts) == 2:
                domain, service = parts
            else:
                domain, service = "notify", "notify"

            # Validate the service exists
            if not hass.services.has_service(domain, service):
                available = [
                    f"notify.{s}" for s in hass.services.async_services().get("notify", {})
                ]
                return {
                    "error": f"Service '{domain}.{service}' not found",
                    "available_notify_services": available,
                }

            service_data: dict[str, Any] = {"message": message}
            if title:
                service_data["title"] = title

            await hass.services.async_call(domain, service, service_data, blocking=True)

            return {"success": True, "target": target, "message": message}
        except Exception as e:
            LOGGER.error("send_notification error: %s", e)
            return {"error": str(e)}


class GetErrorLogTool(ClaudeAssistTool):
    """Tool to get HA error log."""

    name = "get_error_log"
    description = (
        "Get recent Home Assistant error log entries. "
        "Useful for debugging issues with automations, integrations, etc."
    )

    parameters = vol.Schema(
        {
            vol.Optional(
                "lines",
                description="Number of lines to return from the end of the log (default: 50)",
            ): int,
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Get error log."""
        try:
            if not self._is_yolo_mode():
                return {"error": "get_error_log requires yolo mode enabled for this agent"}
            lines = tool_input.tool_args.get("lines", 50)
            # Use hass.config.path() for safe, canonical log path
            log_path = hass.config.path("home-assistant.log")

            def _read_log() -> dict:
                import os
                if not os.path.exists(log_path):
                    return {"error": "Log file not found", "path": log_path}
                with open(log_path, "r", errors="replace") as f:
                    all_lines = f.readlines()
                    tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
                    return {"log": "".join(tail), "total_lines": len(all_lines), "returned_lines": len(tail)}

            return await hass.async_add_executor_job(_read_log)
        except Exception as e:
            LOGGER.error("get_error_log error: %s", e)
            return {"error": str(e)}


class WhoIsHomeTool(ClaudeAssistTool):
    """Tool to check who is home."""

    name = "who_is_home"
    description = (
        "Check which people are home or away based on person and device_tracker entities. "
        "Returns the state of all person entities."
    )

    parameters = vol.Schema({})

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Check who is home."""
        try:
            people = []
            for state in hass.states.async_all("person"):
                if not self._is_entity_allowed(hass, state.entity_id):
                    continue
                people.append({
                    "entity_id": state.entity_id,
                    "name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "source": state.attributes.get("source"),
                })

            # Also check device_trackers not linked to persons
            trackers = []
            for state in hass.states.async_all("device_tracker"):
                if self._is_entity_allowed(hass, state.entity_id):
                    trackers.append({
                        "entity_id": state.entity_id,
                        "name": state.attributes.get("friendly_name", state.entity_id),
                        "state": state.state,
                    })

            return {"persons": people, "device_trackers": trackers}
        except Exception as e:
            LOGGER.error("who_is_home error: %s", e)
            return {"error": str(e)}


class ManageListTool(ClaudeAssistTool):
    """Tool to manage shopping/todo lists."""

    name = "manage_list"
    description = (
        "Add, complete, or get items from todo lists (including shopping lists). "
        "Uses the todo domain services. If no entity_id is provided, "
        "auto-detects the shopping list entity."
    )

    parameters = vol.Schema(
        {
            vol.Required(
                "action",
                description="Action: add, complete, get_items",
            ): vol.In(["add", "complete", "get_items"]),
            vol.Optional(
                "item",
                description="Item name to add or complete",
            ): str,
            vol.Optional(
                "entity_id",
                description="Todo list entity_id (for todo domain). If not provided, uses shopping_list.",
            ): str,
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Manage list."""
        try:
            args = tool_input.tool_args
            action = args["action"]
            item = args.get("item")
            entity_id = args.get("entity_id")

            # Resolve entity_id: if none provided, try to find todo.shopping_list
            if not entity_id:
                # Look for the shopping_list todo entity
                shopping_entity = hass.states.get("todo.shopping_list")
                if shopping_entity:
                    entity_id = "todo.shopping_list"
                else:
                    # Fall back: find any todo entity with "shopping" in name
                    for state in hass.states.async_all("todo"):
                        if "shopping" in state.entity_id.lower():
                            entity_id = state.entity_id
                            break

            if not entity_id:
                return {"error": "No todo/shopping list entity found. Provide entity_id."}

            # Validate entity exists
            if not hass.states.get(entity_id):
                return {"error": f"Entity {entity_id} not found"}
            if not self._is_entity_allowed(hass, entity_id):
                return {"error": f"Entity {entity_id} is not exposed"}

            if action == "add":
                if not item:
                    return {"error": "item is required for add"}
                await hass.services.async_call(
                    "todo", "add_item",
                    {"entity_id": entity_id, "item": item},
                    blocking=True,
                )
                return {"success": True, "action": "added", "item": item, "entity_id": entity_id}

            elif action == "complete":
                if not item:
                    return {"error": "item is required for complete"}
                await hass.services.async_call(
                    "todo", "update_item",
                    {"entity_id": entity_id, "item": item, "status": "completed"},
                    blocking=True,
                )
                return {"success": True, "action": "completed", "item": item, "entity_id": entity_id}

            elif action == "get_items":
                result = await hass.services.async_call(
                    "todo", "get_items",
                    {"entity_id": entity_id, "status": ["needs_action", "completed"]},
                    blocking=True,
                    return_response=True,
                )
                # return_response gives {entity_id: {"items": [...]}}
                if result and entity_id in result:
                    return {"entity_id": entity_id, "items": result[entity_id].get("items", [])}
                return {"entity_id": entity_id, "items": []}

            return {"error": f"Unknown action: {action}"}
        except Exception as e:
            LOGGER.error("manage_list error: %s", e)
            return {"error": str(e)}


class GetCalendarEventsTool(ClaudeAssistTool):
    """Tool to get calendar events."""

    name = "get_calendar_events"
    description = (
        "Get upcoming calendar events from Home Assistant calendar entities. "
        "Returns events within the specified time range."
    )

    parameters = vol.Schema(
        {
            vol.Required(
                "entity_id",
                description="Calendar entity ID (e.g., calendar.personal)",
            ): str,
            vol.Optional(
                "hours_ahead",
                description="How many hours ahead to look (default: 24)",
            ): vol.Coerce(float),
        }
    )

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, subentry_id: str | None = None
    ) -> None:
        """Initialize."""
        super().__init__(hass, entry, subentry_id)

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Get calendar events."""
        try:
            from homeassistant.util import dt as dt_util

            args = tool_input.tool_args
            entity_id = args["entity_id"]
            hours_ahead = min(args.get("hours_ahead", 24), 720)  # Clamp to 30 days max

            if not entity_id.startswith("calendar."):
                return {"error": "entity_id must be a calendar entity"}

            state = hass.states.get(entity_id)
            if state is None:
                return {"error": f"Calendar entity {entity_id} not found"}

            if not self._is_entity_allowed(hass, entity_id):
                return {"error": f"Entity {entity_id} is not exposed"}

            now = dt_util.now()
            end = now + timedelta(hours=hours_ahead)

            result = await hass.services.async_call(
                "calendar", "get_events",
                {
                    "entity_id": entity_id,
                    "start_date_time": now.isoformat(),
                    "end_date_time": end.isoformat(),
                },
                blocking=True,
                return_response=True,
            )

            if result and entity_id in result:
                events = result[entity_id].get("events", [])
                return {"entity_id": entity_id, "events": events}

            return {"entity_id": entity_id, "events": []}
        except Exception as e:
            LOGGER.error("get_calendar_events error: %s", e)
            return {"error": str(e)}


CUSTOM_TOOL_FACTORIES: dict[str, tuple[str, type[llm.Tool]]] = {
    # tool_name: (label, class)
    "set_model": ("Set model", SetModelTool),
    "internet_lookup": ("Internet lookup (read-only)", InternetLookupTool),
    "get_history": ("Get history (recorder)", GetHistoryTool),
    "get_logbook": ("Get logbook", GetLogbookTool),
    "render_template": ("Render template", RenderTemplateTool),
    "get_statistics": ("Get statistics (recorder)", GetStatisticsTool),
    "list_automations": ("List automations", ListAutomationsTool),
    "toggle_automation": ("Toggle automation", ToggleAutomationTool),
    "add_automation": ("Add automation", AddAutomationTool),
    "modify_dashboard": ("Modify dashboard (Lovelace)", ModifyDashboardTool),
    "call_service": ("Call service (YOLO)", CallServiceTool),
    "send_notification": ("Send notification", SendNotificationTool),
    "get_error_log": ("Get error log", GetErrorLogTool),
    "who_is_home": ("Who is home", WhoIsHomeTool),
    "manage_list": ("Manage lists (shopping/todo)", ManageListTool),
    "get_calendar_events": ("Get calendar events", GetCalendarEventsTool),
}


def get_custom_tool_options() -> list[dict[str, str]]:
    """Return tool options for config UI."""
    return [
        {"value": name, "label": label}
        for name, (label, _cls) in CUSTOM_TOOL_FACTORIES.items()
    ]


def get_default_enabled_tools(yolo_mode: bool) -> list[str]:
    """Return safe default enabled tool names for this mode."""
    return default_enabled_tool_names(CUSTOM_TOOL_FACTORIES.keys(), yolo_mode)


def normalize_enabled_tools(
    enabled: list[str] | str | None, yolo_mode: bool
) -> list[str]:
    """Normalize and policy-filter enabled tool names."""
    return normalize_enabled_tool_names(
        enabled, CUSTOM_TOOL_FACTORIES.keys(), yolo_mode
    )


def get_custom_tools(
    hass: HomeAssistant,
    entry: ConfigEntry,
    enabled: list[str] | str | None = None,
    subentry_id: str | None = None,
) -> list[llm.Tool]:
    """Get custom tools for the integration.

    If enabled is provided, only tools with names in enabled are returned.
    """
    yolo_mode = _is_yolo_mode(entry, subentry_id)
    names = normalize_enabled_tools(enabled, yolo_mode)
    tools: list[llm.Tool] = []
    for name in names:
        if name not in CUSTOM_TOOL_FACTORIES:
            continue
        _label, cls = CUSTOM_TOOL_FACTORIES[name]
        tools.append(cls(hass, entry, subentry_id=subentry_id))
    return tools
