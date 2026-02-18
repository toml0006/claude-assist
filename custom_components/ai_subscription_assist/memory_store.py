"""Persistent storage for AI Subscription Assist memory."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN

STORAGE_VERSION = 1
STORAGE_KEY = f"{DOMAIN}.memory"


class MemoryStore:
    """Wrapper over HA Store for memory persistence."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._store: Store[dict[str, Any]] = Store(hass, STORAGE_VERSION, STORAGE_KEY)

    async def async_load(self) -> dict[str, Any]:
        """Load memory document."""
        data = await self._store.async_load()
        if not isinstance(data, dict):
            return {"entries": {}}
        entries = data.get("entries")
        if not isinstance(entries, dict):
            data["entries"] = {}
        return data

    async def async_save(self, data: dict[str, Any]) -> None:
        """Persist memory document."""
        await self._store.async_save(data)
