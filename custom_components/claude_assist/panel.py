"""Frontend panel registration for AI Subscription Assist."""

from __future__ import annotations

from pathlib import Path

from homeassistant.components.frontend import async_remove_panel
from homeassistant.components.http import StaticPathConfig
from homeassistant.components.panel_custom import async_register_panel
from homeassistant.core import HomeAssistant

from .const import (
    DATA_MEMORY_PANEL_REGISTERED,
    DOMAIN,
    LOGGER,
    PANEL_COMPONENT_NAME,
    PANEL_MODULE_URL,
    PANEL_SIDEBAR_ICON,
    PANEL_SIDEBAR_TITLE,
    PANEL_STATIC_BASE_URL,
    PANEL_URL_PATH,
)


async def async_setup_memory_panel(hass: HomeAssistant) -> None:
    """Register the memory/session management panel."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    if domain_data.get(DATA_MEMORY_PANEL_REGISTERED):
        return

    static_dir = Path(__file__).parent / "frontend"
    if not static_dir.exists():
        LOGGER.warning("Panel frontend directory missing: %s", static_dir)
        return

    await hass.http.async_register_static_paths(
        [StaticPathConfig(PANEL_STATIC_BASE_URL, str(static_dir), False)]
    )

    panel_kwargs = {
        "frontend_url_path": PANEL_URL_PATH,
        "webcomponent_name": PANEL_COMPONENT_NAME,
        "module_url": PANEL_MODULE_URL,
        "sidebar_title": PANEL_SIDEBAR_TITLE,
        "sidebar_icon": PANEL_SIDEBAR_ICON,
        "config": {"domain": DOMAIN},
        "require_admin": False,
    }

    # HA API compatibility: some versions support update=, some do not.
    try:
        await async_register_panel(
            hass,
            **panel_kwargs,
            update=True,
        )
    except TypeError:
        await async_register_panel(
            hass,
            **panel_kwargs,
        )
    domain_data[DATA_MEMORY_PANEL_REGISTERED] = True


def async_unload_memory_panel(hass: HomeAssistant) -> None:
    """Remove memory/session panel when no entries remain."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    if not domain_data.get(DATA_MEMORY_PANEL_REGISTERED):
        return
    async_remove_panel(hass, PANEL_URL_PATH, warn_if_unknown=False)
    domain_data[DATA_MEMORY_PANEL_REGISTERED] = False
