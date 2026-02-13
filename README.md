# claude-assist

Use your **Claude subscription** (Pro/Max) as a **Home Assistant Assist conversation agent** — **no Anthropic API key** required.

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/v/release/toml0006/claude-assist)](https://github.com/toml0006/claude-assist/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is this?

**claude-assist** is a Home Assistant custom integration that lets you use your existing Claude subscription as a conversation agent inside Home Assistant’s Assist pipeline.

It authenticates using the **same OAuth flow as the Claude Code CLI**, so you can avoid pay-per-token API billing.

---

## Highlights

- **Subscription OAuth auth** (PKCE) — no API key required
- **Automatic token refresh** (access tokens expire ~8 hours)
- **Assist pipeline compatible** (works like a normal HA conversation agent)
- **Per-agent model selection** in the UI
- **Extended tools** (optional) beyond standard Assist capabilities:
  - History (`get_history`)
  - Logbook (`get_logbook`)
  - Statistics (`get_statistics`)
  - Template rendering (`render_template`)
  - Automations (`list_automations`, `toggle_automation`, `add_automation`)
  - Dashboards (experimental) (`modify_dashboard`)
  - Notifications (`send_notification`)
  - “Who’s home” (`who_is_home`)
  - Shopping/todo (`manage_list`)
  - Calendar (`get_calendar_events`)
- **Per-agent tool allowlist**: choose exactly which *extra* tools each configured agent can use

---

## How it works

```
Siri / HA Voice / UI chat
        ↓
HA Conversation API / Assist Pipeline
        ↓
claude-assist (subscription OAuth)
        ↓
Anthropic API (Claude)
        ↓
Response spoken/displayed by Home Assistant
```

---

## Installation

### Option A: HACS (recommended)

1. Home Assistant → **HACS**
2. **⋮ → Custom repositories**
3. Add `toml0006/claude-assist` as **Integration**
4. Install **Claude Assist**
5. Restart Home Assistant

### Option B: Manual

1. Copy `custom_components/claude_assist` into your HA config directory:
   - `/config/custom_components/claude_assist`
2. Restart Home Assistant

---

## Setup

1. HA → **Settings → Devices & services → Add integration**
2. Search for **Claude Assist**
3. Follow the OAuth instructions (you’ll be sent to a Claude/Anthropic page)
4. Paste the returned code back into HA

After setup you’ll have a **conversation agent** sub-entry (and you can create multiple agents/sub-entries if you want different models/tool permissions).

---

## Configure an agent (model + tools)

For each configured conversation agent/sub-entry:

- **Model**: choose which Claude model to use (4.x models are listed first; <4.x are labeled **Legacy**)
- **Enabled tools**: select which *extra* tools this agent can call
- **LLM API selection (important):**
  - Select **Claude Assist — <agent name>**
  - Do **not** also select the plain **Home Assistant** / **Assist** API in the multi-select, or tools may be namespaced and become harder for the model to call.

---

## Siri Shortcut (optional)

You can build a simple Shortcut:

1. **Dictate Text**
2. **Get Contents of URL**
   - URL: `https://<your-ha>/api/conversation/process`
   - Method: `POST`
   - Headers:
     - `Authorization: Bearer <HA Long-Lived Access Token>`
     - `Content-Type: application/json`
   - Body:
     ```json
     {
       "text": "<dictated text>",
       "language": "en",
       "agent_id": "conversation.<your_claude_assist_entity_id>"
     }
     ```
3. Speak `response.speech.plain.speech`

Note: multi-turn “session” behavior depends on whether your caller reuses `conversation_id`.

---

## FAQ

### Does this use my subscription instead of API billing?
Yes. It uses subscription OAuth (Claude Code-style). No API key needed.

### Is this the same as the official Anthropic integration?
Functionally similar on the HA side (conversation agent), but authentication differs:

- **Official Anthropic integration**: API key (metered)
- **claude-assist**: subscription OAuth (flat subscription)

### Why can’t Claude answer “when did X change” by default?
Standard Assist tools are focused on live state + control. For history/logbook/statistics you must enable the extra tools for that agent.

---

## Development notes (auth details)

This integration works by mimicking Claude Code’s OAuth + request headers.

- Token endpoint: `https://console.anthropic.com/v1/oauth/token`
- Requires beta/header matching (e.g. `anthropic-beta: claude-code-20250219,oauth-2025-04-20`, `x-app: cli`, `user-agent: claude-cli/...`)

---

## Contributing

PRs welcome. If you’re adding new tools, please keep them:
- opt-in per agent (tool allowlist)
- safe by default
- returning structured JSON objects (not pre-serialized JSON strings)

---

## License

MIT — see [LICENSE](LICENSE).

---

*claude-assist is not affiliated with or endorsed by Anthropic.*
