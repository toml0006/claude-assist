# ai-subscription-assist

Use subscription-backed LLMs (OAuth) as **Home Assistant Assist conversation agents**:
- **Claude subscription** (Pro/Max) via **OAuth** (no Anthropic API key required)
- **OpenAI Codex** (ChatGPT subscription) via **OAuth**
- **Google Gemini CLI** (Cloud Code Assist subscription) via **OAuth**
- **OpenAI / compatible** via **API key** (metered)

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/v/release/toml0006/ai-subscription-assist)](https://github.com/toml0006/ai-subscription-assist/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is this?

**ai-subscription-assist** is a Home Assistant custom integration that lets you use multiple LLM providers as conversation agents inside Home Assistant’s Assist pipeline.

For subscription users, it authenticates using OAuth flows compatible with popular CLIs (Claude Code, Codex, Gemini CLI), so you can avoid pay-per-token API billing when your subscription supports it.

---

## Highlights

- **Requires Home Assistant 2026.2.2+** (Python 3.13+)
- **Multi-provider**:
  - Claude subscription OAuth
  - OpenAI Codex (ChatGPT OAuth subscription)
  - Google Gemini CLI (OAuth subscription)
  - OpenAI / compatible API keys
- **Multiple entries supported**: set up more than one provider/account
- **Subscription OAuth auth** (PKCE) — no API key required for Claude/Codex/Gemini
- **Automatic token refresh** (access tokens expire ~8 hours)
- **Assist pipeline compatible** (works like a normal HA conversation agent)
- **Per-agent model selection** in the UI (one provider per config entry; multiple agents per entry)
- **Extended tools** (optional) beyond standard Assist capabilities:
  - History (`get_history`)
  - Logbook (`get_logbook`)
  - Statistics (`get_statistics`)
  - Template rendering (`render_template`)
  - Automations (`list_automations`, `toggle_automation`, `add_automation`)
  - Dashboards (experimental) (`modify_dashboard`)
  - Generic HA service calls (YOLO) (`call_service`)
  - Notifications (`send_notification`)
  - “Who’s home” (`who_is_home`)
  - Shopping/todo (`manage_list`)
  - Calendar (`get_calendar_events`)
- **Per-agent tool allowlist**: choose exactly which *extra* tools each configured agent can use
- **Per-agent YOLO mode**: explicit opt-in to bypass exposure filtering and enable privileged tools

---

## How it works

```
Siri / HA Voice / UI chat
        ↓
HA Conversation API / Assist Pipeline
        ↓
ai-subscription-assist (provider entry + per-agent config)
        ↓
Provider API (Claude / Codex / Gemini / OpenAI-compatible)
        ↓
Response spoken/displayed by Home Assistant
```

---

## Getting started

### Install

You can install via HACS (recommended) or manually.

### Option A: HACS (recommended)

1. Home Assistant → **HACS**
2. **⋮ → Custom repositories**
3. Add `toml0006/ai-subscription-assist` as **Integration**
4. Install **AI Subscription Assist**
5. Restart Home Assistant

### Option B: Manual

1. Copy `custom_components/claude_assist` into your HA config directory:
   - `/config/custom_components/claude_assist`
2. Restart Home Assistant

Note: the integration domain/folder is currently still `claude_assist` for backwards compatibility.

### Set up providers

Repeat this once per provider/account you want (multiple config entries are supported):

1. HA → **Settings → Devices & services → Add integration**
2. Search for **AI Subscription Assist**
3. Choose the provider:
   - **Claude subscription (OAuth)**: follow the OAuth instructions (you’ll be sent to a Claude/Anthropic page)
   - **OpenAI Codex (ChatGPT OAuth subscription)**: follow the OAuth instructions (you’ll be sent to an OpenAI page)
   - **Google Gemini CLI (OAuth subscription)**: follow the OAuth instructions (you’ll be sent to a Google page)
   - **OpenAI / compatible (API key)**: enter your API key and optional base URL
4. Finish setup

After setup you’ll have a **conversation agent** sub-entry (and you can create multiple agents/sub-entries if you want different models/tool permissions).

#### OAuth notes (Codex/Gemini/Claude)

- The OAuth redirect is usually to a `http://localhost:...` URL which may not load in your browser. That’s expected.
- Copy/paste the full redirect URL into Home Assistant (or for Codex you can paste just the `code=` value).
- If you hit **OAuth state mismatch**, just restart the config flow and try again.
- For **Gemini CLI**, some accounts require a Google Cloud project ID. If you see the “project required” error, provide a project ID and re-authenticate.

### Add more agents (different model/tools) under the same provider

1. HA → **Settings → Devices & services**
2. Open the **AI Subscription Assist** entry you want
3. Add a new **conversation agent** sub-entry

Each agent has its own model + tool allowlist/YOLO configuration.

### Configure an agent (model + tools)

For each configured conversation agent/sub-entry:

- **Model**: pick a provider-appropriate model (Claude/OpenAI/Gemini); custom values are allowed
- **Enabled tools**: select which *extra* tools this agent can call
- **LLM API selection (important):**
  - Select **AI Subscription Assist — <entry name> — <agent name>**
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
       "agent_id": "conversation.<your_agent_entity_id>"
     }
     ```
3. Speak `response.speech.plain.speech`

Tip: you can find the agent entity id in **Settings → Devices & services → Entities** (domain `conversation`).

Note: multi-turn “session” behavior depends on whether your caller reuses `conversation_id`.

---

## FAQ

### Does this use my subscription instead of API billing?
Yes, for the OAuth-backed providers (Claude subscription, OpenAI Codex, Gemini CLI). No API key needed for those.

For **OpenAI / compatible (API key)** it uses normal API billing.

### Is this the same as the official Anthropic integration?
Functionally similar on the HA side (conversation agent), but authentication differs:

- **Official Anthropic integration**: API key (metered)
- **ai-subscription-assist**: subscription OAuth (flat subscription)

### Why can’t Claude answer “when did X change” by default?
Standard Assist tools are focused on live state + control. For history/logbook/statistics you must enable the extra tools for that agent.

---

## Documentation

### Tool safety and YOLO mode

By default, extra tools are filtered by Home Assistant’s entity exposure settings and this integration’s tool policy.

If you enable **YOLO mode** on an agent, it:
- bypasses entity exposure filtering
- unlocks privileged tools (service calls, automation/dashboard edits, log access)

Enable YOLO mode only for trusted agents.

## Development notes (auth details)

This integration works by mimicking Claude Code’s OAuth + request headers.

- Token endpoint: `https://console.anthropic.com/v1/oauth/token`
- Requires beta/header matching (e.g. `anthropic-beta: claude-code-20250219,oauth-2025-04-20`, `x-app: cli`, `user-agent: claude-cli/...`)

---

## Roadmap

- **Per-agent memory**: persistent memory/summaries per configured agent (opt-in), so agents can remember preferences and context across sessions.
- More provider backends and better UX around provider auth flows.

## Contributing

Contributions welcome. PRs and issues are appreciated, especially around:
- new subscription provider backends
- improving safety/tooling defaults
- tests and local development workflows
- agent memory design and implementation

If you’re adding new tools, please keep them:
- opt-in per agent (tool allowlist)
- safe by default
- returning structured JSON objects (not pre-serialized JSON strings)

### Local dev harness

Run:

```bash
./scripts/dev_harness.sh
```

This creates/uses `.venv`, runs compile checks, Ruff, and unit tests.

### Local Home Assistant (Docker)

This repo includes a minimal Home Assistant instance for integration development.

Run:

```bash
./scripts/ha_dev.sh bootstrap
./scripts/ha_dev.sh logs
```

Then open `http://localhost:8123`.

Notes:
- The HA config lives in `dev/ha_config/` (state files are gitignored).
- The integration is bind-mounted into `/config/custom_components/claude_assist`.
- After Python code changes, restart HA with `./scripts/ha_dev.sh restart`.

---

## License

MIT — see [LICENSE](LICENSE).

---

*ai-subscription-assist is not affiliated with or endorsed by Anthropic, OpenAI, or Google.*
