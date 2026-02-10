# claude-assist

**Use your Claude subscription as a Home Assistant conversation agent — no API key required.**

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/v/release/toml0006/claude-assist)](https://github.com/toml0006/claude-assist/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is this?

**claude-assist** is a Home Assistant custom integration that lets you use your existing Claude Pro/Max subscription as a conversation agent — the same way the official Anthropic integration works, but without needing a separate API key or pay-per-token billing.

### Why?

- 💰 **Use what you already pay for** — Claude Pro ($20/mo) or Max ($100-200/mo) subscriptions include usage that's separate from API billing
- 🏠 **Full Home Assistant integration** — works with HA's Assist pipeline, exposed entities, and voice assistants
- 🗣️ **Siri / voice ready** — pair with an Apple Shortcut for "Hey Siri, ask Claude..." from anywhere
- 🔑 **No API key needed** — authenticates via OAuth (same flow as Claude Code CLI)

### How it works

```
"Hey Siri, ask Claude why my office lights are still on"
       ↓
  Apple Shortcut → HA Conversation API
       ↓
  claude-assist (OAuth → Anthropic API)
       ↓
  Claude (sees your HA entities, answers naturally)
       ↓
  Siri speaks the response
```

## Features

- 🔐 **OAuth PKCE authentication** — secure login via your Anthropic account (same as Claude Code)
- 🔄 **Automatic token refresh** — access tokens expire every 8 hours; handled transparently
- 🏠 **HA Assist pipeline** — full conversation agent with entity control and state queries
- 🧠 **Model selection** — choose Claude Sonnet, Opus, Haiku, etc.
- ⚡ **Extended thinking** — optional deep reasoning for complex queries
- 🌐 **Web search** — optional server-side web search for real-time info
- 📝 **Custom instructions** — customize Claude's personality and behavior via HA templates

## Installation

### HACS (Recommended)

1. Open HACS in Home Assistant
2. Click the three dots → **Custom repositories**
3. Add `toml0006/claude-assist` as an **Integration**
4. Search for "Claude Assist" and install
5. Restart Home Assistant

### Manual

1. Copy the `custom_components/claude_assist` folder to your HA `config/custom_components/` directory
2. Restart Home Assistant

## Setup

1. Go to **Settings → Devices & Services → Add Integration**
2. Search for **"Claude Assist"**
3. Click **"Authenticate with Anthropic"** — this opens a browser window
4. Sign in with your Claude subscription account
5. Authorize the integration
6. Done! Claude is now available as a conversation agent

### Configure as Voice Assistant

1. Go to **Settings → Voice Assistants → Add Assistant**
2. Name it (e.g., "Claude")
3. Set **Conversation agent** to your Claude Assist instance
4. Optionally configure STT/TTS engines

### Expose Entities

Go to **Settings → Voice Assistants → Expose** tab to choose which entities Claude can see and control.

## Siri Shortcut (Optional)

Create an Apple Shortcut for hands-free access from anywhere:

1. **Create new Shortcut** named "Ask Claude"
2. Add **"Dictate Text"** action
3. Add **"Get Contents of URL"** action:
   - URL: `https://your-ha-instance.com/api/conversation/process`
   - Method: POST
   - Headers: `Authorization: Bearer YOUR_HA_TOKEN`
   - Body (JSON):
     ```json
     {
       "text": "[Dictated Text]",
       "agent_id": "conversation.claude_assist"
     }
     ```
4. Add **"Get Dictionary Value"** for key `response.speech.plain.speech`
5. Add **"Speak Text"** action

Now say: **"Hey Siri, Ask Claude [your question]"**

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| Instructions | (default prompt) | Custom instructions for Claude (supports HA templates) |
| Control Home Assistant | `true` | Allow Claude to see/control exposed entities |
| Model | `claude-sonnet-4-5-20250929` | Which Claude model to use |
| Max tokens | `1024` | Maximum response length |
| Temperature | `1.0` | Response randomness (0.0–1.0) |
| Thinking budget | `0` | Extended thinking tokens (0 = disabled) |
| Web search | `false` | Enable server-side web search |

## How is this different from the official Anthropic integration?

| Feature | Official Anthropic | claude-assist |
|---------|-------------------|---------------|
| Authentication | API key (pay-per-token) | OAuth (subscription) |
| Billing | Metered API usage | Flat monthly subscription |
| Entity control | ✅ | ✅ |
| Assist pipeline | ✅ | ✅ |
| Extended thinking | ✅ | ✅ |
| Web search | ✅ | ✅ |

The conversation agent functionality is identical — the only difference is how you authenticate.

## Requirements

- Home Assistant 2024.10.0 or newer
- Active Claude Pro or Max subscription
- A browser for the initial OAuth login

## FAQ

**Does this count against my subscription limits?**
Yes — usage goes through your subscription's fair-use allowance, same as using claude.ai or Claude Code.

**Can I use this with Claude Pro ($20/mo)?**
Yes! Any Claude subscription tier works.

**Is this against Anthropic's ToS?**
This uses the same OAuth flow as Claude Code CLI, which is an official Anthropic product. We're using the API the same way Claude Code does.

**What happens when my token expires?**
Tokens auto-refresh using your refresh token. You should rarely need to re-authenticate.

## Contributing

Contributions welcome! Please open an issue first to discuss what you'd like to change.

## License

[MIT](LICENSE)

---

*claude-assist is not affiliated with or endorsed by Anthropic.*
