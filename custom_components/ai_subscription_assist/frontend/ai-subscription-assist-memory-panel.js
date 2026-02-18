const WS = {
  ENTRY_LIST: "ai_subscription_assist/entry_list",
  MEMORY_STATUS: "ai_subscription_assist/memory_status",
  MEMORY_LIST: "ai_subscription_assist/memory_list",
  MEMORY_DELETE: "ai_subscription_assist/memory_delete",
  MEMORY_CLEAR: "ai_subscription_assist/memory_clear",
  SESSION_LIST: "ai_subscription_assist/session_list",
  SESSION_GET: "ai_subscription_assist/session_get",
  SESSION_CLEAR: "ai_subscription_assist/session_clear",
};

class AiSubscriptionAssistMemoryPanel extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._hass = null;
    this._initialized = false;
    this._state = {
      loading: false,
      error: "",
      entries: [],
      selectedEntryId: "",
      status: null,
      memoryScope: "mine",
      memoryLimit: 50,
      memorySearch: "",
      memoryTargetUserId: "",
      memories: [],
      filteredMemories: [],
      sessionScope: "mine",
      sessionLimit: 50,
      sessionSubentryId: "",
      sessionTargetUserId: "",
      sessions: [],
      selectedSession: null,
    };
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._initialized) {
      this._initialized = true;
      this._bootstrap();
    } else {
      this._render();
    }
  }

  set panel(_panel) {
    // Home Assistant sets this when rendering a panel.
  }

  connectedCallback() {
    this._render();
  }

  get _isAdmin() {
    return Boolean(this._hass?.user?.is_admin);
  }

  async _bootstrap() {
    try {
      await this._loadEntries();
      await this._refreshAll();
    } catch (err) {
      this._setError(err);
    }
  }

  _setState(patch) {
    this._state = { ...this._state, ...patch };
    this._render();
  }

  _setError(err) {
    const message =
      typeof err === "string"
        ? err
        : err?.message || "Unexpected error while loading memory panel.";
    this._setState({ error: message, loading: false });
  }

  _toast(message) {
    this.dispatchEvent(
      new CustomEvent("hass-notification", {
        detail: { message },
        bubbles: true,
        composed: true,
      })
    );
  }

  async _ws(type, payload = {}) {
    if (!this._hass?.connection) {
      throw new Error("Home Assistant websocket connection is unavailable.");
    }
    return this._hass.connection.sendMessagePromise({ type, ...payload });
  }

  async _loadEntries() {
    const response = await this._ws(WS.ENTRY_LIST);
    const entries = Array.isArray(response?.entries) ? response.entries : [];
    let selectedEntryId = this._state.selectedEntryId;
    if (!selectedEntryId && entries.length > 0) {
      selectedEntryId = entries[0].entry_id;
    } else if (
      selectedEntryId &&
      !entries.some((entry) => entry.entry_id === selectedEntryId)
    ) {
      selectedEntryId = entries.length > 0 ? entries[0].entry_id : "";
    }
    this._setState({ entries, selectedEntryId });
  }

  _selectedEntry() {
    return this._state.entries.find(
      (entry) => entry.entry_id === this._state.selectedEntryId
    );
  }

  _basePayload() {
    return this._state.selectedEntryId
      ? { config_entry_id: this._state.selectedEntryId }
      : {};
  }

  async _refreshStatus() {
    const status = await this._ws(WS.MEMORY_STATUS, this._basePayload());
    this._setState({ status });
  }

  _applyMemorySearch(memories, query) {
    const trimmed = String(query || "").trim().toLowerCase();
    if (!trimmed) {
      return memories;
    }
    return memories.filter((item) => {
      const text = String(item?.text || "").toLowerCase();
      const id = String(item?.id || "").toLowerCase();
      const scope = String(item?.scope || "").toLowerCase();
      return text.includes(trimmed) || id.includes(trimmed) || scope.includes(trimmed);
    });
  }

  async _refreshMemories() {
    const payload = {
      ...this._basePayload(),
      scope: this._state.memoryScope,
      limit: Number(this._state.memoryLimit) || 50,
    };
    if (this._isAdmin && this._state.memoryTargetUserId.trim()) {
      payload.target_user_id = this._state.memoryTargetUserId.trim();
    }
    const response = await this._ws(WS.MEMORY_LIST, payload);
    const memories = Array.isArray(response?.items) ? response.items : [];
    const filteredMemories = this._applyMemorySearch(
      memories,
      this._state.memorySearch
    );
    this._setState({ memories, filteredMemories });
  }

  async _refreshSessions() {
    const payload = {
      ...this._basePayload(),
      scope: this._state.sessionScope,
      limit: Number(this._state.sessionLimit) || 50,
    };
    if (this._state.sessionSubentryId.trim()) {
      payload.subentry_id = this._state.sessionSubentryId.trim();
    }
    if (this._isAdmin && this._state.sessionTargetUserId.trim()) {
      payload.target_user_id = this._state.sessionTargetUserId.trim();
    }
    const response = await this._ws(WS.SESSION_LIST, payload);
    const sessions = Array.isArray(response?.sessions) ? response.sessions : [];
    this._setState({ sessions });
  }

  async _refreshAll() {
    this._setState({ loading: true, error: "" });
    try {
      await Promise.all([
        this._refreshStatus(),
        this._refreshMemories(),
        this._refreshSessions(),
      ]);
      this._setState({ loading: false });
    } catch (err) {
      this._setError(err);
    }
  }

  async _withWrite(action) {
    this._setState({ loading: true, error: "" });
    try {
      await action();
      await this._refreshStatus();
      await this._refreshMemories();
      await this._refreshSessions();
      this._setState({ loading: false });
    } catch (err) {
      this._setError(err);
    }
  }

  async _deleteMemory(memoryId) {
    if (!window.confirm(`Delete memory ${memoryId}?`)) {
      return;
    }
    await this._withWrite(async () => {
      const response = await this._ws(WS.MEMORY_DELETE, {
        ...this._basePayload(),
        memory_id: memoryId,
      });
      if (!response?.deleted) {
        throw new Error("Memory was not deleted (not found or not permitted).");
      }
      this._toast(`Deleted memory ${memoryId}.`);
    });
  }

  async _clearMemories() {
    if (
      !window.confirm(
        `Clear memories for scope '${this._state.memoryScope}'? This cannot be undone.`
      )
    ) {
      return;
    }
    await this._withWrite(async () => {
      const payload = {
        ...this._basePayload(),
        scope: this._state.memoryScope,
        confirm: true,
      };
      if (this._isAdmin && this._state.memoryTargetUserId.trim()) {
        payload.target_user_id = this._state.memoryTargetUserId.trim();
      }
      const response = await this._ws(WS.MEMORY_CLEAR, payload);
      this._toast(`Cleared ${response?.removed || 0} memory item(s).`);
    });
  }

  async _selectSession(sessionId) {
    this._setState({ loading: true, error: "" });
    try {
      const response = await this._ws(WS.SESSION_GET, {
        ...this._basePayload(),
        session_id: sessionId,
        limit: 500,
      });
      this._setState({ selectedSession: response?.session || null, loading: false });
    } catch (err) {
      this._setError(err);
    }
  }

  async _clearSession(sessionId = null) {
    const label = sessionId
      ? `session '${sessionId}'`
      : `scope '${this._state.sessionScope}' sessions`;
    if (!window.confirm(`Clear ${label}? This cannot be undone.`)) {
      return;
    }
    await this._withWrite(async () => {
      const payload = {
        ...this._basePayload(),
        scope: this._state.sessionScope,
        confirm: true,
      };
      if (sessionId) {
        payload.session_id = sessionId;
      } else {
        if (this._state.sessionSubentryId.trim()) {
          payload.subentry_id = this._state.sessionSubentryId.trim();
        }
        if (this._isAdmin && this._state.sessionTargetUserId.trim()) {
          payload.target_user_id = this._state.sessionTargetUserId.trim();
        }
      }

      const response = await this._ws(WS.SESSION_CLEAR, payload);
      if (sessionId && this._state.selectedSession?.session_id === sessionId) {
        this._setState({ selectedSession: null });
      }
      this._toast(
        `Cleared ${response?.removed_sessions || 0} session(s), ${
          response?.removed_messages || 0
        } message(s).`
      );
    });
  }

  _escape(text) {
    return String(text ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  _styles() {
    return `
      :host {
        --panel-gap: 16px;
        display: block;
        padding: 16px;
        box-sizing: border-box;
      }
      .wrap {
        display: grid;
        gap: var(--panel-gap);
        max-width: 1400px;
      }
      .toolbar,
      .controls {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
      }
      .toolbar {
        justify-content: space-between;
      }
      .grid {
        display: grid;
        gap: var(--panel-gap);
        grid-template-columns: 1fr;
      }
      @media (min-width: 1200px) {
        .grid {
          grid-template-columns: 1fr 1fr;
        }
      }
      ha-card {
        padding: 16px;
      }
      h2, h3 {
        margin: 0 0 8px;
      }
      p.meta {
        margin: 0;
        color: var(--secondary-text-color);
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
      }
      th, td {
        border-bottom: 1px solid var(--divider-color);
        padding: 8px;
        text-align: left;
        vertical-align: top;
      }
      td.actions {
        white-space: nowrap;
      }
      .error {
        color: var(--error-color);
        font-weight: 600;
      }
      .pill {
        display: inline-block;
        border: 1px solid var(--divider-color);
        border-radius: 999px;
        padding: 2px 8px;
        font-size: 12px;
      }
      .status-grid {
        display: grid;
        gap: 8px;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        margin-top: 12px;
      }
      .status-item {
        border: 1px solid var(--divider-color);
        border-radius: 8px;
        padding: 8px;
      }
      .status-item .label {
        color: var(--secondary-text-color);
        font-size: 12px;
      }
      .status-item .value {
        font-size: 20px;
        font-weight: 600;
      }
      .messages {
        margin: 0;
        padding: 0;
        list-style: none;
        display: grid;
        gap: 8px;
      }
      .msg {
        border: 1px solid var(--divider-color);
        border-radius: 8px;
        padding: 8px;
      }
      .msg .role {
        font-weight: 700;
      }
      .msg .time {
        font-size: 12px;
        color: var(--secondary-text-color);
      }
      .empty {
        color: var(--secondary-text-color);
      }
    `;
  }

  _render() {
    if (!this.shadowRoot) {
      return;
    }
    const entry = this._selectedEntry();
    const status = this._state.status;
    const memories = this._state.filteredMemories || [];
    const sessions = this._state.sessions || [];
    const selectedSession = this._state.selectedSession;
    const subentryOptions = (entry?.subentries || [])
      .map(
        (subentry) =>
          `<option value="${this._escape(subentry.subentry_id)}">${this._escape(
            subentry.title || subentry.subentry_id
          )}</option>`
      )
      .join("");
    const adminOnlyFields = this._isAdmin
      ? `
          <label>
            Target user:
            <input id="memory-target-user" type="text" value="${this._escape(
              this._state.memoryTargetUserId
            )}" placeholder="Optional user_id">
          </label>
        `
      : "";
    const adminOnlySessionFields = this._isAdmin
      ? `
          <label>
            Scope:
            <select id="session-scope">
              <option value="mine" ${
                this._state.sessionScope === "mine" ? "selected" : ""
              }>mine</option>
              <option value="all" ${
                this._state.sessionScope === "all" ? "selected" : ""
              }>all</option>
            </select>
          </label>
          <label>
            Target user:
            <input id="session-target-user" type="text" value="${this._escape(
              this._state.sessionTargetUserId
            )}" placeholder="Optional user_id">
          </label>
        `
      : `
          <span class="pill">scope: mine</span>
        `;

    this.shadowRoot.innerHTML = `
      <style>${this._styles()}</style>
      <div class="wrap">
        <div class="toolbar">
          <div>
            <h2>AI Subscription Assist Memory Manager</h2>
            <p class="meta">Manage long-term memories and resumable sessions.</p>
          </div>
          <div class="controls">
            <label>
              Provider entry:
              <select id="entry-select">
                ${this._state.entries
                  .map(
                    (item) => `
                      <option value="${this._escape(item.entry_id)}" ${
                        item.entry_id === this._state.selectedEntryId ? "selected" : ""
                      }>
                        ${this._escape(item.title || item.entry_id)}
                      </option>
                    `
                  )
                  .join("")}
              </select>
            </label>
            <mwc-button id="refresh-all" raised>Refresh</mwc-button>
          </div>
        </div>

        ${
          this._state.error
            ? `<div class="error">${this._escape(this._state.error)}</div>`
            : ""
        }
        ${
          this._state.loading
            ? `<div class="pill">Loading…</div>`
            : ""
        }
        ${
          !entry
            ? `<div class="empty">No provider entries found. Add AI Subscription Assist first.</div>`
            : `
              <div class="grid">
                <ha-card>
                  <h3>Status</h3>
                  <p class="meta">Entry: ${this._escape(entry.title || entry.entry_id)}</p>
                  ${
                    status
                      ? `
                      <div class="status-grid">
                        <div class="status-item">
                          <div class="label">Memory enabled</div>
                          <div class="value">${status.memory_enabled ? "yes" : "no"}</div>
                        </div>
                        <div class="status-item">
                          <div class="label">Shared memories</div>
                          <div class="value">${this._escape(
                            status.counts?.shared_memories ?? 0
                          )}</div>
                        </div>
                        <div class="status-item">
                          <div class="label">Your memories</div>
                          <div class="value">${this._escape(
                            status.counts?.own_user_memories ?? 0
                          )}</div>
                        </div>
                        <div class="status-item">
                          <div class="label">Your sessions</div>
                          <div class="value">${this._escape(
                            status.counts?.sessions_mine ?? 0
                          )}</div>
                        </div>
                      </div>
                      `
                      : `<p class="empty">No status loaded yet.</p>`
                  }
                </ha-card>

                <ha-card>
                  <h3>Memory Items</h3>
                  <div class="controls">
                    <label>
                      Scope:
                      <select id="memory-scope">
                        <option value="mine" ${
                          this._state.memoryScope === "mine" ? "selected" : ""
                        }>mine</option>
                        <option value="shared" ${
                          this._state.memoryScope === "shared" ? "selected" : ""
                        }>shared</option>
                        <option value="all" ${
                          this._state.memoryScope === "all" ? "selected" : ""
                        }>all</option>
                      </select>
                    </label>
                    <label>
                      Limit:
                      <input id="memory-limit" type="number" min="1" max="500" value="${this._escape(
                        this._state.memoryLimit
                      )}">
                    </label>
                    ${adminOnlyFields}
                    <mwc-button id="refresh-memory">Refresh</mwc-button>
                    <mwc-button id="clear-memory" outlined>Clear Scope</mwc-button>
                  </div>
                  <div class="controls">
                    <label>
                      Search:
                      <input id="memory-search" type="text" value="${this._escape(
                        this._state.memorySearch
                      )}" placeholder="Filter by id/text/scope">
                    </label>
                  </div>
                  ${
                    memories.length < 1
                      ? `<p class="empty">No memory items for current filter.</p>`
                      : `
                        <table>
                          <thead>
                            <tr>
                              <th>ID</th>
                              <th>Scope</th>
                              <th>Text</th>
                              <th>Updated</th>
                              <th>Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            ${memories
                              .map(
                                (item) => `
                                  <tr>
                                    <td><code>${this._escape(item.id)}</code></td>
                                    <td>${this._escape(item.scope)}</td>
                                    <td>${this._escape(item.text)}</td>
                                    <td>${this._escape(item.updated_at)}</td>
                                    <td class="actions">
                                      <mwc-button data-action="delete-memory" data-memory-id="${this._escape(
                                        item.id
                                      )}" outlined>Delete</mwc-button>
                                    </td>
                                  </tr>
                                `
                              )
                              .join("")}
                          </tbody>
                        </table>
                      `
                  }
                </ha-card>

                <ha-card>
                  <h3>Sessions</h3>
                  <div class="controls">
                    ${adminOnlySessionFields}
                    <label>
                      Subentry:
                      <input id="session-subentry" type="text" value="${this._escape(
                        this._state.sessionSubentryId
                      )}" placeholder="Optional subentry_id" list="session-subentry-options">
                    </label>
                    <datalist id="session-subentry-options">
                      ${subentryOptions}
                    </datalist>
                    <label>
                      Limit:
                      <input id="session-limit" type="number" min="1" max="500" value="${this._escape(
                        this._state.sessionLimit
                      )}">
                    </label>
                    <mwc-button id="refresh-sessions">Refresh</mwc-button>
                    <mwc-button id="clear-sessions" outlined>Clear Scope</mwc-button>
                  </div>
                  ${
                    sessions.length < 1
                      ? `<p class="empty">No sessions for current filter.</p>`
                      : `
                        <table>
                          <thead>
                            <tr>
                              <th>Session ID</th>
                              <th>Subentry</th>
                              <th>Messages</th>
                              <th>Updated</th>
                              <th>Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            ${sessions
                              .map(
                                (session) => `
                                  <tr>
                                    <td><code>${this._escape(session.session_id)}</code></td>
                                    <td>${this._escape(session.subentry_id)}</td>
                                    <td>${this._escape(session.message_count)}</td>
                                    <td>${this._escape(session.updated_at)}</td>
                                    <td class="actions">
                                      <mwc-button data-action="view-session" data-session-id="${this._escape(
                                        session.session_id
                                      )}">View</mwc-button>
                                      <mwc-button data-action="clear-session" data-session-id="${this._escape(
                                        session.session_id
                                      )}" outlined>Clear</mwc-button>
                                    </td>
                                  </tr>
                                `
                              )
                              .join("")}
                          </tbody>
                        </table>
                      `
                  }
                </ha-card>

                <ha-card>
                  <h3>Session Detail</h3>
                  ${
                    !selectedSession
                      ? `<p class="empty">Select a session to inspect transcript context.</p>`
                      : `
                        <div class="controls">
                          <span class="pill"><code>${this._escape(
                            selectedSession.session_id
                          )}</code></span>
                          <span class="pill">messages: ${this._escape(
                            selectedSession.message_count
                          )}</span>
                          <mwc-button id="clear-selected-session" outlined>Clear This Session</mwc-button>
                        </div>
                        <ul class="messages">
                          ${(selectedSession.messages || [])
                            .map(
                              (message) => `
                                <li class="msg">
                                  <div class="role">${this._escape(message.role)}</div>
                                  <div>${this._escape(message.content)}</div>
                                  <div class="time">${this._escape(message.created_at)}</div>
                                </li>
                              `
                            )
                            .join("")}
                        </ul>
                      `
                  }
                </ha-card>
              </div>
            `
        }
      </div>
    `;

    this._bindEvents();
  }

  _bindEvents() {
    const root = this.shadowRoot;
    if (!root) {
      return;
    }

    root.getElementById("entry-select")?.addEventListener("change", async (event) => {
      this._setState({ selectedEntryId: event.target.value, selectedSession: null });
      await this._refreshAll();
    });

    root.getElementById("refresh-all")?.addEventListener("click", async () => {
      await this._loadEntries();
      await this._refreshAll();
    });

    root.getElementById("memory-scope")?.addEventListener("change", async (event) => {
      this._setState({ memoryScope: event.target.value });
      await this._refreshMemories();
    });

    root.getElementById("memory-limit")?.addEventListener("change", async (event) => {
      this._setState({ memoryLimit: Number(event.target.value) || 50 });
      await this._refreshMemories();
    });

    root.getElementById("memory-target-user")?.addEventListener("change", async (event) => {
      this._setState({ memoryTargetUserId: event.target.value });
      await this._refreshMemories();
    });

    root.getElementById("memory-search")?.addEventListener("input", (event) => {
      const memorySearch = event.target.value;
      const filteredMemories = this._applyMemorySearch(
        this._state.memories,
        memorySearch
      );
      this._setState({ memorySearch, filteredMemories });
    });

    root.getElementById("refresh-memory")?.addEventListener("click", async () => {
      await this._refreshMemories();
    });

    root.getElementById("clear-memory")?.addEventListener("click", async () => {
      await this._clearMemories();
    });

    root.getElementById("session-scope")?.addEventListener("change", async (event) => {
      this._setState({ sessionScope: event.target.value, selectedSession: null });
      await this._refreshSessions();
    });

    root.getElementById("session-subentry")?.addEventListener("change", async (event) => {
      this._setState({ sessionSubentryId: event.target.value, selectedSession: null });
      await this._refreshSessions();
    });

    root.getElementById("session-target-user")?.addEventListener("change", async (event) => {
      this._setState({ sessionTargetUserId: event.target.value, selectedSession: null });
      await this._refreshSessions();
    });

    root.getElementById("session-limit")?.addEventListener("change", async (event) => {
      this._setState({ sessionLimit: Number(event.target.value) || 50 });
      await this._refreshSessions();
    });

    root.getElementById("refresh-sessions")?.addEventListener("click", async () => {
      await this._refreshSessions();
    });

    root.getElementById("clear-sessions")?.addEventListener("click", async () => {
      await this._clearSession();
    });

    root.getElementById("clear-selected-session")?.addEventListener("click", async () => {
      const sessionId = this._state.selectedSession?.session_id;
      if (sessionId) {
        await this._clearSession(sessionId);
      }
    });

    root.querySelectorAll("[data-action='delete-memory']").forEach((button) => {
      button.addEventListener("click", async () => {
        await this._deleteMemory(button.dataset.memoryId);
      });
    });

    root.querySelectorAll("[data-action='view-session']").forEach((button) => {
      button.addEventListener("click", async () => {
        await this._selectSession(button.dataset.sessionId);
      });
    });

    root.querySelectorAll("[data-action='clear-session']").forEach((button) => {
      button.addEventListener("click", async () => {
        await this._clearSession(button.dataset.sessionId);
      });
    });
  }
}

customElements.define("ai-subscription-assist-memory-panel", AiSubscriptionAssistMemoryPanel);
