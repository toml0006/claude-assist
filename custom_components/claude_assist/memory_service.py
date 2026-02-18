"""Service-level memory manager for AI Subscription Assist."""

from __future__ import annotations

import asyncio
from typing import Any
import uuid

import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv

from .const import (
    CONF_MEMORY_AUTO_RECALL,
    CONF_MEMORY_AUTO_WRITE,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_MAX_ITEMS_PER_SCOPE,
    CONF_MEMORY_RECALL_TOP_K,
    CONF_MEMORY_RESUME_ENABLED,
    CONF_MEMORY_RESUME_MAX_MESSAGES,
    CONF_MEMORY_TTL_DAYS,
    DATA_MEMORY_SERVICES,
    DOMAIN,
    LOGGER,
    MEMORY_DEFAULTS,
)
from .memory_core import (
    extract_heuristic_memory,
    format_memory_prompt,
    is_duplicate_memory,
    is_slash_command,
    looks_sensitive,
    parse_iso8601,
    parse_slash_command,
    prune_memory_items,
    rank_memory_items,
    utcnow,
    utcnow_iso,
)
from .memory_store import MemoryStore

SERVICE_MEMORY_STATUS = "memory_status"
SERVICE_MEMORY_LIST = "memory_list"
SERVICE_MEMORY_DELETE = "memory_delete"
SERVICE_MEMORY_CLEAR = "memory_clear"
SERVICE_SESSION_LIST = "session_list"
SERVICE_SESSION_GET = "session_get"
SERVICE_SESSION_CLEAR = "session_clear"

VALID_SESSION_SCOPES = {"mine", "all"}
DATA_MEMORY_DOMAIN_SERVICES_REGISTERED = "memory_domain_services_registered"


class _FastEmbedder:
    """Lazy embedding wrapper with graceful fallback."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        self._embedder: Any | None = None
        self._available: bool | None = None
        self._init_lock = asyncio.Lock()

    async def async_embed_one(self, text: str) -> list[float] | None:
        """Embed one string."""
        vectors = await self.async_embed_many([text])
        return vectors[0] if vectors else None

    async def async_embed_many(self, texts: list[str]) -> list[list[float] | None]:
        """Embed many strings."""
        if not texts:
            return []

        embedder = await self._async_get_embedder()
        if embedder is None:
            return [None for _ in texts]

        def _embed_sync() -> list[list[float] | None]:
            vectors: list[list[float] | None] = []
            try:
                for vec in embedder.embed(texts):
                    if hasattr(vec, "tolist"):
                        vectors.append([float(x) for x in vec.tolist()])
                    else:
                        vectors.append([float(x) for x in vec])
            except Exception as err:
                LOGGER.debug("Local embedding failed, falling back to lexical only: %s", err)
                return [None for _ in texts]
            return vectors

        return await self._hass.async_add_executor_job(_embed_sync)

    async def _async_get_embedder(self) -> Any | None:
        if self._available is False:
            return None
        if self._embedder is not None:
            return self._embedder

        async with self._init_lock:
            if self._available is False:
                return None
            if self._embedder is not None:
                return self._embedder

            def _build() -> Any | None:
                try:
                    from fastembed import TextEmbedding  # type: ignore[import-not-found]

                    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                except Exception as err:
                    LOGGER.debug("FastEmbed unavailable, using lexical ranking only: %s", err)
                    return None

            self._embedder = await self._hass.async_add_executor_job(_build)
            self._available = self._embedder is not None
            return self._embedder


class ClaudeAssistMemoryService:
    """Memory manager scoped to a single config entry."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._hass = hass
        self._entry = entry
        self._store = MemoryStore(hass)
        self._lock = asyncio.Lock()
        self._data: dict[str, Any] = {"entries": {}}
        self._embedder = _FastEmbedder(hass)

    async def async_initialize(self) -> None:
        """Load persistent memory data."""
        self._data = await self._store.async_load()

    def is_enabled(self) -> bool:
        """Return whether memory is enabled for this entry."""
        return bool(self._option(CONF_MEMORY_ENABLED))

    async def async_user_is_admin(self, user_id: str | None) -> bool:
        """Return whether the given user id maps to an HA admin user."""
        return await self._async_is_admin(user_id)

    async def async_status_snapshot(
        self, requester_user_id: str | None, *, is_admin: bool
    ) -> dict[str, Any]:
        """Return memory/session status and counts for this entry."""
        async with self._lock:
            doc = self._entry_doc()
            self._prune_entry_locked(doc)

            shared_items = doc.get("shared", [])
            users = doc.get("users", {})
            resume = doc.get("resume", {})

            if not isinstance(shared_items, list):
                shared_items = []
            if not isinstance(users, dict):
                users = {}
            if not isinstance(resume, dict):
                resume = {}

            own_user_count = 0
            if requester_user_id and isinstance(users.get(requester_user_id), list):
                own_user_count = len(users[requester_user_id])

            sessions_total = 0
            sessions_mine = 0
            for by_user in resume.values():
                if not isinstance(by_user, dict):
                    continue
                for owner_id, messages in by_user.items():
                    if not isinstance(messages, list) or not messages:
                        continue
                    sessions_total += 1
                    if requester_user_id and owner_id == requester_user_id:
                        sessions_mine += 1

            await self._store.async_save(self._data)

        snapshot: dict[str, Any] = {
            "entry_id": self._entry.entry_id,
            "memory_enabled": self.is_enabled(),
            "auto_write": bool(self._option(CONF_MEMORY_AUTO_WRITE)),
            "auto_recall": bool(self._option(CONF_MEMORY_AUTO_RECALL)),
            "resume_enabled": bool(self._option(CONF_MEMORY_RESUME_ENABLED)),
            "ttl_days": int(self._option(CONF_MEMORY_TTL_DAYS)),
            "max_items_per_scope": int(self._option(CONF_MEMORY_MAX_ITEMS_PER_SCOPE)),
            "recall_top_k": int(self._option(CONF_MEMORY_RECALL_TOP_K)),
            "resume_max_messages": int(self._option(CONF_MEMORY_RESUME_MAX_MESSAGES)),
            "counts": {
                "shared_memories": len(shared_items),
                "own_user_memories": own_user_count,
                "sessions_mine": sessions_mine,
            },
        }
        if is_admin:
            snapshot["counts"]["user_scopes"] = len(users)
            snapshot["counts"]["sessions_total"] = sessions_total
        return snapshot

    async def async_list_memory_items(
        self,
        *,
        scope: str,
        limit: int,
        requester_user_id: str | None,
        is_admin: bool,
        target_user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List memory items for service-level management."""
        limit = max(1, min(limit, 500))
        if scope not in {"mine", "shared", "all"}:
            scope = "mine"

        async with self._lock:
            doc = self._entry_doc()
            self._prune_entry_locked(doc)
            items = self._collect_memories_for_management(
                doc,
                scope=scope,
                requester_user_id=requester_user_id,
                target_user_id=target_user_id,
                is_admin=is_admin,
            )
            items.sort(
                key=lambda item: parse_iso8601(item.get("updated_at")).timestamp(),
                reverse=True,
            )
            result = [
                self._serialize_memory_item(item, include_user_id=is_admin)
                for item in items[:limit]
            ]
            await self._store.async_save(self._data)
            return result

    async def async_delete_memory_item(
        self, memory_id: str, requester_user_id: str | None, *, is_admin: bool
    ) -> tuple[bool, str]:
        """Delete a memory item by id."""
        return await self._async_delete_memory(memory_id, requester_user_id, is_admin)

    async def async_clear_memory_items(
        self,
        *,
        scope: str,
        requester_user_id: str | None,
        is_admin: bool,
        target_user_id: str | None = None,
    ) -> int:
        """Clear memory items for scope/target user."""
        removed = 0
        async with self._lock:
            doc = self._entry_doc()
            if scope in {"mine", "all"}:
                user_id = self._resolve_target_user(
                    requester_user_id=requester_user_id,
                    target_user_id=target_user_id,
                    is_admin=is_admin,
                    allow_all_admin=False,
                )
                if user_id:
                    users = doc.get("users", {})
                    if isinstance(users, dict) and isinstance(users.get(user_id), list):
                        removed += len(users[user_id])
                        users.pop(user_id, None)

            if scope in {"shared", "all"}:
                if not is_admin:
                    return removed
                shared = doc.get("shared", [])
                if isinstance(shared, list):
                    removed += len(shared)
                    doc["shared"] = []

            if removed:
                await self._store.async_save(self._data)
        return removed

    async def async_list_sessions(
        self,
        *,
        scope: str,
        limit: int,
        requester_user_id: str | None,
        is_admin: bool,
        subentry_id: str | None = None,
        target_user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List saved resumable chat sessions."""
        if scope not in VALID_SESSION_SCOPES:
            scope = "mine"
        if not is_admin:
            scope = "mine"
        limit = max(1, min(limit, 500))

        allowed_user_ids = self._session_filter_user_ids(
            requester_user_id=requester_user_id,
            target_user_id=target_user_id,
            is_admin=is_admin,
            scope=scope,
        )
        if allowed_user_ids == set():
            return []

        async with self._lock:
            doc = self._entry_doc()
            sessions = self._collect_sessions(
                doc,
                allowed_user_ids=allowed_user_ids,
                subentry_id=subentry_id,
                include_user_id=is_admin,
            )
            sessions = sessions[:limit]
            return sessions

    async def async_get_session(
        self,
        *,
        session_id: str,
        requester_user_id: str | None,
        is_admin: bool,
        limit: int = 40,
    ) -> dict[str, Any] | None:
        """Return one resumable session by session id."""
        parsed = self._parse_session_id(session_id)
        if parsed is None:
            return None
        subentry_id, owner_user_id = parsed

        if not is_admin and owner_user_id != requester_user_id:
            return None

        limit = max(1, min(limit, 500))
        async with self._lock:
            doc = self._entry_doc()
            messages = self._resume_messages(
                doc,
                subentry_id=subentry_id,
                user_id=owner_user_id,
                create=False,
            )
            if not messages:
                return None

            start_index = max(0, len(messages) - limit)
            visible = messages[start_index:]
            first_message = messages[0] if isinstance(messages[0], dict) else {}
            last_message = messages[-1] if isinstance(messages[-1], dict) else {}
            created_at = str(first_message.get("created_at", ""))
            updated_at = str(last_message.get("created_at", ""))
            payload: dict[str, Any] = {
                "session_id": self._make_session_id(subentry_id, owner_user_id),
                "subentry_id": subentry_id,
                "message_count": len(messages),
                "created_at": created_at,
                "updated_at": updated_at,
                "messages": [
                    {
                        "role": str(message.get("role", "")),
                        "content": str(message.get("content", "")),
                        "created_at": str(message.get("created_at", "")),
                    }
                    for message in visible
                    if isinstance(message, dict)
                ],
            }
            if is_admin:
                payload["user_id"] = owner_user_id
            return payload

    async def async_clear_sessions(
        self,
        *,
        requester_user_id: str | None,
        is_admin: bool,
        scope: str = "mine",
        subentry_id: str | None = None,
        target_user_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[int, int]:
        """Clear sessions and return (removed_sessions, removed_messages)."""
        if not is_admin:
            scope = "mine"

        async with self._lock:
            doc = self._entry_doc()
            resume = doc.get("resume", {})
            if not isinstance(resume, dict):
                return 0, 0

            removed_sessions = 0
            removed_messages = 0

            if session_id:
                parsed = self._parse_session_id(session_id)
                if parsed is None:
                    return 0, 0
                target_subentry_id, owner_user_id = parsed
                if not is_admin and owner_user_id != requester_user_id:
                    return 0, 0
                by_user = resume.get(target_subentry_id)
                if not isinstance(by_user, dict):
                    return 0, 0
                messages = by_user.pop(owner_user_id, None)
                if isinstance(messages, list):
                    removed_sessions = 1
                    removed_messages = len(messages)
                if not by_user:
                    resume.pop(target_subentry_id, None)
            else:
                allowed_user_ids = self._session_filter_user_ids(
                    requester_user_id=requester_user_id,
                    target_user_id=target_user_id,
                    is_admin=is_admin,
                    scope=scope,
                )
                if allowed_user_ids == set():
                    return 0, 0

                for current_subentry, by_user in list(resume.items()):
                    if subentry_id and current_subentry != subentry_id:
                        continue
                    if not isinstance(by_user, dict):
                        continue
                    for owner_user_id, messages in list(by_user.items()):
                        if allowed_user_ids is not None and owner_user_id not in allowed_user_ids:
                            continue
                        if not isinstance(messages, list):
                            by_user.pop(owner_user_id, None)
                            continue
                        removed_sessions += 1
                        removed_messages += len(messages)
                        by_user.pop(owner_user_id, None)
                    if not by_user:
                        resume.pop(current_subentry, None)

            if removed_sessions > 0:
                await self._store.async_save(self._data)
            return removed_sessions, removed_messages

    async def async_handle_command(
        self,
        user_input: conversation.ConversationInput,
        subentry_id: str,
    ) -> tuple[bool, str]:
        """Handle slash commands locally."""
        parsed = parse_slash_command(user_input.text)
        if parsed is None:
            return False, ""

        kind = str(parsed.get("kind", ""))
        if kind == "unknown":
            return False, ""
        if kind == "error":
            return True, str(parsed.get("error", "Invalid command"))

        if kind == "reset_context":
            cleared = await self.async_clear_resume_context(user_input, subentry_id)
            if cleared:
                return True, "Started a new conversation context."
            return True, "No saved context found. Starting fresh."

        if kind == "memory_help":
            return True, (
                "Memory commands:\n"
                "/memory status\n"
                "/memory add [--shared] <text>\n"
                "/memory list [mine|shared|all] [--limit N]\n"
                "/memory search <query> [--limit N]\n"
                "/memory delete <memory_id>\n"
                "/memory clear mine|shared|all --confirm\n"
                "/memory sessions [mine|all] [--limit N]\n"
                "/memory sessions show <session_id> [--limit N]\n"
                "/memory sessions clear <session_id|mine|all> --confirm\n"
                "Aliases: /remember, /forget, /memories, /sessions, /new, /reset"
            )

        user_id = user_input.context.user_id if user_input.context else None
        is_admin = await self._async_is_admin(user_id)

        if kind == "memory_status":
            return True, self._render_status(user_id, is_admin=is_admin)

        if kind == "session_list":
            scope = str(parsed.get("scope", "mine"))
            limit = int(parsed.get("limit", 20))
            sessions = await self.async_list_sessions(
                scope=scope,
                limit=limit,
                requester_user_id=user_id,
                is_admin=is_admin,
                subentry_id=None,
            )
            if not sessions:
                return True, "No saved sessions found."
            lines = ["Saved sessions:"]
            for session in sessions:
                session_token = str(session.get("session_id", "unknown"))
                subentry = str(session.get("subentry_id", "unknown"))
                message_count = int(session.get("message_count", 0))
                updated_at = str(session.get("updated_at", ""))
                lines.append(
                    f"- `{session_token}` subentry={subentry} messages={message_count} updated={updated_at}"
                )
            return True, "\n".join(lines)

        if kind == "session_show":
            session_id = str(parsed.get("id", "")).strip()
            limit = int(parsed.get("limit", 40))
            if not session_id:
                return True, "Provide a session ID."
            session = await self.async_get_session(
                session_id=session_id,
                requester_user_id=user_id,
                is_admin=is_admin,
                limit=limit,
            )
            if not session:
                return True, "Session not found or not permitted."
            lines = [
                f"Session `{session.get('session_id', session_id)}`",
                f"subentry: {session.get('subentry_id', 'unknown')}",
                f"messages: {session.get('message_count', 0)}",
            ]
            for message in session.get("messages", []):
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", ""))
                content = str(message.get("content", "")).strip()
                created = str(message.get("created_at", ""))
                lines.append(f"- [{role}] {content} ({created})")
            return True, "\n".join(lines)

        if kind == "session_clear":
            target = str(parsed.get("target", "mine")).strip() or "mine"
            confirmed = bool(parsed.get("confirm", False))
            if not confirmed:
                return True, "Add `--confirm` to clear sessions."
            session_id = None
            scope = "mine"
            if target in VALID_SESSION_SCOPES:
                scope = target
            else:
                session_id = target
            removed_sessions, removed_messages = await self.async_clear_sessions(
                requester_user_id=user_id,
                is_admin=is_admin,
                scope=scope,
                subentry_id=subentry_id if scope == "mine" and session_id is None else None,
                session_id=session_id,
            )
            if removed_sessions < 1:
                return True, "No matching sessions to clear."
            return (
                True,
                f"Cleared {removed_sessions} session(s), {removed_messages} message(s).",
            )

        if not self.is_enabled():
            return True, "Memory is disabled for this service entry. Enable it in integration Configure options."

        if kind == "memory_add":
            text = str(parsed.get("text", "")).strip()
            shared = bool(parsed.get("shared", False))
            if not text:
                return True, "Nothing to remember."
            if looks_sensitive(text):
                return True, "I won't store that because it looks sensitive."
            if shared and not is_admin:
                return True, "Only admin users can add shared memory."
            if not shared and not user_id:
                return True, "Cannot store per-user memory without an authenticated user context."
            item_id = await self._async_add_memory(
                text=text,
                scope="shared" if shared else "user",
                user_id=user_id,
                source="slash",
            )
            if item_id is None:
                return True, "That memory already exists."
            return True, f"Saved memory as `{item_id}`."

        if kind == "memory_list":
            scope = str(parsed.get("scope", "all"))
            limit = int(parsed.get("limit", 20))
            return True, await self._async_list_memory(scope, limit, user_id)

        if kind == "memory_search":
            query = str(parsed.get("query", "")).strip()
            limit = int(parsed.get("limit", 10))
            if not query:
                return True, "Provide a search query."
            return True, await self._async_search_memory(query, limit, user_id)

        if kind == "memory_delete":
            memory_id = str(parsed.get("id", "")).strip()
            if not memory_id:
                return True, "Provide a memory ID."
            deleted, scope = await self._async_delete_memory(memory_id, user_id, is_admin)
            if not deleted:
                return True, "Memory not found or not permitted."
            return True, f"Deleted {scope} memory `{memory_id}`."

        if kind == "memory_clear":
            scope = str(parsed.get("scope", "mine"))
            confirmed = bool(parsed.get("confirm", False))
            if not confirmed:
                return True, "Add `--confirm` to clear memory."
            removed = await self._async_clear_memory(scope, user_id, is_admin)
            if removed < 1:
                return True, "No matching memories to clear."
            return True, f"Cleared {removed} memory item(s)."

        return True, "Unsupported memory command."

    async def async_maybe_capture_heuristic(
        self, user_input: conversation.ConversationInput
    ) -> None:
        """Capture explicit user memory hints from normal chat text."""
        if not self.is_enabled() or not bool(self._option(CONF_MEMORY_AUTO_WRITE)):
            return
        if is_slash_command(user_input.text):
            return

        user_id = user_input.context.user_id if user_input.context else None
        if not user_id:
            return

        captured = extract_heuristic_memory(user_input.text)
        if not captured:
            return

        await self._async_add_memory(
            text=captured,
            scope="user",
            user_id=user_id,
            source="heuristic",
        )

    async def async_build_memory_prompt(
        self, user_input: conversation.ConversationInput
    ) -> str | None:
        """Build auto-recalled memory block for prompt injection."""
        if not self.is_enabled() or not bool(self._option(CONF_MEMORY_AUTO_RECALL)):
            return None
        query = user_input.text.strip()
        if not query or query.startswith("/"):
            return None

        user_id = user_input.context.user_id if user_input.context else None
        top_k = int(self._option(CONF_MEMORY_RECALL_TOP_K))

        async with self._lock:
            doc = self._entry_doc()
            self._prune_entry_locked(doc)
            candidates = self._collect_memories(doc, user_id, scope="all")
            if not candidates:
                await self._store.async_save(self._data)
                return None

        query_embedding = await self._embedder.async_embed_one(query)
        ranked = rank_memory_items(
            candidates,
            query=query,
            now=utcnow(),
            top_k=top_k,
            query_embedding=query_embedding,
        )
        if not ranked:
            return None

        now_iso = utcnow_iso()
        ranked_ids = {item.get("id") for item in ranked}
        async with self._lock:
            doc = self._entry_doc()
            for item in self._collect_memories(doc, user_id, scope="all"):
                if item.get("id") in ranked_ids:
                    item["last_accessed_at"] = now_iso
            await self._store.async_save(self._data)

        return format_memory_prompt(ranked)

    async def async_inject_resume_context(
        self,
        chat_log: conversation.ChatLog,
        user_input: conversation.ConversationInput,
        subentry_id: str,
        agent_id: str,
    ) -> None:
        """Inject saved transcript context into new sessions."""
        if not self.is_enabled() or not bool(self._option(CONF_MEMORY_RESUME_ENABLED)):
            return
        if len(chat_log.content) > 2:
            return

        user_id = user_input.context.user_id if user_input.context else None
        if not user_id:
            return

        async with self._lock:
            doc = self._entry_doc()
            messages = self._resume_messages(doc, subentry_id, user_id)
            if not messages:
                return

        restored: list[conversation.Content] = []
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                restored.append(conversation.UserContent(content=content))
            elif role == "assistant":
                restored.append(
                    conversation.AssistantContent(agent_id=agent_id, content=content)
                )

        if not restored or len(chat_log.content) < 2:
            return
        system = chat_log.content[0]
        current_user = chat_log.content[-1]
        chat_log.content = [system, *restored, current_user]

    async def async_record_turn(
        self,
        user_input: conversation.ConversationInput,
        subentry_id: str,
        assistant_text: str | None,
    ) -> None:
        """Persist a user/assistant turn for exact resume."""
        if not self.is_enabled() or not bool(self._option(CONF_MEMORY_RESUME_ENABLED)):
            return
        if not assistant_text or is_slash_command(user_input.text):
            return
        user_id = user_input.context.user_id if user_input.context else None
        if not user_id:
            return

        now_iso = utcnow_iso()
        max_messages = int(self._option(CONF_MEMORY_RESUME_MAX_MESSAGES))

        async with self._lock:
            doc = self._entry_doc()
            messages = self._resume_messages(doc, subentry_id, user_id, create=True)
            messages.append({"role": "user", "content": user_input.text.strip(), "created_at": now_iso})
            messages.append({"role": "assistant", "content": assistant_text.strip(), "created_at": now_iso})
            if len(messages) > max_messages:
                del messages[: len(messages) - max_messages]
            await self._store.async_save(self._data)

    async def async_clear_resume_context(
        self,
        user_input: conversation.ConversationInput,
        subentry_id: str,
    ) -> bool:
        """Clear resumable transcript context for user + agent."""
        user_id = user_input.context.user_id if user_input.context else None
        if not user_id:
            return False
        async with self._lock:
            doc = self._entry_doc()
            by_subentry = doc.setdefault("resume", {}).get(subentry_id)
            if not isinstance(by_subentry, dict) or user_id not in by_subentry:
                return False
            by_subentry.pop(user_id, None)
            await self._store.async_save(self._data)
            return True

    def _option(self, key: str) -> Any:
        return self._entry.options.get(key, MEMORY_DEFAULTS[key])

    def _entry_doc(self) -> dict[str, Any]:
        entries = self._data.setdefault("entries", {})
        doc = entries.setdefault(
            self._entry.entry_id,
            {
                "shared": [],
                "users": {},
                "resume": {},
            },
        )
        if not isinstance(doc.get("shared"), list):
            doc["shared"] = []
        if not isinstance(doc.get("users"), dict):
            doc["users"] = {}
        if not isinstance(doc.get("resume"), dict):
            doc["resume"] = {}
        return doc

    def _resolve_target_user(
        self,
        *,
        requester_user_id: str | None,
        target_user_id: str | None,
        is_admin: bool,
        allow_all_admin: bool,
    ) -> str | None:
        """Resolve which user scope should be used for a management call."""
        if target_user_id:
            if is_admin:
                return target_user_id
            if requester_user_id == target_user_id:
                return target_user_id
            return None
        if requester_user_id:
            return requester_user_id
        if is_admin and allow_all_admin:
            return None
        return None

    def _collect_memories_for_management(
        self,
        doc: dict[str, Any],
        *,
        scope: str,
        requester_user_id: str | None,
        target_user_id: str | None,
        is_admin: bool,
    ) -> list[dict[str, Any]]:
        """Collect memory items with admin-aware user targeting."""
        items: list[dict[str, Any]] = []

        shared = doc.get("shared", [])
        users = doc.get("users", {})
        if not isinstance(shared, list):
            shared = []
        if not isinstance(users, dict):
            users = {}

        if scope in {"shared", "all"}:
            items.extend(item for item in shared if isinstance(item, dict))

        if scope not in {"mine", "all"}:
            return items

        if is_admin and scope == "all" and not target_user_id:
            for scoped_items in users.values():
                if not isinstance(scoped_items, list):
                    continue
                items.extend(item for item in scoped_items if isinstance(item, dict))
            return items

        user_id = self._resolve_target_user(
            requester_user_id=requester_user_id,
            target_user_id=target_user_id,
            is_admin=is_admin,
            allow_all_admin=False,
        )
        if not user_id:
            return items

        scoped_items = users.get(user_id, [])
        if isinstance(scoped_items, list):
            items.extend(item for item in scoped_items if isinstance(item, dict))
        return items

    def _serialize_memory_item(
        self, item: dict[str, Any], *, include_user_id: bool
    ) -> dict[str, Any]:
        """Return memory item payload safe for service responses."""
        payload: dict[str, Any] = {
            "id": str(item.get("id", "")),
            "scope": str(item.get("scope", "user")),
            "text": str(item.get("text", "")),
            "source": str(item.get("source", "")),
            "created_at": str(item.get("created_at", "")),
            "updated_at": str(item.get("updated_at", "")),
            "last_accessed_at": str(item.get("last_accessed_at", "")),
        }
        if include_user_id:
            payload["user_id"] = str(item.get("user_id", ""))
        return payload

    def _make_session_id(self, subentry_id: str, user_id: str) -> str:
        """Build stable session id from (subentry_id, user_id)."""
        return f"{subentry_id}:{user_id}"

    def _parse_session_id(self, session_id: str) -> tuple[str, str] | None:
        """Parse a stable session id."""
        if ":" not in session_id:
            return None
        subentry_id, user_id = session_id.split(":", 1)
        if not subentry_id or not user_id:
            return None
        return subentry_id, user_id

    def _session_filter_user_ids(
        self,
        *,
        requester_user_id: str | None,
        target_user_id: str | None,
        is_admin: bool,
        scope: str,
    ) -> set[str] | None:
        """Return allowed user ids for session queries; None means all users."""
        if not is_admin:
            if requester_user_id:
                return {requester_user_id}
            return set()

        if target_user_id:
            return {target_user_id}

        if scope == "all":
            return None

        if requester_user_id:
            return {requester_user_id}
        return set()

    def _collect_sessions(
        self,
        doc: dict[str, Any],
        *,
        allowed_user_ids: set[str] | None,
        subentry_id: str | None,
        include_user_id: bool,
    ) -> list[dict[str, Any]]:
        """Collect resumable session summaries."""
        resume = doc.get("resume", {})
        if not isinstance(resume, dict):
            return []

        sessions: list[dict[str, Any]] = []
        for current_subentry, by_user in resume.items():
            if subentry_id and current_subentry != subentry_id:
                continue
            if not isinstance(by_user, dict):
                continue
            for owner_user_id, messages in by_user.items():
                if allowed_user_ids is not None and owner_user_id not in allowed_user_ids:
                    continue
                if not isinstance(messages, list) or not messages:
                    continue

                first = messages[0] if isinstance(messages[0], dict) else {}
                last = messages[-1] if isinstance(messages[-1], dict) else {}
                session_payload: dict[str, Any] = {
                    "session_id": self._make_session_id(current_subentry, owner_user_id),
                    "subentry_id": current_subentry,
                    "message_count": len(messages),
                    "created_at": str(first.get("created_at", "")),
                    "updated_at": str(last.get("created_at", "")),
                }
                if include_user_id:
                    session_payload["user_id"] = owner_user_id
                sessions.append(session_payload)

        sessions.sort(
            key=lambda item: parse_iso8601(str(item.get("updated_at", ""))).timestamp(),
            reverse=True,
        )
        return sessions

    def _collect_memories(
        self, doc: dict[str, Any], user_id: str | None, scope: str
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if scope in {"all", "mine"} and user_id:
            user_items = doc.get("users", {}).get(user_id, [])
            if isinstance(user_items, list):
                items.extend(user_items)
        if scope in {"all", "shared"}:
            shared = doc.get("shared", [])
            if isinstance(shared, list):
                items.extend(shared)
        return items

    def _prune_entry_locked(self, doc: dict[str, Any]) -> None:
        ttl_days = int(self._option(CONF_MEMORY_TTL_DAYS))
        max_items = int(self._option(CONF_MEMORY_MAX_ITEMS_PER_SCOPE))
        now = utcnow()
        doc["shared"] = prune_memory_items(
            list(doc.get("shared", [])),
            ttl_days=ttl_days,
            max_items=max_items,
            now=now,
        )
        users = doc.get("users", {})
        if not isinstance(users, dict):
            users = {}
            doc["users"] = users
        for user_id, items in list(users.items()):
            if not isinstance(items, list):
                users[user_id] = []
                continue
            pruned = prune_memory_items(items, ttl_days=ttl_days, max_items=max_items, now=now)
            if pruned:
                users[user_id] = pruned
            else:
                users.pop(user_id, None)

    def _resume_messages(
        self,
        doc: dict[str, Any],
        subentry_id: str,
        user_id: str,
        *,
        create: bool = False,
    ) -> list[dict[str, Any]]:
        resume = doc.setdefault("resume", {})
        if not isinstance(resume, dict):
            doc["resume"] = {}
            resume = doc["resume"]
        by_subentry = resume.get(subentry_id)
        if not isinstance(by_subentry, dict):
            if not create:
                return []
            by_subentry = {}
            resume[subentry_id] = by_subentry
        messages = by_subentry.get(user_id)
        if not isinstance(messages, list):
            if not create:
                return []
            messages = []
            by_subentry[user_id] = messages
        return messages

    async def _async_add_memory(
        self,
        *,
        text: str,
        scope: str,
        user_id: str | None,
        source: str,
    ) -> str | None:
        text = text.strip()
        if not text:
            return None

        embedding = await self._embedder.async_embed_one(text)
        now_iso = utcnow_iso()
        async with self._lock:
            doc = self._entry_doc()
            self._prune_entry_locked(doc)
            if scope == "shared":
                target = doc["shared"]
            else:
                if not user_id:
                    return None
                users = doc.setdefault("users", {})
                target = users.setdefault(user_id, [])

            existing = [str(item.get("text", "")) for item in target if isinstance(item, dict)]
            if is_duplicate_memory(text, existing):
                return None

            item_id = uuid.uuid4().hex[:12]
            target.append(
                {
                    "id": item_id,
                    "text": text,
                    "scope": scope,
                    "user_id": None if scope == "shared" else user_id,
                    "source": source,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                    "last_accessed_at": now_iso,
                    "embedding": embedding,
                }
            )
            self._prune_entry_locked(doc)
            await self._store.async_save(self._data)
            return item_id

    async def _async_list_memory(self, scope: str, limit: int, user_id: str | None) -> str:
        async with self._lock:
            doc = self._entry_doc()
            self._prune_entry_locked(doc)
            items = self._collect_memories(doc, user_id, scope)
            items.sort(
                key=lambda item: parse_iso8601(item.get("updated_at")).timestamp(),
                reverse=True,
            )
            items = items[:limit]
            await self._store.async_save(self._data)

        if not items:
            return "No memories found."
        lines = ["Memories:"]
        for item in items:
            memory_id = item.get("id", "unknown")
            memory_scope = item.get("scope", "user")
            text = str(item.get("text", "")).strip()
            lines.append(f"- `{memory_id}` [{memory_scope}] {text}")
        return "\n".join(lines)

    async def _async_search_memory(self, query: str, limit: int, user_id: str | None) -> str:
        async with self._lock:
            doc = self._entry_doc()
            self._prune_entry_locked(doc)
            items = self._collect_memories(doc, user_id, scope="all")
            await self._store.async_save(self._data)

        if not items:
            return "No memories found."

        query_embedding = await self._embedder.async_embed_one(query)
        ranked = rank_memory_items(
            items,
            query=query,
            top_k=limit,
            now=utcnow(),
            query_embedding=query_embedding,
        )
        if not ranked:
            return "No matching memories."

        lines = [f"Memory search results for '{query}':"]
        for item in ranked:
            memory_id = item.get("id", "unknown")
            memory_scope = item.get("scope", "user")
            text = str(item.get("text", "")).strip()
            lines.append(f"- `{memory_id}` [{memory_scope}] {text}")
        return "\n".join(lines)

    async def _async_delete_memory(
        self, memory_id: str, user_id: str | None, is_admin: bool
    ) -> tuple[bool, str]:
        async with self._lock:
            doc = self._entry_doc()

            shared = doc.get("shared", [])
            if isinstance(shared, list):
                for idx, item in enumerate(shared):
                    if not isinstance(item, dict):
                        continue
                    if item.get("id") != memory_id:
                        continue
                    if not is_admin:
                        return False, "shared"
                    shared.pop(idx)
                    await self._store.async_save(self._data)
                    return True, "shared"

            users = doc.get("users", {})
            if isinstance(users, dict):
                for owner_id, items in list(users.items()):
                    if not isinstance(items, list):
                        continue
                    for idx, item in enumerate(items):
                        if not isinstance(item, dict):
                            continue
                        if item.get("id") != memory_id:
                            continue
                        if not is_admin and owner_id != user_id:
                            return False, "user"
                        items.pop(idx)
                        if not items:
                            users.pop(owner_id, None)
                        await self._store.async_save(self._data)
                        return True, "user"
        return False, "unknown"

    async def _async_clear_memory(
        self, scope: str, user_id: str | None, is_admin: bool
    ) -> int:
        removed = 0
        async with self._lock:
            doc = self._entry_doc()
            if scope in {"mine", "all"}:
                if not user_id:
                    return 0
                users = doc.get("users", {})
                if isinstance(users, dict) and isinstance(users.get(user_id), list):
                    removed += len(users[user_id])
                    users.pop(user_id, None)
            if scope in {"shared", "all"}:
                if not is_admin:
                    return 0
                shared = doc.get("shared", [])
                if isinstance(shared, list):
                    removed += len(shared)
                    doc["shared"] = []
            if removed:
                await self._store.async_save(self._data)
        return removed

    async def _async_is_admin(self, user_id: str | None) -> bool:
        if not user_id:
            return False
        user = await self._hass.auth.async_get_user(user_id)
        return bool(user and user.is_admin)

    def _render_status(self, user_id: str | None, *, is_admin: bool) -> str:
        enabled = self.is_enabled()
        own_memories = 0
        shared_memories = 0
        sessions_mine = 0
        sessions_total = 0
        doc = self._entry_doc()
        shared = doc.get("shared", [])
        if isinstance(shared, list):
            shared_memories = len(shared)
        users = doc.get("users", {})
        if isinstance(users, dict) and user_id and isinstance(users.get(user_id), list):
            own_memories = len(users[user_id])
        resume = doc.get("resume", {})
        if isinstance(resume, dict):
            for by_user in resume.values():
                if not isinstance(by_user, dict):
                    continue
                for owner_id, messages in by_user.items():
                    if not isinstance(messages, list) or not messages:
                        continue
                    sessions_total += 1
                    if user_id and owner_id == user_id:
                        sessions_mine += 1

        counts_text = (
            f"\nShared memories: {shared_memories}\n"
            f"Your memories: {own_memories}\n"
            f"Your sessions: {sessions_mine}"
        )
        if is_admin:
            counts_text += f"\nAll sessions: {sessions_total}"
        return (
            f"Memory enabled: {enabled}\n"
            f"Auto write: {bool(self._option(CONF_MEMORY_AUTO_WRITE))}\n"
            f"Auto recall: {bool(self._option(CONF_MEMORY_AUTO_RECALL))}\n"
            f"Resume context: {bool(self._option(CONF_MEMORY_RESUME_ENABLED))}\n"
            f"TTL days: {int(self._option(CONF_MEMORY_TTL_DAYS))}\n"
            f"Max items/scope: {int(self._option(CONF_MEMORY_MAX_ITEMS_PER_SCOPE))}\n"
            f"Recall top-k: {int(self._option(CONF_MEMORY_RECALL_TOP_K))}"
            f"{counts_text}"
        )


def _get_services(hass: HomeAssistant) -> dict[str, ClaudeAssistMemoryService]:
    domain_data = hass.data.setdefault(DOMAIN, {})
    services = domain_data.get(DATA_MEMORY_SERVICES)
    if not isinstance(services, dict):
        services = {}
        domain_data[DATA_MEMORY_SERVICES] = services
    return services


def _domain_data(hass: HomeAssistant) -> dict[str, Any]:
    return hass.data.setdefault(DOMAIN, {})


def _resolve_entry_memory_service(
    hass: HomeAssistant, config_entry_id: str | None
) -> tuple[str, ClaudeAssistMemoryService]:
    services = _get_services(hass)
    if config_entry_id:
        selected = services.get(config_entry_id)
        if selected is None:
            raise HomeAssistantError(
                f"Config entry '{config_entry_id}' is not loaded for {DOMAIN}."
            )
        return config_entry_id, selected

    if not services:
        raise HomeAssistantError("No loaded AI Subscription Assist entries found.")

    if len(services) == 1:
        entry_id, selected = next(iter(services.items()))
        return entry_id, selected

    raise HomeAssistantError(
        "Multiple AI Subscription Assist entries are loaded. Provide config_entry_id."
    )


async def async_setup_memory_domain_services(hass: HomeAssistant) -> None:
    """Register domain services used for memory/session management."""
    domain_data = _domain_data(hass)
    if domain_data.get(DATA_MEMORY_DOMAIN_SERVICES_REGISTERED):
        return

    base_schema = {vol.Optional("config_entry_id"): cv.string}
    memory_scope = vol.In({"mine", "shared", "all"})
    session_scope = vol.In(VALID_SESSION_SCOPES)

    async def _handle_memory_status(call: ServiceCall) -> dict[str, Any]:
        entry_id, memory_service = _resolve_entry_memory_service(
            hass, call.data.get("config_entry_id")
        )
        requester_user_id = call.context.user_id
        is_admin = await memory_service.async_user_is_admin(requester_user_id)
        snapshot = await memory_service.async_status_snapshot(
            requester_user_id, is_admin=is_admin
        )
        return snapshot

    async def _handle_memory_list(call: ServiceCall) -> dict[str, Any]:
        entry_id, memory_service = _resolve_entry_memory_service(
            hass, call.data.get("config_entry_id")
        )
        requester_user_id = call.context.user_id
        is_admin = await memory_service.async_user_is_admin(requester_user_id)
        items = await memory_service.async_list_memory_items(
            scope=str(call.data.get("scope", "mine")),
            limit=int(call.data.get("limit", 50)),
            requester_user_id=requester_user_id,
            is_admin=is_admin,
            target_user_id=call.data.get("target_user_id"),
        )
        return {"entry_id": entry_id, "count": len(items), "items": items}

    async def _handle_memory_delete(call: ServiceCall) -> dict[str, Any]:
        entry_id, memory_service = _resolve_entry_memory_service(
            hass, call.data.get("config_entry_id")
        )
        requester_user_id = call.context.user_id
        is_admin = await memory_service.async_user_is_admin(requester_user_id)
        deleted, scope = await memory_service.async_delete_memory_item(
            str(call.data["memory_id"]),
            requester_user_id,
            is_admin=is_admin,
        )
        return {
            "entry_id": entry_id,
            "deleted": deleted,
            "memory_id": str(call.data["memory_id"]),
            "scope": scope,
        }

    async def _handle_memory_clear(call: ServiceCall) -> dict[str, Any]:
        if not bool(call.data.get("confirm", False)):
            raise HomeAssistantError("Set confirm: true to clear memories.")
        entry_id, memory_service = _resolve_entry_memory_service(
            hass, call.data.get("config_entry_id")
        )
        requester_user_id = call.context.user_id
        is_admin = await memory_service.async_user_is_admin(requester_user_id)
        removed = await memory_service.async_clear_memory_items(
            scope=str(call.data.get("scope", "mine")),
            requester_user_id=requester_user_id,
            is_admin=is_admin,
            target_user_id=call.data.get("target_user_id"),
        )
        return {"entry_id": entry_id, "removed": removed}

    async def _handle_session_list(call: ServiceCall) -> dict[str, Any]:
        entry_id, memory_service = _resolve_entry_memory_service(
            hass, call.data.get("config_entry_id")
        )
        requester_user_id = call.context.user_id
        is_admin = await memory_service.async_user_is_admin(requester_user_id)
        sessions = await memory_service.async_list_sessions(
            scope=str(call.data.get("scope", "mine")),
            limit=int(call.data.get("limit", 50)),
            requester_user_id=requester_user_id,
            is_admin=is_admin,
            subentry_id=call.data.get("subentry_id"),
            target_user_id=call.data.get("target_user_id"),
        )
        return {"entry_id": entry_id, "count": len(sessions), "sessions": sessions}

    async def _handle_session_get(call: ServiceCall) -> dict[str, Any]:
        entry_id, memory_service = _resolve_entry_memory_service(
            hass, call.data.get("config_entry_id")
        )
        requester_user_id = call.context.user_id
        is_admin = await memory_service.async_user_is_admin(requester_user_id)
        session = await memory_service.async_get_session(
            session_id=str(call.data["session_id"]),
            requester_user_id=requester_user_id,
            is_admin=is_admin,
            limit=int(call.data.get("limit", 100)),
        )
        if session is None:
            raise HomeAssistantError("Session not found or not permitted.")
        return {"entry_id": entry_id, "session": session}

    async def _handle_session_clear(call: ServiceCall) -> dict[str, Any]:
        if not bool(call.data.get("confirm", False)):
            raise HomeAssistantError("Set confirm: true to clear sessions.")
        entry_id, memory_service = _resolve_entry_memory_service(
            hass, call.data.get("config_entry_id")
        )
        requester_user_id = call.context.user_id
        is_admin = await memory_service.async_user_is_admin(requester_user_id)
        removed_sessions, removed_messages = await memory_service.async_clear_sessions(
            requester_user_id=requester_user_id,
            is_admin=is_admin,
            scope=str(call.data.get("scope", "mine")),
            subentry_id=call.data.get("subentry_id"),
            target_user_id=call.data.get("target_user_id"),
            session_id=call.data.get("session_id"),
        )
        return {
            "entry_id": entry_id,
            "removed_sessions": removed_sessions,
            "removed_messages": removed_messages,
        }

    hass.services.async_register(
        DOMAIN,
        SERVICE_MEMORY_STATUS,
        _handle_memory_status,
        schema=vol.Schema(base_schema),
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MEMORY_LIST,
        _handle_memory_list,
        schema=vol.Schema(
            {
                **base_schema,
                vol.Optional("scope", default="mine"): memory_scope,
                vol.Optional("limit", default=50): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=500)
                ),
                vol.Optional("target_user_id"): cv.string,
            }
        ),
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MEMORY_DELETE,
        _handle_memory_delete,
        schema=vol.Schema(
            {
                **base_schema,
                vol.Required("memory_id"): cv.string,
            }
        ),
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MEMORY_CLEAR,
        _handle_memory_clear,
        schema=vol.Schema(
            {
                **base_schema,
                vol.Optional("scope", default="mine"): memory_scope,
                vol.Optional("target_user_id"): cv.string,
                vol.Optional("confirm", default=False): bool,
            }
        ),
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_SESSION_LIST,
        _handle_session_list,
        schema=vol.Schema(
            {
                **base_schema,
                vol.Optional("scope", default="mine"): session_scope,
                vol.Optional("subentry_id"): cv.string,
                vol.Optional("target_user_id"): cv.string,
                vol.Optional("limit", default=50): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=500)
                ),
            }
        ),
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_SESSION_GET,
        _handle_session_get,
        schema=vol.Schema(
            {
                **base_schema,
                vol.Required("session_id"): cv.string,
                vol.Optional("limit", default=100): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=500)
                ),
            }
        ),
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_SESSION_CLEAR,
        _handle_session_clear,
        schema=vol.Schema(
            {
                **base_schema,
                vol.Optional("scope", default="mine"): session_scope,
                vol.Optional("subentry_id"): cv.string,
                vol.Optional("target_user_id"): cv.string,
                vol.Optional("session_id"): cv.string,
                vol.Optional("confirm", default=False): bool,
            }
        ),
        supports_response=SupportsResponse.OPTIONAL,
    )
    domain_data[DATA_MEMORY_DOMAIN_SERVICES_REGISTERED] = True


def _async_unload_memory_domain_services(hass: HomeAssistant) -> None:
    """Remove domain services for memory/session management."""
    domain_data = _domain_data(hass)
    if not domain_data.get(DATA_MEMORY_DOMAIN_SERVICES_REGISTERED):
        return

    for service_name in (
        SERVICE_MEMORY_STATUS,
        SERVICE_MEMORY_LIST,
        SERVICE_MEMORY_DELETE,
        SERVICE_MEMORY_CLEAR,
        SERVICE_SESSION_LIST,
        SERVICE_SESSION_GET,
        SERVICE_SESSION_CLEAR,
    ):
        if hass.services.has_service(DOMAIN, service_name):
            hass.services.async_remove(DOMAIN, service_name)

    domain_data[DATA_MEMORY_DOMAIN_SERVICES_REGISTERED] = False


async def async_setup_memory_service_for_entry(
    hass: HomeAssistant, entry: ConfigEntry
) -> ClaudeAssistMemoryService:
    """Create and register an entry memory service."""
    await async_setup_memory_domain_services(hass)
    service = ClaudeAssistMemoryService(hass, entry)
    await service.async_initialize()
    _get_services(hass)[entry.entry_id] = service
    return service


def async_remove_memory_service_for_entry(hass: HomeAssistant, entry_id: str) -> None:
    """Remove an entry memory service."""
    services = _get_services(hass)
    services.pop(entry_id, None)
    if not services:
        _async_unload_memory_domain_services(hass)


def get_memory_service(
    hass: HomeAssistant, entry_id: str
) -> ClaudeAssistMemoryService | None:
    """Return memory service for entry."""
    return _get_services(hass).get(entry_id)
