"""Service-level memory manager for AI Subscription Assist."""

from __future__ import annotations

import asyncio
from typing import Any
import uuid

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

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
                "Aliases: /remember, /forget, /memories, /new, /reset"
            )

        if kind == "memory_status":
            return True, self._render_status()

        if not self.is_enabled():
            return True, "Memory is disabled for this service entry. Enable it in integration Configure options."

        user_id = user_input.context.user_id if user_input.context else None
        is_admin = await self._async_is_admin(user_id)

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

    def _render_status(self) -> str:
        enabled = self.is_enabled()
        return (
            f"Memory enabled: {enabled}\n"
            f"Auto write: {bool(self._option(CONF_MEMORY_AUTO_WRITE))}\n"
            f"Auto recall: {bool(self._option(CONF_MEMORY_AUTO_RECALL))}\n"
            f"Resume context: {bool(self._option(CONF_MEMORY_RESUME_ENABLED))}\n"
            f"TTL days: {int(self._option(CONF_MEMORY_TTL_DAYS))}\n"
            f"Max items/scope: {int(self._option(CONF_MEMORY_MAX_ITEMS_PER_SCOPE))}\n"
            f"Recall top-k: {int(self._option(CONF_MEMORY_RECALL_TOP_K))}"
        )


def _get_services(hass: HomeAssistant) -> dict[str, ClaudeAssistMemoryService]:
    domain_data = hass.data.setdefault(DOMAIN, {})
    services = domain_data.get(DATA_MEMORY_SERVICES)
    if not isinstance(services, dict):
        services = {}
        domain_data[DATA_MEMORY_SERVICES] = services
    return services


async def async_setup_memory_service_for_entry(
    hass: HomeAssistant, entry: ConfigEntry
) -> ClaudeAssistMemoryService:
    """Create and register an entry memory service."""
    service = ClaudeAssistMemoryService(hass, entry)
    await service.async_initialize()
    _get_services(hass)[entry.entry_id] = service
    return service


def async_remove_memory_service_for_entry(hass: HomeAssistant, entry_id: str) -> None:
    """Remove an entry memory service."""
    _get_services(hass).pop(entry_id, None)


def get_memory_service(
    hass: HomeAssistant, entry_id: str
) -> ClaudeAssistMemoryService | None:
    """Return memory service for entry."""
    return _get_services(hass).get(entry_id)
