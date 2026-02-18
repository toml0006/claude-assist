"""Pure helpers for memory commands, extraction, ranking, and retention."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
import math
import re
import shlex
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9_]+")
SENSITIVE_PATTERNS = (
    re.compile(r"\b(api[\s_-]?key|token|secret|password|passwd|pwd)\b", re.I),
    re.compile(r"\bsk-[a-zA-Z0-9]{16,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b"),
)
HEURISTIC_PATTERNS = (
    re.compile(r"^\s*remember that\s+(.+)$", re.I),
    re.compile(r"^\s*remember\s+(.+)$", re.I),
    re.compile(r"^\s*for future reference[,:\s]+(.+)$", re.I),
    re.compile(r"^\s*my preference is\s+(.+)$", re.I),
    re.compile(r"^\s*i prefer\s+(.+)$", re.I),
    re.compile(r"^\s*call me\s+(.+)$", re.I),
)
SLASH_ALIASES = {"/remember", "/forget", "/memories", "/new", "/reset"}
VALID_MEMORY_SCOPES = {"mine", "shared", "all"}


def utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    """Return UTC now in ISO format."""
    return utcnow().isoformat()


def parse_iso8601(value: str | None) -> datetime:
    """Parse timestamp or return now when invalid."""
    if not value:
        return utcnow()
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return utcnow()


def normalize_text(value: str) -> str:
    """Normalize text for dedupe/ranking."""
    return " ".join(value.strip().lower().split())


def tokenize(value: str) -> set[str]:
    """Tokenize text for lexical matching."""
    return set(TOKEN_RE.findall(normalize_text(value)))


def is_slash_command(text: str) -> bool:
    """Return if user text is a slash command."""
    return text.strip().startswith("/")


def _parse_limit(args: list[str], default: int, maximum: int = 100) -> int:
    """Parse --limit from args and clamp."""
    limit = default
    for idx, token in enumerate(args):
        if token == "--limit" and idx + 1 < len(args):
            try:
                limit = int(args[idx + 1])
            except ValueError:
                pass
            break
    return max(1, min(maximum, limit))


def _without_flags(args: Iterable[str], flags: set[str]) -> list[str]:
    """Remove simple boolean flags from args."""
    return [arg for arg in args if arg not in flags]


def parse_slash_command(text: str) -> dict[str, Any] | None:
    """Parse command text into a normalized action payload."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    try:
        argv = shlex.split(stripped)
    except ValueError:
        return {"kind": "error", "error": "Unable to parse command syntax."}

    if not argv:
        return {"kind": "error", "error": "Empty command."}

    cmd = argv[0].lower()

    # Alias: /sessions ... -> /memory sessions ...
    if cmd == "/sessions":
        argv = ["/memory", "sessions", *argv[1:]]
        cmd = "/memory"

    if cmd in {"/new", "/reset"}:
        return {"kind": "reset_context"}

    if cmd == "/remember":
        args = argv[1:]
        shared = "--shared" in args
        payload = " ".join(_without_flags(args, {"--shared"})).strip()
        if not payload:
            return {"kind": "error", "error": "Usage: /remember [--shared] <text>"}
        return {"kind": "memory_add", "text": payload, "shared": shared}

    if cmd == "/forget":
        if len(argv) < 2:
            return {"kind": "error", "error": "Usage: /forget <memory_id>"}
        return {"kind": "memory_delete", "id": argv[1]}

    if cmd == "/memories":
        args = argv[1:]
        if not args:
            return {"kind": "memory_list", "scope": "all", "limit": 20}
        limit = _parse_limit(args, 20)
        query_parts: list[str] = []
        skip_next = False
        for token in args:
            if skip_next:
                skip_next = False
                continue
            if token == "--limit":
                skip_next = True
                continue
            query_parts.append(token)
        query = " ".join(query_parts).strip()
        if query:
            return {"kind": "memory_search", "query": query, "limit": limit}
        return {"kind": "memory_list", "scope": "all", "limit": limit}

    if cmd != "/memory":
        if cmd in SLASH_ALIASES:
            return {"kind": "error", "error": "Unsupported alias usage."}
        return {"kind": "unknown"}

    if len(argv) == 1:
        return {"kind": "memory_help"}

    subcmd = argv[1].lower()
    args = argv[2:]

    if subcmd == "help":
        return {"kind": "memory_help"}
    if subcmd == "status":
        return {"kind": "memory_status"}
    if subcmd == "add":
        shared = "--shared" in args
        payload = " ".join(_without_flags(args, {"--shared"})).strip()
        if not payload:
            return {"kind": "error", "error": "Usage: /memory add [--shared] <text>"}
        return {"kind": "memory_add", "text": payload, "shared": shared}
    if subcmd == "list":
        scope = "all"
        for token in args:
            if token in VALID_MEMORY_SCOPES:
                scope = token
                break
        return {"kind": "memory_list", "scope": scope, "limit": _parse_limit(args, 20)}
    if subcmd == "search":
        if not args:
            return {"kind": "error", "error": "Usage: /memory search <query> [--limit N]"}
        limit = _parse_limit(args, 10)
        query_parts: list[str] = []
        skip_next = False
        for idx, token in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if token == "--limit":
                skip_next = True
                continue
            if idx > 0 and args[idx - 1] == "--limit":
                continue
            query_parts.append(token)
        query = " ".join(query_parts).strip()
        if not query:
            return {"kind": "error", "error": "Usage: /memory search <query> [--limit N]"}
        return {"kind": "memory_search", "query": query, "limit": limit}
    if subcmd == "delete":
        if not args:
            return {"kind": "error", "error": "Usage: /memory delete <memory_id>"}
        return {"kind": "memory_delete", "id": args[0]}
    if subcmd == "clear":
        scope = "mine"
        if args and args[0] in VALID_MEMORY_SCOPES:
            scope = args[0]
        confirmed = "--confirm" in args
        return {"kind": "memory_clear", "scope": scope, "confirm": confirmed}
    if subcmd == "sessions":
        if not args:
            return {"kind": "session_list", "scope": "mine", "limit": 20}

        action = args[0].lower()
        if action == "show":
            if len(args) < 2:
                return {
                    "kind": "error",
                    "error": "Usage: /memory sessions show <session_id> [--limit N]",
                }
            return {
                "kind": "session_show",
                "id": args[1],
                "limit": _parse_limit(args[2:], 40),
            }

        if action == "clear":
            clear_args = args[1:]
            target = "mine"
            for token in clear_args:
                if token != "--confirm":
                    target = token
                    break
            confirmed = "--confirm" in clear_args
            return {"kind": "session_clear", "target": target, "confirm": confirmed}

        if action in {"mine", "all"}:
            return {
                "kind": "session_list",
                "scope": action,
                "limit": _parse_limit(args[1:], 20),
            }

        if action == "--limit":
            return {"kind": "session_list", "scope": "mine", "limit": _parse_limit(args, 20)}

        return {
            "kind": "error",
            "error": (
                "Usage: /memory sessions [mine|all] [--limit N] | "
                "/memory sessions show <session_id> [--limit N] | "
                "/memory sessions clear <session_id|mine|all> --confirm"
            ),
        }

    return {"kind": "error", "error": f"Unknown /memory subcommand '{subcmd}'."}


def looks_sensitive(text: str) -> bool:
    """Detect obvious secret-like content."""
    normalized = text.strip()
    if len(normalized) < 8:
        return False
    return any(pattern.search(normalized) for pattern in SENSITIVE_PATTERNS)


def extract_heuristic_memory(text: str) -> str | None:
    """Extract memory-worthy content from natural language."""
    if text.strip().startswith("/"):
        return None
    for pattern in HEURISTIC_PATTERNS:
        match = pattern.match(text.strip())
        if not match:
            continue
        candidate = match.group(1).strip().rstrip(".! ")
        if len(candidate) < 4 or len(candidate) > 400:
            return None
        if looks_sensitive(candidate):
            return None
        return candidate
    return None


def jaccard_similarity(a: str, b: str) -> float:
    """Return token-based Jaccard similarity."""
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return intersection / union


def is_duplicate_memory(candidate: str, existing_texts: Iterable[str]) -> bool:
    """Return true if candidate is near-duplicate of existing text."""
    norm_candidate = normalize_text(candidate)
    for text in existing_texts:
        norm_existing = normalize_text(text)
        if norm_existing == norm_candidate:
            return True
        if jaccard_similarity(norm_existing, norm_candidate) >= 0.88:
            return True
    return False


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [0, 1]."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    score = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, (score + 1) / 2 if score < 0 else score))


def lexical_score(query: str, text: str) -> float:
    """Lightweight lexical relevance score."""
    q_tokens = tokenize(query)
    t_tokens = tokenize(text)
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / max(len(q_tokens), 1)


def recency_score(updated_at: str | None, now: datetime) -> float:
    """Compute a decayed recency score in [0, 1]."""
    dt = parse_iso8601(updated_at)
    age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return math.exp(-age_days / 30.0)


def rank_memory_items(
    items: list[dict[str, Any]],
    query: str,
    now: datetime | None = None,
    top_k: int = 5,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Rank memory items by semantic/lexical relevance."""
    if now is None:
        now = utcnow()
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in items:
        text = str(item.get("text", ""))
        if not text:
            continue
        lex = lexical_score(query, text)
        sem = 0.0
        embedding = item.get("embedding")
        if (
            query_embedding
            and isinstance(embedding, list)
            and all(isinstance(v, (int, float)) for v in embedding)
        ):
            sem = cosine_similarity(query_embedding, [float(v) for v in embedding])
            score = 0.55 * sem + 0.35 * lex + 0.10 * recency_score(
                item.get("updated_at"), now
            )
        else:
            score = 0.75 * lex + 0.25 * recency_score(item.get("updated_at"), now)
        scored.append((score, item))

    scored.sort(
        key=lambda pair: (
            pair[0],
            parse_iso8601(pair[1].get("updated_at")).timestamp(),
        ),
        reverse=True,
    )
    return [item for _score, item in scored[: max(1, top_k)]]


def prune_memory_items(
    items: list[dict[str, Any]],
    ttl_days: int,
    max_items: int,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Apply TTL and max-item retention."""
    if now is None:
        now = utcnow()
    ttl_cutoff = now - timedelta(days=max(1, ttl_days))
    retained = [
        item
        for item in items
        if parse_iso8601(item.get("updated_at") or item.get("created_at")) >= ttl_cutoff
    ]
    retained.sort(
        key=lambda item: parse_iso8601(item.get("updated_at") or item.get("created_at")),
        reverse=True,
    )
    return retained[: max(1, max_items)]


def format_memory_prompt(recalled: list[dict[str, Any]]) -> str | None:
    """Create a compact system prompt block with recalled memories."""
    if not recalled:
        return None
    lines = [
        "Long-term memory context:",
        "Use these only when relevant. User's current request has priority.",
    ]
    for item in recalled:
        scope = item.get("scope", "user")
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if scope == "shared":
            lines.append(f"- [shared] {text}")
        else:
            lines.append(f"- [user] {text}")
    if len(lines) <= 2:
        return None
    return "\n".join(lines)
