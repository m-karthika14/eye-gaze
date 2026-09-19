"""Groq-backed word/phrase completion for the product UI (Phase 4). Given a
typed prefix and recent context, returns ranked whole-word candidates (e.g.
"hel" -> "hello", "help", "held" -- Section 9's example). Product-layer
only: never used inside the measured decoder path in /decoder; see
MASTER_PROJECT_DOCUMENT.md Section 13.

Caches repeated (prefix, context) queries, and falls back to a local word
list when Groq is unreachable or times out, so suggestions keep working
(degraded) without the API -- part of Phase 8's robustness requirement,
built in now since it's the natural place for it.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from .groq_client import DEFAULT_SUGGESTION_MODEL, GroqClient

VOCAB_PATH = Path(__file__).resolve().parent.parent / "config" / "vocabulary.txt"
CACHE_MAX_SIZE = 500
GROQ_TIMEOUT_S = 3.0

SYSTEM_PROMPT = (
    "You are a word-completion assistant for an assistive eye-gaze typing "
    "keyboard used by people with severe motor/speech impairments. Given a "
    "partially-typed word prefix and the sentence context so far, return the "
    "most likely English words that complete the prefix, ranked most likely "
    "first. Output ONLY the completed words, one per line, lowercase, no "
    "numbering or punctuation or explanation. Every word must start with the "
    "given prefix exactly."
)


def _load_vocabulary() -> list[str]:
    if not VOCAB_PATH.exists():
        return []
    words = []
    for line in VOCAB_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line.lower())
    return words


class WordSuggester:
    def __init__(self, client: GroqClient | None = None) -> None:
        self._client = client  # lazily created on first use if None, so a missing API key
        self._cache: OrderedDict[tuple[str, str], list[str]] = OrderedDict()
        self._vocabulary = _load_vocabulary()

    def _get_client(self) -> GroqClient:
        if self._client is None:
            self._client = GroqClient()
        return self._client

    def _fallback(self, prefix: str, max_suggestions: int) -> list[str]:
        prefix_lower = prefix.lower()
        matches = [w for w in self._vocabulary if w.startswith(prefix_lower) and w != prefix_lower]
        return matches[:max_suggestions]

    def suggest(self, prefix: str, context: str = "", max_suggestions: int = 3) -> list[str]:
        """Returns up to max_suggestions lowercase word completions for
        `prefix`, most likely first. Falls back to a local word list if the
        Groq API is unreachable or errors."""
        prefix = prefix.strip().lower()
        if not prefix:
            return []

        cache_key = (prefix, context.strip().lower())
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key][:max_suggestions]

        try:
            client = self._get_client()
            user_prompt = f'Context so far: "{context}"\nPartial word: "{prefix}"\nGive {max_suggestions} likely completions.'
            reply = client.chat_completion(
                prompt=user_prompt,
                model=DEFAULT_SUGGESTION_MODEL,
                system=SYSTEM_PROMPT,
                max_tokens=128,
                temperature=0.3,
                timeout=GROQ_TIMEOUT_S,
            )
            candidates = [line.strip().lower() for line in reply.splitlines() if line.strip()]
            # Trust but verify: only keep candidates that actually start with
            # the prefix, in case the model didn't follow instructions exactly.
            suggestions = [w for w in candidates if w.startswith(prefix)][:max_suggestions]
            if not suggestions:
                suggestions = self._fallback(prefix, max_suggestions)
        except Exception:  # noqa: BLE001 -- any Groq/network failure falls back, never crashes the UI
            suggestions = self._fallback(prefix, max_suggestions)

        self._cache[cache_key] = suggestions
        if len(self._cache) > CACHE_MAX_SIZE:
            self._cache.popitem(last=False)
        return suggestions
