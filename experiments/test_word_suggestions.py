"""Live regression check for Phase 4's Groq word-completion suggester.
Requires a working GROQ_API_KEY in config/.env (see llm_services/README.md).
Makes real API calls -- not a pure-offline test.

Run from the repo root:

    .venv/Scripts/python.exe -m experiments.test_word_suggestions
"""

from __future__ import annotations

import sys
import time

from llm_services.word_suggestions import WordSuggester


def check_basic_completion(suggester: WordSuggester) -> bool:
    start = time.monotonic()
    suggestions = suggester.suggest("hel", context="i need", max_suggestions=3)
    latency = time.monotonic() - start

    print(f"[info] 'hel' -> {suggestions}  ({latency * 1000:.0f}ms)")
    if not suggestions:
        print("[FAIL] expected at least one suggestion for 'hel'")
        return False
    if not all(w.startswith("hel") for w in suggestions):
        print(f"[FAIL] not all suggestions start with 'hel': {suggestions}")
        return False
    if latency > 10.0:
        print(f"[FAIL] latency {latency:.1f}s too slow for real-time suggestion display")
        return False
    print("[OK] basic completion: valid, prefix-matching, reasonably fast")
    return True


def check_cache_hit_is_instant(suggester: WordSuggester) -> bool:
    suggester.suggest("wat", context="i want", max_suggestions=3)  # warm the cache
    start = time.monotonic()
    suggester.suggest("wat", context="i want", max_suggestions=3)  # should hit cache
    latency = time.monotonic() - start

    if latency > 0.01:
        print(f"[FAIL] cached call took {latency * 1000:.1f}ms, expected near-instant")
        return False
    print(f"[OK] cached repeat query is near-instant ({latency * 1000:.3f}ms)")
    return True


def check_fallback_works_without_api() -> bool:
    from llm_services.groq_client import GroqClient

    class BrokenClient(GroqClient):
        def __init__(self) -> None:
            pass  # skip real init -- no API key needed

        def chat_completion(self, *args, **kwargs):
            raise RuntimeError("simulated Groq outage")

    suggester = WordSuggester(client=BrokenClient())
    suggestions = suggester.suggest("th", max_suggestions=3)
    print(f"[info] fallback (Groq broken) 'th' -> {suggestions}")
    if not all(w.startswith("th") for w in suggestions):
        print(f"[FAIL] fallback suggestions don't all start with 'th': {suggestions}")
        return False
    print("[OK] falls back to local vocabulary when Groq is unreachable, no crash")
    return True


def main() -> int:
    suggester = WordSuggester()

    checks = [
        lambda: check_basic_completion(suggester),
        lambda: check_cache_hit_is_instant(suggester),
        check_fallback_works_without_api,
    ]
    results = [check() for check in checks]
    if all(results):
        print(f"\n[OK] all {len(results)} word-suggestion checks passed")
        return 0
    print(f"\n[FAIL] {results.count(False)}/{len(results)} checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
