# /llm_services

Groq client wrapper — word/phrase completion suggestions, text-to-speech, and (optionally) speech-to-text for the **product** layer only. Never used inside the measured decoder path (`/decoder`); see `MASTER_PROJECT_DOCUMENT.md` Section 13.

- `groq_client.py` — thin wrapper around the Groq chat-completions API, model IDs read from `config/.env`, supports a per-call `timeout`.
- `test_connection.py` — Phase 0 deliverable: verifies a basic Groq chat-completion call works end to end.
- `word_suggestions.py` — Phase 4 deliverable: `WordSuggester.suggest(prefix, context, max_suggestions)` returns ranked whole-word completions (e.g. `"hel"` → `["help", "hello"]`, matching Section 9's example). Caches repeated `(prefix, context)` queries (in-memory, bounded), and falls back to `config/vocabulary.txt` if Groq errors, times out (3s default), or is unreachable -- never crashes the UI, never blocks indefinitely.

**Status:** Phase 0 (connectivity) and Phase 4 (word suggestions) done. Verified live against the real API: basic completion, cache-hit speed, and fallback-on-failure all tested (`experiments/test_word_suggestions.py`). Fallback/timeout hardening gets a fuller treatment in **Phase 8**.

## Usage

```
python llm_services/test_connection.py
```

Requires `GROQ_API_KEY` set in `config/.env` (copy from `config/.env.example`).
