# /llm_services

Groq client wrapper — word/phrase completion suggestions, text-to-speech, and (optionally) speech-to-text for the **product** layer only. Never used inside the measured decoder path (`/decoder`); see `MASTER_PROJECT_DOCUMENT.md` Section 13.

- `groq_client.py` — thin wrapper around the Groq chat-completions API, model IDs read from `config/.env`.
- `test_connection.py` — Phase 0 deliverable: verifies a basic Groq chat-completion call works end to end.

**Status:** connectivity scaffolding in place (Phase 0). Word/phrase-suggestion prompting, response caching, and the local fallback word list are built in **Phase 4**; the fallback/timeout handling is hardened in **Phase 8**.

## Usage

```
python llm_services/test_connection.py
```

Requires `GROQ_API_KEY` set in `config/.env` (copy from `config/.env.example`).
