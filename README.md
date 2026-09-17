# Calibration-Free Webcam-Based AI Assistive Communication System

Uncertainty-aware, calibration-free webcam gaze decoding for assistive communication, using a joint gaze-probability + language-model decoder. Full research design, literature survey, architecture, and phased roadmap live in [`MASTER_PROJECT_DOCUMENT.md`](./MASTER_PROJECT_DOCUMENT.md) — that document is the single source of truth for this project. This README only covers repository setup.

**Core research question:** when webcam gaze is uncertain, does keeping the full gaze probability distribution and decoding it jointly with language context (S4) produce more accurate intended text than selecting one gaze key first and applying language correction afterward (S1)?

## Repository layout

```
/frontend         React + TypeScript — keyboard UI, gaze cursor, suggestions, text display
/backend          FastAPI + WebSocket server
/vision           Face detection, gaze estimation, uncertainty, temporal processing
/decoder          Key-probability mapping, character LM, joint decoder (noisy-channel)
/llm_services     Groq client — word/phrase suggestions, TTS, (optional) STT
/personalization  Online user model, bias adaptation
/experiments      S0, S1, S2, S3, S4 — runnable, reproducible experiment scripts
/data             Dataset processing, evaluation data, simulation outputs
/models           Gaze model weights, local character-LM weights
/config           Keyboard layout, model paths, decoder parameters, .env (API keys)
```

Each folder has its own `README.md` describing its role and current phase status.

## Setup

1. **Python environment** (backend / vision / decoder / llm_services / personalization / experiments):
   ```
   python -m venv .venv
   .venv/Scripts/activate      # Windows
   pip install -r requirements.txt
   ```

2. **Groq API key** — copy `config/.env.example` to `config/.env` and fill in `GROQ_API_KEY`. Never commit `config/.env` (already git-ignored). Verify connectivity with:
   ```
   python llm_services/test_connection.py
   ```

3. **Frontend** — scaffolded starting Phase 2 (keyboard UI). Not yet initialized.

## Project status

Currently executing **Phase 0 — Specification Freeze** (see Section 19 of `MASTER_PROJECT_DOCUMENT.md`). The research design is frozen; no new features are being brainstormed. Implementation proceeds phase by phase, S0 → S4, per the roadmap.
