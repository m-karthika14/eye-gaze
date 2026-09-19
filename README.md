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

3. **Run the S0 baseline** (backend + frontend, two terminals):
   ```
   .venv/Scripts/python.exe -m uvicorn backend.main:app --port 8000
   ```
   ```
   cd frontend && npm install && npm run dev
   ```
   Open `http://localhost:5173`. Look at a key and dwell ~500ms to type it.

   Note: no `--reload` -- this server holds the webcam open, and hot-reloading
   mid-connection (e.g. after editing a `/vision`, `/decoder`, or `/backend`
   file) tears down the camera and WebSocket abruptly. After any code change,
   stop it (Ctrl+C) and restart manually instead.

## Project status

Phases 0-3 complete:

- **Phase 0** — specification frozen, repo skeleton, Groq connectivity verified end to end.
- **Phase 1** — webcam capture, face detection + pretrained full-face gaze CNN (`uniface`: RetinaFace + MobileGaze), heuristic calibration-free pitch/yaw → screen-point mapping, live debug visualizer (`vision/run_demo.py`).
- **Phase 2** — S0 baseline: nearest-key hard selection + dwell detection + text buffer (`/decoder`, `/experiments/s0.py`), FastAPI WebSocket server (`/backend`), React keyboard UI (`/frontend`). Verified end to end against the live camera.
- **Phase 3** — uncertainty modelling: `vision/fixation.py` groups raw samples into a fixation window and estimates `(mu, sigma)` from real scatter; `decoder/key_probability.py` integrates that Gaussian over the keyboard into a full per-key probability distribution (visualized live in the frontend as a purple heatmap, additive alongside S0 -- not yet consumed for selection). **`experiments/calibrate_uncertainty.py` still needs to be run against real data** to empirically validate sigma before Phase 5 can trust it.

Next: **Phase 4 — Language Model Layer** (local character LM + Groq word suggestions). See Section 19 of `MASTER_PROJECT_DOCUMENT.md` for the full roadmap.
