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

Phases 0-6 complete:

- **Phase 0** — specification frozen, repo skeleton, Groq connectivity verified end to end.
- **Phase 1** — webcam capture, face detection + pretrained full-face gaze CNN (`uniface`: RetinaFace + MobileGaze), heuristic calibration-free pitch/yaw → screen-point mapping, live debug visualizer (`vision/run_demo.py`).
- **Phase 2** — S0 baseline: nearest-key hard selection + dwell detection + text buffer (`/decoder`, `/experiments/s0.py`), FastAPI WebSocket server (`/backend`), React keyboard UI (`/frontend`). Verified end to end against the live camera.
- **Phase 3** — uncertainty modelling: `vision/fixation.py` groups raw samples into a fixation window and estimates `(mu, sigma)` from real scatter; `decoder/key_probability.py` integrates that Gaussian over the keyboard into a full per-key probability distribution (visualized live in the frontend as a purple heatmap, additive alongside S0 -- not yet consumed for selection). `experiments/calibrate_uncertainty.py` has been run against real data: the mapping range was re-derived from real gaze angles, and `sigma_scale=24.45` is a real (large) measurement of webcam gaze imprecision -- not a bug, and exactly what Phase 5's joint decoder is designed to work with.
- **Phase 4** — the two language components: `decoder/char_lm.py`, a local deterministic order-4 character n-gram LM trained on NLTK's Gutenberg corpus (held-out perplexity 5.17, weights committed at `models/char_lm.json`) for Phase 5's decoder; `llm_services/word_suggestions.py`, a Groq-backed whole-word completion suggester (cached, with a local-vocabulary fallback) for the product UI. Both verified independently -- the char LM via synthetic checks, the Groq suggester via live API calls (e.g. `"hel"` → `["help", "hello"]`).
- **Phase 5** — the actual research contribution: S1 through S4 (`/experiments/s1.py`-`s4.py`), all sharing `decoder/beam_search.py`'s noisy-channel search and runnable against identical gaze sequences (`experiments/gaze_sequence.py`). S1 collapses gaze to a hard key first and corrects with a generic confusion model; S4 keeps each observation's real Gaussian key-probability distribution instead. A mechanical synthetic test (`experiments/test_s1_s4_synthetic.py`) already shows H1's predicted pattern working: S4's character-accuracy advantage over S1 grows from +0.018 at low noise to +0.29 at higher noise. (The statistically-powered version of this comparison is Phase 9's job.)
- **Phase 6** — product-layer adaptive keyboard: `backend/adaptive_layout.py` reflows each row's key widths by the local char LM's next-character likelihood (SliceType-style, dampened with a temperature exponent so one dominant prediction doesn't crowd out the rest), and `llm_services/word_suggestions.py` (Phase 4) now drives a live suggestion strip. Word suggestions are accepted by dwelling on the existing, already-calibrated SPACE key rather than a new gaze-interactive region, which would have needed re-running Phase 3's calibration. Fully toggleable live (`{"adaptive_keyboard": false}` over the open socket) without affecting S1-S4, which never touch this code. Verified live: a real dwell selection triggered a real Groq suggestion fetch that appeared correctly a few frames later, and the on/off/on toggle round-tripped cleanly. Also includes a scoped, toggle-gated slice of Phase 8 pulled forward: `backend/main.py` vetoes a dwell selection (undoes it and reports `low_confidence_veto: true` instead) when the peak of the live key-probability distribution is below `config/decoder_params.yaml`'s `fallback.min_confidence_threshold` -- i.e. don't force a character out of a distribution the system itself isn't confident about. `experiments/calibrate_uncertainty.py` also gained `print_accuracy_dashboard()`: a per-trial key-hit/miss table plus aggregate accuracy and a left/right, top/bottom error split, for spotting axis-specific bias (like the yaw-inversion bug) at a glance.

Next: **Phase 7 — Online Personalization**. See Section 19 of `MASTER_PROJECT_DOCUMENT.md` for the full roadmap.
