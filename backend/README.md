# /backend

Python 3.11+, FastAPI, WebSockets. Streams real-time gaze data and S0 decoding from `/vision`, `/decoder` and `/experiments` to `/frontend`, plus Phase 6's product-layer adaptive keyboard and word suggestions.

## Run

```
.venv/Scripts/python.exe -m uvicorn backend.main:app --port 8000
```

Deliberately no `--reload`: this server holds the webcam open for the
duration of a WebSocket connection, and hot-reloading mid-connection tears
down the camera and socket abruptly. After editing `/vision`, `/decoder`, or
`/backend`, stop it (Ctrl+C) and restart manually.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `GET /config/keyboard-layout` | Serves `config/keyboard_layout.json` so the frontend doesn't need its own copy |
| `WS /ws/gaze` | Opens the webcam, runs `vision.GazePipeline` + `experiments.S0System` (+ Phase 6's adaptive layer), streams one JSON message per frame |
| `WS /ws/calibrate` | In-app calibration (`backend/calibration.py`) -- the same live-fixation data collection as `experiments/calibrate_uncertainty.py`'s terminal script, driven over a WebSocket into the real product keyboard instead of a separate OpenCV window |

`/ws/gaze` message shape (extends `MASTER_PROJECT_DOCUMENT.md` Section 16 with `text`, `hovered_key`, `dwell_progress`, `fixation`, `key_layout`, `suggestions`, and `low_confidence_veto`):

```json
{
  "gaze_x": 0.53, "gaze_y": 0.61, "confidence": 0.82,
  "selected_key": null, "hovered_key": "D", "dwell_progress": 0.4, "text": "hel",
  "fixation": { "mu_x": 0.53, "mu_y": 0.61, "sigma_x": 0.02, "sigma_y": 0.015, "n_samples": 12, "duration_ms": 400 },
  "key_probabilities": { "A": 0.03, "S": 0.12, "D": 0.48, "F": 0.27, "G": 0.10 },
  "key_layout": { "Q": {"x": 0.0, "y": 0.0, "width": 0.06, "height": 0.25}, "...": "..." },
  "suggestions": ["hello", "help"],
  "low_confidence_veto": false
}
```

`fixation`/`key_probabilities` are `null` until a fixation window has accumulated enough samples (Phase 3); `key_probabilities` is informational for S0 -- S1-S4 (`/experiments`) are the systems that actually decode against a distribution like it, in their own offline batch harness. `key_layout`/`suggestions` are Phase 6: present only while adaptive mode is on (default). The client can send `{"adaptive_keyboard": false}` over the open socket at any point to fall back to the pure static-keyboard S0 baseline (`key_layout` becomes `null`, `suggestions` becomes `[]`) -- e.g. for a live version of the S1-vs-S4-style controlled comparison, which needs an identical static keyboard.

`low_confidence_veto`: Phase 8's low-confidence fallback ("don't force a selection when the system itself isn't sure"), pulled forward as a scoped addition on top of Phase 6, active only while adaptive mode is on. If a dwell would otherwise commit a key but the peak of `key_probabilities` is below `config/decoder_params.yaml`'s `fallback.min_confidence_threshold` (0.30), the character is immediately undone (`selected_key` reported as `null` for that frame) and this flag is set to `true` instead of typing a guess. BACKSPACE is never vetoed (there's no redo-of-pop primitive to safely undo an undo).

Only one client should connect at a time (one webcam) -- this applies across both `/ws/gaze` and `/ws/calibrate`, since only one of the two can hold the camera. If the camera is already in use, the socket sends `{"error": "..."}` (or, on `/ws/calibrate`, `{"phase": "error", "error": "..."}`) and closes.

`/ws/calibrate` message shape: `{"phase": "warming_up"}` while the startup bias-lock completes, then per-frame `{"phase": "trial", "target_key": "Q", "trial_index": 3, "total_trials": 72, "trials_recorded": 2, "gaze_x": ..., "gaze_y": ..., "fixation_progress": 0.4, "elapsed_s": 1.2}` while a target is shown, `{"phase": "trial_recorded" | "trial_skipped", "target_key": "Q"}` when a trial ends, and finally `{"phase": "done", "trials_collected": 58, "dashboard": {...}, "sigma": {...}, "range": {...} | null}` once `config/gaze_mapping.yaml` and `config/decoder_params.yaml` have been re-derived and written. `dashboard`/`sigma`/`range` are the exact same structured data `experiments/calibrate_uncertainty.py` prints to the terminal (via `compute_accuracy_dashboard`/`compute_sigma_scale`/`analyze_range`, imported directly -- not re-derived, so the two entry points can't silently drift apart). The client can send `{"stop": true}` at any point to end early (same as pressing `q` in the terminal script), as long as at least 8 trials were already recorded. `/ws/gaze`'s uncertainty config (`min_sigma`/`sigma_scale`/`min_confidence_threshold`) is reloaded fresh on every new connection, so a value an in-app calibration run just wrote takes effect on the next typing connection without restarting the backend.

**Status:** Phases 1, 2, 3, and 6 done, plus pulled-forward slices of Phase 7 (continuous drift correction, `/vision`) and Phase 8 (low-confidence veto), and an in-app calibration flow. Verified live against the real camera: `/health`, `/config/keyboard-layout`, `/ws/gaze` (including a real dwell-selected character triggering a real Groq suggestion fetch that appeared a few frames later), and the `adaptive_keyboard` on/off/on toggle. `backend/adaptive_layout.py` (key resizing + suggestion acceptance) has its own synthetic regression tests in `/experiments`.
