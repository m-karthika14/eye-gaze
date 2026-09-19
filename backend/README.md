# /backend

Python 3.11+, FastAPI, WebSockets. Streams real-time gaze data and S0 decoding from `/vision`, `/decoder` and `/experiments` to `/frontend`.

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
| `WS /ws/gaze` | Opens the webcam, runs `vision.GazePipeline` + `experiments.S0System`, streams one JSON message per frame |

`/ws/gaze` message shape (extends `MASTER_PROJECT_DOCUMENT.md` Section 16 with `text`, `hovered_key`, `dwell_progress`, and `fixation`):

```json
{
  "gaze_x": 0.53, "gaze_y": 0.61, "confidence": 0.82,
  "selected_key": null, "hovered_key": "D", "dwell_progress": 0.4, "text": "hel",
  "fixation": { "mu_x": 0.53, "mu_y": 0.61, "sigma_x": 0.02, "sigma_y": 0.015, "n_samples": 12, "duration_ms": 400 },
  "key_probabilities": { "A": 0.03, "S": 0.12, "D": 0.48, "F": 0.27, "G": 0.10 }
}
```

`fixation`/`key_probabilities` are `null` until a fixation window has accumulated enough samples (Phase 3). `key_probabilities` is informational for now -- S0's selection still runs on the hard nearest-key path; S1-S4 (Phase 5) will actually decode against this distribution.

Only one client should connect at a time (one webcam). If the camera is already in use, the socket sends `{"error": "..."}` and closes.

**Status:** Phase 1, 2, and 3 done. Verified: `/health`, `/config/keyboard-layout`, and `/ws/gaze` (including `key_probabilities`) all tested live against the real camera. Phase 4+ will add the local character LM and, in Phase 5, the joint decoder that actually consumes `key_probabilities`.