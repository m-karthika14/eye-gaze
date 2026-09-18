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

`/ws/gaze` message shape (extends `MASTER_PROJECT_DOCUMENT.md` Section 16 with `text` and `hovered_key`):

```json
{ "gaze_x": 0.53, "gaze_y": 0.61, "confidence": 0.82, "selected_key": null, "hovered_key": "D", "text": "hel" }
```

Only one client should connect at a time (one webcam). If the camera is already in use, the socket sends `{"error": "..."}` and closes.

**Status:** Phase 1 (raw gaze streaming groundwork) and Phase 2 (S0 baseline: nearest-key + dwell + text buffer, wired end to end) done. Verified: `/health`, `/config/keyboard-layout`, and `/ws/gaze` all tested live against the real camera. Phase 3+ will add key-probability distributions and the joint decoder in place of S0's hard nearest-key selection.