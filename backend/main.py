"""FastAPI + WebSocket server: streams live gaze + S0 decoding, plus Phase 3's
fixation-aggregated key-probability distribution, to the frontend. Wire
protocol extends MASTER_PROJECT_DOCUMENT.md Section 16 with "text",
"hovered_key", "dwell_progress", and "fixation" fields alongside the
"key_probabilities" the section already specifies.

Run from the repo root (no --reload: this server holds the webcam open for
the duration of a connection, and hot-reloading mid-connection tears it down
abruptly -- restart manually after code changes instead):

    .venv/Scripts/python.exe -m uvicorn backend.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from decoder.key_probability import DEFAULT_MIN_SIGMA, key_probabilities
from decoder.nearest_key import DEFAULT_LAYOUT_PATH, KeyboardLayout
from experiments.s0 import S0System
from vision.pipeline import GazePipeline

DECODER_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "decoder_params.yaml"

app = FastAPI(title="Calibration-Free Gaze Typing -- Backend")

# Dev-only: matches any localhost/127.0.0.1 port, since Vite auto-increments
# past 5173 if that port's already taken. Tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

_layout = KeyboardLayout.from_json()


def _load_uncertainty_config() -> tuple[float, float]:
    if not DECODER_PARAMS_PATH.exists():
        return DEFAULT_MIN_SIGMA, 1.0
    data = yaml.safe_load(DECODER_PARAMS_PATH.read_text(encoding="utf-8")) or {}
    cfg = data.get("gaze_uncertainty", {})
    return float(cfg.get("min_sigma", DEFAULT_MIN_SIGMA)), float(cfg.get("sigma_scale", 1.0))


_min_sigma, _sigma_scale = _load_uncertainty_config()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/config/keyboard-layout")
def keyboard_layout() -> dict:
    return json.loads(DEFAULT_LAYOUT_PATH.read_text(encoding="utf-8"))


@app.websocket("/ws/gaze")
async def ws_gaze(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        pipeline = GazePipeline(camera_index=0, target_fps=30)
    except RuntimeError as exc:
        await websocket.send_json({"error": str(exc)})
        await websocket.close()
        return

    s0 = S0System(layout=_layout)
    loop = asyncio.get_event_loop()
    stream = pipeline.stream()

    def next_sample():
        return next(stream, None)

    try:
        while True:
            try:
                sample = await loop.run_in_executor(None, next_sample)
            except Exception as exc:  # noqa: BLE001 -- surface it to the client, don't die silently
                await websocket.send_json({"error": f"Vision pipeline error: {exc}"})
                break

            if sample is None:
                await websocket.send_json(
                    {"error": "Webcam stream ended (camera disconnected, unplugged, or unreadable)."}
                )
                break

            event = s0.process_sample(sample["gaze_x"], sample["gaze_y"], time.monotonic())

            fixation = sample.get("fixation")
            key_probs = None
            fixation_payload = None
            if fixation is not None:
                key_probs = key_probabilities(
                    fixation.mu_x,
                    fixation.mu_y,
                    fixation.sigma_x * _sigma_scale,
                    fixation.sigma_y * _sigma_scale,
                    _layout,
                    min_sigma=_min_sigma,
                )
                fixation_payload = {
                    "mu_x": fixation.mu_x,
                    "mu_y": fixation.mu_y,
                    "sigma_x": fixation.sigma_x * _sigma_scale,
                    "sigma_y": fixation.sigma_y * _sigma_scale,
                    "n_samples": fixation.n_samples,
                    "duration_ms": fixation.duration_ms,
                }

            await websocket.send_json(
                {
                    "gaze_x": sample["gaze_x"],
                    "gaze_y": sample["gaze_y"],
                    "confidence": sample["confidence"],
                    "selected_key": event.selected_key,
                    "hovered_key": event.hovered_key,
                    "dwell_progress": event.dwell_progress,
                    "text": event.text,
                    "fixation": fixation_payload,
                    "key_probabilities": key_probs,
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        pipeline.close()
