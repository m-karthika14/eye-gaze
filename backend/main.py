"""FastAPI + WebSocket server: streams live gaze + S0 decoding to the
frontend (Phase 2 deliverable). Wire protocol extends
MASTER_PROJECT_DOCUMENT.md Section 16 with "text" and "hovered_key" fields
carrying the live S0 decoder state.

Run from the repo root:

    .venv/Scripts/python.exe -m uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from decoder.nearest_key import DEFAULT_LAYOUT_PATH, KeyboardLayout
from experiments.s0 import S0System
from vision.pipeline import GazePipeline

app = FastAPI(title="Calibration-Free Gaze Typing -- Backend")

# Dev-only: Vite's default port. Tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_layout = KeyboardLayout.from_json()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/config/keyboard-layout")
def keyboard_layout() -> dict:
    import json

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

            await websocket.send_json(
                {
                    "gaze_x": sample["gaze_x"],
                    "gaze_y": sample["gaze_y"],
                    "confidence": sample["confidence"],
                    "selected_key": event.selected_key,
                    "hovered_key": event.hovered_key,
                    "text": event.text,
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        pipeline.close()
