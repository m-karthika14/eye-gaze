"""FastAPI + WebSocket server: streams live gaze + S0 decoding, Phase 3's
fixation-aggregated key-probability distribution, and Phase 6's adaptive
keyboard + word suggestions, to the frontend. Wire protocol extends
MASTER_PROJECT_DOCUMENT.md Section 16 with "text", "hovered_key",
"dwell_progress", "fixation", "key_layout", and "suggestions" alongside the
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

from backend.adaptive_layout import accept_suggestion, compute_adaptive_layout
from backend.calibration import run_calibration_session
from decoder.char_lm import CharLM
from decoder.key_probability import DEFAULT_MIN_SIGMA, key_probabilities
from decoder.nearest_key import DEFAULT_LAYOUT_PATH, KeyboardLayout
from experiments.s0 import S0Event, S0System
from llm_services.word_suggestions import WordSuggester
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
_char_lm = CharLM.load_default()
_word_suggester = WordSuggester()


def _load_uncertainty_config() -> tuple[float, float, float]:
    if not DECODER_PARAMS_PATH.exists():
        return DEFAULT_MIN_SIGMA, 1.0, 0.30
    data = yaml.safe_load(DECODER_PARAMS_PATH.read_text(encoding="utf-8")) or {}
    uncertainty_cfg = data.get("gaze_uncertainty", {})
    fallback_cfg = data.get("fallback", {})
    return (
        float(uncertainty_cfg.get("min_sigma", DEFAULT_MIN_SIGMA)),
        float(uncertainty_cfg.get("sigma_scale", 1.0)),
        float(fallback_cfg.get("min_confidence_threshold", 0.30)),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/config/keyboard-layout")
def keyboard_layout() -> dict:
    return json.loads(DEFAULT_LAYOUT_PATH.read_text(encoding="utf-8"))


def _layout_payload(layout: KeyboardLayout) -> dict:
    return {k.id: {"x": k.x, "y": k.y, "width": k.width, "height": k.height} for k in layout.keys}


@app.websocket("/ws/gaze")
async def ws_gaze(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        pipeline = GazePipeline(camera_index=0, target_fps=30)
    except RuntimeError as exc:
        await websocket.send_json({"error": str(exc)})
        await websocket.close()
        return

    # Reloaded fresh per connection (not once at module import) so a value
    # just written by an in-app calibration run (backend/calibration.py)
    # takes effect on the next connection without restarting the backend.
    min_sigma, sigma_scale, min_confidence_threshold = _load_uncertainty_config()

    s0 = S0System(layout=_layout)
    loop = asyncio.get_event_loop()
    stream = pipeline.stream()

    # Phase 6 state. adaptive_keyboard defaults on; the frontend can send
    # {"adaptive_keyboard": false} at any point to fall back to the pure S0
    # static-keyboard behaviour (e.g. for a live S1-vs-S4-style comparison).
    settings = {"adaptive_keyboard": True}
    last_layout_text: str | None = None
    key_layout_payload: dict | None = None
    suggestions: list[str] = []
    suggestion_task: asyncio.Task | None = None
    pending_prefix = ""

    async def receive_control_messages() -> None:
        try:
            while True:
                msg = await websocket.receive_json()
                if "adaptive_keyboard" in msg:
                    settings["adaptive_keyboard"] = bool(msg["adaptive_keyboard"])
        except Exception:  # noqa: BLE001 -- connection closing; main loop exits too
            pass

    receiver_task = asyncio.create_task(receive_control_messages())

    def next_sample():
        return next(stream, None)

    def fetch_suggestions(prefix: str, context: str) -> list[str]:
        return _word_suggester.suggest(prefix, context, max_suggestions=3)

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

            adaptive_on = settings["adaptive_keyboard"]
            if not adaptive_on and s0.layout is not _layout:
                s0.layout = _layout
                last_layout_text = None
                key_layout_payload = None

            fixation = sample.get("fixation")
            key_probs = None
            fixation_payload = None
            if fixation is not None:
                key_probs = key_probabilities(
                    fixation.mu_x,
                    fixation.mu_y,
                    fixation.sigma_x * sigma_scale,
                    fixation.sigma_y * sigma_scale,
                    _layout,
                    min_sigma=min_sigma,
                )
                fixation_payload = {
                    "mu_x": fixation.mu_x,
                    "mu_y": fixation.mu_y,
                    "sigma_x": fixation.sigma_x * sigma_scale,
                    "sigma_y": fixation.sigma_y * sigma_scale,
                    "n_samples": fixation.n_samples,
                    "duration_ms": fixation.duration_ms,
                }

            event = s0.process_sample(sample["gaze_x"], sample["gaze_y"], time.monotonic())

            low_confidence_veto = False
            if (
                adaptive_on
                and event.selected_key not in (None, "BACKSPACE")
                and key_probs is not None
                and max(key_probs.values()) < min_confidence_threshold
            ):
                # Phase 8's low-confidence fallback, pulled forward: don't
                # force a character out of a distribution the system itself
                # isn't confident about -- undo what S0 just committed and
                # wait for a clearer fixation instead of guessing.
                s0.buffer.apply_key("BACKSPACE")
                event = S0Event(
                    text=s0.buffer.text,
                    selected_key=None,
                    hovered_key=event.hovered_key,
                    dwell_progress=event.dwell_progress,
                )
                low_confidence_veto = True

            if adaptive_on:
                # SPACE accepts the top suggestion mid-word instead of a
                # literal space -- the standard predictive-keyboard pattern,
                # and the only gaze-driven way to accept a suggestion (see
                # experiments/s1.py's sibling docstrings for why: adding a
                # new gaze-interactive region would change calibrated
                # geometry; reusing the existing SPACE key doesn't).
                if event.selected_key == "SPACE" and suggestions and pending_prefix and suggestions[0] != pending_prefix:
                    accept_suggestion(s0.buffer, pending_prefix, suggestions[0])
                    event = S0Event(
                        text=s0.buffer.text,
                        selected_key="SPACE",
                        hovered_key=event.hovered_key,
                        dwell_progress=event.dwell_progress,
                    )
                    suggestions = []
                    suggestion_task = None

                if event.text != last_layout_text:
                    s0.layout = compute_adaptive_layout(_layout, _char_lm, event.text)
                    last_layout_text = event.text
                    key_layout_payload = _layout_payload(s0.layout)

                current_word = event.text.rsplit(" ", 1)[-1] if event.text else ""
                if event.selected_key == "BACKSPACE":
                    suggestions = []  # stale after an edit
                if event.selected_key not in ("SPACE", "BACKSPACE", None) and current_word:
                    if suggestion_task is None or suggestion_task.done():
                        # run_in_executor already returns an awaitable Future
                        # (not a coroutine) -- no create_task() wrapper needed.
                        suggestion_task = loop.run_in_executor(None, fetch_suggestions, current_word, event.text)
                if suggestion_task is not None and suggestion_task.done():
                    try:
                        suggestions = suggestion_task.result()
                    except Exception:  # noqa: BLE001 -- a failed suggestion fetch just means no suggestions
                        suggestions = []
                pending_prefix = current_word
            else:
                key_layout_payload = None
                suggestions = []

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
                    "key_layout": key_layout_payload,
                    "suggestions": suggestions,
                    "low_confidence_veto": low_confidence_veto,
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        receiver_task.cancel()
        pipeline.close()


@app.websocket("/ws/calibrate")
async def ws_calibrate(websocket: WebSocket) -> None:
    """In-app calibration: the same live-fixation flow as
    experiments/calibrate_uncertainty.py, surfaced in the real product UI
    (see backend/calibration.py). Only one of this and /ws/gaze can hold the
    webcam at a time -- make sure the typing socket isn't connected first."""
    await run_calibration_session(websocket)
