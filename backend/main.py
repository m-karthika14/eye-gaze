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
import math
from pathlib import Path
import time

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import yaml

from backend.adaptive_layout import accept_suggestion, compute_adaptive_layout
from backend.calibration import run_calibration_session
from decoder.char_lm import CharLM
from decoder.key_probability import DEFAULT_MIN_SIGMA, key_probabilities
from decoder.nearest_key import DEFAULT_LAYOUT_PATH, KeyboardLayout
from experiments.s0 import S0Event, S0System
from decoder.s4_live import S4LiveDecoder
from llm_services.word_suggestions import WordSuggester
from vision.face_gaze import annotate_eye_tracker_frame
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


def _load_live_sigma_bounds() -> tuple[float, float]:
    """Floor/ceiling for the S4 live decoder's per-dwell gaze sigma -- see
    decoder/s4_live.py and config/decoder_params.yaml."""
    if not DECODER_PARAMS_PATH.exists():
        return 0.08, 0.12
    data = yaml.safe_load(DECODER_PARAMS_PATH.read_text(encoding="utf-8")) or {}
    cfg = data.get("gaze_uncertainty", {})
    return float(cfg.get("live_sigma_floor", 0.08)), float(cfg.get("live_sigma_ceiling", 0.12))


latest_preview_jpeg: bytes | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    """Streams live webcam frames with face mesh, blue eyeball tracking dots,
    and gaze vectors as a multipart MJPEG stream for the frontend preview."""
    def frame_stream():
        while True:
            if latest_preview_jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + latest_preview_jpeg + b"\r\n"
                )
            time.sleep(0.033)

    return StreamingResponse(frame_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


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
    sigma_floor, sigma_ceiling = _load_live_sigma_bounds()
    s4 = S4LiveDecoder(layout=_layout, char_lm=_char_lm, sigma_floor=sigma_floor, sigma_ceiling=sigma_ceiling)
    fps_ema = 0.0
    loop = asyncio.get_event_loop()
    stream = pipeline.stream()

    # Phase 6 state. adaptive_keyboard defaults OFF per user request (no merging keys)
    # decoder_mode defaults to "s4" (joint probabilistic LM beam search)
    settings = {"adaptive_keyboard": False, "decoder_mode": "s4"}
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
                if "decoder_mode" in msg:
                    dm = str(msg["decoder_mode"]).lower()
                    if dm in ("s0", "s4"):
                        if dm != settings.get("decoder_mode"):
                            if dm == "s4":
                                s4.set_text(s0.buffer.text)
                            else:
                                s0.buffer.reset()
                                s0.buffer.text = s4.current_text
                        settings["decoder_mode"] = dm
                if msg.get("clear_text"):
                    s0.reset()
                    s4.reset()
                if "pick_candidate" in msg and settings.get("decoder_mode") == "s4":
                    s4.pick_candidate(int(msg["pick_candidate"]))
                if msg.get("recenter"):
                    pipeline.mapper.recenter()
                if "invert_x" in msg:
                    pipeline.mapper.config.invert_x = bool(msg["invert_x"])
                if "invert_y" in msg:
                    pipeline.mapper.config.invert_y = bool(msg["invert_y"])
                if "pause_typing" in msg:
                    settings["typing_paused"] = bool(msg["pause_typing"])
                if "pitch_sensitivity" in msg:
                    pipeline.mapper._range_pitch = max(0.08, min(0.6, float(msg["pitch_sensitivity"])))
                if "yaw_sensitivity" in msg:
                    pipeline.mapper._range_yaw = max(0.08, min(0.6, float(msg["yaw_sensitivity"])))
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

            # Update live eye tracker preview frame with facial mesh and blue eyeball dots
            global latest_preview_jpeg
            raw_frame = sample.get("frame")
            if raw_frame is not None:
                face_res = sample.get("face_result")
                p_rad = sample.get("pitch")
                y_rad = sample.get("yaw")
                p_d = math.degrees(p_rad) if p_rad is not None else None
                y_d = math.degrees(y_rad) if y_rad is not None else None
                g_pt = (sample["gaze_x"], sample["gaze_y"]) if sample["gaze_x"] is not None else None
                vis = annotate_eye_tracker_frame(
                    raw_frame,
                    face_res,
                    pitch_deg=p_d,
                    yaw_deg=y_d,
                    gaze_screen_pt=g_pt,
                )
                vis_small = cv2.resize(vis, (320, 240))
                ret, buf = cv2.imencode(".jpg", vis_small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ret:
                    latest_preview_jpeg = buf.tobytes()

            sample_fps = sample.get("fps")
            if sample_fps:
                fps_ema = sample_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * sample_fps

            adaptive_on = settings["adaptive_keyboard"]
            if not adaptive_on and s0.layout is not _layout:
                s0.layout = _layout
                s4.layout = _layout
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

            decoder_mode = settings.get("decoder_mode", "s4")
            now_ts = time.monotonic()
            # Gaze that has left the keyboard (e.g. reading the text box above it)
            # must not dwell on the clamped edge key -- treat it as no gaze.
            if sample.get("on_keyboard", True):
                dwell_x, dwell_y = sample["gaze_x"], sample["gaze_y"]
            else:
                dwell_x, dwell_y = None, None
            if decoder_mode == "s4":
                event = s4.process_sample(
                    dwell_x,
                    dwell_y,
                    now_ts,
                    min_sigma=min_sigma,
                    sigma_scale=sigma_scale,
                )
            else:
                event = s0.process_sample(dwell_x, dwell_y, now_ts)

            # When typing is paused (testing / eyeball tracking mode), do not commit key selections
            if settings.get("typing_paused"):
                event = S0Event(
                    text=s4.current_text if decoder_mode == "s4" else s0.buffer.text,
                    selected_key=None,
                    hovered_key=event.hovered_key,
                    dwell_progress=event.dwell_progress,
                )

            low_confidence_veto = False
            if (
                adaptive_on
                and event.selected_key not in (None, "BACKSPACE")
                and key_probs is not None
                and max(key_probs.values()) < min_confidence_threshold
            ):
                # Phase 8's low-confidence fallback, pulled forward: don't
                # force a character out of a distribution the system itself
                # isn't confident about -- undo what was just committed.
                if decoder_mode == "s4":
                    if s4.active_word_obs:
                        s4.active_word_obs.pop()
                        s4.active_word = s4._decode_active(min_sigma=min_sigma)
                else:
                    s0.buffer.apply_key("BACKSPACE")
                event = S0Event(
                    text=s4.current_text if decoder_mode == "s4" else s0.buffer.text,
                    selected_key=None,
                    hovered_key=event.hovered_key,
                    dwell_progress=event.dwell_progress,
                )
                low_confidence_veto = True

            if adaptive_on:
                if event.selected_key == "SPACE" and suggestions and pending_prefix and suggestions[0] != pending_prefix:
                    if decoder_mode == "s0":
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
                    s4.layout = s0.layout
                    last_layout_text = event.text
                    key_layout_payload = _layout_payload(s0.layout)

                current_word = event.text.rsplit(" ", 1)[-1] if event.text else ""
                if event.selected_key == "BACKSPACE":
                    suggestions = []  # stale after an edit
                if event.selected_key not in ("SPACE", "BACKSPACE", None) and current_word:
                    if suggestion_task is None or suggestion_task.done():
                        suggestion_task = loop.run_in_executor(None, fetch_suggestions, current_word, event.text)
                if suggestion_task is not None and suggestion_task.done():
                    try:
                        suggestions = suggestion_task.result()
                    except Exception:  # noqa: BLE001
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
                    "decoder_mode": decoder_mode,
                    "active_word": s4.active_word if decoder_mode == "s4" else "",
                    "candidates": s4.candidates if decoder_mode == "s4" else [],
                    "fps": round(fps_ema, 1),
                    "infer_ms": round(sample.get("infer_ms") or 0.0),
                    "fixation": fixation_payload,
                    "key_probabilities": key_probs,
                    "key_layout": key_layout_payload,
                    "suggestions": suggestions,
                    "low_confidence_veto": low_confidence_veto,
                    "invert_x": pipeline.mapper.config.invert_x,
                    "invert_y": pipeline.mapper.config.invert_y,
                    "face_found": sample.get("face_found", False),
                    "pitch_deg": round(math.degrees(sample.get("pitch", 0.0)), 1) if sample.get("pitch") is not None else None,
                    "yaw_deg": round(math.degrees(sample.get("yaw", 0.0)), 1) if sample.get("yaw") is not None else None,
                    "typing_paused": settings.get("typing_paused", False),
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
