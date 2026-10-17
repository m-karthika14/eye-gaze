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
from decoder.live_modes import LIVE_DECODERS
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


def _calibrated_canvas() -> list[float] | None:
    """Keyboard canvas size (px) the current calibration was recorded at, or None if unknown."""
    from vision.pipeline import CONFIG_PATH

    if not CONFIG_PATH.exists():
        return None
    value = (yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}).get("calibrated_canvas")
    return [float(value[0]), float(value[1])] if isinstance(value, list) and len(value) == 2 else None


def _calibration_stale(calibrated: list[float] | None, current: list[float] | None) -> bool | None:
    """True when the keyboard canvas differs (>4%) from the one calibrated on, or was never stamped
    (a calibration from before the full-screen keyboard). None until the UI reports its size."""
    if current is None:
        return None
    if calibrated is None or calibrated[0] <= 0 or calibrated[1] <= 0:
        return True
    return abs(current[0] - calibrated[0]) / calibrated[0] > 0.04 or abs(current[1] - calibrated[1]) / calibrated[1] > 0.04


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


@app.get("/config/evaluation-phrases")
def evaluation_phrases() -> dict:
    """The fixed phrase set for the typing study (config/evaluation_phrases.txt)."""
    path = Path(__file__).resolve().parent.parent / "config" / "evaluation_phrases.txt"
    phrases = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return {"phrases": phrases}


def _append_study_log(record: dict) -> None:
    """One JSON line per finished phrase, per participant: data/sessions/<participant>.jsonl. Only called when the UI is in study mode."""
    import re

    pid = re.sub(r"[^A-Za-z0-9_-]", "", str(record.get("participant", "")))[:24] or "anonymous"
    out_dir = Path(__file__).resolve().parent.parent / "data" / "sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {**record, "participant": pid, "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(out_dir / f"{pid}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


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
    # All five systems as live decoders (S0-S3: decoder/live_modes.py, S4: decoder/s4_live.py), switchable from the dashboard.
    # They share the same camera, mapping and selection method (blink / Space-Enter / dwell / mouth), so only the decoding differs.
    decoders = {
        m: cls(layout=_layout, char_lm=_char_lm, lexicon=s4.lexicon, sigma_floor=sigma_floor, sigma_ceiling=sigma_ceiling)
        for m, cls in LIVE_DECODERS.items()
    }
    decoders["s4"] = s4
    fps_ema = 0.0
    calibrated_canvas = _calibrated_canvas()
    current_canvas: list[float] | None = None
    loop = asyncio.get_event_loop()
    stream = pipeline.stream()

    # Phase 6 state. adaptive_keyboard defaults OFF per user request (no merging keys)
    # decoder_mode defaults to "s4" (joint probabilistic LM beam search)
    # select_mode: "blink" = look at a key, then hold your eyes shut ~0.5s (natural blinks are ignored);
    # "switch" = look at a key, then press Space/Enter (or click) to select it;
    # "dwell" = hold gaze ~0.8s. Switch is the default because dwell selects whatever you
    # happen to rest your eyes on (reading, hunting for the next key) -- the Midas-touch problem.
    settings = {"adaptive_keyboard": False, "decoder_mode": "s4", "select_mode": "blink"}
    study: dict = {"started": None, "selections": 0, "backspaces": 0, "picks": 0}  # per-phrase counters, only used in study mode
    last_layout_text: str | None = None
    key_layout_payload: dict | None = None
    suggestions: list[str] = []
    suggestion_task: asyncio.Task | None = None
    pending_prefix = ""

    async def receive_control_messages() -> None:
        nonlocal current_canvas
        try:
            while True:
                msg = await websocket.receive_json()
                if isinstance(msg.get("canvas"), list) and len(msg["canvas"]) == 2:
                    current_canvas = [float(msg["canvas"][0]), float(msg["canvas"][1])]
                if "adaptive_keyboard" in msg:
                    settings["adaptive_keyboard"] = bool(msg["adaptive_keyboard"])
                if "decoder_mode" in msg:
                    dm = str(msg["decoder_mode"]).lower()
                    if dm in decoders:
                        previous = settings.get("decoder_mode", "s4")
                        if dm != previous:  # carry the text across so switching modes mid-sentence doesn't lose it
                            decoders[dm].set_text(decoders[previous].current_text)
                        settings["decoder_mode"] = dm
                if "select_mode" in msg and str(msg["select_mode"]) in ("dwell", "switch", "blink", "mouth"):
                    settings["select_mode"] = str(msg["select_mode"])
                if msg.get("select"):
                    settings["pending_select"] = True
                if msg.get("clear_text"):
                    s0.reset()
                    for d in decoders.values():
                        d.reset()
                if msg.get("study_start"):
                    # Study mode: a new phrase begins. Clear the text and start counting on the server (each selection happens exactly once here).
                    study.update(started=time.monotonic(), selections=0, backspaces=0, picks=0)
                    s0.reset()
                    for d in decoders.values():
                        d.reset()
                if isinstance(msg.get("study_done"), dict) and study.get("started") is not None:
                    cur = decoders[settings.get("decoder_mode", "s4")]
                    _append_study_log(
                        {
                            **msg["study_done"],
                            "typed": cur.current_text.strip(),
                            "elapsed_s": round(time.monotonic() - study["started"], 2),
                            "selections": study["selections"],
                            "backspaces": study["backspaces"],
                            "candidate_picks": study["picks"],
                            "decoder_mode": settings.get("decoder_mode"),
                            "select_mode": settings.get("select_mode"),
                            # which mapping this phrase was typed with: a generic one (built from other people) or the person's own
                            "mapping": "generic" if "generic" in str(pipeline.mapper.config.feature_model or "") else "personal",
                        }
                    )
                    study["started"] = None
                    s0.reset()
                    for d in decoders.values():
                        d.reset()
                if "pick_candidate" in msg:
                    if decoders[settings.get("decoder_mode", "s4")].pick_candidate(int(msg["pick_candidate"])) and study.get("started") is not None:
                        study["picks"] += 1
                if msg.get("recenter"):
                    pipeline.mapper.recenter()
                if msg.get("reset_shift"):
                    pipeline.mapper.reset_shift()
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

            decoder_mode = settings.get("decoder_mode", "s4")
            dec = decoders.get(decoder_mode, s4)
            layout_adaptive = bool(getattr(dec, "adaptive_layout", False))  # S2 / S3: confidence-aware adaptive keyboard
            adaptive_on = settings["adaptive_keyboard"]
            if not adaptive_on and not layout_adaptive and s0.layout is not _layout:
                s0.layout = _layout
                for d in decoders.values():
                    d.layout = _layout
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
            for d in decoders.values():
                d.dwell_enabled = settings.get("select_mode", "blink") == "dwell"
            select_mode_now = settings.get("select_mode", "blink")
            pipeline.hold_on_blink = select_mode_now == "blink"  # freeze gaze only for the ACTIVE gesture mode
            pipeline.hold_on_mouth = select_mode_now == "mouth"
            eyes_closed = bool(sample.get("gaze_held"))
            pending_select = settings.pop("pending_select", False)
            # Gaze that has left the keyboard (e.g. reading the text box above it)
            # must not dwell on the clamped edge key -- treat it as no gaze.
            if eyes_closed:
                dwell_x, dwell_y = None, None  # unused: the decoder is held below
            elif sample.get("on_keyboard", True):
                dwell_x, dwell_y = sample["gaze_x"], sample["gaze_y"]
            else:
                dwell_x, dwell_y = None, None
            if eyes_closed:
                event = dec.hold()  # keep the highlight frozen while the eyes are shut
            else:
                event = dec.process_sample(
                    dwell_x,
                    dwell_y,
                    now_ts,
                    min_sigma=min_sigma,
                    sigma_scale=sigma_scale,
                )

            if pending_select and not settings.get("typing_paused"):
                event = dec.select_now(now_ts, sigma_scale=sigma_scale)
            elif (
                (
                    (sample.get("blink_fired") and settings.get("select_mode") == "blink")
                    or (sample.get("mouth_fired") and settings.get("select_mode") == "mouth")
                )
                and not settings.get("typing_paused")
            ):
                # Deliberate blink: select the key highlighted just BEFORE the lids dropped.
                closed_since = (
                    sample.get("blink_closed_since") if settings.get("select_mode") == "blink" else sample.get("mouth_since")
                ) or now_ts
                event = dec.select_now(now_ts, sigma_scale=sigma_scale, at_time=closed_since - 0.25)

            # When typing is paused (testing / eyeball tracking mode), do not commit key selections
            if settings.get("typing_paused"):
                event = S0Event(
                    text=dec.current_text,
                    selected_key=None,
                    hovered_key=event.hovered_key,
                    dwell_progress=event.dwell_progress,
                )

            if event.selected_key and study.get("started") is not None:
                study["selections"] += 1
                if event.selected_key == "BACKSPACE":
                    study["backspaces"] += 1

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
                if dec.active_word_obs:
                    dec.active_word_obs.pop()
                    dec.active_word = dec._decode_active(min_sigma=min_sigma)
                event = S0Event(
                    text=dec.current_text,
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
                    for d in decoders.values():
                        d.layout = s0.layout
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
                if layout_adaptive:
                    # S2 / S3: resize keys by next-letter likelihood after every committed change (no Groq suggestions involved)
                    if event.text != last_layout_text:
                        s0.layout = compute_adaptive_layout(_layout, _char_lm, event.text)
                        for d in decoders.values():
                            d.layout = s0.layout
                        last_layout_text = event.text
                        key_layout_payload = _layout_payload(s0.layout)
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
                    "select_mode": settings.get("select_mode", "blink"),
                    "calibration_stale": _calibration_stale(calibrated_canvas, current_canvas),
                    "blink_progress": round(float(sample.get("blink_progress") or 0.0), 2),
                    "mouth_progress": round(float(sample.get("mouth_progress") or 0.0), 2),
                    "ear": round(float(sample["ear"]), 2) if sample.get("ear") is not None else None,
                    "mar": round(float(sample["mar"]), 2) if sample.get("mar") is not None else None,
                    "eyes_closed": eyes_closed,
                    "gaze_shift": [round(pipeline.mapper.shift[0], 3), round(pipeline.mapper.shift[1], 3)],
                    "active_word": dec.active_word,
                    "candidates": dec.candidates,
                    "online_bias": [round(dec.bias[0], 3), round(dec.bias[1], 3)] if hasattr(dec, "bias") else None,
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
