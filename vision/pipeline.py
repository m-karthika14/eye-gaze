"""End-to-end pipeline: webcam -> face/gaze -> normalized (x, y) point ->
fixation-aggregated (mu, sigma) for uncertainty-aware key decoding.

Yields dicts matching the WebSocket schema in MASTER_PROJECT_DOCUMENT.md
Section 16 (gaze_x, gaze_y, confidence, selected_key), plus debug-only extras
(pitch, yaw, bbox, fps, frame, fixation). The backend forwards only the
JSON-safe subset over the WebSocket and drops the debug extras; `fixation`
(a FixationEvent | None, Phase 3) is not JSON-safe as-is -- the caller
converts it and, if present, runs it through decoder.key_probability for the
per-key distribution (kept out of /vision to keep the uncertainty-estimation
and key-probability-mapping concerns separately owned, per Section 15).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .blink import BlinkDetector, MouthOpenDetector
from .face_gaze import FaceGazeEstimator
from .fixation import FixationAggregator
from .gaze_to_screen import GazeMappingConfig, GazeToScreenMapper
from .webcam import Webcam

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "gaze_mapping.yaml"
DECODER_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "decoder_params.yaml"


def _load_fixation_kwargs() -> dict[str, float]:
    if not DECODER_PARAMS_PATH.exists():
        return {}
    import yaml

    data = yaml.safe_load(DECODER_PARAMS_PATH.read_text(encoding="utf-8")) or {}
    fixation_cfg = data.get("fixation", {})
    kwargs: dict[str, float] = {}
    if "min_window_ms" in fixation_cfg:
        kwargs["min_window_ms"] = float(fixation_cfg["min_window_ms"])
    if "max_window_ms" in fixation_cfg:
        kwargs["max_window_ms"] = float(fixation_cfg["max_window_ms"])
    if "fixation_radius" in fixation_cfg:
        kwargs["fixation_radius"] = float(fixation_cfg["fixation_radius"])
    return kwargs


def _load_estimator_kwargs() -> dict:
    """Optional `gaze_model` / `detect_every` / `detect_size` keys in
    config/gaze_mapping.yaml (unknown keys are ignored by GazeMappingConfig)."""
    if not CONFIG_PATH.exists():
        return {}
    import yaml
    from uniface.constants import GazeWeights

    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    kwargs: dict = {}
    if "gaze_model" in data:
        kwargs["gaze_model"] = GazeWeights[str(data["gaze_model"]).upper()]
    if "detect_every" in data:
        kwargs["detect_every"] = int(data["detect_every"])
    if "detect_size" in data:
        kwargs["detect_size"] = int(data["detect_size"])
    if "ort_threads" in data:
        kwargs["ort_threads"] = int(data["ort_threads"])
    if "eye_mesh" in data:
        kwargs["eye_mesh"] = bool(data["eye_mesh"])
    return kwargs


def _load_blink_min_ms() -> float:
    if not CONFIG_PATH.exists():
        return 450.0
    import yaml

    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return float(data.get("blink_min_ms", 450.0))


class GazePipeline:
    def __init__(self, camera_index: int = 0, target_fps: int = 30, gestures: bool = True) -> None:
        """gestures=False builds a pure gaze pipeline (no face mesh / blink / mouth detection) -- what
        calibration uses, so nothing but the gaze model can influence the data it records."""
        self.webcam = Webcam(camera_index=camera_index, target_fps=target_fps)
        estimator_kwargs = _load_estimator_kwargs()
        # (gestures=False no longer turns the mesh off: calibration needs its eye features. It only means no
        # blink/mouth holds, which are off by default anyway.)
        self.estimator = FaceGazeEstimator(**estimator_kwargs)
        # The gaze point is frozen while a gesture is in progress ONLY for the gesture that is the active
        # select mode (set by the backend). A false positive (looking down lowers the lids; a slightly open
        # mouth) must never freeze the cursor in other modes or during calibration.
        self.hold_on_blink = False
        self.hold_on_mouth = False
        mapping_config = (
            GazeMappingConfig.from_yaml(str(CONFIG_PATH))
            if CONFIG_PATH.exists()
            else GazeMappingConfig()
        )
        self.mapper = GazeToScreenMapper(mapping_config)
        self.fixation = FixationAggregator(**_load_fixation_kwargs())
        self.blink = BlinkDetector(min_ms=_load_blink_min_ms())
        self.mouth = MouthOpenDetector(min_ms=_load_blink_min_ms() * 0.9)
        self._last_mapped = None
        self._angle_buf: deque[tuple[float, float, float]] = deque(maxlen=40)  # (t, pitch, yaw)
        self._feat_buf: deque = deque(maxlen=40)  # (t, eye feature vector)

    def stream(self) -> Iterator[dict[str, Any]]:
        last_time = time.monotonic()
        for frame in self.webcam.frames():
            now = time.monotonic()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            infer_start = time.monotonic()
            result = self.estimator.estimate(frame)
            infer_ms = (time.monotonic() - infer_start) * 1000.0
            if result is None:
                self.mapper.reset()
                self.fixation.reset()
                self.blink.reset()
                self.mouth.reset()
                self._last_mapped = None
                self._angle_buf.clear()
                self._feat_buf.clear()
                yield {
                    "gaze_x": None,
                    "gaze_y": None,
                    "confidence": 0.0,
                    "selected_key": None,
                    "face_found": False,
                    "fixation": None,
                    "fps": fps,
                    "infer_ms": infer_ms,
                    "frame": frame,
                }
                continue

            h, w = frame.shape[:2]
            bx1, by1, bx2, by2 = result.bbox
            face_feat = ((bx1 + bx2) / 2.0 / w, (by1 + by2) / 2.0 / h, (bx2 - bx1) / float(w))
            blink = self.blink.update(result.ear, now)
            mouth = self.mouth.update(result.mar, now)
            hold_gaze = (self.hold_on_blink and blink.eyes_closed) or (self.hold_on_mouth and mouth.active)
            self.estimator.track_through_closure = self.hold_on_blink
            if hold_gaze and self._last_mapped is not None:
                # Eyes shut: the gaze CNN's output is garbage. Don't feed it to the
                # mapper (its bias auto-centring would absorb it), the One Euro filter
                # or the fixation window -- hold the last good point instead.
                mapped = self._last_mapped
            else:
                mapped = self.mapper.map(result.pitch, result.yaw, timestamp=now, face=face_feat, features=result.eye_feat)
                self._last_mapped = mapped
            # Mean of the raw (absolute) CNN angles over the last 0.5s of un-held frames: what calibration records
            # for a trial instead of one noisy frame (per-frame scatter is ~0.05 and would otherwise be baked
            # into every training input).
            if not hold_gaze:
                self._angle_buf.append((now, result.pitch, result.yaw))
                if result.eye_feat is not None:
                    self._feat_buf.append((now, result.eye_feat))
            recent = [(p, y) for t, p, y in self._angle_buf if now - t <= 0.5]
            angle_mean = (sum(p for p, _ in recent) / len(recent), sum(y for _, y in recent) / len(recent)) if len(recent) >= 3 else None
            recent_feats = [f for t, f in self._feat_buf if now - t <= 0.5]
            feat_mean = sum(recent_feats) / len(recent_feats) if len(recent_feats) >= 3 else None
            if hold_gaze:
                fixation_event = None
            elif mapped.on_screen:
                fixation_event = self.fixation.update(mapped.raw_x, mapped.raw_y, now)
            else:
                fixation_event = self.fixation.update(None, None, now)
            yield {
                "gaze_x": mapped.smoothed_x,
                "gaze_y": mapped.smoothed_y,
                "on_keyboard": mapped.on_screen,
                "ear": result.ear,
                "raw_angle_mean": angle_mean,
                "eye_feat_mean": feat_mean,
                "eye_feat_recent": recent_feats,
                "face_feat": face_feat,
                "eyes_closed": blink.eyes_closed,
                "gaze_held": hold_gaze,
                "mar": result.mar,
                "mouth_open": mouth.active,
                "mouth_progress": mouth.progress,
                "mouth_fired": mouth.fired,
                "mouth_since": mouth.since,
                "blink_progress": blink.progress,
                "blink_fired": blink.fired,
                "blink_closed_since": blink.closed_since,
                "confidence": result.detection_confidence,
                "selected_key": None,
                "face_found": True,
                "pitch": result.pitch,
                "yaw": result.yaw,
                "bbox": result.bbox,
                "fixation": fixation_event,
                "fps": fps,
                "infer_ms": infer_ms,
                "frame": frame,
                "face_result": result,
            }

    def close(self) -> None:
        self.webcam.release()

    def __enter__(self) -> "GazePipeline":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
