"""Face detection + full-face appearance-based gaze estimation (Phase 1).

Uses `uniface` (MIT-licensed, ONNX Runtime): RetinaFace for face detection and
MobileGaze (ResNet/MobileNet full-face CNN, pretrained on gaze estimation
datasets — iTracker / full-face-CNN lineage, see MASTER_PROJECT_DOCUMENT.md
Section 1.2/14) for pitch/yaw gaze direction. Weights auto-download on first
use and are SHA-256 verified; no training from scratch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
from uniface.constants import FaceMeshWeights, GazeWeights
from uniface.detection import RetinaFace
from uniface.gaze import MobileGaze
from uniface.landmark import FaceMesh

from .blink import eye_aspect_ratio, mouth_aspect_ratio
from .eye_features import features_from_pixel_landmarks

# Below this EAR the eyes are shut (open is ~0.30). Used to keep tracking the face
# through a closure: face detectors often lose a face whose eyes are closed.
EAR_CLOSED_HINT = 0.20
MAX_CLOSED_TRACK_FRAMES = 30


@dataclass
class FaceGazeResult:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels
    detection_confidence: float
    pitch: float  # radians; + = looking up
    yaw: float  # radians; + = looking right
    landmarks: list[tuple[float, float]] | None = None
    ear: float | None = None  # eye aspect ratio (openness) from the face mesh; None if blink detection is off
    mar: float | None = None  # mouth aspect ratio (openness) from the face mesh
    eye_feat: np.ndarray | None = None  # head-normalised eye/iris landmark features (vision/eye_features.py)


def limit_onnx_threads(n: int) -> None:
    """Cap ONNX Runtime's intra-op thread pool for every session created after
    this call (uniface builds its sessions with default options, which we can't
    pass through). Measured on an 8-logical-core hybrid CPU: MobileOne-S0 gaze
    took 344 ms with the default thread count vs 14 ms with 4 threads -- the
    default pool oversubscribes the cores and spin-waits. Spinning is the other
    half of it: with 3 sessions (detector, gaze, mesh) the idle workers busy-wait
    and burned ~290 ms of CPU per frame vs ~33 ms with spinning off (measured), saturating
    the whole machine and dragging the app to ~7 fps. n <= 0 keeps ORT's default."""
    if n <= 0:
        return
    cv2.setNumThreads(min(n, 2))  # OpenCV's own pool spins the same way during the mesh warp
    base = ort.SessionOptions

    class _Options(base):  # type: ignore[valid-type, misc]
        def __init__(self) -> None:
            super().__init__()
            self.intra_op_num_threads = n
            self.inter_op_num_threads = 1
            self.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.add_session_config_entry("session.intra_op.allow_spinning", "0")

    ort.SessionOptions = _Options


class FaceGazeEstimator:
    """Detects the primary (most central) face in a frame and its gaze direction.

    Speed matters more than it looks: dwell typing needs several gaze samples
    per fixation to average noise down, and the originally-measured cost
    (RetinaFace at 640x640 ~110-200ms + ResNet34 ~140ms per frame) gave ~3 fps,
    i.e. ~2 samples per 600ms dwell. Three changes bring that to ~20 fps:

    - detector runs at `detect_size` (default 320) instead of 640x640;
    - the detector only runs every `detect_every` frames -- in between, the
      previous (EMA-smoothed) face box is reused, since a face moves little
      in 4 frames. Any frame where the detector finds nothing drops tracking;
    - the gaze CNN defaults to MobileOne-S0 (~30ms vs ~140ms for ResNet34;
      uniface's Gaze360 benchmark has them within ~1.3 deg of each other,
      and averaging ~4x more samples per fixation more than makes that up).
    """

    def __init__(
        self,
        gaze_model: GazeWeights = GazeWeights.MOBILEONE_S0,
        detect_size: int = 320,
        detect_every: int = 4,
        ort_threads: int = 2,
        eye_mesh: bool = True,
    ) -> None:
        limit_onnx_threads(ort_threads)
        self._detector = RetinaFace(input_size=(detect_size, detect_size))
        self._gaze = MobileGaze(model_name=gaze_model)
        # Face mesh (468 pts) only for eye openness -> blink-to-select (vision/blink.py).
        # The iris-capable V2 mesh (478 pts): EAR/MAR use the same first-468 layout as V1, and the iris points feed
        # the EyeTrax-style eye features used for calibration/regression.
        self._mesh = FaceMesh(model_name=FaceMeshWeights.V2_478) if eye_mesh else None
        self._detect_every = max(1, detect_every)
        self._frames_since_detect = 0
        self._bbox: np.ndarray | None = None  # smoothed float x1,y1,x2,y2
        self._confidence = 0.0
        self._landmarks: list[tuple[float, float]] | None = None
        self._last_ear: float | None = None
        self._closed_frames = 0
        # Only keep tracking through an apparent eye closure when blink-to-select is the active
        # mode (set by the pipeline). Otherwise a falsely-low eye reading (e.g. looking down)
        # would stop the face detector re-running and let the crop go stale.
        self.track_through_closure = False

    def reset_tracking(self) -> None:
        self._bbox = None
        self._frames_since_detect = 0

    def _detect(self, frame: np.ndarray) -> bool:
        faces = self._detector.detect(frame, max_num=1, metric="max")
        if not faces:
            self._bbox = None
            return False
        face = faces[0]
        new_box = np.asarray(face.bbox, dtype=np.float64)[:4]
        # EMA the box across detections so the crop (and so the gaze CNN's
        # input) doesn't jitter frame to frame.
        self._bbox = new_box if self._bbox is None else 0.5 * self._bbox + 0.5 * new_box
        self._confidence = float(face.confidence)
        self._landmarks = None
        if getattr(face, "landmarks", None) is not None:
            self._landmarks = [(float(pt[0]), float(pt[1])) for pt in face.landmarks]
        return True

    def _mesh_features(self, frame: np.ndarray) -> tuple[float | None, float | None, np.ndarray | None]:
        if self._mesh is None or self._bbox is None:
            return None, None, None
        try:
            kp = np.asarray(self._landmarks, dtype=np.float32) if self._landmarks else None
            res = self._mesh.predict(frame, bboxes=[self._bbox], keypoints=None if kp is None else [kp])[0]
            pts = res.landmarks  # (478, 3): pixel x/y, z on the same scale
            feat = features_from_pixel_landmarks(pts, frame.shape[1], frame.shape[0])
            return eye_aspect_ratio(pts[:, :2]), mouth_aspect_ratio(pts[:, :2]), feat
        except Exception:  # noqa: BLE001 -- a bad mesh frame must not kill gaze tracking
            return None, None, None

    def estimate(self, frame: np.ndarray) -> FaceGazeResult | None:
        # While the eyes are shut, don't re-run the detector (it often loses a face with
        # closed eyes) -- keep tracking the last box until they reopen.
        eyes_shut = self.track_through_closure and (
            self._last_ear is not None
            and self._last_ear < EAR_CLOSED_HINT
            and self._closed_frames < MAX_CLOSED_TRACK_FRAMES
        )
        if self._bbox is None or (self._frames_since_detect >= self._detect_every and not eyes_shut):
            self._frames_since_detect = 0
            previous_bbox = self._bbox
            if not self._detect(frame):
                # Detector found nothing. Either the face left, or the eyes just closed.
                # Probe the previous box with the mesh: if it reads as eyes-closed, keep going.
                self._bbox = previous_bbox
                ear, _, _ = self._mesh_features(frame)
                if previous_bbox is None or not self.track_through_closure or ear is None or ear >= EAR_CLOSED_HINT:
                    self._bbox = None
                    self._last_ear = None
                    return None
        self._frames_since_detect += 1

        x1, y1, x2, y2 = (int(v) for v in self._bbox)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            self._bbox = None
            return None

        gaze = self._gaze.estimate(frame[y1:y2, x1:x2])

        ear, mar, eye_feat = self._mesh_features(frame)
        self._last_ear = ear
        self._closed_frames = self._closed_frames + 1 if (ear is not None and ear < EAR_CLOSED_HINT) else 0

        return FaceGazeResult(
            bbox=(x1, y1, x2, y2),
            detection_confidence=self._confidence,
            pitch=float(gaze.pitch),
            yaw=float(gaze.yaw),
            landmarks=self._landmarks,
            ear=ear,
            mar=mar,
            eye_feat=eye_feat,
        )


def _find_pupil_center(frame: np.ndarray, eye_pt: tuple[float, float], radius: int = 14) -> tuple[int, int]:
    """Finds the darkest circular region (pupil / iris center) around the eye landmark."""
    h, w = frame.shape[:2]
    cx, cy = int(eye_pt[0]), int(eye_pt[1])
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(w, cx + radius), min(h, cy + radius)
    if x2 <= x1 or y2 <= y1:
        return (cx, cy)
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    # The pupil/iris is the darkest spot in the eye crop
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    min_val, _, min_loc, _ = cv2.minMaxLoc(blurred)
    # Smooth with the detector landmark to prevent noise jumps
    px = int(x1 + min_loc[0] * 0.65 + (cx - x1) * 0.35)
    py = int(y1 + min_loc[1] * 0.65 + (cy - y1) * 0.35)
    return (px, py)


def annotate_eye_tracker_frame(
    frame: np.ndarray,
    result: FaceGazeResult | None,
    pitch_deg: float | None = None,
    yaw_deg: float | None = None,
    gaze_screen_pt: tuple[float, float] | None = None,
) -> np.ndarray:
    """Draws a live eye-tracking overlay: facial mesh, eye contours, bright blue

    dots centered on the eyeballs, and gaze vectors.
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    if result is None or not result.landmarks:
        # No face detected indicator
        cv2.putText(
            vis,
            "LOOK AT WEBCAM - NO FACE DETECTED",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        return vis

    # 1. Face bounding box (sleek rounded corner style)
    x1, y1, x2, y2 = result.bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 100), 1)

    lms = result.landmarks
    # RetinaFace 5 landmarks: 0=left eye, 1=right eye, 2=nose, 3=left mouth, 4=right mouth
    pts = [(int(p[0]), int(p[1])) for p in lms]

    # 2. Facial wireframe mesh (cyan lines connecting facial landmarks)
    mesh_color = (255, 200, 50)  # cyan-amber wireframe
    if len(pts) >= 5:
        # Eye-to-eye
        cv2.line(vis, pts[0], pts[1], mesh_color, 1)
        # Eyes to nose
        cv2.line(vis, pts[0], pts[2], mesh_color, 1)
        cv2.line(vis, pts[1], pts[2], mesh_color, 1)
        # Nose to mouth
        cv2.line(vis, pts[2], pts[3], mesh_color, 1)
        cv2.line(vis, pts[2], pts[4], mesh_color, 1)
        # Mouth bar
        cv2.line(vis, pts[3], pts[4], mesh_color, 1)

    # 3. Eyeball tracking & blue dots
    eyes = [lms[0], lms[1]]  # left and right eye centers
    for eye_pt in eyes:
        pupil_x, pupil_y = _find_pupil_center(vis, eye_pt)

        # Outer eye socket outline (white oval)
        cv2.ellipse(vis, (pupil_x, pupil_y), (14, 9), 0, 0, 360, (255, 255, 255), 1)

        # Outer iris boundary (soft cyan)
        cv2.circle(vis, (pupil_x, pupil_y), 7, (255, 200, 0), 1)

        # BRIGHT BLUE DOT centered on the eyeball / pupil
        # Deep blue base + vibrant sky blue center + white pinpoint highlight
        cv2.circle(vis, (pupil_x, pupil_y), 6, (255, 100, 0), -1)  # BGR: blue core
        cv2.circle(vis, (pupil_x, pupil_y), 3, (255, 220, 50), -1)  # bright cyan highlight
        cv2.circle(vis, (pupil_x, pupil_y), 1, (255, 255, 255), -1)  # specular white pinpoint

        # Gaze direction vector ray extending from each eyeball
        p_rad = result.pitch
        y_rad = result.yaw
        ray_len = 35
        ray_dx = int(-math.sin(y_rad) * ray_len)
        ray_dy = int(-math.sin(p_rad) * ray_len)
        cv2.arrowedLine(
            vis,
            (pupil_x, pupil_y),
            (pupil_x + ray_dx, pupil_y + ray_dy),
            (0, 255, 255),  # Yellow gaze ray
            2,
            tipLength=0.3,
        )

    # 4. HUD status header
    p_deg = pitch_deg if pitch_deg is not None else math.degrees(result.pitch)
    y_deg = yaw_deg if yaw_deg is not None else math.degrees(result.yaw)
    hud_text = f"PITCH: {p_deg:+.1f}deg  YAW: {y_deg:+.1f}deg  [EYES: TRACKED]"
    cv2.putText(vis, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)

    if gaze_screen_pt:
        screen_text = f"SCREEN TARGET: X={gaze_screen_pt[0]:.2f} Y={gaze_screen_pt[1]:.2f}"
        cv2.putText(vis, screen_text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 100), 1)

    return vis
