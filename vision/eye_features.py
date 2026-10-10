"""Head-normalised eye / iris landmark features for gaze regression.

Adapted from EyeTrax (https://github.com/ck-zhang/EyeTrax, MIT licence, (c) ck-zhang), `GazeEstimator.extract_features`
and `constants.py`. Same idea and same landmark subset, reimplemented here on top of the face mesh we already run
(uniface `FaceMesh` V2_478 = MediaPipe's Face Landmarker, 478 points with irises, identical index layout):

 1. Build a head-aligned frame from the landmarks: origin at the midpoint of the outer eye corners (33, 263),
    x-axis along the eye line, y-axis towards the top of the head (10) made orthogonal to x, z = x cross y.
 2. Rotate every landmark into that frame and divide by the inter-corner distance. Head translation, in-plane
    rotation, tilt and distance (scale) are largely removed, so what remains is mostly eye-in-socket geometry --
    exactly the signal the iris position carries.
 3. Keep the eye-contour / iris / brow points plus a few face-outline anchors, flatten (x, y, z), and append the
    head yaw / pitch / roll.

The regression that consumes these (standardise + ridge) lives in experiments/calibrate_uncertainty.py
and vision/feature_model.py.
"""

from __future__ import annotations

import numpy as np

# fmt: off
LEFT_EYE_INDICES = [
    107,  66, 105,  63,  70,
     55,  65,  52,  53,  46,
    468, 469, 470, 471, 472,
    133,  33,
    173, 157, 158, 159, 160, 161, 246,
    155, 154, 153, 145, 144, 163,   7,
    243, 190,  56,  28,  27,  29,  30, 247,
    130,  25, 110,  24,  23,  22,  26, 112,
    244, 189, 221, 222, 223, 224, 225, 113,
    226,  31, 228, 229, 230, 231, 232, 233,
    193, 245, 128, 121, 120, 119, 118, 117,
    111,  35, 124, 143, 156,
]

RIGHT_EYE_INDICES = [
    336, 296, 334, 293, 300,
    285, 295, 282, 283, 276,
    473, 476, 475, 474, 477,
    362, 263,
    398, 384, 385, 386, 387, 388, 466,
    382, 381, 380, 374, 373, 390, 249,
    463, 414, 286, 258, 257, 259, 260, 467,
    359, 255, 339, 254, 253, 252, 256, 341,
    464, 413, 441, 442, 443, 444, 445, 342,
    446, 261, 448, 449, 450, 451, 452, 453,
    417, 465, 357, 350, 349, 348, 347, 346,
    340, 265, 353, 372, 383,
]

MUTUAL_INDICES = [
      4,   # nose
     10,   # very top
    151,   # forehead
      9,   # between the brows
    152,   # chin
    234,   # very left
    454,   # very right
     58,   # left jaw
    288,   # right jaw
]
# fmt: on

SUBSET_INDICES = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES
N_FEATURES = len(SUBSET_INDICES) * 3 + 3  # (x, y, z) per point + yaw, pitch, roll
MIN_LANDMARKS = max(SUBSET_INDICES) + 1  # 478: needs the iris-capable mesh


# Feature subsets for small calibration sets. EyeTrax collects many frames per calibration point and regresses on all
# ~490 features; with only a few dozen trials most of those dimensions are noise for gaze, so calibration also tries
# these narrower views and keeps whichever is best on held-out trials. Column indices into the feature vector.
_N_LEFT = len(LEFT_EYE_INDICES)


def _cols(positions: list[int]) -> list[int]:
    return [3 * p + k for p in positions for k in range(3)]


_POSE_COLS = [N_FEATURES - 3, N_FEATURES - 2, N_FEATURES - 1]
FEATURE_SUBSETS: dict[str, list[int]] = {
    # iris centre/edges + the four eye corners of each eye + head pose angles (45 dims)
    "iris_core": _cols(list(range(10, 17)) + [_N_LEFT + p for p in range(10, 17)]) + _POSE_COLS,
    # everything around the eyes (lids, rings), no brows / face outline (~405 dims)
    "eye_region": _cols(list(range(10, _N_LEFT)) + [_N_LEFT + p for p in range(10, _N_LEFT)]) + _POSE_COLS,
    # all of it, like EyeTrax (486 dims)
    "full": list(range(N_FEATURES)),
}


def head_aligned_features(points_xyz: np.ndarray) -> np.ndarray | None:
    """points_xyz: (478, 3) landmarks with x, y as a fraction of the image width / height and z on the same scale
    as x (MediaPipe's convention). Returns the N_FEATURES-vector, or None if the points are unusable."""
    if points_xyz.shape[0] < MIN_LANDMARKS:
        return None
    pts = points_xyz.astype(np.float64)
    left_corner, right_corner, top_of_head = pts[33], pts[263], pts[10]

    eye_center = (left_corner + right_corner) / 2.0
    shifted = pts - eye_center
    x_axis = right_corner - left_corner
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
    y_approx = top_of_head - eye_center
    y_approx = y_approx / (np.linalg.norm(y_approx) + 1e-9)
    y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
    rot = np.column_stack((x_axis, y_axis, z_axis))
    rotated = (rot.T @ shifted.T).T

    inter_eye = np.linalg.norm(rot.T @ (right_corner - eye_center) - rot.T @ (left_corner - eye_center))
    if inter_eye < 1e-7:
        return None
    rotated = rotated / inter_eye

    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    return np.concatenate([rotated[SUBSET_INDICES].ravel(), [yaw, pitch, roll]]).astype(np.float32)


def features_from_pixel_landmarks(landmarks_px: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray | None:
    """Convert uniface FaceMesh output (pixel x/y, z on the pixel scale) to MediaPipe-style normalised coordinates
    and extract the features."""
    if landmarks_px.shape[0] < MIN_LANDMARKS or landmarks_px.shape[1] < 3:
        return None
    pts = landmarks_px[:, :3].astype(np.float64).copy()
    pts[:, 0] /= frame_w
    pts[:, 1] /= frame_h
    pts[:, 2] /= frame_w
    return head_aligned_features(pts)
