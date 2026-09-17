# /vision

Face detection, gaze estimation, uncertainty estimation, and temporal (fixation/dwell) processing.

- **Phase 1:** webcam capture (OpenCV, 30 FPS), face/eye detection (MediaPipe), pretrained appearance-based gaze model (iTracker / full-face-CNN lineage), raw (x, y) gaze point output.
- **Phase 3:** extend to (x, y) + uncertainty (σ), Gaussian gaze modelling `G ~ N(μ, Σ)`, fixation/dwell temporal aggregation, post-hoc uncertainty calibration.
- **Phase 7:** online per-user gaze-bias correction (EyeO-style read-back adaptation), fed back into key-probability mapping.

**Status:** not yet implemented.
