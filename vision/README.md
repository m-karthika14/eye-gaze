# /vision

Face detection, gaze estimation, uncertainty estimation, and temporal (fixation/dwell) processing.

- **Phase 1 (done):** webcam capture (OpenCV, 30 FPS target) → face detection + full-face gaze CNN via [`uniface`](https://github.com/yakhyo/uniface) (RetinaFace + MobileGaze, ResNet34 by default; MIT-licensed, pretrained, weights auto-download + SHA-256 verified — satisfies "pretrained appearance-based gaze model, not trained from scratch") → heuristic calibration-free mapping to a normalized on-screen (x, y) point → live debug visualization.
- **Phase 3:** replace `gaze_to_screen.py`'s heuristic mapping with a calibrated probability distribution: (x, y) + uncertainty (σ), Gaussian gaze modelling `G ~ N(μ, Σ)`, fixation/dwell temporal aggregation, post-hoc uncertainty calibration.
- **Phase 7:** online per-user gaze-bias correction (EyeO-style read-back adaptation), fed back into key-probability mapping.

## Files

| File | Role |
|---|---|
| `webcam.py` | `Webcam` — OpenCV capture generator paced to a target FPS |
| `face_gaze.py` | `FaceGazeEstimator` — RetinaFace detection + MobileGaze pitch/yaw via `uniface` |
| `gaze_to_screen.py` | `GazeToScreenMapper` — heuristic pitch/yaw → normalized screen point, EMA-smoothed (placeholder until Phase 3) |
| `pipeline.py` | `GazePipeline` — ties the above together; yields dicts matching the Section 16 WebSocket schema plus debug extras |
| `run_demo.py` | Interactive live visualization: `.venv/Scripts/python.exe -m vision.run_demo` (press `q` to quit) |
| `smoke_test.py` | Headless N-frame check, no GUI: `.venv/Scripts/python.exe -m vision.smoke_test [num_frames]` |

The mapping heuristic's constants live in `config/gaze_mapping.yaml`.

**Status:** Phase 1 complete. Face/gaze estimation verified importable and the API contract confirmed; run `run_demo.py` yourself to visually confirm live tracking (requires a real webcam + display, which isn't available in the tool sandbox this was built in).
