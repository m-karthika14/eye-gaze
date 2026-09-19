# /vision

Face detection, gaze estimation, uncertainty estimation, and temporal (fixation/dwell) processing.

- **Phase 1 (done):** webcam capture (OpenCV, 30 FPS target) → face detection + full-face gaze CNN via [`uniface`](https://github.com/yakhyo/uniface) (RetinaFace + MobileGaze, ResNet34 by default; MIT-licensed, pretrained, weights auto-download + SHA-256 verified — satisfies "pretrained appearance-based gaze model, not trained from scratch") → heuristic calibration-free mapping to a normalized on-screen (x, y) point → live debug visualization.
- **Phase 3 (done):** `gaze_to_screen.py`'s `map()` now also exposes the raw (pre-filter) point, which `fixation.py`'s `FixationAggregator` groups into fixation windows (200–800ms) and summarizes as `(mu, sigma)` from real sample-to-sample scatter — not a fixed guess. `/decoder/key_probability.py` integrates that Gaussian over the keyboard to get a full per-key distribution (see `/decoder`'s README). `/experiments/calibrate_uncertainty.py` empirically validates sigma against known targets.
- **Phase 7:** online per-user gaze-bias correction (EyeO-style read-back adaptation), fed back into key-probability mapping.

## Files

| File | Role |
|---|---|
| `webcam.py` | `Webcam` — OpenCV capture generator paced to a target FPS |
| `face_gaze.py` | `FaceGazeEstimator` — RetinaFace detection + MobileGaze pitch/yaw via `uniface` |
| `gaze_to_screen.py` | `GazeToScreenMapper` — heuristic pitch/yaw → normalized screen point (bias-corrected, auto-ranged, One Euro Filter-smoothed); returns both the raw and smoothed point |
| `fixation.py` | `FixationAggregator` — groups raw samples into a fixation window, returns `FixationEvent(mu, sigma, n_samples, duration_ms)` |
| `pipeline.py` | `GazePipeline` — ties the above together; yields dicts matching the Section 16 WebSocket schema plus debug extras (including `fixation`) |
| `one_euro_filter.py` | `OneEuroFilter` — adaptive smoothing (Casiez et al. 2012), fast during saccades, stable during fixation |
| `run_demo.py` | Interactive live visualization: `.venv/Scripts/python.exe -m vision.run_demo` (press `q` to quit) |
| `smoke_test.py` | Headless N-frame check, no GUI: `.venv/Scripts/python.exe -m vision.smoke_test [num_frames]` |

The mapping heuristic's constants live in `config/gaze_mapping.yaml`; fixation window and calibration constants live in `config/decoder_params.yaml`.

**Status:** Phase 1 and Phase 3 complete and verified — synthetic tests confirm `FixationAggregator` recovers injected noise levels correctly, and the pipeline was tested live end to end through `/backend`. **`config/decoder_params.yaml`'s `gaze_uncertainty.sigma_scale` still needs to be set from real data** — run `experiments/calibrate_uncertainty.py` yourself (needs your webcam + eyes) to complete Phase 3's calibration requirement.
