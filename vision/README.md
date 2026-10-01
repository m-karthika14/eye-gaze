# /vision

Face detection, gaze estimation, uncertainty estimation, and temporal (fixation/dwell) processing.

- **Phase 1 (done):** webcam capture (OpenCV, 30 FPS target) → face detection + full-face gaze CNN via [`uniface`](https://github.com/yakhyo/uniface) (RetinaFace + MobileGaze, ResNet34 by default; MIT-licensed, pretrained, weights auto-download + SHA-256 verified — satisfies "pretrained appearance-based gaze model, not trained from scratch") → heuristic calibration-free mapping to a normalized on-screen (x, y) point → live debug visualization.
- **Phase 3 (done):** `gaze_to_screen.py`'s `map()` now also exposes the raw (pre-filter) point, which `fixation.py`'s `FixationAggregator` groups into fixation windows (200–800ms) and summarizes as `(mu, sigma)` from real sample-to-sample scatter — not a fixed guess. `/decoder/key_probability.py` integrates that Gaussian over the keyboard to get a full per-key distribution (see `/decoder`'s README). `/experiments/calibrate_uncertainty.py` empirically validates sigma against known targets.
- **Phase 7 (scoped slice pulled forward):** the one-time startup bias lock has a real failure mode on longer sessions -- head/posture drift partway through silently invalidates it (confirmed on real calibration data: per-key raw-angle readings crept monotonically across repeats within one multi-minute session). `gaze_to_screen.py`'s `GazeToScreenMapper` now keeps the bias continuously chasing a long-time-constant (`drift_tau_s`, default 30s) exponential moving average of the raw signal after the initial lock, so sustained drift self-corrects while a single key's dwell (under ~2s) is too short to meaningfully move it. Verified synthetically: a 1.5s dwell at an extreme gaze angle barely nudges the bias (<0.01 rad), while a genuine 60s drift to a new resting position is substantially corrected. Toggleable via `drift_correction_enabled` in `config/gaze_mapping.yaml`. Full Phase 7 (EyeO-style read-back adaptation, anchored on specific UI regions rather than an undifferentiated running average) is still future work.

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

**Status:** Phase 1 and Phase 3 complete, plus a scoped Phase 7 slice (continuous drift correction). Synthetic tests confirm `FixationAggregator` recovers injected noise correctly, and `experiments/calibrate_uncertainty.py` has been run against real data: `config/gaze_mapping.yaml`'s `half_fov_*_rad` were re-derived from real gaze angles (fixing a range that had gotten permanently stuck too wide -- see its commit history), and `config/decoder_params.yaml`'s `gaze_uncertainty.sigma_scale` reflects a real, empirically-measured uncertainty level, not a guess. That value is large -- webcam appearance-based gaze estimation genuinely is this imprecise, especially at extreme angles -- which is exactly the uncertainty Phase 5's joint decoder is designed to work with rather than assume away.
