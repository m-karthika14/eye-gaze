# /data

Dataset processing, evaluation data, and simulation outputs.

| Dataset | Role |
|---|---|
| MPIIGaze / MPIIFaceGaze | Primary — realistic laptop-style gaze estimation |
| Gaze360 | Robustness — head-pose/distance/indoor-outdoor variation; provides uncertainty estimates |
| GazeCapture | Generalisation/pretraining |
| Custom collected dataset | Webcam validation, keyboard-gaze, online adaptation, natural interaction |
| Text corpus | Training/evaluating the local character-level LM and word completion |

Raw and processed data are git-ignored (see root `.gitignore`) — this folder holds processing scripts, not the datasets themselves. See `config/evaluation_phrases.txt` for the current evaluation phrase set (Phase 0).

`experiments/calibrate_uncertainty.py` (Phase 3) writes its recorded trials to `data/uncertainty_calibration_samples.jsonl` — also git-ignored, since it's personal to one person's camera/screen setup and meant to be regenerated, not version-controlled.

**Status:** empty (aside from git-ignored generated files). Populated starting **Phase 1** (gaze datasets) and **Phase 3–4** (validation/text corpus).
