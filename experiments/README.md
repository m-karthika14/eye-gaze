# /experiments

Runnable, reproducible scripts for the five systems compared in this research (see `MASTER_PROJECT_DOCUMENT.md` Section 10):

| System | What It Does |
|---|---|
| S0 | Gaze → nearest key → text (baseline, no language, no adaptation) |
| S1 | Hard-selected key → post-hoc language correction |
| S2 | S1 + adaptive / confidence-aware keyboard |
| S3 | S2 + online user adaptation |
| S4 | S3 + joint gaze-probability + language decoding (core method) |

- **Phase 2 (done):** `s0.py` — `S0System` composes `decoder.nearest_key` + `decoder.dwell` + `decoder.text_buffer` into the S0 baseline. Used live by `/backend` (one sample at a time via `process_sample`) and, later, by Phase 9 for batch replay of recorded/simulated gaze sequences via `run_sequence`. `test_s0_synthetic.py` is a regression check against a fabricated gaze path (no webcam needed): `.venv/Scripts/python.exe -m experiments.test_s0_synthetic`.
- **Phase 3 (done):** `calibrate_uncertainty.py` — shows a sequence of known target keys, records predicted `(mu, sigma)` against ground truth, and computes/writes back a `sigma_scale` correction factor so Phase 3's uncertainty isn't trusted blindly (needs your webcam + eyes, run it yourself: `.venv/Scripts/python.exe -m experiments.calibrate_uncertainty`). `test_uncertainty_synthetic.py` is the no-webcam regression check for the fixation aggregation, Gaussian key-probability math, and the calibration analysis itself.
- **Phase 5:** S1–S4 implemented, all runnable against identical recorded gaze sequences (critical for the S1-vs-S4 comparison).
- **Phase 9:** the four research experiments (H1 main comparison, noise sensitivity, H2 online-adaptation comparison, real-world validation) run and logged here.

**Status:** S0 and Phase 3's uncertainty machinery done and verified via synthetic tests + live end-to-end runs through `/backend`'s WebSocket. `calibrate_uncertainty.py` still needs to be run against real data to set `sigma_scale` (see `/vision`'s README).
