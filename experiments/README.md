# /experiments

Runnable, reproducible scripts for the five systems compared in this research (see `MASTER_PROJECT_DOCUMENT.md` Section 10):

| System | What It Does |
|---|---|
| S0 | Gaze → nearest key → text (baseline, no language, no adaptation) |
| S1 | Hard-selected key → post-hoc language correction |
| S2 | S1 + adaptive / confidence-aware keyboard |
| S3 | S2 + online user adaptation |
| S4 | S3 + joint gaze-probability + language decoding (core method) |

- **Phase 2:** S0 baseline.
- **Phase 5:** S1–S4 implemented, all runnable against identical recorded gaze sequences (critical for the S1-vs-S4 comparison).
- **Phase 9:** the four research experiments (H1 main comparison, noise sensitivity, H2 online-adaptation comparison, real-world validation) run and logged here.

**Status:** not yet implemented.
