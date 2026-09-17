# /personalization

Lightweight online user model — learns per-user gaze bias from natural interaction, with no explicit calibration session (C3, secondary research contribution; tests H2).

- **Phase 7:** implement the EyeO-style mechanism — use typed-text read-back moments (user looking at what they just typed) as an implicit, known-target calibration signal; estimate a running per-user gaze-bias correction; feed the corrected distribution back into `/decoder`'s key-probability mapping; wire into S3 (replacing the Phase-5 stub).

**Status:** not yet implemented.
