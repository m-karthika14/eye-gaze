"""S2: S1 + adaptive/confidence-aware keyboard (MASTER_PROJECT_DOCUMENT.md
Section 10).

The adaptive keyboard is a LIVE interaction feature (Phase 6): it resizes
keys during actual use, which changes what the user's eyes land on and so
changes the gaze that gets recorded. Replaying an already-recorded sequence
in this batch harness can't retroactively change what was recorded, so for
batch/replay evaluation S2's DECODING logic is identical to S1's -- they
only diverge in a live interactive session. This matches Section 10 anyway:
the adaptive keyboard is explicitly disabled during the S1-vs-S4 controlled
comparison, to avoid confounding the visual stimulus.
"""

from __future__ import annotations

from experiments.s1 import S1System


class S2System(S1System):
    """Decoding-wise identical to S1 for batch/recorded-sequence evaluation
    -- see module docstring."""
