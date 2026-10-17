"""Live (interactive) versions of S0-S3, next to the existing live S4 (decoder/s4_live.py), so the dashboard can switch between
all five systems on the same camera, mapping and selection method (blink / Space-Enter / dwell / mouth).

They subclass S4LiveDecoder, so hover, dwell, "select now" (blink etc.), cursor hold and candidate picking are identical to S4;
only what happens to a selected key differs (MASTER_PROJECT_DOCUMENT.md Section 10):

  S0  gaze -> nearest key -> text. No language model, no correction.
  S1  hard-selected keys during the word; at SPACE the word is corrected post hoc with the character LM using a generic,
      position-blind confusion model (the SAME maths as experiments/s1.py -- language only knows "this key was selected, typos
      happen sometimes", nothing about which neighbouring keys were plausible).
  S2  S1 + the confidence-aware adaptive keyboard (keys resize by next-letter likelihood; wired in backend/main.py).
  S3  S2 + online adaptation: a running estimate of the user's gaze bias, learned with no calibration screen from moments where the
      user's own behaviour reveals the intended key, and subtracted from later gaze:
        (a) BACKSPACE-and-retry: the gaze that picked the deleted key vs the key then chosen instead;
        (b) an LM correction of a real word: the gaze at each letter the LM changed vs the letter it changed it to.
      Letters the user typed correctly teach nothing (the gaze was inside the key either way), so they are not used.
  S4  the existing joint gaze-distribution + lexicon decoder (unchanged, decoder/s4_live.py).

Not part of the S4 research comparison: that runs in experiments/ on recorded or simulated sequences.
"""

from __future__ import annotations

import statistics
from typing import Optional

from decoder.beam_search import beam_search_decode
from decoder.nearest_key import nearest_key
from decoder.s4_live import S4LiveDecoder
from experiments.gaze_sequence import GazeObservation
from experiments.s0 import S0Event
from experiments.s1 import generic_confusion_likelihoods

# S1's correction strength. experiments/s1.py's shipped defaults (match 0.85, LM weight 0.25) never override a selected key -- S1 would
# behave exactly like S0 -- so the live S1 uses values tuned by grid search on the measured eye-landmark error (experiments/run_results.py
# tuning split, results/tuning_landmarks_loo.csv). Same tuning effort S4 gets.
S1_LIVE_MATCH_PROB = 0.35  # = the values chosen by the full tuning run on the within-run eye-landmark noise pool (results/tuning_landmarks_loo.csv)
S1_LIVE_LM_WEIGHT = 0.5


class S0LiveDecoder(S4LiveDecoder):
    """S0: nearest key, committed as typed."""

    mode = "s0"
    adaptive_layout = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._chars: list[str] = []  # the key actually selected for each letter of the current word (lowercase)

    # -- lifecycle ----------------------------------------------------------------------------------------------
    def reset(self) -> None:
        super().reset()
        self._chars.clear()

    def set_text(self, text: str) -> None:
        super().set_text(text)
        self._chars.clear()

    def pick_candidate(self, index: int) -> bool:
        ok = super().pick_candidate(index)
        if ok:
            self._chars.clear()
        return ok

    # -- hooks the S1-S3 subclasses override ------------------------------------------------------------------
    def _refresh(self) -> None:
        self.candidates = []

    def _word_to_commit(self) -> str:
        return "".join(self._chars)

    def _before_word_end(self, raw: str, committed: str) -> None:  # noqa: B027 -- intentional no-op hook
        return

    def _decode_active(self, **_unused: object) -> str:
        # Keep _chars aligned with the observations if a caller popped one (backend's low-confidence veto does).
        del self._chars[len(self.active_word_obs):]
        return "".join(self._chars)

    # -- selection ------------------------------------------------------------------------------------------------
    def _commit(
        self,
        mu_x: float,
        mu_y: float,
        std_x: float,
        std_y: float,
        sigma_scale: float,
        sigma_override: Optional[float] = None,
    ) -> Optional[str]:
        key = nearest_key(mu_x, mu_y, self.layout)

        if key == "BACKSPACE":
            if self.active_word_obs:
                self.active_word_obs.pop()
                if self._chars:
                    self._chars.pop()
                self.active_word = "".join(self._chars)
                self._refresh()
            elif self.committed_text:
                self.committed_text = self.committed_text[:-1]
                self.active_word = ""

        elif key == "SPACE":
            if self.active_word_obs:
                word = self._word_to_commit()
                self._before_word_end("".join(self._chars), word)
                self.committed_text = self.committed_text + word + " "
                self.active_word_obs.clear()
                self._chars.clear()
                self.active_word = ""
                self.candidates = []
            else:
                self.committed_text += " "

        elif key is not None and len(key) == 1:  # a letter key
            sigma = sigma_override if sigma_override is not None else self.sigma_floor
            self.active_word_obs.append(GazeObservation(mu_x=mu_x, mu_y=mu_y, sigma_x=sigma, sigma_y=sigma))
            self._chars.append(key.lower())
            self.active_word = "".join(self._chars)
            self._refresh()

        return key


class S1LiveDecoder(S0LiveDecoder):
    """S1: hard keys while typing, post-hoc character-LM correction of the whole word at SPACE."""

    mode = "s1"

    def __init__(
        self,
        *args,
        match_prob: float = S1_LIVE_MATCH_PROB,
        correction_lm_weight: float = S1_LIVE_LM_WEIGHT,
        correction_beam: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.match_prob = match_prob
        self.correction_lm_weight = correction_lm_weight
        self.correction_beam = correction_beam

    def _correct(self) -> str:
        """experiments/s1.py's decoder applied to this word: generic confusion likelihood around each selected key, character LM
        prior, beam search. Letters only inside a word (a "correction" must not invent a space)."""
        if not self._chars:
            return ""
        likelihoods = []
        for c in self._chars:
            full = generic_confusion_likelihoods(c, self.match_prob)
            likelihoods.append({k: v for k, v in full.items() if k != " "})
        context = self.committed_text[-3:] if self.committed_text else ""
        return beam_search_decode(
            likelihoods, self.char_lm, beam_width=self.correction_beam, initial_prefix=context, lm_weight=self.correction_lm_weight
        )

    def _refresh(self) -> None:
        raw = "".join(self._chars)
        corrected = self._correct()
        # first = what SPACE will commit; second = exactly what was typed
        self.candidates = [w for w in dict.fromkeys([corrected, raw]) if w][: self.max_candidates]

    def _word_to_commit(self) -> str:
        return self._correct()


class S2LiveDecoder(S1LiveDecoder):
    """S2: S1 + the adaptive keyboard. Decoding is identical to S1 (as in experiments/s2.py); the key resizing is a layout
    effect applied by backend/main.py whenever this mode is active (`adaptive_layout = True`)."""

    mode = "s2"
    adaptive_layout = True


class S3LiveDecoder(S2LiveDecoder):
    """S3: S2 + online bias adaptation (see the module docstring for what it learns from, and what it deliberately ignores)."""

    mode = "s3"

    def __init__(
        self,
        *args,
        bias_rate: float = 0.25,
        bias_clip: float = 0.15,
        max_residual: float = 0.25,
        min_word_len: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bias_rate = bias_rate
        self.bias_clip = bias_clip
        self.max_residual = max_residual  # a bigger gaze-vs-key gap than this is a wild miss, not a bias
        self.min_word_len = min_word_len
        self.bias = [0.0, 0.0]
        self._gaze_pts: list[tuple[float, float]] = []  # the real (bias-corrected) gaze behind each selected letter
        self._pending_gaze: tuple[float, float] | None = None
        self._deleted: tuple[tuple[float, float], str] | None = None  # (gaze, key) of the letter just backspaced

    # -- lifecycle ----------------------------------------------------------------------------------------------
    def reset(self) -> None:
        super().reset()
        self.bias = [0.0, 0.0]
        self._gaze_pts.clear()
        self._pending_gaze = None
        self._deleted = None

    def set_text(self, text: str) -> None:
        super().set_text(text)
        self._gaze_pts.clear()
        self._deleted = None

    def pick_candidate(self, index: int) -> bool:
        ok = super().pick_candidate(index)
        if ok:
            self._gaze_pts.clear()
            self._deleted = None
        return ok

    # -- bias handling ------------------------------------------------------------------------------------------
    def process_sample(self, gaze_x, gaze_y, timestamp, **kwargs) -> S0Event:
        """The learned bias is subtracted from the gaze BEFORE hover / dwell / selection see it."""
        if gaze_x is not None and gaze_y is not None:
            gaze_x, gaze_y = gaze_x - self.bias[0], gaze_y - self.bias[1]
        return super().process_sample(gaze_x, gaze_y, timestamp, **kwargs)

    def select_now(self, timestamp: float, sigma_scale: float = 1.0, window_s: float = 0.4, at_time: Optional[float] = None) -> S0Event:
        # The base class commits the HIGHLIGHTED key's centre (so the observation carries no gaze information); remember where the
        # eyes actually were so the bias estimator has something to learn from.
        hi = at_time if at_time is not None else timestamp
        pts = [(x, y) for t, x, y in self._recent if hi - window_s <= t <= hi]
        self._pending_gaze = (statistics.median(p[0] for p in pts), statistics.median(p[1] for p in pts)) if pts else None
        try:
            return super().select_now(timestamp, sigma_scale=sigma_scale, window_s=window_s, at_time=at_time)
        finally:
            self._pending_gaze = None

    def _center(self, key_id: str) -> tuple[float, float] | None:
        key = next((k for k in self.layout.keys if k.id == key_id), None)
        return key.center() if key else None

    def _learn(self, residual: tuple[float, float]) -> None:
        if abs(residual[0]) > self.max_residual or abs(residual[1]) > self.max_residual:
            return
        for i in (0, 1):
            self.bias[i] = max(-self.bias_clip, min(self.bias_clip, self.bias[i] + self.bias_rate * residual[i]))

    def _commit(self, mu_x, mu_y, std_x, std_y, sigma_scale, sigma_override=None) -> Optional[str]:
        gaze = self._pending_gaze or (mu_x, mu_y)
        before = len(self._chars)
        removed = self._chars[-1] if self._chars else None
        removed_gaze = self._gaze_pts[-1] if self._gaze_pts else None
        key = super()._commit(mu_x, mu_y, std_x, std_y, sigma_scale, sigma_override)

        if key is not None and len(key) == 1 and len(self._chars) == before + 1:  # a letter was added
            self._gaze_pts.append(gaze)
            if self._deleted is not None:
                (g_wrong, key_wrong), self._deleted = self._deleted, None
                center = self._center(key)
                if center is not None and key.lower() != key_wrong:  # BACKSPACE-and-retry: the retry key is the intended one
                    self._learn((g_wrong[0] - center[0], g_wrong[1] - center[1]))
        elif key == "BACKSPACE" and len(self._chars) == before - 1 and removed is not None and removed_gaze is not None:
            self._gaze_pts.pop()
            self._deleted = (removed_gaze, removed)
        elif key == "SPACE":
            self._gaze_pts.clear()
            self._deleted = None
        return key

    def _before_word_end(self, raw: str, committed: str) -> None:
        # LM correction of a real word: where the LM changed a letter, the gaze at that letter vs the letter it was changed to.
        lexicon = self.lexicon
        if (
            lexicon is None
            or committed == raw
            or len(committed) != len(raw)
            or len(committed) < self.min_word_len
            or committed not in lexicon.word_logp
            or len(self._gaze_pts) != len(raw)
        ):
            return
        for i, (c_raw, c_new) in enumerate(zip(raw, committed)):
            if c_raw != c_new:
                center = self._center(c_new.upper())
                if center is not None:
                    self._learn((self._gaze_pts[i][0] - center[0], self._gaze_pts[i][1] - center[1]))


LIVE_DECODERS = {"s0": S0LiveDecoder, "s1": S1LiveDecoder, "s2": S2LiveDecoder, "s3": S3LiveDecoder}
