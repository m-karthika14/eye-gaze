"""Interactive S4 Live Decoder: Joint Gaze-Probability + Language Model
decoder for real-time interactive typing.

Unlike raw nearest-key selection, S4 maintains gaze uncertainty P(G|T) for
each letter selection and decodes the most probable word:
    P(T|G) ~ P(G|T) * P(T)

The word prior P(T) is a lexicon unigram (decoder/lexicon.py): every word with
as many letters as there are dwell selections is scored by its gaze likelihood
times its frequency. Strings outside the lexicon are still reachable through
character-n-gram beam search, at a fixed out-of-vocabulary penalty. On
simulated gaze one key-width off, this recovers ~80% of words vs ~25% for
character beam search alone (see README, "Live decoder"). This class is the
interactive PRODUCT path; the S1-vs-S4 research comparison in experiments/
does not use it, so that comparison is unchanged.

Features:
- RobustSpatialDwell with post-selection refractory cooldown (450ms) to
  prevent Midas Touch and accidental transit key selections while saccading.
- 600ms intentional dwell threshold.
- One-fire-per-fixation lock until gaze moves away.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import List, Optional, Tuple

from decoder.beam_search import beam_search_decode
from decoder.char_lm import CharLM
from decoder.lexicon import Lexicon
from decoder.key_probability import DEFAULT_MIN_SIGMA, key_probabilities
from decoder.nearest_key import KeyboardLayout, nearest_key
from experiments.gaze_sequence import GazeObservation, key_id_to_char
from experiments.s0 import S0Event


def gaze_to_char_likelihoods(
    obs: GazeObservation,
    layout: KeyboardLayout,
    min_sigma: float = 0.05,
) -> dict[str, float]:
    """Converts the observation's Gaussian into a per-character likelihood over
    lowercase alphabet 'a'-'z'. Drops BACKSPACE and SPACE so letter dwell
    evaluates against alphabet candidates."""
    key_probs = key_probabilities(
        obs.mu_x, obs.mu_y, obs.sigma_x, obs.sigma_y, layout, min_sigma=min_sigma
    )
    key_probs.pop("BACKSPACE", None)
    key_probs.pop("SPACE", None)
    total = sum(key_probs.values())
    if total <= 0.0:
        return {chr(c): 1.0 / 26.0 for c in range(ord("a"), ord("z") + 1)}
    return {key_id_to_char(key_id): p / total for key_id, p in key_probs.items()}


class RobustSpatialDwell:
    """Fixation/dwell detector based on spatial clustering with refractory cooldown.

    Built for noisy ~10-15 fps webcam gaze, where single frames routinely land
    far from where the eye really is (blinks, CNN jitter):

    - dwell_ms: 600ms of steady gaze fires a selection.
    - The fixation centre is the running mean of the samples so far, not the
      first sample (which is often still mid-saccade).
    - Outlier tolerance: a sample beyond `radius` of the centre does NOT reset
      the dwell. It is set aside, and only if gaze stays away for `grace_ms`
      is the fixation considered over and a new one started from those samples.
      Progress simply holds still during a brief excursion.
    - cooldown_ms: after a selection, no new dwell starts for this long, so the
      eye passing over neighbouring keys on its way to the next one can't fire.
    - one-fire-per-fixation: once fired, stays idle until gaze really moves on.
    """

    def __init__(
        self,
        dwell_ms: float = 600.0,
        radius: float = 0.12,
        cooldown_ms: float = 450.0,
        grace_ms: float = 250.0,
    ) -> None:
        self.dwell_ms = dwell_ms
        self.radius = radius
        self.cooldown_ms = cooldown_ms
        self.grace_ms = grace_ms
        self.reset()

    def reset(self) -> None:
        self.anchor_x: Optional[float] = None
        self.anchor_y: Optional[float] = None
        self.start_t: Optional[float] = None
        self.fired = False
        self.cooldown_until = 0.0
        self.samples: list[tuple[float, float]] = []
        self.away_since: Optional[float] = None
        self.away_samples: list[tuple[float, float]] = []

    def _start(self, x: float, y: float, t: float, samples: Optional[list[tuple[float, float]]] = None) -> None:
        self.samples = list(samples) if samples else [(x, y)]
        self.anchor_x = statistics.fmean(p[0] for p in self.samples)
        self.anchor_y = statistics.fmean(p[1] for p in self.samples)
        self.start_t = t
        self.fired = False
        self.away_since = None
        self.away_samples = []

    def update(
        self, x: Optional[float], y: Optional[float], t: float
    ) -> tuple[Optional[tuple[float, float, float, float]], float]:
        """Returns ((mu_x, mu_y, std_x, std_y), progress 0..1)."""
        if x is None or y is None:
            self.reset()
            return None, 0.0

        # During refractory cooldown, no new dwell starts
        if t < self.cooldown_until:
            return None, 0.0

        if self.anchor_x is None:
            self._start(x, y, t)
            return None, 0.0

        progress = 0.0 if self.fired else min(max((t - self.start_t) * 1000.0 / self.dwell_ms, 0.0), 1.0)

        if math.hypot(x - self.anchor_x, y - self.anchor_y) > self.radius:
            # Excursion: hold position, and only abandon the fixation if it persists.
            if self.away_since is None:
                self.away_since = t
            self.away_samples.append((x, y))
            if (t - self.away_since) * 1000.0 > self.grace_ms:
                self._start(x, y, self.away_since, self.away_samples)
                return None, 0.0
            return None, progress

        self.away_since = None
        self.away_samples = []
        if self.fired:
            return None, 0.0  # already fired for this fixation

        self.samples.append((x, y))
        self.anchor_x = statistics.fmean(p[0] for p in self.samples)
        self.anchor_y = statistics.fmean(p[1] for p in self.samples)
        elapsed_ms = (t - self.start_t) * 1000.0
        progress = min(max(elapsed_ms / self.dwell_ms, 0.0), 1.0)

        if elapsed_ms >= self.dwell_ms:
            self.fired = True
            self.cooldown_until = t + (self.cooldown_ms / 1000.0)
            # Median centre: robust to the odd bad frame inside the window.
            mu_x = statistics.median(p[0] for p in self.samples)
            mu_y = statistics.median(p[1] for p in self.samples)
            # Raw within-dwell scatter; the decoder turns it into a calibrated
            # sigma (scale + floor + ceiling) -- see S4LiveDecoder._sigma.
            std_x = statistics.pstdev(p[0] for p in self.samples) if len(self.samples) > 1 else 0.0
            std_y = statistics.pstdev(p[1] for p in self.samples) if len(self.samples) > 1 else 0.0
            return (mu_x, mu_y, std_x, std_y), 1.0

        return None, progress


class S4LiveDecoder:
    def __init__(
        self,
        layout: Optional[KeyboardLayout] = None,
        char_lm: Optional[CharLM] = None,
        dwell_ms: float = 800.0,
        beam_width: int = 12,
        lm_weight: float = 0.25,
        lexicon: Optional[Lexicon] = None,
        prior_weight: float = 1.0,
        sigma_floor: float = 0.08,
        sigma_ceiling: float = 0.12,
        max_candidates: int = 3,
    ) -> None:
        """sigma_floor/ceiling bound the per-dwell gaze sigma (normalized
        screen units, 0.10 = one key width). The floor is the honest
        "webcam gaze is about this imprecise" level -- without it a perfectly
        steady dwell would claim near-certainty about a position the sensor
        can't actually resolve; the ceiling stops an over-large calibrated
        sigma_scale from flattening the gaze term so far that the language
        prior alone picks the word."""
        self.layout = layout or KeyboardLayout.from_json()
        self.char_lm = char_lm or CharLM.load_default()
        self.lexicon = lexicon if lexicon is not None else Lexicon.load_default()
        self.prior_weight = prior_weight
        self.sigma_floor = sigma_floor
        self.sigma_ceiling = max(sigma_ceiling, sigma_floor)
        self.max_candidates = max_candidates
        self.candidates: List[str] = []
        self.dwell_ms = dwell_ms
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.dwell = RobustSpatialDwell(dwell_ms=dwell_ms)
        self._hover: Optional[str] = None
        self.dwell_enabled = True  # False = selection only via select_now()
        self._recent: deque = deque(maxlen=60)  # (t, x, y) of recent on-keyboard gaze
        self._hover_hist: deque = deque(maxlen=120)  # (t, highlighted key id or None)

        self.committed_text: str = ""
        self.active_word_obs: List[GazeObservation] = []
        self.active_word: str = ""

    def reset(self) -> None:
        self.dwell.reset()
        self.committed_text = ""
        self.active_word_obs.clear()
        self.active_word = ""
        self.candidates = []

    def set_text(self, text: str) -> None:
        self.committed_text = text
        self.active_word_obs.clear()
        self.active_word = ""
        self.candidates = []

    @property
    def current_text(self) -> str:
        return self.committed_text + self.active_word

    def _decode_active(self, **_unused: object) -> str:
        if not self.active_word_obs:
            self.candidates = []
            return ""
        likelihoods = [
            gaze_to_char_likelihoods(obs, self.layout, min_sigma=min(self.sigma_floor, obs.sigma_x, obs.sigma_y))
            for obs in self.active_word_obs
        ]
        context = self.committed_text[-3:] if self.committed_text else ""
        beam_word = beam_search_decode(
            likelihoods,
            self.char_lm,
            beam_width=self.beam_width,
            initial_prefix=context,
            lm_weight=self.lm_weight,
        )

        if self.lexicon is None:
            self.candidates = [beam_word] if beam_word else []
            return beam_word

        scored: List[Tuple[str, float]] = self.lexicon.rank(
            likelihoods, self.prior_weight, top_k=self.max_candidates
        )
        if beam_word and all(beam_word != w for w, _ in scored):
            # Out-of-vocabulary candidate: same gaze term, flat OOV prior, so
            # it only wins when the gaze evidence clearly beats every word.
            gaze_ll = sum(
                math.log(max(likelihoods[i].get(c, 0.0), 1e-12)) for i, c in enumerate(beam_word)
            )
            scored.append((beam_word, gaze_ll + self.prior_weight * self.lexicon.oov_logp))
        scored.sort(key=lambda item: -item[1])
        self.candidates = [w for w, _ in scored[: self.max_candidates]]
        return self.candidates[0] if self.candidates else beam_word

    def pick_candidate(self, index: int) -> bool:
        """Commit candidates[index] (plus a space) instead of the top-ranked
        word -- the manual override for when the decoder's first guess is
        wrong but the intended word is among the alternatives."""
        if not (0 <= index < len(self.candidates)):
            return False
        self.committed_text += self.candidates[index] + " "
        self.active_word_obs.clear()
        self.active_word = ""
        self.candidates = []
        return True

    def _sticky_hover(self, x: Optional[float], y: Optional[float], margin: float = 0.03) -> Optional[str]:
        """Highlighted key with hysteresis: it only changes once gaze is clearly
        (by `margin`) inside another key, so the highlight doesn't flicker
        between neighbours while the point sits on a border."""
        if x is None or y is None:
            self._hover = None
            return None
        if self._hover is not None:
            key = next((k for k in self.layout.keys if k.id == self._hover), None)
            if key and key.x - margin <= x <= key.x + key.width + margin and key.y - margin <= y <= key.y + key.height + margin:
                return self._hover
        self._hover = nearest_key(x, y, self.layout)
        return self._hover

    def _sigma(self, std: float, sigma_scale: float) -> float:
        return min(max(std * sigma_scale, self.sigma_floor), self.sigma_ceiling)

    def _commit(
        self,
        mu_x: float,
        mu_y: float,
        std_x: float,
        std_y: float,
        sigma_scale: float,
        sigma_override: Optional[float] = None,
    ) -> Optional[str]:
        """Apply one selection at (mu_x, mu_y) -- shared by dwell and by an
        explicit "select now" confirmation. Returns the key id selected."""
        sigma_x = sigma_override if sigma_override is not None else self._sigma(std_x, sigma_scale)
        sigma_y = sigma_override if sigma_override is not None else self._sigma(std_y, sigma_scale)
        target_key = nearest_key(mu_x, mu_y, self.layout)

        if target_key == "BACKSPACE":
            if self.active_word_obs:
                self.active_word_obs.pop()
                self.active_word = self._decode_active()
            elif self.committed_text:
                self.committed_text = self.committed_text[:-1]
                self.active_word = ""

        elif target_key == "SPACE":
            if self.active_word:
                self.committed_text = self.committed_text + self.active_word + " "
                self.active_word_obs.clear()
                self.active_word = ""
                self.candidates = []
            else:
                self.committed_text += " "

        elif target_key is not None:
            # Letter key
            self.active_word_obs.append(
                GazeObservation(mu_x=mu_x, mu_y=mu_y, sigma_x=sigma_x, sigma_y=sigma_y)
            )
            self.active_word = self._decode_active()

        return target_key

    def process_sample(
        self,
        gaze_x: Optional[float],
        gaze_y: Optional[float],
        timestamp: float,
        min_sigma: float = DEFAULT_MIN_SIGMA,
        sigma_scale: float = 1.0,
    ) -> S0Event:
        hovered_key = self._sticky_hover(gaze_x, gaze_y)
        self._hover_hist.append((timestamp, hovered_key))

        if gaze_x is not None and gaze_y is not None:
            self._recent.append((timestamp, gaze_x, gaze_y))
        else:
            self._recent.clear()

        selected_key: Optional[str] = None
        progress = 0.0
        if self.dwell_enabled:
            fired_tuple, progress = self.dwell.update(gaze_x, gaze_y, timestamp)
            if fired_tuple is not None:
                selected_key = self._commit(*fired_tuple, sigma_scale)
        else:
            self.dwell.reset()

        return S0Event(
            text=self.current_text,
            selected_key=selected_key,
            hovered_key=hovered_key,
            dwell_progress=progress,
        )

    def hold(self) -> S0Event:
        """No gaze update this frame (eyes closed: the gaze CNN output is garbage
        and must not move the highlight or feed the dwell)."""
        return S0Event(text=self.current_text, selected_key=None, hovered_key=self._hover, dwell_progress=0.0)

    def select_now(
        self,
        timestamp: float,
        sigma_scale: float = 1.0,
        window_s: float = 0.4,
        at_time: Optional[float] = None,
    ) -> S0Event:
        """Explicit confirmation ("click"): select the key the user has been
        looking at over the last `window_s` seconds. This is the cure for the
        Midas-touch problem of dwell selection -- looking at the keyboard,
        reading the text or hunting for the next key never selects anything;
        only the deliberate confirm does."""
        selected_key: Optional[str] = None
        hover_id = self._hover
        if at_time is not None:
            # Blink-select: the key that was highlighted just BEFORE the eyes closed.
            # (As the lids drop, the gaze estimate drifts for a frame or two.)
            for t, h in reversed(self._hover_hist):
                if t <= at_time and h is not None:
                    hover_id = h
                    break
        hovered = next((k for k in self.layout.keys if k.id == hover_id), None) if hover_id else None
        if hovered is not None:
            # The highlighted (blue) key IS the selection -- exactly what the user
            # sees and confirms. Pin the gaze observation to that key with a tight
            # sigma so the word decoder can't swap the letter for a different one.
            cx, cy = hovered.center()
            selected_key = self._commit(cx, cy, 0.0, 0.0, sigma_scale, sigma_override=0.02)
            return S0Event(
                text=self.current_text,
                selected_key=selected_key,
                hovered_key=self._hover,
                dwell_progress=0.0,
            )

        recent = [(x, y) for t, x, y in self._recent if timestamp - t <= window_s]
        if recent:
            xs = [p[0] for p in recent]
            ys = [p[1] for p in recent]
            std_x = statistics.pstdev(xs) if len(xs) > 1 else 0.0
            std_y = statistics.pstdev(ys) if len(ys) > 1 else 0.0
            selected_key = self._commit(statistics.median(xs), statistics.median(ys), std_x, std_y, sigma_scale)
        return S0Event(
            text=self.current_text,
            selected_key=selected_key,
            hovered_key=self._hover,
            dwell_progress=0.0,
        )
