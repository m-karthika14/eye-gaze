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
    
    - dwell_ms: 600ms (intentional, deliberate dwell).
    - cooldown_ms: 450ms (prevents transit triggers while eye moves to next key).
    - radius: 0.10 (allows natural webcam scatter around the target).
    - one-fire-per-fixation: once a key fires, stays idle until eye moves away.
    """
    def __init__(
        self,
        dwell_ms: float = 600.0,
        radius: float = 0.10,
        cooldown_ms: float = 450.0,
    ) -> None:
        self.dwell_ms = dwell_ms
        self.radius = radius
        self.cooldown_ms = cooldown_ms
        self.anchor_x: Optional[float] = None
        self.anchor_y: Optional[float] = None
        self.start_t: Optional[float] = None
        self.fired: bool = False
        self.cooldown_until: float = 0.0
        self.samples: list[tuple[float, float]] = []

    def reset(self) -> None:
        self.anchor_x = None
        self.anchor_y = None
        self.start_t = None
        self.fired = False
        self.cooldown_until = 0.0
        self.samples.clear()

    def update(
        self, x: Optional[float], y: Optional[float], t: float
    ) -> tuple[Optional[tuple[float, float, float, float]], float]:
        """Returns ((mu_x, mu_y, sigma_x, sigma_y), progress 0..1)."""
        if x is None or y is None:
            self.reset()
            return None, 0.0

        # During refractory cooldown, no new dwell starts
        if t < self.cooldown_until:
            return None, 0.0

        if self.anchor_x is None:
            self.anchor_x = x
            self.anchor_y = y
            self.start_t = t
            self.fired = False
            self.samples = [(x, y)]
            return None, 0.0

        dist = math.hypot(x - self.anchor_x, y - self.anchor_y)
        if dist > self.radius:
            # Gaze moved away to a new location
            self.anchor_x = x
            self.anchor_y = y
            self.start_t = t
            self.fired = False
            self.samples = [(x, y)]
            return None, 0.0

        if self.fired:
            # Already fired for this fixation; don't re-fire until gaze moves away
            return None, 0.0

        self.samples.append((x, y))
        elapsed_ms = (t - self.start_t) * 1000.0
        progress = min(max(elapsed_ms / self.dwell_ms, 0.0), 1.0)

        if elapsed_ms >= self.dwell_ms and not self.fired:
            self.fired = True
            self.cooldown_until = t + (self.cooldown_ms / 1000.0)
            mu_x = statistics.fmean(p[0] for p in self.samples)
            mu_y = statistics.fmean(p[1] for p in self.samples)
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
        dwell_ms: float = 600.0,
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
        self.dwell = RobustSpatialDwell(dwell_ms=dwell_ms, radius=0.10, cooldown_ms=450.0)

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
            gaze_to_char_likelihoods(obs, self.layout, min_sigma=self.sigma_floor)
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

    def _sigma(self, std: float, sigma_scale: float) -> float:
        return min(max(std * sigma_scale, self.sigma_floor), self.sigma_ceiling)

    def process_sample(
        self,
        gaze_x: Optional[float],
        gaze_y: Optional[float],
        timestamp: float,
        min_sigma: float = DEFAULT_MIN_SIGMA,
        sigma_scale: float = 1.0,
    ) -> S0Event:
        hovered_key = (
            nearest_key(gaze_x, gaze_y, self.layout)
            if gaze_x is not None and gaze_y is not None
            else None
        )

        fired_tuple, progress = self.dwell.update(gaze_x, gaze_y, timestamp)
        selected_key: Optional[str] = None

        if fired_tuple is not None:
            mu_x, mu_y, std_x, std_y = fired_tuple
            sigma_x = self._sigma(std_x, sigma_scale)
            sigma_y = self._sigma(std_y, sigma_scale)
            target_key = nearest_key(mu_x, mu_y, self.layout)
            selected_key = target_key

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
                obs = GazeObservation(
                    mu_x=mu_x,
                    mu_y=mu_y,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                )
                self.active_word_obs.append(obs)
                self.active_word = self._decode_active()

        return S0Event(
            text=self.current_text,
            selected_key=selected_key,
            hovered_key=hovered_key,
            dwell_progress=progress,
        )
