"""Local character-level language model (Phase 4): the LM used inside the
research decoder (Stage 4/5, C2's P(T) term) to score candidate text. Kept
fully local/deterministic -- never Groq-backed -- for reproducibility; see
MASTER_PROJECT_DOCUMENT.md Section 13.

An order-4 (3-character context) interpolated n-gram model over a 27-symbol
alphabet (a-z + space), matching config/keyboard_layout.json's key set --
no punctuation or digits. Linear (Jelinek-Mercer-style) interpolation across
orders 4..1 with fixed weights handles unseen higher-order contexts by
falling back toward lower-order (and ultimately uniform) estimates, so every
symbol always gets a nonzero, well-defined probability.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ALPHABET = "abcdefghijklmnopqrstuvwxyz "
VOCAB_SIZE = len(ALPHABET)  # 27

ORDER = 4  # context length + 1 (i.e. 3 preceding characters -> next char)
# Interpolation weights for orders 4,3,2,1 (context lengths 3,2,1,0), highest
# order weighted most but never trusted exclusively. Sums to 1.0.
INTERPOLATION_WEIGHTS = (0.55, 0.25, 0.13, 0.07)
EPSILON = 0.05  # additive smoothing constant within each order


def clean_text(raw: str) -> str:
    """Lowercases and strips to the same alphabet the keyboard produces:
    a-z and space only, whitespace collapsed."""
    out = []
    prev_space = True  # collapse leading whitespace too
    for ch in raw.lower():
        if ch.isalpha() and ch in ALPHABET:
            out.append(ch)
            prev_space = False
        elif ch.isspace():
            if not prev_space:
                out.append(" ")
                prev_space = True
        # anything else (punctuation, digits) is dropped, not replaced with
        # a space, so "don't" -> "dont" rather than "don t"
    return "".join(out).strip()


class CharLM:
    def __init__(self) -> None:
        # counts[k] maps a context of length k -> Counter of next-char counts
        self.counts: dict[int, dict[str, Counter]] = {k: defaultdict(Counter) for k in range(ORDER)}

    def train(self, text: str) -> None:
        padded = " " * (ORDER - 1) + text
        for i in range(len(text)):
            next_char = text[i]
            for k in range(ORDER):
                context = padded[i + (ORDER - 1) - k : i + (ORDER - 1)]
                self.counts[k][context][next_char] += 1

    def _order_prob(self, k: int, context: str, char: str) -> float:
        context = context[len(context) - k :] if k > 0 else ""
        counter = self.counts[k].get(context)
        total = sum(counter.values()) if counter else 0
        count_c = counter[char] if counter else 0
        return (count_c + EPSILON) / (total + EPSILON * VOCAB_SIZE)

    def next_char_probs(self, prefix: str) -> dict[str, float]:
        """Returns a normalized {char: probability} over ALPHABET, given the
        (already-cleaned) prefix typed so far."""
        context = prefix[-(ORDER - 1) :] if prefix else ""
        context = context.rjust(ORDER - 1, " ")

        probs: dict[str, float] = {}
        for char in ALPHABET:
            p = 0.0
            for k, weight in enumerate(reversed(INTERPOLATION_WEIGHTS)):
                # reversed() so k=0 (order-1/unigram) pairs with the last
                # weight and k=ORDER-1 (order-4) with the first -- see
                # INTERPOLATION_WEIGHTS' comment (highest order listed first).
                p += weight * self._order_prob(k, context, char)
            probs[char] = p

        total = sum(probs.values())
        return {c: p / total for c, p in probs.items()}

    def log_prob(self, text: str) -> float:
        """Sum of log P(char | preceding chars) over the (cleaned) text --
        the P(T) term Phase 5's noisy-channel decoder needs."""
        text = clean_text(text)
        total = 0.0
        for i in range(len(text)):
            probs = self.next_char_probs(text[:i])
            total += math.log(probs[text[i]])
        return total

    def save(self, path: Path) -> None:
        serializable = {
            str(k): {ctx: dict(counter) for ctx, counter in ctx_counts.items()}
            for k, ctx_counts in self.counts.items()
        }
        path.write_text(json.dumps(serializable), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharLM":
        data = json.loads(path.read_text(encoding="utf-8"))
        lm = cls()
        for k_str, ctx_counts in data.items():
            k = int(k_str)
            for ctx, counter in ctx_counts.items():
                lm.counts[k][ctx] = Counter(counter)
        return lm
