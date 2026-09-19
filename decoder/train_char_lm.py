"""Trains the local character-level LM (Phase 4) on NLTK's Gutenberg corpus
(18 public-domain English texts, ~11.8M raw characters) and saves it to
/models/char_lm.json.

A classic-literature corpus is a starter/placeholder text source -- not
conversational register, which matters more once this feeds real decoding
in Phase 5 -- but it's freely available, large enough to train a solid
order-4 model, and enough to validate the LM mechanics now. Swappable later
without changing CharLM itself.

Run from the repo root:

    .venv/Scripts/python.exe -m decoder.train_char_lm
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import nltk

from decoder.char_lm import CharLM, clean_text

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "char_lm.json"

HELD_OUT_FRACTION = 0.05  # last 5% of each file held out for validation, never trained on


def load_corpus() -> tuple[str, str]:
    """Returns (train_text, held_out_text), both already cleaned."""
    try:
        from nltk.corpus import gutenberg
    except LookupError:
        nltk.download("gutenberg")
        from nltk.corpus import gutenberg

    train_parts: list[str] = []
    held_out_parts: list[str] = []
    for file_id in gutenberg.fileids():
        raw = gutenberg.raw(file_id)
        split_at = int(len(raw) * (1 - HELD_OUT_FRACTION))
        train_parts.append(raw[:split_at])
        held_out_parts.append(raw[split_at:])

    train_text = clean_text("\n".join(train_parts))
    held_out_text = clean_text("\n".join(held_out_parts))
    return train_text, held_out_text


def main() -> int:
    print("Loading corpus...")
    train_text, held_out_text = load_corpus()
    print(f"[OK] train: {len(train_text):,} chars, held-out: {len(held_out_text):,} chars")

    print("Training order-4 interpolated character LM...")
    lm = CharLM()
    lm.train(train_text)

    MODELS_DIR.mkdir(exist_ok=True)
    lm.save(MODEL_PATH)
    print(f"[OK] saved model to {MODEL_PATH}")

    # Perplexity on held-out text -- the standard held-out evaluation Phase
    # 4's success criterion asks for ("sensible probabilities... on held-out
    # text"). Lower is better; a well-trained char-level model on English
    # prose typically lands somewhere in the ballpark of 3-6 (i.e. ~1.5-2.5
    # effective "surprising" bits per character).
    sample = held_out_text[:20000]  # full held-out set would take a while at pure-Python speed
    log_prob_sum = lm.log_prob(sample)
    perplexity = math.exp(-log_prob_sum / len(sample))
    print(f"[OK] held-out perplexity (first {len(sample):,} chars): {perplexity:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
