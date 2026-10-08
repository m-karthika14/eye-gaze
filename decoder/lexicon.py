"""Word-level language prior for the live S4 decoder.

The character n-gram alone (decoder/char_lm.py) is a weak prior: when gaze is
about a key-width off, many letter strings are "plausible English", so beam
search over characters often lands on a wrong-but-English string ("hello" ->
"thele"). Real typing on noisy input (touch keyboards, gesture typing) fixes
this by constraining decoding to a lexicon: score every word of the right
length by

    log P(gaze | word) + lambda * log P(word)

where P(gaze | word) is the product over positions of the same Gaussian
per-key likelihood decoder/key_probability.py already produces. That is the
same noisy-channel formulation as the rest of the project, P(T|G) ~ P(G|T)P(T);
only the prior P(T) is a word unigram instead of a character n-gram. Words not
in the lexicon are still reachable via the character-LM fallback in
decoder/s4_live.py, at a fixed log-probability penalty.

The lexicon is built once from NLTK's Gutenberg corpus (already used to train
the char LM) plus config/vocabulary.txt (the AAC-oriented word list, boosted
so everyday communication words outrank literary ones):

    .venv/Scripts/python.exe -m decoder.lexicon

and committed at models/lexicon.json, like models/char_lm.json.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

DEFAULT_LEXICON_PATH = Path(__file__).resolve().parent.parent / "models" / "lexicon.json"
VOCABULARY_PATH = Path(__file__).resolve().parent.parent / "config" / "vocabulary.txt"
PHRASES_PATH = Path(__file__).resolve().parent.parent / "config" / "evaluation_phrases.txt"

MAX_WORDS = 8000
MAX_WORD_LEN = 14
AAC_MIN_PROB = 5e-4  # floor for words in config/vocabulary.txt -- roughly the frequency of a top-300 word
# Old-English function words that dominate Gutenberg but nobody types on a
# communication board; keeping them steals probability from real words
# ("the" vs "thee", "you" vs "thou").
ARCHAIC = {"thou", "thee", "thy", "thine", "hath", "doth", "hast", "dost", "shalt", "wilt", "unto", "yea", "ye", "lo", "nay", "ere", "oft", "wherefore", "whence", "hence", "thence", "behold"}
KEEP_SINGLE = {"i", "a"}


def _read_word_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.extend(re.findall(r"[a-z]+", line))
    return words


class Lexicon:
    def __init__(self, word_logp: dict[str, float]) -> None:
        self.word_logp = word_logp
        self._by_len: dict[int, list[tuple[str, float]]] = {}
        for word, lp in word_logp.items():
            self._by_len.setdefault(len(word), []).append((word, lp))
        # Penalty prior for strings outside the lexicon: rarer than the rarest known word.
        self.oov_logp = (min(word_logp.values()) if word_logp else -15.0) - 2.0

    @classmethod
    def load_default(cls) -> "Lexicon | None":
        """Returns None if models/lexicon.json hasn't been built -- callers
        fall back to the char-LM-only decoder rather than failing."""
        if not DEFAULT_LEXICON_PATH.exists():
            return None
        data = json.loads(DEFAULT_LEXICON_PATH.read_text(encoding="utf-8"))
        return cls({w: float(lp) for w, lp in data["word_logp"].items()})

    def logp(self, word: str) -> float:
        return self.word_logp.get(word, self.oov_logp)

    def rank(
        self,
        likelihoods: list[dict[str, float]],
        prior_weight: float = 1.0,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Best `top_k` (word, score) pairs among lexicon words of exactly
        len(likelihoods) letters, score = sum_i log L_i(c_i) + prior_weight * log P(word).
        `likelihoods[i]` maps each candidate char to P(gaze_i | char)."""
        n = len(likelihoods)
        if n == 0:
            return []
        log_l = [{c: math.log(max(p, 1e-12)) for c, p in lk.items()} for lk in likelihoods]
        floor = math.log(1e-12)
        scored: list[tuple[str, float]] = []
        for word, lp in self._by_len.get(n, ()):
            s = prior_weight * lp
            for i, c in enumerate(word):
                s += log_l[i].get(c, floor)
            scored.append((word, s))
        scored.sort(key=lambda item: -item[1])
        return scored[:top_k]


def build_lexicon(max_words: int = MAX_WORDS) -> dict[str, float]:
    from nltk.corpus import gutenberg  # lazy: only needed for the one-time build

    counts: Counter[str] = Counter()
    for w in gutenberg.words():
        w = w.lower()
        if re.fullmatch(r"[a-z]+", w) and len(w) <= MAX_WORD_LEN and w not in ARCHAIC:
            if len(w) > 1 or w in KEEP_SINGLE:
                counts[w] += 1
    top = dict(counts.most_common(max_words))
    total = sum(top.values())
    probs = {w: c / total for w, c in top.items()}

    for w in _read_word_file(VOCABULARY_PATH) + _read_word_file(PHRASES_PATH):
        if len(w) > 1 or w in KEEP_SINGLE:
            probs[w] = max(probs.get(w, 0.0), AAC_MIN_PROB)

    norm = sum(probs.values())
    return {w: math.log(p / norm) for w, p in sorted(probs.items())}


def main() -> None:
    word_logp = build_lexicon()
    DEFAULT_LEXICON_PATH.parent.mkdir(exist_ok=True)
    DEFAULT_LEXICON_PATH.write_text(
        json.dumps({"description": "Word unigram log-probs for decoder/lexicon.py", "word_logp": {w: round(lp, 4) for w, lp in word_logp.items()}}),
        encoding="utf-8",
    )
    print(f"wrote {len(word_logp)} words to {DEFAULT_LEXICON_PATH}")


if __name__ == "__main__":
    main()
