# /decoder

Key-probability mapping, the local character-level LM, and the joint noisy-channel decoder — the core research contribution (C1/C2).

- **Phase 2 (done):** `nearest_key.py` — hard nearest-key lookup against `config/keyboard_layout.json` (S0's degenerate, single-choice version of key-probability mapping). `dwell.py` — first, simple fixation/dwell detector (fixed threshold, no temporal aggregation yet). `text_buffer.py` — shared key-sequence → text buffer used by every S-system.
- **Phase 3:** replace `nearest_key`'s hard choice with a real Gaussian integrated over each key's region → full per-key probability distribution; replace `dwell.py`'s fixed threshold with proper temporal aggregation over a fixation window.
- **Phase 4:** local character-level LM (compact Transformer / GRU / n-gram) — deliberately not Groq-backed; see `MASTER_PROJECT_DOCUMENT.md` Section 13 for why.
- **Phase 5:** S0–S4 ablation ladder, culminating in S4's joint decoding `P(T|G) ∝ P(G|T)·P(T)` (beam search over key-probability distribution × LM prior).

This is the component the S1-vs-S4 experiment (H1) directly measures — keep it fully local and deterministic, no cloud LLM calls in this path.
