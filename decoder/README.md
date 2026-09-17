# /decoder

Key-probability mapping, the local character-level LM, and the joint noisy-channel decoder — the core research contribution (C1/C2).

- **Phase 3:** integrate the gaze Gaussian over each key's screen region → per-key probability distribution.
- **Phase 4:** local character-level LM (compact Transformer / GRU / n-gram) — deliberately not Groq-backed; see `MASTER_PROJECT_DOCUMENT.md` Section 13 for why.
- **Phase 5:** S0–S4 ablation ladder, culminating in S4's joint decoding `P(T|G) ∝ P(G|T)·P(T)` (beam search over key-probability distribution × LM prior).

**Status:** not yet implemented. This is the component the S1-vs-S4 experiment (H1) directly measures — keep it fully local and deterministic, no cloud LLM calls in this path.
