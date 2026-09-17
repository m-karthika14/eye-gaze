# /config

Frozen Phase-0 configuration: keyboard layout, target vocabulary / evaluation phrases, decoder parameters, model paths, and API keys.

| File | Purpose |
|---|---|
| `keyboard_layout.json` | Static QWERTY key regions (normalized 0–1 coordinates) used by S0–S4 and the adaptive keyboard |
| `evaluation_phrases.txt` | The evaluation phrase set used across Phase 9 experiments (H1/H2) |
| `vocabulary.txt` | Starter target vocabulary for early LM/keyboard testing (expanded with the full text corpus in Phase 4) |
| `decoder_params.yaml` | Placeholder defaults for fixation window, gaze uncertainty prior, beam width, etc. — tuned in Phase 3/5/10 |
| `model_paths.yaml` | Paths to gaze model and character-LM weights under `/models` |
| `.env.example` | Template for Groq API key and model IDs — copy to `.env` (git-ignored) and fill in |

`.env` (the real, filled-in file) is git-ignored. Never commit API keys.
