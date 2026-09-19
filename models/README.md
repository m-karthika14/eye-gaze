# /models

Trained/pretrained model weights: the gaze estimation model and the local character-level LM.

- **Phase 1:** the gaze CNN's weights are fetched by `uniface` into its own cache directory (not here) — see `/vision`'s README. Not git-ignored/committed here since they're not stored in this folder.
- **Phase 4 (done):** `char_lm.json` — the trained local character-level LM (order-4 n-gram, `decoder/char_lm.py`). Small (~585KB) and deterministic, so it's **committed**, not git-ignored — Section 13 specifically wants this component reproducible/versioned, and committing it means nobody needs to retrain (`decoder/train_char_lm.py`, needs the NLTK Gutenberg corpus) just to run the decoder.
- **Phase 10:** fine-tuned/optimised variants from NVIDIA AI Factory runs.

Paths referenced from `config/model_paths.yaml`. Only large/binary formats (`.pt`, `.pth`, `.onnx`, `.bin`, `.safetensors`) are git-ignored here (see root `.gitignore`) — plain-text/small artifacts like `char_lm.json` are committed on a case-by-case basis.

**Status:** `char_lm.json` present and committed.
