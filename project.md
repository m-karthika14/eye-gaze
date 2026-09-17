Complete Project Roadmap
Calibration-Free Webcam-Based AI Assistive Communication System
Phased Implementation Plan with Groq API Integration
13 Phases · Research + Product Tracks · Groq-Powered Language Layer · Full Tech Stack · Timeline ·
Deliverables per Phase
Companion document to the Final System Design Report — this roadmap turns the frozen research design into a
buildable, phased product plan.
How to Read This Roadmap
Every phase below is tagged by type, so it is always clear whether a task exists to prove the research hypothesis or to
make the system usable as a product. Mixing these up is the most common way student AI projects lose focus.
Tag Meaning
RESEARCH Required to test H1 / H2 and produce the paper's results. Must be controlled, measurable, reproducible.
PRODUCT Makes the system usable/demoable but is not itself a research claim.
INFRA Underlying engineering/tooling that supports both tracks.
The one rule that governs every phase: don't build the product first and invent a paper around it. Build around one
testable question — when webcam gaze is uncertain, does keeping the full gaze probability distribution and
decoding it jointly with language context produce more accurate intended text than selecting one gaze key first and
applying language correction afterward? Everything else supports that question or makes the system a complete,
usable product.
Groq API Integration Strategy
Groq is used throughout the product layer for fast language generation, but the core research decoder (C2, the S1-vs-S4
experiment) keeps its own small local model. This split is deliberate — the reasoning is below.
Where Groq is used
Component Recommended Groq Model Why Groq Fits Here
Word / phrase completion suggestions
(product UI)
openai/gpt-oss-20b ~1000 tokens/sec on Groq's LPU hardware — fast
enough to suggest words in real time as the user
gazes
SpeakFaster-style phrase/abbreviation
expansion (optional advanced layer)
openai/gpt-oss-120b Higher quality for expanding short/abbreviated input
into full phrases when the extra ~2x latency is
acceptable
Text-to-speech output Canopy Labs Orpheus TTS
(preview, via Groq)
Keeps the whole language + speech stack on one
API instead of a separate TTS engine
Optional voice input (future extension) whisper-large-v3-turbo Low-cost, fast speech-to-text if a
caregiver/voice-command mode is added later
Where Groq is deliberately NOT used
The character-level language model prior inside the joint decoder (Stage 5 / C2, the component the S1-vs-S4 experiment
directly tests) stays a small, local, offline model (a compact char-level Transformer, GRU, or n-gram model — chosen in
Phase 4). Three reasons:
• Reproducibility: a controlled scientific comparison needs a deterministic, versioned model — a cloud API can change
behind the scenes between experiment runs
• Scale of experimentation: the noise-sensitivity and simulation experiments (Phase 9) run thousands of trials; hitting a
rate-limited cloud API that many times is slow, costly, and adds uncontrolled network latency as a confound
• Timing integrity: the S1-vs-S4 comparison is partly about interaction speed; a local model guarantees consistent,
network-independent latency so any measured difference comes from the decoding method, not network jitter
In short: Groq powers everything the user sees and hears; a small local model powers the one component being
scientifically measured.
Practical notes
• Groq's model lineup changes fairly often — confirm the current production model list and pricing at
console.groq.com/docs/models before finalising model IDs in code
• Cache repeated suggestion queries (e.g. common prefixes) to cut latency and API usage further
• Store the API key in environment variables / a .env file, never hard-coded (see Phase 0 config)
• Design a local fallback (a small cached n-gram word list) for the product-layer suggestions too, so the system keeps
working if the Groq API is briefly unreachable — this becomes part of Phase 8 (Robustness)
Tech Stack Summary
Layer Technology
Frontend React + Vite + TypeScript + Tailwind CSS — keyboard UI, gaze cursor, probability visualisation,
suggestions, text display, accessibility controls
Backend Python 3.11+, FastAPI, WebSockets — real-time gaze → UI streaming
Computer vision OpenCV, MediaPipe (face/landmark preprocessing), PyTorch (gaze model inference)
Gaze estimation Pretrained appearance-based gaze model (iTracker / full-face-CNN lineage) — not trained from
scratch
Uncertainty & decoder NumPy, SciPy — Gaussian gaze modelling, key-probability integration, noisy-channel decoding
Character-level LM (research) Small local model — compact Transformer / GRU / n-gram (chosen in Phase 4)
Word/phrase LM (product) Groq API — openai/gpt-oss-20b (fast) / openai/gpt-oss-120b (quality)
Text-to-speech Groq API — Orpheus TTS (preview)
Personalization Lightweight online bias-adaptation model, updated from natural interaction + read-back (EyeO-style)
Experiment tooling scikit-learn, statistical analysis scripts for S0–S4 comparisons
Training / large-scale simulation PyTorch on the university's NVIDIA AI Factory (DGX + Jetson) — see Phase 10
Software Architecture (Repository Layout)
/frontend
React + TypeScript — keyboard UI, gaze cursor, suggestions, text display
/backend
FastAPI + WebSocket server
/vision
Face detection, gaze estimation, uncertainty, temporal processing
/decoder
Key-probability mapping, character LM, joint decoder (noisy-channel)
/llm_services
Groq client — word/phrase suggestions, TTS, (optional) STT
/personalization
Online user model, bias adaptation
/experiments
S0, S1, S2, S3, S4 — runnable, reproducible experiment scripts
/data
Dataset processing, evaluation data, simulation outputs
/models
Gaze model weights, local character-LM weights
/config
Keyboard layout, model paths, decoder parameters, .env (API keys)
Communication Protocol (WebSocket)
Vision backend streams gaze results to the frontend continuously:
{ "gaze_x": 0.53, "gaze_y": 0.61, "confidence": 0.82, "selected_key": null }
Once a fixation/dwell event resolves, key probabilities are pushed:
{ "key_probabilities": { "A": 0.03, "S": 0.12, "D": 0.48, "F": 0.27, "G": 0.10 } }
Datasets
Dataset Role
MPIIGaze / MPIIFaceGaze Primary — realistic laptop-style gaze estimation (213,659 images, 15 users)
Gaze360 Robustness — broad head-pose/distance/indoor-outdoor variation; provides uncertainty
estimates
GazeCapture Generalisation/pretraining — large scale, though collected on phones/tablets not laptop
webcams
Custom collected dataset Webcam validation, keyboard-gaze, online adaptation, natural interaction — collected by the
team
Text corpus Training/evaluating the local character-level LM and word completion
Metrics
Primary:
• Character accuracy
• Word accuracy
• Sentence/phrase accuracy
Secondary:
• Words per minute (WPM)
• Character Error Rate (CER) / Word Error Rate (WER)
• Correction rate
• Number of selections
• Interaction time
• Decoding latency
• Rejected/uncertain selections (fallback triggers)
User experience (if a human study is run):
• NASA-TLX or similar workload measure
• System Usability Scale (SUS)
• Preference/usability questionnaire
• Whether generated/suggested text preserves the user's intended meaning and style (a concern the AAC literature
explicitly raises — see Valencia et al., 2023)
The 13 Phases
Each phase lists its goal, concrete tasks, where Groq fits (if at all), the deliverable, and how to know the phase is actually
done. Phases are meant to be completed in order — do not jump to S4 (Phase 5) before S0–S3 exist; the baselines are
what make the paper's comparison valid.
PHASE 0 — Specification Freeze INFRA
Goal Lock the architecture, research questions, S0–S4 design, datasets, metrics, and UI requirements before
writing implementation code.
Key Tasks • Finalise system architecture diagram and this roadmap
• Confirm research gap, hypotheses (H1, H2), and success criteria
• Confirm keyboard layout, target vocabulary, and evaluation phrase set
• Set up the repository skeleton (folders from the Software Architecture section)
• Create the .env config for Groq API key and model IDs
Groq / Tools Create and test a Groq API key; verify a basic chat-completion call works end to end.
Deliverable A written technical specification (this document + the System Design Report) and an empty, structured
repository.
Success Criteria No more feature brainstorming happens after this phase — only implementation against the frozen spec.
PHASE 1 — Webcam & Gaze Foundation RESEARCH
Goal Get a stable, real-time gaze point from an ordinary webcam.
Key Tasks • Integrate webcam capture (OpenCV) at 30 FPS
• Face detection + eye-region cropping (MediaPipe or equivalent)
• Load a pretrained appearance-based gaze model (iTracker / full-face-CNN lineage)
• Output raw (x, y) gaze point per frame
• Basic on-screen gaze-point visualisation for debugging
Deliverable A running pipeline: webcam → face/eye crop → gaze model → (x, y) point, visualised live.
Success Criteria A stable, visibly-tracking gaze point on screen as the user looks around, at interactive frame rates.
PHASE 2 — Keyboard Interaction & Baseline (S0) RESEARCH
Goal Build the first working gaze-to-text loop — the S0 baseline.
Key Tasks • Build the virtual keyboard UI (React) with defined key regions
• Map gaze point to nearest key
• Implement fixation/dwell detection (a first simple version)
• Build the text buffer and on-screen text output
• Wire gaze → key → text end to end
Deliverable S0 working: gaze → nearest key → text, with no language intelligence.
Success Criteria A user can type simple words by looking at keys, however imperfectly — this is the baseline everything else
is measured against.
PHASE 3 — Uncertainty Modelling & Calibration RESEARCH
Goal Turn a raw gaze point into a calibrated probability distribution over keyboard keys.
Key Tasks • Extend the gaze model output to (x, y) + uncertainty (σ)
• Model gaze as G ~ N(µ, Σ) around the predicted point
• Integrate the Gaussian over each key's screen region to get per-key probabilities
• Build fixation/dwell temporal aggregation (30fps samples → one selection-event distribution)
• Validate/calibrate the uncertainty (post-hoc calibration, e.g. temperature scaling, or empirical error
modelling against a validation set) — do not trust raw model confidence blindly
Deliverable A key-probability map per selection event (e.g. A: 0.03, S: 0.12, D: 0.48, F: 0.27, G: 0.10), with the
uncertainty calibrated and validated.
Success Criteria Predicted uncertainty is empirically checked to be well-calibrated (not systematically over- or
under-confident) before it is trusted by the decoder.
PHASE 4 — Language Model Layer RESEARCH
Goal Build the two separate language components the system needs, and integrate Groq for the product-facing
one.
Key Tasks • Choose and train/fine-tune a small local character-level LM (compact Transformer, GRU, or n-gram) for
the research decoder
• Validate the character LM independently: input prefix → next-character probabilities
• Build the Groq client wrapper (llm_services/) for word/phrase completion
• Design prompts for Groq word-completion (e.g. given typed prefix + recent context, return ranked word
candidates)
• Add response caching for common prefixes to cut Groq latency/cost
• Add a local fallback word list for when Groq is unreachable
Groq / Tools openai/gpt-oss-20b for low-latency word/phrase suggestions in the product UI. Keep this fully separate from
the local character-level LM used inside the research decoder.
Deliverable Two working, independently-testable language components: a local char-level LM and a Groq-backed
word/phrase suggester.
Success Criteria Both LMs return sensible probabilities/suggestions on held-out text, and the Groq call latency is measured
and acceptable for real-time suggestion display.
PHASE 5 — Joint Decoder (S4) & Ablation Ladder (S1–S3) RESEARCH
Goal Implement the actual research contribution: noisy-channel joint decoding, plus the intermediate ablation
systems needed to isolate its effect.
Key Tasks • Implement S1: hard-selected key → post-hoc local character-LM correction
• Implement S2: S1 + confidence-aware adaptive keyboard (key resizing)
• Implement S3: S2 + online user adaptation (stub is acceptable until Phase 7 is complete)
• Implement S4: P(T|G) ∝ P(G|T)·P(T) — joint decoding over the full key-probability distribution and the
character-level LM prior (e.g. via beam search)
• Ensure S1 and S4 can be run on identical recorded gaze sequences for the controlled comparison
Groq / Tools Not used here — S1–S4 all use the local character-level LM for a controlled, reproducible comparison.
Deliverable All five systems (S0–S4) runnable independently and against the same recorded gaze data.
Success Criteria S4 produces valid decoded text on test sequences, and a same-input S1-vs-S4 run produces two directly
comparable outputs.
PHASE 6 — Adaptive Keyboard & Suggestions UI PRODUCT
Goal Build the product-facing interface enhancements: key resizing and Groq-powered word suggestions.
Key Tasks • Implement live key resizing driven by language-model-predicted likely next characters
• Display word-completion suggestions (via Groq) as the user types
• Enlarge a suggestion visually when gaze probability is concentrated on it
• Ensure this layer can be fully disabled (flag) for the controlled S1-vs-S4 comparison
Groq / Tools openai/gpt-oss-20b powers the live suggestion strip (e.g. “HEL” → HELLO / HELP / HELD).
Deliverable A keyboard that visibly adapts to language predictions, with suggestions updating in real time.
Success Criteria Likely keys/suggestions are visibly larger and easier to select; the feature can be toggled off without breaking
S1/S4.
PHASE 7 — Online Personalization RESEARCH
Goal Learn user-specific gaze bias from natural interaction, without an explicit calibration screen.
Key Tasks • Use typed-text read-back moments (user looking at what they just typed) as implicit, known-target
calibration signal — the EyeO-style mechanism
• Estimate a running per-user gaze-bias correction from these observations
• Feed the corrected gaze distribution back into the key-probability mapping (Stage 3)
• Wire this into S3 (replacing the Phase-5 stub)
Deliverable A user model that updates continuously during normal use and measurably shifts key probabilities toward
the user's true gaze target over time.
Success Criteria Decoding accuracy for a given user improves over a session without the user ever seeing a calibration
screen.
PHASE 8 — Robustness & Fallback Handling PRODUCT
Goal Make the system behave safely under real-world failure conditions instead of guessing.
Key Tasks • Detect face lost / not detected → pause input
• Detect blink / eyes closed → pause input
• Detect gaze off-screen → pause input
• Detect low confidence across all keys → do not force a selection; wait for more evidence
• Add a Groq API fallback: if the suggestion/TTS call fails or times out, fall back to the local cached word list /
a basic local TTS, and retry in the background
Groq / Tools Fallback and timeout handling specifically for the Groq-dependent product features (suggestions, TTS).
Deliverable A fallback/low-confidence state machine covering gaze failures and API failures.
Success Criteria The system never emits a forced, garbage character under any of the tested failure conditions, and
continues to function (degraded) if Groq is unreachable.
PHASE 9 — Research Experiments RESEARCH
Goal Run the actual experiments that test H1 and H2.
Key Tasks • Experiment 1 (main, H1): S1 vs S4 on identical recorded gaze sequences, same vocabulary/context
• Experiment 2 (noise sensitivity): compare S1 vs S4 across low/medium/high/very-high simulated gaze
noise
• Experiment 3 (H2): decoding performance with vs. without online adaptation, over repeated interaction
• Experiment 4 (real-world validation): repeat key comparisons with real webcam + real participants, not just
simulation
• Log all primary and secondary metrics defined earlier for every run
Groq / Tools Not used in the measured decoder path (see Groq strategy above); may be used to help generate varied,
realistic evaluation sentences for the test set.
Deliverable A results dataset and statistical analysis for H1 and H2, across simulated and real conditions.
Success Criteria A clear, statistically supported answer to whether S4 outperforms S1, and whether the gap widens as gaze
noise increases (the specific claim this project makes).
PHASE 10 — NVIDIA AI Factory Utilization INFRA
Goal Use the university's DGX/Jetson facility for what a laptop can't do: scale.
Key Tasks • Run large-scale simulation (thousands of noisy-gaze trials) for Experiment 2
• Train/fine-tune the local character-level LM and any gaze-model adaptation
• Run hyperparameter sweeps: uncertainty calibration method, temporal window size, decoder weighting,
beam-search width
• Optimise inference (e.g. TensorRT) for the on-device/real-time gaze + decoder path
Deliverable Trained/tuned local models, and simulation results that would be impractical on a laptop.
Success Criteria The facility is a genuine part of the research pipeline (large-scale, reproducible experiments), not just a
mentioned resource.
PHASE 11 — Full Product Integration PRODUCT
Goal Combine every component into one coherent, accessible, demoable application.
Key Tasks • Integrate webcam + gaze + uncertainty + decoder + adaptive keyboard + suggestions + personalization +
Groq TTS into one running app
• Polish the accessibility of the UI itself (contrast, key sizing defaults, clear feedback states)
• Add a simple onboarding flow (no calibration screen, but a short “here's how it works” intro)
• Stress-test the end-to-end latency budget (gaze → decoded text → spoken output)
Groq / Tools Groq now powers suggestions and speech output live, in the full integrated product.
Deliverable A single running product: a person can sit down, use only their webcam and gaze, and communicate by
voice.
Success Criteria An end-to-end demo works reliably for a new user with no setup beyond sitting in front of the webcam.
PHASE 12 — Paper Writing RESEARCH
Goal Write up the research as a submittable paper.
Key Tasks • Structure: Introduction, Related Work, Research Gap, Proposed Method, Uncertainty Estimation, Key
Probability Mapping, Joint Decoder, Online Adaptation, Experimental Setup, Results, Ablation, Discussion,
Limitations, Conclusion
• Report all metrics from Phase 9, with statistical tests where relevant
• State novelty as “to the best of our knowledge…”, not an absolute claim
• Run one final targeted literature search immediately before submission to confirm the gap still holds
Deliverable A complete, submission-ready paper draft.
Success Criteria Every claim in the paper is backed by a specific result from Phase 9, and every design choice traces back to
a phase in this roadmap.
Suggested Timeline (14-Week Semester)
Weeks Phases Focus
1 0 Specification freeze, repo setup
2–3 1 Webcam + gaze foundation working end to end
4 2 S0 baseline: gaze → key → text
5–6 3 Uncertainty modelling + calibration validated
7 4 Local character LM + Groq word-suggestion integration
8–9 5 S1–S4 implemented and cross-validated on same data
10 6 Adaptive keyboard + suggestions UI (product layer)
11 7 Online personalization wired into S3
11–12 8 Robustness + Groq fallback handling
12–13 9, 10 Run experiments (incl. AI Factory large-scale runs)
13 11 Full product integration + demo polish
14 12 Paper writing and submission
Phases 6–8 (product layer) can run partly in parallel with Phases 9–10 (experiments) once S0–S4 are stable, since they
touch different parts of the codebase — adjust based on team size.
What NOT to Add
These would increase scope without strengthening the central research question:
• Seven “AI agents” just for presentation
• Fatigue detection, unless experiments show a genuine need
• A completely new gaze CNN trained from scratch
• Calling Groq (or any cloud LLM) inside the measured decoder path, per character, during controlled experiments
• Complex emotion detection
• Voice recognition, unless a concrete use case requires it
• Eye-blink communication as a separate research problem
• Gesture/path typing (EyeSwipe-style) — out of scope for this design
Final Master Flow
USER
↓
WEBCAM → GAZE ESTIMATION → POINT + UNCERTAINTY
↓
TEMPORAL AGGREGATION (fixation / dwell)
↓
KEY PROBABILITIES (A .05 S .10 D .50 ...)
↓ ↓
CHARACTER LM (local) ADAPTIVE KEYBOARD (product)
↓
JOINT DECODER — gaze + language
↓
INTENDED TEXT → WORD SUGGESTIONS (Groq) → TEXT-TO-SPEECH (Groq) → SPEECH
ONLINE USER MODEL learns gaze bias from natural interaction + read-back,
continuously feeding back into gaze estimation and decoding.
The rule that ties it all together: S4 is the research. Everything else — adaptive keyboard, Groq-powered suggestions,
personalization, TTS, UI — makes it a complete, usable system.