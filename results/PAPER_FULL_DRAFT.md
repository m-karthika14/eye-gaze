# Paper draft (paste-ready): Abstract, Introduction, Literature Survey, Methodology (A-N), Algorithms

Everything in Section III and the Algorithms describes what is implemented in this repository (file names in brackets). Numbers come from
`results/` (perception.json, tables.md) and are quoted again in `results/PERCEPTION_RESULTS.md` (Sections 4.1 and 4.2). Before submission:
check every reference against the publisher page (details were written from memory), and keep the claims limited to the evidence listed in
the Limitations paragraph; no draft can guarantee acceptance, but a paper whose claims match its evidence is the one reviewers accept.

---

# Calibration-Free Webcam Eye-Gaze Typing by Joint Gaze-Distribution and Language-Model Decoding

## Abstract

Eye-gaze typing could give people who cannot use their hands an inexpensive communication aid with an ordinary laptop webcam, but two
obstacles stand in the way: webcam gaze is only accurate to about one key width, and the mapping from eye to screen is fitted in a
calibration that has to be repeated whenever the seating or head position changes. We present a webcam typing system that avoids a
per-session calibration step. We use the term operationally: the mapping is not refitted for the sitting (it comes from earlier
sittings), the seat is at a fixed position, and the user may give one re-centring glance at the middle of the keyboard. To make typing
usable with a mapping that is not refitted, the decoder keeps the gaze uncertainty instead of selecting one key: each fixation becomes a
two-dimensional Gaussian over the keyboard, integrated over every key, and decoded jointly with a word lexicon and a character
language model by noisy-channel search (S4), while an online learner removes slow offsets from the user's own corrections (S3, S4 with
adaptation). We compare S4 with nearest-key selection (S0), hard selection with language-model correction (S1), a stronger
adjacency-aware hard-selection baseline (S1+), and an adaptive-keyboard variant (S2), each with and without recalibration, on gaze errors
measured from seven real calibration runs of two participants, in a paired simulation with equally tuned baselines. A mapping that was
not refitted raised the character error rate of every system three- to four-fold (S4: 0.081 to 0.258; S1: 0.201 to 0.649;
S1+: 0.140 to 0.540), but S4 without recalibration (0.258) approached nearest-key selection with a fresh calibration (0.226) and was
52-60% lower than the hard-selection baselines without recalibration. Under simulated posture drift, online bias learning kept S4's
last-ten-phrase error at 0.13 where the static decoder reached 0.53. Evidence for users who have never calibrated at all is limited to
one held-out person; a controlled user study is the necessary next step.

## Index Terms

Calibration-free eye tracking, gaze typing, webcam, assistive communication (AAC), noisy-channel decoding, language models, uncertainty
estimation, online adaptation, human-computer interaction.

---

## I. INTRODUCTION

### A. Background

People with severe motor impairments, such as amyotrophic lateral sclerosis, spinal cord injury or locked-in syndrome, often keep control of
their eyes when they lose control of their hands and voice. Eye-gaze typing on an on-screen keyboard is therefore a main route to
communication in augmentative and alternative communication (AAC) [Majaranta and Raiha 2002]. Commercial systems rely on dedicated
infrared eye trackers, which are accurate (below about one degree) but costly, and need setup that many users cannot afford. A laptop
webcam is present in almost every household, and appearance-based gaze estimation from ordinary video has improved quickly
[Zhang et al. 2015; Krafka et al. 2016; Zhang et al. 2017]. Off-the-shelf webcam estimates remain much coarser, with reported errors of
3-6 degrees for calibrated and uncalibrated use [Garcia de la Santa Ramos et al. 2023], which on a laptop screen corresponds to one to
four keys of a standard keyboard.

### B. Problem Statement

Three problems separate webcam gaze from usable typing. (1) **Accuracy.** The estimated gaze point is noisy and biased, so the key under
the estimate is often not the key the user means. The usual remedy is to select one key per fixation (a hard decision) and correct the
resulting text with a language model afterwards; this throws away how close the neighbouring keys were, which is exactly what a language
model could use to tell neighbouring letters apart. (2) **Calibration.** Mappings fitted in one sitting lose a large share of their
accuracy in the next, because head position, seating and lighting change; requiring a calibration at every use is a burden for the
people who most need the system. Removing it, so that a user can sit down and type, is an open problem even for commercial
webcam trackers [Garcia de la Santa Ramos et al. 2023]. (3) **Selection.** Dwell-based selection suffers from the Midas touch and is slow on noisy signals;
users who cannot use any switch need robust alternatives such as blinks or mouth opening that do not disturb the gaze signal.

### C. Proposed Approach

We target typing without a per-session calibration. We define the term precisely, because it is easy to over-claim: a system is
*calibration-free* here when (i) the gaze mapping is not refitted for the current sitting (it is taken from earlier sittings, or, in the
strongest case, from other people), (ii) the seat is at a roughly fixed position relative to the screen, and (iii) the user gives at
most one re-centring glance at the middle of the keyboard. We then ask what makes typing possible in that regime. Our answer is to
keep the uncertainty: each fixation is modelled as a Gaussian with the measured scatter, the probability mass of every key is computed
in closed form, and a noisy-channel decoder searches for the text that best explains the whole sequence of fixations jointly with a word
lexicon and a character language model. A mapping that is not refitted is mainly off by a constant offset, so an online learner
estimates that offset from the user's own corrections, without any calibration screen. The pipeline uses head-normalised eye-landmark
features and a ridge-regression mapping, a gaze-CNN mapping as a second back end, adaptive smoothing, self-calibrating blink and
mouth-opening selection, and an adaptive keyboard. Five decoders (S0-S4) share the same camera, mapping and selection method, so that
differences can be attributed to the decoding alone; every baseline receives the same tuning effort as the proposed method; and every
decoder is evaluated both with a freshly calibrated mapping and with a mapping that is not refitted, so that the cost of removing
calibration, and how much of it each decoder recovers, can be read directly from the results.

### D. Contributions

1. An operational definition of calibration-free webcam gaze typing (no per-session refit, fixed seat, optional one-point re-centre) and an
   evaluation that reports it next to the calibrated case, for every decoder, so that the claim can be checked.
2. S4, a joint decoder that integrates a Gaussian gaze estimate over each key and combines it with a word lexicon and a character n-gram
   language model by noisy-channel search, which keeps typing usable when the mapping is not refitted; it is compared with a strong
   adjacency-aware hard-selection baseline (S1+) as well as with nearest-key (S0) and hard-selection-plus-correction (S1) baselines.
3. An online bias learner (S3, S4 with adaptation) that removes slow offsets from language-model corrections and backspace-retry, with no
   calibration screen, evaluated under simulated posture drift.
4. A complete, open webcam gaze-typing system (face and landmark detection, head-normalised eye-landmark mapping with a CNN alternative,
   smoothing, blink/mouth/switch/dwell selection, decoders S0-S4, and a web dashboard that switches between them) running at about 20
   frames per second on a laptop CPU.
5. A calibration procedure that trains on individual frames and chooses the feature subset and ridge strength by leave-one-trial-out error
   over held-out frames, with within-sitting and across-sitting accuracy reported for seven runs of two participants.
6. A fair, reproducible protocol (paired trace-driven simulation on measured errors, tuning on disjoint phrases and seeds for every
   system, bootstrap intervals and paired tests, and an explicit statement of what the evidence does and does not show).

---

## II. LITERATURE SURVEY

We surveyed twelve core papers from 2016 to 2025 in four sets: webcam-based gaze estimation, calibration-free and adaptive gaze
estimation, gaze-based text entry, and language-model-assisted augmentative and alternative communication (AAC). Each set establishes one
ingredient of a calibration-light webcam typing system and leaves a different part open. We add the classical work on which the sets rest
and the tools our implementation uses.

**A. Webcam-based gaze estimation.** Appearance-based methods regress gaze from face or eye images [Zhang et al. 2015]. Krafka et al.
trained a multi-region CNN (iTracker) on GazeCapture, 2.4 million frames from 1,474 people, taking both eyes, the face and a face-grid
mask as inputs, and reached 1.71 cm (phone) and 2.53 cm (tablet) error without calibration [Krafka et al. 2016]. The error is still
centimetre-level, the target devices are phones and tablets rather than laptops, the output is a single point without uncertainty, and no
typing task is evaluated. Zhang et al. showed that a full-face input with learned spatial weights, which suppress uninformative regions,
reaches 4.8 degrees person-independent 3D error on MPIIGaze and improves on iTracker by 14.3% (MPIIGaze) and 27.7% (EYEDIAP)
[Zhang et al. 2017]; it needs a calibrated camera and a geometric normalisation, and it does not address a communication task. Gaze360
added a temporal model that predicts a per-sample uncertainty next to the 3D gaze direction, trained on 238 people in indoor and outdoor
scenes [Kellnhofer et al. 2019]; the uncertainty is reported but not consumed by any downstream decision or interaction. Browser-based
tracking such as WebGazer learns a regression from eye features to the screen from the user's interactions [Papoutsaki et al. 2016], and
lightweight open-source tools expose iris and pupil positions [Lame, GazeTracking] or regress head-normalised landmark features to the
screen with a ridge regression after a calibration [EyeTrax, MIT licence]; a gaze-typing project built on such a pipeline exists
[Gaze_typing, MIT licence]. We adopt the head-normalised landmark features and the ridge mapping, add frame-level training with a
noise-aware selection criterion and a CNN alternative, and evaluate the whole system with a decoder that uses the gaze uncertainty.
Together these works establish that an ordinary camera can estimate gaze and that its uncertainty is measurable. They leave open how the
estimate is used for typing and how much accuracy survives when the mapping to the screen is not fitted for the user.

**B. Calibration and calibration-free or adaptive gaze estimation.** Calibration fits a generic gaze model to the individual user
[Blignaut et al. 2014]. Removing it is an open problem: saliency maps and interaction events have been used for implicit calibration
[Sugano et al. 2010; Huang et al. 2016], and recent work evaluates calibration-free webcam tracking for user modelling [Hutt and D'Mello
2022]. A 2023 doctoral-consortium paper reports about 3 degrees of error with calibration and 6-8 degrees without, on a commercial webcam
tracker, and names head-movement robustness as the main obstacle [Garcia de la Santa Ramos et al. 2023]. FAZE disentangles appearance,
head pose and gaze in a latent space and uses meta-learning to adapt to a person from as few as three labelled samples, reaching 3.18
degrees [Park et al. 2019]; the adaptation still needs explicit labelled samples, happens offline, and is not evaluated on text entry.
Chen and Shi decompose the predicted gaze into a subject-independent term and a subject-dependent bias, motivated by inter-subject bias
exceeding intra-subject variance; their calibration-free mode sets the bias to zero and improves on the prior state of the art by up to
10%, and a one-point calibration of about nine samples reduces the error by up to 35.6% [Chen and Shi 2020]. EyeO exploits the difference
between gaze as input (typing, where the user compensates for a miscalibration) and gaze as output (reading the typed character back,
where the user does not) to correct the calibration offset silently during typing, evaluated with 20 participants and stakeholders from
the ALS community on a dedicated eye tracker [Saran et al. 2025]; it corrects a point estimate and does not use a language model. EyeO is
the closest precedent to removing the explicit calibration phase. Our online bias learner has the same aim but a different cue: it
estimates the offset from the user's own corrections (language-model corrections and backspace-retry) and works on a webcam. The
decomposition of Chen and Shi motivates treating the residual error of an unrefitted mapping as mostly a constant offset, and our
measurements agree in direction: a mapping fitted in one sitting loses most of its accuracy in the next, and mostly through a constant
offset that a one-point re-centre or an online bias learner can remove. None of these papers removes calibration altogether: FAZE and Chen
and Shi need samples or discard the bias, and EyeO needs a dedicated tracker.

**C. Gaze typing and gaze-based text entry.** Eye typing has been studied for decades; the review by Majaranta and Raiha covers the
systems and design issues of dwell-based typing [Majaranta and Raiha 2002]. Dwell selection suffers from the Midas touch problem, in which
everything the user looks at is selected [Jacob 1990], and from the trade-off between dwell time and error; adaptive and cascading dwell
times reduce it [Mott et al. 2017]. Dwell-free approaches decode text from the gaze path against a language model and show that large
speed gains are possible with accurate trackers [Kristensson and Vertanen 2012]; Dasher replaces the keyboard with continuous
language-model-driven zooming [Ward et al. 2000]. EyeSwipe lets the user trace a word with the gaze path instead of dwelling on each
letter, matched against a dictionary, and reaches about 11.7 words per minute after practice, but it needs a dedicated tracker,
calibration and a reverse-crossing gesture that takes practice [Kurauchi et al. 2016]. SliceType uses a language model to shrink unlikely
keys and enlarge likely ones, with inline word predictions, and was faster and preferred over Dasher and GazeTalk in a study with 37
novices; it assumes a reasonably accurate calibrated signal, and the adaptation is in the interface only [Benligiray et al. 2018].
Abdrabou et al. type without calibration by correlating the gaze with the motion of moving letters (smooth pursuit) and add
corpus-trained text suggestions, reaching 3.41 words per minute with 26 participants on a dedicated eye tracker [Abdrabou et al. 2019];
this is the closest precedent that combines calibration-free typing with language-based suggestions, but it needs continuously moving
stimuli, a different interaction from pointing at a static keyboard. Soft keyboards treat the touch point as a noisy observation and
combine a spatial model with a language model [Goodman et al. 2002], and sentence-based decoders for touchscreen input follow the same
noisy-channel formulation [Vertanen et al. 2015]. These works show that decoding strategy and language-model-driven key adaptation matter
as much as raw accuracy. Our S4 decoder applies the noisy-channel formulation to gaze: each fixation is a Gaussian whose width is measured
and calibrated, decoded with a word lexicon and a character n-gram model, and it keeps discrete key selection, which suits users with
limited control. We compare it with a hard-selection baseline that is given the keyboard geometry and the calibrated noise (S1+), which
isolates the benefit of keeping the continuous gaze distribution.

**D. Language-model-assisted AAC.** SpeakFaster expands highly abbreviated gaze-typed input into full phrases with a fine-tuned language
model, with 57% fewer motor actions in simulation and 29-60% higher text-entry rate for two ALS eye-gaze users [Cai et al. 2024]. A
character-level adaptation of language models for AAC predicts the next character from limited input per keystroke, trained on simulated
typing because no real AAC-user data was available [arXiv:2501.10582, 2025]. Valencia et al. studied live language-model suggestions with
AAC users and found that they can speed communication but that users worried the suggestions did not reflect their own voice
[Valencia et al. 2023]. All three assume an already working, accurate gaze or keyboard input: they use already-selected characters rather
than a noisy per-key signal. The study of Valencia et al. is also a caution for our evaluation, which should judge whether the decoded text
keeps the intended meaning and not only the error rate.

**E. Signal processing and gestures.** The One Euro filter smooths noisy interactive signals with a cutoff that adapts to speed [Casiez et
al. 2012]; blinks can be detected from an eye aspect ratio computed from facial landmarks [Soukupova and Cech 2016]. Face detection and
landmarks come from RetinaFace [Deng et al. 2020] and a MediaPipe-style mesh with iris points [Kartynnik et al. 2019]; the gaze CNN uses a
MobileOne backbone [Vasu et al. 2023]. Ridge regression is the classical regulariser for the ill-posed fit [Hoerl and Kennard 1970].

**Gap.** Within the work surveyed, gaze estimators report a point estimate (or an uncertainty that nothing consumes), adaptive and
calibration-light methods still need labelled samples, discard the personal bias or require a dedicated tracker, text-entry methods assume
an accurate signal or moving stimuli, and language-model AAC starts after the gaze has been resolved into characters. Prior webcam systems
either require a calibration, report accuracy only, or select one key and correct afterwards. We did not find among them a system that (i)
states what calibration-free means for typing and reports it next to the calibrated case for the same decoders, (ii) keeps the
per-fixation Gaussian over keys and decodes it jointly with a word lexicon and a character language model on an ordinary webcam, (iii)
compares it with an equally tuned, geometry-aware hard-selection baseline on errors measured from real calibration runs, (iv) learns the
residual offset online from the user's own corrections, and (v) reports within-sitting and across-sitting accuracy of the mapping together
with the decoder results. We do not claim to eliminate calibration. In this paper "calibration-free" means that the mapping is not refitted
for the sitting, the seat is fixed, and the user gives at most one re-centring glance (Section I-C), and the evidence for people who have
never calibrated is limited (see Limitations). Our uncertainty is the measured scatter of each fixation, not a network-predicted value as
in Gaze360.

---

## III. METHODOLOGY

### A. System Overview

The system has five stages (Fig. 1): (1) video acquisition, (2) face detection, landmarks and gaze angles, (3) gaze-to-keyboard mapping
and smoothing, (4) selection by dwell, blink, mouth opening or a switch, and (5) decoding of the selected fixations into text by one of
five decoders (S0-S4). A calibration mode fits the mapping in stage 3. The backend is a Python FastAPI WebSocket server
([backend/main.py]); the frontend is a React/Vite dashboard with a full-screen on-screen keyboard shared by calibration and live
typing, a switch between S0-S4, and a study bar for phrase prompts and logging. Gaze positions are expressed in normalised keyboard
coordinates (x to the right, y downwards, both in [0,1]); the 28-key QWERTY layout (26 letters, SPACE, BACKSPACE) lies on a 10 x 4 grid
in which a key is 0.10 wide and 0.25 tall ([config/keyboard_layout.json]).

*Operating regimes.* Throughout, a decoder is evaluated in two regimes: **calibrated** (the mapping is fitted on the user in the current
sitting, Section F) and **calibration-free** (the mapping is fitted on other sittings, or on other people, and not refitted; the
only personal input allowed is the one-point re-centre of Section H, plus the online bias learner of Section M for S3 and S4). The first is
the upper bound, the second is the claim of this paper, and the difference is the cost of removing calibration.

Fig. 1 (to draw): Camera -> detect (RetinaFace) -> landmarks (FaceMesh) + gaze CNN -> features -> mapping -> filter -> fixation/dwell ->
key probabilities -> decoder (S0-S4) -> text.

### B. Video Acquisition

Frames come from a threaded grabber that always keeps only the newest frame, so that a slow downstream step never works on stale video
([vision/webcam.py]). The camera runs at 640 x 480. ONNX Runtime is limited to two threads with thread spinning disabled; the default
configuration used about 290 ms of CPU per frame and gave 3-7 frames per second, against about 20 frames per second with the cap
([vision/face_gaze.py]). Face detection runs on every fourth frame at 320 pixels, and the last face box is reused in between.

### C. Face Detection, Landmarks and Gaze Angles

The face is found with RetinaFace; a 478-point face mesh with iris landmarks (FaceMesh V2, the MediaPipe layout) gives eye, lip and head
landmarks; a MobileOne-S0 gaze network (MobileGaze) gives the gaze pitch theta and yaw psi in camera coordinates from the face crop. All
three are loaded through the uniface library. The landmarks feed the eye features (Section D), the blink and mouth detectors
(Section I), and the head-pose terms; the gaze angles feed the CNN mapping (Section E). During eye closure the landmark track is
continued, so that the gaze can resume after a blink without a new detection ([vision/face_gaze.py]).

### D. Head-Normalised Eye Features

Let p_i in R^3 be the landmarks in image coordinates (x, y as a fraction of the frame size, z on the scale of x). A head-aligned frame is
built from the outer eye corners p_33 and p_263 and the top of the head p_10:

  e = (p_33 + p_263) / 2   (origin)
  x_hat = (p_263 - p_33) / ||p_263 - p_33||
  y' = (p_10 - e) / ||p_10 - e||,   y_hat = (y' - (y' . x_hat) x_hat) / ||y' - (y' . x_hat) x_hat||
  z_hat = x_hat x y_hat   (cross product),   R = [x_hat  y_hat  z_hat]

Every landmark is rotated into this frame and divided by the inter-corner distance d:

  q_i = R^T (p_i - e) / d,   d = ||R^T (p_263 - e) - R^T (p_33 - e)||

so that translation, in-plane rotation and distance are largely removed and the remaining variation is mainly the eye in its socket. The
head angles are read from R:

  yaw = atan2(R_10, R_00),   pitch = atan2(-R_20, sqrt(R_21^2 + R_22^2)),   roll = atan2(R_21, R_22).

The feature vector f in R^486 stacks (x, y, z) of 161 landmarks (76 per eye: lids, rings, brows and the iris; and nine face-outline
anchors) and the three angles ([vision/eye_features.py], after EyeTrax). Three column subsets are considered by calibration: iris_core
(iris centre and edge points plus the eye corners of both eyes and the three angles, 45 dimensions), eye_region (all eye-contour and iris
points, 399 dimensions) and full (486 dimensions).

### E. Gaze-CNN Affine Mapping (second back end)

The gaze network gives absolute angles (theta, psi), with no session-specific zero point. The mapping from angles to a keyboard position
is affine, optionally with the head position h = (c_x, c_y, w) (face centre and width as fractions of the frame, clamped to +-0.08):

  x = a_0 + a_1 psi + a_2 theta + a_3^T (h - h_ref),   y = b_0 + b_1 psi + b_2 theta + b_3^T (h - h_ref).

Three fits are compared by leave-one-trial-out RMSE and the best is kept: a forward fit (position regressed on angles), an inverse fit
(angles regressed on position, then inverted, which avoids regression dilution), and the inverse fit with head terms
([experiments/calibrate_uncertainty.py] fit_affine). Pressing a recentre key shifts the output so the current gaze maps to the keyboard
centre (Section H).

### F. Calibration Protocol

The user looks at 16 highlighted keys (Q, W, P, O, A, S, L, Z, X, M, T, D, H, B, SPACE, BACKSPACE; edges weighted) twice, in random order,
32 trials in about 2.5 minutes, on the same full-screen keyboard used for typing. For each trial the first 0.9 s are discarded
(settle delay: the eye is still travelling), a fixation of at least 500 ms with at least four samples must form (or, after a 7 s timeout,
a best fixation of at least 300 ms is accepted, else the trial is skipped). For each accepted trial we store the target centre
(x*, y*), the mean of the features over the last 0.5 s (f_bar), up to ten of the individual frames in that window (f_1..f_10), the mean
absolute gaze angles and the head position. Each run is archived with the participant identifier and the keyboard size, because a
calibration is only valid for the keyboard canvas it was recorded on ([backend/calibration.py]).

### G. Ridge-Regression Mapping

Let F in R^(n x d) be the stacked training features (individual frames, n about 10 x 32) and Y in R^(n x 2) their targets. Each column is
standardised, z = (f - mu) / max(sigma, 5e-4) (the floor prevents near-constant features from being amplified), and the mapping is the ridge
solution, computed in dual form because n is much smaller than d:

  W = Z^T (Z Z^T + alpha I)^(-1) (Y - y_bar),   y_hat(f) = z^T W + y_bar.

The feature subset s in {iris_core, eye_region, full} and the strength alpha in {10, 100, 1000} are chosen by leave-one-trial-out
cross-validation on held-out frames: for every trial k, a model fitted on the frames of all other trials predicts the frames of trial k,
and

  RMSE(s, alpha) = sqrt( (1/N) sum_frames || y_hat - y* ||^2 ).

Scoring held-out single frames puts the noise of live use into the criterion; scoring on averaged samples alone always prefers the least
regularised model, which then amplifies frame noise in use (a measured example: 31% key hit for the least regularised iris-only model against 45% for a regularised
full model on frame-like input). Runs without stored frames use the averaged-sample fit with synthetic frame noise added to the held-out trial. The feature model is
used only if its held-out RMSE is below 0.92 times that of the CNN affine mapping; otherwise the CNN mapping is used.

### H. Online Mapping, Smoothing and Re-Centring

*Feature averaging.* During typing the feature vector is the mean of the last N = 5 frames (matching the 0.5 s means used in calibration):
f_bar_t = (1/N) sum_{j=0}^{N-1} f_{t-j}; the mapping gives (x_t, y_t) = y_hat(f_bar_t) + delta.

*One Euro filter.* Each coordinate is filtered with the One Euro filter [Casiez et al. 2012] at irregular sample times t_e:

  dx_hat = a_d x' + (1 - a_d) dx_hat_prev,   a = r/(r+1),  r = 2 pi f_c t_e,   f_c = f_min + beta |dx_hat|,
  x_hat = a x + (1 - a) x_hat_prev,

with f_min = 0.4 (more smoothing at rest), beta = 1.5, and f_d = 1.0 for the derivative.

*Cursor leash.* The displayed point (s_x, s_y) only moves when the filtered point u lies farther than r = 0.07 from it, and then trails at
distance r: s <- s + (u - s)(d - r)/d, d = ||u - s||. This hides sub-key scatter during a fixation.

*Off-keyboard test.* A point counts as on the keyboard if the filtered point is within 0.08 of the left, right and bottom edges and
within 0.12 of the top edge (the top row is where the estimate is noisiest); a single noisy frame beyond the edge does not drop the key.

*Re-centring.* Pressing C while looking at the middle of the keyboard sets the offset delta = (0.5, 0.5) - median over the last second of the
un-shifted predictions; the median keeps one noisy frame from shifting everything. The offset is reset when the page reconnects.

### I. Selection Gestures

*Blink.* The eye aspect ratio from the contour landmarks is
EAR = (||p_u1 - p_l1|| + ||p_u2 - p_l2||) / (2 ||p_out - p_in||), averaged over both eyes. An eye is closed when EAR falls below 0.62 of
a self-adapting open-eye baseline (80th percentile of the last 90 open-eye frames, available after 10 frames and at least 0.12) and open again above 0.75 (hysteresis); a selection fires once
the closure lasts at least 450 ms, so natural blinks (100-300 ms) never type. While the eyes are closed the pointer is held, and the key
selected is the one at the time the closure started minus 0.25 s.

*Mouth opening.* MAR = ||p_13 - p_14|| / ||p_78 - p_308||; the mouth counts as open when MAR exceeds the closed-mouth baseline by 0.14
(released at 0.08, with a minimum absolute value of 0.18) for at least 400 ms. The gaze is held only for the gesture currently selected.

*Switch and dwell.* A keypress (Space/Enter) selects the highlighted key. Dwell selection uses a spatial detector: the fixation centre is
the running mean of samples, a sample farther than 0.12 from the centre does not reset the dwell and is set aside unless the gaze stays away
for 250 ms, a selection fires after 600-800 ms, and a 450 ms refractory period after a selection prevents transit keys from firing
([decoder/s4_live.py] RobustSpatialDwell).

### J. Fixation Aggregation and Gaze Uncertainty

The raw (unfiltered) mapped points of a fixation window (200-800 ms, new fixation when a point is farther than 0.18 from the running
mean) are summarised as the mean mu = (mu_x, mu_y) and the within-window scatter sigma_raw = (sigma_x, sigma_y); the unfiltered points are used
because filtering would shrink the scatter ([vision/fixation.py]). The observation noise is the calibrated width

  sigma = clip(s sigma_raw, sigma_floor, sigma_ceil),   s = 1.823,  sigma_floor = 0.08,  sigma_ceil = 0.12,

where the scale s is the median ratio between the measured error and the within-fixation scatter on calibration data, and the floor and
ceiling reflect the real imprecision of webcam gaze (about one key width) and stop the gaze term from vanishing or overwhelming the
language prior. Note that the scatter measured within a fixation underestimates the true error, which includes bias; the scale s corrects
for it empirically.

### K. Key Probabilities

The gaze at a selection is modelled as G ~ N(mu, diag(sigma_x^2, sigma_y^2)); the probability that the user means key K_i with
rectangle [x_i, x_i + w_i] x [y_i, y_i + h_i] is the Gaussian mass over its region, which separates into error-function terms:

  P(K_i | G) proportional to [ Phi((x_i + w_i - mu_x)/sigma_x) - Phi((x_i - mu_x)/sigma_x) ] x [ Phi((y_i + h_i - mu_y)/sigma_y) - Phi((y_i - mu_y)/sigma_y) ],

with Phi(z) = (1 + erf(z / sqrt 2)) / 2, normalised over keys; sigma has a floor to avoid false certainty. For decoding, BACKSPACE is
removed and the mass is renormalised over the 27 symbols (a-z and space) ([decoder/key_probability.py]).

### L. Language Models

*Character model.* An order-4 interpolated n-gram over the alphabet A = {a..z, space} (|A| = 27), trained on the NLTK Gutenberg corpus
(95% of each text for training, 5% held out):

  P(c | h) = sum_{k=0}^{3} lambda_k (n(h_k c) + eps) / (n(h_k) + eps |A|),   lambda = (0.07, 0.13, 0.25, 0.55) for context lengths 0..3,
  eps = 0.05,

where h_k is the last k characters of the history, and the result is normalised ([decoder/char_lm.py]).

*Word lexicon.* A word unigram built from the 8,000 most frequent corpus words of up to 14 letters (archaic function words removed),
plus an AAC word list, raised to a floor probability of 5e-4; log-probabilities log P(w) are stored in `models/lexicon.json`
([decoder/lexicon.py]). Note for reproduction: the AAC list and the evaluation phrases are both added to the floor, so the live S4
lexicon contains the words of the evaluation phrases; the simulated comparison below uses only the character model.

### M. Decoders S0-S4

All decoders receive the same sequence of fixation observations G_1..G_n (n = number of selected letters); they differ in how a
fixation becomes a letter. Let h_i be the key containing mu_i, and c_k the centre of key k.

*S0 (nearest key).* The text is h_1..h_n; there is no correction.

*S1 (hard selection plus language-model correction).* Gaze is collapsed to h_i; a position-blind confusion likelihood is used,
L_i(c) = m if c = h_i, else (1 - m)/(|A| - 1), and the text is found by beam search (width 8) maximising

  T* = argmax_T sum_i [ log L_i(T_i) + lambda log P(T_i | T_{<i}) ],

with the character LM and (m, lambda) = (0.35, 0.5) live (tuned; the library defaults (0.85, 0.25) never override a key and make S1
identical to S0).

*S1+ (adjacency-aware hard selection).* The strongest baseline that still collapses gaze to a key: the probability that the user meant
c given selected key h_i falls with the key distance, L_i(c) proportional to
exp(-1/2 [((c_x - h_x)/(kappa sigma_x))^2 + ((c_y - h_y)/(kappa sigma_y))^2]), with the calibrated sigma and a tuned multiplier kappa, and
the same beam search. Compared with S4 it differs in one thing: whether the continuous position inside and around the key is kept.

*S2 (adaptive keyboard).* S1 plus a keyboard in which, in each row, key widths are proportional to
max(P_LM(c | typed text), 0.01)^0.5 (temperature 0.5 so one likely key cannot swallow the row), keeping row heights and bounds
([backend/adaptive_layout.py]). On recorded sequences S2 decodes identically to S1, so it has no separate row in the batch results.

*S3 (online adaptation).* S2 plus a running gaze-bias estimate b, learned only from moments when the user's behaviour reveals the intended
key: a language-model correction of a dictionary word (the gaze at each changed letter against the letter it was changed to), and
backspace-and-retry. For each such letter with residual r = mu - c_new (kept only if |r_x|, |r_y| <= 0.25):

  b <- clip(b + eta r, -0.15, 0.15),   eta = 0.25,   and later fixations use mu' = mu - b.

Letters typed correctly teach nothing (the gaze was inside the key either way), so they are not used; the true text is never used.

*S4 (joint gaze-distribution decoding).* Each fixation gives a distribution over letters L_i(c) = P(K_c | G_i) (Section K), and the word
is decoded by the noisy-channel rule P(T | G) proportional to P(G | T) P(T). With n letters selected for the current word, every lexicon word
w of length n is scored

  score(w) = sum_{i=1}^n log L_i(w_i) + gamma log P(w),   gamma = 1,

an out-of-vocabulary candidate from the character-model beam search (width 12) is added with the same gaze term and a flat penalty prior
(log P_oov = lowest lexicon log-probability minus 2), and the top three candidates are shown for one-press selection. In the
simulation (Section N) S1+ and S4 use the same character model and beam search, so the comparison isolates how the gaze is used; the live
S4 additionally uses the lexicon.

### N. Evaluation Protocol

*Regimes.* "Calibrated" uses within-run held-out errors; "calibration-free" uses across-sitting errors (a mapping fitted on all other runs
and applied to this one without refitting). For six of the seven runs the other runs are mostly the same person on other sittings, so this
regime tests the absence of a per-session refit; the case of a person with no data at all is the single held-out participant (and the
planned study).

*Perception.* For each calibration run we report the key-hit rate (the nearest key to the predicted point is the target; chance with 28
keys is about 3.6%) and the RMSE in keyboard units. Within a sitting, each trial is predicted by a model that did not see it
(leave-one-trial-out); across sittings, a model is fitted on all other runs and predicts the held-out run, which is the realistic case of an
unrefitted mapping (for one run, the other runs are from a different person).

*Typing simulation.* The decoders are compared in a paired, trace-driven simulation: for each character of an evaluation phrase, the true key
centre plus an error vector sampled from the measured error pool gives the observation, with the calibrated sigma; every system decodes the
identical sequences. Twenty fixed phrases (config/evaluation_phrases.txt) are each simulated 100 times. The free parameters of S1 (m, lambda),
S1+ (lambda, kappa) and S4 (lambda, sigma multiplier) are tuned by grid search on 120 different phrases with different random seeds, with
the same budget for every system, separately per condition.

*Drift.* Forty-phrase sessions with a gaze bias growing linearly to (-0.14, -0.10) (the offset measured between a stale and a fresh
mapping; half and double are also run), 100 sessions per setting; S3 and S4 may adapt after each phrase.

*Metrics and statistics.* Character error rate CER = Levenshtein(ref, hyp) / |ref|, word error rate, character accuracy and exact
sentence match. The unit of analysis is the phrase (scores averaged over the 100 draws); intervals are 95% bootstrap intervals over the 20
phrases (10,000 resamples), differences are tested with the paired Wilcoxon signed-rank test, and the relative reduction is (A - B)/A.

*User study (planned; protocol in the repository).* Participants first type with a generic mapping fitted on other people (no
calibration of their own), then calibrate once, then type again; system order is counterbalanced by a Latin square; the same selection
method is used for everyone. Leave-one-person-out accuracy needs at least three distinct people and is not reported here.

---

## IV. ALGORITHMS

**Algorithm 1: Calibration (per user, fixed seat)**
1. Show the full-screen keyboard; start the camera and wait until a face is tracked.
2. Build the trial list: the 16 target keys repeated twice, in random order.
3. For each trial: highlight the target key, discard the first 0.9 s, then track the fixation.
4. When a fixation of at least 500 ms with at least four samples forms (or the 7 s timeout passes with a best fixation of at least 300 ms),
   store the target centre, the mean feature vector over the last 0.5 s, up to ten frames from that window, the mean gaze angles and the
   head position; otherwise skip the trial.
5. Build the frame-level training set: every stored frame is a row labelled with its trial's target.
6. For each feature subset s and each alpha: for every trial k, fit the ridge model (Section G) on the frames of the other trials, predict
   the frames of trial k, and accumulate the squared error.
7. Choose the (s, alpha) with the smallest held-out RMSE; fit the final model on all frames.
8. Fit the CNN affine mapping (forward, inverse, inverse with head terms) and keep the best by leave-one-trial-out RMSE.
9. Use the feature model if its RMSE is below 0.92 times the CNN RMSE; otherwise use the CNN mapping. Save the model and the keyboard size.
10. Archive the run with the participant identifier.

**Algorithm 2: Live gaze-to-keyboard mapping (per frame)**
1. Grab the newest frame; every fourth frame run face detection, otherwise track the box.
2. Extract the landmarks, build the head-aligned frame, and compute the feature vector f (Section D).
3. Average the last five feature vectors; predict (x, y) with the ridge model; add the re-centre offset delta.
4. Filter x and y with the One Euro filter on the unclamped point; judge on/off keyboard from the filtered point.
5. Apply the cursor leash; display the leashed point; highlight the key under it.
6. If a gesture selection is active, hold the pointer while the gesture is held.
7. Feed the unfiltered point to the fixation aggregator (mean and scatter).

**Algorithm 3: Selection and S4 decoding**
1. Wait for a selection event (dwell completes, blink or mouth gesture fires, or a switch is pressed); take the fixation's mu and
   sigma_raw; compute sigma = clip(s sigma_raw, floor, ceiling).
2. Compute the letter distribution L_i from the Gaussian mass over each key (Section K).
3. Append L_i to the current word's observations.
4. For every lexicon word of the same length, compute score(w) = sum log L_i(w_i) + log P(w); add the beam-search out-of-vocabulary
   candidate with a flat penalty prior.
5. Show the top three candidates; the best is the working text.
6. On SPACE, commit the word; on BACKSPACE, remove the last letter and re-decode; the user may pick another candidate.

**Algorithm 4: Online bias learning (S3 and S4 with adaptation)**
1. Initialise b = (0, 0).
2. After each committed word, compare the decoded word with the hard-selected keys.
3. For each letter that the language model changed (word in the lexicon, at least three letters): compute the residual r between the
   gaze at that letter and the centre of the letter it was changed to; skip it if |r_x| or |r_y| exceeds 0.25.
4. Update b <- clip(b + 0.25 r, -0.15, 0.15) per axis.
5. Subtract b from the gaze of all later fixations; also update b from a backspace-and-retry pair.
6. Never use the true text; unchanged letters teach nothing.

**Algorithm 5: Evaluation (simulation)**
1. For each condition, load the measured error pool (within a sitting, across sittings, or per back end).
2. Tune the parameters of S1, S1+ and S4 by grid search on 120 tuning phrases with tuning seeds.
3. For each of the 20 test phrases and each of 100 draws: build observations (true key centre plus a sampled error), decode with every system,
   and compute CER and WER.
4. Average within phrase; compute bootstrap intervals over phrases and paired Wilcoxon tests.
5. For drift: run 40-phrase sessions with the growing bias, with and without online adaptation, 100 sessions per setting.

---

## Results (to place after Section IV; text in `results/PERCEPTION_RESULTS.md`)

- 4.1 Gaze perception accuracy (Table 1, Figure 1).
- 4.2 Decoding on measured gaze noise (Table 2, Figure 2).
- 4.3 Noise sweep: S4 CER 0.019 / 0.141 / 0.320 / 0.464 / 0.555 against S1+ 0.036 / 0.270 / 0.517 / 0.655 / 0.735 at per-axis noise 0.04 /
  0.08 / 0.12 / 0.16 / 0.20 (relative reduction against S1+: 47% / 48% / 38% / 29% / 25%).
- 4.4 Online adaptation under drift (measured drift): last-ten-phrase CER 0.132 with online bias against 0.534 for static S4 and
  0.818 for S2 (0.567 for S3).
- 4.5 Calibrated versus calibration-free, every decoder (Table 3, below).
- 4.6 Typing study: pending participants.

### Table 3. Cost of removing calibration (CER, simulation on measured errors; mean over 20 phrases)

| System | Calibrated (within sitting) | Calibration-free (not refitted, eye-region features) | Factor | Calibration-free vs calibrated S0 |
|---|---|---|---|---|
| S0 nearest key | 0.226 | 0.655 | 2.9x | 2.9x |
| S1 hard + LM correction | 0.201 | 0.649 | 3.2x | 2.9x |
| S1+ adjacency-aware hard selection | 0.140 | 0.540 | 3.9x | 2.4x |
| S4 joint decoding | **0.081** | **0.258** | 3.2x | **1.1x** |

Reading the table: dropping the calibration roughly triples the character error rate of every system, so calibration-free typing is
harder, not free. What S4 buys is that its calibration-free error (0.258) is close to what nearest-key selection achieves *with* a fresh
calibration (0.226), while the hard-selection systems without recalibration stay at 0.54-0.65. With iris-only features the calibration-free
S4 error is 0.281; with the gaze CNN, 0.419 (calibrated 0.254). The one-point re-centre and online bias learning are tested separately
(Section 4.4 for drift); the across-people case is a single held-out person and the planned study.

## Limitations (paste in Section V or VI)

Only two participants contributed calibration data (six of the seven runs are from one person), so accuracy figures describe these
people and this setup (one laptop, one seat, one lighting condition); the typing results are from a paired simulation on measured errors
and not from users typing; the live S4 lexicon includes the words of the evaluation phrases, so live typing results with these phrases
would favour S4; across-sitting folds combine head pose, lighting and time; and no result here supports a calibration-free claim for
people who have never calibrated, beyond one held-out participant; "calibration-free" in this paper means no refit for the sitting, a fixed seat and at most one
re-centring glance. These limits are the content of the planned user study.

## References (verify each before submission)

[1] P. Majaranta and K.-J. Raiha, "Twenty years of eye typing: systems and design issues," ETRA 2002.
[2] R. J. K. Jacob, "What you look at is what you get: eye movement-based interaction techniques," CHI 1990.
[3] M. E. Mott, S. Williams, J. O. Wobbrock, M. R. Morris, "Improving dwell-based gaze typing with dynamic, cascading dwell times," CHI 2017.
[4] P. O. Kristensson and K. Vertanen, "The potential of dwell-free eye-typing for fast assistive gaze communication," ETRA 2012.
[5] D. J. Ward, A. F. Blackwell, D. J. C. MacKay, "Dasher - a data entry interface using continuous gestures and language models," UIST 2000.
[6] X. Zhang, Y. Sugano, M. Fritz, A. Bulling, "Appearance-based gaze estimation in the wild," CVPR 2015.
[7] X. Zhang, Y. Sugano, M. Fritz, A. Bulling, "It's written all over your face: full-face appearance-based gaze estimation," CVPRW 2017.
[8] K. Krafka et al., "Eye tracking for everyone," CVPR 2016.
[9] A. Papoutsaki et al., "WebGazer: scalable webcam eye tracking using user interactions," IJCAI 2016.
[10] ck-zhang, "EyeTrax," GitHub repository (MIT licence); A. Lame, "GazeTracking," GitHub repository (MIT licence); Ai-DuDe, "Gaze_typing,"
     GitHub repository (MIT licence).
[11] Y. Sugano, Y. Matsushita, Y. Sato, "Calibration-free gaze sensing using saliency maps," CVPR 2010.
[12] M. X. Huang, T. C. K. Kwok, G. Ngai, S. C. F. Chan, H. V. Leong, "Building a personalized, auto-calibrating eye tracker from user
     interactions," CHI 2016.
[13] S. Hutt and S. D'Mello, "Evaluating calibration-free webcam-based eye tracking for gaze-based user modeling," ICMI 2022.
[14] A. Garcia de la Santa Ramos, R. Cabeza, A. Villanueva, "Calibration free eye tracking solution for mobile and embedded devices," ETRA 2023,
     doi:10.1145/3588015.3589539.
[15] P. Blignaut, K. Holmqvist, M. Nystrom, R. Dewhurst, "Improving the accuracy of video-based eye tracking in real time through
     post-calibration regression," in Current Trends in Eye Tracking Research, Springer 2014.
[16] J. Goodman, G. Venolia, K. Steury, C. Parker, "Language modeling for soft keyboards," IUI 2002.
[17] K. Vertanen, H. Memmi, J. Emge, S. Reyal, P. O. Kristensson, "VelociTap: investigating fast mobile text entry using sentence-based
     decoding of touchscreen keyboard input," CHI 2015.
[18] G. Casiez, N. Roussel, D. Vogel, "1 Euro filter: a simple speed-based low-pass filter for noisy input in interactive systems," CHI 2012.
[19] T. Soukupova and J. Cech, "Real-time eye blink detection using facial landmarks," CVWW 2016.
[20] J. Deng et al., "RetinaFace: single-shot multi-level face localisation in the wild," CVPR 2020.
[21] Y. Kartynnik et al., "Real-time facial surface geometry from monocular video on mobile GPUs," arXiv:1907.06724, 2019.
[22] P. K. A. Vasu et al., "MobileOne: an improved one millisecond mobile backbone," CVPR 2023.
[23] A. E. Hoerl and R. W. Kennard, "Ridge regression: biased estimation for nonorthogonal problems," Technometrics 1970.
[24] P. Kellnhofer, A. Recasens, S. Stent, W. Matusik, A. Torralba, "Gaze360: physically unconstrained gaze estimation in the wild," ICCV 2019.
[25] S. Park, S. De Mello, P. Molchanov, U. Iqbal, O. Hilliges, J. Kautz, "Few-shot adaptive gaze estimation," ICCV 2019.
[26] Z. Chen and B. E. Shi, "Offset calibration for appearance-based gaze estimation via gaze decomposition," WACV 2020.
[27] V. Saran, J. Alber, C. Zhang, J. Paradiso, D. Bragg, J. Langford, "EyeO: autocalibrating gaze output with gaze input for gaze typing," CHI EA 2025.
[28] A. M. Kurauchi, W. Feng, A. Joshi, C. H. Morimoto, M. Betke, "EyeSwipe: dwell-free text entry using gaze paths," CHI 2016.
[29] B. Benligiray, C. Topal, C. Akinlar, "SliceType: fast gaze typing with a merging keyboard," J. Multimodal User Interfaces 13(4), 2019.
[30] V. Abdrabou, M. Mostafa, M. Khamis, W. Elmougy, "Calibration-free text entry using smooth pursuit eye movements," ETRA 2019.
[31] S. Cai et al., "Using large language models to accelerate communication for eye gaze typing users with ALS," Nature Communications, 2024.
[32] "Adapting large language models for character-based augmentative and alternative communication," arXiv:2501.10582, 2025.
[33] S. Valencia, R. Cave, K. Kallarackal, K. Seaver, M. Terry, S. K. Kane, "'The less I type, the better': how AI language models can enhance or impede communication for AAC users," CHI 2023.
