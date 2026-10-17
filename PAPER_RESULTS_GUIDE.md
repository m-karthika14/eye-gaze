# Writing the Results section: complete guide

For: you, the author of the paper. Repository state as of 17 Oct 2026. Every number in section 5 was produced by the commands in section 1 and is copied from `results/`; if you change the code or data, re-run and re-copy. Do not edit numbers by hand.

---

## 0. What you can claim, and what you cannot (read this first)

| Claim in your design | Evidence in the repo now | What to write |
|---|---|---|
| **H1**: keeping the full gaze probability distribution and decoding it jointly with a language model beats selecting one key first and correcting afterwards | Trace-driven **simulation** on gaze error measured on a real webcam; S1 and S4 tuned with equal effort; an additional, stronger hard-selection baseline (S1+); paired statistics (section 5.2, 5.3) | "In simulation driven by measured gaze error, S4 reduced CER by X% relative to ..." Not "users type X% better". |
| **H2**: online adaptation helps over repeated use without an explicit calibration | **Simulated** posture drift; S3 learns bias from LM corrections (section 5.4) | "Under simulated drift, online adaptation reduced late-session CER by X%". It is a mechanism demonstration, not a user result. |
| "Calibration-free" (your title) | **Not supported.** The deployed system needs a ~2.5 minute calibration, and a mapping fitted in one sitting loses a large part of its accuracy in the next (section 5.1) | Do not use the phrase. Use "short per-sitting calibration" or reframe the contribution as *reducing the calibration burden*. Or build Phase 7 properly and test it with people. |
| Typing speed (WPM), usability, assistive benefit | **Not measured.** No participants, no timing logs | Say so under limitations. These need a user study (section 9). |
| Generalises across people, cameras, lighting | **One person, one camera, one afternoon** (2 calibration runs) | State it as the main external-validity limitation. |

The honest paper you can write today: *a method (S4) + a rigorous simulation study driven by real measured webcam error + a real-world measurement of the perception layer, including the finding that cross-sitting generalisation is the real bottleneck, + a live prototype.* That is a legitimate contribution if framed that way.

---

## 1. What was built for results, and how to regenerate everything

| File | What it is |
|---|---|
| `experiments/sim_core.py` | Trace-driven simulation core: measured error pools, the five decoders in batch form, the S3/S4 online-bias learner, metrics |
| `experiments/run_results.py` | The experiment runner: tuning, test, noise sweep, drift, statistics, tables, figures |
| `experiments/perception_summary.py` | Perception table: within-run and across-sitting accuracy per calibration run |
| `data/calibration_history/*.jsonl` | Every Calibrate run, archived by the app. **This is your raw data.** Back it up; do not delete runs |
| `results/tables.md`, `summary.json`, `per_phrase.csv`, `tuning_*.csv`, `perception.md`, `figures/*.png` | Outputs |
| `decoder/live_modes.py` + dashboard switch | S0-S3 as live modes next to the existing live S4, for demos and a future user study |

```
.venv/Scripts/python.exe -m experiments.run_results          # ~30 min on 8 cores; use --quick (~2 min) only to check it runs
.venv/Scripts/python.exe -m experiments.perception_summary   # seconds
.venv/Scripts/python.exe -m experiments.test_sim_core        # tests of the pipeline itself
```

More calibration runs make the error pools larger and more representative: run **Calibrate** at the start of every session (different days, lighting, seating) and re-run the two scripts. Two runs from one afternoon is the minimum, not a dataset.

---

## 2. Methods text you can adapt

State the protocol *before* the results; a reader must be able to see that the comparison is fair.

**2.1 Perception and measured noise.** "Gaze was estimated from a laptop webcam (480x640, ~20 fps) by a face-mesh landmark model (478 points including irises), converted to head-normalised eye features (after Zhang, EyeTrax, MIT licence) and mapped to a position on the on-screen keyboard by ridge regression fitted during a short calibration (16 keys x 2 repeats, ~2.5 min). As an alternative back-end we used a pretrained full-face gaze CNN (MobileOne-S0) with an affine mapping. Each calibration trial records the true key and the position the mapping assigned to it. We use the resulting error vectors in two ways: *within-run* (each trial predicted by a model fitted on the other trials of the same run) and *across sittings* (fitted on the other runs, predicting this run)."

**2.2 Systems.** "S0 selects the key nearest to the gaze point. S1 collapses gaze to a hard-selected key and then corrects the character sequence with a character n-gram LM (order 4, trained on Gutenberg) using a position-blind confusion model. S1+ is the strongest hard-selection baseline: the same collapse to a key, but the corrector knows the keyboard geometry and the calibrated per-axis sigma (the probability that the user meant key *c* given selected key *h* decays with the distance between their centres). S2 adds a confidence-aware adaptive keyboard; for recorded sequences it decodes identically to S1, so it has no separate row in the batch results. S3 adds online adaptation: a running gaze-bias estimate learned only from LM corrections of dictionary words (letters the LM leaves unchanged teach nothing; the true text is never used). S4 keeps each observation's Gaussian key-probability distribution and decodes it jointly with the same LM: P(T|G) proportional to P(G|T)P(T)."

**2.3 Tuning (the fairness paragraph; reviewers look for it).** "Every system's free parameters were tuned by grid search on a tuning split (120 phrases built from an AAC word list, disjoint from the test phrases, with disjoint random seeds): S1 over confusion match probability and LM weight, S1+ over LM weight and sigma multiplier, S4 over LM weight and sigma multiplier. The shipped S1 defaults never override a selected key (S1 would be identical to S0), so tuning is what makes S1 a real baseline. The grids, the chosen values and all tuning scores are in `results/tuning_*.csv`."

**2.4 Test protocol.** "The 20 fixed evaluation phrases (config/evaluation_phrases.txt; lowercase letters and spaces) were each simulated 100 times. Each simulated character is the true key's centre plus one error vector sampled from the measured pool (or iid Gaussian noise for the sweep). All systems decode the identical observation sequences (paired design)."

**2.5 Metrics and statistics.** "We report character error rate (CER, Levenshtein over characters / reference length), word error rate (WER), character accuracy (fixed-length, position-wise) and sentence exact match. The unit of analysis is the phrase: scores are averaged over the 100 draws within each phrase, and 95% confidence intervals are percentile bootstrap intervals over the 20 phrases (10,000 resamples). System differences use paired differences per phrase with a Wilcoxon signed-rank test and a bootstrap CI of the mean difference. Relative reduction is (A-B)/A."

**2.6 Drift experiment.** "To test online adaptation we simulate 40-phrase sessions in which a gaze bias grows linearly from zero to a final value (measured between a stale and a fresh mapping: about -0.14 horizontally and -0.10 vertically in normalised canvas units; half and double that are also reported). 100 sessions per setting, identical gaze for every system, session as the unit of analysis."

**2.7 Reproducibility.** Seeds (tune 1,000,000; test 50,000,000; session 90,000,000), Python/numpy versions and the date are in `results/summary.json`. Release `data/calibration_history/` and the scripts.

---

## 3. Recommended structure of the Results section

1. **Perception: how accurate is the measured gaze?** (table of within-run vs across-sitting accuracy; the generalisation gap)
2. **Decoding on measured noise** (main table; **H1**, against S1 *and* S1+)
3. **Sensitivity to noise level** (sweep figure; H1's prediction that the advantage grows with noise)
4. **Online adaptation under drift** (drift figure and table; **H2**)
5. **Ablation ladder S0 to S4** (one paragraph or small table: what each rung adds, including why S2 = S1 here)
6. **Negative and unexpected results** (keep them: they make the rest credible)

Each subsection: one sentence on what question it answers, the table or figure, 3-5 sentences reading it, one sentence of caveat.

---

## 4. How to write each paragraph

**The formula.** *Claim, then evidence, then interpretation, then caveat.* "S4 reduced CER from 0.218 to 0.063 (-71%; 95% CI of the difference [0.145, 0.163]; Wilcoxon p = 2e-6, n = 20 phrases). The advantage comes from keeping which side of a key boundary the gaze fell on, which a hard selection discards. This is a simulation on error measured in two calibration runs of one user."

**Do**
- Give the number, its interval, the test, the n. Lead with the effect size; the p-value is supporting.
- Name the condition in every sentence: "within-run", "across sittings", "simulated drift".
- Use "simulation", "simulated", "trace-driven". Never "users", "participants", "typed".
- Report the baselines' tuned values, and that S1's defaults equal S0.
- Say all 20 phrases improved (it is true, and it explains the p-value).
- Report sentence-level exact match; it is the number a reader finds most intuitive ("46.9% of phrases decoded perfectly").

**Do not**
- Do not claim "calibration-free", "real-time typing speed", or any benefit for users with motor impairment.
- Do not compare S4 only to the untuned S1.
- Do not mix within-run and across-sitting numbers in one sentence.
- Do not over-read identical p-values: with n = 20 and every difference in the same direction, the two-sided Wilcoxon p-value has a floor of 1.9e-6, so many comparisons print exactly that. Say "all 20 phrases improved (p = 1.9e-6, the minimum attainable with n = 20)" and let the CI and effect size carry the argument.
- Do not present WER above 1.0 as a bug: word segmentation can break when spaces are mis-decoded, producing insertions.
- Do not state the within-run accuracy as what a user will experience later (section 5.1 shows why).

**Useful sentence stems**
- "Across all five measured-noise conditions, S4 had the lowest CER; the advantage over S1+ ranged from X% to Y%."
- "The advantage shrinks as noise grows (from X% at sigma = 0.04 to Y% at sigma = 0.20), consistent with the information that S4 preserves becoming less informative when the gaze is nearly uninformative."
- "Online adaptation did not help when drift exceeded the capture range of the learner (double drift), because the learner needs mostly-correct decodings to bootstrap itself."
- "These results are from a simulation driven by 64 measured trials from two calibration runs of a single user and should be read as a mechanism demonstration, not as a usability result."

---


---

## 5. The results (copy numbers from here, not from memory)

Generated on 17 Oct 2026 by `experiments/run_results.py` (run time 1678 s): 20 fixed test phrases x 100 noise draws, 120 tuning phrases x 2 draws, 100 drift sessions per setting. Noise is drawn from error vectors **measured in 2 calibration runs** (n = 64 measured trials per pool): one person, one camera, one afternoon. Everything below is **simulation**.

### 5.1 Perception: how accurate is the measured gaze?

Source: `results/perception.md`. A key counts as a hit when the nearest key to the predicted point is the target (chance with 40 keys is about 2.5%).

| Run | Trials | Eye-landmark, within run (held-out trials) | Gaze CNN, within run | Mapping that was live during the run |
|---|---|---|---|---|
| 20261010_144100_run | 32 | 69% key hit, RMSE 0.117 | 22% key hit, RMSE 0.186 | 25% key hit, RMSE 0.226 |
| 20261010_144635 | 32 | 84% key hit, RMSE 0.118 | 56% key hit, RMSE 0.128 | 25% key hit, RMSE 0.263 |

The last column is how the mapping that happened to be live during that run performed: for run 1 an older model from earlier that afternoon, for run 2 the model fitted on run 1 a few minutes before. It is the closest thing to a real later-sitting test in the data, and it is poor (25% both times).

**Across sittings** (fit on the other run, predict this one; RMSE in normalised canvas units, a key is 0.10 wide and 0.25 tall):

| Configuration | Key hit (run 1, run 2) | RMSE (run 1, run 2) |
|---|---|---|
| Eye-landmark, eye-region features, ridge alpha 10 | 34%, 47% | 0.199, 0.152 |
| Eye-landmark, iris-only features, alpha 10 (**what Calibrate currently deploys**) | 12%, 25% | 0.287, 0.263 |
| Gaze CNN, absolute-angle affine | 25%, 41% | 0.218, 0.189 |

**How to write it.** "Within a calibration run, the eye-landmark mapping placed 69-84% of held-out trials on the correct key (chance ~2.5%), against 22-56% for the gaze-CNN mapping. Applied to a later calibration run, accuracy fell sharply (best configuration: 34%, 47%), i.e. a mapping fitted in one sitting does not transfer well even a few minutes later (the two runs were about five minutes apart, and the face had moved about 9% of the frame height between them). This gap, not the decoder, is the main obstacle to calibration-free use."

**Be careful:** the ranking of feature sets is based on one pair of runs. Do not write "eye-region features are better" as a general fact; write "in our two runs". The iris-only configuration, which seemed the most principled, generalised worst: say so (it is a finding, not an embarrassment).

### 5.2 Decoding on measured noise (H1)

Main condition (freshly calibrated eye-landmark perception, within-run errors). Mean [95% CI over 20 phrases]:

| System | CER | WER | Char. accuracy | Sentence exact |
|---|---|---|---|---|
| S0 nearest key | 0.252 [0.246, 0.257] | 0.780 | 74.8% | 1.2% |
| S1 hard key + LM correction (tuned) | 0.218 [0.211, 0.224] | 0.660 | 78.2% | 5.0% |
| S1+ hard key + adjacency-aware correction (tuned) | 0.160 [0.152, 0.169] | 0.517 | 83.9% | 13.8% |
| **S4 joint gaze distribution + LM** | 0.063 [0.052, 0.074] | 0.186 | 93.7% | 46.9% |

S2 decodes identically to S1 on recorded sequences, so it is not a separate row.

**How to write it.** "S4 reduced character error rate from 0.218 (S1) to 0.063, a 71% relative reduction (mean paired difference 0.155, 95% CI [0.145, 0.163], Wilcoxon signed-rank p = 1.9e-06; all 20 phrases improved). Against the strongest hard-selection baseline S1+, which uses the same hard-selected key but knows the keyboard geometry and calibrated sigma, S4 still reduced CER by 61% (0.160 to 0.063; CI of the difference [0.088, 0.107]). 46.9% of simulated phrases were decoded exactly by S4, against 5.0% for S1 and 13.8% for S1+."

**All five measured conditions** (CER; reduction columns are S4 relative to S1 and to S1+, with the CI of the S1+ reduction in brackets, in percent):

| Condition | S0 | S1 | S1+ | S4 | vs S1 | vs S1+ |
|---|---|---|---|---|---|---|
| Eye-landmark, freshly calibrated | 0.252 | 0.218 | 0.160 | **0.063** | 71% | 61% [55, 67] |
| Gaze CNN, freshly calibrated | 0.584 | 0.572 | 0.418 | **0.223** | 61% | 47% [42, 51] |
| Eye-landmark (eye-region), across sittings | 0.639 | 0.629 | 0.527 | **0.274** | 57% | 48% [43, 53] |
| Eye-landmark (iris-only), across sittings | 0.831 | 0.802 | 0.733 | **0.488** | 39% | 33% [29, 37] |
| Gaze CNN, across sittings | 0.668 | 0.658 | 0.584 | **0.308** | 53% | 47% [44, 50] |

**How to write it.** "The advantage held in all five measured-noise conditions: S4 reduced CER relative to S1+ by 33% to 61%. The absolute error is dominated by the perception condition: across sittings even S4 has CER 0.27-0.49 and almost no phrase is decoded exactly (6.0% at best), which shows that decoding cannot compensate for a stale calibration."

Tuned parameters (report in a supplementary table; full grids in `results/tuning_*.csv`):

| Condition | S1 (match, LM weight) | S1+ (LM weight, sigma multiplier) | S4 (LM weight, sigma multiplier) |
|---|---|---|---|
| landmarks_loo | 0.35, 0.5 | 2.0, 0.8 | 1.0, 1.0 |
| cnn_loo | 0.35, 0.5 | 0.25, 2.0 | 2.0, 0.6 |
| landmarks_xrun_eye | 0.35, 0.5 | 1.5, 1.0 | 1.5, 0.8 |
| landmarks_xrun_iris | 0.95, 2.0 | 1.5, 1.25 | 2.0, 0.6 |
| cnn_xrun | 0.35, 0.5 | 1.0, 0.8 | 1.0, 0.8 |

Figure 3 (`results/figures/fig_real_conditions.png`) shows the two freshly-calibrated conditions; the across-sitting conditions are in the table above (add them to the figure yourself if you want them plotted).

### 5.3 Sensitivity to noise (the sweep)

CER with iid Gaussian noise of per-axis sigma s (normalised canvas units; one key is 0.10 wide and 0.25 tall):

| sigma | S0 | S1 | S1+ | S4 | S4 vs S1 | S4 vs S1+ | absolute gap S1+ minus S4 | Wilcoxon p (S1+ vs S4) |
|---|---|---|---|---|---|---|---|---|
| 0.04 | 0.161 | 0.132 | 0.036 | 0.019 | 86% | 47% | 0.017 | 0.00071 |
| 0.08 | 0.467 | 0.451 | 0.270 | 0.141 | 69% | 48% | 0.129 | 1.9e-06 |
| 0.12 | 0.641 | 0.631 | 0.517 | 0.320 | 49% | 38% | 0.197 | 1.9e-06 |
| 0.16 | 0.732 | 0.726 | 0.655 | 0.464 | 36% | 29% | 0.192 | 1.9e-06 |
| 0.2 | 0.781 | 0.772 | 0.735 | 0.555 | 28% | 24% | 0.180 | 1.9e-06 |

**How to write it.** "The relative advantage over S1+ was largest at low noise (47% at sigma = 0.04) and shrank as noise grew (24% at sigma = 0.2), while the absolute CER gap rose from 0.017 to a plateau of about 0.20 around sigma = 0.12-0.16 before declining. With extreme noise the gaze carries little information for any system."

**Be careful (this differs from your original prediction).** The design document predicted that S4's advantage *grows* with noise. In relative terms it does not: it falls from 47% to 24% against S1+ because CER saturates for every system. The absolute gap does rise and then plateau. Report both honestly and phrase H1 as "S4 outperforms at every noise level", not "the advantage grows". At sigma = 0.04 the gap to S1+ is small (0.017); the p-value there (0.0007) is the only one above the n = 20 floor, which is itself informative.

### 5.4 Online adaptation under simulated drift (H2)

40-phrase sessions, gaze bias growing linearly to the stated total, 100 sessions per setting. Mean [95% CI over sessions]; "late" = last 10 phrases of the session.

| Drift (x, y) | System | Session CER | Late CER |
|---|---|---|---|
| half (-0.07, -0.05) | S2 (no adaptation) | 0.416 [0.413, 0.419] | 0.654 [0.646, 0.662] |
| half (-0.07, -0.05) | S3 (S2 + online bias) | 0.399 [0.374, 0.424] | 0.467 [0.428, 0.507] |
| half (-0.07, -0.05) | S4 (static) | 0.128 [0.125, 0.132] | 0.234 [0.224, 0.244] |
| half (-0.07, -0.05) | S4 + online bias | 0.131 [0.122, 0.140] | 0.129 [0.118, 0.142] |
| measured (-0.14, -0.1) | S2 (no adaptation) | 0.613 [0.611, 0.616] | 0.844 [0.840, 0.848] |
| measured (-0.14, -0.1) | S3 (S2 + online bias) | 0.516 [0.488, 0.543] | 0.675 [0.636, 0.713] |
| measured (-0.14, -0.1) | S4 (static) | 0.348 [0.344, 0.352] | 0.687 [0.680, 0.694] |
| measured (-0.14, -0.1) | S4 + online bias | 0.124 [0.114, 0.137] | 0.123 [0.104, 0.147] |
| double (-0.28, -0.2) | S2 (no adaptation) | 0.744 [0.742, 0.747] | 0.890 [0.886, 0.893] |
| double (-0.28, -0.2) | S3 (S2 + online bias) | 0.706 [0.692, 0.720] | 0.877 [0.871, 0.882] |
| double (-0.28, -0.2) | S4 (static) | 0.601 [0.598, 0.604] | 0.905 [0.901, 0.909] |
| double (-0.28, -0.2) | S4 + online bias | 0.245 [0.229, 0.265] | 0.554 [0.538, 0.572] |

| Drift | S2 to S3, session CER | S2 to S3, late CER | S4 static to S4 + online bias, late CER |
|---|---|---|---|
| half | 4% (CI of diff [-0.008, +0.042]) | 29% (CI [+0.148, +0.225]) | 45% (CI [+0.090, +0.118]) |
| measured | 16% (CI of diff [+0.070, +0.125]) | 20% (CI [+0.131, +0.209]) | 82% (CI [+0.538, +0.586]) |
| double | 5% (CI of diff [+0.025, +0.052]) | 1% (CI [+0.008, +0.018]) | 39% (CI [+0.332, +0.367]) |

**How to write it.** "With the measured amount of drift, a static S4 started the session at a CER of 0.08 but degraded to 0.69 over the last ten phrases; adding online bias adaptation held it at 0.12 throughout (an 82% reduction in late-session CER; averaged over the whole session, 0.12 versus 0.35). The same mechanism on the S2 decoder (S3) helped less, reducing late CER by 20% (0.84 to 0.68), because with hard selection fewer letters are corrected for the learner to learn from."

**Be careful, report these too.**
- *Early-session cost.* S3 is slightly **worse** than S2 for the first ~10 phrases (see the figure: the orange curve starts above the blue one) because the estimate is still noisy. Over a whole session with small drift (4% at half drift) the CI of the session difference includes zero [-0.008, +0.042] even though the Wilcoxon test is significant: lead with the CI.
- *Double drift defeats it.* At twice the measured drift the learner recovers little for S3 (1% late) and only part for S4 (late CER 0.55): learning from LM corrections needs mostly-correct decodings to bootstrap. This is a real limit of the mechanism; say so.
- *Optimism.* The learner confirms corrections against a word list that contains the test vocabulary; drift is linear and its size is known. Real drift is irregular and the vocabulary is open.
- *Not calibration-free.* All drift conditions start from a fresh calibration (zero bias at the start of the session).

### 5.5 The ablation ladder in one paragraph

"Each rung adds one ingredient (freshly calibrated eye-landmark noise): S0 (nearest key) CER 0.252; S1 adds post-hoc LM correction, 0.218 (14% better than S0, small because a position-blind corrector cannot tell which key was the plausible mistake); S2 adds the adaptive keyboard, which cannot be evaluated by replaying recorded gaze and is therefore identical to S1 in this study; S3 adds online adaptation, which only matters under drift (section 5.4); S4 adds joint decoding of the gaze distribution, 0.063 (71% better than S1). The extra baseline S1+ (hard key, geometry-aware correction) sits between S1 and S4 at 0.160: roughly 37% of S4's gain over S1 comes from knowing the keyboard geometry, and the remaining 63% from keeping the continuous gaze position."

That last sentence is the cleanest statement of what your contribution adds. Use it (with the numbers from your final run).

---

## 6. Negative and unexpected results: report these, they make the rest believable

1. **The shipped S1 equals S0.** With the default parameters (match probability 0.85, LM weight 0.25), S1 never overrides a selected key; the unit test `test_sim_core` asserts it. A paper that compared S4 to that S1 would exaggerate H1. Say you tuned it, and that you added S1+.
2. **A tuned S1 helps only a little** (14% over S0): post-hoc correction with a position-blind confusion model is weak. That is the point of H1, but also a reason a reader may find S1 an easy baseline, which is why S1+ is there.
3. **The cross-sitting drop.** Accuracy measured inside one calibration run is roughly twice as good in key-hit terms (69-84% within-run vs 34-47% at best across runs), and RMSE rises by 1.3-2.4 times (about 0.12 vs 0.15-0.29) when the mapping is applied to the next run. This is the most important practical finding and contradicts "calibration-free".
4. **The iris-only feature set generalised worst** (12%, 25% key hit across runs vs 34%, 47% for eye-region features). The more "physically meaningful" features were not the more robust ones in this data.
5. **The relative advantage of S4 shrinks with noise** (section 5.3), unlike the original prediction.
6. **S3 is slightly worse than S2 early in a session** and cannot recover from very large drift (section 5.4).
7. **The gaze CNN is not uniformly worse:** within-run it ranged from 22% to 56% key hit between the two runs, so run-to-run variability is large; with two runs, avoid strong claims about which back-end is better.
8. **WER can exceed 1.0** when mis-decoded spaces break word boundaries.

---



## 7. Limitations: paste-ready, adapt wording

- **Simulation, not users.** Real typing includes dwell/blink timing, fatigue and error-correction behaviour; the errors in real use are temporally and spatially correlated, which bootstrapping independent vectors does not reproduce.
- **Tiny noise sample.** 64 measured error vectors from 2 runs, one person, one camera, one afternoon. The cross-sitting conditions rest on a single pair of runs.
- **Perception is the bottleneck.** The accuracy gap between within-run and across-sitting results means a fresh calibration is needed per sitting; the system is not calibration-free.
- **Language model and vocabulary.** The character LM is trained on literary text (Gutenberg), not AAC speech. The online-bias learner confirms corrections against a word list that includes the evaluation vocabulary, which is optimistic. The live product decoder (word lexicon) is a different component from the batch S4 evaluated here.
- **Fixed-length formulation.** S1-S4 decode one observation per intended character, including spaces; insertions and deletions by the user are not modelled in the batch study.
- **Tuning on simulated phrases.** Tuning phrases come from the same word list family as the test phrases, though disjoint.
- **Drift model.** Linear drift of known magnitude; real posture drift is irregular.

---

## 8. Questions a reviewer will ask, and answers

- *"Your baseline is a strawman."* → S1 is tuned with the same budget as S4, and S1+ is the strongest hard-selection baseline (geometry-aware, calibrated sigma). The remaining gap is exactly what keeping the continuous gaze position buys.
- *"Where are the users?"* → Not done; this is a simulation study with real measured noise. Section 9 describes the user study as the next step. Do not hide this.
- *"Why is the S2 row missing?"* → The adaptive keyboard changes the stimulus, so it cannot be evaluated by replaying recorded gaze; on replay it decodes identically to S1. It matters in live use and belongs in the user study.
- *"Is S3 just a stub?"* → In the batch study it learns a bias from LM corrections (rule stated in 2.2). It is a first mechanism, not the final Phase 7 design.
- *"Does this need calibration?"* → Yes, per sitting. The across-sitting results quantify the cost of not recalibrating.
- *"Is the noise realistic?"* → It is measured, from the real system, but from one user in two runs; see limitations.
- *"Multiple comparisons?"* → The primary comparison (S4 vs S1+ on the main condition) is pre-specified; the rest are descriptive. If you test many pairs, apply Holm correction.

---

## 9. What turns this into a full study (the user study)

The dashboard now switches between S0-S4 live on the same camera, mapping and selection method, so a **within-subject** study is possible. The **🧪 Study** bar now prompts the fixed phrases and logs each finished phrase (typed text, time, selections, backspaces, system, selection method) per participant in `data/sessions/`; `experiments/study_summary.py` turns that into per-system tables (CER, WER, WPM, selections per character, backspaces). Not implemented: SUS / NASA-TLX questionnaires (use paper or a form) and any automatic counterbalancing (follow the Latin square in `STUDY_PROTOCOL.md`).

- **Design:** within-subject, counterbalanced order (Latin square) over S0, S1, S3, S4 (S2 changes the keyboard layout; include it if you want to test the adaptive keyboard). One short calibration at the start of each block.
- **Task:** copy 10 phrases per condition from a fixed list (use the 20 evaluation phrases split into two sets); a short practice phrase before each condition.
- **N:** at least 8-12 participants for a first study (state this is a pilot). Able-bodied participants first; a clinical population needs ethics review and partner organisations.
- **Measures:** CER/WER of what was typed, WPM and keystrokes per character, backspaces per character, number of candidate picks, plus SUS and NASA-TLX after each condition.
- **Analysis:** Friedman test across conditions, Wilcoxon signed-rank for the pre-specified pairs with Holm correction, report effect sizes and CIs; a mixed-effects model with participant as a random effect if you have enough data.
- **Ethics:** informed consent, data handling (webcam video is sensitive: this system stores no video, only feature vectors in `data/calibration_history/`; state it), approval from your institution.

---

## 10. Checklists

**Before submission**
- [ ] The words "calibration-free" appear nowhere unless Phase 7 is implemented and tested on people.
- [ ] Every table says whether noise is within-run, across-sitting, Gaussian or drifted.
- [ ] Baseline tuning, grids and chosen values are reported (or in a supplement).
- [ ] CIs on every reported mean; the unit of analysis (phrase / session) is stated.
- [ ] "Simulation" is in the abstract.
- [ ] Limitations section includes the single-user, single-camera, two-run facts.
- [ ] You cite EyeTrax (Zhang, 2025, MIT) for the feature extraction idea, MediaPipe for the face mesh, and your literature survey papers.
- [ ] Numbers in the text were copied from `results/` after the final run.

**Figures to include** (all generated into `results/figures/`)
1. `fig_noise_sweep.png`: *CER versus gaze noise for S0, S1, S1+ and S4 (95% CI over phrases).* Shows the advantage and how it shrinks.
2. `fig_drift.png`: *CER over a 40-phrase session under simulated posture drift for S2, S3, S4 and S4 with online bias.* Shows adaptation recovering accuracy.
3. `fig_real_conditions.png`: *CER on measured error pools, per system.* The headline comparison.


---

## 11. Running the calibration-free test and the typing study

`STUDY_PROTOCOL.md` is the step-by-step plan (timings, Latin square, exact commands). The tools: participant-ID box in the accuracy check, the **🧪 Study** bar (target phrase and per-phrase logging in `data/sessions/`), `experiments/loo_person.py` (calibration-free accuracy across people), `experiments/use_mapping.py` (switch to a generic mapping built without the participant), `experiments/study_summary.py` (typing results per system). The numbers in section 5 come from your two calibration runs and are simulation only; replace the perception conditions with the across-people ones once 3+ people exist (`python -m experiments.run_results` picks them up automatically), and add the typing study as a separate results subsection.

---

## Appendix: the raw generated tables (identical to `results/tables.md`)


### Real noise: eye-landmark perception, freshly calibrated (within-run held-out errors)

| System | CER ↓ | WER ↓ | Char. acc. ↑ | Sentence exact ↑ |
|---|---|---|---|---|
| S0 | 0.252 [0.246, 0.257] | 0.780 [0.727, 0.842] | 74.8% [74.3, 75.4] | 1.2% [0.6, 2.0] |
| S1 | 0.218 [0.211, 0.224] | 0.660 [0.608, 0.724] | 78.2% [77.5, 78.9] | 5.0% [3.5, 6.6] |
| S1P | 0.160 [0.152, 0.169] | 0.517 [0.468, 0.574] | 83.9% [83.1, 84.7] | 13.8% [10.9, 16.9] |
| S2 | 0.218 [0.211, 0.224] | 0.660 [0.608, 0.724] | 78.2% [77.5, 78.9] | 5.0% [3.5, 6.6] |
| S4 | 0.063 [0.052, 0.074] | 0.186 [0.155, 0.219] | 93.7% [92.6, 94.7] | 46.9% [37.5, 55.6] |

S2 decodes identically to S1 on recorded sequences (experiments/s2.py), so its row equals S1. S1 tuned: match=0.35, LM weight=0.5; S4 tuned: LM weight=1.0, sigma x1.0. S1+ (adjacency-aware, hard-selected key) tuned: LM weight=2.0, sigma x0.8. Mean [95% bootstrap CI over the 20 phrases]. **H1 (S1 vs S4)**: CER reduction 71.0% (mean diff +0.155, 95% CI [+0.145, +0.163], Wilcoxon p=1.9e-06, n=20 phrases). **H1 vs the strongest hard-selection baseline (S1+ vs S4)**: CER reduction 60.6% (mean diff +0.097, 95% CI [+0.088, +0.107], Wilcoxon p=1.9e-06).

### Real noise: gaze-CNN perception, freshly calibrated (within-run held-out errors)

| System | CER ↓ | WER ↓ | Char. acc. ↑ | Sentence exact ↑ |
|---|---|---|---|---|
| S0 | 0.584 [0.571, 0.597] | 1.096 [1.043, 1.163] | 41.5% [40.2, 42.8] | 0.0% [0.0, 0.0] |
| S1 | 0.572 [0.559, 0.586] | 1.081 [1.019, 1.160] | 42.5% [41.1, 43.9] | 0.0% [0.0, 0.0] |
| S1P | 0.418 [0.400, 0.435] | 0.894 [0.825, 0.973] | 58.0% [56.2, 59.7] | 0.9% [0.4, 1.5] |
| S2 | 0.572 [0.559, 0.586] | 1.081 [1.019, 1.160] | 42.5% [41.1, 43.9] | 0.0% [0.0, 0.0] |
| S4 | 0.223 [0.198, 0.249] | 0.481 [0.429, 0.531] | 77.6% [74.9, 80.2] | 11.3% [6.3, 17.2] |

S2 decodes identically to S1 on recorded sequences (experiments/s2.py), so its row equals S1. S1 tuned: match=0.35, LM weight=0.5; S4 tuned: LM weight=2.0, sigma x0.6. S1+ (adjacency-aware, hard-selected key) tuned: LM weight=0.25, sigma x2.0. Mean [95% bootstrap CI over the 20 phrases]. **H1 (S1 vs S4)**: CER reduction 61.0% (mean diff +0.349, 95% CI [+0.323, +0.373], Wilcoxon p=1.9e-06, n=20 phrases). **H1 vs the strongest hard-selection baseline (S1+ vs S4)**: CER reduction 46.6% (mean diff +0.195, 95% CI [+0.175, +0.214], Wilcoxon p=1.9e-06).

### Real noise: eye-landmark perception, eye-region features, across sittings (fit on other runs)

| System | CER ↓ | WER ↓ | Char. acc. ↑ | Sentence exact ↑ |
|---|---|---|---|---|
| S0 | 0.639 [0.629, 0.649] | 1.139 [1.074, 1.220] | 35.9% [34.9, 36.9] | 0.0% [0.0, 0.0] |
| S1 | 0.629 [0.621, 0.638] | 1.128 [1.063, 1.210] | 36.6% [35.7, 37.5] | 0.0% [0.0, 0.0] |
| S1P | 0.527 [0.514, 0.539] | 1.058 [0.980, 1.150] | 46.7% [45.4, 48.0] | 0.2% [0.0, 0.5] |
| S2 | 0.629 [0.621, 0.638] | 1.128 [1.063, 1.210] | 36.6% [35.7, 37.5] | 0.0% [0.0, 0.0] |
| S4 | 0.274 [0.242, 0.306] | 0.605 [0.539, 0.668] | 72.5% [69.3, 75.7] | 6.0% [2.8, 10.0] |

S2 decodes identically to S1 on recorded sequences (experiments/s2.py), so its row equals S1. S1 tuned: match=0.35, LM weight=0.5; S4 tuned: LM weight=1.5, sigma x0.8. S1+ (adjacency-aware, hard-selected key) tuned: LM weight=1.5, sigma x1.0. Mean [95% bootstrap CI over the 20 phrases]. **H1 (S1 vs S4)**: CER reduction 56.5% (mean diff +0.356, 95% CI [+0.322, +0.388], Wilcoxon p=1.9e-06, n=20 phrases). **H1 vs the strongest hard-selection baseline (S1+ vs S4)**: CER reduction 48.1% (mean diff +0.254, 95% CI [+0.225, +0.282], Wilcoxon p=1.9e-06).

### Real noise: eye-landmark perception, iris-only features, across sittings (fit on other runs)

| System | CER ↓ | WER ↓ | Char. acc. ↑ | Sentence exact ↑ |
|---|---|---|---|---|
| S0 | 0.831 [0.820, 0.841] | 1.209 [1.135, 1.309] | 15.6% [14.5, 16.8] | 0.0% [0.0, 0.0] |
| S1 | 0.802 [0.791, 0.813] | 1.215 [1.137, 1.314] | 14.5% [13.6, 15.5] | 0.0% [0.0, 0.0] |
| S1P | 0.733 [0.715, 0.749] | 1.224 [1.134, 1.337] | 23.3% [21.7, 24.9] | 0.0% [0.0, 0.0] |
| S2 | 0.802 [0.791, 0.813] | 1.215 [1.137, 1.314] | 14.5% [13.6, 15.5] | 0.0% [0.0, 0.0] |
| S4 | 0.488 [0.455, 0.521] | 0.911 [0.847, 0.977] | 50.8% [47.4, 54.1] | 0.4% [0.0, 1.1] |

S2 decodes identically to S1 on recorded sequences (experiments/s2.py), so its row equals S1. S1 tuned: match=0.95, LM weight=2.0; S4 tuned: LM weight=2.0, sigma x0.6. S1+ (adjacency-aware, hard-selected key) tuned: LM weight=1.5, sigma x1.25. Mean [95% bootstrap CI over the 20 phrases]. **H1 (S1 vs S4)**: CER reduction 39.2% (mean diff +0.314, 95% CI [+0.285, +0.342], Wilcoxon p=1.9e-06, n=20 phrases). **H1 vs the strongest hard-selection baseline (S1+ vs S4)**: CER reduction 33.4% (mean diff +0.245, 95% CI [+0.216, +0.273], Wilcoxon p=1.9e-06).

### Real noise: gaze-CNN perception, across sittings (fit on other runs)

| System | CER ↓ | WER ↓ | Char. acc. ↑ | Sentence exact ↑ |
|---|---|---|---|---|
| S0 | 0.668 [0.657, 0.680] | 1.165 [1.092, 1.257] | 32.8% [31.6, 34.0] | 0.0% [0.0, 0.0] |
| S1 | 0.658 [0.647, 0.669] | 1.160 [1.083, 1.257] | 33.6% [32.4, 34.8] | 0.0% [0.0, 0.0] |
| S1P | 0.584 [0.570, 0.598] | 1.120 [1.038, 1.219] | 40.9% [39.5, 42.3] | 0.1% [0.0, 0.2] |
| S2 | 0.658 [0.647, 0.669] | 1.160 [1.083, 1.257] | 33.6% [32.4, 34.8] | 0.0% [0.0, 0.0] |
| S4 | 0.308 [0.283, 0.335] | 0.654 [0.592, 0.719] | 68.9% [66.2, 71.5] | 4.8% [2.5, 8.1] |

S2 decodes identically to S1 on recorded sequences (experiments/s2.py), so its row equals S1. S1 tuned: match=0.35, LM weight=0.5; S4 tuned: LM weight=1.0, sigma x0.8. S1+ (adjacency-aware, hard-selected key) tuned: LM weight=1.0, sigma x0.8. Mean [95% bootstrap CI over the 20 phrases]. **H1 (S1 vs S4)**: CER reduction 53.2% (mean diff +0.350, 95% CI [+0.323, +0.372], Wilcoxon p=1.9e-06, n=20 phrases). **H1 vs the strongest hard-selection baseline (S1+ vs S4)**: CER reduction 47.2% (mean diff +0.276, 95% CI [+0.257, +0.293], Wilcoxon p=1.9e-06).

### Noise sweep (iid Gaussian gaze noise, per-axis sigma in normalised canvas units; a key is 0.10 wide, 0.25 tall)

| sigma | S0 CER | S1 CER | S1+ CER | S4 CER | S1→S4 reduction | S1+→S4 reduction | Wilcoxon p (S1+ vs S4) |
|---|---|---|---|---|---|---|---|
| 0.04 | 0.161 | 0.132 | 0.036 | 0.019 | 85.7% | 46.9% | 0.00071 |
| 0.08 | 0.467 | 0.451 | 0.270 | 0.141 | 68.8% | 47.8% | 1.9e-06 |
| 0.12 | 0.641 | 0.631 | 0.517 | 0.320 | 49.3% | 38.1% | 1.9e-06 |
| 0.16 | 0.732 | 0.726 | 0.655 | 0.464 | 36.1% | 29.3% | 1.9e-06 |
| 0.2 | 0.781 | 0.772 | 0.735 | 0.555 | 28.1% | 24.5% | 1.9e-06 |

### Drift: 40-phrase sessions with the gaze bias growing linearly from 0 to the stated total (posture drift)

| drift (x, y) | System | session CER | last-10-phrase CER |
|---|---|---|---|
| drift_half (-0.07, -0.05) | S2 (no adaptation) | 0.416 [0.413, 0.419] | 0.654 [0.646, 0.662] |
| drift_half (-0.07, -0.05) | S3 (S2 + online bias) | 0.399 [0.374, 0.424] | 0.467 [0.428, 0.507] |
| drift_half (-0.07, -0.05) | S4 (static) | 0.128 [0.125, 0.132] | 0.234 [0.224, 0.244] |
| drift_half (-0.07, -0.05) | S4 + online bias | 0.131 [0.122, 0.140] | 0.129 [0.118, 0.142] |
| drift_measured (-0.14, -0.1) | S2 (no adaptation) | 0.613 [0.611, 0.616] | 0.844 [0.840, 0.848] |
| drift_measured (-0.14, -0.1) | S3 (S2 + online bias) | 0.516 [0.488, 0.543] | 0.675 [0.636, 0.713] |
| drift_measured (-0.14, -0.1) | S4 (static) | 0.348 [0.344, 0.352] | 0.687 [0.680, 0.694] |
| drift_measured (-0.14, -0.1) | S4 + online bias | 0.124 [0.114, 0.137] | 0.123 [0.104, 0.147] |
| drift_double (-0.28, -0.2) | S2 (no adaptation) | 0.744 [0.742, 0.747] | 0.890 [0.886, 0.893] |
| drift_double (-0.28, -0.2) | S3 (S2 + online bias) | 0.706 [0.692, 0.720] | 0.877 [0.871, 0.882] |
| drift_double (-0.28, -0.2) | S4 (static) | 0.601 [0.598, 0.604] | 0.905 [0.901, 0.909] |
| drift_double (-0.28, -0.2) | S4 + online bias | 0.245 [0.229, 0.265] | 0.554 [0.538, 0.572] |

| drift | H2: S2→S3, session CER | H2: S2→S3, last-10 CER | S4 static→adaptive, last-10 CER |
|---|---|---|---|
| drift_half | 4.2% (diff +0.017, CI [-0.008, +0.042], p=0.0045, n=100 sessions) | 28.6% (diff +0.187, CI [+0.148, +0.225], p=2.5e-12, n=100 sessions) | 44.7% (diff +0.104, CI [+0.090, +0.118], p=5.5e-16, n=100 sessions) |
| drift_measured | 15.9% (diff +0.098, CI [+0.070, +0.125], p=1.2e-08, n=100 sessions) | 20.0% (diff +0.169, CI [+0.131, +0.209], p=2.2e-12, n=100 sessions) | 82.1% (diff +0.564, CI [+0.538, +0.586], p=4.4e-18, n=100 sessions) |
| drift_double | 5.1% (diff +0.038, CI [+0.025, +0.052], p=5.5e-07, n=100 sessions) | 1.5% (diff +0.013, CI [+0.008, +0.018], p=1.6e-05, n=100 sessions) | 38.8% (diff +0.351, CI [+0.332, +0.367], p=4.4e-18, n=100 sessions) |
