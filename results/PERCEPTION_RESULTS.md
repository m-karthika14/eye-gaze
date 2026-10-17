# Perception results (one file: paper text, table, figure, and a prompt to regenerate them)

Everything numeric below comes from `results/perception.json` (made by `python -m experiments.perception_summary`) and the figure from
`python -m experiments.perception_figure` -> `results/figures/fig_perception.png`. Regenerate both after any new calibration run and
re-check the numbers in the text.

Status of the evidence: **two participants (participant 1 = the developer, six runs; participant 2, one run), seven calibration runs.** "Held out" means held-out trials (within
a run) or a held-out run (across runs). It does not mean held-out people, except for the single participant-2 run, which was predicted from participant 1's runs. State this in the paper.

---

## 1. Paper text (Results, Section 4.1)

### 4.1 Gaze perception accuracy

We first evaluated how accurately the measured gaze identifies the key a user is looking at, before any language-model decoding. Seven
calibration runs from two participants were analysed (participant 1: six runs on two days; participant 2: one run), with 19-32
fixation trials per run (29 on average) on keys of the 28-key on-screen keyboard. For each run we fitted two mappings from the camera
to keyboard coordinates: an eye-landmark regression (ridge regression on head-normalised eye and iris landmark features) and a gaze-CNN
mapping (an affine map of the CNN's absolute gaze angles). We report the key-hit rate, i.e. the share of trials in which the key nearest
to the predicted point is the target key (chance is 3.6% with 28 keys), and the root-mean-square error (RMSE) in normalised keyboard
coordinates, where a key is 0.10 wide and 0.25 tall.

**Within a sitting.** Evaluated leave-one-trial-out, the eye-landmark mapping reached a mean key-hit rate of 76% (range 47-96%) and a
mean RMSE of 0.11 (0.06-0.17), against 38% (19-57%) and 0.17 (0.13-0.24) for the gaze-CNN mapping (Table 1, Figure 1). The four most
recent runs, for which the model was trained on individual video frames instead of one averaged sample per trial, reached 88-96% key hit
(mean 93%) and an RMSE of 0.06-0.11 (mean 0.08), i.e. the nearest key was identified correctly in roughly nine of ten fixations, with a
typical error below one key width. This includes the single run of participant 2 (96%, RMSE 0.056). Because these runs also differ in
other respects (later in the day, same seat), we do not attribute the improvement to the training change alone.

**Across sittings and across people.** Accuracy dropped when a mapping fitted on the other runs was applied to a new one. The
eye-landmark mapping fell to a mean key-hit rate of 35% (0-63%) and an RMSE of 0.19 (0.12-0.31); the gaze-CNN mapping fell to 25%
(13-39%) and 0.22 (0.14-0.28). In one run the cross-run eye-landmark mapping identified no key correctly. For participant 2, whose run
was predicted by a mapping fitted only on participant 1's runs (i.e. a different person), the eye-landmark mapping identified the key in
4% of trials (RMSE 0.311) and the gaze-CNN mapping in 39% (RMSE 0.143). The mapping that was live during each run, which had been
fitted in an earlier session, identified the target key in only 18% of trials on average (3-31%). In a follow-up analysis of two
consecutive sittings of participant 1 about 50 minutes apart, 70% of the cross-sitting error was a single constant offset (RMSE 0.39
before and 0.14 after removing the mean offset; response gain about 1), which is consistent with a shift in head pose or seating between
sittings and not with a change in the eye-to-screen relation itself.

**Summary.** The eye-landmark mapping was more accurate than the gaze-CNN mapping within a sitting. Its accuracy is high when the
calibration and the use occur in the same sitting and the user's position is unchanged, and it degrades substantially across sittings
and, in the one case we could test, across people (the gaze-CNN mapping degraded less in that case). We therefore treat the
across-sitting figures as the realistic estimate for an unrecalibrated mapping, and the within-sitting figures as the accuracy available
after a short calibration in a fixed seating position.

**Limitations of this analysis.** Only two participants contributed data, and one of them contributed six of the seven runs, so the
results do not show how the mapping generalises to people in general; the across-run folds mix sittings from two different days and so
combine head-pose, lighting and time effects; and the number of trials per run is small (19-32), so individual run values are
imprecise. No significance tests are reported for this section.

### Table 1. Held-out accuracy per calibration run (key hit / RMSE)

| Run | Participant | Date | Trials | Eye-landmark, within sitting | Eye-landmark, across sittings | Gaze CNN, within sitting | Gaze CNN, across sittings | Live mapping at the time |
|---|---|---|---|---|---|---|---|---|
| R1 | 1 | 10 Oct | 32 | 47% / 0.154 | 38% / 0.141 | 22% / 0.186 | 25% / 0.222 | 25% |
| R2 | 1 | 10 Oct | 32 | 56% / 0.175 | 47% / 0.177 | 56% / 0.128 | 28% / 0.277 | 25% |
| R3 | 1 | 17 Oct | 32 | 56% / 0.142 | 44% / 0.145 | 19% / 0.242 | 22% / 0.249 | 28% |
| R4 | 1 | 17 Oct | 30 | 93% / 0.063 | 63% / 0.116 | 40% / 0.140 | 13% / 0.239 | 3% |
| R5 | 1 | 17 Oct | 19 | 95% / 0.109 | 0% / 0.310 | 47% / 0.171 | 21% / 0.244 | 5% |
| R6 | 1 | 17 Oct | 32 | 88% / 0.075 | 50% / 0.141 | 22% / 0.169 | 25% / 0.189 | 31% |
| R7 | 2 | 17 Oct | 28 | 96% / 0.056 | 4% / 0.311 | 57% / 0.129 | 39% / 0.143 | 7% |
| **Mean** | | | | **76% / 0.11** | **35% / 0.19** | **38% / 0.17** | **25% / 0.22** | **18%** |

Key hit = nearest key to the predicted point equals the target (chance 3.6%). RMSE in normalised keyboard units (key = 0.10 x 0.25).
Within sitting = leave-one-trial-out inside the run. Across sittings = fit on the other six runs, predict this run (for R7 this is a
different person; for R1-R6 the other runs include the same person). Runs R4-R7 were fitted on individual frames; R1-R3 on one averaged
sample per trial.

### Figure 1 (caption)

**Figure 1.** Gaze perception accuracy per calibration run. (a) Key-hit rate for the eye-landmark and gaze-CNN mappings, evaluated within
a sitting (leave-one-trial-out) and across sittings (fit on the other runs); the dotted line is chance (3.6%). (b) Position error (RMSE, in
keyboard units where one key is 0.10 wide) for the same four conditions; each dot is one run. Seven runs, two participants. File: `results/figures/fig_perception.png`.

---

## 2. Prompt to regenerate the table or the figure (paste to any assistant with code access)

```
You have results/perception.json from my gaze-typing project. Structure:
  per_run: list of {run, trials, as_recorded:{key_hit,rmse}, landmarks:{key_hit,rmse,...}, cnn:{key_hit,rmse,...}}
  cross_run: {lm_eye:{key_hit:[...],rmse:[...]}, cnn:{...}}   (one value per run, in per_run order)

Task: produce (1) a LaTeX table and (2) a two-panel figure in matplotlib for a research paper.

Table: one row per run (R1..R7) and a final Mean row; columns: Trials; eye-landmark within-sitting key-hit/RMSE;
eye-landmark across-sittings key-hit/RMSE; gaze-CNN within; gaze-CNN across. Key hit as a percentage (no decimals), RMSE with 3 decimals
(Mean row 2 decimals). booktabs style, caption "Held-out accuracy per calibration run", note under the table defining key hit
(nearest key = target, chance 3.6% with 28 keys), RMSE units (key = 0.10 x 0.25) and what within/across mean.

Figure: 10 x 3.6 inches, 200 dpi. Panel (a) grouped bars of key-hit rate per run, four groups: eye-landmark within, eye-landmark across,
gaze-CNN within, gaze-CNN across (dark/light blue, dark/light orange); y from 0 to 1; dotted chance line at 1/28; legend inside.
Panel (b) boxplot of RMSE with the individual runs overlaid as dots for the same four conditions (no fliers drawn twice).
Colour-blind safe, no title text inside the figure beyond "(a)" and "(b)" panel titles, font size >= 8 pt.

Rules: use only the numbers in the JSON, do not invent values or add significance stars, and print the mean and range of each of the
four conditions so I can check the text. State in the caption that the seven runs come from two participants.
```

---

## 3. Before you submit: checklist

- Re-run `perception_summary` and `perception_figure` after any new calibration, then update the numbers in section 1.
- Say clearly that this is **two participants** (one of them contributing six of seven runs); "calibration-free for every person" is not supported by this data.
- Report across-sitting results next to within-sitting ones, never within-sitting alone.
- R4-R6 use the frame-level fit and R1-R3 do not: say so, and do not credit the improvement to the method alone.
- Describe the app's "Accuracy Check" as a calibration.


---
---

# Section 4.2: Decoding on measured noise (H1)

Numbers from `results/tables.md` (regenerated 17 Oct on all seven calibration runs by `python -m experiments.run_results`). This is a
**trace-driven simulation**, not live typing: gaze errors are sampled from the held-out errors measured on the real runs (Section 4.1) and
fed to each decoder on 20 evaluation phrases. Statistics are over phrases, not over people.

## 4.2 Text (paste)

### 4.2 Decoding on measured gaze noise

We next asked whether decoding the gaze distribution jointly with a language model (S4) reduces typing errors compared with selecting a
hard key and correcting it afterwards (S0, S1), and compared with a stronger hard-selection baseline (S1+). S0 selects the nearest key.
S1 selects the nearest key and then lets a position-blind character language model correct it. S1+ also selects a hard key, but its
correction is adjacency-aware and uses a calibrated per-axis noise estimate, so that it uses as much positional information as a
hard-selection method can. S4 keeps the full gaze distribution over keys and decodes it jointly with a word lexicon and a character
language model. To make the comparison fair, the free parameters of S1, S1+ and S4 were tuned on phrases disjoint from the evaluation
phrases, and all systems were evaluated on the same 20 phrases and the same sampled gaze errors.

Gaze errors were not synthetic: for each condition we sampled errors from the held-out errors measured in Section 4.1. The conditions
were (i) the eye-landmark mapping within a sitting (the accuracy after a fresh calibration), (ii) the eye-landmark mapping across
sittings (a mapping fitted on other runs, i.e. no recalibration), in two feature variants, and (iii) the same two cases for the gaze-CNN
mapping. We report character error rate (CER), word error rate (WER), character accuracy and the share of sentences typed exactly,
with 95% bootstrap confidence intervals over the 20 phrases, and a paired Wilcoxon signed-rank test over phrases.

**Within a sitting (eye-landmark mapping).** S4 reached a CER of 0.081 [0.066, 0.097], against 0.226 for S0, 0.201 for S1 and 0.140 for
S1+ (Table 2). Relative to S1, S4 reduced CER by 59.9% (mean difference 0.121, 95% CI [0.108, 0.133], Wilcoxon p = 1.9e-06, n = 20
phrases); relative to the strongest hard-selection baseline, S1+, the reduction was 42.1% (0.059, [0.050, 0.067], p = 1.9e-06). S4 typed
38.9% of the sentences exactly, against 4.4% for S1 and 16.6% for S1+.

**Across sittings (no recalibration).** With the mapping fitted on other runs, all systems degraded, but S4 degraded least. Using the
eye-landmark mapping with eye-region features, CER was 0.258 [0.230, 0.288] for S4, 0.649 for S1 and 0.540 for S1+ (S4 vs S1: 60.3%
lower; S4 vs S1+: 52.3% lower, p = 1.9e-06). With iris-only features the CER was 0.281 for S4 against 0.653 for S1 and 0.530 for S1+
(56.9% and 46.9% lower). For the gaze-CNN mapping, whose gaze estimates are noisier, CER was 0.419 for S4 against 0.683 for S1 and 0.632
for S1+ (38.7% and 33.7% lower); the CNN mapping within a sitting gave 0.254 for S4 against 0.577 and 0.467 (56.0% and 45.7% lower).
In every condition S4 had the lowest CER and its confidence interval did not overlap with those of the other systems.

**Hard selection plus correction adds little.** S1 improved on S0 by at most about 0.03 CER in any condition, and S2 (S1 with an adaptive
keyboard) decodes identically to S1 on recorded gaze sequences, so we report it with S1. The largest gain among the hard-selection
methods came from S1+, which uses key adjacency and calibrated noise, but S4 improved on it in all conditions.

**Interpretation.** Because S4 weighs every key by the probability that the gaze belongs to it, instead of committing to one key
first, an error of one key width, which is typical of these gaze estimates (RMSE 0.06-0.19, Section 4.1), still leaves the intended key
in the candidate set, and the language model resolves between neighbours. The advantage is largest where the gaze is most accurate and
the language model has the most to work with (within a sitting), and it persists, with smaller relative gains, for the noisier CNN mapping.

**Limitations.** (1) The results are simulated typing on measured gaze errors, not a user study; they show what the decoders can do with
the noise we measured and need confirmation with participants. (2) The error pools come from two participants over seven runs (six from one of them), so they
describe mostly one person's noise. (3) p-values are over 20 phrases under the simulation and say that the decoders differ on these
phrases, not that the effect generalises to people; the confidence intervals are the more informative result. (4) S1, S1+ and S4 share the same
character n-gram language model in this simulation, so the comparison isolates how the gaze is used, not the quality of the language model;
the live S4 additionally uses a word lexicon, which was not part of this simulation.

### Table 2. CER by system and condition (mean [95% CI over 20 phrases])

| Condition | S0 | S1 | S1+ | S4 | S4 vs S1 | S4 vs S1+ |
|---|---|---|---|---|---|---|
| Eye-landmark, within sitting | 0.226 [0.219, 0.232] | 0.201 [0.195, 0.208] | 0.140 [0.128, 0.153] | **0.081** [0.066, 0.097] | -59.9% | -42.1% |
| Eye-landmark (eye-region), across sittings | 0.655 [0.644, 0.666] | 0.649 [0.637, 0.660] | 0.540 [0.525, 0.554] | **0.258** [0.230, 0.288] | -60.3% | -52.3% |
| Eye-landmark (iris-only), across sittings | 0.660 [0.649, 0.672] | 0.653 [0.642, 0.665] | 0.530 [0.513, 0.549] | **0.281** [0.254, 0.312] | -56.9% | -46.9% |
| Gaze CNN, within sitting | 0.587 [0.576, 0.599] | 0.577 [0.565, 0.589] | 0.467 [0.449, 0.486] | **0.254** [0.231, 0.277] | -56.0% | -45.7% |
| Gaze CNN, across sittings | 0.690 [0.681, 0.700] | 0.683 [0.674, 0.694] | 0.632 [0.621, 0.643] | **0.419** [0.395, 0.443] | -38.7% | -33.7% |

CER = character error rate, lower is better. S1+ is the adjacency-aware hard-selection baseline. Paired Wilcoxon p-values over the 20
phrases: S4 vs S1 and S4 vs S1+ both 1.9e-06 in all rows, which is the smallest value possible with
n = 20. Tuned parameters per condition are in `results/tables.md`.

### Figure 2 (caption)

**Figure 2.** Character error rate of S0, S1, S1+ and S4 on gaze errors measured from real calibration runs, within and across
sittings (error bars: 95% bootstrap CI over 20 phrases). Simulated typing; errors from two participants (six of seven runs from one). Source: `results/tables.md`.
(Existing figure: `results/figures/fig_real_conditions.png`.)

## Prompt to regenerate Table 2 / Figure 2

```
You have results/summary.json and results/tables.md from experiments.run_results (trace-driven simulation of a gaze-typing system).
Task: (1) a LaTeX booktabs table of CER by system (S0, S1, S1+ [called S1P in the files], S4) for the five "real noise" conditions,
with mean [95% CI] and two extra columns: relative CER reduction of S4 vs S1 and vs S1+; (2) a grouped bar chart (matplotlib, 8 x 3.5
in, 200 dpi) with the five conditions on the x axis, four bars per condition (S0 grey, S1 light blue, S1+ dark blue, S4 orange), y = CER
from 0 to 0.8, error bars = the 95% CI, a legend, and axis labels. Use only numbers from the files, do not invent values or stars. State
in the caption: simulated typing, errors measured from two participants, CIs over 20 phrases. Print the numbers used so I can check them.
```

## Notes

- These numbers replace the earlier 71% / 61% (computed when only two runs existed). The backups are `results/tables.before_6runs.md`
  and `results/summary.before_6runs.json`. The appendix of `PAPER_RESULTS_GUIDE.md` still shows the old tables; use this file.
- S2 = S1 in simulation by construction (the adaptive keyboard only acts live).
