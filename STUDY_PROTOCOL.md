# Study protocol: calibration-free test + typing study (for tomorrow)

For: you, running participants. Everything below uses tools that exist and are tested. Results appear as soon as data exists, but **nobody can produce them before the participants have been run**; this protocol is how to get them in one day.

## 0. What the study answers

| Question | Data collected | Table you get |
|---|---|---|
| **Does the system work with NO calibration of the user** (mapping learned from other people only)? | each person's calibration run (Phase A) + a typing block with a generic mapping (Phase B1) | `results/lopo.md` (accuracy), `results/study.md` (typing) |
| How much better is it after a short personal accuracy check? | Phase B2 | same |
| Which system types best: S0, S1, S3, S4? (S2 optional) | Phase B3, counterbalanced | `results/study.md` |

The accuracy check in the app (Calibrate) **is a calibration**: it fits and saves the user's mapping. Tell participants exactly that (consent), and describe it as a calibration in the paper. Only the typing block with the generic mapping is the calibration-free condition.

## 1. Before anyone arrives (15 minutes)

1. **Back up your own setup** (the study overwrites it): `copy models\eye_feature_model.npz models\eye_feature_model.MINE.npz` and `copy config\gaze_mapping.yaml config\gaze_mapping.MINE.yaml`. (They are also in git.)
2. **Mark the seat position**: fix the distance from the screen (e.g. 60 cm, tape on the desk), the laptop angle, and the camera height. Same room, same lighting for everyone. Write the numbers down; they go in the paper.
3. Browser **F11 or a fixed window**, then do not resize it between participants (the calibration is tied to the keyboard size; the app warns if it differs).
4. Start the backend and the frontend as usual. In the app press **🧪 Study** once to check the phrase prompt appears, then turn it off.
5. Decide the number of people. **Fewer than 3 gives no calibration-free result** (the generic mapping would be one person's). Aim for 8+; 5 is a pilot. Your own earlier runs count as one extra person ("legacy") in the generic mapping, but only if their file has eye features (the two from 10 October do).
6. Give everyone an ID: P01, P02, ... Keep a sheet: ID, glasses (y/n), eye colour, notes.

## 2. Phase A: everyone's calibration run (about 3 minutes each)

For each person, at the marked seat: **Calibrate** (the screen says "Accuracy Check"), type the ID in the **Participant ID** box, **Start accuracy check**, look at each highlighted key until it fills. It saves `data/calibration_history/<time>_<ID>.jsonl`. Do **not** let people see the result screen long enough to turn it into a game; just press Done.

After the last person: `python -m experiments.loo_person` gives the calibration-free accuracy table immediately (needs 3+ people). This alone is already a result: *how good is a mapping learned from other people, applied to a new person with no calibration?*

## 3. Phase B: per participant (about 20 minutes)

Sequence (the generic block must come **before** their own fresh accuracy check, otherwise it is not calibration-free):

**B1. Calibration-free block (about 4 minutes)**
1. `python -m experiments.use_mapping build-generic --exclude P03` (use this participant's ID; nothing of theirs is in the model)
2. `python -m experiments.use_mapping use generic`, then refresh the page.
3. Turn on **🧪 Study**, enter the ID, choose **S4** (and then S0 if time allows). The person may press **C once while looking at the middle of the keyboard** (a one-point re-centre; say so in the paper) but nothing else.
4. They type phrases 1-4: press **Start**, type (blink / Space-Enter selection), press **Done ✓**. The app logs typed text, time, selections, backspaces, and moves on to the next phrase by itself.

**B2. Personal accuracy check (about 3 minutes)**: Calibrate with their ID again (fresh, same sitting). This overwrites the mapping with their own.

**B3. Typing blocks, counterbalanced (about 12 minutes)**: S0, S1, S3, S4 (add S2 if you want the adaptive keyboard), 4 phrases per system (phrases 5-20 in order: the 20-phrase list is used exactly once per person: 4 in B1, 16 in B3). Order the systems by Latin square so each system appears in each position, and therefore with each group of phrases, equally often:

| participant (mod 4) | order |
|---|---|
| 1 | S0, S1, S3, S4 |
| 2 | S1, S4, S0, S3 |
| 3 | S3, S0, S4, S1 |
| 4 | S4, S3, S1, S0 |

Use the dashboard **S0 S1 S2 S3 S4** switch between blocks (the phrase prompt keeps advancing; press ◀/▶ if you need to move it). Always the same selection method for everyone (recommend **Space/Enter** for able-bodied participants, so that selection errors are not mixed with gaze errors; blink if you want to test the hands-free case).

Between participants: `python -m experiments.use_mapping use personal` is not needed (the accuracy check wrote their model); the next participant's B1 starts with `build-generic --exclude <next ID>`.

## 4. After the last participant (one command each, minutes)

```
python -m experiments.use_mapping use personal                 # restore, then run YOUR OWN accuracy check once, or copy back the .MINE files
python -m experiments.loo_person                                # calibration-free accuracy across people  -> results/lopo.md
python -m experiments.study_summary                             # what people typed, per system          -> results/study.md
python -m experiments.perception_summary                        # per-run and across-sitting accuracy     -> results/perception.md
python -m experiments.run_results                               # (about 30 minutes) simulation now also uses the across-people error pools
```

Then paste numbers into the paper from `results/` only (see `PAPER_RESULTS_GUIDE.md`, sections 4-5, for how to write each paragraph).

## 5. What to expect, so nothing is a surprise

- **Calibration-free accuracy across people is likely much lower than within-person accuracy.** In your own two runs (same person, minutes apart) the key-hit rate fell from 69-84% to 25-47%; across people it is commonly worse. With 40 keys, the generic mapping may be too inaccurate for useful typing, and the S4 decoder (which tolerates noise) is what recovers part of it. Report whatever comes out, with the worst person.
- **The one-point re-centre may or may not help** (it adds the noise of one trial); `loo_person.py` reports both.
- **Few participants = wide intervals.** Call it a pilot, report CIs, and do not claim significance you cannot support.

## 6. If you can only do part of it

| You have | What you can report |
|---|---|
| Fewer than 3 people | The calibrated typing study (Phase B2-B3) only, as a pilot; no calibration-free claim. |
| 3-5 people | Calibration-free accuracy (loo_person) with wide CIs + pilot typing results. |
| 8+ people | Both, with usable intervals. |

## 7. Ethics and data

- Get informed consent that says what is recorded: **no video is stored**, only numeric eye-landmark features, gaze angles, calibration positions and typed text, tagged with the participant ID.
- Describe the accuracy check as what it is: a calibration.
- Check what your institution requires before collecting data from people.
