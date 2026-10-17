# Calibration-free test: leave-one-person-out

3 people. Each person is scored with a mapping fitted only on the other people (no calibration of their own). Key hit chance with 40 keys is about 2.5%. RMSE in normalised canvas units (key = 0.10 wide, 0.25 tall).

| person | eye-landmark, zero-shot | eye-landmark, + one-point re-centre | eye-landmark (all features), zero-shot | gaze CNN, zero-shot | gaze CNN, + one-point re-centre |
|---|---|---|---|---|---|
| 1 | 23% / 0.218 | 36% / 0.347 | 26% / 0.238 | 12% / 0.265 | 15% / 0.222 |
| 2 | 4% / 0.311 | 33% / 0.175 | 7% / 0.298 | 39% / 0.143 | 30% / 0.160 |
| legacy | 29% / 0.363 | 32% / 0.335 | 27% / 0.418 | 21% / 0.285 | 13% / 0.261 |

| summary over people (key hit / RMSE) | mean [95% CI] key hit | median | worst person | mean RMSE |
|---|---|---|---|---|
| eye-landmark, zero-shot | 19% [4, 29] | 23% | 4% (2) | 0.297 |
| eye-landmark, + one-point re-centre | 34% [32, 36] | 33% | 32% (legacy) | 0.286 |
| eye-landmark (all features), zero-shot | 20% [7, 27] | 26% | 7% (2) | 0.318 |
| gaze CNN, zero-shot | 24% [12, 39] | 21% | 12% (1) | 0.231 |
| gaze CNN, + one-point re-centre | 19% [13, 30] | 15% | 13% (legacy) | 0.215 |

Report the worst person as well as the mean: a calibration-free claim is only as good as its worst user. With few people the intervals are wide; say how many people there were.
