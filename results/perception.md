# Perception: held-out accuracy per calibration run

| run | trials | eye-landmark: key hit / RMSE (subset) | gaze CNN: key hit / RMSE | live mapping during run: key hit / RMSE |
|---|---|---|---|---|
| 20261010_144100_run | 32 | 47% / 0.154 (full) | 22% / 0.186 | 25% / 0.226 |
| 20261010_144635 | 32 | 56% / 0.175 (full) | 56% / 0.128 | 25% / 0.263 |
| 20261017_193829 | 32 | 56% / 0.142 (full) | 19% / 0.242 | 28% / 0.368 |
| 20261017_202853_1 | 30 | 93% / 0.063 (eye_region) | 40% / 0.140 | 3% / 0.376 |
| 20261017_203528_1 | 19 | 95% / 0.109 (eye_region) | 47% / 0.171 | 5% / 0.478 |
| 20261017_210920_1 | 32 | 88% / 0.075 (full) | 22% / 0.169 | 31% / 0.236 |
| 20261017_222602_2 | 28 | 96% / 0.056 (eye_region) | 57% / 0.129 | 7% / 0.303 |

## Across sittings: fit on the other run(s), predict this one

| configuration | key hit (per held-out run) | RMSE (per held-out run) |
|---|---|---|
| eye-landmark, eye-region features, alpha 10 | 38%, 47%, 44%, 63%, 0%, 50%, 4% | 0.141, 0.177, 0.145, 0.116, 0.310, 0.141, 0.311 |
| eye-landmark, iris-only features, alpha 10 | 22%, 34%, 19%, 43%, 5%, 50%, 18% | 0.217, 0.232, 0.303, 0.098, 0.401, 0.138, 0.157 |
| gaze-CNN absolute-angle affine | 25%, 28%, 22%, 13%, 21%, 25%, 39% | 0.222, 0.277, 0.249, 0.239, 0.244, 0.189, 0.143 |

Compare with the within-run numbers above: a mapping fitted in one sitting loses a large part of its accuracy in the next (head position, lighting and seating differ). The iris-only configuration generalised worst in this data.

Key hit = nearest key to the predicted point equals the target key (chance with 40 keys is about 2.5%). RMSE is in normalised canvas units (a key is 0.10 wide and 0.25 tall). Held-out = leave-one-trial-out within the run.
