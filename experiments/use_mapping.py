"""Switch the live app between YOUR personal mapping and a generic one built from other people (the calibration-free trial).

    .venv/Scripts/python.exe -m experiments.use_mapping status
    .venv/Scripts/python.exe -m experiments.use_mapping build-generic --exclude P03   # fit on everyone EXCEPT P03 (leave-one-person-out)
    .venv/Scripts/python.exe -m experiments.use_mapping use generic                   # P03 now types with no calibration of their own
    .venv/Scripts/python.exe -m experiments.use_mapping use personal                  # back to the normal mapping

`use generic` only changes the `feature_model` line of config/gaze_mapping.yaml (a backup is made the first time) and remembers the personal
path in config/mapping_choice.json. Restart is not needed: the page reconnects and picks it up. Running the app's accuracy check (Calibrate) overwrites
`feature_model` with a fresh personal model, so run `use generic` AFTER any calibration you do not want to apply.

For a clean calibration-free trial of participant X: build the generic model excluding X (so nothing of X's is in it), switch to it, and let X type the
phrases in the Study bar. The one-point re-centre (C while looking at the middle of the keyboard) is allowed and is reported separately in loo_person.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

from experiments.sim_core import ROOT, load_runs_by_person
from vision.eye_features import FEATURE_SUBSETS
from vision.feature_model import fit_ridge

CONFIG = ROOT / "config" / "gaze_mapping.yaml"
CHOICE = ROOT / "config" / "mapping_choice.json"
GENERIC = "models/eye_feature_model_generic.npz"
yaml = YAML()
yaml.preserve_quotes = True


def read_feature_model(config: Path = CONFIG) -> str | None:
    with open(config, encoding="utf-8") as f:
        data = yaml.load(f)
    return data.get("feature_model")


def write_feature_model(value: str | None, config: Path = CONFIG) -> None:
    with open(config, encoding="utf-8") as f:
        data = yaml.load(f)
    if value is None:
        data.pop("feature_model", None)
    else:
        data["feature_model"] = value
    with open(config, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def build_generic(exclude: str | None, history_dir: Path | None = None, out: Path | None = None) -> tuple[int, int]:
    """Pool every person's trials (minus `exclude`) into one eye-region ridge model. Returns (people, trials)."""
    by_person = load_runs_by_person(history_dir) if history_dir else load_runs_by_person()
    people = [p for p in sorted(by_person) if p != exclude]
    rows = [x for p in people for run in by_person[p] for x in run]
    if len(people) < 1 or len(rows) < 12:
        raise SystemExit("not enough calibration data from other people to build a generic mapping")
    x = np.array([r["eye_feat"] for r in rows])
    y = np.array([[r["true_x"], r["true_y"]] for r in rows])
    model = fit_ridge(x, y, 10.0, columns=FEATURE_SUBSETS["eye_region"])
    model.save(out or (ROOT / GENERIC))
    return len(people), len(rows)


def use(which: str, config: Path = CONFIG, choice: Path = CHOICE) -> str | None:
    current = read_feature_model(config)
    state = json.loads(choice.read_text(encoding="utf-8")) if choice.exists() else {}
    if which == "generic":
        if current != GENERIC:
            state["personal"] = current
            choice.write_text(json.dumps(state, indent=2), encoding="utf-8")
            backup = config.with_name("gaze_mapping.backup_before_generic.yaml")
            if not backup.exists():
                shutil.copyfile(config, backup)
        write_feature_model(GENERIC, config)
        return GENERIC
    personal = state.get("personal")
    write_feature_model(personal, config)
    return personal


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    b = sub.add_parser("build-generic")
    b.add_argument("--exclude", default=None, help="participant id to leave out (leave-one-person-out)")
    u = sub.add_parser("use")
    u.add_argument("which", choices=("generic", "personal"))
    a = ap.parse_args()
    if a.cmd == "status":
        print("active feature_model:", read_feature_model())
        print("generic model exists:", (ROOT / GENERIC).exists())
        print("remembered personal model:", json.loads(CHOICE.read_text())["personal"] if CHOICE.exists() else None)
        by_person = load_runs_by_person()
        print("people with usable calibration data:", {p: len(r) for p, r in by_person.items()})
    elif a.cmd == "build-generic":
        people, trials = build_generic(a.exclude)
        print(f"built {GENERIC} from {people} other person(s), {trials} trials (excluded: {a.exclude})")
    else:
        print("feature_model is now:", use(a.which))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
