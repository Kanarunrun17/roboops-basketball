"""バスケットボールDatasetの検査。PushTの共通検査にtask固有の検査を足します。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from roboops.basketball import ACTION_DIM, ENVIRONMENT_DIM, STATE_DIM, WORLD_SIZE
from roboops.data import (
    ACTION_KEY,
    AGENT_STATE_KEY,
    ENVIRONMENT_STATE_KEY,
    EPISODE_KEY,
    load_validation_data,
)
from validate_dataset import Check, validate

DEFAULT_DATA = Path("data/processed/basketball.npz")
EXPECTED_SHAPES = {
    AGENT_STATE_KEY: (STATE_DIM,),
    ENVIRONMENT_STATE_KEY: (ENVIRONMENT_DIM,),
    ACTION_KEY: (ACTION_DIM,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the basketball dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--min-human", type=int, default=0)
    return parser.parse_args()


def validate_basketball(values: dict[str, np.ndarray], min_human: int = 0) -> list[Check]:
    checks = validate(values, EXPECTED_SHAPES)
    if not all(check.passed for check in checks):
        return checks
    state = np.asarray(values[AGENT_STATE_KEY])
    action = np.asarray(values[ACTION_KEY])
    episodes = np.asarray(values[EPISODE_KEY]).reshape(-1)

    grip_values = np.unique(np.concatenate((state[:, 2], action[:, 2])))
    binary = set(grip_values.tolist()) <= {0.0, 1.0}
    checks.append(Check("grip / held are binary", binary, f"values={grip_values[:6].tolist()}"))

    targets_ok = bool(((action[:, :2] >= 0.0) & (action[:, :2] <= WORLD_SIZE)).all())
    checks.append(Check("action target inside the court", targets_ok, f"0 <= x, y <= {WORLD_SIZE}"))

    bad_release = []
    for episode in np.unique(episodes):
        mask = episodes == episode
        held, grip = state[mask, 2], action[mask, 2]
        releases = int(np.sum((held > 0.5) & (grip < 0.5)))
        if held[0] < 0.5 or releases != 1:
            bad_release.append(int(episode))
    checks.append(
        Check(
            "each episode starts holding and releases once",
            not bad_release,
            "all episodes" if not bad_release else f"bad episodes: {bad_release[:10]}",
        )
    )

    if "source" in values:
        source = np.asarray(values["source"]).reshape(-1)
        human = np.unique(episodes[source == 0]).size
        synthetic = np.unique(episodes[source == 1]).size
        checks.append(
            Check(
                "human demos included",
                human >= min_human,
                f"human={human}, synthetic={synthetic} (minimum human={min_human})",
            )
        )
    return checks


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data}\nRun: python src/generate_basketball.py")
    checks = validate_basketball(load_validation_data(args.data), args.min_human)
    print("## Basketball Dataset Validation")
    print(f"Data: {args.data}")
    print()
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'}: {check.label} — {check.detail}")
    passed = all(check.passed for check in checks)
    print()
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
