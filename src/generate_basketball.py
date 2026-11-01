"""人間のデモから癖を測り、人間らしい合成デモを足してDatasetを作ります。

1. ``data/human/episode_*.npz`` を読み、成功したデモから癖(HumanStyle)を測る
2. その分布からサンプルしたパラメータでエキスパートを動かし、成功episodeだけを残す
3. 人間のデモと合成デモを1つのNPZ(PushTと同じkey)へまとめ、IDとrevisionを記録する
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from roboops.basketball import BasketballEnv
from roboops.basketball_expert import HumanStyle, measure_human_style, run_expert_episode
from roboops.data import ACTION_KEY, AGENT_STATE_KEY, ENVIRONMENT_STATE_KEY, EPISODE_KEY, FRAME_KEY

DATASET_ID = "local/basketball-human-synthetic"
SYNTHETIC_SEED_OFFSET = 20_000
SOURCE_HUMAN, SOURCE_SYNTHETIC = 0, 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the basketball dataset.")
    parser.add_argument("--human", type=Path, default=Path("data/human"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/basketball.npz"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/dataset/basketball.json"))
    parser.add_argument("--episodes", type=int, default=400, help="successful synthetic episodes")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_human_episodes(directory: Path) -> list[dict[str, np.ndarray]]:
    episodes = []
    for path in sorted(directory.glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as values:
            episode = {key: np.asarray(values[key]) for key in values.files}
        episode["path"] = str(path)
        episodes.append(episode)
    return episodes


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    human = [episode for episode in load_human_episodes(args.human) if bool(episode["success"])]
    style = measure_human_style(human) if human else HumanStyle()
    print("## Basketball Dataset Generation")
    print(f"Human demos (success): {len(human)} from {args.human}")
    for key, value in style.to_dict().items():
        print(f"  {key:>22s} = {value:8.3f}")

    env = BasketballEnv()
    rng = np.random.default_rng(args.seed)
    episodes = [{**episode, "source": SOURCE_HUMAN} for episode in human]
    attempts = 0
    while len(episodes) - len(human) < args.episodes:
        result = run_expert_episode(env, SYNTHETIC_SEED_OFFSET + attempts, style, rng)
        attempts += 1
        if result["success"]:
            episodes.append({**result, "seed": SYNTHETIC_SEED_OFFSET + attempts - 1,
                             "source": SOURCE_SYNTHETIC})
    synthetic_success = args.episodes / max(attempts, 1)
    print(f"Synthetic: {args.episodes} successes / {attempts} attempts ({synthetic_success:.0%})")

    columns: dict[str, list[np.ndarray]] = {
        AGENT_STATE_KEY: [], ENVIRONMENT_STATE_KEY: [], ACTION_KEY: [],
        EPISODE_KEY: [], FRAME_KEY: [], "source": [], "seed": [],
    }
    for index, episode in enumerate(episodes):
        length = len(episode["action"])
        columns[AGENT_STATE_KEY].append(episode["agent_state"])
        columns[ENVIRONMENT_STATE_KEY].append(episode["environment_state"])
        columns[ACTION_KEY].append(episode["action"])
        columns[EPISODE_KEY].append(np.full(length, index))
        columns[FRAME_KEY].append(np.arange(length))
        columns["source"].append(np.full(length, episode["source"]))
        columns["seed"].append(np.full(length, int(episode["seed"])))
    arrays = {key: np.concatenate(value) for key, value in columns.items()}
    for key in (AGENT_STATE_KEY, ENVIRONMENT_STATE_KEY, ACTION_KEY):
        arrays[key] = arrays[key].astype(np.float32)

    # 内容から決まるhashをrevisionにして、同じ入力なら同じrevisionになるようにします。
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    revision = digest.hexdigest()[:16]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, dataset_id=DATASET_ID, dataset_revision=revision, **arrays)
    report = {
        "dataset": {"id": DATASET_ID, "revision": revision, "path": str(args.output)},
        "episodes": len(episodes),
        "frames": int(arrays[ACTION_KEY].shape[0]),
        "human_episodes": [episode["path"] for episode in human],
        "synthetic": {
            "episodes": args.episodes,
            "attempts": attempts,
            "success_rate": synthetic_success,
            "seed": args.seed,
            "env_seed_offset": SYNTHETIC_SEED_OFFSET,
        },
        "human_style": style.to_dict(),
        "seconds": time.perf_counter() - started_at,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset: {args.output} ({len(episodes)} episodes, {report['frames']} frames)")
    print(f"Revision: {revision}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
