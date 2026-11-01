"""保存したdemo(人間 / 合成)をseedとactionからreplayし、動画にします。

物理演算は決定的なので、同じseedとactionを与えると記録時と同じ動きになります。
replay後の観測が記録と一致するかも確認します。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from roboops.data import AGENT_STATE_KEY, ENVIRONMENT_STATE_KEY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay basketball demos into a video.")
    parser.add_argument("--human", type=Path, default=Path("data/human"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/basketball.npz"))
    parser.add_argument("--synthetic", type=int, default=4, help="synthetic episodes to include")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/replay-demos.mp4"))
    return parser.parse_args()


def replay(env, seed: int, actions: np.ndarray, caption: str) -> tuple[list, np.ndarray]:
    observation, _ = env.reset(seed=seed)
    frames, states = [], []
    for action in actions:
        states.append(np.concatenate((observation["agent_pos"], observation["environment_state"])))
        frames.append(env.render(caption=caption))
        observation, *_ = env.step(action)
    frames.extend([env.render(caption=caption)] * 15)
    return frames, np.asarray(states)


def main() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import imageio.v3 as iio

    from roboops.basketball import FPS, BasketballEnv

    args = parse_args()
    env = BasketballEnv(render_size=512)
    frames: list[np.ndarray] = []
    for path in sorted(args.human.glob("episode_*.npz")):
        with np.load(path) as values:
            if not bool(values["success"]):
                continue
            recorded = np.concatenate((values["agent_state"], values["environment_state"]), 1)
            clip, states = replay(env, int(values["seed"]), values["action"], f"HUMAN {path.stem}")
        error = float(np.abs(states - recorded).max())
        print(f"{path}: max replay error={error:.4f}")
        frames += clip
    if args.synthetic and args.dataset.exists():
        with np.load(args.dataset) as values:
            source, episode = values["source"], values["episode_index"]
            for index in np.unique(episode[source == 1])[: args.synthetic]:
                mask = episode == index
                recorded = np.concatenate(
                    (values[AGENT_STATE_KEY][mask], values[ENVIRONMENT_STATE_KEY][mask]), 1
                )
                seed = int(values["seed"][mask][0])
                clip, states = replay(
                    env, seed, values["action"][mask], f"SYNTHETIC (human-style) #{index}"
                )
                print(f"synthetic #{index}: max replay error={np.abs(states - recorded).max():.4f}")
                frames += clip
    args.output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output, np.stack(frames), fps=FPS, codec="libx264")
    print(f"Video: {args.output}")


if __name__ == "__main__":
    main()
