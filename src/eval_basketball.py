"""学習したACTを固定seedのバスケットボールtaskで評価し、成功率と動画を記録します。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from roboops.config import load_yaml
from roboops.policy import load_normalization, normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ACT in the basketball task.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_basketball.yaml"))
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    config = load_yaml(args.config)
    evaluation = config["evaluation"]
    if bool(evaluation["headless"]):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from lerobot.policies.act.modeling_act import ACTPolicy

    from roboops.basketball import FPS, WORLD_SIZE, BasketballEnv

    checkpoint_dir = Path(evaluation["checkpoint"])
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_dir}. "
            "Run: python src/train.py --config configs/train_basketball.yaml"
        )
    policy = ACTPolicy.from_pretrained(checkpoint_dir)
    policy.eval()
    stats = load_normalization(checkpoint_dir / "normalization.npz")
    training_metrics = json.loads(
        (checkpoint_dir / "training_metrics.json").read_text(encoding="utf-8")
    )

    episodes = int(evaluation["episodes"])
    max_steps = int(evaluation["max_steps"])
    seed = int(evaluation["seed"])
    save_video = args.save_video or bool(evaluation["save_video"])
    video_episodes = int(evaluation.get("video_episodes", 1))
    env = BasketballEnv(render_size=512, max_steps=max_steps)
    frames: list[np.ndarray] = []
    episode_results: list[dict[str, Any]] = []

    print("## Official LeRobot ACT Basketball Evaluation")
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        policy.reset()
        record = save_video and episode < video_episodes
        caption = f"ACT policy | episode {episode + 1}"
        info: dict[str, Any] = {"is_success": False, "release_step": None}
        steps = 0
        for _ in range(max_steps):
            if record:
                frames.append(env.render(caption=caption))
            batch = {
                "observation.state": torch.from_numpy(
                    normalize(observation["agent_pos"], stats, "observation.state")
                ).float().unsqueeze(0),
                "observation.environment_state": torch.from_numpy(
                    normalize(
                        observation["environment_state"], stats, "observation.environment_state"
                    )
                ).float().unsqueeze(0),
            }
            with torch.inference_mode():
                normalized_action = policy.select_action(batch)[0].cpu().numpy()
            action = normalized_action * stats["action"]["std"] + stats["action"]["mean"]
            action[:2] = np.clip(action[:2], 0.0, WORLD_SIZE)
            observation, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        if record:
            frames.extend([env.render(caption=caption)] * (FPS // 2))
        success = bool(info["is_success"])
        episode_results.append(
            {
                "episode": episode + 1,
                "seed": seed + episode,
                "hoop": env.hoop.round(1).tolist(),
                "steps": steps,
                "release_step": info["release_step"],
                "success": success,
            }
        )
        print(
            f"Episode {episode + 1}/{episodes} | hoop={env.hoop.round(0).tolist()} "
            f"| release={info['release_step']} | success={'yes' if success else 'no'}"
        )

    if save_video and frames:
        import imageio.v3 as iio

        video_path = Path(evaluation["video_path"])
        video_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(video_path, np.stack(frames), fps=FPS, codec="libx264")
        print(f"Video: {video_path}")

    success_rate = float(np.mean([item["success"] for item in episode_results]))
    release_rate = float(np.mean([item["release_step"] is not None for item in episode_results]))
    min_success_rate = float(evaluation["min_success_rate"])
    passed = success_rate >= min_success_rate
    elapsed_seconds = time.perf_counter() - started_at
    metrics = {
        "result": "PASS" if passed else "FAIL",
        "architecture": "lerobot_act",
        "environment": "roboops.basketball.BasketballEnv",
        "checkpoint": str(checkpoint_dir),
        "dataset": training_metrics["dataset"],
        "model": training_metrics["model"],
        "versions": {
            "lerobot": importlib.metadata.version("lerobot"),
            "pymunk": importlib.metadata.version("pymunk"),
        },
        "episodes": episodes,
        "seed": seed,
        "success_rate": success_rate,
        "release_rate": release_rate,
        "elapsed_seconds": elapsed_seconds,
        "thresholds": {"min_success_rate": min_success_rate},
        "episode_results": episode_results,
    }
    metrics_path = Path(config["output"]["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Release rate: {release_rate:.0%}")
    print(f"Success rate: {success_rate:.0%}")
    print(f"Evaluation: {elapsed_seconds:.1f}s")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
