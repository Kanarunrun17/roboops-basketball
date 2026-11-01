"""マウスでロボットアームを操作し、人間のシュートデモを記録します(Mac本体で実行)。

操作(トラックパッドはカーソル移動だけ。握る/離すはスペースキーか左ボタン):
  スペースを押す      記録開始(押している間ボールを握り、手先がカーソルへ追従)
  カーソルを動かす    振りかぶって、リング方向へ素早く動かす
  スペースを離す      ボールを離す(シュート)
  S / R / N           保存して次へ / 同じリングでやり直し / 保存せず次のリングへ
  Q / Esc             終了

保存先: data/human/episode_XXX.npz(seedとactionを保存するのでreplayで再現できます)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame

from roboops.basketball import FPS, WORLD_SIZE, BasketballEnv

HUMAN_SEED_OFFSET = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record human basketball demos.")
    parser.add_argument("--output", type=Path, default=Path("data/human"))
    parser.add_argument("--size", type=int, default=640, help="window size in pixels")
    parser.add_argument("--speed", type=float, default=1.0, help="0.5 = slow motion")
    return parser.parse_args()


def next_episode_index(output: Path) -> int:
    existing = sorted(output.glob("episode_*.npz"))
    return int(existing[-1].stem.split("_")[1]) + 1 if existing else 0


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pygame.init()
    screen = pygame.display.set_mode((args.size, args.size))
    pygame.display.set_caption("Basketball teleop")
    clock = pygame.time.Clock()
    scale = args.size / WORLD_SIZE
    env = BasketballEnv(render_size=args.size)

    episode = next_episode_index(args.output)
    attempt = episode
    seed = HUMAN_SEED_OFFSET + attempt
    observation, _ = env.reset(seed=seed)
    mode = "ready"
    buffer: dict[str, list[np.ndarray]] = {}
    success = False
    print(f"episode {episode:03d} (seed {seed}): スペースを押して開始")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif mode == "done" and event.key in (pygame.K_s, pygame.K_r, pygame.K_n):
                    if event.key == pygame.K_s:
                        path = args.output / f"episode_{episode:03d}.npz"
                        np.savez(
                            path,
                            seed=seed,
                            success=success,
                            fps=FPS,
                            **{key: np.asarray(value, dtype=np.float32)
                               for key, value in buffer.items()},
                        )
                        print(f"  saved {path} ({len(buffer['action'])} frames)")
                        episode += 1
                    if event.key != pygame.K_r:
                        attempt += 1
                        seed = HUMAN_SEED_OFFSET + attempt
                    observation, _ = env.reset(seed=seed)
                    mode = "ready"
                    print(f"episode {episode:03d} (seed {seed}): スペースを押して開始")
            elif mode == "ready" and (
                (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE)
                or (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1)
            ):
                mode = "recording"
                buffer = {"agent_state": [], "environment_state": [], "action": []}

        mouse_x, mouse_y = pygame.mouse.get_pos()
        target = np.array([mouse_x / scale, WORLD_SIZE - mouse_y / scale])
        if mode == "recording":
            pressed = pygame.key.get_pressed()[pygame.K_SPACE] or pygame.mouse.get_pressed()[0]
            grip = 1.0 if pressed else 0.0
            action = np.array([*target, grip])
            buffer["agent_state"].append(observation["agent_pos"])
            buffer["environment_state"].append(observation["environment_state"])
            buffer["action"].append(action)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                mode = "done"
                success = bool(info["is_success"])
                print(f"  {'SCORE!' if success else 'miss'}  S:保存 R:やり直し N:次へ")

        captions = {
            "ready": "press & hold SPACE to start",
            "recording": "swing cursor, release SPACE to shoot",
            "done": "S: save   R: retry   N: next   Q: quit",
        }
        frame = env.render(caption=f"ep {episode:03d} | {captions[mode]}")
        screen.blit(pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), (0, 0))
        pygame.draw.circle(screen, (200, 40, 160), (mouse_x, mouse_y), 6, 2)
        pygame.display.flip()
        clock.tick(FPS * args.speed)

    pygame.quit()


if __name__ == "__main__":
    main()
