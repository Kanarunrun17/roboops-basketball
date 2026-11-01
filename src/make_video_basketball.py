"""提出用の動画(1280x720, 等倍速)を作ります。

構成: タイトル → 人間のデモ → 人間らしい合成デモ → ACTのシュート(固定20 seed) → 結果
すべてseedとactionからのreplay / 再実行なので、同じcheckpointなら同じ動画になります。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720
GAME = 640
BACKGROUND = (24, 28, 38)
TEXT = (235, 238, 245)
ACCENT = (255, 170, 60)
GOOD, BAD = (90, 210, 120), (240, 90, 80)
DEFAULT_FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
DEFAULT_FONT_LIGHT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make the submission video.")
    parser.add_argument("--human", type=Path, default=Path("data/human"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/basketball.npz"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/dataset/basketball.json"))
    parser.add_argument("--eval-config", type=Path, default=Path("configs/eval_basketball.yaml"))
    parser.add_argument("--synthetic", type=int, default=4)
    parser.add_argument("--title", default="PAI最終課題_ruonruon")
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--font-light", default=DEFAULT_FONT_LIGHT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/video/PAI最終課題_ruonruon.mp4"))
    return parser.parse_args()


class Canvas:
    def __init__(self, font: str, font_light: str) -> None:
        self.font_path, self.light_path = font, font_light

    def font(self, size: int, light: bool = False) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(self.light_path if light else self.font_path, size)

    def card(self, lines: list[tuple[str, int, tuple[int, int, int], bool]]) -> np.ndarray:
        image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(image)
        heights = [size * 1.6 for _, size, _, _ in lines]
        y = (HEIGHT - sum(heights)) / 2
        for (text, size, color, light), height in zip(lines, heights, strict=True):
            font = self.font(size, light)
            width = draw.textlength(text, font=font)
            draw.text(((WIDTH - width) / 2, y), text, font=font, fill=color)
            y += height
        return np.asarray(image)

    def panel(self, game: np.ndarray, header: str, lines: list[str], status: str = "",
              status_color: tuple[int, int, int] = TEXT) -> np.ndarray:
        image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        image.paste(Image.fromarray(game).resize((GAME, GAME)), (24, (HEIGHT - GAME) // 2))
        draw = ImageDraw.Draw(image)
        x = GAME + 64
        draw.text((x, 70), header, font=self.font(34), fill=ACCENT)
        y = 140
        for line in lines:
            draw.text((x, y), line, font=self.font(24, light=True), fill=TEXT)
            y += 40
        if status:
            draw.text((x, HEIGHT - 150), status, font=self.font(40), fill=status_color)
        draw.text((WIDTH - 150, HEIGHT - 44), "1x speed", font=self.font(20, True), fill=(150, 155, 170))
        return np.asarray(image)


class FrameSink:
    """frameをメモリに溜めず、そのままffmpegへ書き出します。"""

    def __init__(self, path: Path, fps: int) -> None:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                         macro_block_size=8)
        self.count = 0

    def append(self, frame: np.ndarray) -> None:
        self.writer.append_data(frame)
        self.count += 1

    def extend(self, frames: list[np.ndarray]) -> None:
        for frame in frames:
            self.append(frame)

    def __iadd__(self, frames: list[np.ndarray]) -> FrameSink:
        self.extend(frames)
        return self

    def close(self) -> None:
        self.writer.close()


def hold(frame: np.ndarray, seconds: float, fps: int) -> list[np.ndarray]:
    return [frame] * int(seconds * fps)


def main() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    from roboops.basketball import FPS, WORLD_SIZE, BasketballEnv
    from roboops.config import load_yaml
    from roboops.policy import load_normalization, normalize

    args = parse_args()
    canvas = Canvas(args.font, args.font_light)
    env = BasketballEnv(render_size=GAME)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    style = report["human_style"]
    frames = FrameSink(args.output, FPS)

    frames += hold(canvas.card([
        ("AI × ロボティクス 最終課題", 34, ACCENT, False),  # noqa: RUF001
        ("人間のデモ3本から学ぶ", 56, TEXT, False),
        ("バスケットボール・シュートロボット", 56, TEXT, False),
        ("2D物理シミュレータ(pymunk) × LeRobot ACT × RoboOps", 26, TEXT, True),  # noqa: RUF001
        (args.title + "   |   等倍速", 24, (150, 155, 170), True),
    ]), 5, FPS)

    def replay(seed: int, actions: np.ndarray, header: str, lines: list[str]) -> None:
        env.reset(seed=seed)
        for action in actions:
            frames.append(canvas.panel(env.render(), header, lines))
            env.step(action)
        status, color = ("SCORE!", GOOD) if env.scored else ("MISS", BAD)
        frames.extend(hold(canvas.panel(env.render(), header, lines, status, color), 0.6, FPS))

    # 1. 人間のデモ
    frames += hold(canvas.card([
        ("① 人間がロボットを操作してデモを記録", 44, TEXT, False),
        ("トラックパッドで手先を動かし、スペースキーを離すとシュート", 28, TEXT, True),
        ("成功した3本だけを保存(seedとactionをGitで管理)", 28, TEXT, True),
    ]), 3.5, FPS)
    human_paths = [path for path in sorted(args.human.glob("episode_*.npz"))
                   if bool(np.load(path)["success"])]
    for number, path in enumerate(human_paths, 1):
        with np.load(path) as values:
            replay(int(values["seed"]), values["action"], f"人間のデモ {number}/{len(human_paths)}", [
                "操作: トラックパッド + スペースキー",
                "観測: 手先・ボール・リングの位置",
                "行動: 目標座標(x, y) + 握る/離す",
                "リングの位置は毎回ランダム",
            ])

    # 2. 人間らしい合成デモ
    frames += hold(canvas.card([
        ("② 人間の癖を測り、人間らしい合成デモを400本作る", 40, TEXT, False),
        (f"投げる角度 {np.degrees(style['launch_angle']):.0f}°  /  構え {style['windup_frames']:.0f} frame"
         f"  /  手ブレ {style['jitter']:.1f}px  /  フォロースルー {style['follow_through']:.0f}px", 26, ACCENT, True),
        ("この分布からサンプル + 手ブレ(OU過程) + リリース速度の誤差", 26, TEXT, True),
        (f"合成デモの成功率 {report['synthetic']['success_rate']:.0%} → 成功したものだけ学習に使う", 26, TEXT, True),
    ]), 4.5, FPS)
    with np.load(args.dataset) as values:
        source, episode = values["source"], values["episode_index"]
        for number, index in enumerate(np.unique(episode[source == 1])[: args.synthetic], 1):
            mask = episode == index
            replay(int(values["seed"][mask][0]), values["action"][mask],
                   f"合成デモ {number}/{args.synthetic}", [
                       "人間の癖を真似たエキスパート",
                       "放物線からリリース速度を逆算",
                       "人間らしいブレ・タイミング差を付与",
                       f"Dataset: {report['episodes']} episodes",
                   ])

    # 3. ACTのシュート
    evaluation = load_yaml(args.eval_config)["evaluation"]
    checkpoint = Path(evaluation["checkpoint"])
    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.eval()
    stats = load_normalization(checkpoint / "normalization.npz")
    episodes, seed0 = int(evaluation["episodes"]), int(evaluation["seed"])
    frames += hold(canvas.card([
        ("③ LeRobot ACT で学習 → AIが自分でシュート", 44, TEXT, False),
        ("学習に使っていない固定20 seed(リング位置)で評価", 28, TEXT, True),
        ("人の操作なし・すべてAIの推論", 28, ACCENT, True),
    ]), 3.5, FPS)
    successes = 0
    for number in range(episodes):
        observation, _ = env.reset(seed=seed0 + number)
        policy.reset()
        header = f"AI(ACT)のシュート {number + 1}/{episodes}"
        done = False
        while not done:
            lines = [f"成功 {successes} / {number}", "入力: 手先・ボール・リング・経過時間",
                     "出力: 目標座標 + 握る/離す", "temporal ensemble なし(毎step推論)"]
            frames.append(canvas.panel(env.render(), header, lines))
            batch = {
                "observation.state": torch.from_numpy(
                    normalize(observation["agent_pos"], stats, "observation.state")).float()[None],
                "observation.environment_state": torch.from_numpy(
                    normalize(observation["environment_state"], stats,
                              "observation.environment_state")).float()[None],
            }
            with torch.inference_mode():
                action = policy.select_action(batch)[0].numpy()
            action = action * stats["action"]["std"] + stats["action"]["mean"]
            action[:2] = np.clip(action[:2], 0.0, WORLD_SIZE)
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        successes += int(env.scored)
        status, color = ("SCORE!", GOOD) if env.scored else ("MISS", BAD)
        lines[0] = f"成功 {successes} / {number + 1}"
        frames += hold(canvas.panel(env.render(), header, lines, status, color), 0.5, FPS)

    # 4. 結果
    frames += hold(canvas.card([
        ("結果(固定20 seed)", 44, ACCENT, False),
        ("授業と同じACT(temporal ensemble)              成功率   0%", 28, TEXT, True),
        ("ensembleなし(毎step推論)                         成功率 20%", 28, TEXT, True),
        ("+ 経過時間を観測に追加(合成デモのみ)          成功率 70%", 28, TEXT, True),
        (f"+ 人間のデモ3本 + 人間らしい合成デモ          成功率 {successes / episodes:.0%}", 28, GOOD, False),
        ("Dataset生成 → 検証 → 学習 → 評価 を GitHub Actions で自動化", 24, TEXT, True),
        ("github.com/Kanarunrun17/roboops-basketball", 24, (150, 155, 170), True),
    ]), 8, FPS)

    frames.close()
    print(f"Video: {args.output} ({frames.count / FPS:.1f}s, AI success {successes}/{episodes})")


if __name__ == "__main__":
    main()
