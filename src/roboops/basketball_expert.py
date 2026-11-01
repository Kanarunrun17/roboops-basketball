"""人間のデモの癖を真似てシュートする合成エキスパート。

人間のデモから「待ち時間・振りかぶり・投げる角度・加速時間・手ブレ・フォロースルー」を
測り(``measure_human_style``)、その分布からサンプルしたパラメータで1 episodeを作ります。
リリース速度だけは放物線から逆算するので、人間より安定して入ります。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np

from roboops.basketball import (
    FPS,
    HAND_FRAME_DECAY,
    HAND_GAIN,
    BasketballEnv,
    clamp_to_workspace,
    required_release_speed,
)

AIM_ABOVE_RIM = 6.0


@dataclass(frozen=True)
class HumanStyle:
    """1つの値は平均、``*_std`` は人間ごとのばらつき。"""

    idle_frames: float = 10.0
    idle_frames_std: float = 4.0
    windup_frames: float = 14.0
    windup_frames_std: float = 4.0
    windup_offset_x: float = -50.0
    windup_offset_x_std: float = 15.0
    windup_offset_y: float = -60.0
    windup_offset_y_std: float = 15.0
    throw_frames: float = 6.0
    throw_frames_std: float = 1.5
    launch_angle: float = math.radians(55.0)
    launch_angle_std: float = math.radians(5.0)
    jitter: float = 2.0
    follow_through: float = 40.0
    follow_through_std: float = 15.0
    release_speed_error: float = 0.012

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> HumanStyle:
        names = {field.name for field in fields(cls)}
        return cls(**{key: float(value) for key, value in values.items() if key in names})


def measure_human_style(
    episodes: list[dict[str, np.ndarray]],
    default: HumanStyle | None = None,
) -> HumanStyle:
    """人間のepisode(agent_state / environment_state / action)から癖を測ります。"""
    default = default or HumanStyle()
    rows: dict[str, list[float]] = {key: [] for key in ("idle", "windup", "dx", "dy", "throw",
                                                         "angle", "jitter", "follow")}
    for episode in episodes:
        state, environment, action = (
            episode["agent_state"], episode["environment_state"], episode["action"],
        )
        released = np.flatnonzero((state[:, 2] > 0.5) & (action[:, 2] < 0.5))
        if released.size == 0:
            continue
        release = int(released[0])
        speed = np.linalg.norm(environment[:, 2:4], axis=1)
        throw_start = release
        while throw_start > 0 and speed[throw_start - 1] > 150.0:
            throw_start -= 1
        moving = np.flatnonzero(speed[:throw_start] > 30.0)
        idle = int(moving[0]) if moving.size else throw_start
        rows["idle"].append(idle)
        rows["windup"].append(max(throw_start - idle, 1))
        windup_offset = state[throw_start, :2] - state[0, :2]
        rows["dx"].append(float(windup_offset[0]))
        rows["dy"].append(float(windup_offset[1]))
        rows["throw"].append(max(release - throw_start, 1))
        vx, vy = environment[release, 2:4]
        rows["angle"].append(math.atan2(float(vy), float(vx)))
        targets = action[:release, :2]
        if len(targets) >= 5:
            kernel = np.ones(5) / 5
            smooth = np.stack([np.convolve(targets[:, i], kernel, "same") for i in range(2)], 1)
            rows["jitter"].append(float(np.std((targets - smooth)[2:-2])))
        after = action[release:, :2]
        direction = np.array([math.cos(rows["angle"][-1]), math.sin(rows["angle"][-1])])
        if len(after):
            rows["follow"].append(float(np.max((after - state[release, :2]) @ direction)))

    def mean_std(key: str, mean: float, std: float, floor: float) -> tuple[float, float]:
        values = rows[key]
        if not values:
            return mean, std
        return float(np.mean(values)), max(float(np.std(values)), floor)

    idle, idle_std = mean_std("idle", default.idle_frames, default.idle_frames_std, 2.0)
    windup, windup_std = mean_std("windup", default.windup_frames, default.windup_frames_std, 2.0)
    dx, dx_std = mean_std("dx", default.windup_offset_x, default.windup_offset_x_std, 8.0)
    dy, dy_std = mean_std("dy", default.windup_offset_y, default.windup_offset_y_std, 8.0)
    throw, throw_std = mean_std("throw", default.throw_frames, default.throw_frames_std, 1.0)
    angle, angle_std = mean_std(
        "angle", default.launch_angle, default.launch_angle_std, math.radians(3.0)
    )
    follow, follow_std = mean_std(
        "follow", default.follow_through, default.follow_through_std, 8.0
    )
    jitter = float(np.mean(rows["jitter"])) if rows["jitter"] else default.jitter
    return HumanStyle(
        idle_frames=idle, idle_frames_std=idle_std,
        windup_frames=windup, windup_frames_std=windup_std,
        windup_offset_x=dx, windup_offset_x_std=dx_std,
        windup_offset_y=dy, windup_offset_y_std=dy_std,
        throw_frames=throw, throw_frames_std=throw_std,
        launch_angle=float(np.clip(angle, math.radians(35.0), math.radians(75.0))),
        launch_angle_std=angle_std,
        jitter=float(np.clip(jitter, 0.5, 6.0)),
        follow_through=follow, follow_through_std=follow_std,
        release_speed_error=default.release_speed_error,
    )


def _min_jerk(start: np.ndarray, end: np.ndarray, frames: int) -> list[np.ndarray]:
    phases = (np.arange(1, frames + 1) / frames)[:, None]
    blend = 10 * phases**3 - 15 * phases**4 + 6 * phases**5
    return list(start + (end - start) * blend)


def run_expert_episode(
    env: BasketballEnv,
    seed: int,
    style: HumanStyle,
    rng: np.random.Generator,
    record_frames: bool = False,
) -> dict[str, object]:
    """1 episodeを実行し、観測・action・成否(と任意でframe)を返します。"""
    observation, _ = env.reset(seed=seed)

    def sample(mean: float, std: float, low: float = -math.inf) -> float:
        return max(float(rng.normal(mean, std)), low)

    angle = sample(style.launch_angle, style.launch_angle_std)
    direction = np.array([math.cos(angle), math.sin(angle)])
    windup = clamp_to_workspace(
        env.hand
        + np.array([
            sample(style.windup_offset_x, style.windup_offset_x_std),
            sample(style.windup_offset_y, style.windup_offset_y_std),
        ])
    )
    idle_frames = round(sample(style.idle_frames, style.idle_frames_std, 0))
    windup_frames = round(sample(style.windup_frames, style.windup_frames_std, 4))
    throw_frames = round(sample(style.throw_frames, style.throw_frames_std, 3))
    follow = sample(style.follow_through, style.follow_through_std, 0)
    speed_error = float(rng.normal(1.0, style.release_speed_error))
    aim = env.hoop + np.array([0.0, AIM_ABOVE_RIM])

    # 手ブレ: 平滑化した乱数(Ornstein-Uhlenbeck過程)
    jitter = np.zeros(2)

    def shaky(point: np.ndarray) -> np.ndarray:
        nonlocal jitter
        jitter = 0.7 * jitter + rng.normal(0.0, style.jitter * 0.7, size=2)
        return np.asarray(point) + jitter

    plan: list[tuple[np.ndarray, float]] = []
    start = env.hand.copy()
    plan += [(shaky(start), 1.0) for _ in range(idle_frames)]
    plan += [(shaky(point), 1.0) for point in _min_jerk(start, windup, windup_frames)]

    agent_states, environment_states, actions, frames = [], [], [], []

    def act(action: np.ndarray) -> tuple[bool, dict]:
        nonlocal observation
        agent_states.append(observation["agent_pos"])
        environment_states.append(observation["environment_state"])
        actions.append(np.asarray(action, dtype=np.float32))
        if record_frames:
            frames.append(env.render())
        observation, _, terminated, truncated, info = env.step(action)
        return terminated or truncated, info

    done = False
    for target, grip in plan:
        done, info = act(np.array([*target, grip]))
        if done:
            break

    # 投げ: 発射方向へ加速し、放物線から逆算した速度に揃えてから離します。
    frame = 0
    while not done and env.held:
        hand = env.hand.copy()
        velocity = env.hand_velocity
        speed = float(np.linalg.norm(velocity))
        actual_angle = math.atan2(velocity[1], velocity[0]) if speed > 1.0 else angle
        need = required_release_speed(hand, actual_angle, aim) * speed_error
        if frame >= throw_frames and abs(speed / need - 1.0) < 0.02:
            release_target = hand + direction * follow * 0.5
            done, info = act(np.array([*release_target, 0.0]))
            break
        if frame > throw_frames + 6:
            done, info = act(np.array([*hand, 0.0]))
            break
        progress = min(1.0, (frame + 1) / throw_frames)
        predicted = hand + direction * required_release_speed(hand, angle, aim) * progress / FPS
        desired_speed = required_release_speed(predicted, angle, aim) * speed_error * progress
        # 目標を1 frame先へ置き、最後のsubstepで狙った速度になるようにします。
        target = hand + direction * desired_speed / (HAND_GAIN * HAND_FRAME_DECAY)
        done, info = act(np.array([*target, 1.0]))
        frame += 1

    # フォロースルー → 休止位置へ戻る
    if not done:
        release_hand = env.hand.copy()
        after = _min_jerk(release_hand, release_hand + direction * follow, 5)
        after += _min_jerk(after[-1], start, 15)
        index = 0
        while not done:
            point = after[min(index, len(after) - 1)]
            done, info = act(np.array([*shaky(point), 0.0]))
            index += 1

    return {
        "agent_state": np.asarray(agent_states, dtype=np.float32),
        "environment_state": np.asarray(environment_states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "success": bool(info["is_success"]),
        "frames": frames,
    }


__all__ = [
    "HumanStyle",
    "measure_human_style",
    "run_expert_episode",
]
