import math

import numpy as np

from roboops.basketball import (
    ACTION_DIM,
    ENVIRONMENT_DIM,
    REACH_MAX,
    SHOULDER,
    STATE_DIM,
    BasketballEnv,
    clamp_to_workspace,
    required_release_speed,
)
from roboops.basketball_expert import HumanStyle, measure_human_style, run_expert_episode
from validate_basketball import validate_basketball


def test_reset_is_deterministic_and_has_schema() -> None:
    first, _ = BasketballEnv().reset(seed=3)
    second, _ = BasketballEnv().reset(seed=3)
    np.testing.assert_array_equal(first["environment_state"], second["environment_state"])
    assert first["agent_pos"].shape == (STATE_DIM,)
    assert first["environment_state"].shape == (ENVIRONMENT_DIM,)
    assert first["agent_pos"][2] == 1.0


def test_hand_stays_inside_arm_workspace() -> None:
    point = clamp_to_workspace(np.array([500.0, 500.0]))
    assert np.linalg.norm(point - SHOULDER) <= REACH_MAX + 1e-6


def test_required_speed_is_infinite_when_target_is_behind() -> None:
    assert math.isinf(required_release_speed(np.array([300.0, 100.0]), 0.9, np.array([200.0, 300.0])))


def test_ball_falls_after_release() -> None:
    env = BasketballEnv()
    observation, _ = env.reset(seed=0)
    hand = observation["agent_pos"][:2]
    observation, *_ = env.step(np.array([*hand, 0.0]))
    assert observation["agent_pos"][2] == 0.0
    for _ in range(5):
        observation, *_ = env.step(np.array([*hand, 0.0]))
    assert observation["environment_state"][3] < 0.0


def test_replaying_actions_reproduces_episode() -> None:
    env = BasketballEnv()
    episode = run_expert_episode(env, 7, HumanStyle(), np.random.default_rng(0))
    env.reset(seed=7)
    for state, action in zip(episode["agent_state"], episode["action"], strict=True):
        np.testing.assert_allclose(env._observation()["agent_pos"], state, atol=1e-4)
        env.step(action)
    assert episode["action"].shape[1] == ACTION_DIM


def test_expert_scores_more_often_than_not() -> None:
    env = BasketballEnv()
    rng = np.random.default_rng(1)
    results = [run_expert_episode(env, 100 + seed, HumanStyle(), rng)["success"] for seed in range(10)]
    assert sum(results) >= 5


def test_measured_style_follows_demo_angle() -> None:
    env = BasketballEnv()
    rng = np.random.default_rng(2)
    demos = [run_expert_episode(env, seed, HumanStyle(), rng) for seed in range(3)]
    style = measure_human_style([demo for demo in demos if demo["success"]])
    assert math.radians(35.0) <= style.launch_angle <= math.radians(75.0)


def test_validator_rejects_episode_without_release() -> None:
    frames = 6
    values = {
        "observation.state": np.tile([150.0, 200.0, 1.0, 0.0], (frames, 1)),
        "observation.environment_state": np.tile([150.0, 200.0, 0.0, 0.0, 400.0, 300.0], (frames, 1)),
        "action": np.tile([150.0, 200.0, 1.0], (frames, 1)),
        "episode_index": np.zeros(frames, dtype=int),
    }
    checks = validate_basketball(values)
    release = next(check for check in checks if check.label.startswith("each episode"))
    assert not release.passed
