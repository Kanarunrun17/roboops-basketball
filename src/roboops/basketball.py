"""2Dバスケットボールのシュートtask。gym-pushtと同じpymunkで物理演算します。

座標系はpymunkに合わせてy軸が上向きの512x512です。
- Observation: ``agent_pos`` = ``[hand_x, hand_y, held, elapsed_seconds]``、
  ``environment_state`` = ``[ball_x, ball_y, ball_vx, ball_vy, hoop_x, hoop_y]``
- Action: ``[target_x, target_y, grip]``。手先は目標座標へ一次遅れで追従し、
  ``grip < 0.5`` になった瞬間にボールを離します(一度離したら再び掴めません)。
- Success: リリース後のボールが上からリングを通過すること。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pymunk

WORLD_SIZE = 512.0
FPS = 30
SUBSTEPS = 4
DT = 1.0 / (FPS * SUBSTEPS)
GRAVITY = -900.0
FLOOR_Y = 24.0
SHOULDER = np.array([96.0, 110.0])
LINKS = (120.0, 110.0)
REACH_MIN = 40.0
REACH_MAX = LINKS[0] + LINKS[1] - 4.0
HAND_GAIN = 18.0
HAND_MAX_SPEED = 2000.0
# 一定の目標を与えたとき、frame最後のsubstepの速度は初速のこの倍率になります。
HAND_FRAME_DECAY = (1.0 - HAND_GAIN * DT) ** (SUBSTEPS - 1)
REST_HAND = SHOULDER + np.array([60.0, 70.0])
BALL_RADIUS = 10.0
RIM_HALF_WIDTH = 34.0
HOOP_X_RANGE = (340.0, 430.0)
HOOP_Y_RANGE = (240.0, 360.0)
MAX_STEPS = 150
SCORE_LINGER_STEPS = 18
MISS_LINGER_STEPS = 8

STATE_DIM = 4
ENVIRONMENT_DIM = 6
ACTION_DIM = 3


def clamp_to_workspace(point: np.ndarray) -> np.ndarray:
    """2関節アームが届く範囲と床より上に手先を収めます。"""
    offset = np.asarray(point, dtype=np.float64) - SHOULDER
    distance = float(np.linalg.norm(offset))
    if distance < 1e-6:
        offset, distance = np.array([REACH_MIN, 0.0]), REACH_MIN
    clamped = SHOULDER + offset / distance * np.clip(distance, REACH_MIN, REACH_MAX)
    clamped[1] = max(clamped[1], FLOOR_Y + 20.0)
    return clamped


def elbow_position(hand: np.ndarray) -> np.ndarray:
    """描画用の逆運動学。肘が下になる解を返します。"""
    offset = hand - SHOULDER
    distance = float(np.clip(np.linalg.norm(offset), 1e-6, LINKS[0] + LINKS[1]))
    base_angle = math.atan2(offset[1], offset[0])
    cos_alpha = (LINKS[0] ** 2 + distance**2 - LINKS[1] ** 2) / (2 * LINKS[0] * distance)
    alpha = math.acos(float(np.clip(cos_alpha, -1.0, 1.0)))
    angle = base_angle - alpha
    return SHOULDER + LINKS[0] * np.array([math.cos(angle), math.sin(angle)])


def required_release_speed(
    position: np.ndarray,
    angle: float,
    aim: np.ndarray,
) -> float:
    """位置と発射角からaimを通る放物線の初速を返します。届かなければinf。"""
    dx = float(aim[0] - position[0])
    dy = float(aim[1] - position[1])
    cos_angle = math.cos(angle)
    denominator = 2.0 * cos_angle**2 * (dx * math.tan(angle) - dy)
    if dx <= 0.0 or cos_angle <= 1e-6 or denominator <= 0.0:
        return math.inf
    return math.sqrt(-GRAVITY * dx**2 / denominator)


class BasketballEnv:
    """Gymnasiumに似たAPIを持つ軽量なシュートtask。"""

    def __init__(self, render_size: int = 512, max_steps: int = MAX_STEPS) -> None:
        self.render_size = render_size
        self.max_steps = max_steps
        self._surface = None
        self._font = None
        self.reset(seed=0)

    # ------------------------------------------------------------------ setup
    def _build_space(self) -> None:
        space = pymunk.Space()
        space.gravity = (0.0, GRAVITY)
        static = space.static_body
        floor = pymunk.Segment(static, (-100, FLOOR_Y), (WORLD_SIZE + 100, FLOOR_Y), 2)
        floor.elasticity, floor.friction = 0.55, 0.8
        hx, hy = self.hoop
        board_x = hx + RIM_HALF_WIDTH + 8.0
        board = pymunk.Segment(static, (board_x, hy - 12), (board_x, hy + 90), 3)
        board.elasticity, board.friction = 0.6, 0.5
        rims = [
            pymunk.Circle(static, 3.0, (hx - RIM_HALF_WIDTH - 3.0, hy)),
            pymunk.Circle(static, 3.0, (hx + RIM_HALF_WIDTH + 3.0, hy)),
            pymunk.Segment(static, (hx + RIM_HALF_WIDTH + 3.0, hy), (board_x, hy), 2),
        ]
        for rim in rims:
            rim.elasticity, rim.friction = 0.5, 0.6
        space.add(floor, board, *rims)

        moment = pymunk.moment_for_circle(0.6, 0, BALL_RADIUS)
        self.ball = pymunk.Body(0.6, moment)
        self.ball.position = tuple(self.hand)
        ball_shape = pymunk.Circle(self.ball, BALL_RADIUS)
        ball_shape.elasticity, ball_shape.friction = 0.7, 0.6
        space.add(self.ball, ball_shape)
        self.space = space

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        options = options or {}
        hoop = options.get("hoop")
        if hoop is None:
            hoop = (self.rng.uniform(*HOOP_X_RANGE), self.rng.uniform(*HOOP_Y_RANGE))
        self.hoop = np.asarray(hoop, dtype=np.float64)
        self.hand = clamp_to_workspace(REST_HAND + self.rng.uniform(-8.0, 8.0, size=2))
        self.hand_velocity = np.zeros(2)
        self.held = True
        self.scored = False
        self.missed = False
        self.steps = 0
        self.release_step: int | None = None
        self.end_step: int | None = None
        self._build_space()
        return self._observation(), self._info()

    # ------------------------------------------------------------------- step
    def _move_hand(self, target: np.ndarray) -> None:
        velocity = HAND_GAIN * (target - self.hand)
        speed = float(np.linalg.norm(velocity))
        if speed > HAND_MAX_SPEED:
            velocity *= HAND_MAX_SPEED / speed
        moved = clamp_to_workspace(self.hand + velocity * DT)
        self.hand_velocity = (moved - self.hand) / DT
        self.hand = moved

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(ACTION_DIM)
        target = np.clip(action[:2], 0.0, WORLD_SIZE)
        if self.held and action[2] < 0.5:
            self.held = False
            self.release_step = self.steps
        for _ in range(SUBSTEPS):
            self._move_hand(target)
            if self.held:
                self.ball.position = tuple(self.hand)
                self.ball.velocity = tuple(self.hand_velocity)
            previous_y = self.ball.position.y
            self.space.step(DT)
            if not self.held:
                self._update_outcome(previous_y)
        self.steps += 1

        terminated = False
        if self.end_step is not None:
            linger = SCORE_LINGER_STEPS if self.scored else MISS_LINGER_STEPS
            terminated = self.steps - self.end_step >= linger
        truncated = not terminated and self.steps >= self.max_steps
        reward = 1.0 if self.scored else 0.0
        return self._observation(), reward, terminated, truncated, self._info()

    def _update_outcome(self, previous_y: float) -> None:
        if self.end_step is not None:
            return
        x, y = self.ball.position
        hx, hy = self.hoop
        if previous_y > hy >= y and abs(x - hx) < RIM_HALF_WIDTH - BALL_RADIUS * 0.3:
            self.scored = True
            self.end_step = self.steps
        elif y <= FLOOR_Y + BALL_RADIUS + 1.0 or not -40.0 < x < WORLD_SIZE + 40.0:
            self.missed = True
            self.end_step = self.steps

    def _observation(self) -> dict[str, np.ndarray]:
        x, y = self.ball.position
        vx, vy = self.ball.velocity
        return {
            "agent_pos": np.array(
                [*self.hand, float(self.held), self.steps / FPS], dtype=np.float32
            ),
            "environment_state": np.array([x, y, vx, vy, *self.hoop], dtype=np.float32),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "is_success": self.scored,
            "released": not self.held,
            "release_step": self.release_step,
            "hand_velocity": self.hand_velocity.copy(),
        }

    # ----------------------------------------------------------------- render
    def _to_screen(self, point) -> tuple[int, int]:
        scale = self.render_size / WORLD_SIZE
        return int(point[0] * scale), int((WORLD_SIZE - point[1]) * scale)

    def render(self, caption: str | None = None) -> np.ndarray:
        import pygame

        if self._surface is None:
            pygame.font.init()
            self._surface = pygame.Surface((self.render_size, self.render_size))
            self._font = pygame.font.SysFont(None, max(18, self.render_size // 20))
        surface = self._surface
        scale = self.render_size / WORLD_SIZE
        surface.fill((236, 242, 248))
        floor_top = self._to_screen((0, FLOOR_Y))[1]
        pygame.draw.rect(
            surface, (205, 160, 110), (0, floor_top, self.render_size, self.render_size)
        )

        # Hoop: backboard, net, rim
        hx, hy = self.hoop
        board_x = hx + RIM_HALF_WIDTH + 8.0
        pygame.draw.line(
            surface, (90, 90, 100), self._to_screen((board_x, hy - 12)),
            self._to_screen((board_x, hy + 90)), max(2, int(6 * scale)),
        )
        for i in range(5):
            top = hx - RIM_HALF_WIDTH + i * RIM_HALF_WIDTH / 2
            bottom = hx - RIM_HALF_WIDTH * 0.6 + i * RIM_HALF_WIDTH * 0.3
            pygame.draw.line(
                surface, (250, 250, 250), self._to_screen((top, hy)),
                self._to_screen((bottom, hy - 40)), 1,
            )
        pygame.draw.line(
            surface, (220, 60, 40), self._to_screen((hx - RIM_HALF_WIDTH - 3, hy)),
            self._to_screen((board_x, hy)), max(2, int(5 * scale)),
        )

        # Robot: pedestal + 2-link arm
        pedestal = [
            self._to_screen((SHOULDER[0] - 22, FLOOR_Y)),
            self._to_screen((SHOULDER[0] + 22, FLOOR_Y)),
            self._to_screen((SHOULDER[0] + 10, SHOULDER[1])),
            self._to_screen((SHOULDER[0] - 10, SHOULDER[1])),
        ]
        pygame.draw.polygon(surface, (70, 80, 100), pedestal)
        elbow = elbow_position(self.hand)
        width = max(4, int(12 * scale))
        for start, end in ((SHOULDER, elbow), (elbow, self.hand)):
            pygame.draw.line(
                surface, (60, 110, 200), self._to_screen(start), self._to_screen(end), width
            )
        for joint in (SHOULDER, elbow):
            pygame.draw.circle(surface, (30, 40, 60), self._to_screen(joint), width // 2 + 2)
        hand_color = (40, 170, 90) if self.held else (150, 160, 170)
        pygame.draw.circle(surface, hand_color, self._to_screen(self.hand), int(9 * scale))

        # Ball
        ball_center = self._to_screen(self.ball.position)
        radius = max(3, int(BALL_RADIUS * scale))
        pygame.draw.circle(surface, (240, 130, 30), ball_center, radius)
        pygame.draw.circle(surface, (90, 40, 10), ball_center, radius, 1)
        angle = self.ball.angle
        for offset in (0.0, math.pi / 2):
            dx = math.cos(angle + offset) * radius
            dy = math.sin(angle + offset) * radius
            pygame.draw.line(
                surface, (90, 40, 10),
                (ball_center[0] - dx, ball_center[1] + dy),
                (ball_center[0] + dx, ball_center[1] - dy), 1,
            )

        status = "SCORE!" if self.scored else "MISS" if self.missed else ""
        lines = [f"step {self.steps:3d}"] + ([caption] if caption else [])
        for row, text in enumerate(lines):
            surface.blit(self._font.render(text, True, (40, 40, 50)), (8, 6 + row * 22))
        if status:
            color = (30, 150, 60) if self.scored else (200, 50, 40)
            label = pygame.font.SysFont(None, self.render_size // 7).render(status, True, color)
            surface.blit(label, label.get_rect(center=(self.render_size // 2, self.render_size // 4)))
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()
