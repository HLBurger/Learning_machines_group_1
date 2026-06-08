"""
robobo_env.py — Gymnasium-compatible environment wrapping IRobobo for TD3.

The environment exposes:
    observation_space : Box(21,)   continuous, see reward.build_observation()
    action_space      : Box(2,)    [left_wheel, right_wheel] in [-1, 1]

Usage (sim):
    from robobo_interface import SimulationRobobo
    from learning_machines.robobo_env import RoboboEnv

    rob = SimulationRobobo(identifier=1)
    env = RoboboEnv(rob)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

import numpy as np
import gym
from gym import spaces

from robobo_interface import IRobobo, SimulationRobobo

from .constants import (
    OBS_DIM, MAX_WHEEL_SPEED, MOVE_DURATION,
    FRONT_INDICES, IR_THRESHOLD,
    TD3_MAX_STEPS
)
from .reward import compute_reward_continuous, build_observation


class RoboboEnv(gym.Env):
    """
    Single-robot Gymnasium environment for Robobo obstacle avoidance / exploration.

    Episode termination
    -------------------
    - Truncated after TD3_MAX_STEPS steps (time limit, not a failure).
    - Never terminated early on collision; collisions are penalised via reward
      so the agent learns to avoid them rather than exploiting early termination.

    Observation (21-dim, dtype float32)
    ------------------------------------
    See reward.build_observation() and constants.OBS_DIM.

    Action (2-dim, dtype float32, range [-1, 1])
    --------------------------------------------
    [left_wheel_speed, right_wheel_speed]
    Scaled by MAX_WHEEL_SPEED before being sent to the robot.
    """

    metadata = {"render_modes": []}

    def __init__(self, rob: IRobobo):
        super().__init__()
        self.rob = rob

        # ── Spaces ────────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low  = np.full(OBS_DIM, -1.0, dtype=np.float32),
            high = np.full(OBS_DIM,  1.0, dtype=np.float32),
            dtype = np.float32,
        )
        self.action_space = spaces.Box(
            low  = np.full(2, -1.0, dtype=np.float32),
            high = np.full(2,  1.0, dtype=np.float32),
            dtype = np.float32,
        )

        # ── Episode state ─────────────────────────────────────────────────────
        self._step_count          = 0
        self._prev_irs            = [0] * 8
        self._last_action         = np.zeros(2, dtype=np.float32)
        self._visited_cells       = set()
        self._total_reward        = 0.0

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self):
        super().reset()

        # (Re)start simulation if in sim mode
        if isinstance(self.rob, SimulationRobobo):
            self.rob.play_simulation()

        # Reset episode counters
        self._step_count           = 0
        self._visited_cells        = set()
        self._total_reward         = 0.0
        self._last_action          = np.zeros(2, dtype=np.float32)

        # Initial sensor reading
        self._prev_irs = self.rob.read_irs()

        obs = build_observation(
            self._prev_irs,
            self._prev_irs,
            self._last_action
        )
        return obs

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # ── Scale and send motor command ───────────────────────────────────────
        left_cmd  = float(action[0]) * MAX_WHEEL_SPEED
        right_cmd = float(action[1]) * MAX_WHEEL_SPEED
        self.rob.move_blocking(left_cmd, right_cmd, MOVE_DURATION)

        # ── Read new sensor state ─────────────────────────────────────────────
        curr_irs = self.rob.read_irs()

        # ── Position (sim only) ───────────────────────────────────────────────
        position = (
            self.rob.get_position()
            if isinstance(self.rob, SimulationRobobo)
            else None
        )

        # ── Compute reward ────────────────────────────────────────────────────
        prev_cell_count = len(self._visited_cells)

        reward, self._visited_cells, info = compute_reward_continuous(
            action,
            curr_irs,
            self._prev_irs,
            position,
            self._visited_cells,
        )


        # ── Build next observation ────────────────────────────────────────────
        obs = build_observation(
            curr_irs,
            self._prev_irs,
            action
        )

        # ── Bookkeeping ───────────────────────────────────────────────────────
        self._prev_irs    = curr_irs
        self._last_action = action
        self._step_count += 1
        self._total_reward += reward

        terminated = False 
        truncated  = self._step_count >= TD3_MAX_STEPS
        done = terminated or truncated

        if done and isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()

        info["step"]              = self._step_count
        info["total_reward"]      = self._total_reward

        return obs, reward, done, info

    def close(self):
        if isinstance(self.rob, SimulationRobobo):
            try:
                self.rob.stop_simulation()
            except Exception:
                pass
