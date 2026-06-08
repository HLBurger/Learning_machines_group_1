"""
reward.py — expanded reward function for both Q-learning and TD3.

compute_reward()          : original signature, used by Q-learning train loop.
compute_reward_continuous(): new signature for TD3 / Gym env.
front_blocked()           : utility used by train_rl.py.
"""

import numpy as np
from .constants import (
    ACTIONS, FRONT_INDICES, BACK_INDICES, LEFT_INDICES, RIGHT_INDICES,
    IR_THRESHOLD, IR_NEAR,
    W_SPEED, W_ROTATION, W_PROXIMITY,
    EXPLORATION_BONUS, AVOIDANCE_BONUS,
    COLLISION_PENALTY,
    GRID_SIZE, MAX_WHEEL_SPEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared internals
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_ir(irs: list, indices: list) -> float:
    """Return max IR reading for the given sensor indices, normalised to [0, 1]."""
    return max(irs[i] for i in indices) / 255.0


def _delta_ir(curr_irs: list, prev_irs: list, indices: list) -> float:
    """
    Return the change in max-IR for the given sensor group.
    Positive means the robot is getting closer; negative means moving away.
    """
    return _normalise_ir(curr_irs, indices) - _normalise_ir(prev_irs, indices)


def _exploration_bonus(position, visited_cells: set):
    """Return an exploration bonus if a new grid cell was entered."""
    if position is None:
        return 0.0, visited_cells
    cell = (int(position.x / GRID_SIZE), int(position.y / GRID_SIZE))
    if cell not in visited_cells:
        visited_cells = visited_cells | {cell}   # immutable-style update
        return EXPLORATION_BONUS, visited_cells
    return 0.0, visited_cells

# ─────────────────────────────────────────────────────────────────────────────
# Q-learning interface  (original signature — backward compatible)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    action: int,
    irs: list,
    prev_irs: list,
    position,
    visited_cells: set,
) -> tuple:
    """
    Compute reward for one Q-learning step.

    Parameters
    ----------
    action               : int   — action index into ACTIONS dict
    irs                  : list  — current IR readings (length 8)
    prev_irs             : list  — previous IR readings (length 8)
    position             : Position with .x, .y (sim only) or None
    visited_cells        : set   — grid cells visited so far this episode
    steps_since_new_cell : int   — how many steps since a new cell was entered

    Returns
    -------
    reward        : float
    visited_cells : set (updated)
    """
    left_speed, right_speed, _ = ACTIONS[action]

    # ── 1. Forward speed (0 → 1) ─────────────────────────────────────────────
    s_trans = max((left_speed + right_speed) / (2 * MAX_WHEEL_SPEED), 0.0)

    # ── 2. Rotation magnitude (0 → 1) ────────────────────────────────────────
    s_rot = abs(left_speed - right_speed) / (2 * MAX_WHEEL_SPEED)

    # ── 3. Front proximity (0 → 1, higher = closer) ──────────────────────────
    v_front     = _normalise_ir(irs, FRONT_INDICES)
    prev_v_front = _normalise_ir(prev_irs, FRONT_INDICES)

    # ── 6. Collision check ───────────────────────────────────────────────────
    collision = any(irs[i] > IR_THRESHOLD for i in FRONT_INDICES)

    # ── 7. Avoidance bonus (was near, now clear) ──────────────────────────────
    was_near  = prev_v_front > 0.4
    is_clear  = v_front < 0.2
    avoidance = AVOIDANCE_BONUS if (was_near and is_clear) else 0.0

    # ── 9. Exploration bonus ──────────────────────────────────────────────────
    exploration, visited_cells = _exploration_bonus(position, visited_cells)
    # ── 11. Combine ───────────────────────────────────────────────────────────
    reward = (
        W_SPEED            * s_trans
        + W_ROTATION       * (1.0 - s_rot)
        + W_PROXIMITY      * (1.0 - v_front)
        + avoidance
        + exploration
        + (COLLISION_PENALTY if collision else 0.0)
    )

    return reward, visited_cells


# ─────────────────────────────────────────────────────────────────────────────
# TD3 / continuous interface
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward_continuous(
    action: np.ndarray,
    irs: list,
    prev_irs: list,
    position,
    visited_cells: set,
) -> tuple:
    """
    Compute reward for one TD3 step.

    Parameters
    ----------
    action               : np.ndarray shape (2,) — [left, right] in [-1, 1]
    irs                  : list  — current IR readings (length 8)
    prev_irs             : list  — previous IR readings (length 8)
    position             : Position with .x, .y (sim only) or None
    visited_cells        : set   — grid cells visited so far this episode
    steps_since_new_cell : int   — steps since a new cell was entered

    Returns
    -------
    reward        : float
    visited_cells : set (updated)
    info          : dict  — individual reward components for logging
    """
    left_norm, right_norm = float(action[0]), float(action[1])

    # ── 1. Forward speed  ─────────────────────────────────────────────────────
    # Average of both wheels, clamped to forward motion only.
    s_trans = max((left_norm + right_norm) / 2.0, 0.0)

    # ── 2. Rotation magnitude ─────────────────────────────────────────────────
    s_rot = abs(left_norm - right_norm) / 2.0

    # ── 3. Front proximity ────────────────────────────────────────────────────
    v_front      = _normalise_ir(irs, FRONT_INDICES)
    prev_v_front = _normalise_ir(prev_irs, FRONT_INDICES)
    

    # ── 6. Collision ──────────────────────────────────────────────────────────
    front_collision = any(irs[i] > IR_THRESHOLD for i in FRONT_INDICES)
    back_collision = any(irs[i] > IR_THRESHOLD for i in BACK_INDICES)

    # ── 7. Avoidance bonus ────────────────────────────────────────────────────
    was_near  = prev_v_front > 0.4
    is_clear  = v_front < 0.2
    avoidance = AVOIDANCE_BONUS if (was_near and is_clear) else 0.0

    # ── 9. Exploration bonus ──────────────────────────────────────────────────
    exploration, visited_cells = _exploration_bonus(position, visited_cells)

    # ── 11. Combine ───────────────────────────────────────────────────────────
    components = dict(
        speed          = W_SPEED        * s_trans,
        rotation       = W_ROTATION     * (1.0 - s_rot),
        proximity      = W_PROXIMITY    * (1.0 - v_front),
        avoidance      = avoidance,
        exploration    = exploration,
        front_collision = COLLISION_PENALTY if front_collision else 0.0,
        back_collision = COLLISION_PENALTY if back_collision else 0.0,
    )
    reward = sum(components.values())

    info = {
        **components,
        "front_collision" : front_collision,
        "back_collision" : back_collision,
        "v_front"       : v_front,
        "s_trans"       : s_trans,
        "s_rot"         : s_rot,
        "cells_visited" : len(visited_cells),
    }

    return reward, visited_cells, info


# ─────────────────────────────────────────────────────────────────────────────
# Observation builder
# ─────────────────────────────────────────────────────────────────────────────

def build_observation(irs, prev_irs, last_action):
    curr_norm = np.array([irs[i] / 255.0 for i in range(8)], dtype=np.float32)
    prev_norm  = np.array([prev_irs[i] / 255.0 for i in range(8)], dtype=np.float32)
    delta_norm = np.clip(curr_norm - prev_norm, -1.0, 1.0)

    la        = np.asarray(last_action, dtype=np.float32)
    fwd_speed  = np.clip((la[0] + la[1]) / 2.0, 0.0, 1.0)
    rot_mag    = abs(la[0] - la[1]) / 2.0

    return np.concatenate([curr_norm, delta_norm, la, np.array([fwd_speed, rot_mag], dtype = np.float32)])


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def front_blocked(irs: list) -> bool:
    """True if any front sensor exceeds IR_NEAR."""
    return any(irs[i] > IR_NEAR for i in FRONT_INDICES)
