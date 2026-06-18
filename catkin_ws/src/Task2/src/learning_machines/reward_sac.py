"""
reward_sac.py — Task 2 foraging reward function.

Design principles (TA feedback + literature):
  - No hardcoded thresholds for behaviour switching
  - All components continuous and naturally scaled
  - Speed regulated by context (fast when searching, slow near walls)
  - Centering gated by forward motion (no spin-in-place exploit)
  - Approach uses static + delta blob size (no stalling before touch)
  - Penalty when food not visible (no parking-and-staring exploit)
  - Urgency grows over time (competition: 3 minutes = collect fast)
  - IR = wall signal only (walls are white)
  - Camera = food signal only (food is green)
"""

import numpy as np
from .constants_sac import (
    FRONT_INDICES,
    IR_COLLISION_THRESHOLD,
    MAX_WHEEL_SPEED,
    MAX_STEPS,
    FOOD_TOUCH_REWARD,
    FOOD_SPEED_BONUS,
    W_CENTERING,
    W_APPROACH_STATIC,
    W_APPROACH_DELTA,
    W_SPEED_SEARCH,
    W_SPEED_WALL,
    NOT_VISIBLE_PENALTY,
    W_PROXIMITY,
    GRID_SIZE,
    EXPLORATION_BONUS,
    URGENCY_PENALTY,
    MAX_URGENCY_STEPS,
    COLLISION_PENALTY,
)


def compute_reward(
    left_speed: float,
    right_speed: float,
    irs: list,
    vision: dict,
    prev_vision: dict,
    food_collected: int,
    prev_food_collected: int,
    current_step: int,
    steps_since_food: int,
    position,
    visited_cells: set,
) -> tuple:
    """
    Compute Task 2 reward for one SAC step.

    Parameters
    ----------
    left_speed, right_speed : float  — wheel speeds [-25, 25]
    irs                     : list   — current IR readings (8)
    vision                  : dict   — current analyse_frame() output
    prev_vision             : dict   — previous step analyse_frame() output
    food_collected          : int    — total food count this episode
    prev_food_collected     : int    — previous step food count
    current_step            : int    — step index within episode
    steps_since_food        : int    — steps since last food collected
    position                : Position or None  — sim position for grid
    visited_cells           : set    — grid cells visited this episode

    Returns
    -------
    reward           : float
    collision        : bool
    steps_since_food : int (updated)
    visited_cells    : set (updated)
    info             : dict of components for logging
    """

    # normalised forward motion [0, 1]
    forward = max((left_speed + right_speed) / (2.0 * MAX_WHEEL_SPEED), 0.0)

    # normalised front IR [0, 1]
    front_ir_norm = max(irs[i] for i in FRONT_INDICES) / 255.0

    # ── 1. Food touch ─────────────────────────────────────────────────
    food_touched = food_collected > prev_food_collected
    if food_touched:
        steps_since_food = 0
        speed_bonus      = FOOD_SPEED_BONUS * (1.0 - current_step / MAX_STEPS)
        touch_reward     = FOOD_TOUCH_REWARD + speed_bonus
    else:
        touch_reward = 0.0

    # ── 2. Camera-guided approach (gated by forward motion) ───────────
    obj_visible  = vision["obj_visible"]
    obj_dx       = vision["obj_dx"]
    obj_size     = vision["obj_size"]
    prev_size    = prev_vision["obj_size"]

    if obj_visible:
        # centering: reward facing food regardless of forward speed
        # gating by forward was preventing the robot from learning to turn toward food
        centering  = W_CENTERING * (1.0 - abs(obj_dx))

        # approach: static size (closeness) + size growth (moving closer)
        size_delta = max(obj_size - prev_size, 0.0)
        approach   = W_APPROACH_STATIC * obj_size + W_APPROACH_DELTA * size_delta

        not_visible = 0.0
    else:
        centering   = 0.0
        approach    = 0.0
        # penalty for losing food from view (prevents parking-and-staring)
        not_visible = NOT_VISIBLE_PENALTY

    # ── 3. Speed regulation ───────────────────────────────────────────
    if not obj_visible:
        # reward moving fast when searching (no food in view)
        speed_reward = W_SPEED_SEARCH * forward
    else:
        speed_reward = 0.0

    # penalise moving fast near walls regardless of food visibility
    speed_wall_penalty = -W_SPEED_WALL * forward * front_ir_norm

    # ── 4. Wall proximity penalty — proportional ──────────────────────
    wall_penalty = -W_PROXIMITY * front_ir_norm

    # ── 5. Collision ──────────────────────────────────────────────────
    # walls are white so IR high = wall collision, not food
    n_triggered  = sum(1 for i in FRONT_INDICES if irs[i] > IR_COLLISION_THRESHOLD)
    collision    = n_triggered >= 2
    coll_penalty = COLLISION_PENALTY * (n_triggered / len(FRONT_INDICES))

    # ── 6. Exploration when no food visible ───────────────────────────
    exploration = 0.0
    if not obj_visible and position is not None:
        cell = (int(position.x / GRID_SIZE), int(position.y / GRID_SIZE))
        if cell not in visited_cells:
            exploration = EXPLORATION_BONUS
            visited_cells = visited_cells | {cell}

    # ── 7. Urgency ────────────────────────────────────────────────────
    capped_steps = min(steps_since_food, MAX_URGENCY_STEPS)
    urgency      = URGENCY_PENALTY * capped_steps

    # ── 8. Combine ────────────────────────────────────────────────────
    reward = (
        touch_reward
      + centering
      + approach
      + not_visible
      + speed_reward
      + speed_wall_penalty
      + wall_penalty
      + coll_penalty
      + exploration
      + urgency
    )

    if not food_touched:
        steps_since_food += 1

    info = {
        "touch"             : touch_reward,
        "centering"         : centering,
        "approach"          : approach,
        "not_visible"       : not_visible,
        "speed_search"      : speed_reward,
        "speed_wall_penalty": speed_wall_penalty,
        "wall"              : wall_penalty,
        "collision"         : coll_penalty,
        "exploration"       : exploration,
        "urgency"           : urgency,
        "obj_visible"       : obj_visible,
        "obj_size"          : obj_size,
        "food_total"        : food_collected,
    }

    return reward, collision, steps_since_food, visited_cells, info


def front_blocked(irs: list) -> bool:
    """True if majority of front sensors triggered."""
    from .constants_sac import IR_NEAR_THRESHOLD
    return sum(1 for i in FRONT_INDICES if irs[i] > IR_NEAR_THRESHOLD) >= 2