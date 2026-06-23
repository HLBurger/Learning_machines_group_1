"""
reward_sac.py — Task 3 pushing reward function.

Design principles:
  - No hardcoded thresholds for behaviour switching; all components continuous.
  - Two-phase structure handled implicitly:
      Phase 1 (search/approach): robot rewarded for finding the red object,
        centering it, and driving toward it.
      Phase 2 (push): once the red object is large in frame (close contact),
        robot rewarded for keeping the object centred AND the green goal
        centred, and for sustained forward motion.
  - Terminal reward fires when red object centre pixel falls inside the
    green goal mask (goal_reached flag from vision).
  - Speed regulated by context: fast when searching, penalised near walls.
  - Collision with walls incurs full penalty; contact with the red object
    is not a collision and is detected via vision.
  - Anti-stuck penalty prevents oscillation.
  - IR = wall signal only (walls are white).
  - Camera = red object + green goal area.
"""

import numpy as np
from .constants_sac import (
    FRONT_INDICES,
    IR_COLLISION_THRESHOLD,
    MAX_WHEEL_SPEED,
    MAX_STEPS,
    W_SPEED_SEARCH,
    W_SPEED_WALL,
    W_PROXIMITY,
    W_STUCK_PENALTY,
    STUCK_SPEED_THRESHOLD,
    GRID_SIZE,
    EXPLORATION_BONUS,
    URGENCY_PENALTY,
    MAX_URGENCY_STEPS,
    COLLISION_PENALTY,
    OBJECT_IN_GOAL_REWARD,
    FAST_COMPLETION_BONUS,
    W_RED_CENTERING,
    W_RED_APPROACH,
    W_GOAL_CENTERING,
    RED_LOST_PENALTY,
    PUSH_FORWARD_REWARD,
)
from .vision import classify_collision


def compute_reward(
    left_speed: float,
    right_speed: float,
    irs: list,
    vision: dict,
    prev_vision: dict,
    goal_reached: bool,
    current_step: int,
    steps_since_contact: int,
    position,
    visited_cells: set,
    frame=None,
) -> tuple:
    """
    Compute Task 3 pushing reward for one SAC step.

    Parameters
    ----------
    left_speed, right_speed : float  — wheel speeds [-25, 25]
    irs                     : list   — current IR readings (8)
    vision                  : dict   — current analyse_frame() output
    prev_vision             : dict   — previous step analyse_frame() output
    goal_reached            : bool   — True when red object centre is inside
                                       green goal mask this step
    current_step            : int    — step index within episode
    steps_since_contact     : int    — steps since red object was last visible
    position                : Position or None  — sim position for grid
    visited_cells           : set    — grid cells visited this episode
    frame                   : np.ndarray or None — raw BGR frame for collision
                                                   type classification

    Returns
    -------
    reward              : float
    collision           : bool
    steps_since_contact : int (updated)
    visited_cells       : set (updated)
    info                : dict of reward components for logging
    """

    # ── Derived scalars ────────────────────────────────────────────────
    # Normalised forward motion in [0, 1]; reverse motion maps to 0.
    forward = max((left_speed + right_speed) / (2.0 * MAX_WHEEL_SPEED), 0.0)

    # Normalised front IR reading in [0, 1].
    front_ir_norm = max(irs[i] for i in FRONT_INDICES) / 255.0

    # Classify what is triggering the front IR sensors.
    front_vals   = [irs[i] for i in FRONT_INDICES]
    contact_type = (
        classify_collision(frame, front_vals, IR_COLLISION_THRESHOLD)
        if frame is not None
        else ("wall" if max(front_vals) > IR_COLLISION_THRESHOLD else "none")
    )

    # ── Extract vision features ────────────────────────────────────────
    red_visible  = vision["red_visible"]
    red_dx       = vision["red_dx"]       # horizontal offset [-1, 1], 0 = centred
    red_size     = vision["red_size"]     # blob area / frame area [0, 1]

    goal_visible = vision["goal_visible"]
    goal_dx      = vision["goal_dx"]      # horizontal offset of green goal [-1, 1]

    # ── 1. Terminal: object successfully pushed into goal ──────────────
    if goal_reached:
        steps_remaining   = MAX_STEPS - current_step
        fast_bonus        = FAST_COMPLETION_BONUS * (steps_remaining / MAX_STEPS)
        terminal_reward   = OBJECT_IN_GOAL_REWARD + fast_bonus
        steps_since_contact = 0
    else:
        terminal_reward = 0.0

    # ── 2. Approach red object (phase 1 / ongoing) ────────────────────
    # Centering: reward for keeping red object in horizontal centre.
    # Approach: reward for blob size — larger means robot is closer.
    if red_visible:
        red_centering = W_RED_CENTERING * (1.0 - abs(red_dx))
        red_approach  = W_RED_APPROACH  * red_size
        steps_since_contact = 0
        red_lost      = 0.0
    else:
        red_centering = 0.0
        red_approach  = 0.0
        red_lost      = RED_LOST_PENALTY
        # urgency grows with time since object was last visible
        search_urgency  = min(steps_since_contact / MAX_URGENCY_STEPS, 1.0)
        red_lost       *= (1.0 + search_urgency)

    # ── 3. Align with goal while pushing (phase 2) ────────────────────
    # Only fires when the robot already has the red object close
    # (red_size large) and the green goal is visible.
    # This naturally activates during the push phase without a hard threshold.
    if goal_visible and red_visible:
        goal_centering = W_GOAL_CENTERING * (1.0 - abs(goal_dx)) * red_size
    else:
        goal_centering = 0.0

    # ── 4. Forward push reward ────────────────────────────────────────
    # Fires when robot is moving forward with red object in near-contact.
    # red_size acts as a smooth gate: large = close = pushing.
    push_reward = PUSH_FORWARD_REWARD * forward * red_size if red_visible else 0.0

    # ── 5. Speed regulation ───────────────────────────────────────────
    # Reward forward motion at all times (searching or pushing).
    speed_reward = W_SPEED_SEARCH * forward

    # Penalise moving fast when a wall is directly ahead.
    speed_wall_penalty = -W_SPEED_WALL * forward * front_ir_norm

    # ── 6. Wall proximity penalty ─────────────────────────────────────
    # Applied when front IR is high and the obstacle is a wall, not the object.
    wall_penalty = -W_PROXIMITY * front_ir_norm if contact_type != "object" else 0.0

    # ── 7. Collision ──────────────────────────────────────────────────
    # Wall collision only; contact with the red object is desired and
    # detected as contact_type == "object" via vision.
    n_triggered  = sum(1 for i in FRONT_INDICES if irs[i] > IR_COLLISION_THRESHOLD)
    collision    = contact_type == "wall" and n_triggered >= 2
    coll_penalty = COLLISION_PENALTY if collision else 0.0

    # ── 8. Anti-stuck ─────────────────────────────────────────────────
    net_speed     = abs(left_speed + right_speed) / 2.0
    stuck_penalty = W_STUCK_PENALTY if net_speed < STUCK_SPEED_THRESHOLD else 0.0

    # ── 9. Exploration when red object not visible ────────────────────
    exploration = 0.0
    if not red_visible and position is not None:
        cell = (int(position.x / GRID_SIZE), int(position.y / GRID_SIZE))
        if cell not in visited_cells:
            exploration   = EXPLORATION_BONUS
            visited_cells = visited_cells | {cell}

    # ── 10. Urgency ───────────────────────────────────────────────────
    capped_steps = min(steps_since_contact, MAX_URGENCY_STEPS)
    urgency      = URGENCY_PENALTY * capped_steps

    # ── 11. Combine ───────────────────────────────────────────────────
    reward = (
        terminal_reward
      + red_centering
      + red_approach
      + goal_centering
      + push_reward
      + red_lost
      + speed_reward
      + speed_wall_penalty
      + wall_penalty
      + coll_penalty
      + stuck_penalty
      + exploration
      + urgency
    )

    if not red_visible:
        steps_since_contact += 1

    info = {
        "terminal"          : terminal_reward,
        "red_centering"     : red_centering,
        "red_approach"      : red_approach,
        "goal_centering"    : goal_centering,
        "push_reward"       : push_reward,
        "red_lost"          : red_lost,
        "speed_search"      : speed_reward,
        "speed_wall_penalty": speed_wall_penalty,
        "wall"              : wall_penalty,
        "collision"         : coll_penalty,
        "stuck"             : stuck_penalty,
        "exploration"       : exploration,
        "urgency"           : urgency,
        "red_visible"       : red_visible,
        "red_size"          : red_size,
        "goal_visible"      : goal_visible,
        "goal_reached"      : goal_reached,
    }

    return reward, collision, steps_since_contact, visited_cells, info


def front_blocked(irs: list) -> bool:
    """True if majority of front sensors are triggered above near-threshold."""
    from .constants_sac import IR_NEAR_THRESHOLD
    return sum(1 for i in FRONT_INDICES if irs[i] > IR_NEAR_THRESHOLD) >= 2
