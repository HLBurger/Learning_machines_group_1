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
    W_RED_CLOSING,
    W_RED_APPROACH,
    W_GOAL_CENTERING,
    SWITCH_BONUS,
    GOAL_URGENCY_PENALTY,
    MAX_GOAL_URGENCY_STEPS,
)
from .vision import classify_collision
 
 
def compute_reward(
    mode: str,
    switch_mode: str,
    left_speed: float,
    right_speed: float,
    irs: list,
    vision: dict,
    prev_vision: dict,
    goal_reached: bool,
    current_step: int,
    steps_since_contact: int,
    steps_since_goal_visible: int,
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
 
    Notes on simplification from the previous version
    ---------------------------------------------------
    - Removed `red_lost`: it duplicated `urgency` (both keyed off
      `steps_since_contact` with a monotonic penalty). `urgency` alone
      now carries that signal.
    - Removed `speed_wall_penalty` and merged its intent into
      `wall_penalty`, which now scales with forward speed itself
      (`1 + forward`) instead of being a second, separately-weighted
      term off the same IR signal.
    - `speed_reward` (search-phase forward incentive) is now only
      applied while the red object is NOT visible. Once `red_visible`
      is True, `red_approach` already rewards closing the distance, so
      keeping `speed_reward` unconditional was double-counting forward
      motion and let the agent drift off-centre while still scoring
      "speed" points.
    - Removed `push_reward` entirely in goal mode: it rewarded raw
      forward motion gated only by `red_size`, with no awareness of
      `goal_dx`. That meant a robot pushing the object in a straight
      line away from the goal scored identically to one pushing it
      toward the goal, and it actively fought `goal_centering` whenever
      a corrective arc was needed. `goal_centering` (already scaled by
      `red_size`) is now the sole "push correctly" signal.
    - Fixed a bug where `terminal_reward` was summed into the goal-mode
      reward twice.
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
    if goal_reached and mode == "green":
        steps_remaining     = MAX_STEPS - current_step
        fast_bonus          = FAST_COMPLETION_BONUS * (steps_remaining / MAX_STEPS)
        terminal_reward     = OBJECT_IN_GOAL_REWARD + fast_bonus
        steps_since_contact = 0
    else:
        terminal_reward = 0.0
    
 
    # ── 2. Approach red object (phase 1 / ongoing) ────────────────────
    # Centering: reward for keeping red object in horizontal centre.
    # Approach: reward for blob size — larger means robot is closer.
    if red_visible:
        red_centering = W_RED_CENTERING * (1.0 - abs(red_dx))
        red_approach  = W_RED_APPROACH  * red_size
        closing_speed = W_RED_CLOSING * forward
        steps_since_contact = 0
    else:
        red_centering = 0.0
        red_approach  = 0.0
        closing_speed = 0.0
 
    # ── 3. Align with goal while pushing (phase 2) ────────────────────
    # Only fires when the robot already has the red object close
    # (red_size large) and the green goal is visible.
    # This naturally activates during the push phase without a hard threshold.
    # This is now the sole "push correctly" signal — it rewards forward
    # progress implicitly through red_size growing as the robot closes
    # the gap, while keeping the goal centred (i.e. aimed) along the way.
    if goal_visible and red_visible:
        goal_centering = W_GOAL_CENTERING * (1.0 - abs(goal_dx)) * red_size
    else:
        goal_centering = 0.0
 
    # ── 4. Speed regulation ────────────────────────────────────────────
    # Reward forward motion only while searching (red not yet visible).
    # Once the object is visible, red_approach / goal_centering already
    # reward the right kind of forward motion, so an unconditional speed
    # bonus here would double-count and could mask poor centring.
    speed_reward = W_SPEED_SEARCH * forward if not red_visible else 0.0
 
    # ── 5. Wall proximity penalty ─────────────────────────────────────
    # Applied when front IR is high and the obstacle is a wall, not the
    # object. Scales with forward speed so that approaching a wall fast
    # is penalised more than sitting near one — this absorbs what used
    # to be a separate speed_wall_penalty term.
    wall_penalty = (
        -W_PROXIMITY * front_ir_norm * (1.0 + forward)
        if contact_type != "object"
        else 0.0
    )
 
    # ── 6. Collision ──────────────────────────────────────────────────
    # Wall collision only; contact with the red object is desired and
    # detected as contact_type == "object" via vision.
    n_triggered  = sum(1 for i in FRONT_INDICES if irs[i] > IR_COLLISION_THRESHOLD)
    collision    = contact_type == "wall" and n_triggered >= 2
    coll_penalty = COLLISION_PENALTY if collision else 0.0
 
    # ── 7. Anti-stuck ─────────────────────────────────────────────────
    net_speed     = abs(left_speed + right_speed) / 2.0
    stuck_penalty = W_STUCK_PENALTY if net_speed < STUCK_SPEED_THRESHOLD else 0.0
 
    # ── 8. Exploration when red object not visible ────────────────────
    exploration = 0.0
    if not red_visible and position is not None:
        cell = (int(position.x / GRID_SIZE), int(position.y / GRID_SIZE))
        if cell not in visited_cells:
            exploration   = EXPLORATION_BONUS
            visited_cells = visited_cells | {cell}
 
    # ── 9. Urgency ─────────────────────────────────────────────────────
    # Sole penalty for "time spent without seeing the red object" —
    # previously duplicated by red_lost, which is now removed.
    capped_steps = min(steps_since_contact, MAX_URGENCY_STEPS)
    urgency      = URGENCY_PENALTY * capped_steps

    capped_goal_steps = min(steps_since_goal_visible, MAX_GOAL_URGENCY_STEPS)
    goal_urgency       = GOAL_URGENCY_PENALTY * capped_goal_steps

    switch_reward = 0
    if switch_mode == "green":
        switch_reward = SWITCH_BONUS

    elif switch_mode == "red":
        switch_reward = -SWITCH_BONUS

    
 
    # ── 10. Combine ────────────────────────────────────────────────────
    if mode == "red":
 
        reward = (
              red_centering
            + red_approach
            + speed_reward
            + closing_speed
            + wall_penalty
            + coll_penalty
            + stuck_penalty
            + exploration
            + urgency
            + switch_reward
        )
    else:
 
        reward = (
              terminal_reward
            + goal_centering
            + wall_penalty
            + coll_penalty
            + stuck_penalty
            + switch_reward
            + goal_urgency
        )
 
    if not red_visible:
        steps_since_contact += 1
 
    info = {
        "terminal"          : terminal_reward,
        "red_centering"     : red_centering,
        "red_approach"      : red_approach,
        "goal_centering"    : goal_centering,
        "speed_search"      : speed_reward,
        "wall"              : wall_penalty,
        "collision"         : coll_penalty,
        "stuck"             : stuck_penalty,
        "exploration"       : exploration,
        "urgency"           : urgency,
        "goal urgency"      : goal_urgency,
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
