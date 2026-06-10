from .constants_sac import (
    FRONT_INDICES,
    IR_COLLISION_THRESHOLD,
    IR_NEAR_THRESHOLD,
    W_SPEED,
    W_ROTATION,
    W_PROXIMITY,
    EXPLORATION_BONUS,
    COLLISION_PENALTY,
    AVOIDANCE_BONUS,
    GRID_SIZE,
    MAX_WHEEL_SPEED,
)


def compute_reward(
    left_speed: float,
    right_speed: float,
    irs: list,
    prev_irs: list,
    position,
    visited_cells: set,
) -> tuple:
    """
    Compute the reward for one SAC step.

    Parameters
    ----------
    left_speed    : float — left wheel speed chosen by SAC
    right_speed   : float — right wheel speed chosen by SAC
    irs           : list  — current raw IR sensor readings, length 8
    prev_irs      : list  — previous step IR readings, length 8
    position      : Position object with .x and .y, or None
    visited_cells : set   — grid cells already visited this episode

    Returns
    -------
    reward        : float — scalar reward for this step
    visited_cells : set   — updated visited cells
    collision     : bool  — whether robot collided / should end episode
    """

    max_speed = float(MAX_WHEEL_SPEED)

    # 1. Forward speed reward, normalized 0 -> 1
    s_trans = (left_speed + right_speed) / (2.0 * max_speed)
    s_trans = max(s_trans, 0.0)

    # 2. Rotation score, normalized 0 -> 1
    # High if robot drives straight, low if it spins.
    s_rot = abs(left_speed - right_speed) / (2.0 * max_speed)
    s_rot = min(s_rot, 1.0)

    # 3. Proximity score
    front_vals = [irs[i] for i in FRONT_INDICES]
    v_sens = max(front_vals) / 255.0
    v_sens = min(max(v_sens, 0.0), 1.0)

    # 4. Collision check
    collision = any(irs[i] > IR_COLLISION_THRESHOLD for i in FRONT_INDICES)

    # 5. Avoidance bonus
    prev_front_vals = [prev_irs[i] for i in FRONT_INDICES]
    prev_v_sens = max(prev_front_vals) / 255.0
    prev_v_sens = min(max(prev_v_sens, 0.0), 1.0)

    was_near = prev_v_sens > 0.4
    is_clear = v_sens < 0.2
    avoidance = AVOIDANCE_BONUS if (was_near and is_clear) else 0.0

    # 6. Exploration bonus
    exploration = 0.0
    if position is not None:
        cell = (int(position.x / GRID_SIZE), int(position.y / GRID_SIZE))
        if cell not in visited_cells:
            exploration = EXPLORATION_BONUS
            visited_cells.add(cell)

    # 7. Combine
    reward = (
        W_SPEED * s_trans
        + W_ROTATION * (1.0 - s_rot)
        + W_PROXIMITY * (1.0 - v_sens)
        + avoidance
        + exploration
        + (COLLISION_PENALTY if collision else 0.0)
    )

    return reward, visited_cells, collision


def front_blocked(irs: list) -> bool:
    """True if any front sensor exceeds IR_NEAR."""
    return any(irs[i] > IR_NEAR_THRESHOLD for i in FRONT_INDICES)