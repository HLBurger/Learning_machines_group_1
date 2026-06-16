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
    MAX_WHEEL_SPEED,
)


def compute_reward(
    left_speed: float,
    right_speed: float,
    irs: list,
    prev_irs: list,
    odom_map,           # OdometryMap instance
    sim_position=None,  # simulation Position object, or None
) -> tuple:
    """
    Compute the reward for one SAC step and advance the odometry map.

    Parameters
    ----------
    left_speed    : float        — left wheel speed chosen by SAC
    right_speed   : float        — right wheel speed chosen by SAC
    irs           : list         — current raw IR sensor readings, length 8
    prev_irs      : list         — previous step IR readings, length 8
    odom_map      : OdometryMap  — live map; will be mutated in-place
    sim_position  : object with .x .y, or None
                    When provided (simulation only), used for ground-truth XY.

    Returns
    -------
    reward    : float — scalar reward for this step
    new_cell  : bool  — True if the agent entered a previously-unseen cell
    collision : bool  — whether the robot collided
    """

    max_speed = float(MAX_WHEEL_SPEED)

    # 1. Forward speed reward, normalized 0 -> 1
    s_trans = (left_speed + right_speed) / (2.0 * max_speed)
    s_trans = max(s_trans, 0.0)

    # 2. Rotation penalty, 0 (straight) -> 1 (full spin)
    s_rot = abs(left_speed - right_speed) / (2.0 * max_speed)
    s_rot = min(s_rot, 1.0)

    # 3. Proximity score (0 = nothing near, 1 = very close)
    front_vals = [irs[i] for i in FRONT_INDICES]
    v_sens = max(front_vals) / 255.0
    v_sens = min(max(v_sens, 0.0), 1.0)

    # 4. Collision check
    collision = any(irs[i] > IR_COLLISION_THRESHOLD for i in FRONT_INDICES)

    # 5. Avoidance bonus (was near → now clear)
    prev_front_vals = [prev_irs[i] for i in FRONT_INDICES]
    prev_v_sens = max(prev_front_vals) / 255.0
    prev_v_sens = min(max(prev_v_sens, 0.0), 1.0)

    was_near  = prev_v_sens > 0.4
    is_clear  = v_sens < 0.2
    avoidance = AVOIDANCE_BONUS if (was_near and is_clear) else 0.0

    # 6. Odometry map update + exploration bonus
    #    OdometryMap.update() returns True when a new cell is entered.
    new_cell  = odom_map.update(left_speed, right_speed, sim_position=sim_position)
    exploration = EXPLORATION_BONUS if new_cell else 0.0

    # 7. Combine
    reward = (
        W_SPEED     *  s_trans
        + W_ROTATION * (1.0 - s_rot)
        + W_PROXIMITY * (1.0 - v_sens)
        + avoidance
        + exploration
        + (COLLISION_PENALTY if collision else 0.0)
    )

    return reward, new_cell, collision


def front_blocked(irs: list) -> bool:
    """True if any front sensor exceeds IR_NEAR_THRESHOLD."""
    return any(irs[i] > IR_NEAR_THRESHOLD for i in FRONT_INDICES)
