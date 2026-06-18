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
    W_OBJECT_APPROACH,
    OBJECT_CENTERING_BONUS,
    OBJECT_APPROACH_BONUS,
    SIZE_APPROACH_MIN_DELTA,
    IR_APPROACH_MIN_DELTA,
)
from .vision import classify_collision, analyse_frame


def compute_reward(
    left_speed: float,
    right_speed: float,
    irs: list,
    prev_irs: list,
    odom_map,           # OdometryMap instance
    sim_position=None,  # simulation Position object, or None
    frame=None,         # camera frame (np.ndarray BGR) or None
    prev_frame=None,    # previous step's camera frame, for approach detection
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
    frame         : np.ndarray or None
                    Front camera frame used by classify_collision.
                    When None, collision_type is inferred from IR only.
    prev_frame    : np.ndarray or None
                    Previous step's camera frame, used to detect whether
                    the robot is closing in on a food object.

    Returns
    -------
    reward         : float — scalar reward for this step
    new_cell       : bool  — True if the agent entered a previously-unseen cell
    collision      : bool  — whether the robot collided
    collision_type : str   — "none" | "wall" | "object"
    """

    max_speed = float(MAX_WHEEL_SPEED)

    # 1. Forward speed reward, normalized 0 -> 1
    s_trans = (left_speed + right_speed) / (2.0 * max_speed)
    s_trans = max(s_trans, 0.0)

    # 2. Rotation penalty, 0 (straight) -> 1 (full spin)
    s_rot = abs(left_speed - right_speed) / (2.0 * max_speed)
    s_rot = min(s_rot, 1.0)

    # 3. Collision check + type classification
    front_vals = [irs[i] for i in FRONT_INDICES]
    front_max = max(front_vals)
    if frame is not None:
        # Pass the full front IR list so classify_collision can use the median
        collision_type = classify_collision(frame, front_vals, IR_COLLISION_THRESHOLD)
    else:
        # No camera feed: IR-only fallback (cannot detect objects without vision)
        collision_type = "wall" if front_max > IR_COLLISION_THRESHOLD else "none"
    collision = collision_type == "wall"

    # 4. Proximity score (0 = nothing near, 1 = very close)
    # front_vals already computed above
    v_sens = min(max(max(front_vals) / 255.0, 0.0), 1.0)

    wall_proximity = 0.0
    object_proximity = 0.0
    ir_object_reward = 0.0
    proximity_reward = 0.0

    if collision_type == "wall":
        wall_proximity = v_sens
    elif collision_type == "object":
        object_proximity = v_sens
        ir_object_reward += object_proximity

    # stay away from walls
    proximity_reward -= W_PROXIMITY * wall_proximity

    # get close to objects
    proximity_reward += W_PROXIMITY * object_proximity

    # 5. Avoidance bonus (was near → now clear)
    prev_front_vals = [prev_irs[i] for i in FRONT_INDICES]
    prev_v_sens = max(prev_front_vals) / 255.0
    prev_v_sens = min(max(prev_v_sens, 0.0), 1.0)

    avoidance = 0.0

    if collision_type != "object":
        was_near = prev_v_sens > 0.4
        is_clear = v_sens < 0.2
        

        if was_near and is_clear:
            avoidance = AVOIDANCE_BONUS
    
    if collision_type == "object":
        
        if v_sens - prev_v_sens > IR_APPROACH_MIN_DELTA:
            ir_object_reward += OBJECT_APPROACH_BONUS

    # 6. Food object approach incentive
    object_reward = 0.0
    if frame is not None:
        feats = analyse_frame(frame)
        if feats["object_visible"]:
            # Reward for facing the object: 1.0 when centred, 0.0 at the edge
            centering = 1.0 - abs(feats["object_dx"])
            object_reward = W_OBJECT_APPROACH * centering * feats["object_size"]

            # Bonus for being well-centred (actively steering toward it)
            if abs(feats["object_dx"]) < 0.2:
                object_reward += OBJECT_CENTERING_BONUS

            # Bonus when the object grew since the last frame (robot closing in)
            if prev_frame is not None:
                prev_feats = analyse_frame(prev_frame)

                
                if prev_feats["object_visible"] and feats["object_size"] - prev_feats["object_size"] > SIZE_APPROACH_MIN_DELTA:
                    object_reward += OBJECT_APPROACH_BONUS

    # 7. Odometry map update + exploration bonus
    #    OdometryMap.update() returns True when a new cell is entered.
    new_cell  = odom_map.update(left_speed, right_speed, sim_position=sim_position)
    exploration = EXPLORATION_BONUS if new_cell else 0.0

    # 8. Collision penalties/reward
    collision_reward = 0.0

    if collision_type == "wall":
        collision_reward = COLLISION_PENALTY

    elif collision_type == "object":
        collision_reward = abs(COLLISION_PENALTY)


    # 9. Combine
    reward = (
        W_SPEED * s_trans
        + W_ROTATION * (1.0 - s_rot)
        + proximity_reward
        + avoidance
        + object_reward
        + ir_object_reward
        + exploration
        + collision_reward
    )

    return reward, new_cell, collision, collision_type


def front_blocked(irs: list) -> bool:
    """True if any front sensor exceeds IR_NEAR_THRESHOLD."""
    return any(irs[i] > IR_NEAR_THRESHOLD for i in FRONT_INDICES)
