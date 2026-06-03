from collections import deque
from .constants import *
from robobo_interface import IRobobo, SimulationRobobo
from .visualize_metrics import SimMetrics

def front_blocked(IR_list):
    return(
        IR_list[IR_indices["FRONT_R"]] > IR_THRESHOLD or
        IR_list[IR_indices["FRONT_L"]] > IR_THRESHOLD or
        IR_list[IR_indices["FRONT_C"]] > IR_THRESHOLD
    )

def side_clear_score(history: deque, side: str):
    
    if side == "left":
        indices = LEFT_INDICES
    elif side == "right":
        indices = RIGHT_INDICES
    
    score = 0
    for weight, IR_list in enumerate(history, start = 1):
        step_blocked = any([IR_list[i] > IR_THRESHOLD for i in indices])
        score += weight * (1 if not step_blocked else -1)
    return score


def avoid_obstacles(rob: IRobobo, max_iter: int = 200):

    metrics = SimMetrics()

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    ir_history = deque(maxlen = HISTORY_LEN)

    for step in range(max_iter):
        irs = rob.read_irs()
        metrics.record(irs)
        ir_history.append(irs)

        print(f"[step: {step}]"
              f"FrontC: {irs[IR_indices['FRONT_C']]}"
              f"FrontL: {irs[IR_indices['FRONT_L']]}"
              f"FrontLL: {irs[IR_indices['FRONT_LL']]}"
              f"FrontR: {irs[IR_indices['FRONT_R']]}"
              f"FrontRR: {irs[IR_indices['FRONT_RR']]}"
              f"BACKC: {irs[IR_indices['BACK_C']]}"
              f"BACKL: {irs[IR_indices['BACK_L']]}"
              f"BACKR: {irs[IR_indices['BACK_R']]}"
        )

        if front_blocked(irs):
            left_score = side_clear_score(ir_history, "left")
            right_score = side_clear_score(ir_history, "right")

            print(
                f"OBSTACLE DETECTED"
                f"left_score = {left_score:.1f}   right_score = {right_score:.1f}"
            )

            if right_score >= left_score:
                # Turning right
                print("TURNING RIGHT")
                rob.move_blocking(TURN_SPEED, -TURN_SPEED, TURN_MS)
            else:
                # Turning left
                print("TURNING LEFT")
                rob.move_blocking(-TURN_SPEED, TURN_SPEED, TURN_MS)
        else:
            # Clear path, moving forwards
            rob.move_blocking(DRIVE_SPEED, DRIVE_SPEED, DRIVE_MS)
    
    metrics.plot()
    print(metrics.summary())

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()