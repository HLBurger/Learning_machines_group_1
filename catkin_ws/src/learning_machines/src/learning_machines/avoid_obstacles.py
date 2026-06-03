from collections import deque

from robobo_interface import IRobobo, SimulationRobobo

# Initial stats
DRIVE_SPEED = 40
TURN_SPEED = 40
DRIVE_MS = 500
TURN_MS = 600
IR_THRESHOLD = 100           # > means an obstacle
HISTORY_LEN = 10            # number of steps before to determine turning angle

# IR indexation
FRONT_L_INDEX = 2
FRONT_R_INDEX = 3
FRONT_C_INDEX = 4
FRONT_RR_INDEX = 5
FRONT_LL_INDEX = 7
BACK_L_INDEX = 0
BACK_R_INDEX = 1
BACK_C_INDEX = 6

def front_blocked(IR_list):
    return(
        IR_list[FRONT_R_INDEX] > IR_THRESHOLD or
        IR_list[FRONT_L_INDEX] > IR_THRESHOLD or
        IR_list[FRONT_C_INDEX] > IR_THRESHOLD
    )

def side_clear_score(history: deque, side: str):
    
    if side == "left":
        indices = [FRONT_LL_INDEX, FRONT_L_INDEX, BACK_L_INDEX]
    elif side == "right":
        indices = [FRONT_RR_INDEX, FRONT_R_INDEX, BACK_R_INDEX]
    
    score = 0
    n = len(history)
    for weight, IR_list in enumerate(history, start = 1):
        step_blocked = any([IR_list[i] > IR_THRESHOLD for i in indices])
        score += weight * (1 if not step_blocked else -1)
    return score


def avoid_obstacles(rob: IRobobo, max_iter: int = 200):

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    ir_history = deque(maxlen = HISTORY_LEN)

    for step in range(max_iter):
        irs = rob.read_irs()
        ir_history.append(irs)

        print(f"[step: {step}]"
              f"FrontC: {irs[FRONT_C_INDEX]}"
              f"FrontL: {irs[FRONT_L_INDEX]}"
              f"FrontLL: {irs[FRONT_LL_INDEX]}"
              f"FrontR: {irs[FRONT_R_INDEX]}"
              f"FrontRR: {irs[FRONT_RR_INDEX]}"
              f"BACKC: {irs[BACK_C_INDEX]}"
              f"BACKL: {irs[BACK_L_INDEX]}"
              f"BACKR: {irs[FRONT_R_INDEX]}"
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
        
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()