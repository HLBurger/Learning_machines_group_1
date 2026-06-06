# Initial stats
DRIVE_SPEED = 15  # Slower forward movement for better visibility
TURN_SPEED = 20   # Slower turning
DRIVE_MS = 500    # Longer duration between steps
TURN_MS = 800     # Longer turning time
IR_THRESHOLD = 100           # > means an obstacle
HISTORY_LEN = 10            # number of steps before to determine turning angle

IR_indices = {
    "FRONT_L" : 2,
    "FRONT_R" : 3,
    "FRONT_C" : 4,
    "FRONT_RR" : 5,
    "FRONT_LL" : 7,
    "BACK_L" : 0,
    "BACK_R" : 1,
    "BACK_C" : 6,
}

FRONT_INDICES = [value for key, value in IR_indices.items() if key.startswith('FRONT')]
BACK_INDICES = [value for key, value in IR_indices.items() if key.startswith('BACK')]
LEFT_INDICES = [value for key, value in IR_indices.items() if key.endswith('L')]
RIGHT_INDICES = [value for key, value in IR_indices.items() if key.endswith('R')]
