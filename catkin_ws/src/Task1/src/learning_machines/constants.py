# ─────────────────────────────────────────────
# IR sensor indices and thresholds
# ─────────────────────────────────────────────
IR_indices = {
    "FRONT_L"  : 2,
    "FRONT_R"  : 3,
    "FRONT_C"  : 4,
    "FRONT_RR" : 5,
    "FRONT_LL" : 7,
    "BACK_L"   : 0,
    "BACK_R"   : 1,
    "BACK_C"   : 6,
}

# Task 0 threshold (kept for visualize_metrics compatibility)
IR_THRESHOLD = 100

# Task 1: 3-level discretisation thresholds
IR_CLEAR = 100   # 0–100   → level 0 (clear)
IR_NEAR  = 200   # 100–200 → level 1 (near)
                 # 200+    → level 2 (close)

FRONT_INDICES = [v for k, v in IR_indices.items() if k.startswith("FRONT")]
BACK_INDICES  = [v for k, v in IR_indices.items() if k.startswith("BACK")]
LEFT_INDICES  = [v for k, v in IR_indices.items() if k.endswith("L")]
RIGHT_INDICES = [v for k, v in IR_indices.items() if k.endswith("R")]

# ─────────────────────────────────────────────
# Action space (6 discrete actions)
# ─────────────────────────────────────────────
# Each action is (left_speed, right_speed, duration_ms)
ACTIONS = {
    0: ( 25,  25,  300),   # forward fast
    1: ( 15,  15,  300),   # forward medium
    2: ( 10,  10,  300),   # forward slow
    3: (-15,  15,  400),   # turn left
    4: ( 15, -15,  400),   # turn right
    5: (-25,  25,  400),   # sharp left
    6: ( 25, -25,  400),   # sharp right
}
ACTION_NAMES = {
    0: "forward fast",
    1: "forward medium",
    2: "forward slow",
    3: "turn left",
    4: "turn right",
    5: "sharp left",
    6: "sharp right",
}
N_ACTIONS = len(ACTIONS)

# ─────────────────────────────────────────────
# Q-learning hyperparameters
# ─────────────────────────────────────────────
ALPHA        = 0.1    # learning rate
GAMMA        = 0.9    # discount factor
EPSILON      = 1.0    # starting exploration rate
EPSILON_MIN  = 0.05   # minimum exploration rate
EPSILON_DECAY = 0.9   # decay per episode

N_EPISODES   = 100    # total training episodes
MAX_STEPS    = 100    # max steps per episode

# ─────────────────────────────────────────────
# Reward function weights
# ─────────────────────────────────────────────
W_SPEED       = 0.4   # weight for forward speed reward
W_ROTATION    = 0.3   # weight for penalising spinning
W_PROXIMITY   = 0.3   # weight for penalising obstacle proximity
EXPLORATION_BONUS  = 0.3   # reward for visiting a new grid cell
AVOIDANCE_BONUS = 0.5      # reward for escaping a near-obstacle situation
COLLISION_PENALTY  = -1.0  # hard penalty for collision
GRID_SIZE          = 0.2   # metres per grid cell for coverage tracking