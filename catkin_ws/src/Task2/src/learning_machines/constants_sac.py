# ─────────────────────────────────────────────
# IR sensor layout
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

IR_THRESHOLD           = 100   # kept for visualize_metrics
IR_COLLISION_THRESHOLD = 120
IR_NEAR_THRESHOLD      = 60
IR_MAX_VALUE           = 400

FRONT_INDICES = [v for k, v in IR_indices.items() if k.startswith("FRONT")]
BACK_INDICES  = [v for k, v in IR_indices.items() if k.startswith("BACK")]
LEFT_INDICES  = [v for k, v in IR_indices.items() if k.endswith("L")]
RIGHT_INDICES = [v for k, v in IR_indices.items() if k.endswith("R")]

# ─────────────────────────────────────────────
# State dimension
# Walls are white, food is green — no classification needed.
# IR tells us about walls, camera tells us about food.
#
# [IR (8)] + [obj_visible, obj_dx, obj_size (3)] + [last_action (2)] = 13
# ─────────────────────────────────────────────
IR_STATE_DIM  = len(IR_indices)   # 8
VISION_DIM    = 3                 # obj_visible, obj_dx, obj_size
LAST_ACTION_DIM = 2               # left, right from previous step
STATE_DIM     = IR_STATE_DIM + VISION_DIM + LAST_ACTION_DIM  # 13

# ─────────────────────────────────────────────
# Continuous action space
# ─────────────────────────────────────────────
ACTION_DIM         = 2
MIN_WHEEL_SPEED    = -25.0
MAX_WHEEL_SPEED    = 25.0
ACTION_DURATION_MS = 300

# ─────────────────────────────────────────────
# SAC hyperparameters
# ─────────────────────────────────────────────
GAMMA              = 0.99
TAU                = 0.005
ACTOR_LR           = 3e-4
CRITIC_LR          = 3e-4
ALPHA_LR           = 3e-4
BATCH_SIZE         = 256
REPLAY_BUFFER_SIZE = 100_000
LEARNING_STARTS    = 1000    # ~5 episodes of random exploration before learning
UPDATES_PER_STEP   = 1
N_EPISODES         = 200
MAX_STEPS          = 200     # ~60s per episode at 300ms/step

# ─────────────────────────────────────────────
# Entropy / exploration
# ─────────────────────────────────────────────
AUTO_ENTROPY_TUNING = True
TARGET_ENTROPY      = -ACTION_DIM
INIT_ALPHA          = 0.2

# ─────────────────────────────────────────────
# Network architecture
# ─────────────────────────────────────────────
HIDDEN_DIM  = 128
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# ─────────────────────────────────────────────
# Reward — Task 2 foraging
# Walls are white → IR = wall signal only
# Green = food always → camera = food signal only
# ─────────────────────────────────────────────

# Food collection
FOOD_TOUCH_REWARD  = 5.0    # large bonus when food touched
FOOD_SPEED_BONUS   = 1.0    # additional bonus scaled by steps remaining

# Camera-guided approach
W_CENTERING        = 1.5    # facing food reward (decoupled from forward gate)
W_APPROACH_STATIC  = 2.0    # reward for blob size (closeness) — was 0.5
W_APPROACH_DELTA   = 3.0    # reward for blob growing (moving closer) — was 0.5

# Speed regulation
W_SPEED_SEARCH     = 0.5    # reward fast movement when no food visible
W_SPEED_WALL       = 0.5    # penalty for moving fast near wall

# Not visible penalty (from literature — prevents parking-and-staring)
NOT_VISIBLE_PENALTY = -0.05  # was -0.01 — stronger push to find food

# Wall avoidance (IR-based, proportional)
W_PROXIMITY        = 0.5

# Exploration when no food visible (simple grid, no odometry map)
GRID_SIZE          = 0.2
EXPLORATION_BONUS  = 0.5    # was 0.3

# Urgency — grows if no food collected for a while
URGENCY_PENALTY    = -0.03  # was -0.005 — actually felt now
MAX_URGENCY_STEPS  = 100    # was 50

# Collision
COLLISION_PENALTY  = -2.0   # was -1.0 — stronger wall aversion

# ─────────────────────────────────────────────
# Vision / colour detection
# Green HSV ranges — walls are white so no confusion
# ─────────────────────────────────────────────
GREEN_LOWER_SIM = (40,  60,  40)
GREEN_UPPER_SIM = (85, 255, 255)
GREEN_LOWER_HW  = (35,  50,  40)   # broader for lab lighting
GREEN_UPPER_HW  = (90, 255, 255)
MIN_BLOB_AREA_FRAC = 0.003          # ignore tiny specks < 0.3% of frame