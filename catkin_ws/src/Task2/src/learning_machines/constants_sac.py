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
IR_COLLISION_THRESHOLD = 150
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
UPDATES_PER_STEP   = 4     # more gradient steps per env step = faster learning
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
HIDDEN_DIM  = 64     # 13-dim state is simple; smaller net converges faster
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# ─────────────────────────────────────────────
# Reward — Task 2 foraging
# Walls are white → IR = wall signal only
# Green = food always → camera = food signal only
# ─────────────────────────────────────────────

# Food collection
FOOD_TOUCH_REWARD  = 15.0   # large bonus per package — each one is worth chasing
FOOD_SPEED_BONUS   = 3.0    # extra bonus scaled by steps remaining (fast = bigger)

# Camera-guided approach
W_CENTERING        = 2.0    # reward for facing food (no forward gate)
W_APPROACH_STATIC  = 2.0    # reward for blob size (closeness)
W_APPROACH_DELTA   = 5.0    # reward for blob growing — strongest signal to move IN

# Speed regulation
W_SPEED_SEARCH     = 1.0    # strong reward for moving fast when no food visible
W_SPEED_WALL       = 0.5    # penalty for moving fast near wall

# Not visible penalty — must be strong enough to make searching worthwhile
NOT_VISIBLE_PENALTY = -0.1  # -0.1/step × 200 steps = -20 baseline; touching food (+10) beats 100 steps searching

# Wall avoidance (IR-based, proportional)
W_PROXIMITY        = 1.0    # stronger wall push-away

# Anti-stuck: penalty for near-zero speed (robot oscillating or frozen)
W_STUCK_PENALTY    = -0.3   # per step where |left+right| < threshold
STUCK_SPEED_THRESHOLD = 2.0 # abs(left+right)/2 below this = stuck

# Exploration when no food visible (simple grid, no odometry map)
GRID_SIZE          = 0.2
EXPLORATION_BONUS  = 1.0    # strong incentive to visit new cells

# Urgency — grows if no food collected for a while
URGENCY_PENALTY    = -0.05  # -0.05/step × 100 cap = -5 max; meaningful but not dominant
MAX_URGENCY_STEPS  = 100

# Collision — full penalty, not fractional
COLLISION_PENALTY  = -3.0

# ─────────────────────────────────────────────
# Vision / colour detection
# Green HSV ranges — walls are white so no confusion
# ─────────────────────────────────────────────
GREEN_LOWER_SIM = (40,  60,  40)
GREEN_UPPER_SIM = (85, 255, 255)
GREEN_LOWER_HW  = (35,  50,  40)   # broader for lab lighting
GREEN_UPPER_HW  = (90, 255, 255)
MIN_BLOB_AREA_FRAC = 0.001          # detect packages from further away