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

FRONT_INDICES = [v for k, v in IR_indices.items() if k.startswith("FRONT")]
BACK_INDICES  = [v for k, v in IR_indices.items() if k.startswith("BACK")]
LEFT_INDICES  = [v for k, v in IR_indices.items() if k.endswith("L")]
RIGHT_INDICES = [v for k, v in IR_indices.items() if k.endswith("R")]

STATE_DIM = len(IR_indices)

IR_MAX_VALUE = 400
IR_COLLISION_THRESHOLD = 120
IR_NEAR_THRESHOLD = 60


# ─────────────────────────────────────────────
# Continuous action space
# ─────────────────────────────────────────────
ACTION_DIM = 2

ACTION_LOW = -1.0
ACTION_HIGH = 1.0

MIN_WHEEL_SPEED = -25.0
MAX_WHEEL_SPEED = 25.0

ACTION_DURATION_MS = 300


# ─────────────────────────────────────────────
# SAC hyperparameters
# ─────────────────────────────────────────────
GAMMA = 0.99
TAU = 0.005

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 100_000
LEARNING_STARTS = 1_000
UPDATES_PER_STEP = 1

N_EPISODES = 100
MAX_STEPS = 100

# ─────────────────────────────────────────────
# SAC smoke-test hyperparameters
# ─────────────────────────────────────────────
# BATCH_SIZE = 4
# REPLAY_BUFFER_SIZE = 100
# LEARNING_STARTS = 0
# UPDATES_PER_STEP = 1

# N_EPISODES = 1
# MAX_STEPS = 10

# ─────────────────────────────────────────────
# Entropy / exploration
# ─────────────────────────────────────────────
AUTO_ENTROPY_TUNING = True
TARGET_ENTROPY = -ACTION_DIM
INIT_ALPHA = 0.2


# ─────────────────────────────────────────────
# Network architecture
# ─────────────────────────────────────────────
HIDDEN_DIM = 128
NUM_HIDDEN_LAYERS = 2

LOG_STD_MIN = -20
LOG_STD_MAX = 2


# ─────────────────────────────────────────────
# Reward function weights
# ─────────────────────────────────────────────
W_SPEED = 0.6
W_ROTATION = 0.1
W_PROXIMITY = 0.5

W_ACTION_SMOOTHNESS = 0.05
W_MOTOR_POWER = 0.01

EXPLORATION_BONUS = 0.8
AVOIDANCE_BONUS = 0.3
COLLISION_PENALTY = -1.0

GRID_SIZE = 0.2


# ─────────────────────────────────────────────
# Safety limits
# ─────────────────────────────────────────────
EMERGENCY_STOP_DISTANCE_THRESHOLD = 230
MAX_ABS_WHEEL_SPEED = 25.0
MAX_SPEED_CHANGE_PER_STEP = 10.0

STOP_ON_COLLISION = True