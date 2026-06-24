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
BACK_INDICES  = [v for k, v in IR_indices.items() if k.endswith("R")]
LEFT_INDICES  = [v for k, v in IR_indices.items() if k.endswith("L")]
RIGHT_INDICES = [v for k, v in IR_indices.items() if k.endswith("R")]

# ─────────────────────────────────────────────
# State dimension
# Walls are white — IR signals wall proximity only.
# Camera detects both the bright red object and the green goal area.
#
# [IR (8)] + [obj_visible, obj_dx, obj_size,
#              goal_visible, goal_dx, goal_size (6)] + [last_action (2)] = 16
# ─────────────────────────────────────────────
IR_STATE_DIM    = len(IR_indices)                     # 8
VISION_DIM      = 6   # red_visible, red_dx, red_size, goal_visible, goal_dx, goal_size
LAST_ACTION_DIM = 2   # left_speed, right_speed from previous step
STATE_DIM       = IR_STATE_DIM + VISION_DIM + LAST_ACTION_DIM  # 16

# ─────────────────────────────────────────────
# Continuous action space
# ─────────────────────────────────────────────
ACTION_DIM         = 2
MIN_WHEEL_SPEED    = -25.0
MAX_WHEEL_SPEED    = 100
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
UPDATES_PER_STEP   = 4      # more gradient steps per env step = faster learning
N_EPISODES         = 200
MAX_STEPS          = 200    # 3 min per episode at 300ms/step (matches real competition)

# ─────────────────────────────────────────────
# Entropy / exploration
# ─────────────────────────────────────────────
AUTO_ENTROPY_TUNING = True
TARGET_ENTROPY      = -ACTION_DIM
INIT_ALPHA          = 0.2

# ─────────────────────────────────────────────
# Network architecture
# ─────────────────────────────────────────────
HIDDEN_DIM  = 64     # 16-dim state is simple; smaller net converges faster
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# ─────────────────────────────────────────────
# Reward — Task 3 pushing
# Walls are white → IR = wall signal only.
# Red object = push target; green area = goal.
# Robot must align with the red object, push it forward, and
# drive it into the green goal area.
#
# Weighting philosophy: the two real objectives are (1) reach the red
# object and (2) push it into the green goal. Those terms — and the
# terminal success reward — should dominate. Everything else here
# (speed regulation, exploration, wall proximity, anti-stuck) is
# scaffolding that exists to make the two objectives achievable; it is
# sized below the task terms so it nudges behaviour without competing
# with it for the agent's attention.
# ─────────────────────────────────────────────

# Success
OBJECT_IN_GOAL_REWARD = 50.0   # large terminal bonus when red centre lands inside green mask
FAST_COMPLETION_BONUS = 15.0   # extra bonus scaled by steps remaining (faster = bigger)

# Approach red object (objective 1)
W_RED_CENTERING   = 0.8   # reward for keeping red object centred horizontally
W_RED_APPROACH    = 1.0   # reward for red blob size (robot getting closer to object) — raised above centering, since closing distance matters more than fine alignment early on

# Align with goal once object is close (objective 2)
W_GOAL_CENTERING  = 3.0   # reward for keeping green goal centred while pushing — raised so active goal-pushing clearly outweighs scaffolding terms below

# Forward push reward — fires when robot is moving forward with red object in contact
PUSH_FORWARD_REWARD = 0.5  # reward per step of active forward pushing

# Penalty when red object is lost from view during pushing phase
RED_LOST_PENALTY  = -0.3   # small per-step penalty to keep robot seeking object

W_SPEED_SEARCH    = 0.4   # reward for moving fast when red object not visible
W_SPEED_WALL      = 0.5   # penalty coefficient for moving fast near a wall

W_PROXIMITY       = 0.6   # penalty weight proportional to front IR normalised reading


W_STUCK_PENALTY      = -0.3   # applied per step where net speed < threshold
STUCK_SPEED_THRESHOLD = 2.0   # abs(left+right)/2 below this = stuck

GRID_SIZE         = 0.2
EXPLORATION_BONUS = 0.5   # incentive to visit new grid cells while searching

# Urgency — grows if no successful push event for a while
URGENCY_PENALTY   = -0.05  # -0.05/step × 100 cap = -5 max
MAX_URGENCY_STEPS = 100

# Collision — full penalty, not fractional; applied only for wall contacts
COLLISION_PENALTY = -3.0

# ─────────────────────────────────────────────
# Vision / colour detection
# Walls are white → no wall/food confusion in HSV space.
# Red object wraps around H=0/180 so two ranges are needed.
# Green goal area uses a single range.
# ─────────────────────────────────────────────
RED1_LOWER_SIM = (0,   120,  80)
RED1_UPPER_SIM = (10,  255, 255)
RED2_LOWER_SIM = (170, 120,  80)
RED2_UPPER_SIM = (180, 255, 255)
RED1_LOWER_HW  = (0,   100,  60)
RED1_UPPER_HW  = (12,  255, 255)
RED2_LOWER_HW  = (165, 100,  60)
RED2_UPPER_HW  = (180, 255, 255)

GREEN_LOWER_SIM = (40,  60,  40)
GREEN_UPPER_SIM = (85, 255, 255)
GREEN_LOWER_HW  = (35,  50,  40)   # broader for lab lighting
GREEN_UPPER_HW  = (90, 255, 255)

MIN_RED_AREA_FRAC  = 0.001   # minimum red blob size as fraction of frame area
MIN_GOAL_AREA_FRAC = 0.002   # minimum green goal size as fraction of frame area

GOAL_REACHED_DILATE_ITERS = 3  # tune based on tile size vs. object size

RED_ACQUIRED_SIZE = 0.02
RED_CENTER_THRESHOLD = 0.10

RED_ACQUIRED_STEPS = 10
MIN_AGENT_STEPS = 10
RED_LOST_STEPS = 10  # consecutive steps red must be absent before green -> red switch-back