from pathlib import Path
import numpy as np
import cv2

from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo

from .constants_sac import (
    N_EPISODES,
    MAX_STEPS,
    LEARNING_STARTS,
    UPDATES_PER_STEP,
    RED_ACQUIRED_SIZE,
    RED_CENTER_THRESHOLD,
    RED_ACQUIRED_STEPS,
    MIN_AGENT_STEPS,
    RED_LOST_STEPS,
)
from .sac import SAC_RL
from .vision import analyse_frame, vision_features
from .reward_sac import compute_reward
from .visualize_metrics import RLMetrics

RESULTS_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def _ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_episode(
    rob: IRobobo,
    agentred: SAC_RL,
    agentgreen: SAC_RL,
    metrics: RLMetrics,
    training: bool = True,
    hardware: bool = False,
) -> tuple:
    """
    Run one SAC episode for Task 3 pushing.
 
    The robot must locate a bright red object, push it using its front
    grabber, and drive it into a green goal area.
 
    State: [IR (8)] + [vision (6)] + [last_action (2)] = 16 dims
 
    Vision features (6):
        red_visible, red_dx, red_size,
        goal_visible, goal_dx, goal_size
 
    Agent switching:
        - "red"   -> "green": once the red object has been centred and
                     close enough for RED_ACQUIRED_STEPS consecutive
                     steps (and at least MIN_AGENT_STEPS have passed
                     since the last switch).
        - "green" -> "red": once the red object has been NOT visible
                     for RED_LOST_STEPS consecutive steps while the
                     green agent is active (and at least MIN_AGENT_STEPS
                     have passed since the last switch). This is the
                     safeguard for losing contact with the object while
                     pushing — without it the episode would keep running
                     the push agent with nothing to push.
 
    Episode terminates on:
        - goal_reached: red object centre pixel is inside the green goal mask
        - 3 consecutive wall collision steps after LEARNING_STARTS
        - MAX_STEPS reached
 
    Returns
    -------
    total_reward : float
    successes    : int — 1 if goal reached this episode, else 0
    """
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
 
        # Set simulation speed to 4x
        rob._sim.setInt32Param(rob._sim.intparam_speedmodifier, 4)
 
    use_sim              = isinstance(rob, SimulationRobobo)
    visited_cells        = set()
    total_reward         = 0.0
    steps_since_contact  = 0
    steps_since_goal_visible = 0
    last_action          = np.zeros(2, dtype=np.float32)
    consecutive_collisions = 0
    steps_centered_red   = 0
    steps_red_lost       = 0
    steps_since_switch   = 0
    current_agent        = "red"
    agent                = agentred
 
    # Tilt camera downward to keep the red object and ground-level goal in frame.
    rob.set_phone_tilt_blocking(220, 200)
 
    irs = rob.read_irs()
    try:
        img = rob.read_image_front()
        vis = analyse_frame(img, hardware=hardware)
    except Exception:
        vis = {
            "red_visible": False, "red_dx": 0.0, "red_size": 0.0,
            "goal_visible": False, "goal_dx": 0.0, "goal_size": 0.0,
            "goal_reached": False,
        }
 
    for step in range(MAX_STEPS):
 
        # 1. Build vision feature vector [red_visible, red_dx, red_size,
        #                                  goal_visible, goal_dx, goal_size]
        if step > 0:
            img = rob.read_image_front()
            vis_arr = vision_features(
                img,
                hardware=hardware,
            )
            cv2.imwrite(str(FIGURES_DIR / "photo.png"), img)
 
        else:
            vis_arr = np.array([
                1.0 if vis["red_visible"]  else 0.0,
                vis["red_dx"],
                vis["red_size"],
                1.0 if vis["goal_visible"] else 0.0,
                vis["goal_dx"],
                vis["goal_size"],
            ], dtype=np.float32)
 
        # 2. Select action
        motor_command, raw_action = agent.select_action(
            ir_values=irs,
            vision_feats=vis_arr,
            last_action=last_action,
            evaluate=not training,
        )
        left, right, duration = motor_command
 
        # 3. Execute
        rob.move_blocking(left, right, duration)
 
        # 4. Observe
        next_irs = rob.read_irs()
        try:
            next_img = rob.read_image_front()
            next_vis = analyse_frame(next_img, hardware=hardware)
        except Exception:
            next_vis = {
                "red_visible": False, "red_dx": 0.0, "red_size": 0.0,
                "goal_visible": False, "goal_dx": 0.0, "goal_size": 0.0,
                "goal_reached": False,
            }
 
        next_vis_arr = np.array([
            1.0 if next_vis["red_visible"]  else 0.0,
            next_vis["red_dx"],
            next_vis["red_size"],
            1.0 if next_vis["goal_visible"] else 0.0,
            next_vis["goal_dx"],
            next_vis["goal_size"],
        ], dtype=np.float32)
 
        steps_since_switch += 1

        # ── Track sustained "red centred & close" for red -> green ────
        if (
            next_vis["red_visible"]
            and abs(next_vis["red_dx"]) < RED_CENTER_THRESHOLD
            and next_vis["red_size"] > RED_ACQUIRED_SIZE
        ):
            steps_centered_red += 1
        else:
            steps_centered_red -= 1

        steps_centered_red = max(0, min(10, steps_centered_red))

        if not next_vis["goal_visible"]:
            steps_since_goal_visible += 1
        else:
            steps_since_goal_visible = 0

        # ── Track sustained "red not visible" for green -> red ────────
        if not next_vis["red_visible"]:
            steps_red_lost += 1
        else:
            steps_red_lost -= 1

        steps_red_lost = max(0, min(10, steps_red_lost))

        if (
            current_agent == "red"
            and steps_centered_red >= RED_ACQUIRED_STEPS
            and steps_since_switch >= MIN_AGENT_STEPS
        ):
            current_agent = "green"
            agent = agentgreen
            steps_since_switch = 0
            steps_red_lost = 0
            switch_mode = current_agent
            print("switch_to_green")
 
        elif (
            current_agent == "green"
            and steps_red_lost >= RED_LOST_STEPS
            and steps_since_switch >= MIN_AGENT_STEPS
        ):
            current_agent = "red"
            agent = agentred
            steps_since_switch = 0
            steps_centered_red = 0
            switch_mode = current_agent
            print("switch_to_red")
        else:
            switch_mode = None
 
        # 5. Position for exploration grid (sim only; hardware has no odometry)
        position = rob.get_position() if use_sim else None
        
        # 6. Compute reward
        reward, collision, steps_since_contact, visited_cells, info = compute_reward(
            mode = current_agent,
            switch_mode=switch_mode,
            left_speed=left,
            right_speed=right,
            irs=next_irs,
            vision=next_vis,
            prev_vision=vis,
            goal_reached=next_vis["goal_reached"],
            current_step=step,
            steps_since_contact=steps_since_contact,
            steps_since_goal_visible=steps_since_goal_visible,
            position=position,
            visited_cells=visited_cells,
            frame=next_img if "next_img" in dir() else None,
        )
        total_reward += reward
 
        # 7. Done — require 3 consecutive wall-collision steps to avoid
        #    terminating on a single IR spike; always terminate on goal_reached.
        consecutive_collisions = (consecutive_collisions + 1) if collision else 0
        done = (
            next_vis["goal_reached"]
            or (consecutive_collisions >= 3 and agent.total_steps > LEARNING_STARTS)
        )
 
        # 8. Store transition & update
        if training:
            agent.store_transition(
                state_ir=irs,
                action=raw_action,
                reward=reward,
                next_state_ir=next_irs,
                done=done,
                vision_feats=vis_arr,
                next_vision_feats=next_vis_arr,
                last_action=last_action,
                next_last_action=raw_action,
            )
            for _ in range(UPDATES_PER_STEP):
                agent.update()
 
        # 9. Log
        s_trans = max((left + right) / (2.0 * 25.0), 0.0)
        metrics.record_step(
            action={"left": float(left), "right": float(right)},
            irs=next_irs,
            reward=reward,
            epsilon=0.0,
            speed=s_trans,
            collision=collision,
            avoidance=max(info["red_approach"], 0.0),
            food_collected=1 if next_vis["goal_reached"] else 0,
            obj_visible=next_vis["red_visible"],
            obj_size=next_vis["red_size"],
        )
 
        # 10. Shift state
        vis         = next_vis
        irs         = next_irs
        last_action = raw_action
 
        if done:
            break
 
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
 
    successes = 1 if next_vis["goal_reached"] else 0
    return total_reward, successes


def train(rob: IRobobo) -> tuple:
    """Full SAC training loop for Task 3 pushing."""
    _ensure_dirs()
    agentred    = SAC_RL()
    agentgreen  = SAC_RL()
    metrics  = RLMetrics(label="Training")
    hardware = isinstance(rob, HardwareRobobo)

    print(f"Starting SAC training — {N_EPISODES} episodes, {MAX_STEPS} steps each")
    print(f"State dimension: {agentred.state_dim}  (IR + vision + last_action)")
    print(f"Results -> {RESULTS_DIR}")

    best_reward = float("-inf")

    for ep in range(N_EPISODES):
        ep_reward, ep_success = run_episode(
            rob=rob, agentred=agentred, agentgreen=agentgreen, metrics=metrics,
            training=True, hardware=hardware,
        )
        cells = metrics.cells_this_episode
        metrics.end_episode(ep_reward, food_collected=ep_success)

        if ep >= 20 and ep_reward > best_reward:
            best_reward = ep_reward
            agentred.save("sac_agentred_best.pt")
            agentgreen.save("sac_agentgreen_best.pt")
            print(f"  [ep {ep:>3}] New best: {best_reward:.2f} (success={ep_success})")

        print(
            f"Episode {ep:>4}/{N_EPISODES} | "
            f"reward: {ep_reward:>7.2f} | "
            f"success: {ep_success} | "
            f"buffer: {len(agentred.replay_buffer)} | "
            f"cells: {cells}"
        )

    agentred.save("sac_agentred_final.pt")
    agentgreen.save("sac_agentgreen_final.pt")
    metrics.save_raw(str(RESULTS_DIR / "training_data_sac.json"))
    metrics.plot_training(save_path=str(FIGURES_DIR) + "/")
    metrics.plot_push_metrics(save_path=str(FIGURES_DIR) + "/")

    print(f"Training complete. Best reward: {best_reward:.2f}")
    return agentred, agentgreen, metrics


def validate(
    rob: IRobobo,
    agentred: SAC_RL,
    agentgreen: SAC_RL,
    train_metrics: RLMetrics = None,
    n_runs: int = 5,
    label: str = "val1",
) -> RLMetrics:
    """Run SAC validation with deterministic policy."""
    _ensure_dirs()
    metrics  = RLMetrics(label=label)
    hardware = isinstance(rob, HardwareRobobo)

    print(f"\nValidating SAC — {n_runs} deterministic runs [{label}]")

    for run in range(n_runs):
        ep_reward, ep_success = run_episode(
            rob=rob, agentred=agentred, agentgreen=agentgreen, metrics=metrics,
            training=False, hardware=hardware,
        )
        metrics.end_episode(ep_reward, food_collected=ep_success)
        print(f"  Run {run + 1}/{n_runs} | reward: {ep_reward:.2f} | success: {ep_success}")

    metrics.save_raw(str(RESULTS_DIR / f"validation_data_sac_{label}.json"))
    metrics.plot_training(save_path=str(FIGURES_DIR) + "/")
    metrics.plot_push_metrics(save_path=str(FIGURES_DIR) + "/")

    if train_metrics is not None:
        RLMetrics.plot_training_vs_validation(
            train_metrics, metrics, save_path=str(FIGURES_DIR) + "/",
        )
        RLMetrics.plot_boxplot(
            train_metrics, metrics, save_path=str(FIGURES_DIR) + "/",
        )

    return metrics
