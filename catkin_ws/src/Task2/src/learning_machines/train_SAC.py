import copy
from pathlib import Path
import numpy as np

from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo

from .constants_sac import (
    N_EPISODES, MAX_STEPS, FRONT_INDICES, ACTION_DURATION_MS,
)
from .sac import SAC_RL
from .vision import analyse_frame, vision_features, no_vision
from .reward_sac import compute_reward, front_blocked
from .visualize_metrics import RLMetrics

RESULTS_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def _ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_episode(
    rob: IRobobo,
    agent: SAC_RL,
    metrics: RLMetrics,
    training: bool = True,
    hardware: bool = False,
) -> tuple:
    """
    Run one SAC episode for Task 2 foraging.

    State: [IR (8)] + [vision (3)] + [last_action (2)] = 13 dims
    No odometry map — walls are white, food is green, no classification needed.

    Returns
    -------
    total_reward  : float
    food_count    : int — packages collected this episode
    """
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    use_sim           = isinstance(rob, SimulationRobobo)
    visited_cells     = set()
    total_reward      = 0.0
    food_collected    = 0
    prev_food         = 0
    steps_since_food  = 0
    last_action       = np.zeros(2, dtype=np.float32)

    irs       = rob.read_irs()
    try:
        img = rob.read_image_front()
        vis = analyse_frame(img, hardware=hardware)
    except Exception:
        vis = {"obj_visible": False, "obj_dx": 0.0, "obj_size": 0.0}
    prev_vis = vis.copy()

    for step in range(MAX_STEPS):

        # 1. Build vision feature vector
        vis_arr = vision_features(
            rob.read_image_front() if step > 0 else img,
            hardware=hardware
        ) if step > 0 else np.array([
            1.0 if vis["obj_visible"] else 0.0,
            vis["obj_dx"], vis["obj_size"]
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
            next_img  = rob.read_image_front()
            next_vis  = analyse_frame(next_img, hardware=hardware)
        except Exception:
            next_vis  = {"obj_visible": False, "obj_dx": 0.0, "obj_size": 0.0}

        next_vis_arr = np.array([
            1.0 if next_vis["obj_visible"] else 0.0,
            next_vis["obj_dx"], next_vis["obj_size"]
        ], dtype=np.float32)

        # 5. Food count (sim only)
        if use_sim:
            try:
                food_collected = rob.get_nr_food_collected()
            except Exception:
                food_collected = prev_food

        # 6. Position for exploration grid (sim only)
        position = rob.get_position() if use_sim else None

        # 7. Compute reward
        reward, collision, steps_since_food, visited_cells, info = compute_reward(
            left_speed=left,
            right_speed=right,
            irs=next_irs,
            vision=next_vis,
            prev_vision=vis,
            food_collected=food_collected,
            prev_food_collected=prev_food,
            current_step=step,
            steps_since_food=steps_since_food,
            position=position,
            visited_cells=visited_cells,
        )
        total_reward += reward

        # 8. Done — terminate on collision so bad states don't pollute the buffer
        done = collision

        # 9. Store & update
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
            agent.update()

        # 10. Log
        s_trans = max((left + right) / (2.0 * 25.0), 0.0)
        metrics.record_step(
            action={"left": float(left), "right": float(right)},
            irs=next_irs,
            reward=reward,
            epsilon=0.0,
            speed=s_trans,
            collision=collision,
            avoidance=max(info["approach"], 0.0),
            food_collected=food_collected,
            obj_visible=next_vis["obj_visible"],
            obj_size=next_vis["obj_size"],
        )

        # 11. Shift
        prev_food   = food_collected
        prev_vis    = next_vis
        vis         = next_vis
        vis_arr_old = vis_arr
        irs         = next_irs
        last_action = raw_action

        if done:
            break

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    return total_reward, food_collected


def train(rob: IRobobo) -> tuple:
    """Full SAC training loop."""
    _ensure_dirs()
    agent    = SAC_RL()
    metrics  = RLMetrics(label="Training")
    hardware = isinstance(rob, HardwareRobobo)

    print(f"Starting SAC training — {N_EPISODES} episodes, {MAX_STEPS} steps each")
    print(f"State dimension: {agent.state_dim}  (IR + vision + last_action)")
    print(f"Results -> {RESULTS_DIR}")

    best_reward = float("-inf")

    for ep in range(N_EPISODES):
        ep_reward, ep_food = run_episode(
            rob=rob, agent=agent, metrics=metrics,
            training=True, hardware=hardware,
        )
        cells = metrics.cells_this_episode
        metrics.end_episode(ep_reward, food_collected=ep_food)

        if ep >= 20 and ep_reward > best_reward:
            best_reward = ep_reward
            agent.save("sac_agent_best.pt")
            print(f"  [ep {ep:>3}] New best: {best_reward:.2f} (food={ep_food})")

        print(
            f"Episode {ep:>4}/{N_EPISODES} | "
            f"reward: {ep_reward:>7.2f} | "
            f"food: {ep_food} | "
            f"buffer: {len(agent.replay_buffer)} | "
            f"cells: {cells}"
        )

    agent.save("sac_agent_final.pt")
    metrics.save_raw(str(RESULTS_DIR / "training_data_sac.json"))
    metrics.plot_training(save_path=str(FIGURES_DIR) + "/")
    metrics.plot_food_metrics(save_path=str(FIGURES_DIR) + "/")

    print(f"Training complete. Best: {best_reward:.2f}")
    return agent, metrics


def validate(
    rob: IRobobo,
    agent: SAC_RL,
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
        ep_reward, ep_food = run_episode(
            rob=rob, agent=agent, metrics=metrics,
            training=False, hardware=hardware,
        )
        metrics.end_episode(ep_reward, food_collected=ep_food)
        print(f"  Run {run + 1}/{n_runs} | reward: {ep_reward:.2f} | food: {ep_food}")

    metrics.save_raw(str(RESULTS_DIR / f"validation_data_sac_{label}.json"))
    metrics.plot_training(save_path=str(FIGURES_DIR) + "/")
    metrics.plot_food_metrics(save_path=str(FIGURES_DIR) + "/")

    if train_metrics is not None:
        RLMetrics.plot_training_vs_validation(
            train_metrics, metrics, save_path=str(FIGURES_DIR) + "/",
        )
        RLMetrics.plot_boxplot(
            train_metrics, metrics, save_path=str(FIGURES_DIR) + "/",
        )

    return metrics