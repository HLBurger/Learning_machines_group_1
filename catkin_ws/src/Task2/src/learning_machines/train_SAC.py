import copy
from pathlib import Path
import numpy as np

from robobo_interface import IRobobo, SimulationRobobo

from .constants_sac import (
    N_EPISODES,
    MAX_STEPS,
    FRONT_INDICES,
    AVOIDANCE_BONUS,
)
from .sac import SAC_RL
from .vision import classify_collision, analyse_frame
from .odometry_map import OdometryMap
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
    stop_on_collision: bool = True,
) -> float:
    """
    Run one SAC episode with odometry-based map exploration.

    The agent observes:
        [IR sensors (8)] + [pose (4)] + [local occupancy window (11×11=121)]
    Total state dimension: 133 (STATE_DIM in constants_sac.py)

    The OdometryMap is reset at the start of every episode.  In simulation
    it uses ground-truth XY position; on real hardware it dead-reckons from
    wheel commands.

    Returns
    -------
    total_reward : float
    """

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # ── Odometry map, reset each episode ──────────────────────────────
    use_sim = isinstance(rob, SimulationRobobo)
    odom_map = OdometryMap(use_sim_position=use_sim)

    total_reward = 0.0

    irs      = rob.read_irs()
    prev_irs = irs
    prev_img = None

    for step in range(MAX_STEPS):

        # 1. Build current observation ─────────────────────────────────
        pose      = odom_map.normalised_pose()   # (4,)
        local_map = odom_map.local_window()      # (LOCAL_MAP_WINDOW²,)
        img       = rob.read_image_front()

        vis       = analyse_frame(img)
        vision_feats = np.array([
            float(vis["object_visible"]),
            vis["object_dx"],
            vis["object_size"],
            float(vis["wall_visible"]),
            vis["wall_frac"],
        ], dtype=np.float32)

        # 2. Select action ─────────────────────────────────────────────
        motor_command, raw_action = agent.select_action(
            ir_values=irs,
            pose=pose,
            local_map=local_map,
            vision_feats=vision_feats,
            evaluate=not training,
        )

        left, right, duration = motor_command

        # 3. Execute ───────────────────────────────────────────────────
        rob.move_blocking(left, right, duration)

        # 4. Observe next IR state ─────────────────────────────────────
        next_irs = rob.read_irs()

        # 5. Ground-truth position (sim only) ──────────────────────────
        sim_position = rob.get_position() if use_sim else None

        # 6. Compute reward & advance odometry map ─────────────────────
        #    compute_reward now calls odom_map.update() internally, which
        #    integrates the pose AND marks the new cell on the map.
        reward, new_cell, collision, collision_type = compute_reward(
            left_speed=left,
            right_speed=right,
            irs=next_irs,
            prev_irs=prev_irs,
            odom_map=odom_map,
            sim_position=sim_position,
            frame=img,
            prev_frame=prev_img,
        )

        total_reward += reward
        print(f"collision_type: {collision_type}")
        if new_cell:
            metrics.record_new_cell()

        # 7. Build NEXT observation (map already updated by compute_reward)
        next_pose      = odom_map.normalised_pose()
        next_local_map = odom_map.local_window()
        next_img       = rob.read_image_front()

        next_vis        = analyse_frame(next_img)
        next_vision_feats = np.array([
            float(next_vis["object_visible"]),
            next_vis["object_dx"],
            next_vis["object_size"],
            float(next_vis["wall_visible"]),
            next_vis["wall_frac"],
        ], dtype=np.float32)

        # 8. Termination ───────────────────────────────────────────────
        done = bool(collision and stop_on_collision)

        # 9. Store transition & update SAC ─────────────────────────────
        if training:
            agent.store_transition(
                state_ir=irs,
                action=raw_action,
                reward=reward,
                next_state_ir=next_irs,
                done=done,
                pose=pose,
                next_pose=next_pose,
                local_map=local_map,
                next_local_map=next_local_map,
                vision_feats=vision_feats,
                next_vision_feats=next_vision_feats,
            )

            losses = agent.update()
        else:
            losses = None

        # 10. Logging ──────────────────────────────────────────────────
        max_speed = 25.0
        s_trans = max((left + right) / (2.0 * max_speed), 0.0)

        prev_v = max(prev_irs[i] for i in FRONT_INDICES) / 255.0
        curr_v = max(next_irs[i] for i in FRONT_INDICES) / 255.0
        avoidance = AVOIDANCE_BONUS if (prev_v > 0.4 and curr_v < 0.2) else 0.0

        action_for_logging = {
            "left": float(left),
            "right": float(right),
            "duration": int(duration),
        }

        metrics.record_step(
            action=action_for_logging,
            irs=next_irs,
            reward=reward,
            epsilon=0.0,
            speed=s_trans,
            collision=collision,
            avoidance=avoidance,
            vision_feats=vision_feats,
        )

        prev_irs     = next_irs
        irs          = next_irs
        prev_img     = img
        vision_feats = next_vision_feats
        img          = next_img

        if done:
            break

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    return total_reward


def train(rob: IRobobo) -> tuple:
    """Full SAC training loop over N_EPISODES."""
    _ensure_dirs()

    agent   = SAC_RL()
    metrics = RLMetrics(label="Training")

    print(f"Starting SAC training — {N_EPISODES} episodes, max {MAX_STEPS} steps each")
    print(f"State dimension: {agent.state_dim}  (IR + pose + local map)")
    print(f"Results will be saved to: {RESULTS_DIR}")

    best_reward    = float("-inf")
    best_model_path = RESULTS_DIR / "sac_agent_best.pt"

    for ep in range(N_EPISODES):
        ep_reward = run_episode(
            rob=rob,
            agent=agent,
            metrics=metrics,
            training=True,
            stop_on_collision=True,
        )

        cells = metrics.cells_this_episode
        metrics.end_episode(ep_reward)

        if ep >= 20 and ep_reward > best_reward:
            best_reward = ep_reward
            agent.save("sac_agent_best.pt")
            print(
                f"  [ep {ep:>3}] New best reward: "
                f"{best_reward:.2f} — sac_agent_best.pt updated"
            )

        if ep % 10 == 0:
            print(
                f"Episode {ep:>4}/{N_EPISODES} | "
                f"reward: {ep_reward:>7.2f} | "
                f"buffer size: {len(agent.replay_buffer)} | "
                f"cells visited: {cells}"
            )

    agent.save("sac_agent_final.pt")
    metrics.save_raw(str(RESULTS_DIR / "training_data_sac.json"))
    metrics.plot_training(save_path=str(FIGURES_DIR) + "/")

    print("SAC training complete.")
    print(f"  Best reward achieved : {best_reward:.2f} -> sac_agent_best.pt")
    print("  Final SAC model      -> sac_agent_final.pt")

    return agent, metrics


def validate(
    rob: IRobobo,
    agent: SAC_RL,
    train_metrics: RLMetrics = None,
    n_runs: int = 5,
    label: str = "val1",
) -> RLMetrics:
    """Run SAC validation episodes with deterministic actions."""
    _ensure_dirs()

    metrics = RLMetrics(label="Validation")

    print(f"\nValidating SAC — {n_runs} deterministic runs")

    for run in range(n_runs):
        ep_reward = run_episode(
            rob=rob,
            agent=agent,
            metrics=metrics,
            training=False,
            stop_on_collision=True,
        )

        metrics.end_episode(ep_reward)
        print(f"  Validation run {run + 1}/{n_runs} | reward: {ep_reward:.2f}")

    metrics.save_raw(str(RESULTS_DIR / f"validation_data_sac_{label}.json"))
    metrics.plot_training(save_path=str(FIGURES_DIR) + "/")
    metrics.plot_vision_validation(save_path=str(FIGURES_DIR) + "/")

    if train_metrics is not None:
        RLMetrics.plot_training_vs_validation(
            train_metrics, metrics, save_path=str(FIGURES_DIR) + "/",
        )
        RLMetrics.plot_boxplot(
            train_metrics, metrics, save_path=str(FIGURES_DIR) + "/",
        )

    return metrics
