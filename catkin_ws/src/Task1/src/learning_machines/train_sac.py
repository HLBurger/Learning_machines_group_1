import copy
from pathlib import Path

from robobo_interface import IRobobo, SimulationRobobo

from .constants import (
    N_EPISODES,
    MAX_STEPS,
    FRONT_INDICES,
    AVOIDANCE_BONUS,
)
from .sac import SAC_RL
from .reward import compute_reward, front_blocked
from .visualize_metrics import RLMetrics
from .vision import build_state


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
    stop_on_collision: bool = False,
) -> float:
    """
    Run one SAC episode.

    SAC uses continuous actions:
        raw_action = normalized action in [-1, 1]
        motor_command = left_speed, right_speed, duration_ms

    The replay buffer stores:
        state, raw_action, reward, next_state, done

    Parameters
    ----------
    rob:
        Robobo interface.

    agent:
        SAC_RL agent.

    metrics:
        RLMetrics logger.

    training:
        If True, use stochastic SAC actions and update the networks.
        If False, use deterministic actions and do not update.

    stop_on_collision:
        If True, end episode immediately after collision.
        If False, continue until MAX_STEPS, like your Q-learning version.

    Returns
    -------
    total_reward:
        Total reward collected during the episode.
    """

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    visited_cells = set()
    total_reward = 0.0

    state = build_state(rob)
    prev_irs = state[:8]

    previous_raw_action = None

    for step in range(MAX_STEPS):

        # 1. Select continuous SAC action
        # motor_command = scaled command for robot
        # raw_action = normalized action in [-1, 1] for replay buffer
        motor_command, raw_action = agent.select_action(
            state_values=state,
            evaluate=not training,
        )

        left, right, duration = motor_command

        # 2. Execute motor command
        rob.move_blocking(left, right, duration)

        # 3. Observe next state
        next_state = build_state(rob)
        next_irs = next_state[:8]

        # 4. Get position, simulation only
        position = rob.get_position() if isinstance(rob, SimulationRobobo) else None

        # 5. Compute reward
        prev_cell_count = len(visited_cells)

        reward, visited_cells, collision = compute_reward(
            left, right, next_irs, prev_irs, position, visited_cells, next_state=next_state,
        )

        total_reward += reward

        # 6. Track new explored cells
        if len(visited_cells) > prev_cell_count:
            metrics.record_new_cell()

        # 7. Determine episode termination
        done = bool(collision and stop_on_collision)

        # 8. Store transition and update SAC
        if training:
            agent.store_transition(
                state=state,
                action=raw_action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            # SAC can do one or more updates per environment step.
            # The agent.update() method should internally skip updates
            # until the replay buffer has enough samples.
            losses = agent.update()
        else:
            losses = None

        # 9. Compute logging components
        max_speed = 25.0
        s_trans = max((left + right) / (2.0 * max_speed), 0.0)

        prev_v = max(prev_irs[i] for i in FRONT_INDICES) / 255.0
        curr_v = max(next_irs[i] for i in FRONT_INDICES) / 255.0
        avoidance = AVOIDANCE_BONUS if (prev_v > 0.4 and curr_v < 0.2) else 0.0

        # SAC has no discrete action index.
        # For old metrics code that expects "action", log a readable tuple instead.
        action_for_logging = {
            "left": float(left),
            "right": float(right),
            "duration": int(duration),
        }

        # 10. Log step metrics
        # If your RLMetrics.record_step only accepts an int action,
        # see the compatibility option below.
        metrics.record_step(
            action=action_for_logging,
            irs=next_irs,
            reward=reward,
            epsilon=0.0,
            speed=s_trans,
            collision=collision,
            avoidance=avoidance,
        )

        previous_raw_action = raw_action
        prev_irs = next_irs
        state = next_state

        if done:
            break

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    return total_reward

def train(rob: IRobobo) -> tuple:
    """
    Full SAC training loop over N_EPISODES.
    """
    _ensure_dirs()

    agent = SAC_RL()
    metrics = RLMetrics(label="Training")

    print(f"Starting SAC training — {N_EPISODES} episodes, max {MAX_STEPS} steps each")
    print(f"Results will be saved to: {RESULTS_DIR}")

    best_reward = float("-inf")
    best_model_path = RESULTS_DIR / "sac_agent_best.pt"

    for ep in range(N_EPISODES):
        ep_reward = run_episode(
            rob=rob,
            agent=agent,
            metrics=metrics,
            training=True,
            stop_on_collision=False,
        )

        cells = metrics.cells_this_episode
        metrics.end_episode(ep_reward)

        # Save best model after some initial exploration/training
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

    # Save final SAC model
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
    """
    Run SAC validation episodes.

    During validation, SAC uses deterministic actions:
        evaluate=True

    There is no epsilon in SAC validation.
    """
    _ensure_dirs()

    metrics = RLMetrics(label="Validation")

    print(f"\nValidating SAC — {n_runs} deterministic runs")

    for run in range(n_runs):
        ep_reward = run_episode(
            rob=rob,
            agent=agent,
            metrics=metrics,
            training=False,
            stop_on_collision=False,
        )

        metrics.end_episode(ep_reward)

        print(
            f"  Validation run {run + 1}/{n_runs} | "
            f"reward: {ep_reward:.2f}"
        )

    metrics.save_raw(str(RESULTS_DIR / f"validation_data_sac_{label}.json"))
    metrics.plot_training(save_path=str(FIGURES_DIR) + "/")

    if train_metrics is not None:
        RLMetrics.plot_training_vs_validation(
            train_metrics,
            metrics,
            save_path=str(FIGURES_DIR) + "/",
        )

        RLMetrics.plot_boxplot(
            train_metrics,
            metrics,
            save_path=str(FIGURES_DIR) + "/",
        )

    return metrics