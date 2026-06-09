import os
import copy
from pathlib import Path
from robobo_interface import IRobobo, SimulationRobobo

from .constants import (
    ACTIONS, N_EPISODES, MAX_STEPS, W_SPEED,
    FRONT_INDICES, AVOIDANCE_BONUS,
)
from .q_learning import QLearning, discretise
from .reward_original import compute_reward, front_blocked
from .visualize_metrics_qlearning import RLMetrics

RESULTS_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def _ensure_dirs():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def run_episode(
    rob: IRobobo,
    agent: QLearning,
    metrics: RLMetrics,
    training: bool = True,
) -> float:
    """
    Run one full episode of MAX_STEPS — no early termination on collision.
    Collisions are penalised via reward but episode always runs to completion.
    Returns total reward for this episode.
    """
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    visited_cells = set()
    steps_since_new_cell = 0
    total_reward  = 0.0

    irs      = rob.read_irs()
    prev_irs = irs
    state    = discretise(irs)

    for step in range(MAX_STEPS):

        # 1. Select action
        action = agent.select_action(state)
        left, right, duration = ACTIONS[action]

        # 2. Execute action
        rob.move_blocking(left, right, duration)

        # 3. Observe next state
        next_irs   = rob.read_irs()
        next_state = discretise(next_irs)

        # 4. Get position (sim only)
        position = rob.get_position() if isinstance(rob, SimulationRobobo) else None

        # 5. Compute reward
        prev_cell_count = len(visited_cells)
        reward, visited_cells = compute_reward(
            action, next_irs, prev_irs, position, visited_cells,
            steps_since_new_cell
        )
        total_reward += reward

        # 6. Track new cells
        if len(visited_cells) > prev_cell_count:
            steps_since_new_cell = 0
            metrics.record_new_cell()
        else:
            steps_since_new_cell += 1

        # 7. Update Q-table (training only)
        if training:
            agent.update(state, action, reward, next_state, done=False)

        # 8. Compute components for logging
        max_speed = 25.0
        s_trans   = max((left + right) / (2 * max_speed), 0.0)
        collision = any(next_irs[i] > 100 for i in FRONT_INDICES)
        prev_v    = max(prev_irs[i] for i in FRONT_INDICES) / 255.0
        curr_v    = max(next_irs[i] for i in FRONT_INDICES) / 255.0
        avoidance = AVOIDANCE_BONUS if (prev_v > 0.4 and curr_v < 0.2) else 0.0

        # 9. Log step metrics
        metrics.record_step(
            action=action,
            irs=next_irs,
            reward=reward,
            epsilon=agent.epsilon,
            speed=s_trans,
            collision=collision,
            avoidance=avoidance,
        )

        prev_irs = next_irs
        state    = next_state

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    return total_reward


def train(rob: IRobobo) -> tuple:
    """Full training loop over N_EPISODES."""
    _ensure_dirs()
    agent   = QLearning()
    metrics = RLMetrics(label="Training")

    print(f"Starting training — {N_EPISODES} episodes, max {MAX_STEPS} steps each")
    print(f"Initial epsilon: {agent.epsilon:.2f}")
    print(f"Results will be saved to: {RESULTS_DIR}")

    best_reward    = float("-inf")
    best_q_table   = None

    for ep in range(N_EPISODES):
        ep_reward = run_episode(rob, agent, metrics, training=True)
        cells     = metrics.cells_this_episode

        agent.decay_epsilon()
        metrics.end_episode(ep_reward)

        # save best Q-table snapshot whenever we beat the best reward
        # only consider episodes after epsilon has decayed enough (ep >= 20)
        # to avoid saving a lucky random episode early on
        if ep >= 20 and ep_reward > best_reward:
            best_reward  = ep_reward
            best_q_table = copy.deepcopy(agent.q_table)
            agent.save(str(RESULTS_DIR / "q_table_best.pkl"))
            print(f"  [ep {ep:>3}] New best reward: {best_reward:.2f} — q_table_best.pkl updated")

        if ep % 10 == 0:
            print(
                f"Episode {ep:>4}/{N_EPISODES} | "
                f"reward: {ep_reward:>7.2f} | "
                f"epsilon: {agent.epsilon:.3f} | "
                f"cells visited: {cells}"
            )

    # save final Q-table (last episode) separately from best
    agent.save(str(RESULTS_DIR / "q_table_final.pkl"))
    metrics.save_raw(str(RESULTS_DIR / "training_data.json"))

    # generate plots
    metrics.plot_training(save_path=str(FIGURES_DIR) + "/")
    RLMetrics.plot_epsilon_decay(metrics, save_path=str(FIGURES_DIR) + "/")

    print(f"Training complete.")
    print(f"  Best reward achieved : {best_reward:.2f}  -> q_table_best.pkl")
    print(f"  Final Q-table        -> q_table_final.pkl")

    return agent, metrics


def validate(
    rob: IRobobo,
    agent: QLearning,
    train_metrics: RLMetrics = None,
    n_runs: int = 5,
    label: str = "val1",
) -> RLMetrics:
    metrics = RLMetrics(label=label)
    """Run validation episodes with epsilon=0 (pure exploitation)."""
    _ensure_dirs()
    original_epsilon = agent.epsilon
    agent.epsilon    = 0.0

    metrics = RLMetrics(label="Validation")
    print(f"\nValidating — {n_runs} runs, epsilon=0 (greedy policy)")

    for run in range(n_runs):
        ep_reward = run_episode(rob, agent, metrics, training=False)
        metrics.end_episode(ep_reward)
        print(f"  Validation run {run + 1}/{n_runs} | reward: {ep_reward:.2f}")

    metrics.save_raw(str(RESULTS_DIR / f"validation_data_{label}.json"))
    metrics.plot_training(save_path=str(FIGURES_DIR) + "/")

    if train_metrics is not None:
        RLMetrics.plot_training_vs_validation(
            train_metrics, metrics,
            save_path=str(FIGURES_DIR) + "/"
        )
        RLMetrics.plot_boxplot(
            train_metrics, metrics,
            save_path=str(FIGURES_DIR) + "/"
        )

    agent.epsilon = original_epsilon
    return metrics