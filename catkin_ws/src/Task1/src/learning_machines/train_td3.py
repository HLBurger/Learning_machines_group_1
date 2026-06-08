import os
import json
import numpy as np
from pathlib import Path

from robobo_interface import IRobobo, SimulationRobobo

# ── Optional SB3 import with clear error ─────────────────────────────────────
try:
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as e:
    raise ImportError(
        "stable-baselines3 is required for TD3 training.\n"
        "Install it with:  pip install stable-baselines3>=1.6 gym\n"
        f"Original error: {e}"
    )

from .robobo_env import RoboboEnv
from .constants import (
    TD3_TOTAL_TIMESTEPS, TD3_LEARNING_STARTS, TD3_LEARNING_RATE,
    TD3_BATCH_SIZE, TD3_BUFFER_SIZE, TD3_TRAIN_FREQ, TD3_GRADIENT_STEPS,
    TD3_ACTION_NOISE_STD, TD3_GAMMA, TD3_TAU, TD3_POLICY_DELAY,
    TD3_MAX_STEPS,
)


RESULTS_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR  = RESULTS_DIR / "models"


def _ensure_dirs():
    for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics container (mirrors RLMetrics interface for visualize_metrics.py)
# ─────────────────────────────────────────────────────────────────────────────

class TD3Metrics:
    """
    Lightweight metrics tracker compatible with the existing RLMetrics plots.
    Episode data is appended by the SB3 callback during training.
    """

    def __init__(self, label: str = "TD3_Training"):
        self.label = label

        self.episode_rewards    = []
        self.episode_steps      = []
        self.episode_cells      = []
        self.episode_speeds     = []
        self.episode_collisions = []
        self.episode_avoidances = []

        # filled by callback
        self._step_rewards     = []
        self._step_speeds      = []
        self._step_collisions  = []
        self._step_avoidances  = []
        self.cells_this_episode = 0
        self.epsilon_history   = []   # not used by TD3, kept for API compat

    def record_step(self, reward, speed, collision, avoidance):
        self._step_rewards.append(reward)
        self._step_speeds.append(speed)
        self._step_collisions.append(float(collision))
        self._step_avoidances.append(avoidance)

    def record_new_cell(self):
        self.cells_this_episode += 1

    def end_episode(self, total_reward: float):
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(len(self._step_rewards))
        self.episode_cells.append(self.cells_this_episode)
        self.episode_speeds.append(
            float(np.mean(self._step_speeds)) if self._step_speeds else 0.0
        )
        self.episode_collisions.append(
            float(np.mean(self._step_collisions)) if self._step_collisions else 0.0
        )
        self.episode_avoidances.append(
            float(np.sum(self._step_avoidances)) if self._step_avoidances else 0.0
        )
        self._step_rewards     = []
        self._step_speeds      = []
        self._step_collisions  = []
        self._step_avoidances  = []
        self.cells_this_episode = 0

    def save_raw(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "label"              : self.label,
            "episode_rewards"    : self.episode_rewards,
            "episode_steps"      : self.episode_steps,
            "episode_cells"      : self.episode_cells,
            "episode_speeds"     : self.episode_speeds,
            "episode_collisions" : self.episode_collisions,
            "episode_avoidances" : self.episode_avoidances,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Raw metrics saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SB3 callback — records per-episode metrics and saves checkpoints
# ─────────────────────────────────────────────────────────────────────────────

class TD3MetricsCallback(BaseCallback):
    """
    SB3 callback that:
    - Captures per-step info dicts from the environment.
    - Aggregates them into TD3Metrics episode summaries.
    - Saves a model checkpoint whenever a new best episode reward is set.
    """

    def __init__(self, metrics: TD3Metrics, save_dir: Path, verbose: int = 1):
        super().__init__(verbose)
        self.metrics      = metrics
        self.save_dir     = save_dir
        self.best_reward  = float("-inf")
        self._ep_reward   = 0.0
        self._ep_step     = 0

    def _on_step(self) -> bool:
        # SB3 stores the most recent info dict in self.locals["infos"]
        info   = self.locals["infos"][0]
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]

        self._ep_reward += reward
        self._ep_step   += 1

        # record step-level data
        speed     = float(info.get("s_trans", 0.0))
        collision = bool(info.get("collision", False))
        avoidance = float(info.get("avoidance", 0.0))
        self.metrics.record_step(reward, speed, collision, avoidance)

        if bool(info.get("cells_visited", 0)) > self.metrics.cells_this_episode:
            self.metrics.record_new_cell()

        if done:
            self.metrics.end_episode(self._ep_reward)

            ep_num = len(self.metrics.episode_rewards)
            if self.verbose and ep_num % 10 == 0:
                print(
                    f"  [ep {ep_num:>4}] "
                    f"reward: {self._ep_reward:>7.2f} | "
                    f"steps: {self._ep_step} | "
                    f"cells: {self.metrics.episode_cells[-1]}"
                )

            # checkpoint best model (skip first 5 episodes of random warmup)
            if ep_num > 5 and self._ep_reward > self.best_reward:
                self.best_reward = self._ep_reward
                self.model.save(str(self.save_dir / "td3_best"))
                if self.verbose:
                    print(
                        f"  → New best reward {self.best_reward:.2f} "
                        f"— td3_best saved"
                    )

            self._ep_reward = 0.0
            self._ep_step   = 0

        return True   # returning False would stop training


# ─────────────────────────────────────────────────────────────────────────────
# Public training entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_td3(rob: IRobobo) -> tuple:
    """
    Build a RoboboEnv, configure TD3 with action noise, run training.

    Returns
    -------
    model   : stable_baselines3.TD3
    metrics : TD3Metrics
    """
    _ensure_dirs()

    env     = RoboboEnv(rob)
    metrics = TD3Metrics(label="TD3_Training")

    # ── Action noise (Gaussian) ───────────────────────────────────────────────
    n_actions    = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean  = np.zeros(n_actions),
        sigma = TD3_ACTION_NOISE_STD * np.ones(n_actions),
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = TD3(
        policy         = "MlpPolicy",
        env            = env,
        action_noise   = action_noise,
        learning_rate  = TD3_LEARNING_RATE,
        buffer_size    = TD3_BUFFER_SIZE,
        learning_starts= TD3_LEARNING_STARTS,
        batch_size     = TD3_BATCH_SIZE,
        tau            = TD3_TAU,
        gamma          = TD3_GAMMA,
        train_freq     = TD3_TRAIN_FREQ,
        gradient_steps = TD3_GRADIENT_STEPS,
        policy_delay   = TD3_POLICY_DELAY,
        verbose        = 0,
        device         = "cuda"
    )

    callback = TD3MetricsCallback(metrics, save_dir=MODELS_DIR, verbose=1)

    print(f"Starting TD3 training — {TD3_TOTAL_TIMESTEPS:,} total timesteps")
    print(f"Observation dim : {env.observation_space.shape[0]}")
    print(f"Action dim      : {env.action_space.shape[0]}")
    print(f"Results dir     : {RESULTS_DIR}")

    model.learn(
        total_timesteps = TD3_TOTAL_TIMESTEPS,
        callback        = callback
    )

    # ── Save final model and metrics ──────────────────────────────────────────
    model.save(str(MODELS_DIR / "td3_final"))
    metrics.save_raw(str(RESULTS_DIR / "td3_training_data.json"))

    print(f"\nTraining complete.")
    print(f"  Best reward   : {callback.best_reward:.2f} → models/td3_best")
    print(f"  Final model   : models/td3_final")

    env.close()
    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Public validation entry point
# ─────────────────────────────────────────────────────────────────────────────

def validate_td3(
    rob: IRobobo,
    model: TD3,
    n_runs: int = 5,
    label: str = "TD3_Validation",
) -> TD3Metrics:
    """
    Run n_runs greedy episodes (no exploration noise) and record metrics.

    Parameters
    ----------
    rob     : IRobobo instance (sim or hardware)
    model   : trained TD3 model
    n_runs  : number of validation episodes
    label   : string label for saved files

    Returns
    -------
    metrics : TD3Metrics
    """
    _ensure_dirs()

    env     = RoboboEnv(rob)
    metrics = TD3Metrics(label=label)

    print(f"\nValidating TD3 — {n_runs} runs (deterministic policy)")

    for run in range(n_runs):
        obs = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            # deterministic=True disables action noise
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            speed     = float(info.get("s_trans", 0.0))
            collision = bool(info.get("collision", False))
            avoidance = float(info.get("avoidance", 0.0))
            metrics.record_step(reward, speed, collision, avoidance)

            if info.get("cells_visited", 0) > metrics.cells_this_episode:
                metrics.record_new_cell()

            ep_reward += reward

        metrics.end_episode(ep_reward)
        print(f"  Validation run {run + 1}/{n_runs} | reward: {ep_reward:.2f}")

    metrics.save_raw(str(RESULTS_DIR / f"td3_validation_{label}.json"))
    env.close()
    return metrics
