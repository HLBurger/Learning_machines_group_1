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
from .visualize_metrics_td3 import TD3Metrics, TD3MetricsCallback

RESULTS_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR  = RESULTS_DIR / "models"


def _ensure_dirs():
    for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)




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
    metrics.plot_training(save_path=str(FIGURES_DIR) + "/")
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
            collision = bool(info.get("front_collision", False))
            avoidance = float(info.get("avoidance", 0.0))
            metrics.record_step(reward, speed, collision, avoidance)

            if info.get("cells_visited", 0) > metrics.cells_this_episode:
                metrics.record_new_cell()

            ep_reward += reward

        metrics.end_episode(ep_reward)
        print(f"  Validation run {run + 1}/{n_runs} | reward: {ep_reward:.2f}")

    metrics.save_raw(str(RESULTS_DIR / f"td3_validation_{label}.json"))
    env.close()

    train_metrics = TD3Metrics.load_raw(str(RESULTS_DIR / "td3_training_data.json"))
    val_metrics   = TD3Metrics.load_raw(str(RESULTS_DIR / f"td3_validation_{label}.json"))
    

    train_metrics.plot_training(save_path=FIGURES_DIR)
    TD3Metrics.plot_training_vs_validation(train_metrics, val_metrics, save_path=FIGURES_DIR)
    TD3Metrics.plot_boxplot(train_metrics, val_metrics, save_path=FIGURES_DIR)

    return metrics
