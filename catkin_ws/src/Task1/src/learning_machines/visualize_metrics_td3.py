import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


# ─────────────────────────────────────────────────────────────────────────────
# Metrics container
# ─────────────────────────────────────────────────────────────────────────────

class TD3Metrics:
    """
    Lightweight metrics tracker for TD3 training.
    Episode data is appended by TD3MetricsCallback during training.
    Includes plotting methods compatible with the RLMetrics visualisation API.
    """

    def __init__(self, label: str = "TD3_Training"):
        self.label = label

        self.episode_rewards    = []
        self.episode_steps      = []
        self.episode_cells      = []
        self.episode_speeds     = []
        self.episode_collisions = []
        self.episode_avoidances = []

        # filled per-step by the callback
        self._step_rewards     = []
        self._step_speeds      = []
        self._step_collisions  = []
        self._step_avoidances  = []
        self.cells_this_episode = 0

        # not used by TD3 (no ε-greedy), kept for API compatibility
        self.epsilon_history   = []

    # ── Per-step recording ────────────────────────────────────────────────────

    def record_step(self, reward: float, speed: float,
                    collision: bool, avoidance: float) -> None:
        self._step_rewards.append(reward)
        self._step_speeds.append(speed)
        self._step_collisions.append(float(collision))
        self._step_avoidances.append(avoidance)

    def record_new_cell(self) -> None:
        self.cells_this_episode += 1

    # ── Per-episode recording ─────────────────────────────────────────────────

    def end_episode(self, total_reward: float) -> None:
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

    # ── Persist raw data ──────────────────────────────────────────────────────

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

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _rolling(arr, window):
        return np.convolve(arr, np.ones(window) / window, mode="valid")

    @staticmethod
    def _style(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel, fontsize=8, labelpad=4)
        ax.set_ylabel(ylabel, fontsize=8, labelpad=4)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4, linestyle="--")

    @staticmethod
    def _smart_ylim(ax, data):
        """Set y-axis using percentiles to ignore early outliers."""
        lo = np.percentile(data, 10)
        hi = np.percentile(data, 100)
        margin = (hi - lo) * 0.15
        ax.set_ylim(lo - margin, hi + margin)
    
    @classmethod
    def load_raw(cls, path: str) -> "TD3Metrics":
        with open(path, "r") as f:
            data = json.load(f)
        obj = cls(label=data.get("label", ""))
        obj.episode_rewards    = data["episode_rewards"]
        obj.episode_steps      = data["episode_steps"]
        obj.episode_cells      = data["episode_cells"]
        obj.episode_speeds     = data["episode_speeds"]
        obj.episode_collisions = data["episode_collisions"]
        obj.episode_avoidances = data["episode_avoidances"]
        return obj

    # ── Plot 1: Training component curves ─────────────────────────────────────

    def plot_training(self, save_path: str = "results/figures/") -> None:
        """
        3-panel plot: Total Reward | Mean Speed | Collision Rate.
        Each panel shows the per-episode line, rolling mean, and ±1 std band.
        """
        if not self.episode_rewards:
            print("No episodes recorded yet.")
            return

        episodes = np.arange(len(self.episode_rewards))
        window   = min(10, len(episodes))

        rewards    = np.array(self.episode_rewards)
        speeds     = np.array(self.episode_speeds)
        collisions = np.array(self.episode_collisions)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"Training Metrics — {self.label} ({len(episodes)} episodes)",
            fontsize=13, fontweight="bold"
        )

        configs = [
            (axes[0], rewards,    "Total Reward",   "Total Reward",       False),
            (axes[1], speeds,     "Mean Speed",     "Speed (normalised)", True),
            (axes[2], collisions, "Collision Rate", "Collision Rate",     True),
        ]

        for ax, data, title, ylabel, fixed_ylim in configs:
            roll   = TD3Metrics._rolling(data, window)
            roll_x = np.arange(window - 1, len(data))
            roll_std = np.array([
                np.std(data[max(0, i - window):i + 1])
                for i in range(window - 1, len(data))
            ])

            ax.plot(episodes, data, alpha=0.35, linewidth=0.9,
                    color="tab:blue", label="Per episode")
            ax.plot(roll_x, roll, linewidth=2.0, color="black",
                    label=f"Rolling mean ({window})")
            ax.fill_between(roll_x, roll - roll_std, roll + roll_std,
                            alpha=0.2, color="tab:red", label="±1 std")
            ax.legend(fontsize=7, framealpha=0.2)
            TD3Metrics._style(ax, title, "Episode", ylabel)

            if fixed_ylim:
                ax.set_ylim(-0.05, 1.05)
            else:
                TD3Metrics._smart_ylim(ax, data)

        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            path = os.path.join(save_path, f"components_{self.label}.png")
            fig.savefig(path, dpi=300)
            print(f"Figure saved -> {path}")
        plt.show()

    # ── Plot 2: Training vs Validation curves ─────────────────────────────────

    @staticmethod
    def plot_training_vs_validation(
        train_metrics: "TD3Metrics",
        val_metrics:   "TD3Metrics",
        save_path: str = "results/figures/",
    ) -> None:
        """
        Training curve (blue) overlaid with a validation mean/std band (green).
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            "Training vs Validation Performance",
            fontsize=13, fontweight="bold"
        )

        pairs = [
            (
                np.array(train_metrics.episode_rewards),
                np.array(val_metrics.episode_rewards),
                "Total Reward", "Total Reward", False,
            ),
            (
                np.array(train_metrics.episode_speeds),
                np.array(val_metrics.episode_speeds),
                "Mean Speed", "Speed (normalised)", True,
            ),
            (
                np.array(train_metrics.episode_collisions),
                np.array(val_metrics.episode_collisions),
                "Collision Rate", "Collision Rate", True,
            ),
        ]

        for ax, (train_data, val_data, title, ylabel, fixed_ylim) in zip(axes, pairs):
            t_x    = np.arange(len(train_data))
            window = min(10, len(train_data))

            ax.plot(t_x, train_data, color="tab:blue",
                    linewidth=1.0, alpha=0.4, label="Training (per episode)")

            roll     = TD3Metrics._rolling(train_data, window)
            roll_x   = np.arange(window - 1, len(train_data))
            roll_std = np.array([
                np.std(train_data[max(0, i - window):i + 1])
                for i in range(window - 1, len(train_data))
            ])
            ax.plot(roll_x, roll, color="tab:blue",
                    linewidth=2.0, label="Training rolling mean")
            ax.fill_between(roll_x, roll - roll_std, roll + roll_std,
                            alpha=0.2, color="tab:blue")

            v_mean = np.mean(val_data)
            v_std  = np.std(val_data)
            ax.axhline(v_mean, color="tab:green", linewidth=2.0,
                       linestyle="--", label=f"Validation mean ({v_mean:.2f})")
            ax.axhspan(v_mean - v_std, v_mean + v_std,
                       alpha=0.15, color="tab:green", label="Validation ±1 std")

            ax.legend(fontsize=7, framealpha=0.2)
            TD3Metrics._style(ax, title, "Episode", ylabel)

            if fixed_ylim:
                ax.set_ylim(-0.05, 1.05)
            else:
                TD3Metrics._smart_ylim(ax, train_data)

        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            path = os.path.join(save_path, "train_vs_validation.png")
            fig.savefig(path, dpi=300)
            print(f"Figure saved -> {path}")
        plt.show()

    # ── Plot 3: Box plot training vs validation ───────────────────────────────

    @staticmethod
    def plot_boxplot(
        train_metrics: "TD3Metrics",
        val_metrics:   "TD3Metrics",
        save_path: str = "results/figures/",
    ) -> None:
        """
        Box plot comparing the converged training policy against validation runs.
        Uses the last 20 training episodes as a proxy for the converged policy.
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))
        fig.suptitle(
            "Training vs Validation — Final Policy Comparison",
            fontsize=13, fontweight="bold"
        )

        n_converged = min(20, len(train_metrics.episode_rewards))

        pairs = [
            (
                train_metrics.episode_rewards[-n_converged:],
                val_metrics.episode_rewards,
                "Total Reward",
            ),
            (
                train_metrics.episode_speeds[-n_converged:],
                val_metrics.episode_speeds,
                "Mean Speed",
            ),
            (
                train_metrics.episode_collisions[-n_converged:],
                val_metrics.episode_collisions,
                "Collision Rate",
            ),
        ]

        for ax, (train_data, val_data, title) in zip(axes, pairs):
            bp = ax.boxplot(
                [train_data, val_data],
                labels=["Training\n(last 20 ep)", "Validation\n(5 runs)"],
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker="o", markersize=4, alpha=0.5),
                widths=0.5,
            )
            bp["boxes"][0].set_facecolor("#2d6a4f")
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_facecolor("#e63946")
            bp["boxes"][1].set_alpha(0.7)

            for i, data in enumerate([train_data, val_data], 1):
                ax.plot(i, np.mean(data), "o", color="yellow",
                        markersize=7, zorder=5,
                        label="Mean" if i == 1 else "")

            ax.legend(fontsize=7, framealpha=0.2)
            TD3Metrics._style(ax, title, "", title)

        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            path = os.path.join(save_path, "boxplot_train_vs_val.png")
            fig.savefig(path, dpi=300)
            print(f"Figure saved -> {path}")
        plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# SB3 callback
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
        self.metrics     = metrics
        self.save_dir    = save_dir
        self.best_reward = float("-inf")
        self._ep_reward  = 0.0
        self._ep_step    = 0

    def _on_step(self) -> bool:
        info   = self.locals["infos"][0]
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]

        self._ep_reward += reward
        self._ep_step   += 1

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

            # save checkpoint when a new best is reached (skip first 5 warmup eps)
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

        return True  # returning False would stop training