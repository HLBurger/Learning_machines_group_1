import os
import json
import numpy as np
import matplotlib.pyplot as plt
from .constants_sac import FRONT_INDICES, IR_THRESHOLD


class RLMetrics:
    """
    Tracks per-step and per-episode metrics for Task 2 SAC foraging.

    Per step  : reward, speed, collision, avoidance, food count, blob visibility
    Per episode: total reward, steps, food collected, avg steps per food,
                 blob visible fraction
    """

    def __init__(self, label="Training"):
        self.label = label

        # per-step buffers
        self._step_rewards    = []
        self._step_speeds     = []
        self._step_collisions = []
        self._step_avoidances = []
        self._step_food       = []
        self._step_obj_vis    = []
        self._step_obj_size   = []
        self.cells_this_episode = 0

        # per-episode storage
        self.episode_rewards    = []
        self.episode_steps      = []
        self.episode_cells      = []
        self.episode_speeds     = []
        self.episode_collisions = []
        self.episode_food       = []        # food collected per episode
        self.episode_steps_per_food = []   # avg steps to collect one food
        self.episode_obj_vis    = []        # fraction of steps food was visible
        self.epsilon_history    = []

    # ── Per-step recording ─────────────────────────────────────────────

    def record_step(
        self,
        action,
        irs: list,
        reward: float,
        epsilon: float,
        speed: float,
        collision: bool,
        avoidance: float,
        food_collected: int = 0,
        obj_visible: bool = False,
        obj_size: float = 0.0,
    ):
        self._step_rewards.append(reward)
        self._step_speeds.append(speed)
        self._step_collisions.append(1.0 if collision else 0.0)
        self._step_avoidances.append(avoidance)
        self._step_food.append(food_collected)
        self._step_obj_vis.append(1.0 if obj_visible else 0.0)
        self._step_obj_size.append(obj_size)
        self.epsilon_history.append(epsilon)

    def record_new_cell(self):
        self.cells_this_episode += 1

    # ── Per-episode recording ──────────────────────────────────────────

    def end_episode(self, total_reward: float, food_collected: int = 0):
        n_steps = len(self._step_rewards)
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(n_steps)
        self.episode_cells.append(self.cells_this_episode)
        self.episode_food.append(food_collected)
        self.episode_speeds.append(
            float(np.mean(self._step_speeds)) if self._step_speeds else 0.0
        )
        self.episode_collisions.append(
            float(np.mean(self._step_collisions)) if self._step_collisions else 0.0
        )
        self.episode_obj_vis.append(
            float(np.mean(self._step_obj_vis)) if self._step_obj_vis else 0.0
        )
        # avg steps per food — avoids division by zero
        if food_collected > 0:
            self.episode_steps_per_food.append(n_steps / food_collected)
        else:
            self.episode_steps_per_food.append(float(n_steps))  # no food = all steps wasted

        # reset
        self._step_rewards    = []
        self._step_speeds     = []
        self._step_collisions = []
        self._step_avoidances = []
        self._step_food       = []
        self._step_obj_vis    = []
        self._step_obj_size   = []
        self.cells_this_episode = 0

    # ── Save raw data ──────────────────────────────────────────────────

    def save_raw(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "label"                   : self.label,
            "episode_rewards"         : self.episode_rewards,
            "episode_steps"           : self.episode_steps,
            "episode_cells"           : self.episode_cells,
            "episode_food"            : self.episode_food,
            "episode_steps_per_food"  : self.episode_steps_per_food,
            "episode_speeds"          : self.episode_speeds,
            "episode_collisions"      : self.episode_collisions,
            "episode_obj_vis"         : self.episode_obj_vis,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Raw data saved -> {path}")

    # ── Helpers ────────────────────────────────────────────────────────

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
        lo = np.percentile(data, 10)
        hi = np.percentile(data, 100)
        margin = (hi - lo) * 0.15
        ax.set_ylim(lo - margin, hi + margin)

    @staticmethod
    def _plot_with_band(ax, data, window, color, label):
        episodes = np.arange(len(data))
        roll     = RLMetrics._rolling(data, window)
        roll_x   = np.arange(window - 1, len(data))
        roll_std = np.array([
            np.std(data[max(0, i - window):i + 1])
            for i in range(window - 1, len(data))
        ])
        ax.plot(episodes, data, alpha=0.35, linewidth=0.9, color=color, label="Per episode")
        ax.plot(roll_x, roll, linewidth=2.0, color="black", label=f"Rolling mean ({window})")
        ax.fill_between(roll_x, roll - roll_std, roll + roll_std, alpha=0.2, color="tab:red", label="±1 std")

    # ── Plot 1: Training curves ────────────────────────────────────────

    def plot_training(self, save_path: str = "results/figures/") -> None:
        """
        3-panel: Total Reward | Mean Speed | Collision Rate
        Matches lecture 'Analysing metrics' layout.
        """
        if not self.episode_rewards:
            print("No episodes recorded yet.")
            return

        window = min(10, len(self.episode_rewards))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"Training Metrics — {self.label} ({len(self.episode_rewards)} episodes)",
            fontsize=13, fontweight="bold"
        )

        configs = [
            (axes[0], np.array(self.episode_rewards),    "Total Reward",   "Total Reward",       False),
            (axes[1], np.array(self.episode_speeds),     "Mean Speed",     "Speed (normalised)", True),
            (axes[2], np.array(self.episode_collisions), "Collision Rate", "Collision Rate",     True),
        ]

        for ax, data, title, ylabel, fixed_ylim in configs:
            RLMetrics._plot_with_band(ax, data, window, "tab:blue", "Per episode")
            ax.legend(fontsize=7, framealpha=0.2)
            RLMetrics._style(ax, title, "Episode", ylabel)
            if fixed_ylim:
                ax.set_ylim(-0.05, 1.05)
            else:
                RLMetrics._smart_ylim(ax, data)

        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            path = os.path.join(save_path, f"components_{self.label}.png")
            fig.savefig(path, dpi=300)
            print(f"Figure saved -> {path}")
        plt.show()

    # ── Plot 2: Food metrics ───────────────────────────────────────────

    def plot_food_metrics(self, save_path: str = "results/figures/") -> None:
        """
        Task 2 specific plots matching the paper (Fig 3):
          - Food collected per episode (bar chart)
          - Avg steps per food per episode (line chart)
          - Food visible fraction per episode
        """
        if not self.episode_food:
            print("No food data recorded.")
            return

        episodes = np.arange(len(self.episode_food))
        window   = min(10, len(episodes))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"Food Collection Metrics — {self.label}",
            fontsize=13, fontweight="bold"
        )

        # AX1: food collected per episode (bar)
        ax1 = axes[0]
        food = np.array(self.episode_food)
        ax1.bar(episodes, food, color="tab:green", alpha=0.7, label="Food collected")
        if len(food) >= window:
            roll = RLMetrics._rolling(food.astype(float), window)
            ax1.plot(np.arange(window - 1, len(food)), roll,
                     color="black", linewidth=2.0, label=f"Rolling mean ({window})")
        ax1.legend(fontsize=7, framealpha=0.2)
        RLMetrics._style(ax1, "Food Collected per Episode", "Episode", "Packages")

        # AX2: avg steps per food (line — lower = faster)
        ax2 = axes[1]
        spf = np.array(self.episode_steps_per_food)
        RLMetrics._plot_with_band(ax2, spf, window, "tab:orange", "Steps/food")
        ax2.legend(fontsize=7, framealpha=0.2)
        RLMetrics._style(ax2, "Avg Steps per Food (lower = faster)", "Episode", "Steps / package")
        RLMetrics._smart_ylim(ax2, spf)

        # AX3: fraction of steps food was visible
        ax3 = axes[2]
        vis = np.array(self.episode_obj_vis)
        RLMetrics._plot_with_band(ax3, vis, window, "tab:blue", "Visible fraction")
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(fontsize=7, framealpha=0.2)
        RLMetrics._style(ax3, "Fraction of Steps Food Visible", "Episode", "Fraction")

        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            path = os.path.join(save_path, f"food_metrics_{self.label}.png")
            fig.savefig(path, dpi=300)
            print(f"Figure saved -> {path}")
        plt.show()

    # ── Plot 3: Training vs Validation ────────────────────────────────

    @staticmethod
    def plot_training_vs_validation(train_metrics, val_metrics, save_path="results/figures/"):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Training vs Validation Performance", fontsize=13, fontweight="bold")

        pairs = [
            (np.array(train_metrics.episode_rewards),    np.array(val_metrics.episode_rewards),    "Total Reward",   False),
            (np.array(train_metrics.episode_food),       np.array(val_metrics.episode_food),       "Food Collected", False),
            (np.array(train_metrics.episode_collisions), np.array(val_metrics.episode_collisions), "Collision Rate", True),
        ]

        for ax, (train_data, val_data, title, fixed_ylim) in zip(axes, pairs):
            window = min(10, len(train_data))
            t_x    = np.arange(len(train_data))
            ax.plot(t_x, train_data, color="tab:blue", linewidth=1.0, alpha=0.4, label="Training")

            roll     = RLMetrics._rolling(train_data.astype(float), window)
            roll_x   = np.arange(window - 1, len(train_data))
            roll_std = np.array([np.std(train_data[max(0, i-window):i+1]) for i in range(window-1, len(train_data))])
            ax.plot(roll_x, roll, color="tab:blue", linewidth=2.0, label="Training mean")
            ax.fill_between(roll_x, roll - roll_std, roll + roll_std, alpha=0.2, color="tab:blue")

            v_mean = np.mean(val_data)
            v_std  = np.std(val_data)
            ax.axhline(v_mean, color="tab:green", linewidth=2.0, linestyle="--",
                       label=f"Validation mean ({v_mean:.2f})")
            ax.axhspan(v_mean - v_std, v_mean + v_std, alpha=0.15, color="tab:green")

            ax.legend(fontsize=7, framealpha=0.2)
            RLMetrics._style(ax, title, "Episode", title)
            if fixed_ylim:
                ax.set_ylim(-0.05, 1.05)
            else:
                RLMetrics._smart_ylim(ax, train_data)

        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, "train_vs_validation.png"), dpi=300)
            print(f"Figure saved -> {os.path.join(save_path, 'train_vs_validation.png')}")
        plt.show()

    # ── Plot 4: Boxplot ────────────────────────────────────────────────

    @staticmethod
    def plot_boxplot(train_metrics, val_metrics, save_path="results/figures/"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))
        fig.suptitle("Training vs Validation — Final Policy", fontsize=13, fontweight="bold")

        n = min(20, len(train_metrics.episode_rewards))
        pairs = [
            (train_metrics.episode_rewards[-n:],    val_metrics.episode_rewards,    "Total Reward"),
            (train_metrics.episode_food[-n:],        val_metrics.episode_food,        "Food Collected"),
            (train_metrics.episode_collisions[-n:],  val_metrics.episode_collisions,  "Collision Rate"),
        ]

        for ax, (train_data, val_data, title) in zip(axes, pairs):
            bp = ax.boxplot(
                [train_data, val_data],
                labels=["Training\n(last 20)", "Validation\n(5 runs)"],
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                widths=0.5,
            )
            bp["boxes"][0].set_facecolor("#2d6a4f"); bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_facecolor("#e63946"); bp["boxes"][1].set_alpha(0.7)
            for i, d in enumerate([train_data, val_data], 1):
                ax.plot(i, np.mean(d), "o", color="yellow", markersize=7, zorder=5)
            RLMetrics._style(ax, title, "", title)

        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, "boxplot_train_vs_val.png"), dpi=300)
        plt.show()