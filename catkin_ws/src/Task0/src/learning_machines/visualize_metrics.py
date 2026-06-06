import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
from .constants import *
 
 
class SimMetrics:
 
    def __init__(self, label = "Simulation"):
        self.label = label
        self.ir_history = [[] for _ in IR_indices.keys()]
        self.max_front = []
        self.max_back = []
        self.front_blocked = []
        self.obstacle_count = []
        self.sensor_load = []
        self.hits = 0
 
        # multi-run storage
        self.all_runs_front = []
        self.all_runs_back = []
        self.all_runs_ir = []
        self.all_runs_load = []
        self.run_summaries = []
 
    def record(self, irs):
        vals = [irs[i] for i in range(len(IR_indices.keys()))]
 
        for i, v in enumerate(vals):
            self.ir_history[i].append(v)
 
        max_front = max(vals[i] for i in FRONT_INDICES)
        max_back = max(vals[i] for i in BACK_INDICES)
        blocked = int(max_front > IR_THRESHOLD)
 
        self.hits += blocked
 
        self.max_front.append(max_front)
        self.max_back.append(max_back)
        self.front_blocked.append(blocked)
        self.obstacle_count.append(self.hits)
        self.sensor_load.append(sum(v > IR_THRESHOLD for v in vals) / len(vals))
 
    def end_run(self):
        """Call after each run to save data and reset buffers for next run."""
        if not self.max_front:
            return
 
        self.all_runs_front.append(list(self.max_front))
        self.all_runs_back.append(list(self.max_back))
        self.all_runs_ir.append(np.array(self.ir_history))
        self.all_runs_load.append(list(self.sensor_load))
        self.run_summaries.append(self.summary())
 
        # reset buffers
        self.ir_history = [[] for _ in IR_indices.keys()]
        self.max_front = []
        self.max_back = []
        self.front_blocked = []
        self.obstacle_count = []
        self.sensor_load = []
        self.hits = 0
 
    def _style(self, ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize = 9, fontweight = "bold", pad = 6)
        ax.set_xlabel(xlabel, fontsize = 8, labelpad = 4)
        ax.set_ylabel(ylabel, fontsize = 8, labelpad = 4)
        ax.tick_params(labelsize = 7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis = "y", linewidth = 0.4, alpha = 0.4, linestyle = "--")
 
    def _padded_matrix(self, list_of_lists):
        """Pad runs to equal length with NaN so they can be stacked."""
        max_len = max(len(r) for r in list_of_lists)
        out = np.full((len(list_of_lists), max_len), np.nan)
        for i, row in enumerate(list_of_lists):
            out[i, :len(row)] = row
        return out
 
    def plot(self, save_path = "../results/figures/"):
 
        if not self.max_front:
            print("Nothing to plot yet. Run the simulation first")
            return
 
        steps = np.arange(len(self.max_front))
 
        fig = plt.figure(figsize = (14, 12))
        gs = gridspec.GridSpec(3, 2, figure = fig, hspace = 0.48, wspace = 0.32)
 
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, :])
 
        # AX1: IR sensor values history
        for i, name in enumerate(IR_indices.keys()):
            ax1.plot(steps, self.ir_history[i],
                     label = name, linewidth = 0.9,
                     alpha = 0.85
            )
        ax1.axhline(IR_THRESHOLD, linewidth = 1.0,
                    linestyle = "--", alpha = 0.6,
                    color = "red", label = "threshold"
        )
        ax1.set_ylim(0, 200)
        ax1.legend(fontsize = 7, ncol = 2,
                    framealpha = 0.15, loc = "upper right"
        )
        self._style(ax1, "All IR Sensor Values", "Time Step", "IR Value (0–255)")
 
        # AX2: Front and Back max ranges
        ax2.fill_between(steps, self.max_front, alpha = 0.25)
        ax2.fill_between(steps, self.max_back, alpha = 0.25)
        ax2.plot(steps, self.max_front, linewidth = 1.3,
                 label = "Front max"
        )
        ax2.plot(steps, self.max_back, linewidth = 1.3,
                 label = "Back max"
        )
        ax2.axhline(IR_THRESHOLD, linewidth = 1,
                    linestyle = "--", alpha = 0.6,
                    color = "red", label = "threshold"
        )
        ax2.set_ylim(0, 200)
        ax2.legend(fontsize = 7, framealpha = 0.15,
                   loc = "upper right")
        self._style(ax2, "Front vs Back — Peak IR", "Time Step", "Max IR Value (0–255)")
 
        # AX3: Cumulative obstacle hits (recomputed as proper integer count)
        hits_cum = np.cumsum(np.array(self.max_front) > IR_THRESHOLD).astype(int)
        ax3.step(steps, hits_cum, where = "post",
                 linewidth = 1.3
        )
        ax3.fill_between(steps, hits_cum,
                         step = "post", alpha = 0.15
        )
        total = int(hits_cum[-1])
        ax3.annotate(f"Total: {total}",
                     xy = (steps[len(steps) // 2], total),
                     xytext = (0, 10), textcoords = "offset points",
                     fontsize = 7, fontweight = "bold"
        )
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        self._style(ax3, "Cumulative Obstacle Detections", "Time Step", "Detection Count")
 
        # AX4: Sensor load
        ax4.bar(steps, self.sensor_load,
                width = 1.0, alpha = 0.75
        )
        avg = np.mean(self.sensor_load)
        ax4.axhline(avg, linewidth = 1.3,
                    linestyle = "--", label = f"Mean {avg:.2f}"
        )
        ax4.set_ylim(0, 1)
        ax4.yaxis.set_major_formatter(ticker.PercentFormatter(xmax = 1))
        ax4.legend(fontsize = 7, framealpha = 0.15)
        self._style(ax4, "Fraction of Sensors Above Threshold per Step",
                    "Time Step", "Sensor Load (%)")
 
        # AX5: IR heatmap (all sensors x all steps)
        ir_mat = np.array(self.ir_history)
        im = ax5.imshow(ir_mat, aspect = "auto", cmap = "YlOrRd",
                        vmin = 0, vmax = 255, interpolation = "nearest")
        ax5.set_yticks(range(len(IR_indices)))
        ax5.set_yticklabels(list(IR_indices.keys()), fontsize = 7.5)
        cbar = fig.colorbar(im, ax = ax5, orientation = "vertical",
                            fraction = 0.02, pad = 0.01)
        cbar.set_label("IR Value", fontsize = 7)
        cbar.ax.tick_params(labelsize = 6.5)
        self._style(ax5, "IR Sensor Activation Heatmap",
                    "Time Step", "Sensor")
 
        # Title
        fig.suptitle(f"Robobo Simulation Sensor Metrics — {self.label}",
                     fontsize = 14, fontweight = "bold")
 
        plt.tight_layout(rect = [0, 0, 1, 1])
 
        if save_path:
            fig.savefig(save_path + "metrics.png", dpi = 300)
            print(f"Figure saved to: {save_path}metrics.png")
 
        plt.show()
 
    def plot_runs(self, save_path = "../results/figures/"):
        """Overlay all runs on shared axes to show run-to-run variance."""
 
        if not self.all_runs_front:
            print("No completed runs found. Call end_run() after each run.")
            return
 
        n_runs = len(self.all_runs_front)
        palette = plt.cm.tab10(np.linspace(0, 0.9, n_runs))
 
        mat_front = self._padded_matrix(self.all_runs_front)
        mat_back  = self._padded_matrix(self.all_runs_back)
        mat_load  = self._padded_matrix(self.all_runs_load)
 
        fig, axes = plt.subplots(2, 2, figsize = (14, 9))
 
        ax1, ax2, ax3, ax4 = axes.flat
 
        # AX1: Front IR all runs
        for r in range(n_runs):
            ax1.plot(mat_front[r], color = palette[r],
                     linewidth = 0.9, alpha = 0.6, label = f"Run {r + 1}"
            )
        mean_f = np.nanmean(mat_front, axis = 0)
        std_f  = np.nanstd(mat_front,  axis = 0)
        ax1.plot(mean_f, color = "black", linewidth = 1.8,
                 linestyle = "--", label = "Mean", zorder = 5
        )
        ax1.fill_between(np.arange(mat_front.shape[1]),
                         mean_f - std_f, mean_f + std_f,
                         alpha = 0.12, label = "±1 std"
        )
        ax1.axhline(IR_THRESHOLD, linewidth = 1.2, linestyle = "--",
                    color = "red", alpha = 0.6, label = "threshold"
        )
        ax1.set_ylim(0, 210)
        ax1.legend(fontsize = 6.5, ncol = 2, framealpha = 0.2)
        self._style(ax1, "Front Max IR — All Runs", "Time Step", "IR Value (0–255)")
 
        # AX2: Back IR all runs
        for r in range(n_runs):
            ax2.plot(mat_back[r], color = palette[r],
                     linewidth = 0.9, alpha = 0.6, label = f"Run {r + 1}"
            )
        mean_b = np.nanmean(mat_back, axis = 0)
        std_b  = np.nanstd(mat_back,  axis = 0)
        ax2.plot(mean_b, color = "black", linewidth = 1.8,
                 linestyle = "--", label = "Mean", zorder = 5
        )
        ax2.fill_between(np.arange(mat_back.shape[1]),
                         mean_b - std_b, mean_b + std_b,
                         alpha = 0.12, label = "±1 std"
        )
        ax2.axhline(IR_THRESHOLD, linewidth = 1.2, linestyle = "--",
                    color = "red", alpha = 0.6, label = "threshold"
        )
        ax2.set_ylim(0, 210)
        ax2.legend(fontsize = 6.5, ncol = 2, framealpha = 0.2)
        self._style(ax2, "Back Max IR — All Runs", "Time Step", "IR Value (0–255)")
 
        # AX3: Cumulative hits per run
        for r in range(n_runs):
            front_r  = np.array(self.all_runs_front[r])
            hits_r   = np.cumsum(front_r > IR_THRESHOLD).astype(int)
            ax3.step(np.arange(len(hits_r)), hits_r,
                     where = "post", color = palette[r],
                     linewidth = 0.9, alpha = 0.75, label = f"Run {r + 1}"
            )
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(integer = True))
        ax3.legend(fontsize = 6.5, ncol = 2, framealpha = 0.2)
        self._style(ax3, "Cumulative Obstacle Detections per Run",
                    "Time Step", "Detection Count")
 
        # AX4: Sensor load all runs
        for r in range(n_runs):
            ax4.plot(mat_load[r], color = palette[r],
                     linewidth = 0.8, alpha = 0.5, label = f"Run {r + 1}"
            )
        mean_l = np.nanmean(mat_load, axis = 0)
        std_l  = np.nanstd(mat_load,  axis = 0)
        ax4.plot(mean_l, color = "black", linewidth = 1.8,
                 linestyle = "--", label = "Mean", zorder = 5
        )
        ax4.fill_between(np.arange(mat_load.shape[1]),
                         mean_l - std_l, mean_l + std_l,
                         alpha = 0.12, label = "±1 std"
        )
        ax4.set_ylim(0, 1.05)
        ax4.yaxis.set_major_formatter(ticker.PercentFormatter(xmax = 1))
        ax4.legend(fontsize = 6.5, ncol = 2, framealpha = 0.2)
        self._style(ax4, "Sensor Load per Step — All Runs",
                    "Time Step", "Sensor Load (%)")
 
        fig.suptitle(f"Robobo Multi-Run Overview — {self.label}  ({n_runs} runs)",
                     fontsize = 13, fontweight = "bold")
 
        plt.tight_layout()
 
        if save_path:
            fig.savefig(save_path + f"multi_run_{self.label}.png", dpi = 300)
            print(f"Figure saved to: {save_path}multi_run_{self.label}.png")
 
        plt.show()
 
    @staticmethod
    def plot_reality_gap(sim, hw, save_path = "../results/figures/"):
        """
        Side-by-side comparison of simulation vs hardware sensor behaviour.
        Both SimMetrics objects should have end_run() called at least once.
        """
 
        def _get_matrices(m):
            if m.all_runs_front:
                return (m._padded_matrix(m.all_runs_front),
                        m._padded_matrix(m.all_runs_back),
                        m._padded_matrix(m.all_runs_load))
            return (np.array([m.max_front]),
                    np.array([m.max_back]),
                    np.array([m.sensor_load]))
 
        sim_front, sim_back, sim_load = _get_matrices(sim)
        hw_front,  hw_back,  hw_load  = _get_matrices(hw)
 
        fig, axes = plt.subplots(2, 3, figsize = (16, 9))
 
        configs = [
            (axes[0, 0], axes[1, 0], sim_front, hw_front, "Front Max IR",  "IR Value (0–255)"),
            (axes[0, 1], axes[1, 1], sim_back,  hw_back,  "Back Max IR",   "IR Value (0–255)"),
            (axes[0, 2], axes[1, 2], sim_load,  hw_load,  "Sensor Load",   "Fraction Active"),
        ]
 
        for ax_sim, ax_hw, s_mat, h_mat, title, ylabel in configs:
            for ax, mat, lbl in [
                (ax_sim, s_mat, sim.label),
                (ax_hw,  h_mat, hw.label),
            ]:
                x    = np.arange(mat.shape[1])
                mean = np.nanmean(mat, axis = 0)
                std  = np.nanstd(mat,  axis = 0)
 
                for r in range(mat.shape[0]):
                    ax.plot(x, mat[r], linewidth = 0.7, alpha = 0.35)
 
                ax.plot(x, mean, color = "black",
                        linewidth = 1.6, linestyle = "--", label = "Mean"
                )
                ax.fill_between(x, mean - std, mean + std,
                                alpha = 0.18, label = "±1 std"
                )
                ax.axhline(IR_THRESHOLD, linewidth = 1.1, linestyle = ":",
                           color = "red", alpha = 0.75, label = "threshold"
                )
                ax.legend(fontsize = 6.5, framealpha = 0.2)
                sim._style(ax, f"{title} — {lbl}", "Time Step", ylabel)
 
                if title != "Sensor Load":
                    ax.set_ylim(0, 210)
 
        fig.suptitle(f"Reality Gap — {sim.label} vs {hw.label}",
                     fontsize = 13, fontweight = "bold")
 
        plt.tight_layout()
 
        if save_path:
            fig.savefig(save_path + "reality_gap.png", dpi = 300)
            print(f"Figure saved to: {save_path}reality_gap.png")
 
        plt.show()
 
    def summary(self):
        if not self.max_front:
            return {}
 
        return {
            "steps" : len(self.max_front),
            "total_obstacle_hits" : self.hits,
            "hit_rate" : self.hits / len(self.max_front),
            "avg_front_IR" : float(np.mean(self.max_front)),
            "avg_back_IR" : float(np.mean(self.max_back)),
            "avg_sensor_load" : float(np.mean(self.sensor_load)),
            "max_front_IR_seen" : float(np.max(self.max_front))
        }