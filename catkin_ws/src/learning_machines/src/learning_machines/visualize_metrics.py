import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from .constants import *


class SimMetrics:

    def __init__(self):
        self.ir_history = [[] for _ in IR_indices.keys()]
        self.max_front = []
        self.max_back = []
        self.front_blocked = []
        self.obstacle_count = []
        self.sensor_load = []
        self.hits = 0

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

    def plot(self, save_path = "../results/figures/"):

        if not self.max_front:
            print("Nothing to plot yet. Run the simulation first")
            return

        steps = np.arange(len(self.max_front))

        fig = plt.figure(figsize = (14, 10))
        gs = gridspec.GridSpec(2, 2, figure = fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # AX1: IR sensor values history
        for i, name in enumerate(IR_indices.keys()):
            ax1.plot(steps, self.ir_history[i], 
                     label = name, linewidth = 0.9,
                     alpha = 0.85
            )
        ax1.axhline(IR_THRESHOLD, linewidth = 1.0,
                    linestyle = "--", alpha = 0.6,
                    label = "threshold"
        )
        ax1.set_ylim(0, 200)
        ax1.legend(fontsize = 7, ncol = 2,
                    framealpha = 0.15, loc = "upper right"
        )
        #plt.style(ax1, "All IR sensor values", "Step", "IR Value")

        #AX2: Front and Back max ranges
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
                    label = "threshold"
        )
        ax2.set_ylim(0, 200)
        ax2.legend(fontsize = 7, framealpha = 0.15,
                   loc = "upper right")
        #_style(ax2, "Front vs Back — Peak IR", "Step", "Max IR Value")

        #AX3: Obstacle hits
        ax3.step(steps, self.obstacle_count, where = "post",
                 linewidth = 1.3         
        )
        ax3.fill_between(steps,self.obstacle_count, 
                         step = "post", alpha = 0.15
        )
        total = self.obstacle_count[-1]
        ax3.annotate(f"Total: {total}",
                     xy = (steps[-1], total),
                     xytext = (-60, -16), textcoords = "offset points",
                     fontsize = 7, fontweight = "bold"            
        )
        #_style(ax3, "Cumulative Obstacle Detections", "Step", "Hit Count")

        # AX4: Sensor loads
        ax4.bar(steps, self.sensor_load,
                width = 1.0, alpha = 0.75        
        )
        avg = np.mean(self.sensor_load)
        ax4.axhline(avg, linewidth = 1.3,
                    linestyle = "--", label = f"Mean {avg:.2f}"            
        )
        ax4.set_ylim(0, 1)
        ax4.legend(fontsize = 7, framealpha = 0.15)
        #_style(ax4, "Sensor Load per Step", "Step", "Fraction of sensors active")

        # Title
        fig.suptitle("Robobo Simulation Sensor Metrics",
                     fontsize = 14, fontweight = "bold")
        
        plt.tight_layout(rect = [0, 0, 1, 1])

        if save_path:
            fig.savefig(save_path + "metrics.png", dpi = 300)
            print(f"Figure saved to: {save_path}metrics.png")

        plt.show()


    def summary(self):
        if not self.max_front:
            return {}
        
        return {
            "steps" : len(self.max_front),
            "total_obstacle hits" : self.hits,
            "hit_rate" : self.hits / len(self.max_front),
            "avg_front_IR" : float(np.mean(self.max_front)),
            "avg_back_IR" : float(np.mean(self.max_back)),
            "avg_sensor_load" : float(np.mean(self.sensor_load)),
            "max_front_IR_seen" : float(np.max(self.max_front))
        }