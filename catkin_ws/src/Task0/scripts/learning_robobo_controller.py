#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions, avoid_obstacles
from learning_machines.visualize_metrics import SimMetrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera = True)
        hw_metrics = avoid_obstacles(rob, n_runs = 5)

    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        sim_metrics = avoid_obstacles(rob, n_runs = 5)

    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")
