#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines.train_rl import train, validate
from learning_machines.q_learning import QLearning
from learning_machines.visualize_metrics import RLMetrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            """Pass one of:
            --train-sim         train in simulation
            --validate-sim      validate in simulation (loads saved Q-table)
            --train-validate    train then immediately validate
            --validate-hw       validate on hardware (loads saved Q-table)
            """
        )

    mode = sys.argv[1]

    if mode == "--train-sim":
        rob = SimulationRobobo(identifier=2)
        agent, train_metrics = train(rob)

    elif mode == "--validate-sim":
        rob   = SimulationRobobo(identifier=2)
        agent = QLearning()
        agent.load()
        validate(rob, agent, train_metrics=None, n_runs=5)

    elif mode == "--train-validate":
        # train then immediately validate with comparison plots
        rob = SimulationRobobo(identifier=2)
        agent, train_metrics = train(rob)
        validate(rob, agent, train_metrics=train_metrics, n_runs=5)

    elif mode == "--validate-hw":
        rob   = HardwareRobobo(camera=False)
        agent = QLearning()
        agent.load()
        validate(rob, agent, train_metrics=None, n_runs=5)

    else:
        raise ValueError(f"Unknown argument: {mode}")