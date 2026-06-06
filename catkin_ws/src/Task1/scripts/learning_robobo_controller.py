#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines.train_rl import train, validate, RESULTS_DIR
from learning_machines.q_learning import QLearning


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
        rob = SimulationRobobo(identifier=1)
        agent, train_metrics = train(rob)

    elif mode == "--validate-sim":
        label = sys.argv[2] if len(sys.argv) > 2 else "val1"
        rob   = SimulationRobobo(identifier=1)
        agent = QLearning()
        agent.load(agent.load(str(RESULTS_DIR / "q_table_final.pkl")))
        validate(rob, agent, train_metrics=None, n_runs=5, label=label)


    elif mode == "--validate-hw":
        rob   = HardwareRobobo(camera=False)
        agent = QLearning()
        agent.load(str(RESULTS_DIR / "q_table_best.pkl"))
        validate(rob, agent, train_metrics=None, n_runs=5)

    else:
        raise ValueError(f"Unknown argument: {mode}")