#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines.train_rl import train, validate, RESULTS_DIR
from learning_machines.q_learning import QLearning

# SAC imports

from learning_machines.train_SAC import train as train_sac
from learning_machines.train_SAC import validate as validate_sac
from learning_machines.train_SAC import RESULTS_DIR as RESULTS_DIR_SAC
from learning_machines.sac import SAC_RL


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

    elif mode == "--train-sim-sac":
        rob = SimulationRobobo(identifier=1)
        agent, train_metrics = train_sac(rob)

    elif mode == "--validate-sim":
        label = sys.argv[2] if len(sys.argv) > 2 else "val1"

        rob = SimulationRobobo(identifier=1)

        agent = QLearning()
        agent.load(str(RESULTS_DIR / "q_table_final.pkl"))

        validate(
            rob,
            agent,
            train_metrics=None,
            n_runs=5,
            label=label,
        )

    elif mode == "--validate-sim-sac":
        label = sys.argv[2] if len(sys.argv) > 2 else "val1"

        rob = SimulationRobobo(identifier=1)

        agent = SAC_RL()
        agent.load(str(RESULTS_DIR_SAC / "sac_agent_best.pt"))

        validate_sac(
            rob,
            agent,
            train_metrics=None,
            n_runs=5,
            label=label,
        )

    elif mode == "--validate-hw":
        rob = HardwareRobobo(camera=False)

        agent = QLearning()
        agent.load(str(RESULTS_DIR / "q_table_best.pkl"))

        validate(
            rob,
            agent,
            train_metrics=None,
            n_runs=5,
        )

    elif mode == "--validate-hw-sac":
        rob = HardwareRobobo(camera=False)

        agent = SAC_RL()
        agent.load(str(RESULTS_DIR_SAC / "sac_agent_best.pt"))

        validate_sac(
            rob,
            agent,
            train_metrics=None,
            n_runs=5,
            label="hw_sac",
        )

    else:
        raise ValueError(f"Unknown argument: {mode}")