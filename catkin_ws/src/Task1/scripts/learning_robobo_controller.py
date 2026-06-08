#!/usr/bin/env python3
import sys
from stable_baselines3 import TD3

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines.train_rl import train, validate, RESULTS_DIR
from learning_machines.q_learning import QLearning
from learning_machines.train_td3 import train_td3, validate_td3

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

    elif mode == "--train-td3-sim":
        rob = SimulationRobobo(identifier=1)
        model, train_metrics = train_td3(rob)

    elif mode == "--validate-td3-sim":
        label = sys.argv[2] if len(sys.argv) > 2 else "TD3_Validation"
        rob   = SimulationRobobo(identifier=1)
        model = TD3.load("../results/models/td3_best.zip", print_system_info = True)
        validate_td3(rob, model, n_runs=5, label=label)

    elif mode == "--validate-td3-hw":
        rob   = HardwareRobobo(camera=False)
        model = TD3.load(str(RESULTS_DIR / "models" / "td3_best"))
        validate_td3(rob, model, n_runs=5, label="TD3_HW_Validation")

    else:
        raise ValueError(f"Unknown argument: {mode}")