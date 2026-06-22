#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines.train_sac import train, validate, RESULTS_DIR

# SAC imports

from learning_machines.train_sac import train as train_sac
from learning_machines.train_sac import validate as validate_sac
from learning_machines.train_sac import RESULTS_DIR as RESULTS_DIR_SAC
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

    if mode == "--train-sim-sac":
        rob = SimulationRobobo()
        agent, train_metrics = train_sac(rob)

    elif mode == "--validate-sim-sac":
        label = sys.argv[2] if len(sys.argv) > 2 else "val1"

        rob = SimulationRobobo()

        agent = SAC_RL()
        agent.load(str(RESULTS_DIR_SAC / "sac_model_final.pt"))

        validate_sac(
            rob,
            agent,
            train_metrics=None,
            n_runs=5,
            label=label,
        )

    elif mode == "--validate-hw-sac":
        rob = HardwareRobobo(camera=True)

        class IntSpeedRobobo:
            def _init_(self, wrapped_robobo, max_speed=100):
                self._rob = wrapped_robobo
                self.max_speed = max_speed

            def move_blocking(self, left, right, duration):
                left = int(round(np.clip(left, -self.max_speed, self.max_speed)))
                right = int(round(np.clip(right, -self.max_speed, self.max_speed)))

                return self._rob.move_blocking(left, right, duration)

            def _getattr_(self, name):
                return getattr(self._rob, name)

        rob = IntSpeedRobobo(rob, max_speed=100)
        agent = SAC_RL()
        agent.load(str(RESULTS_DIR_SAC / "sac_agent_best.pt"))

        validate_sac(
            rob,
            agent,
            train_metrics=None,
            n_runs=10,
            label="hw_sac",
        )

    else:
        raise ValueError(f"Unknown argument: {mode}")