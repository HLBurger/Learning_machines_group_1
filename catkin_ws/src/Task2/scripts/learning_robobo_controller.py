#!/usr/bin/env python3
import sys
import numpy as np

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines.train_SAC import train as train_sac
from learning_machines.train_SAC import validate as validate_sac
from learning_machines.train_SAC import RESULTS_DIR as RESULTS_DIR_SAC
from learning_machines.sac import SAC_RL

# Arena 2 — foraging scene with green food packages
ARENA_IDENTIFIER = 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            """Pass one of:
            --train-sim-sac          train SAC in simulation
            --validate-sim-sac       validate in simulation (loads sac_agent_final.pt)
            --validate-sim-sac val2  validate with val2 label
            --validate-hw-sac        validate on hardware   (loads sac_agent_best.pt)
            """
        )

    mode = sys.argv[1]

    if mode == "--train-sim-sac":
        rob = SimulationRobobo(identifier=ARENA_IDENTIFIER)
        train_sac(rob)

    elif mode == "--validate-sim-sac":
        label = sys.argv[2] if len(sys.argv) > 2 else "val1"
        rob   = SimulationRobobo(identifier=ARENA_IDENTIFIER)

        agentred = SAC_RL()
        model_path = RESULTS_DIR_SAC / "sac_agentred_best.pt"
        if not model_path.exists():
            model_path = RESULTS_DIR_SAC / "sac_agentred_final.pt"
        agentred.load(str(model_path))

        agentgreen = SAC_RL()
        model_path = RESULTS_DIR_SAC / "sac_agentgreen_best.pt"
        if not model_path.exists():
            model_path = RESULTS_DIR_SAC / "sac_agentgreen_final.pt"
        agentgreen.load(str(model_path))

        validate_sac(rob, agentred, agentgreen, train_metrics=None, n_runs=5, label=label)

    elif mode == "--validate-hw-sac":

        class IntSpeedRobobo:
            """Rounds float motor speeds to int for hardware compatibility."""
            def __init__(self, wrapped, max_speed=100):
                self._rob      = wrapped
                self.max_speed = max_speed

            def move_blocking(self, left, right, duration):
                left  = int(round(np.clip(left,  -self.max_speed, self.max_speed)))
                right = int(round(np.clip(right, -self.max_speed, self.max_speed)))
                return self._rob.move_blocking(left, right, duration)

            def __getattr__(self, name):
                return getattr(self._rob, name)

        rob   = HardwareRobobo(camera=True)   # camera=True required for Task 2
        rob   = IntSpeedRobobo(rob)

        agentred = SAC_RL()
        model_path = RESULTS_DIR_SAC / "sac_agentred_best.pt"
        if not model_path.exists():
            model_path = RESULTS_DIR_SAC / "sac_agentred_final.pt"
        agentred.load(str(model_path))

        agentgreen = SAC_RL()
        model_path = RESULTS_DIR_SAC / "sac_agentgreen_best.pt"
        if not model_path.exists():
            model_path = RESULTS_DIR_SAC / "sac_agentgreen_final.pt"
        agentgreen.load(str(model_path))

        validate_sac(rob, agentred, agentgreen, train_metrics=None, n_runs=5, label="hw_sac")

    else:
        raise ValueError(f"Unknown argument: {mode}")