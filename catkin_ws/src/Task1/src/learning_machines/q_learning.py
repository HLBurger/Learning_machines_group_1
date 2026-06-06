import numpy as np
import random
import pickle
from pathlib import Path

from .constants import (
    IR_indices, IR_CLEAR, IR_NEAR,
    N_ACTIONS, ALPHA, GAMMA,
    EPSILON, EPSILON_MIN, EPSILON_DECAY,
)


RESULTS_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "results"

N_SENSORS = len(IR_indices)       # 8
N_LEVELS  = 3                     # clear / near / close
N_STATES  = N_LEVELS ** N_SENSORS # 3^8 = 6561


def discretise(irs: list) -> tuple:
    """
    Convert raw IR readings into a discrete state tuple.
    Each sensor maps to:
        0 -> clear  (value <= IR_CLEAR)
        1 -> near   (IR_CLEAR < value <= IR_NEAR)
        2 -> close  (value > IR_NEAR)
    """
    state = []
    for i in range(N_SENSORS):
        v = irs[i]
        if v <= IR_CLEAR:
            state.append(0)
        elif v <= IR_NEAR:
            state.append(1)
        else:
            state.append(2)
    return tuple(state)


def state_to_index(state: tuple) -> int:
    """Convert an 8-tuple of {0,1,2} into a flat Q-table row index."""
    idx = 0
    for s in state:
        idx = idx * N_LEVELS + s
    return idx


class QLearning:
    """
    Tabular Q-learning agent.
    Q-table shape: (N_STATES, N_ACTIONS) = (6561, 6)
    """

    def __init__(self):
        self.q_table = np.zeros((N_STATES, N_ACTIONS))
        self.epsilon = EPSILON

    def select_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            # 50% chance forward, 50% random turn
            if random.random() < 0.5:
                return random.randint(0, 2)  # forward actions
            return random.randint(3, 6)      # turn actions
        idx = state_to_index(state)
        return int(np.argmax(self.q_table[idx]))

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        """Standard Q-learning update rule."""
        idx      = state_to_index(state)
        next_idx = state_to_index(next_state)

        current_q = self.q_table[idx, action]
        target_q  = reward + (0.0 if done else GAMMA * np.max(self.q_table[next_idx]))

        self.q_table[idx, action] += ALPHA * (target_q - current_q)

    def decay_epsilon(self) -> None:
        """Call once per episode after training."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def save(self, path: str = None) -> None:
        if path is None:
            path = str(RESULTS_DIR / "q_table.pkl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)
        print(f"Q-table saved -> {path}")

    def load(self, path: str = None) -> None:
        if path is None:
            path = str(RESULTS_DIR / "q_table.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        print(f"Q-table loaded <- {path}")