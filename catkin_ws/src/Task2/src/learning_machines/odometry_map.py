"""
OdometryMap
===========
Maintains a global boolean occupancy grid updated by wheel odometry (or
ground-truth simulation position) and provides a local-window observation
centred on the agent.

Grid convention
---------------
- Origin (0, 0) is placed at the centre of the grid on reset.
- x increases to the right, y increases upward (standard robot frame).
- Each cell represents a square of side GRID_SIZE metres.

Coordinate helpers
------------------
  world  -> grid index : col = int(x / GRID_SIZE) + half_w
                         row = int(y / GRID_SIZE) + half_h
  grid index -> world  : x = (col - half_w) * GRID_SIZE
                         y = (row - half_h) * GRID_SIZE
"""

import math
import numpy as np

from .constants_sac import (
    GRID_SIZE,
    MAP_WIDTH_CELLS,
    MAP_HEIGHT_CELLS,
    LOCAL_MAP_WINDOW,
    WHEEL_BASE,
    MAX_WHEEL_SPEED,
    ACTION_DURATION_MS,
)


class OdometryMap:
    """
    Occupancy map updated by dead-reckoning odometry.

    Parameters
    ----------
    use_sim_position : bool
        If True, ``update()`` expects a real (x, y) position from the
        simulator rather than integrating wheel speeds.  The orientation
        is still integrated from wheel commands so that the local window
        is always robot-centric.
    """

    def __init__(self, use_sim_position: bool = False):
        self.use_sim_position = use_sim_position
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Clear the map and reset pose to the grid centre."""
        self.grid = np.zeros(
            (MAP_HEIGHT_CELLS, MAP_WIDTH_CELLS), dtype=np.bool_
        )
        self._half_w = MAP_WIDTH_CELLS  // 2
        self._half_h = MAP_HEIGHT_CELLS // 2

        # Robot pose in world coordinates (metres / radians)
        self.x: float = 0.0
        self.y: float = 0.0
        self.theta: float = 0.0  # heading, radians, 0 = +x axis

        # Mark starting cell
        self._mark_current_cell()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        left_speed: float,
        right_speed: float,
        sim_position=None,
    ) -> bool:
        """
        Advance the odometry one step and mark the new cell.

        Parameters
        ----------
        left_speed, right_speed : float
            Wheel speeds in the same units as MAX_WHEEL_SPEED
            (i.e. ±25 cm/s for the Robobo).
        sim_position : object with .x, .y attributes, or None
            When ``use_sim_position=True`` and this is provided, the
            (x, y) position is taken directly from the simulator; only
            heading is integrated from wheels.

        Returns
        -------
        bool
            True if the agent entered a previously-unseen cell.
        """
        dt = ACTION_DURATION_MS / 1000.0  # seconds

        # Convert wheel speed to m/s (Robobo units are cm/s)
        v_l = left_speed  / 100.0
        v_r = right_speed / 100.0

        # Heading integration (always from wheels)
        omega = (v_r - v_l) / WHEEL_BASE
        self.theta += omega * dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

        if self.use_sim_position and sim_position is not None:
            # Ground-truth XY from simulator
            self.x = float(sim_position.x)
            self.y = float(sim_position.y)
        else:
            # Dead-reckoning
            v = (v_l + v_r) / 2.0
            self.x += v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt

        return self._mark_current_cell()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def local_window(self) -> np.ndarray:
        """
        Return a flattened float32 array of shape (LOCAL_MAP_WINDOW**2,)
        representing the boolean occupancy in a square patch of side
        LOCAL_MAP_WINDOW centred on the agent.

        Out-of-bounds cells are treated as explored (1.0) so walls are
        not confused with unexplored space.
        """
        half = LOCAL_MAP_WINDOW // 2
        col, row = self._world_to_grid(self.x, self.y)

        # Build window with boundary fill = 1.0 (visited / wall)
        window = np.ones((LOCAL_MAP_WINDOW, LOCAL_MAP_WINDOW), dtype=np.float32)

        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                r = row + dr
                c = col + dc
                wr = dr + half
                wc = dc + half
                if 0 <= r < MAP_HEIGHT_CELLS and 0 <= c < MAP_WIDTH_CELLS:
                    window[wr, wc] = float(self.grid[r, c])

        return window.flatten()

    def normalised_pose(self) -> np.ndarray:
        """
        Return (x_norm, y_norm, cos_theta, sin_theta) as a float32 array.
        x_norm, y_norm are in [-1, 1] relative to the map extents.
        cos/sin encode heading without the ±pi discontinuity.
        """
        max_x = (MAP_WIDTH_CELLS  // 2) * GRID_SIZE
        max_y = (MAP_HEIGHT_CELLS // 2) * GRID_SIZE
        x_n = np.clip(self.x / max_x, -1.0, 1.0)
        y_n = np.clip(self.y / max_y, -1.0, 1.0)
        return np.array(
            [x_n, y_n, math.cos(self.theta), math.sin(self.theta)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_to_grid(self, x: float, y: float):
        col = int(x / GRID_SIZE) + self._half_w
        row = int(y / GRID_SIZE) + self._half_h
        col = max(0, min(MAP_WIDTH_CELLS  - 1, col))
        row = max(0, min(MAP_HEIGHT_CELLS - 1, row))
        return col, row

    def _mark_current_cell(self) -> bool:
        col, row = self._world_to_grid(self.x, self.y)
        if self.grid[row, col]:
            return False          # already visited
        self.grid[row, col] = True
        return True               # newly explored

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def cells_visited(self) -> int:
        return int(self.grid.sum())
