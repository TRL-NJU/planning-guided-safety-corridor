"""Physical workspace walls as shared environment inputs, not a controller.

Wall half-spaces feed the existing obstacle BaS interface; sampled wall faces
feed the existing SCP interface. Occupancy and disk-wall clearance are exact.
No command clipping, state projection, or extra avoidance policy is applied.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WorkspaceWall:
    start: np.ndarray
    end: np.ndarray
    inward: np.ndarray
    name: str = "workspace_wall"
    _samples: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.start = np.asarray(self.start, float)
        self.end = np.asarray(self.end, float)
        self.inward = np.asarray(self.inward, float)
        self.center = (self.start + self.end) / 2

    def bounding_radius(self, clearance=0.0):
        return float(np.linalg.norm(self.end - self.start) / 2 + clearance)

    def h_and_grad(self, point, clearance=0.0):
        return float(
            (np.asarray(point) - self.start) @ self.inward - clearance
        ), self.inward.copy()

    def contains(self, point, clearance=0.0):
        return self.h_and_grad(point, clearance)[0] <= 0

    def boundary_points(self, spacing, clearance=0.0):
        key = (float(spacing), float(clearance))
        if key not in self._samples:
            count = max(
                1,
                int(
                    np.ceil(np.linalg.norm(self.end - self.start) / max(spacing, 1e-4))
                ),
            )
            self._samples[key] = (
                np.linspace(self.start, self.end, count + 1) + clearance * self.inward
            )
        return self._samples[key]


def workspace_walls(width, height):
    return [
        WorkspaceWall([0, 0], [0, height], [1, 0]),
        WorkspaceWall([width, 0], [width, height], [-1, 0]),
        WorkspaceWall([0, 0], [width, 0], [0, 1]),
        WorkspaceWall([0, height], [width, height], [0, -1]),
    ]


def wall_clearance(point, width, height, radius):
    x, y = np.asarray(point, float)
    return float(min(x, y, width - x, height - y) - radius)


def apply_wall_occupancy(grid, width, height, resolution, clearance):
    """Inflate the four physical walls by robot radius + safety margin."""
    xs = np.arange(grid.shape[0]) * resolution
    ys = np.arange(grid.shape[1]) * resolution
    grid[(xs <= clearance) | (xs >= width - clearance), :] = 100.0
    grid[:, (ys <= clearance) | (ys >= height - clearance)] = 100.0
    return grid
