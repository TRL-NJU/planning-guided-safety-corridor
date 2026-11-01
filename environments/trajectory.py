from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PiecewiseLinearTrajectory:
    times: np.ndarray
    positions: np.ndarray

    def __post_init__(self):
        self.times = np.asarray(self.times, float)
        self.positions = np.asarray(self.positions, float)
        if self.positions.shape != (len(self.times), 2):
            raise ValueError("positions must have shape (len(times), 2)")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing")

    def position(self, t):
        t = float(t)
        if t <= self.times[0]:
            return self.positions[0].copy()
        if t >= self.times[-1]:
            return self.positions[-1].copy()
        i = int(np.searchsorted(self.times, t, side="right") - 1)
        a = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
        return (1 - a) * self.positions[i] + a * self.positions[i + 1]

    def velocity(self, t):
        if t <= self.times[0] or t >= self.times[-1]:
            return np.zeros(2)
        i = int(np.searchsorted(self.times, t, side="right") - 1)
        return (self.positions[i + 1] - self.positions[i]) / (
            self.times[i + 1] - self.times[i]
        )

    def to_dict(self):
        return {
            "type": "piecewise_linear",
            "times": self.times.tolist(),
            "positions": self.positions.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["times"], float), np.asarray(d["positions"], float))


@dataclass
class PingPongTrajectory:
    """Constant-speed periodic A -> B -> A motion."""

    a: np.ndarray
    b: np.ndarray
    speed: float

    def __post_init__(self):
        self.a = np.asarray(self.a, float).reshape(2)
        self.b = np.asarray(self.b, float).reshape(2)
        self.speed = float(self.speed)
        if self.speed <= 0:
            raise ValueError("PingPongTrajectory speed must be positive")
        if np.linalg.norm(self.b - self.a) <= 1e-12:
            raise ValueError("PingPongTrajectory endpoints A and B must differ")

    @property
    def length(self):
        return float(np.linalg.norm(self.b - self.a))

    @property
    def period(self):
        return 2.0 * self.length / self.speed

    def position(self, t):
        L = self.length
        d = (max(float(t), 0.0) * self.speed) % (2.0 * L)
        direction = (self.b - self.a) / L
        if d <= L:
            return self.a + d * direction
        return self.b - (d - L) * direction

    def velocity(self, t):
        L = self.length
        d = (max(float(t), 0.0) * self.speed) % (2.0 * L)
        direction = (self.b - self.a) / L
        return self.speed * direction if d < L else -self.speed * direction

    def to_dict(self):
        return {
            "type": "ping_pong",
            "a": self.a.tolist(),
            "b": self.b.tolist(),
            "speed": self.speed,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            np.asarray(d["a"], float), np.asarray(d["b"], float), float(d["speed"])
        )


def trajectory_from_dict(d):
    # Backward-compatible: v1/v2 files had no explicit type.
    typ = d.get("type", "piecewise_linear")
    if typ == "piecewise_linear":
        return PiecewiseLinearTrajectory.from_dict(d)
    if typ == "ping_pong":
        return PingPongTrajectory.from_dict(d)
    raise ValueError(f"Unknown trajectory type: {typ}")
