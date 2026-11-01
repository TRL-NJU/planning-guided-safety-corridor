from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RobotState:
    p: np.ndarray
    v: np.ndarray


class SingleIntegrator:
    def __init__(self, speed_max):
        self.speed_max = float(speed_max)

    def step(self, state, u, dt):
        vel = np.asarray(u, float)
        if not np.all(np.isfinite(vel)) or np.linalg.norm(vel) >= self.speed_max:
            raise ValueError(
                "Applied velocity must satisfy the strict input constraint"
            )
        return RobotState(state.p + 0.5 * (state.v + vel) * dt, vel)

    def hold_control(self, state, cfg):
        return np.zeros(2)


def make_dynamics(name, cfg):
    if name == "single":
        return SingleIntegrator(float(cfg["limits"]["single_speed_max"]))
    raise ValueError(name)
