"""Input-state extension with an inverse-barrier velocity constraint."""

from dataclasses import dataclass

import numpy as np


@dataclass
class InputStateUpdate:
    velocity: np.ndarray
    rate: np.ndarray
    margin: float
    barrier_state: float
    barrier_rate: float
    barrier_feedback_norm: float
    equation_residual: float


class InputBarrierController:
    """Integrate dot(u) = k_v(w-u) - k_u grad(W_u) without output clipping.

    g(u) = 1 - ||u||^2 / U^2, z_u = 1/g - 1, W_u = z_u^2/2.
    The implicit Euler equation is radial and has one root inside ||u||<U.
    The scalar solve is a time-integration step, not a velocity projection.
    """

    def __init__(self, speed_limit, tracking_gain=200.0, barrier_gain=0.02):
        self.speed_limit = float(speed_limit)
        self.tracking_gain = float(tracking_gain)
        self.barrier_gain = float(barrier_gain)
        if not all(
            np.isfinite(x) and x > 0
            for x in (self.speed_limit, self.tracking_gain, self.barrier_gain)
        ):
            raise ValueError(
                "Input-state gains and speed limit must be positive and finite"
            )

    def barrier(self, velocity):
        velocity = np.asarray(velocity, float)
        margin = 1.0 - float(velocity @ velocity) / self.speed_limit**2
        if margin <= 0:
            raise ValueError("Input state must start strictly inside the speed ball")
        state = 1.0 / margin - 1.0
        gradient = 2.0 * velocity / (self.speed_limit**2 * margin**2)
        return margin, state, gradient

    def advance(self, velocity, target, dt):
        velocity = np.asarray(velocity, float).reshape(2)
        target = np.asarray(target, float).reshape(2)
        if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(target)):
            raise ValueError("Velocity states and targets must be finite")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("The input integration step must be positive and finite")
        self.barrier(velocity)
        rhs = velocity + dt * self.tracking_gain * target
        magnitude = float(np.linalg.norm(rhs))
        if not np.isfinite(magnitude):
            raise ValueError("Input target exceeds numerical range")
        if magnitude == 0:
            next_velocity = np.zeros(2)
        else:
            linear = 1.0 + dt * self.tracking_gain
            low = 0.0
            high = min(magnitude / linear, np.nextafter(self.speed_limit, 0.0))
            # The nonlinear term grows to infinity at the input boundary.
            for _ in range(64):
                radius = (low + high) / 2.0
                g = 1.0 - (radius / self.speed_limit) ** 2
                z = 1.0 / g - 1.0
                damping = 2.0 * self.barrier_gain * z / (self.speed_limit**2 * g**2)
                if (linear + dt * damping) * radius > magnitude:
                    high = radius
                else:
                    low = radius
            next_velocity = ((low + high) / 2.0) * (rhs / magnitude)
        margin, z, gradient = self.barrier(next_velocity)
        rate = (next_velocity - velocity) / dt
        correction = -self.barrier_gain * z * gradient
        expected_rate = self.tracking_gain * (target - next_velocity) + correction
        return InputStateUpdate(
            next_velocity,
            rate,
            margin,
            z,
            float(gradient @ rate),
            float(np.linalg.norm(correction)),
            float(np.linalg.norm(rate - expected_rate)),
        )
