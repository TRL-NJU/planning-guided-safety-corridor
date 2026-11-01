from __future__ import annotations
import numpy as np


class InverseBarrier:
    def __init__(self, epsilon=1e-3):
        self.epsilon = float(epsilon)

    def safe_h(self, h):
        return max(float(h), self.epsilon)

    def value(self, h):
        return 1.0 / self.safe_h(h)

    def derivative(self, h):
        hs = self.safe_h(h)
        return -1.0 / (hs * hs)

    def state(self, h, h_eq):
        return self.value(h) - self.value(h_eq)

    def coupling_direction(self, h, grad_h):
        return self.derivative(h) * np.asarray(grad_h, float)
