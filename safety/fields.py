from __future__ import annotations
import numpy as np
from scipy.special import logsumexp


class ConstantField:
    def __init__(self, h=10.0):
        self.h = float(h)

    def value_grad(self, p):
        return self.h, np.zeros(2)

    def value(self, p):
        return self.h

    def values(self, points):
        return np.full(len(np.asarray(points).reshape(-1, 2)), self.h)


class ObstacleAggregateField:
    """Positive-domain harmonic aggregation from Barrier States theory."""

    def __init__(self, obstacles, clearance, free_value=10.0):
        self.obstacles = list(obstacles)
        self.clearance = float(clearance)
        self.free_value = float(free_value)

    def value_grad(self, p):
        if not self.obstacles:
            return self.free_value, np.zeros(2)
        vals = []
        grads = []
        for o in self.obstacles:
            h, g = o.h_and_grad(p, self.clearance)
            vals.append(h)
            grads.append(g)
        vals = np.asarray(vals, float)
        grads = np.asarray(grads, float)
        if np.min(vals) <= 0:
            i = int(np.argmin(vals))
            return float(vals[i]), grads[i].copy()
        total = float(np.sum(1.0 / vals))
        h = 1.0 / total
        grad = h * h * np.sum(grads / (vals[:, None] ** 2), axis=0)
        return float(h), grad

    def value(self, p):
        return self.value_grad(p)[0]


class SCPField:
    def __init__(self, A, b, kappa, seed=None, polygon=None):
        self.A = np.asarray(A, float)
        self.b = np.asarray(b, float).reshape(-1)
        self.kappa = float(kappa)
        self.seed = None if seed is None else np.asarray(seed, float).reshape(2)
        self.polygon = None if polygon is None else np.asarray(polygon, float)

    def facet_values(self, p):
        return self.A @ np.asarray(p, float).reshape(2) + self.b

    def values(self, points):
        points = np.asarray(points, float).reshape(-1, 2)
        hs = points @ self.A.T + self.b[None, :]
        return -logsumexp(-self.kappa * hs, axis=1) / self.kappa

    def value_grad(self, p):
        hs = self.facet_values(p)
        x = -self.kappa * hs
        lse = logsumexp(x)
        H = -lse / self.kappa
        w = np.exp(x - lse)
        return float(H), np.sum(w[:, None] * self.A, axis=0)

    def value(self, p):
        return float(self.values(np.asarray(p, float).reshape(1, 2))[0])


class CorridorField:
    """Smooth union of local SCP fields with adaptive compensation."""

    def __init__(
        self,
        local_fields,
        softmax_kappa,
        fusion_mode="adaptive_entropy_weighted_average",
    ):
        if not local_fields:
            raise ValueError("Need >=1 local field")
        self.local_fields = list(local_fields)
        self.kappa = float(softmax_kappa)
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in {
            "adaptive_entropy_weighted_average",
            "fixed_compensated_lse",
        }:
            raise ValueError("Unsupported corridor fusion mode.")
        self._lengths = np.asarray([len(field.b) for field in self.local_fields], int)
        self._starts = np.r_[0, np.cumsum(self._lengths[:-1])]
        self._A = np.vstack([field.A for field in self.local_fields])
        self._b = np.concatenate([field.b for field in self.local_fields])
        self._local_kappa = np.asarray(
            [field.kappa for field in self.local_fields], float
        )
        self._facet_kappa = np.repeat(self._local_kappa, self._lengths)

    def _local_values(self, points):
        points = np.asarray(points, float).reshape(-1, 2)
        hs = points @ self._A.T + self._b[None, :]
        logits = -hs * self._facet_kappa[None, :]
        maxima = np.maximum.reduceat(logits, self._starts, axis=1)
        exponent = np.exp(logits - np.repeat(maxima, self._lengths, axis=1))
        sums = np.add.reduceat(exponent, self._starts, axis=1)
        return -(maxima + np.log(sums)) / self._local_kappa[None, :]

    def _local_value_grads(self, p):
        hs = self._A @ np.asarray(p, float).reshape(2) + self._b
        logits = -hs * self._facet_kappa
        maxima = np.maximum.reduceat(logits, self._starts)
        exponent = np.exp(logits - np.repeat(maxima, self._lengths))
        sums = np.add.reduceat(exponent, self._starts)
        values = -(maxima + np.log(sums)) / self._local_kappa
        facet_weights = exponent / np.repeat(sums, self._lengths)
        gradients = np.add.reduceat(
            facet_weights[:, None] * self._A, self._starts, axis=0
        )
        return values, gradients

    def local_values(self, p):
        return self._local_values(np.asarray(p, float).reshape(1, 2))[0]

    def _weights(self, local):
        logits = self.kappa * np.asarray(local, float)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        weights = np.exp(logits)
        return weights / np.sum(weights, axis=-1, keepdims=True)

    def values(self, points):
        local = self._local_values(points)
        if self.fusion_mode == "fixed_compensated_lse":
            return (
                logsumexp(self.kappa * local, axis=1) - np.log(local.shape[1])
            ) / self.kappa
        weights = self._weights(local)
        return np.sum(weights * local, axis=1)

    def value_grad(self, p):
        """Scalar corridor value and its exact spatial gradient."""
        local, gradient = self._local_value_grads(p)
        logits = self.kappa * local
        lse = logsumexp(logits)
        weights = np.exp(logits - lse)
        if self.fusion_mode == "fixed_compensated_lse":
            H = float((lse - np.log(len(local))) / self.kappa)
            coefficients = weights
        else:
            H = float(weights @ local)
            coefficients = weights * (1.0 + self.kappa * (local - H))
        return H, np.sum(coefficients[:, None] * gradient, axis=0)

    def value(self, p):
        return float(self.values(np.asarray(p, float).reshape(1, 2))[0])

    def weights(self, p):
        return self._weights(self.local_values(p))

    def adaptive_compensation(self, p):
        if self.fusion_mode == "fixed_compensated_lse":
            return float(np.log(len(self.local_fields)) / self.kappa)
        weights = self.weights(p)
        positive = weights[weights > 0.0]
        return float(-np.sum(positive * np.log(positive)) / self.kappa)
