from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class CircleObstacle:
    center: np.ndarray
    radius: float
    name: str = "circle"

    def __post_init__(self):
        self.center = np.asarray(self.center, float).reshape(2)
        self.radius = float(self.radius)

    def h_and_grad(self, p, clearance=0.0):
        p = np.asarray(p, float).reshape(2)
        R = self.radius + float(clearance)
        d = p - self.center
        dist = float(np.linalg.norm(d))
        if dist <= 1.0e-12:
            return -R, np.zeros(2)
        return dist - R, d / dist

    def contains(self, p, clearance=0.0):
        return self.h_and_grad(p, clearance)[0] <= 0.0

    def boundary_points(self, spacing, clearance=0.0):
        R = self.radius + float(clearance)
        n = max(24, int(math.ceil(2 * math.pi * R / max(spacing, 1e-4))))
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return self.center + np.c_[R * np.cos(t), R * np.sin(t)]

    def bounding_radius(self, clearance=0.0):
        return self.radius + float(clearance)

    def moved_to(self, center):
        return CircleObstacle(np.asarray(center, float), self.radius)

    def to_dict(self):
        return {"type": "circle", "center": self.center.tolist(), "radius": self.radius}


@dataclass
class SuperellipseObstacle:
    center: np.ndarray
    a: float
    b: float
    power: int = 8
    angle: float = 0.0
    name: str = "superellipse"

    def __post_init__(self):
        self.center = np.asarray(self.center, float).reshape(2)
        self.a, self.b = float(self.a), float(self.b)
        self.power, self.angle = int(self.power), float(self.angle)
        if self.power < 2 or self.power % 2:
            raise ValueError("power must be an even integer >=2")

    @property
    def R(self):
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.array([[c, -s], [s, c]], float)

    def _local(self, p):
        return self.R.T @ (np.asarray(p, float).reshape(2) - self.center)

    def h_and_grad(self, p, clearance=0.0):
        # Distance-like level set: preserve h=0 while bounding the gradient.
        q = self._local(p)
        A = self.a + clearance
        B = self.b + clearance
        n = self.power
        x, y = float(q[0]), float(q[1])
        ax = abs(x / A)
        ay = abs(y / B)
        power_sum = ax**n + ay**n
        scale = min(A, B)
        if power_sum <= 1.0e-24:
            return -scale, np.zeros(2)
        radius = power_sum ** (1.0 / n)
        common = power_sum ** (1.0 / n - 1.0)
        gl = scale * np.array(
            [
                common * np.sign(x) * ax ** (n - 1) / A,
                common * np.sign(y) * ay ** (n - 1) / B,
            ],
            float,
        )
        h = scale * (radius - 1.0)
        return float(h), self.R @ gl

    def contains(self, p, clearance=0.0):
        return self.h_and_grad(p, clearance)[0] <= 0.0

    def boundary_points(self, spacing, clearance=0.0):
        """Sample the closed boundary with a true maximum spatial spacing."""
        cache_key = (
            float(spacing),
            float(clearance),
            float(self.a),
            float(self.b),
            int(self.power),
            float(self.angle),
            float(self.center[0]),
            float(self.center[1]),
        )
        cache = getattr(self, "_boundary_point_cache", {})
        cached = cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        A = self.a + clearance
        B = self.b + clearance
        max_spacing = max(float(spacing), 1.0e-4)
        exponent = 2.0 / self.power

        def point(t):
            ct, st = math.cos(t), math.sin(t)
            return np.array(
                [
                    A * math.copysign(abs(ct) ** exponent, ct),
                    B * math.copysign(abs(st) ** exponent, st),
                ],
                float,
            )

        points = []

        # Uniform parameter-angle sampling leaves metre-scale gaps on long,
        # high-power superellipses, so refine each quadrant by arc length.
        def refine(t0, p0, t1, p1, depth=0):
            tm = 0.5 * (t0 + t1)
            pm = point(tm)
            arc_est = float(np.linalg.norm(pm - p0) + np.linalg.norm(p1 - pm))
            if arc_est <= max_spacing or depth >= 30:
                points.append(p1)
                return
            refine(t0, p0, tm, pm, depth + 1)
            refine(tm, pm, t1, p1, depth + 1)

        quarter = 0.5 * math.pi
        points.append(point(0.0))
        for quadrant in range(4):
            t0 = quadrant * quarter
            t1 = (quadrant + 1) * quarter
            refine(t0, point(t0), t1, point(t1))
        local = np.asarray(points[:-1], float)  # omit duplicated 2*pi endpoint
        output = self.center + local @ self.R.T
        if len(cache) >= 8:
            cache.clear()
        cache[cache_key] = output.copy()
        self._boundary_point_cache = cache
        return output

    def bounding_radius(self, clearance=0.0):
        return math.hypot(self.a + clearance, self.b + clearance)

    def moved_to(self, center):
        return SuperellipseObstacle(
            np.asarray(center, float), self.a, self.b, self.power, self.angle
        )

    def to_dict(self):
        return {
            "type": "superellipse",
            "center": self.center.tolist(),
            "a": self.a,
            "b": self.b,
            "power": self.power,
            "angle": self.angle,
        }


def obstacle_from_dict(d):
    if d["type"] == "circle":
        return CircleObstacle(np.asarray(d["center"], float), float(d["radius"]))
    if d["type"] == "superellipse":
        return SuperellipseObstacle(
            np.asarray(d["center"], float),
            float(d["a"]),
            float(d["b"]),
            int(d.get("power", 8)),
            float(d.get("angle", 0)),
        )
    raise ValueError(d["type"])
