from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


class ClearanceEvaluator:
    def __init__(self, static_obstacles, robot_radius, boundary_spacing=0.04):
        self.static = list(static_obstacles)
        self.robot_radius = float(robot_radius)
        self.spacing = min(float(boundary_spacing), 0.05)
        pts = [o.boundary_points(self.spacing, 0.0) for o in self.static]
        self.points = np.vstack(pts) if pts else np.empty((0, 2))
        self.tree = cKDTree(self.points) if len(self.points) else None

    def _dist_set(self, p, obstacles, tree=None):
        if not obstacles:
            return np.inf
        p = np.asarray(p, float)
        best = np.inf
        for o in obstacles:
            if o.name == "circle":
                best = min(best, np.linalg.norm(p - o.center) - o.radius)
        if tree is not None:
            d, _ = tree.query(p)
            inside = any(o.contains(p, 0.0) for o in obstacles)
            best = min(best, -float(d) if inside else float(d))
        else:
            for o in obstacles:
                if o.name != "circle":
                    pts = o.boundary_points(self.spacing, 0.0)
                    d = float(np.min(np.linalg.norm(pts - p, axis=1)))
                    best = min(best, -d if o.contains(p, 0.0) else d)
        return float(best)

    def clearance(self, p, dynamic_obstacles=None):
        ds = self._dist_set(p, self.static, self.tree)
        dd = self._dist_set(p, list(dynamic_obstacles or []), None)
        return float(min(ds, dd) - self.robot_radius)
