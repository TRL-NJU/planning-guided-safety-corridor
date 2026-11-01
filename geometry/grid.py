from __future__ import annotations
import numpy as np


def rasterize_obstacles(obstacles, width, height, resolution, clearance):
    nx = int(round(width / resolution))
    ny = int(round(height / resolution))
    xs = np.arange(nx) * resolution
    ys = np.arange(ny) * resolution
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    occ = np.zeros((nx, ny), bool)
    for obs in obstacles:
        if obs.name == "circle":
            R = obs.radius + clearance
            occ |= (X - obs.center[0]) ** 2 + (Y - obs.center[1]) ** 2 <= R * R
        else:
            c, s = np.cos(obs.angle), np.sin(obs.angle)
            dx = X - obs.center[0]
            dy = Y - obs.center[1]
            qx = c * dx + s * dy
            qy = -s * dx + c * dy
            A = obs.a + clearance
            B = obs.b + clearance
            n = obs.power
            occ |= (qx / A) ** n + (qy / B) ** n <= 1.0
    grid = np.zeros((nx, ny), float)
    grid[occ] = 100.0
    grid[0, :] = grid[-1, :] = 100.0
    grid[:, 0] = grid[:, -1] = 100.0
    return grid


def point_to_grid(p, resolution, shape):
    p = np.asarray(p, float).reshape(2)
    ix = int(round(p[0] / resolution))
    iy = int(round(p[1] / resolution))
    return min(max(ix, 0), shape[0] - 1), min(max(iy, 0), shape[1] - 1)
