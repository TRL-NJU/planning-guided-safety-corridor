from __future__ import annotations
from collections import deque
from geometry.grid import rasterize_obstacles, point_to_grid


def grid_connectivity_feasible(obstacles, start, goal, width, height, resolution, rho):
    grid = rasterize_obstacles(obstacles, width, height, resolution, rho)
    s = point_to_grid(start, resolution, grid.shape)
    g = point_to_grid(goal, resolution, grid.shape)
    if grid[s] > 50 or grid[g] > 50:
        return False
    q = deque([s])
    seen = {s}
    nx, ny = grid.shape
    nbr = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while q:
        x, y = q.popleft()
        if (x, y) == g:
            return True
        for dx, dy in nbr:
            xx, yy = x + dx, y + dy
            if (
                not (0 <= xx < nx and 0 <= yy < ny)
                or grid[xx, yy] > 50
                or (xx, yy) in seen
            ):
                continue
            if dx and dy and (grid[x + dx, y] > 50 or grid[x, y + dy] > 50):
                continue
            seen.add((xx, yy))
            q.append((xx, yy))
    return False
