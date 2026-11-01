from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time

import numpy as np


@dataclass
class JPSPlanResult:
    path: np.ndarray | None
    success: bool
    computation_time: float


@dataclass
class _Node:
    x: int
    y: int
    cost: float
    parent_index: int
    dx: int = 0
    dy: int = 0


class JPSWrapper:
    """Self-contained, grid-based 8-connected Jump Point Search.

    ``grid`` follows the project convention: axis 0 is x, axis 1 is y,
    and values above 50 are occupied.  The optional grid-cell dilation is
    deliberately a planning-only clearance margin; it does not alter the
    physical collision model or the SCP/BaS safety field.
    """

    _NSZ = ((8, 0), (1, 2), (3, 2))
    _NS = (
        ((-1, 0, -1), (0, -1, -1)),
        ((0,), (-1,)),
        ((1, 0, 1), (0, -1, -1)),
        ((-1,), (0,)),
        ((1, -1, 0, 1, -1, 0, 1, -1), (0, 0, 1, 1, 1, -1, -1, -1)),
        ((1,), (0,)),
        ((-1, 0, -1), (0, 1, 1)),
        ((0,), (1,)),
        ((1, 0, 1), (0, 1, 1)),
    )
    _F1 = (
        ((1, 0), (0, 1)),
        ((1, -1), (0, 0)),
        ((-1, 0), (0, 1)),
        ((0, 0), (1, -1)),
        ((), ()),
        ((0, 0), (1, -1)),
        ((1, 0), (0, -1)),
        ((1, -1), (0, 0)),
        ((-1, 0), (0, -1)),
    )
    _F2 = (
        ((1, -1), (-1, 1)),
        ((1, -1), (-1, -1)),
        ((-1, 1), (-1, 1)),
        ((-1, -1), (1, -1)),
        ((), ()),
        ((1, 1), (1, -1)),
        ((1, -1), (1, -1)),
        ((1, -1), (1, 1)),
        ((-1, 1), (1, -1)),
    )

    def __init__(
        self,
        width,
        height,
        resolution,
        expected_version=None,
        verbose=False,
        extra_dilation_cells=1,
    ):
        del expected_version
        self.width = float(width)
        self.height = float(height)
        self.resolution = float(resolution)
        self.verbose = bool(verbose)
        self.extra_dilation_cells = max(0, int(extra_dilation_cells))
        self._raw_grid_cache = None
        self._occupancy_cache = None

    @staticmethod
    def _dilate(occupied: np.ndarray, iterations: int) -> np.ndarray:
        result = np.asarray(occupied, dtype=bool).copy()
        for _ in range(iterations):
            padded = np.pad(result, 1, mode="constant", constant_values=False)
            grown = np.zeros_like(result)
            nx, ny = result.shape
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    grown |= padded[1 + ox : 1 + ox + nx, 1 + oy : 1 + oy + ny]
            result = grown
        return result

    def _occupancy(self, grid: np.ndarray) -> np.ndarray:
        raw = np.asarray(grid, dtype=float) > 50.0
        if self._raw_grid_cache is not None and np.array_equal(
            raw, self._raw_grid_cache
        ):
            return self._occupancy_cache
        occupied = self._dilate(raw, self.extra_dilation_cells)
        self._raw_grid_cache = raw.copy()
        self._occupancy_cache = occupied
        return occupied

    @staticmethod
    def _dir_id(dx: int, dy: int) -> int:
        return int(dx) + 1 + 3 * (int(dy) + 1)

    @staticmethod
    def _grid_index(node: _Node, ny: int) -> int:
        return node.x * ny + node.y

    @staticmethod
    def _heuristic(x: int, y: int, gx: int, gy: int) -> float:
        dx = abs(x - gx)
        dy = abs(y - gy)
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)

    @staticmethod
    def _is_walkable(occupied: np.ndarray, x: int, y: int) -> bool:
        return (
            0 <= x < occupied.shape[0]
            and 0 <= y < occupied.shape[1]
            and not occupied[x, y]
        )

    @staticmethod
    def _is_occupied(occupied: np.ndarray, x: int, y: int) -> bool:
        return (
            0 <= x < occupied.shape[0] and 0 <= y < occupied.shape[1] and occupied[x, y]
        )

    def _has_forced(
        self, occupied: np.ndarray, x: int, y: int, dx: int, dy: int
    ) -> bool:
        direction = self._dir_id(dx, dy)
        f1x, f1y = self._F1[direction]
        for i in range(self._NSZ[abs(dx) + abs(dy)][1]):
            if self._is_occupied(occupied, x + f1x[i], y + f1y[i]):
                return True
        return False

    def _jump(
        self,
        occupied: np.ndarray,
        x: int,
        y: int,
        dx: int,
        dy: int,
        gx: int,
        gy: int,
    ) -> tuple[int, int] | None:
        nx = x + dx
        ny = y + dy
        while True:
            if not self._is_walkable(occupied, nx, ny):
                return None
            if (nx == gx and ny == gy) or self._has_forced(occupied, nx, ny, dx, dy):
                return nx, ny

            direction = self._dir_id(dx, dy)
            num_natural = self._NSZ[abs(dx) + abs(dy)][0]
            nsx, nsy = self._NS[direction]
            for i in range(num_natural - 1):
                if self._jump(occupied, nx, ny, nsx[i], nsy[i], gx, gy) is not None:
                    return nx, ny

            next_x = nx + dx
            next_y = ny + dy
            if not self._is_walkable(occupied, next_x, next_y):
                return nx, ny
            nx, ny = next_x, next_y

    def _successors(
        self, occupied: np.ndarray, current: _Node, gx: int, gy: int
    ) -> list[_Node]:
        direction = self._dir_id(current.dx, current.dy)
        num_natural, num_forced = self._NSZ[abs(current.dx) + abs(current.dy)]
        nsx, nsy = self._NS[direction]
        successors: list[_Node] = []

        for i in range(num_natural):
            jump = self._jump(occupied, current.x, current.y, nsx[i], nsy[i], gx, gy)
            if jump is not None:
                jx, jy = jump
                successors.append(
                    _Node(
                        jx,
                        jy,
                        current.cost + math.hypot(jx - current.x, jy - current.y),
                        -1,
                        nsx[i],
                        nsy[i],
                    )
                )

        f1x, f1y = self._F1[direction]
        f2x, f2y = self._F2[direction]
        for i in range(num_forced):
            if self._is_occupied(
                occupied, current.x + f1x[i], current.y + f1y[i]
            ) and self._is_walkable(occupied, current.x + f2x[i], current.y + f2y[i]):
                dx = f2x[i]
                dy = f2y[i]
                jump = self._jump(occupied, current.x, current.y, dx, dy, gx, gy)
                if jump is not None:
                    jx, jy = jump
                    successors.append(
                        _Node(
                            jx,
                            jy,
                            current.cost + math.hypot(jx - current.x, jy - current.y),
                            -1,
                            dx,
                            dy,
                        )
                    )
        return successors

    def _search(
        self, occupied: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
    ):
        nx, ny = occupied.shape
        sx, sy = start
        gx, gy = goal
        start_node = _Node(sx, sy, 0.0, -1)
        start_id = self._grid_index(start_node, ny)
        nodes: dict[int, _Node] = {start_id: start_node}
        best_cost: dict[int, float] = {start_id: 0.0}
        closed: set[int] = set()
        heap = [(self._heuristic(sx, sy, gx, gy), 0.0, start_id)]

        while heap:
            _, queued_cost, current_id = heapq.heappop(heap)
            if current_id in closed:
                continue
            current = nodes[current_id]
            if queued_cost > current.cost + 1.0e-12:
                continue
            closed.add(current_id)
            if current.x == gx and current.y == gy:
                return current_id, nodes

            for node in self._successors(occupied, current, gx, gy):
                node_id = self._grid_index(node, ny)
                if (
                    node_id in closed
                    or node.cost >= best_cost.get(node_id, math.inf) - 1.0e-12
                ):
                    continue
                node.parent_index = current_id
                best_cost[node_id] = node.cost
                nodes[node_id] = node
                score = node.cost + self._heuristic(node.x, node.y, gx, gy)
                heapq.heappush(heap, (score, node.cost, node_id))
        return None, nodes

    @staticmethod
    def _dense_cells(goal_id: int, nodes: dict[int, _Node]) -> np.ndarray:
        waypoints = []
        node_id = goal_id
        while node_id != -1:
            node = nodes[node_id]
            waypoints.append((node.x, node.y))
            node_id = node.parent_index
        waypoints.reverse()

        dense = [waypoints[0]]
        for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
            dx = int(np.sign(x1 - x0))
            dy = int(np.sign(y1 - y0))
            count = max(abs(x1 - x0), abs(y1 - y0))
            dense.extend((x0 + k * dx, y0 + k * dy) for k in range(1, count + 1))
        return np.asarray(dense, dtype=float)

    def plan(self, grid, start, goal):
        t0 = time.perf_counter()
        occupied = self._occupancy(np.asarray(grid, dtype=float))
        start = np.asarray(start, dtype=float).reshape(2)
        goal = np.asarray(goal, dtype=float).reshape(2)
        start_cell = tuple(np.rint(start / self.resolution).astype(int))
        goal_cell = tuple(np.rint(goal / self.resolution).astype(int))

        if not self._is_walkable(occupied, *start_cell) or not self._is_walkable(
            occupied, *goal_cell
        ):
            return JPSPlanResult(None, False, time.perf_counter() - t0)
        if start_cell == goal_cell:
            return JPSPlanResult(
                np.vstack((start, goal)), True, time.perf_counter() - t0
            )

        goal_id, nodes = self._search(occupied, start_cell, goal_cell)
        elapsed = time.perf_counter() - t0
        if goal_id is None:
            return JPSPlanResult(None, False, elapsed)

        cells = self._dense_cells(goal_id, nodes)
        path = cells * self.resolution
        path[0] = start
        path[-1] = goal
        keep = np.ones(len(path), dtype=bool)
        if len(path) > 1:
            keep[1:] = np.linalg.norm(np.diff(path, axis=0), axis=1) > 1.0e-10
        path = path[keep]
        if self.verbose:
            print(
                f"[JPS] success: {len(path)} path points, "
                f"{elapsed * 1.0e3:.2f} ms, clearance buffer "
                f"{self.extra_dilation_cells} cell(s)"
            )
        return JPSPlanResult(path, True, elapsed)
