from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
import time
import numpy as np
from safety.fields import ObstacleAggregateField
from safety.corridor import build_single_scp, build_lookahead_corridor


@dataclass
class SafetyBuild:
    success: bool
    field: object | None
    p_eq: np.ndarray | None
    reason: str
    safety_time: float
    scp_time: float = 0.0
    fusion_time: float = 0.0
    seeds: np.ndarray | None = None
    polygons: list | None = None
    diagnostics: dict = dataclass_field(default_factory=dict)


def sensed_obstacle_points(obstacles, current, sensing_range, spacing):
    points = []
    current = np.asarray(current, float)
    for obstacle in obstacles:
        if (
            np.linalg.norm(obstacle.center - current) - obstacle.bounding_radius()
            > sensing_range
        ):
            continue
        boundary = obstacle.boundary_points(spacing, 0.0)
        mask = np.linalg.norm(boundary - current, axis=1) <= sensing_range
        if np.any(mask):
            points.append(boundary[mask])
    return np.vstack(points) if points else np.empty((0, 2))


def sensed_obstacles(obstacles, current, sensing_range):
    return [
        obstacle
        for obstacle in obstacles
        if np.linalg.norm(obstacle.center - current) - obstacle.bounding_radius()
        <= sensing_range
    ]


def choose_path_equilibrium(path, current, field, cfg):
    s0, _, _ = path.project(current)
    ds = float(cfg["barrier"]["equilibrium_sample_ds"])
    smax = min(path.length, s0 + float(cfg["barrier"]["equilibrium_horizon"]))
    margin = float(cfg["barrier"]["equilibrium_margin"])
    best = None
    s = s0
    while s <= smax + 1e-12:
        q = path.point_at(s)
        if field.value(q) > margin:
            best = q
        s += ds
    return best


def choose_goal_equilibrium(path, field, cfg):
    """Use the actual goal only after it is robustly inside the current field."""
    goal = np.asarray(path.points[-1], float).copy()
    margin = float(cfg["barrier"]["equilibrium_margin"])
    return goal if field.value(goal) > margin else None


def build_method_safety(
    method,
    current,
    path,
    all_current_obstacles,
    local_points,
    cfg,
    rho,
    poisson_reconstructor=None,
):
    if method == "jps":
        return SafetyBuild(True, None, None, "not_applicable", 0.0)
    if method == "jps_bas":
        t0 = time.perf_counter()
        obstacles = sensed_obstacles(
            all_current_obstacles, current, float(cfg["perception"]["range"])
        )
        field = ObstacleAggregateField(
            obstacles, rho, float(cfg["barrier"]["free_field_value"])
        )
        equilibrium = choose_path_equilibrium(path, current, field, cfg)
        elapsed = time.perf_counter() - t0
        if field.value(current) <= 0:
            return SafetyBuild(
                False,
                field,
                equilibrium,
                "current_outside_obstacle_bas_safe_domain",
                elapsed,
            )
        if equilibrium is None:
            return SafetyBuild(False, field, None, "no_admissible_equilibrium", elapsed)
        return SafetyBuild(True, field, equilibrium, "ok", elapsed)
    if method == "single_scp_bas":
        result, elapsed = build_single_scp(current, local_points, cfg, rho)
        if not result.success:
            return SafetyBuild(False, None, None, result.reason, elapsed, elapsed, 0)
        seeds = np.asarray([current])
        polygons = [result.polygon]
        equilibrium = choose_goal_equilibrium(path, result.field, cfg)
        mode = (
            "goal_equilibrium" if equilibrium is not None else "absolute_barrier_state"
        )
        return SafetyBuild(
            True, result.field, equilibrium, mode, elapsed, elapsed, 0, seeds, polygons
        )
    if method == "corridor_bas":
        result = build_lookahead_corridor(
            current, path, local_points, cfg, rho, poisson_reconstructor
        )
        if not result.success:
            return SafetyBuild(
                False,
                None,
                None,
                result.reason,
                result.computation_time,
                result.scp_time,
                result.fusion_time,
                diagnostics=result.diagnostics,
            )
        equilibrium = choose_goal_equilibrium(path, result.field, cfg)
        mode = (
            "goal_equilibrium" if equilibrium is not None else "absolute_barrier_state"
        )
        return SafetyBuild(
            True,
            result.field,
            equilibrium,
            mode,
            result.computation_time,
            result.scp_time,
            result.fusion_time,
            result.seeds,
            result.polygons,
            result.diagnostics,
        )
    raise ValueError(method)
