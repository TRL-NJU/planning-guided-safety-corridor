from __future__ import annotations
from dataclasses import dataclass
import time
from dataclasses import field as dataclass_field
from copy import deepcopy
from .acceptance import accept_corridor, reference_prefix
import numpy as np
from .fields import CorridorField
from .scp import build_scp


@dataclass
class CorridorBuildResult:
    success: bool
    field: object | None
    local_fields: list
    seeds: np.ndarray
    polygons: list
    reason: str
    computation_time: float
    scp_time: float
    fusion_time: float
    reference_points: object = None
    diagnostics: dict = dataclass_field(default_factory=dict)


def _segment_clearances(p0, endpoints, obstacle_points):
    """Minimum point-cloud clearance for several p0-to-endpoint segments."""
    p0 = np.asarray(p0, float).reshape(2)
    ends = np.asarray(endpoints, float).reshape(-1, 2)
    pts = np.asarray(obstacle_points, float).reshape(-1, 2)
    if not len(ends):
        return np.empty(0)
    if not len(pts):
        return np.full(len(ends), np.inf)
    v = ends - p0
    den = np.sum(v * v, axis=1)
    rel = pts - p0
    tau = np.divide(
        rel @ v.T,
        den[None, :],
        out=np.zeros((len(pts), len(ends))),
        where=den[None, :] > 1e-14,
    )
    tau = np.clip(tau, 0.0, 1.0).T
    projection = p0 + tau[:, :, None] * v[:, None, :]
    distances = np.linalg.norm(pts[None, :, :] - projection, axis=2)
    return np.min(distances, axis=1)


def _candidate_arc_lengths(s, limit, max_spacing, resolution):
    end = min(float(limit), float(s) + float(max_spacing))
    if end <= s + 1e-12:
        return np.empty(0)
    count = max(0, int(np.floor((end - s) / resolution + 1e-12)))
    values = float(s) + float(resolution) * np.arange(1, count + 1, dtype=float)
    if not len(values) or values[-1] < end - 1e-10:
        values = np.r_[values, end]
    return values


def _feasible_prefix(
    current, path, s_values, previous_field, obstacle_points, rho, margin
):
    candidates = path.points_at(s_values)
    pts = np.asarray(obstacle_points, float).reshape(-1, 2)
    if len(pts):
        seed_clearance = np.min(
            np.linalg.norm(candidates[:, None, :] - pts[None, :, :], axis=2), axis=1
        )
        line_clearance = _segment_clearances(current, candidates, pts)
    else:
        seed_clearance = np.full(len(candidates), np.inf)
        line_clearance = np.full(len(candidates), np.inf)
    inside = previous_field.values(candidates) > float(margin)
    valid = (seed_clearance > rho) & (line_clearance > rho) & inside
    invalid = np.flatnonzero(~valid)
    count = int(invalid[0]) if len(invalid) else len(valid)
    return candidates[:count], np.asarray(s_values, float)[:count]


def _build_local_sequence(
    current, path, obstacle_points, cfg, rho, poisson_reconstructor=None
):
    """Build overlapping local regions; final fusion still needs acceptance.

    Every new seed lies in the strict smooth interior of the preceding SCP and
    is the certified interior seed of its own SCP. The seed is therefore a
    known common interior point, so no post-hoc overlap search is required.
    """
    tt = time.perf_counter()
    sc = cfg["scp"]
    ws = cfg["workspace"]
    margin = float(sc["smooth_current_margin"])
    build_args = (
        obstacle_points,
        rho,
        float(ws["width"]),
        float(ws["height"]),
        float(sc["construction_half_width"]),
        float(sc["softmin_kappa"]),
        int(sc["max_obstacle_facets"]),
        float(sc["candidate_tol"]),
        float(sc["seed_clearance_tol"]),
    )

    t0 = time.perf_counter()
    first = build_scp(
        np.asarray(current, float), *build_args, required_seed_margin=margin
    )
    tscp = time.perf_counter() - t0
    if not first.success:
        return CorridorBuildResult(
            False,
            None,
            [],
            np.empty((0, 2)),
            [],
            "first_scp:" + first.reason,
            time.perf_counter() - tt,
            tscp,
            0.0,
        )

    fields = [first.field]
    polys = [first.polygon]
    seeds = [np.asarray(current, float).copy()]
    s0, _, _ = path.project(current)
    horizon = min(path.length, s0 + float(sc["max_lookahead"]))
    # A closest projection can lie behind a newly observed obstacle. Search
    # forward for a reachable connection inside the first convex smooth SCP.
    search = np.arange(s0, horizon, float(sc["seed_search_resolution"]))
    search = np.unique(np.r_[search, horizon])
    candidates = path.points_at(search)
    reachable = first.field.values(candidates) > margin
    indices = np.flatnonzero(reachable)
    if not len(indices):
        # Keep safety feedback active when the global route is temporarily
        # unreachable. This is a fresh local domain, not a stale corridor.
        result = CorridorBuildResult(
            True,
            None,
            fields,
            np.asarray(seeds),
            polys,
            "local_safety_reference",
            time.perf_counter() - tt,
            tscp,
            0.0,
        )
        result.reference_points = np.vstack((current, current))
        result.diagnostics = {
            "reference_start_s": s0,
            "reference_end_s": s0,
            "local_reference_only": True,
        }
        return result
    s0 = float(search[indices[0]])
    s = s0
    max_spacing = float(sc["max_seed_spacing"])
    resolution = float(sc["seed_search_resolution"])
    min_progress = float(sc["min_seed_progress"])
    max_iterations = int(np.ceil(max(0.0, horizon - s0) / min_progress)) + 1

    for _ in range(max_iterations):
        s_values = _candidate_arc_lengths(s, horizon, max_spacing, resolution)
        if not len(s_values):
            break
        candidates, valid_s = _feasible_prefix(
            seeds[-1], path, s_values, fields[-1], obstacle_points, float(rho), margin
        )
        progress = valid_s - s
        admissible = np.flatnonzero(progress >= min_progress - 1e-12)
        if not len(admissible):
            break

        accepted = None
        for j in admissible[::-1]:
            seed = candidates[int(j)]
            t0 = time.perf_counter()
            result = build_scp(seed, *build_args, required_seed_margin=margin)
            tscp += time.perf_counter() - t0
            if result.success:
                accepted = (seed, float(valid_s[int(j)]), result)
                break
        if accepted is None:
            break

        seed, s, result = accepted
        fields.append(result.field)
        polys.append(result.polygon)
        seeds.append(seed.copy())
        if s >= horizon - 1e-12:
            break

    result = CorridorBuildResult(
        True,
        None,
        fields,
        np.asarray(seeds),
        polys,
        "ok",
        time.perf_counter() - tt,
        tscp,
        0.0,
    )
    result.reference_points = reference_prefix(current, path, s0, s)
    result.diagnostics = {"reference_start_s": s0, "reference_end_s": s}
    return result


def build_lookahead_corridor(
    current, path, obstacle_points, cfg, rho, poisson_reconstructor=None
):
    """Refine seeds first, then shorten the prefix, accepting only checked fields."""
    started = time.perf_counter()
    current = np.asarray(current, float)
    settings = cfg.get(
        "corridor_acceptance",
        {
            "field_margin": 1e-4,
            "max_refinements": 2,
            "path_max_depth": 14,
            "grid_resolution": 0.1,
            "max_grid_nodes": 40000,
        },
    )
    settings = dict(settings)
    settings["connectivity_via_poisson"] = bool(
        cfg.get("poisson", {}).get("enabled", False)
    )
    attempts = []
    scp_time = 0.0
    maximum = int(settings["max_refinements"])
    last_reason = "no_local_region"
    for refinement in range(maximum + 1):
        attempt_cfg = deepcopy(cfg)
        for name in ("max_seed_spacing", "seed_search_resolution", "min_seed_progress"):
            attempt_cfg["scp"][name] = float(cfg["scp"][name]) / (2**refinement)
        result = _build_local_sequence(current, path, obstacle_points, attempt_cfg, rho)
        scp_time += result.scp_time
        if not result.success:
            last_reason = result.reason
            attempts.append({"refinement": refinement, "reason": last_reason})
            continue
        count = len(result.local_fields)
        while count >= 1:
            fields = result.local_fields[:count]
            polygons = result.polygons[:count]
            seeds = result.seeds[:count]
            start_s = result.diagnostics["reference_start_s"]
            end_s = start_s if count == 1 else path.project(seeds[-1])[0]
            local_only = result.diagnostics.get("local_reference_only", False)
            reference = (
                np.vstack((current, current))
                if local_only
                else reference_prefix(current, path, start_s, end_s)
            )
            geometry = CorridorField(fields, float(cfg["scp"]["softmax_kappa"]))
            check = accept_corridor(geometry, polygons, current, reference, settings)
            last_reason = check.reason
            active_field = geometry
            if local_only:
                # A single smooth SCP has a convex positive domain. Use its
                # unnormalized clearance field: normalizing a shrinking local
                # Poisson domain can place the current state near H=1 and
                # suppress the existing barrier feedback while an obstacle closes.
                check.grid_components = 1
            if (
                check.success
                and not local_only
                and cfg.get("poisson", {}).get("enabled", False)
            ):
                if poisson_reconstructor is None:
                    from .poisson import PoissonReconstructor

                    poisson_reconstructor = PoissonReconstructor(cfg)
                poisson = poisson_reconstructor.reconstruct(
                    geometry, polygons, current, path
                )
                last_reason = poisson.reason
                check.success = poisson.success
                active_field = poisson.field
                if poisson.success:
                    check.grid_components = int(active_field.source_components)
            attempts.append(
                {
                    "refinement": refinement,
                    "regions": count,
                    "reason": last_reason,
                    "path_lower_bound": check.path_lower_bound,
                    "grid_components": check.grid_components,
                }
            )
            if check.success:
                active_field.reference_end_s = float(end_s)
                if local_only:
                    active_field.reference_hold_point = current.copy()
                active_field.acceptance_diagnostics = {
                    "reference_mode": "local_safety" if local_only else "forward_path",
                    "refinements": refinement,
                    "removed_regions": len(result.local_fields) - count,
                    "path_lower_bound": check.path_lower_bound,
                    "grid_components": check.grid_components,
                    "connectivity_check": "current_component_four_neighbour_grid",
                    "reference_end_s": float(end_s),
                    "attempts": attempts,
                }
                elapsed = time.perf_counter() - started
                return CorridorBuildResult(
                    True,
                    active_field,
                    fields,
                    seeds,
                    polygons,
                    "ok",
                    elapsed,
                    scp_time,
                    elapsed - scp_time,
                    reference,
                    active_field.acceptance_diagnostics,
                )
            if refinement < maximum:
                break
            count -= 1
    elapsed = time.perf_counter() - started
    return CorridorBuildResult(
        False,
        None,
        [],
        np.empty((0, 2)),
        [],
        last_reason,
        elapsed,
        scp_time,
        elapsed - scp_time,
        None,
        {"attempts": attempts},
    )


def build_single_scp(current, obstacle_points, cfg, rho):
    sc = cfg["scp"]
    ws = cfg["workspace"]
    t0 = time.perf_counter()
    res = build_scp(
        current,
        obstacle_points,
        rho,
        float(ws["width"]),
        float(ws["height"]),
        float(sc["construction_half_width"]),
        float(sc["softmin_kappa"]),
        int(sc["max_obstacle_facets"]),
        float(sc["candidate_tol"]),
        float(sc["seed_clearance_tol"]),
        required_seed_margin=float(sc["smooth_current_margin"]),
    )
    elapsed = time.perf_counter() - t0
    return res, elapsed
