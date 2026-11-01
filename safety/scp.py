from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .fields import SCPField


@dataclass
class SCPResult:
    success: bool
    field: SCPField | None
    polygon: np.ndarray | None
    reason: str
    obstacle_facets: int


def _domain(seed, half_width, width, height):
    sx, sy = map(float, seed)
    xmin = max(0.0, sx - half_width)
    xmax = min(width, sx + half_width)
    ymin = max(0.0, sy - half_width)
    ymax = min(height, sy + half_width)
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float)
    b = np.array([-xmin, xmax, -ymin, ymax], float)
    poly = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], float)
    return A, b, poly, (xmin, xmax, ymin, ymax)


def _clip(poly, a, b, tol=1e-10):
    if poly is None or len(poly) == 0:
        return np.empty((0, 2))
    out = []
    for i in range(len(poly)):
        P = poly[i]
        Q = poly[(i + 1) % len(poly)]
        hp = float(a @ P + b)
        hq = float(a @ Q + b)
        ip = hp >= -tol
        iq = hq >= -tol
        if ip and iq:
            out.append(Q)
        elif ip and not iq:
            den = hp - hq
            if abs(den) > 1e-14:
                out.append(P + (hp / den) * (Q - P))
        elif (not ip) and iq:
            den = hp - hq
            if abs(den) > 1e-14:
                out.append(P + (hp / den) * (Q - P))
            out.append(Q)
    return np.asarray(out, float) if out else np.empty((0, 2))


def build_scp(
    seed,
    obstacle_points,
    rho,
    width,
    height,
    half_width,
    kappa,
    max_obstacle_facets=64,
    candidate_tol=1e-10,
    seed_clearance_tol=1e-5,
    required_seed_margin=None,
):
    seed = np.asarray(seed, float).reshape(2)
    pts = (
        np.asarray(obstacle_points, float).reshape(-1, 2)
        if len(obstacle_points)
        else np.empty((0, 2))
    )
    domain_A, domain_b, poly, bounds = _domain(seed, half_width, width, height)
    xmin, xmax, ymin, ymax = bounds
    if not (xmin < seed[0] < xmax and ymin < seed[1] < ymax):
        return SCPResult(False, None, None, "seed_not_in_domain_interior", 0)
    if len(pts):
        # Include exterior points whose distance to the domain is at most rho.
        dx = np.maximum(np.maximum(xmin - pts[:, 0], pts[:, 0] - xmax), 0.0)
        dy = np.maximum(np.maximum(ymin - pts[:, 1], pts[:, 1] - ymax), 0.0)
        mask = np.hypot(dx, dy) <= float(rho)
        candidates = pts[mask].copy()
    else:
        candidates = np.empty((0, 2))
    obstacle_A = []
    obstacle_b = []
    facet_count = 0
    while len(candidates):
        distance = np.linalg.norm(candidates - seed, axis=1)
        index = int(np.argmin(distance))
        obstacle = candidates[index]
        nearest = float(distance[index])
        if nearest <= rho + seed_clearance_tol:
            return SCPResult(False, None, None, "seed_clearance_violation", facet_count)
        a = (seed - obstacle) / nearest
        b = -float(a @ obstacle) - float(rho)
        if float(a @ seed + b) <= 0:
            return SCPResult(
                False, None, None, "invalid_separator_orientation", facet_count
            )
        obstacle_A.append(a.copy())
        obstacle_b.append(b)
        facet_count += 1
        poly = _clip(poly, a, b)
        if len(poly) < 3:
            return SCPResult(False, None, None, "empty_polytope", facet_count)
        candidates = candidates[(candidates - obstacle) @ a > candidate_tol]
        if facet_count >= max_obstacle_facets and len(candidates):
            return SCPResult(False, None, None, "max_facets_reached", facet_count)
    if obstacle_A:
        A = np.vstack((np.asarray(obstacle_A), domain_A))
        b = np.r_[np.asarray(obstacle_b), domain_b]
    else:
        A, b = domain_A, domain_b
    raw_seed_margin = A @ seed + b
    if np.min(raw_seed_margin) <= 0:
        return SCPResult(
            False, None, None, "seed_not_strictly_inside_final_scp", facet_count
        )

    effective_kappa = float(kappa)
    required = None if required_seed_margin is None else float(required_seed_margin)
    if required is not None and np.min(raw_seed_margin) <= required:
        return SCPResult(
            False,
            None,
            None,
            "seed_raw_margin_below_required_smooth_margin",
            facet_count,
        )
    field = SCPField(A, b, effective_kappa, seed, poly)
    if required is not None:
        for _ in range(20):
            if field.value(seed) > required:
                break
            effective_kappa *= 2.0
            field = SCPField(A, b, effective_kappa, seed, poly)
        else:
            return SCPResult(
                False, None, None, "adaptive_kappa_failed_to_include_seed", facet_count
            )
    return SCPResult(True, field, poly, "ok", facet_count)
