"""Continuous path certificates and numerical corridor connectivity checks."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label


@dataclass
class AcceptanceResult:
    success: bool
    reason: str
    path_lower_bound: float = float("nan")
    grid_components: int = 0


def reference_prefix(current, path, start_s, end_s):
    """Include the current-to-path connection and every intervening vertex."""
    internal = (path.cum > start_s) & (path.cum < end_s)
    return np.vstack(
        (current, path.point_at(start_s), path.points[internal], path.point_at(end_s))
    )


def certify_path(field, vertices, margin, max_depth=14):
    """Certify full segments with batched midpoint bounds.

    Unit facet normals give ||grad H_k|| <= 1. Softmax weights imply
    sum(w_k * |H_k-H|) <= 2*(max(H_k)-H) <= 2*log(N)/kappa.
    Hence ||grad H|| <= 1+2*log(N), globally. The bound is conservative
    but independent of the spread between distant local fields.
    """
    vertices = np.asarray(vertices, float)
    if len(vertices) < 2 or not np.all(np.isfinite(vertices)):
        return False, float("nan")
    left, right = vertices[:-1], vertices[1:]
    lipschitz = 1.0 + 2.0 * np.log(len(field.local_fields))
    lower_bound = float("inf")
    for depth in range(max_depth + 1):
        centers = (left + right) / 2.0
        radii = np.linalg.norm(right - left, axis=1) / 2.0
        values = field.values(centers)
        if not np.all(np.isfinite(values)) or np.any(values < margin):
            return False, float(np.min(values))
        bounds = values - lipschitz * radii
        done = bounds >= margin
        if np.any(done):
            lower_bound = min(lower_bound, float(np.min(bounds[done])))
        if np.all(done):
            return True, lower_bound
        if depth == max_depth:
            return False, float(np.min(bounds))
        a, b, c = left[~done], right[~done], centers[~done]
        left = np.vstack((a, c))
        right = np.vstack((c, b))
    return False, lower_bound


def check_connectivity(field, polygons, current, resolution, max_nodes):
    """Select the positive grid component containing the current position.

    This is a four-neighbour grid test, not a proof of continuum topology.
    The grid is anchored at the current position and covers the full SCP union.
    """
    vertices = np.vstack(polygons)
    low = np.min(vertices, axis=0) - resolution
    high = np.max(vertices, axis=0) + resolution
    origin = current - np.ceil((current - low) / resolution) * resolution
    shape = np.ceil((high - origin) / resolution).astype(int) + 1
    if np.prod(shape) > max_nodes:
        return False, 0, "acceptance_grid_too_large"
    x = origin[0] + resolution * np.arange(shape[0])
    y = origin[1] + resolution * np.arange(shape[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    values = np.empty(len(points))
    for start in range(0, len(points), 2048):
        values[start : start + 2048] = field.values(points[start : start + 2048])
    if not np.all(np.isfinite(values)):
        return False, 0, "nonfinite_corridor_values"
    mask = values.reshape(tuple(shape)) > 0.0
    labels, count = label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    index = np.rint((current - origin) / resolution).astype(int)
    selected = int(labels[tuple(index)])
    if selected == 0:
        return False, int(count), "current_outside_discrete_corridor"
    # Remote islands do not invalidate the active component. Continuous path
    # coverage above places the reference in the current analytic component.
    return True, int(count), "ok"


def accept_corridor(field, polygons, current, reference, settings):
    margin = float(settings["field_margin"])
    if field.value(current) < margin:
        return AcceptanceResult(False, "current_margin_failed")
    covered, bound = certify_path(
        field, reference, margin, int(settings["path_max_depth"])
    )
    if not covered:
        return AcceptanceResult(False, "reference_coverage_failed", bound)
    if settings.get("connectivity_via_poisson", False):
        return AcceptanceResult(True, "ok", bound)
    connected, count, reason = check_connectivity(
        field,
        polygons,
        current,
        float(settings["grid_resolution"]),
        int(settings["max_grid_nodes"]),
    )
    return AcceptanceResult(connected, reason, bound, count)
