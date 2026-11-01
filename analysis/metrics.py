from __future__ import annotations
import numpy as np


def _field_values(field, points):
    points = np.asarray(points, float).reshape(-1, 2)
    if hasattr(field, "values"):
        return np.asarray(field.values(points), float)
    return np.asarray([field.value(point) for point in points], float)


def forward_safe_path_coverage(path, current, field, ds, max_horizon):
    if field is None:
        return np.nan
    s0, _, _ = path.project(current)
    smax = min(path.length, s0 + float(max_horizon))
    count = max(0, int(np.floor((smax - s0) / float(ds) + 1e-12)))
    samples = s0 + float(ds) * np.arange(count + 1, dtype=float)
    if not len(samples) or samples[-1] < smax - 1e-10:
        samples = np.r_[samples, smax]
    safe = _field_values(field, path.points_at(samples)) > 0.0
    unsafe = np.flatnonzero(~safe)
    last = int(unsafe[0]) - 1 if len(unsafe) else len(samples) - 1
    return 0.0 if last < 0 else float(samples[last] - s0)


def safe_region_area(field, center, window, resolution):
    if field is None:
        return np.nan
    half = float(window) / 2.0
    center = np.asarray(center, float)
    xs = np.arange(center[0] - half, center[0] + half + resolution, resolution)
    ys = np.arange(center[1] - half, center[1] + half + resolution, resolution)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    points = np.c_[X.ravel(), Y.ravel()]
    return float(
        np.count_nonzero(_field_values(field, points) >= 0.0) * resolution * resolution
    )
