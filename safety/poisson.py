from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spsolve


@dataclass
class PoissonBuildResult:
    success: bool
    field: object | None
    reason: str
    computation_time: float
    grid_nodes: int = 0
    iterations: int = 0


class PoissonCorridorField:
    """Bilinearly interpolated Poisson field on one connected corridor grid."""

    fusion_mode = "poisson_reconstruction"

    def __init__(
        self,
        origin,
        resolution,
        h_grid,
        phi_grid=None,
        component_mask=None,
        geometry_field=None,
        field_scale=1.0,
    ):
        self.origin = np.asarray(origin, float).reshape(2)
        self.resolution = float(resolution)
        self.h_grid = np.asarray(h_grid, float)
        self.phi_grid = (
            np.zeros_like(self.h_grid)
            if phi_grid is None
            else np.asarray(phi_grid, float)
        )
        self.component_mask = (
            self.h_grid >= 0.0
            if component_mask is None
            else np.asarray(component_mask, bool)
        )
        self.geometry_field = geometry_field
        self.field_scale = float(field_scale)
        self.local_fields = list(getattr(geometry_field, "local_fields", []))
        self.kappa = float(getattr(geometry_field, "kappa", 1.0))

    def _bilinear(self, array, points, outside):
        points = np.asarray(points, float).reshape(-1, 2)
        nx, ny = array.shape
        fx = (points[:, 0] - self.origin[0]) / self.resolution
        fy = (points[:, 1] - self.origin[1]) / self.resolution
        valid = (fx >= 0.0) & (fy >= 0.0) & (fx <= nx - 1) & (fy <= ny - 1)
        values = np.full(len(points), float(outside), float)
        if not np.any(valid) or nx < 2 or ny < 2:
            return values
        xv = np.clip(fx[valid], 0.0, nx - 1.0)
        yv = np.clip(fy[valid], 0.0, ny - 1.0)
        i0 = np.minimum(np.floor(xv).astype(int), nx - 2)
        j0 = np.minimum(np.floor(yv).astype(int), ny - 2)
        tx = xv - i0
        ty = yv - j0
        a00 = array[i0, j0]
        a10 = array[i0 + 1, j0]
        a01 = array[i0, j0 + 1]
        a11 = array[i0 + 1, j0 + 1]
        values[valid] = (
            (1.0 - tx) * (1.0 - ty) * a00
            + tx * (1.0 - ty) * a10
            + (1.0 - tx) * ty * a01
            + tx * ty * a11
        )
        return values

    def values(self, points):
        return self._bilinear(self.h_grid, points, -self.field_scale)

    def value(self, p):
        return float(self.values(np.asarray(p, float).reshape(1, 2))[0])

    def phi_values(self, points):
        return self._bilinear(self.phi_grid, points, 0.0)

    def value_grad(self, p):
        p = np.asarray(p, float).reshape(2)
        nx, ny = self.h_grid.shape
        fx = (p[0] - self.origin[0]) / self.resolution
        fy = (p[1] - self.origin[1]) / self.resolution
        if nx < 2 or ny < 2 or fx < 0.0 or fy < 0.0 or fx > nx - 1 or fy > ny - 1:
            return -self.field_scale, np.zeros(2)
        i0 = min(int(math.floor(fx)), nx - 2)
        j0 = min(int(math.floor(fy)), ny - 2)
        tx = fx - i0
        ty = fy - j0
        h00 = self.h_grid[i0, j0]
        h10 = self.h_grid[i0 + 1, j0]
        h01 = self.h_grid[i0, j0 + 1]
        h11 = self.h_grid[i0 + 1, j0 + 1]
        value = (
            (1.0 - tx) * (1.0 - ty) * h00
            + tx * (1.0 - ty) * h10
            + (1.0 - tx) * ty * h01
            + tx * ty * h11
        )
        gx = ((1.0 - ty) * (h10 - h00) + ty * (h11 - h01)) / self.resolution
        gy = ((1.0 - tx) * (h01 - h00) + tx * (h11 - h10)) / self.resolution
        return float(value), np.array([gx, gy], float)


class PoissonReconstructor:
    """Stateful real-time Poisson reconstruction with previous-solution warm start."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.previous_field = None

    @staticmethod
    def _axis_grid(low, high, anchor, resolution, padding):
        lower = float(low) - padding
        upper = float(high) + padding
        left_steps = int(math.ceil(max(0.0, float(anchor) - lower) / resolution))
        origin = float(anchor) - left_steps * resolution
        count = int(math.ceil(max(0.0, upper - origin) / resolution)) + 1
        return origin, max(count, 3)

    @staticmethod
    def _path_progress(points, path, current, horizon, sample_ds):
        s0, _, _ = path.project(current)
        s1 = min(float(path.length), float(s0) + float(horizon))
        length = max(s1 - s0, 1.0e-12)
        count = max(2, int(math.ceil(length / sample_ds)) + 1)
        s_values = np.linspace(s0, s1, count)
        samples = path.points_at(s_values)
        output = np.empty(len(points), float)
        chunk = 2048
        for start in range(0, len(points), chunk):
            batch = points[start : start + chunk]
            distance2 = np.sum((batch[:, None, :] - samples[None, :, :]) ** 2, axis=2)
            nearest = np.argmin(distance2, axis=1)
            output[start : start + chunk] = (s_values[nearest] - s0) / length
        return np.clip(output, 0.0, 1.0)

    @staticmethod
    def _matrix(component):
        nodes = np.argwhere(component)
        count = len(nodes)
        index = np.full(component.shape, -1, int)
        index[component] = np.arange(count)
        rows = [np.arange(count)]
        cols = [np.arange(count)]
        data = [np.full(count, 4.0)]
        nx, ny = component.shape
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            xx = nodes[:, 0] + dx
            yy = nodes[:, 1] + dy
            in_bounds = (xx >= 0) & (xx < nx) & (yy >= 0) & (yy < ny)
            neighbour = np.full(count, -1, int)
            neighbour[in_bounds] = index[xx[in_bounds], yy[in_bounds]]
            linked = neighbour >= 0
            rows.append(np.flatnonzero(linked))
            cols.append(neighbour[linked])
            data.append(np.full(int(np.sum(linked)), -1.0))
        matrix = coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(count, count),
        ).tocsr()
        return matrix, nodes

    def reconstruct(self, geometry_field, polygons, current, path):
        started = time.perf_counter()
        pcfg = self.cfg["poisson"]
        resolution = float(pcfg["grid_resolution"])
        padding = float(pcfg["grid_padding"])
        vertices = np.vstack(
            [np.asarray(poly, float) for poly in polygons if len(poly)]
        )
        if not len(vertices):
            return PoissonBuildResult(False, None, "empty_corridor_geometry", 0.0)
        x0, nx = self._axis_grid(
            vertices[:, 0].min(), vertices[:, 0].max(), current[0], resolution, padding
        )
        y0, ny = self._axis_grid(
            vertices[:, 1].min(), vertices[:, 1].max(), current[1], resolution, padding
        )
        if nx * ny > int(pcfg["max_grid_nodes"]):
            return PoissonBuildResult(
                False,
                None,
                "poisson_grid_too_large",
                time.perf_counter() - started,
                nx * ny,
            )
        xs = x0 + resolution * np.arange(nx)
        ys = y0 + resolution * np.arange(ny)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        points = np.c_[xx.ravel(), yy.ravel()]
        # The grid approximates the positive domain of the final
        # two-level smooth corridor: softmin inside every SCP, followed by the
        # compensated softmax across SCPs. Its Dirichlet boundary is therefore
        # the manuscript's closed H_corr=0 contour, not the raw SCP union.
        fused_values = geometry_field.values(points).reshape(nx, ny)
        mask = fused_values > 0.0
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], int)
        labels, component_count = label(mask, structure=structure)
        ci = int(round((float(current[0]) - x0) / resolution))
        cj = int(round((float(current[1]) - y0) / resolution))
        current_h = float(geometry_field.value(current))
        if current_h <= 0.0:
            return PoissonBuildResult(
                False,
                None,
                "current_outside_fused_corridor",
                time.perf_counter() - started,
                nx * ny,
            )
        if not (0 <= ci < nx and 0 <= cj < ny) or labels[ci, cj] == 0:
            return PoissonBuildResult(
                False,
                None,
                "current_outside_discrete_corridor",
                time.perf_counter() - started,
                nx * ny,
            )
        component = labels == labels[ci, cj]
        matrix, nodes = self._matrix(component)
        node_points = np.c_[xs[nodes[:, 0]], ys[nodes[:, 1]]]
        progress = self._path_progress(
            node_points,
            path,
            current,
            float(self.cfg["scp"]["max_lookahead"]),
            float(pcfg["progress_sample_ds"]),
        )
        source = 1.0 - float(pcfg["remote_decay"]) * progress
        rhs = resolution * resolution * source
        x_init = None
        if bool(pcfg["warm_start"]) and self.previous_field is not None:
            x_init = self.previous_field.phi_values(node_points)
        iterations = 0

        def callback(_):
            nonlocal iterations
            iterations += 1

        solution, info = cg(
            matrix,
            rhs,
            x0=x_init,
            rtol=float(pcfg["cg_tolerance"]),
            atol=0.0,
            maxiter=int(pcfg["cg_max_iterations"]),
            callback=callback,
        )
        if info != 0 or not np.all(np.isfinite(solution)):
            if not bool(pcfg["direct_fallback"]):
                return PoissonBuildResult(
                    False,
                    None,
                    f"poisson_cg_failed:{info}",
                    time.perf_counter() - started,
                    len(nodes),
                    iterations,
                )
            solution = spsolve(matrix, rhs)
        solution = np.maximum(np.asarray(solution, float), 0.0)
        maximum = float(np.max(solution)) if len(solution) else 0.0
        epsilon = float(pcfg["normalization_epsilon"])
        if not np.isfinite(maximum) or maximum <= epsilon:
            return PoissonBuildResult(
                False,
                None,
                "degenerate_poisson_solution",
                time.perf_counter() - started,
                len(nodes),
                iterations,
            )
        phi_grid = np.zeros((nx, ny), float)
        phi_grid[nodes[:, 0], nodes[:, 1]] = solution
        scale = float(pcfg["field_scale"])
        outside_distance = distance_transform_edt(~component) * resolution
        h_grid = -np.minimum(scale, float(pcfg["outside_slope"]) * outside_distance)
        if bool(pcfg.get("normalize", True)):
            h_grid[component] = scale * solution / (maximum + epsilon)
        else:
            # Preserve the computed Poisson solution and its physical scale.
            h_grid[component] = solution
        field = PoissonCorridorField(
            np.array([x0, y0]),
            resolution,
            h_grid,
            phi_grid,
            component,
            geometry_field,
            scale,
        )
        field.source_components = int(component_count)
        self.previous_field = field
        return PoissonBuildResult(
            True, field, "ok", time.perf_counter() - started, len(nodes), iterations
        )
