from __future__ import annotations
import numpy as np


class PolylinePath:
    def __init__(self, points):
        pts = np.asarray(points, float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("points must be (N,2), N>=2")
        self.points = pts
        self.seg = pts[1:] - pts[:-1]
        self.seg_len = np.linalg.norm(self.seg, axis=1)
        self.cum = np.r_[0.0, np.cumsum(self.seg_len)]
        self.length = float(self.cum[-1])
        self._projection_hint = None

    def _project_range(self, p, lo, hi):
        """Project onto [lo, hi) segments in one vectorised operation."""
        indices = np.arange(int(lo), int(hi), dtype=int)
        valid = self.seg_len[indices] > 1e-12
        indices = indices[valid]
        if not len(indices):
            return np.inf, 0.0, self.points[0].copy(), 0
        a = self.points[indices]
        d = self.seg[indices]
        length = self.seg_len[indices]
        tau = np.clip(np.sum((p - a) * d, axis=1) / (length * length), 0.0, 1.0)
        q = a + tau[:, None] * d
        dist2 = np.sum((q - p) ** 2, axis=1)
        j = int(np.argmin(dist2))
        i = int(indices[j])
        return float(dist2[j]), float(self.cum[i] + tau[j] * length[j]), q[j].copy(), i

    def project(self, p):
        p = np.asarray(p, float).reshape(2)
        nseg = len(self.seg_len)
        if nseg == 0:
            return 0.0, self.points[0].copy(), 0
        if self._projection_hint is None:
            best = self._project_range(p, 0, nseg)
        else:
            hint = int(np.clip(self._projection_hint, 0, nseg - 1))
            center = float(self.cum[hint])
            lo = max(0, int(np.searchsorted(self.cum, center - 1.0, side="right") - 1))
            hi = min(nseg, int(np.searchsorted(self.cum, center + 6.0, side="right")))
            hi = max(lo + 1, hi)
            best = self._project_range(p, lo, hi)
            at_open_boundary = (best[3] == lo and lo > 0) or (
                best[3] == hi - 1 and hi < nseg
            )
            if best[0] > 4.0 or at_open_boundary:
                best = self._project_range(p, 0, nseg)
        self._projection_hint = int(best[3])
        return best[1], best[2], best[3]

    def point_at(self, s):
        return self.points_at([s])[0]

    def points_at(self, s_values):
        """Vectorised counterpart of point_at for an array of arc lengths."""
        s = np.clip(np.asarray(s_values, float).reshape(-1), 0.0, self.length)
        if not len(s):
            return np.empty((0, 2))
        indices = np.searchsorted(self.cum, s, side="right") - 1
        indices = np.clip(indices, 0, len(self.seg_len) - 1)
        length = self.seg_len[indices]
        tau = np.divide(
            s - self.cum[indices], length, out=np.zeros_like(s), where=length > 1e-12
        )
        return self.points[indices] + tau[:, None] * self.seg[indices]

    def tangent_at(self, s):
        s = float(np.clip(s, 0, self.length))
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(max(i, 0), len(self.seg_len) - 1)
        d = self.seg[i]
        n = np.linalg.norm(d)
        return d / n if n > 1e-12 else np.array([1.0, 0.0])

    def point_ahead(self, p, distance):
        s0, _, _ = self.project(p)
        s = min(self.length, s0 + float(distance))
        return self.point_at(s), s

    def sample_forward(self, p, spacing, max_lookahead, include_current=True):
        s0, _, _ = self.project(p)
        spacing = float(spacing)
        max_lookahead = float(max_lookahead)
        count = max(0, int(np.floor(max_lookahead / spacing + 1e-12)))
        ss = s0 + spacing * np.arange(1, count + 1, dtype=float)
        ss = ss[ss <= self.length + 1e-12]
        if s0 < self.length and self.length - s0 <= max_lookahead + 1e-12:
            if not len(ss) or ss[-1] < self.length - 1e-8:
                ss = np.r_[ss, self.length]
        pts = self.points_at(ss)
        if include_current:
            pts = np.vstack((np.asarray(p, float).reshape(1, 2), pts))
            ss = np.r_[s0, ss]
        return [q.copy() for q in pts], [float(x) for x in ss]
