from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GuidedReference:
    point: np.ndarray
    path_s: float
    tangent: np.ndarray
    lateral_offset: float
    field_value: float
    used_guidance: bool


class CorridorReferenceGuide:
    """Shift the current JPS reference using only the current Poisson field."""

    def __init__(self, cfg, dynamics="single"):
        base = dict(cfg["corridor_guidance"])
        override = base.pop(dynamics, {})
        base.pop("single", None)
        base.pop("double", None)
        if override:
            base.update(override)
        self.cfg = base
        self.previous_offset = 0.0

    def reset(self) -> None:
        self.previous_offset = 0.0

    def on_replan(self) -> None:
        self.reset()

    @staticmethod
    def _unit_tangent(path, path_s):
        tangent = np.asarray(path.tangent_at(path_s), float).reshape(2)
        norm = float(np.linalg.norm(tangent))
        return tangent / norm if norm > 1.0e-12 else np.array([1.0, 0.0])

    def reference(self, path, current, goal, field) -> GuidedReference:
        current = np.asarray(current, float).reshape(2)
        goal = np.asarray(goal, float).reshape(2)
        lookahead = float(self.cfg["lookahead"])
        base, path_s = path.point_ahead(current, lookahead)
        if field is not None and hasattr(field, "reference_end_s"):
            path_s = min(path_s, field.reference_end_s)
            base = path.point_at(path_s)
        base = np.asarray(base, float).reshape(2)
        tangent = self._unit_tangent(path, path_s)

        if (
            not bool(self.cfg.get("enabled", True))
            or field is None
            or np.linalg.norm(goal - current)
            <= float(self.cfg["terminal_disable_distance"])
        ):
            self.previous_offset = 0.0
            return GuidedReference(base, path_s, tangent, 0.0, np.nan, False)

        normal = np.array([-tangent[1], tangent[0]], float)
        lateral_range = float(self.cfg["lateral_range"])
        spacing = float(self.cfg["sample_spacing"])
        offsets = np.arange(
            -lateral_range,
            lateral_range + 0.5 * spacing,
            spacing,
            dtype=float,
        )
        offsets = np.clip(offsets, -lateral_range, lateral_range)
        offsets = np.unique(np.r_[offsets, 0.0])

        maximum_change = float(self.cfg["max_offset_change"])
        reachable = np.abs(offsets - self.previous_offset) <= maximum_change + 1.0e-12
        candidates = base[None, :] + offsets[:, None] * normal[None, :]
        values = np.asarray(field.values(candidates), float)
        valid = (
            reachable
            & np.isfinite(values)
            & (values > float(self.cfg["minimum_field_value"]))
        )
        if not np.any(valid):
            self.previous_offset = 0.0
            return GuidedReference(base, path_s, tangent, 0.0, np.nan, False)

        valid_values = values[valid]
        scale = max(float(np.max(valid_values)), 1.0e-12)
        normalized_field = values / scale
        normalized_offset = offsets / max(lateral_range, 1.0e-12)
        normalized_change = (offsets - self.previous_offset) / max(
            maximum_change, spacing
        )
        score = (
            float(self.cfg["field_weight"]) * normalized_field
            - float(self.cfg["lateral_penalty"]) * normalized_offset**2
            - float(self.cfg["smoothness_penalty"]) * normalized_change**2
        )
        score[~valid] = -np.inf
        index = int(np.argmax(score))
        selected_offset = float(offsets[index])
        self.previous_offset = selected_offset
        return GuidedReference(
            candidates[index].copy(),
            path_s,
            tangent,
            selected_offset,
            float(values[index]),
            True,
        )
