from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safety.barrier import InverseBarrier


def clip_norm(v, max_norm):
    v = np.asarray(v, float)
    norm = np.linalg.norm(v)
    return v * (max_norm / norm) if norm > max_norm > 0 else v


@dataclass
class ControlEvaluation:
    target_velocity: np.ndarray
    u_nom: np.ndarray
    u_safe: np.ndarray
    h: float
    h_eq: float
    z: float
    safety_violation: bool
    safe_input_raw_norm: float = 0.0
    safe_input_limited: bool = False


class NavigationController:
    def __init__(self, cfg, dynamics, *, dynamic_scene=False):
        self.cfg = cfg
        self.dynamics = dynamics
        self.dynamic_scene = bool(dynamic_scene)
        self.full_dynamic_actuation = self.dynamic_scene and bool(
            cfg["controller"].get("dynamic_full_actuation", False)
        )
        self.barrier = InverseBarrier(float(cfg["barrier"]["epsilon"]))

    def nominal(self, p, v, p_ref, v_ref):
        if self.dynamics != "single":
            raise ValueError(f"Unsupported dynamics: {self.dynamics}")

        return float(self.cfg["controller"]["Kp_single"]) * (p_ref - p)

    def evaluate(
        self,
        method,
        p,
        v,
        p_ref,
        v_ref,
        safety_field=None,
        p_eq=None,
        field_time_rate=0.0,
    ):
        u_nom = self.nominal(p, v, p_ref, v_ref)
        if method == "jps" or safety_field is None:
            return ControlEvaluation(
                u_nom.copy(), u_nom, np.zeros(2), np.nan, np.nan, 0.0, False
            )

        h, grad = safety_field.value_grad(p)
        grad = np.asarray(grad, float)
        if p_eq is None:
            baseline = self.cfg["barrier"]["transit_B_act"]
            b_act = float(baseline[method] if isinstance(baseline, dict) else baseline)
            h_eq = 1.0 / b_act
            z = self.barrier.value(h) - b_act
        else:
            h_eq = safety_field.value(p_eq)
            z = self.barrier.state(h, h_eq)

        if bool(self.cfg["controller"].get("positive_z_only", False)):
            z_control = max(z, 0.0)
        else:
            z_control = z
        gain_cfg = self.cfg["controller"]["kB"][method]
        gain = float(
            gain_cfg[self.dynamics] if isinstance(gain_cfg, dict) else gain_cfg
        )
        u_safe = gain * z_control * grad

        # The BaS feedback equation above is unchanged. In moving-obstacle
        # scenes, a separate small cap on u_safe can prevent it from cancelling
        # u_nom, even though the physical actuator permits reverse motion.
        # Dynamic scenes pass this sum to the shared final actuator norm limit
        # in Simulator/SingleIntegrator. Static scenes retain the component cap.
        raw_norm = float(np.linalg.norm(u_safe))
        cap_cfg = self.cfg["controller"]["safe_input_cap"]
        cap = float(cap_cfg[self.dynamics] if isinstance(cap_cfg, dict) else cap_cfg)
        limited = not self.full_dynamic_actuation and raw_norm > cap
        if not self.full_dynamic_actuation:
            u_safe = clip_norm(u_safe, cap)
        return ControlEvaluation(
            u_nom + u_safe,
            u_nom,
            u_safe,
            float(h),
            float(h_eq),
            float(z),
            bool(h <= 0.0),
            raw_norm,
            bool(limited),
        )
