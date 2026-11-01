from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from core.config import rho_from_config
from geometry.grid import rasterize_obstacles
from geometry.collision import ClearanceEvaluator
from planning.jps_wrapper import JPSWrapper
from planning.path_utils import PolylinePath
from planning.corridor_guidance import CorridorReferenceGuide
from controllers.navigation_controller import NavigationController
from controllers.input_barrier_controller import InputBarrierController
from dynamics.models import RobotState, make_dynamics
from experiments.methods import build_method_safety, sensed_obstacle_points
from analysis.metrics import forward_safe_path_coverage, safe_region_area
from safety.poisson import PoissonReconstructor
from .termination import StallDetector
from .logger import TrialLogger
from .workspace import workspace_walls, wall_clearance, apply_wall_occupancy


@dataclass
class SimulationResult:
    summary: dict
    logger: TrialLogger
    scenario: object
    method: str
    dynamics: str
    snapshots: list


class Simulator:
    def __init__(self, cfg, scenario, method, dynamics):
        self.cfg = cfg
        self.scenario = scenario
        self.method = method
        self.dynamics_name = dynamics
        self.rho = rho_from_config(cfg)
        ws = cfg["workspace"]
        self.boundary_is_wall = bool(ws.get("boundary_is_wall", False))
        self.walls = (
            workspace_walls(scenario.width, scenario.height)
            if self.boundary_is_wall
            else []
        )
        self.planner = JPSWrapper(
            float(ws["width"]),
            float(ws["height"]),
            float(ws["grid_resolution"]),
            verbose=bool(cfg["planning"]["jps_verbose"]),
            extra_dilation_cells=int(
                cfg["planning"].get("jps_extra_dilation_cells", 1)
            ),
        )
        self.model = make_dynamics(dynamics, cfg)
        self.controller = NavigationController(
            cfg, dynamics, dynamic_scene=bool(scenario.dynamic_specs)
        )
        self.corridor_guide = (
            CorridorReferenceGuide(cfg, dynamics)
            if method == "corridor_bas"
            and bool(cfg.get("corridor_guidance", {}).get("enabled", False))
            else None
        )
        settings = cfg["input_barrier"]
        self.input_controller = InputBarrierController(
            cfg["limits"]["single_speed_max"],
            settings["tracking_gain"],
            settings["barrier_gain"],
        )
        self.last_guidance_offset = 0.0
        self.last_guidance_value = np.nan
        self.last_guidance_used = False
        self.poisson_reconstructor = (
            PoissonReconstructor(cfg)
            if method == "corridor_bas"
            and bool(cfg.get("poisson", {}).get("enabled", False))
            else None
        )
        self.clearance_eval = ClearanceEvaluator(
            scenario.static_obstacles, float(cfg["robot"]["radius"])
        )
        self.logger = TrialLogger()
        self.snapshots = []

    def _wall_clearance(self, point):
        if not self.boundary_is_wall:
            return np.inf
        return wall_clearance(
            point,
            self.scenario.width,
            self.scenario.height,
            float(self.cfg["robot"]["radius"]),
        )

    def _reference(self, path, state, safety_field=None):
        if safety_field is not None and hasattr(safety_field, "reference_hold_point"):
            self.last_guidance_offset = 0.0
            self.last_guidance_value = np.nan
            self.last_guidance_used = False
            return state.p.copy(), np.zeros(2)
        pr, sr = path.point_ahead(
            state.p, float(self.cfg["planning"]["reference_lookahead"])
        )
        if safety_field is not None and hasattr(safety_field, "reference_end_s"):
            sr = min(sr, safety_field.reference_end_s)
            pr = path.point_at(sr)
        self.last_guidance_offset = 0.0
        self.last_guidance_value = np.nan
        self.last_guidance_used = False
        if self.corridor_guide is not None and safety_field is not None:
            guided = self.corridor_guide.reference(
                path, state.p, self.scenario.goal, safety_field
            )
            pr = guided.point
            sr = guided.path_s
            self.last_guidance_offset = guided.lateral_offset
            self.last_guidance_value = guided.field_value
            self.last_guidance_used = guided.used_guidance
        if self.dynamics_name == "single":
            vr = np.zeros(2)
        else:
            goal_delta = np.asarray(self.scenario.goal, float) - state.p
            remaining = float(np.linalg.norm(goal_delta))
            terminal_radius = float(self.cfg["planning"]["reference_lookahead"])
            if remaining <= terminal_radius and remaining > 1.0e-12:
                direction = goal_delta / remaining
            elif self.last_guidance_used:
                # Apply lateral guidance through p_ref only. Keep v_ref
                # tangent to the causal JPS path for double-integrator damping.
                direction = self.corridor_guide._unit_tangent(path, sr)
            else:
                direction = path.tangent_at(sr)
            speed = min(float(self.cfg["planning"]["reference_speed"]), remaining)
            vr = speed * direction
        return pr, vr

    def run(self):
        c = self.cfg
        dt = float(c["simulation"]["dt"])
        dynamic_scene = bool(self.scenario.dynamic_specs)
        jps_dt = float(
            c["planning"]["dt"] if dynamic_scene else c["planning"]["static_dt"]
        )
        tmax = float(c["simulation"]["max_time"])
        goal_r = float(c["robot"]["goal_radius"])
        dsafe = float(c["robot"]["safety_margin"])
        metric_dt = float(c["metrics"]["sample_dt"])
        snapshot_dt = float(c["simulation"]["snapshot_dt"])

        state = RobotState(np.asarray(self.scenario.start, float).copy(), np.zeros(2))
        stall = StallDetector(c)
        path = None
        sbuild = None
        valid = False
        next_jps = 0.0
        next_metric = 0.0
        next_snapshot = 0.0
        last_safety_valid = False

        plen = effort = smooth = 0.0
        lastu = np.zeros(2)
        minclr = minsaf = np.inf
        min_wall_clearance = self._wall_clearance(state.p)
        margin_violate = False
        field_violate = False

        jt = []
        mt = []
        st = []
        sct = []
        ft = []
        ht = []
        ct = []
        covs = []
        areas = []
        guidance_offsets = []
        guidance_values = []
        guidance_used_steps = 0
        reverse_steps = 0
        safe_input_limited_steps = 0
        target_outside_limit_steps = 0
        n_jps_updates = 0
        n_safety_updates = 0

        status = "navigation_failure"
        reason = "timeout"
        t = 0.0
        steps = 0

        if min_wall_clearance <= 0:
            status, reason = "collision", "initial_wall_collision"
            minclr, minsaf = min_wall_clearance, min_wall_clearance - dsafe
            margin_violate = True
            self.logger.event(0.0, "collision", reason=reason)

        while t <= tmax + 1.0e-12 and status != "collision":
            high_start = time.perf_counter()
            dyn = self.scenario.dynamic_obstacles(t)
            allobs = list(self.scenario.static_obstacles) + list(dyn)
            safety_obstacles = allobs + self.walls

            # JPS retains its configured period. A failed refresh does not erase
            # the most recent successful path; the real-time safety update below
            # will truncate unsafe corridor seeds on that retained path.
            jps_updated = bool(t + 1.0e-12 >= next_jps)
            jps_success = True
            path_reused = False
            map_t = 0.0
            jps_t = 0.0
            if jps_updated:
                n_jps_updates += 1
                tm = time.perf_counter()
                # All four methods receive the same causal JPS information:
                # static obstacles and the current dynamic-obstacle positions.
                # No future trajectory or swept-volume prediction is used.
                grid = rasterize_obstacles(
                    allobs,
                    float(c["workspace"]["width"]),
                    float(c["workspace"]["height"]),
                    float(c["workspace"]["grid_resolution"]),
                    self.rho,
                )
                if self.boundary_is_wall:
                    apply_wall_occupancy(
                        grid,
                        self.scenario.width,
                        self.scenario.height,
                        float(c["workspace"]["grid_resolution"]),
                        self.rho,
                    )
                map_t = time.perf_counter() - tm
                pres = self.planner.plan(grid, state.p, self.scenario.goal)
                jps_t = float(pres.computation_time)
                jt.append(jps_t)
                mt.append(map_t)
                jps_success = bool(pres.success)
                if jps_success:
                    path = PolylinePath(pres.path)
                    if self.corridor_guide is not None:
                        self.corridor_guide.on_replan()
                else:
                    path_reused = path is not None
                    self.logger.event(
                        t,
                        "jps_failure_path_reused" if path_reused else "jps_failure",
                    )
                next_jps += jps_dt
                while next_jps <= t + 1.0e-12:
                    next_jps += jps_dt

            # The safety representation has no independent period. It is rebuilt
            # after every completed control step from the latest robot/obstacle
            # state and the latest successful JPS path.
            if path is not None:
                localpts = sensed_obstacle_points(
                    safety_obstacles,
                    state.p,
                    float(c["perception"]["range"]),
                    float(c["perception"]["boundary_spacing"]),
                )
                sbuild = build_method_safety(
                    self.method,
                    state.p,
                    path,
                    safety_obstacles,
                    localpts,
                    c,
                    self.rho,
                    self.poisson_reconstructor,
                )
                valid = bool(sbuild.success)
                n_safety_updates += 1
                st.append(float(sbuild.safety_time))
                sct.append(float(sbuild.scp_time))
                ft.append(float(sbuild.fusion_time))

                if valid and sbuild.field is not None and t + 1.0e-12 >= next_metric:
                    covs.append(
                        forward_safe_path_coverage(
                            path,
                            state.p,
                            sbuild.field,
                            float(c["metrics"]["path_coverage_ds"]),
                            float(c["scp"]["max_lookahead"]),
                        )
                    )
                    if bool(c["metrics"]["safe_area_enabled"]):
                        areas.append(
                            safe_region_area(
                                sbuild.field,
                                state.p,
                                float(c["metrics"]["safe_area_window"]),
                                float(c["metrics"]["safe_area_resolution"]),
                            )
                        )
                    next_metric += metric_dt
                    while next_metric <= t + 1.0e-12:
                        next_metric += metric_dt

                if (
                    bool(c["simulation"]["save_snapshots"])
                    and valid
                    and t + 1.0e-12 >= next_snapshot
                ):
                    self.snapshots.append(
                        {
                            "t": float(t),
                            "path": path.points.copy(),
                            "p": state.p.copy(),
                            "field": sbuild.field,
                            "p_eq": None if sbuild.p_eq is None else sbuild.p_eq.copy(),
                            "seeds": None
                            if sbuild.seeds is None
                            else sbuild.seeds.copy(),
                            "polygons": []
                            if sbuild.polygons is None
                            else [x.copy() for x in sbuild.polygons],
                        }
                    )
                    next_snapshot += snapshot_dt
                    while next_snapshot <= t + 1.0e-12:
                        next_snapshot += snapshot_dt
            else:
                sbuild = None
                valid = False

            if valid != last_safety_valid:
                if valid:
                    self.logger.event(t, "realtime_safety_update_recovered")
                else:
                    self.logger.event(
                        t,
                        "realtime_safety_build_failure",
                        reason="no_jps_path" if sbuild is None else sbuild.reason,
                    )
            last_safety_valid = valid

            total = time.perf_counter() - high_start
            ht.append(total)
            self.logger.log_high_level(
                t=float(t),
                jps_updated=jps_updated,
                jps_success=jps_success,
                path_reused=path_reused,
                path_available=path is not None,
                method_success=valid,
                map_time=float(map_t),
                jps_time=float(jps_t),
                safety_time=0.0 if sbuild is None else float(sbuild.safety_time),
                total_time=float(total),
                n_scp=0
                if sbuild is None or sbuild.seeds is None
                else int(len(sbuild.seeds)),
                reason="no_jps_path" if sbuild is None else sbuild.reason,
                corridor_acceptance={} if sbuild is None else sbuild.diagnostics,
            )

            if valid and path is not None:
                pref, vref = self._reference(
                    path,
                    state,
                    None if sbuild is None else sbuild.field,
                )
                guidance_offsets.append(float(self.last_guidance_offset))
                if np.isfinite(self.last_guidance_value):
                    guidance_values.append(float(self.last_guidance_value))
                guidance_used_steps += int(self.last_guidance_used)
                tc0 = time.perf_counter()
                ce = self.controller.evaluate(
                    self.method,
                    state.p,
                    state.v,
                    pref,
                    vref,
                    None if sbuild is None else sbuild.field,
                    None if sbuild is None else sbuild.p_eq,
                    0.0,
                )
                u = ce.target_velocity
                ct.append(time.perf_counter() - tc0)
                field_violate |= ce.safety_violation
            else:
                pref = state.p.copy()
                vref = np.zeros(2)
                self.last_guidance_offset = 0.0
                self.last_guidance_value = np.nan
                self.last_guidance_used = False
                u = self.model.hold_control(state, c)
                ce = None

            cap = float(c["limits"]["single_speed_max"])
            input_update = self.input_controller.advance(state.v, u, dt)
            ua = input_update.velocity
            reference_delta = pref - state.p
            reference_norm = float(np.linalg.norm(reference_delta))
            reference_velocity = (
                float(ua @ reference_delta / reference_norm)
                if reference_norm > 1.0e-12
                else 0.0
            )
            reverse_steps += int(reference_velocity < -1.0e-6)
            safe_input_limited_steps += int(ce is not None and ce.safe_input_limited)
            target_outside_limit_steps += int(np.linalg.norm(u) > cap)
            old = state.p.copy()
            old_velocity = state.v.copy()
            state = self.model.step(state, ua, dt)
            plen += float(np.linalg.norm(state.p - old))
            effort += float(
                (old_velocity @ old_velocity + old_velocity @ ua + ua @ ua) * dt / 3.0
            )
            smooth += float((ua - lastu) @ (ua - lastu))
            lastu = ua.copy()

            clr = self.clearance_eval.clearance(
                state.p, self.scenario.dynamic_obstacles(t + dt)
            )
            wall_clr = self._wall_clearance(state.p)
            min_wall_clearance = min(min_wall_clearance, wall_clr)
            clr = min(clr, wall_clr)
            minclr = min(minclr, clr)
            sm = clr - dsafe
            minsaf = min(minsaf, sm)
            margin_violate |= sm < 0
            self.logger.log_step(
                t=float(t + dt),
                px=float(state.p[0]),
                py=float(state.p[1]),
                vx=float(state.v[0]),
                vy=float(state.v[1]),
                ux=float(ua[0]),
                uy=float(ua[1]),
                pref_x=float(pref[0]),
                pref_y=float(pref[1]),
                guidance_offset=float(self.last_guidance_offset),
                guidance_field_value=float(self.last_guidance_value),
                guidance_used=int(self.last_guidance_used),
                clearance=float(clr),
                wall_clearance=float(wall_clr),
                signed_safety_margin=float(sm),
                h=np.nan if ce is None else float(ce.h),
                h_eq=np.nan if ce is None else float(ce.h_eq),
                z=np.nan if ce is None else float(ce.z),
                plan_valid=int(valid),
                u_nom_x=0.0 if ce is None else float(ce.u_nom[0]),
                u_nom_y=0.0 if ce is None else float(ce.u_nom[1]),
                u_safe_x=0.0 if ce is None else float(ce.u_safe[0]),
                u_safe_y=0.0 if ce is None else float(ce.u_safe[1]),
                safe_input_raw_norm=0.0 if ce is None else ce.safe_input_raw_norm,
                safe_input_limited=int(ce is not None and ce.safe_input_limited),
                velocity_target_exceeds_limit=int(np.linalg.norm(u) > cap),
                velocity_target_x=float(u[0]),
                velocity_target_y=float(u[1]),
                input_rate_x=float(input_update.rate[0]),
                input_rate_y=float(input_update.rate[1]),
                input_margin=float(input_update.margin),
                input_barrier_state=float(input_update.barrier_state),
                input_barrier_rate=float(input_update.barrier_rate),
                input_barrier_feedback_norm=float(input_update.barrier_feedback_norm),
                input_equation_residual=float(input_update.equation_residual),
                reference_velocity=reference_velocity,
                reverse_motion=int(reference_velocity < -1.0e-6),
            )

            t += dt
            steps += 1
            if clr <= 0:
                status = "collision"
                reason = "wall_collision" if wall_clr <= 0 else "physical_collision"
                self.logger.event(t, "collision")
                break
            if np.linalg.norm(state.p - self.scenario.goal) <= goal_r:
                status = "success"
                reason = ""
                self.logger.event(t, "goal_reached")
                break
            if stall.update(
                t,
                state.p,
                self.scenario.goal,
                float(np.linalg.norm(state.v)),
            ):
                status = "navigation_failure"
                reason = "deadlock"
                self.logger.event(t, "deadlock")
                break

        if t > tmax and status not in ("success", "collision"):
            status = "navigation_failure"
            reason = "timeout"
            self.logger.event(min(t, tmax), "timeout")

        d0 = float(np.linalg.norm(self.scenario.goal - self.scenario.start))

        def avg(x, default=np.nan):
            return float(np.mean(x)) if x else default

        def p95(x, default=np.nan):
            return float(np.percentile(x, 95)) if x else default

        def mx(x, default=np.nan):
            return float(np.max(x)) if x else default

        summary = {
            "scenario": self.scenario.scenario_type,
            "subtype": self.scenario.subtype,
            "environment_id": getattr(self.scenario, "environment_id", ""),
            "seed": int(self.scenario.seed),
            "method": self.method,
            "dynamics": self.dynamics_name,
            "terminal_status": status,
            "failure_reason": reason,
            "success": int(status == "success"),
            "collision": int(status == "collision"),
            "navigation_failure": int(status == "navigation_failure"),
            "deadlock": int(reason == "deadlock"),
            "safety_margin_violation": int(margin_violate),
            "safety_field_violation": int(field_violate),
            "travel_time": float(min(t, tmax)),
            "executed_path_length": float(plen),
            "normalized_path_length": float(plen / max(d0, 1.0e-9)),
            "min_clearance": float(minclr),
            "min_signed_safety_margin": float(minsaf),
            "control_effort": float(effort),
            "control_smoothness": float(smooth),
            "mean_forward_safe_coverage": avg(covs),
            "min_forward_safe_coverage": float(np.min(covs)) if covs else np.nan,
            "mean_safe_region_area": avg(areas),
            "jps_time_mean": avg(jt),
            "jps_time_p95": p95(jt),
            "jps_time_max": mx(jt),
            "map_time_mean": avg(mt),
            "safety_time_mean": avg(st, 0.0),
            "safety_time_p95": p95(st, 0.0),
            "scp_time_mean": avg(sct, 0.0),
            "fusion_time_mean": avg(ft, 0.0),
            "high_level_time_mean": avg(ht),
            "high_level_time_p95": p95(ht),
            "high_level_time_max": mx(ht),
            "controller_time_mean": avg(ct),
            "guidance_offset_mean_abs": avg(
                [abs(value) for value in guidance_offsets], 0.0
            ),
            "guidance_offset_max_abs": mx(
                [abs(value) for value in guidance_offsets], 0.0
            ),
            "guidance_field_value_mean": avg(guidance_values),
            "guidance_usage_rate": float(
                guidance_used_steps / max(len(guidance_offsets), 1)
            ),
            "n_control_steps": steps,
            "n_high_level_updates": len(ht),
            "n_jps_updates": n_jps_updates,
            "n_safety_updates": n_safety_updates,
            "dynamic_full_actuation": bool(self.controller.full_dynamic_actuation),
            "n_reverse_steps": reverse_steps,
            "n_safe_input_limited_steps": safe_input_limited_steps,
            "n_target_outside_limit_steps": target_outside_limit_steps,
            "input_constraint_method": "dynamic_extension_inverse_barrier",
            "minimum_input_margin": min(row["input_margin"] for row in self.logger.rows)
            if self.logger.rows
            else float("nan"),
            "maximum_input_equation_residual": max(
                row["input_equation_residual"] for row in self.logger.rows
            )
            if self.logger.rows
            else float("nan"),
            "single_speed_max": float(c["limits"]["single_speed_max"]),
            "workspace_boundary_is_wall": self.boundary_is_wall,
            "min_wall_clearance": float(min_wall_clearance),
        }
        return SimulationResult(
            summary,
            self.logger,
            self.scenario,
            self.method,
            self.dynamics_name,
            self.snapshots,
        )
