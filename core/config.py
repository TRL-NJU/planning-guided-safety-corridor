from __future__ import annotations
from pathlib import Path
import math
import warnings
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    validate_config(cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def rho_from_config(cfg: dict) -> float:
    return float(cfg["robot"]["radius"]) + float(cfg["robot"]["safety_margin"])


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg["workspace"].get("boundary_is_wall", False), bool):
        raise ValueError("workspace.boundary_is_wall must be a boolean.")
    dt = float(cfg["simulation"]["dt"])
    planning_dt = float(cfg["planning"]["dt"])
    if dt <= 0 or planning_dt <= 0:
        raise ValueError("simulation.dt and planning.dt must be positive.")
    if planning_dt < dt:
        raise ValueError("planning.dt must be >= simulation.dt.")
    static_dt = float(cfg["planning"]["static_dt"])
    metric_dt = float(cfg["metrics"]["sample_dt"])
    snapshot_dt = float(cfg["simulation"]["snapshot_dt"])
    if static_dt < dt:
        raise ValueError("planning.static_dt must be >= simulation.dt.")
    if metric_dt <= 0 or snapshot_dt <= 0:
        raise ValueError(
            "metrics.sample_dt and simulation.snapshot_dt must be positive."
        )
    if cfg["robot"]["radius"] <= 0:
        raise ValueError("robot.radius must be positive.")
    if cfg["robot"]["safety_margin"] < 0:
        raise ValueError("robot.safety_margin must be non-negative.")
    transit = cfg["barrier"]["transit_B_act"]
    transit_values = transit.values() if isinstance(transit, dict) else (transit,)
    if any(float(value) <= 0.0 for value in transit_values):
        raise ValueError("barrier.transit_B_act values must be positive.")
    if isinstance(transit, dict) and not {
        "jps_bas",
        "single_scp_bas",
        "corridor_bas",
    } <= set(transit):
        raise ValueError("barrier.transit_B_act must define all safety methods.")

    scp = cfg["scp"]
    softmin_kappa = float(scp["softmin_kappa"])
    softmax_kappa = float(scp["softmax_kappa"])
    max_spacing = float(scp["max_seed_spacing"])
    search_resolution = float(scp["seed_search_resolution"])
    min_progress = float(scp["min_seed_progress"])
    max_lookahead = float(scp["max_lookahead"])
    if softmin_kappa <= 0 or softmax_kappa <= 0:
        raise ValueError("scp.softmin_kappa and scp.softmax_kappa must be positive.")
    if max_spacing <= 0:
        raise ValueError("scp.max_seed_spacing must be positive.")
    if search_resolution <= 0 or search_resolution > max_spacing:
        raise ValueError(
            "scp.seed_search_resolution must lie in (0, max_seed_spacing]."
        )
    if min_progress <= 0 or min_progress > max_spacing:
        raise ValueError("scp.min_seed_progress must lie in (0, max_seed_spacing].")
    if max_lookahead <= 0:
        raise ValueError("scp.max_lookahead must be positive.")

    acceptance = cfg.get("corridor_acceptance", {})
    for name in ("field_margin", "grid_resolution"):
        value = float(acceptance.get(name, 1e-4 if name == "field_margin" else 0.1))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"corridor_acceptance.{name} must be finite and positive.")
    for name, default in (
        ("max_refinements", 2),
        ("path_max_depth", 14),
        ("max_grid_nodes", 40000),
    ):
        value = acceptance.get(name, default)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < (0 if name == "max_refinements" else 1)
        ):
            raise ValueError(
                f"corridor_acceptance.{name} must be an integer in its valid range."
            )

    guidance = cfg.get("corridor_guidance", {})
    if bool(guidance.get("enabled", False)):
        base = {
            key: value
            for key, value in guidance.items()
            if key
            not in {
                "single",
            }
        }
        for dynamics in ("single",):
            current = dict(base)
            current.update(guidance.get(dynamics, {}) or {})
            prefix = f"corridor_guidance.{dynamics}"
            for name in (
                "lookahead",
                "lateral_range",
                "sample_spacing",
                "field_weight",
                "max_offset_change",
                "terminal_disable_distance",
            ):
                if float(current[name]) <= 0.0:
                    raise ValueError(f"{prefix}.{name} must be positive.")
            for name in (
                "minimum_field_value",
                "lateral_penalty",
                "smoothness_penalty",
            ):
                if float(current[name]) < 0.0:
                    raise ValueError(f"{prefix}.{name} must be non-negative.")
            if float(current["sample_spacing"]) > 2.0 * float(current["lateral_range"]):
                raise ValueError(f"{prefix}.sample_spacing is too large.")
            if float(current["max_offset_change"]) < float(current["sample_spacing"]):
                raise ValueError(
                    f"{prefix}.max_offset_change must be >= sample_spacing."
                )
            if float(current["lookahead"]) > max_lookahead:
                raise ValueError(f"{prefix}.lookahead must be <= scp.max_lookahead.")
    poisson = cfg.get("poisson", {})
    if bool(poisson.get("enabled", False)):
        for name in (
            "grid_resolution",
            "grid_padding",
            "field_scale",
            "normalization_epsilon",
            "progress_sample_ds",
            "cg_tolerance",
            "outside_slope",
        ):
            if float(poisson[name]) <= 0.0:
                raise ValueError(f"poisson.{name} must be positive.")
        decay = float(poisson["remote_decay"])
        if not 0.0 <= decay < 1.0:
            raise ValueError("poisson.remote_decay must lie in [0,1).")
        if int(poisson["cg_max_iterations"]) <= 0:
            raise ValueError("poisson.cg_max_iterations must be positive.")
        if int(poisson["max_grid_nodes"]) < 100:
            raise ValueError("poisson.max_grid_nodes must be >=100.")
        if float(poisson["grid_resolution"]) > float(scp["construction_half_width"]):
            raise ValueError(
                "poisson.grid_resolution is too coarse for the SCP domain."
            )

    extra_dilation = int(cfg["planning"]["jps_extra_dilation_cells"])
    if extra_dilation < 0:
        raise ValueError("planning.jps_extra_dilation_cells must be non-negative.")

    limits = cfg["limits"]
    for name in ("single_speed_max",):
        if not math.isfinite(float(limits[name])) or float(limits[name]) <= 0.0:
            raise ValueError(f"limits.{name} must be positive.")

    for name in ("tracking_gain", "barrier_gain"):
        value = float(cfg["input_barrier"][name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"input_barrier.{name} must be positive and finite.")

    safe_cap = cfg["controller"]["safe_input_cap"]
    full_actuation = cfg["controller"].get("dynamic_full_actuation", False)
    if not isinstance(full_actuation, bool):
        raise ValueError("controller.dynamic_full_actuation must be a boolean.")
    for dynamics in ("single", "double"):
        value = safe_cap[dynamics] if isinstance(safe_cap, dict) else safe_cap
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"controller.safe_input_cap.{dynamics} must be positive.")

    half_width = float(scp["construction_half_width"])
    sensing_range = float(cfg["perception"]["range"])
    required = max_lookahead + math.sqrt(2.0) * half_width
    if required > sensing_range:
        warnings.warn(
            "L_max + sqrt(2)*R_D exceeds sensing range; a future seed's square "
            "construction domain may extend outside the sensed region.",
            RuntimeWarning,
        )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_run_config(path: str | Path | None = None) -> dict:
    path = project_root() / "config" / "run.yaml" if path is None else Path(path)
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else project_root() / path
