from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np

from environments.scenario import Scenario
from safety.fields import CorridorField, SCPField
from safety.poisson import PoissonCorridorField


PREVIEW_FILENAME = "corridor_preview.json.gz"
PREVIEW_LEVELS = np.linspace(0.0, 1.0, 6)


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _progress_indices(positions):
    if len(positions) <= 1:
        return np.zeros(6, dtype=int)
    cumulative = np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    ]
    if cumulative[-1] <= 1.0e-12:
        return np.linspace(0, len(positions) - 1, 6).round().astype(int)
    targets = PREVIEW_LEVELS * cumulative[-1]
    return np.asarray(
        [int(np.argmin(np.abs(cumulative - target))) for target in targets]
    )


def _snapshot_at_or_before(snapshots, t):
    eligible = [x for x in snapshots if float(x["t"]) <= float(t) + 1.0e-9]
    return eligible[-1] if eligible else snapshots[0]


def _serialize_field(field):
    if isinstance(field, PoissonCorridorField):
        return {
            "type": "poisson_grid",
            "origin": field.origin.tolist(),
            "resolution": float(field.resolution),
            "h_grid": field.h_grid.tolist(),
            "phi_grid": field.phi_grid.tolist(),
            "component_mask": field.component_mask.astype(np.uint8).tolist(),
            "field_scale": float(field.field_scale),
        }
    local_fields = list(getattr(field, "local_fields", [field]))
    return {
        "type": "smooth_union",
        "kappa": float(getattr(field, "kappa", local_fields[0].kappa)),
        "fusion": str(
            getattr(field, "fusion_mode", "adaptive_entropy_weighted_average")
        ),
        "local_fields": [
            {
                "A": np.asarray(local.A, float).tolist(),
                "b": np.asarray(local.b, float).tolist(),
                "kappa": float(local.kappa),
                "seed": None
                if local.seed is None
                else np.asarray(local.seed, float).tolist(),
                "polygon": None
                if local.polygon is None
                else np.asarray(local.polygon, float).tolist(),
            }
            for local in local_fields
        ],
    }


def _deserialize_field(data):
    if data.get("type") == "poisson_grid":
        return PoissonCorridorField(
            np.asarray(data["origin"], float),
            float(data["resolution"]),
            np.asarray(data["h_grid"], float),
            np.asarray(data["phi_grid"], float),
            np.asarray(data["component_mask"], bool),
            field_scale=float(data["field_scale"]),
        )
    local_fields = [
        SCPField(
            np.asarray(item["A"], float),
            np.asarray(item["b"], float),
            float(item["kappa"]),
            None if item["seed"] is None else np.asarray(item["seed"], float),
            None if item["polygon"] is None else np.asarray(item["polygon"], float),
        )
        for item in data["local_fields"]
    ]
    return CorridorField(
        local_fields,
        float(data["kappa"]),
        fusion_mode=str(data.get("fusion", "fixed_compensated_lse")),
    )


def save_corridor_preview_data(out_dir, result):
    """Persist six reproducible corridor frames from the completed trial."""
    if result.method != "corridor_bas" or not result.snapshots:
        return None
    arrays = result.logger.arrays()
    if "t" not in arrays or "px" not in arrays:
        return None

    times = np.r_[0.0, np.asarray(arrays["t"], float)]
    positions = np.vstack(
        (np.asarray(result.scenario.start, float), np.c_[arrays["px"], arrays["py"]])
    )
    indices = _progress_indices(positions)
    frames = []
    for level, index in zip(PREVIEW_LEVELS, indices):
        t = float(times[index])
        snapshot = _snapshot_at_or_before(result.snapshots, t)
        frames.append(
            {
                "progress_level": float(level),
                "trajectory_index": int(index),
                "t": t,
                "current": positions[index].tolist(),
                "path": np.asarray(snapshot["path"], float).tolist(),
                "p_eq": None
                if snapshot.get("p_eq") is None
                else np.asarray(snapshot["p_eq"], float).tolist(),
                "seeds": []
                if snapshot.get("seeds") is None
                else np.asarray(snapshot["seeds"], float).tolist(),
                "polygons": [
                    np.asarray(p, float).tolist() for p in snapshot.get("polygons", [])
                ],
                "field": _serialize_field(snapshot["field"]),
            }
        )

    payload = {
        "schema_version": 2,
        "method": result.method,
        "dynamics": result.dynamics,
        "summary": _json_ready(result.summary),
        "scenario": _json_ready(result.scenario.to_dict()),
        "trajectory": positions.tolist(),
        "times": times.tolist(),
        "frames": frames,
    }
    path = Path(out_dir) / PREVIEW_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, path)
    return path


def load_corridor_preview_data(path):
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if int(payload.get("schema_version", -1)) not in (1, 2):
        raise ValueError(
            f"Unsupported corridor preview schema: {payload.get('schema_version')}"
        )
    payload["scenario"] = Scenario.from_dict(payload["scenario"])
    payload["trajectory"] = np.asarray(payload["trajectory"], float)
    payload["times"] = np.asarray(payload["times"], float)
    for frame in payload["frames"]:
        frame["current"] = np.asarray(frame["current"], float)
        frame["path"] = np.asarray(frame["path"], float)
        frame["p_eq"] = (
            None if frame["p_eq"] is None else np.asarray(frame["p_eq"], float)
        )
        frame["seeds"] = np.asarray(frame["seeds"], float).reshape(-1, 2)
        frame["polygons"] = [np.asarray(p, float) for p in frame["polygons"]]
        frame["field"] = _deserialize_field(frame["field"])
    payload["source_file"] = path
    return payload


def discover_corridor_previews(roots):
    files = []
    for root in roots:
        root = Path(root)
        if root.exists():
            files.extend(root.rglob(PREVIEW_FILENAME))
    return sorted(set(p.resolve() for p in files), key=lambda p: str(p).lower())
