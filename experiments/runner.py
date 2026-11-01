from __future__ import annotations

from pathlib import Path
import json
import time

import pandas as pd

from core.config import load_config
from core.benchmark import trial_provenance
from environments.scenario import Scenario
from simulation.simulator import Simulator
from visualization.corridor_preview_data import (
    PREVIEW_FILENAME,
    save_corridor_preview_data,
)


def trial_output_dir(root, scenario, dynamics, method):
    env_id = scenario.environment_id or f"seed_{scenario.seed:06d}"
    return Path(root) / dynamics / scenario.scenario_type / method / env_id


def _read_complete_summary(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def run_one(config_path, scenario_path, results_root, dynamics, method, resume=True):
    cfg = load_config(config_path)
    scenario = Scenario.load(scenario_path)
    out = trial_output_dir(results_root, scenario, dynamics, method)
    summary_path = out / "summary.json"
    preview_path = out / PREVIEW_FILENAME
    provenance = trial_provenance(config_path, scenario_path)
    if (
        resume
        and summary_path.exists()
        and (method != "corridor_bas" or preview_path.exists())
        and (out / "events.json").exists()
        and (out / "high_level.json").exists()
        and (
            not cfg["simulation"]["save_timeseries"]
            or (out / "timeseries.npz").exists()
        )
    ):
        existing = _read_complete_summary(summary_path)
        if (
            existing is not None
            and existing.get("provenance") == provenance
            and existing.get("scenario") == scenario.scenario_type
            and existing.get("environment_id") == scenario.environment_id
            and existing.get("method") == method
            and existing.get("dynamics") == dynamics
        ):
            return existing

    if method == "corridor_bas":
        cfg["simulation"]["save_snapshots"] = True
    trial_start = time.perf_counter()
    result = Simulator(cfg, scenario, method, dynamics).run()
    result.summary["wall_time_seconds"] = float(time.perf_counter() - trial_start)
    result.summary["provenance"] = provenance
    result.summary["environment_id"] = scenario.environment_id
    result.summary["environment_file"] = str(Path(scenario_path).resolve())
    if method == "corridor_bas":
        result.summary["preview_data_file"] = str(preview_path.resolve())
        if save_corridor_preview_data(out, result) is None:
            result.summary.pop("preview_data_file", None)
    result.logger.save(out, result.summary, bool(cfg["simulation"]["save_timeseries"]))
    return result.summary


def collect_summaries(results_root):
    rows = []
    for path in Path(results_root).rglob("summary.json"):
        summary = _read_complete_summary(path)
        if summary is None:
            raise ValueError(f"Incomplete or invalid summary file: {path}")
        rows.append(summary)
    return pd.DataFrame(rows)
