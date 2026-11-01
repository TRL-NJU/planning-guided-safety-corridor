
import argparse
import json

import numpy as np

from core.config import load_config, load_run_config, resolve_project_path
from environments.scenario import Scenario


def audit_trial(summary_path, cfg):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scenario = Scenario.load(summary["environment_file"])
    with np.load(summary_path.parent / "timeseries.npz") as data:
        velocity = np.column_stack((data["ux"], data["uy"]))
        target = np.column_stack((data["velocity_target_x"], data["velocity_target_y"]))
        rate = np.column_stack((data["input_rate_x"], data["input_rate_y"]))
        position = np.column_stack((data["px"], data["py"]))
        dt = float(cfg["simulation"]["dt"])
        limit = float(cfg["limits"]["single_speed_max"])
        margin = 1.0 - np.sum(velocity**2, axis=1) / limit**2
        if not len(margin) or not np.all(np.isfinite(margin)) or np.any(margin <= 0):
            raise ValueError(f"Invalid input trajectory: {summary_path}")
        z = 1.0 / margin - 1.0
        gradient = 2.0 * velocity / (limit**2 * margin[:, None] ** 2)
        expected_rate = cfg["input_barrier"]["tracking_gain"] * (target - velocity)
        expected_rate -= cfg["input_barrier"]["barrier_gain"] * z[:, None] * gradient
        previous_velocity = np.vstack((np.zeros(2), velocity[:-1]))
        previous_position = np.vstack((scenario.start, position[:-1]))
        input_error = np.max(np.linalg.norm(rate - expected_rate, axis=1))
        integration_error = np.max(
            np.linalg.norm((velocity - previous_velocity) / dt - rate, axis=1)
        )
        position_error = np.max(
            np.linalg.norm(
                position
                - previous_position
                - 0.5 * dt * (previous_velocity + velocity),
                axis=1,
            )
        )
        if not np.allclose(data["input_barrier_state"], z, rtol=1e-9, atol=1e-9):
            raise ValueError(f"Inconsistent input barrier state: {summary_path}")
        if not np.allclose(data["input_margin"], margin, rtol=1e-9, atol=1e-9):
            raise ValueError(f"Inconsistent input margin: {summary_path}")
        if input_error > 1e-6 or integration_error > 1e-8 or position_error > 1e-9:
            raise ValueError(f"Input dynamics audit failed: {summary_path}")
        return {
            "scenario": summary["scenario"],
            "environment": summary["environment_id"],
            "method": summary["method"],
            "samples": len(margin),
            "maximum_speed": float(np.max(np.linalg.norm(velocity, axis=1))),
            "minimum_input_margin": float(np.min(margin)),
            "maximum_rate_equation_error": float(input_error),
            "maximum_position_integration_error": float(position_error),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root")
    args = parser.parse_args()
    settings = load_run_config()["benchmark"]
    root = resolve_project_path(args.results_root or settings["results_root"])
    cfg = load_config(resolve_project_path(settings["config"]))
    paths = sorted(root.glob("single/*/*/*/summary.json"))
    if not paths:
        raise ValueError("No saved trial summaries found")
    manifest_path = root / "benchmark_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(paths) != manifest["expected_trials"]:
            raise ValueError("The benchmark is incomplete")
    trials = [audit_trial(path, cfg) for path in paths]
    report = {
        "trials": len(trials),
        "samples": sum(row["samples"] for row in trials),
        "maximum_speed": max(row["maximum_speed"] for row in trials),
        "minimum_input_margin": min(row["minimum_input_margin"] for row in trials),
        "maximum_rate_equation_error": max(
            row["maximum_rate_equation_error"] for row in trials
        ),
        "trial_details": trials,
    }
    output = root / "input_constraint_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verified {report['trials']} trials and {report['samples']} input samples")
    print(f"Maximum applied speed: {report['maximum_speed']:.8f} m/s")
    print(f"Minimum input margin: {report['minimum_input_margin']:.8g}")
    print(f"Audit: {output}")


if __name__ == "__main__":
    main()
