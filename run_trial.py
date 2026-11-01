
import argparse

from core.config import load_run_config, resolve_project_path
from core.constants import METHODS
from experiments.runner import run_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("S1", "S2"), required=True)
    parser.add_argument("--environment", type=int, required=True)
    parser.add_argument("--method", choices=METHODS, default="corridor_bas")
    parser.add_argument(
        "--force", action="store_true", help="Recompute an existing trial"
    )
    args = parser.parse_args()
    options = load_run_config()["benchmark"]
    scenario = resolve_project_path(options["environment_root"]) / args.scenario
    scenario /= f"environment_{args.environment:03d}.json"
    if not scenario.is_file():
        parser.error(f"Environment does not exist: {scenario}")
    summary = run_one(
        resolve_project_path(options["config"]),
        scenario,
        resolve_project_path(options["results_root"]),
        "single",
        args.method,
        resume=not args.force,
    )
    print(f"Status: {summary['terminal_status']}")
    print(f"Reason: {summary.get('failure_reason') or 'goal reached'}")


if __name__ == "__main__":
    main()
