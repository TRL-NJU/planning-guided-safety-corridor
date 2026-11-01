 from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from core.config import load_config, load_run_config, resolve_project_path
from core.constants import METHODS, DYNAMICS
from core.benchmark import discover_environments, benchmark_manifest, write_manifest
from experiments.runner import run_one, collect_summaries
from environments.scenario import Scenario


def main():
    opt = load_run_config()["benchmark"]
    config_path = str(resolve_project_path(opt["config"]))
    environment_root = resolve_project_path(opt["environment_root"])
    results_root = resolve_project_path(opt["results_root"])
    workers = int(opt.get("workers", 1))
    resume = bool(opt.get("resume", True))

    env_files = discover_environments(environment_root)
    manifest = benchmark_manifest(environment_root, config_path)
    if not env_files:
        raise RuntimeError(
            f"No saved environments under {environment_root}.\n"
            "Run: python edit_environment.py"
        )

    # Validate category labels before launching workers.
    scenarios = [Scenario.load(p) for p in env_files]
    cfg = load_config(config_path)
    for scenario in scenarios:
        if scenario.width != float(
            cfg["workspace"]["width"]
        ) or scenario.height != float(cfg["workspace"]["height"]):
            raise ValueError(
                f"Environment workspace differs from benchmark config: {scenario.environment_id}"
            )
    write_manifest(results_root / "benchmark_manifest.json", manifest)
    counts = Counter(sc.scenario_type for sc in scenarios)
    n_env = len(env_files)
    n_trials = n_env * len(DYNAMICS) * len(METHODS)

    print("=" * 78)
    print("EXPERIMENT-I PAIRED BENCHMARK")
    print(f"Saved environments : {n_env}")
    print(f"  S1                : {counts.get('S1', 0)}")
    print(f"  S2                : {counts.get('S2', 0)}")
    print(f"Dynamics            : {len(DYNAMICS)} ({', '.join(DYNAMICS)})")
    print(f"Methods             : {len(METHODS)} ({', '.join(METHODS)})")
    print(
        f"Total trials        : {n_env} × {len(DYNAMICS)} × {len(METHODS)} = {n_trials}"
    )
    print("=" * 78)

    tasks = [(p, d, m) for p in env_files for d in DYNAMICS for m in METHODS]

    if workers <= 1:
        for i, (p, d, m) in enumerate(tasks, 1):
            print(f"[{i}/{n_trials}] {p.parent.name}/{p.name} | {d} | {m}")
            summary = run_one(config_path, str(p), str(results_root), d, m, resume)
            if summary["terminal_status"] != "success":
                print(
                    f"    >>> FAILURE | {m} | "
                    f"{summary.get('failure_reason', 'unknown')}"
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    run_one,
                    config_path,
                    str(p),
                    str(results_root),
                    d,
                    m,
                    resume,
                )
                for p, d, m in tasks
            ]
            for i, future in enumerate(as_completed(futures), 1):
                summary = future.result()
                tag = (
                    "SUCCESS" if summary["terminal_status"] == "success" else "FAILURE"
                )
                print(
                    f"[{i}/{n_trials}] {summary['scenario']} "
                    f"{summary.get('environment_id', '')} "
                    f"{summary['dynamics']} {summary['method']} -> {tag}"
                )

    df = collect_summaries(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    summary_path = results_root / "summary_all.csv"
    df.to_csv(summary_path, index=False)

    print("\nBenchmark finished.")
    print(f"Environment count : {n_env}")
    print(f"Expected trials   : {n_trials}")
    print(f"Results           : {results_root}")
    print(f"Summary           : {summary_path}")

    from export_benchmark_table import generate_tables

    _, table_outputs = generate_tables(results_root, manifest["environments"])
    for dynamics, csv_path, png_path in table_outputs:
        print(f"{dynamics:6s} table CSV: {csv_path}")
        print(f"{dynamics:6s} table PNG: {png_path}")



if __name__ == "__main__":
    main()
