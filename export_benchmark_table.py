from __future__ import annotations

from pathlib import Path
import json
import math

import pandas as pd
from core.config import load_run_config, resolve_project_path
from core.benchmark import environment_inventory
from core.constants import METHODS, DYNAMICS, BENCHMARK_SCENARIOS

SCENARIOS = BENCHMARK_SCENARIOS

SCENARIO_NAMES = {
    "S1": "S1 - Static",
    "S2": "S2 - Dynamic",
    "Overall": "Overall",
}
DYNAMICS_NAMES = {"single": "Velocity-controlled robot"}
METHOD_NAMES = {
    "jps": "JPS",
    "jps_bas": "JPS-BaS",
    "single_scp_bas": "JPS-Single-SCP-BaS",
    "corridor_bas": "JPS-Poisson-Corridor-BaS<br/>(Ours)",
}


def _results_root() -> Path:
    return resolve_project_path(load_run_config()["benchmark"]["results_root"])


def collect_summaries(root: Path) -> pd.DataFrame:
    return pd.DataFrame(
        json.loads(path.read_text(encoding="utf-8"))
        for path in root.rglob("summary.json")
    )


def validate_trials(df: pd.DataFrame, expected_environments=None) -> None:
    required = {
        "scenario",
        "environment_id",
        "dynamics",
        "method",
        "success",
        "collision",
        "navigation_failure",
        "wall_time_seconds",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns in trial summaries: {missing}")
    if expected_environments is None:
        expected_environments = environment_inventory(
            resolve_project_path(load_run_config()["benchmark"]["environment_root"])
        )
    if set(expected_environments) != set(SCENARIOS):
        raise ValueError("The paired benchmark must contain S1 and S2.")
    for scenario, ids in expected_environments.items():
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicated environment inventory: {scenario}")
    key = ["scenario", "environment_id", "dynamics", "method"]
    if df.duplicated(key, keep=False).any():
        raise ValueError("Duplicate paired trials detected.")
    expected = {
        (scenario, env, dynamics, method)
        for scenario, ids in expected_environments.items()
        for env in ids
        for dynamics in DYNAMICS
        for method in METHODS
    }
    actual = set(df[key].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(
            f"Paired trial inventory mismatch: expected {len(expected)}, found {len(df)}; "
            f"missing={sorted(expected - actual)[:8]}, unexpected={sorted(actual - expected)[:8]}"
        )
    for column in ("success", "collision", "navigation_failure"):
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"Invalid binary outcome: {column}")
    if not (df[["success", "collision", "navigation_failure"]].sum(axis=1) == 1).all():
        raise ValueError("Each trial must have exactly one terminal outcome.")
    if (
        df["wall_time_seconds"].isna().any()
        or not df["wall_time_seconds"]
        .map(lambda value: math.isfinite(float(value)) and float(value) >= 0)
        .all()
    ):
        raise ValueError("Run times must be finite and nonnegative.")


def benchmark_subtitle(summary, dynamics):
    block = summary[
        (summary["dynamics"] == dynamics) & summary["scenario"].isin(SCENARIOS)
    ]
    counts = {}
    for scenario in SCENARIOS:
        values = block.loc[block["scenario"] == scenario, "n"].unique()
        if len(values) != 1:
            raise ValueError(f"Inconsistent paired counts in {scenario}/{dynamics}")
        counts[scenario] = int(values[0])
    total = sum(counts.values())
    detail = ", ".join(f"{scenario}: {count}" for scenario, count in counts.items())
    return (
        f"{total} environments ({detail}), {len(METHODS)} methods, "
        f"and {total * len(METHODS)} paired trials."
    )


def aggregate_trials(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    successful = work["success"] == 1
    work["runtime_excluded"] = False
    work["successful_wall_time_s"] = work["wall_time_seconds"].where(successful)
    work["successful_navigation_time_s"] = work["travel_time"].where(successful)
    work["safety_update_ms"] = 1000.0 * work["safety_time_mean"]
    aggregations = dict(
        n=("success", "size"),
        success_rate=("success", "mean"),
        successful_trials=("success", "sum"),
        mean_wall_time_s=("successful_wall_time_s", "mean"),
        runtime_excluded=("runtime_excluded", "sum"),
        mean_navigation_time_s=("successful_navigation_time_s", "mean"),
        mean_safety_update_ms=("safety_update_ms", "mean"),
        collision_rate=("collision", "mean"),
        navigation_failure_rate=("navigation_failure", "mean"),
    )
    grouped = (
        work.groupby(["scenario", "dynamics", "method"], observed=False)
        .agg(**aggregations)
        .reset_index()
    )
    overall = (
        work.groupby(["dynamics", "method"], observed=False)
        .agg(**aggregations)
        .reset_index()
    )
    overall.insert(0, "scenario", "Overall")
    out = pd.concat([grouped, overall], ignore_index=True)
    for column in ("success_rate", "collision_rate", "navigation_failure_rate"):
        out[column] *= 100.0
    scenario_order = {name: i for i, name in enumerate((*SCENARIOS, "Overall"))}
    method_order = {name: i for i, name in enumerate(METHODS)}
    return (
        out.assign(
            _scenario=out["scenario"].map(scenario_order),
            _method=out["method"].map(method_order),
        )
        .sort_values(["dynamics", "_scenario", "_method"])
        .drop(columns=["_scenario", "_method"])
        .reset_index(drop=True)
    )


def save_table_png(
    summary: pd.DataFrame, output_path: Path, dynamics: str
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
        }
    )
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    navy = "#263B50"
    slate = "#DCE5EC"
    grid = "#AAB7C4"
    text_dark = "#102A43"
    text_mid = "#52606D"
    ours = "#2F6F73"
    ours_fill = "#E9F3F2"
    ours_overall = "#DCECEA"
    overall_fill = "#EEF2F6"

    left, right = 0.04, 0.96
    scenario_width = 0.13
    metric_width = (right - left - scenario_width) / 8.0
    x_edges = [left, left + scenario_width]
    for _ in range(8):
        x_edges.append(x_edges[-1] + metric_width)

    header1 = 0.078
    header2 = 0.067
    body_height = 0.073
    table_top = 0.82
    y_edges = [table_top, table_top - header1, table_top - header1 - header2]
    for _ in range(len(SCENARIOS) + 1):
        y_edges.append(y_edges[-1] - body_height)

    def cell(
        x0,
        x1,
        y0,
        y1,
        face,
        label="",
        color=text_dark,
        weight="normal",
        size=7.0,
        align="center",
        linewidth=0.45,
    ):
        ax.add_patch(
            Rectangle(
                (x0, y1),
                x1 - x0,
                y0 - y1,
                facecolor=face,
                edgecolor=grid,
                linewidth=linewidth,
            )
        )
        if label:
            xpos = (x0 + x1) / 2.0 if align == "center" else x0 + 0.008
            ax.text(
                xpos,
                (y0 + y1) / 2.0,
                label,
                ha=align,
                va="center",
                color=color,
                fontsize=size,
                fontweight=weight,
                linespacing=1.15,
            )

    ax.text(
        0.047,
        0.886,
        f"Experiment I - {DYNAMICS_NAMES[dynamics]} Benchmark",
        ha="left",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color="#1F2D3D",
    )
    ax.text(
        0.047,
        0.853,
        benchmark_subtitle(summary, dynamics),
        ha="left",
        va="center",
        fontsize=8.4,
        color=text_mid,
    )

    cell(
        x_edges[0],
        x_edges[1],
        y_edges[0],
        y_edges[2],
        navy,
        "Scenario",
        "white",
        "bold",
        7.4,
    )
    for method_index, method in enumerate(METHODS):
        c0 = 1 + 2 * method_index
        header_color = ours if method == "corridor_bas" else navy
        label = METHOD_NAMES[method].replace("<br/>", "\n")
        cell(
            x_edges[c0],
            x_edges[c0 + 2],
            y_edges[0],
            y_edges[1],
            header_color,
            label,
            "white",
            "bold",
            7.4,
        )
        cell(
            x_edges[c0],
            x_edges[c0 + 1],
            y_edges[1],
            y_edges[2],
            slate,
            "Success\nrate (%)",
            "#243B53",
            "bold",
            6.8,
        )
        cell(
            x_edges[c0 + 1],
            x_edges[c0 + 2],
            y_edges[1],
            y_edges[2],
            slate,
            "Mean navigation\ntime (s)",
            "#243B53",
            "bold",
            6.8,
        )

    dynamics_summary = summary[summary["dynamics"] == dynamics]
    for row_index, scenario in enumerate((*SCENARIOS, "Overall")):
        y0, y1 = y_edges[2 + row_index], y_edges[3 + row_index]
        block = (
            dynamics_summary[dynamics_summary["scenario"] == scenario]
            .set_index("method")
            .reindex(METHODS)
        )
        best_success = float(block["success_rate"].max())
        finite_runtime = block["mean_navigation_time_s"].dropna()
        best_runtime = (
            float(finite_runtime.min()) if not finite_runtime.empty else math.nan
        )
        base_fill = overall_fill if scenario == "Overall" else "white"
        cell(
            x_edges[0],
            x_edges[1],
            y0,
            y1,
            base_fill,
            SCENARIO_NAMES[scenario],
            text_dark,
            "bold",
            8.0,
            "left",
        )
        for method_index, method in enumerate(METHODS):
            c0 = 1 + 2 * method_index
            success = float(block.loc[method, "success_rate"])
            runtime = float(block.loc[method, "mean_navigation_time_s"])
            method_fill = base_fill
            if method == "corridor_bas":
                method_fill = ours_overall if scenario == "Overall" else ours_fill
            cell(
                x_edges[c0],
                x_edges[c0 + 1],
                y0,
                y1,
                method_fill,
                f"{success:.1f}",
                text_dark,
                "bold" if abs(success - best_success) < 1.0e-9 else "normal",
                8.0,
            )
            cell(
                x_edges[c0 + 1],
                x_edges[c0 + 2],
                y0,
                y1,
                method_fill,
                "-" if math.isnan(runtime) else f"{runtime:.2f}",
                text_dark,
                "bold"
                if not math.isnan(runtime) and abs(runtime - best_runtime) < 1.0e-9
                else "normal",
                8.0,
            )

    ax.plot([left, right], [table_top, table_top], color=text_dark, linewidth=1.2)
    ax.plot([left, right], [y_edges[2], y_edges[2]], color=text_mid, linewidth=0.9)
    ax.plot([left, right], [y_edges[-2], y_edges[-2]], color=text_mid, linewidth=0.9)
    ax.plot([left, right], [y_edges[-1], y_edges[-1]], color=text_dark, linewidth=1.2)
    ax.text(
        0.047,
        y_edges[-1] - 0.031,
        "Notes.",
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=text_mid,
    )
    ax.text(
        0.082,
        y_edges[-1] - 0.031,
        "Success rate is the percentage of trials reaching the goal without collision, deadlock, or timeout. "
        "Navigation time is simulated elapsed time from start to goal, averaged over successful trials only.",
        ha="left",
        va="center",
        fontsize=6.8,
        color=text_mid,
    )
    ax.text(
        0.047,
        y_edges[-1] - 0.052,
        "Bold marks the highest success rate and the lowest successful-trial navigation time in each row independently.",
        ha="left",
        va="center",
        fontsize=6.8,
        color=text_mid,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path.with_suffix(".png"), format="png", dpi=300, facecolor="white"
    )
    plt.close(fig)


def generate_tables(root: Path, expected_environments=None):
    root = Path(root)
    trials = collect_summaries(root)
    if trials.empty:
        raise RuntimeError(f"No summary.json found under {root}")
    if expected_environments is None and (root / "benchmark_manifest.json").exists():
        expected_environments = json.loads(
            (root / "benchmark_manifest.json").read_text(encoding="utf-8")
        )["environments"]
    validate_trials(trials, expected_environments)
    summary = aggregate_trials(trials)
    table_dir = root / "tables"
    figure_dir = root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for dynamics in DYNAMICS:
        csv_path = table_dir / f"benchmark_success_runtime_{dynamics}.csv"
        png_path = figure_dir / f"benchmark_success_runtime_{dynamics}.png"
        summary[summary["dynamics"] == dynamics].to_csv(
            csv_path, index=False, float_format="%.4f"
        )
        save_table_png(summary, png_path, dynamics)
        outputs.append((dynamics, csv_path, png_path))
    return len(trials), outputs


def main() -> None:
    n_trials, outputs = generate_tables(_results_root())
    print("Benchmark tables finished.")
    print(f"Trials : {n_trials}")
    for dynamics, csv_path, png_path in outputs:
        print(f"{dynamics:6s} CSV: {csv_path}")
        print(f"{dynamics:6s} PNG: {png_path}")


if __name__ == "__main__":
    main()
