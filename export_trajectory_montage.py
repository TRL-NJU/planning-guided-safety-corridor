from __future__ import annotations


import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from core.config import project_root
from core.constants import DYNAMICS, METHOD_LABELS, METHODS
from environments.scenario import Scenario
from visualization.corridor_preview_data import (
    PREVIEW_FILENAME,
    load_corridor_preview_data,
)
from visualization.scene import _draw_dynamic_obstacles, _draw_obstacle
from visualization.style import (
    COLORS,
    apply_publication_style,
    style_axes,
)


# ------------------------- User-editable defaults -------------------------
DEFAULT_SCENARIO = "S1"
DEFAULT_METHOD = "corridor_bas"
DEFAULT_DYNAMICS = "single"
DEFAULT_COLUMNS = 6
DEFAULT_RESULTS_ROOT = "results/benchmark"
DEFAULT_SHOW = True

# Only these environment numbers are shown. Use None to show every environment.
# Examples:
DEFAULT_ENVIRONMENT_IDS = (1,4,7,8,9,16,17,11,12,13,14,18)
# DEFAULT_ENVIRONMENT_IDS = tuple(range(1, 26))
#  DEFAULT_ENVIRONMENT_IDS = None
# DEFAULT_ENVIRONMENT_IDS = None
# Vertical gap between adjacent rows, expressed as a fraction of the average
# panel height. Recommended range: 0.05 (compact) to 0.40 (loose).
DEFAULT_ROW_SPACING = 0.0
# Equal horizontal gap between all adjacent columns.
DEFAULT_COLUMN_SPACING = 0.08

# Match view_corridor_evolution.py and visualization/style.py.
AXIS_LINEWIDTH = 0.8
FONT_WEIGHT = "normal"
# Smaller values move x(m) and y(m) closer to the axes; negative values work.
AXIS_LABEL_PAD = 2.0
# Shared export width used by both montage and corridor-preview figures.
EXPORT_WIDTH_INCHES = 22.0
EXPORT_DPI = 300
# Shared six-column geometry. Keep these values identical to
# view_corridor_evolution.py so the two PNG figures stack exactly.
FIGURE_LEFT = 0.045
FIGURE_RIGHT = 0.995
FIGURE_TOP = 0.880
FIGURE_BOTTOM = 0.075
LEGEND_Y = 0.98
CORRIDOR_TRAJECTORY_COLOR = "#0072B2"
TRAJECTORY_LINEWIDTH = 2.35
PLANNED_PATH_LINEWIDTH = 1.25
# -------------------------------------------------------------------------


def _arguments():
    parser = argparse.ArgumentParser(
        description="Assemble saved trajectories into one grid figure."
    )
    parser.add_argument("--scenario", choices=("S1", "S2"), default=DEFAULT_SCENARIO)
    parser.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    parser.add_argument("--dynamics", choices=DYNAMICS, default=DEFAULT_DYNAMICS)
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def _natural_environment_key(path):
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return path.stem


def _select_environment_files(environment_files, selected_ids):
    """Return the requested environments in the order written above."""
    if selected_ids is None:
        return environment_files

    requested = []
    for value in selected_ids:
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "DEFAULT_ENVIRONMENT_IDS must contain integer environment numbers."
            ) from exc
        if number < 0:
            raise ValueError("Environment numbers must be non-negative.")
        environment_id = f"environment_{number:03d}"
        if environment_id not in requested:
            requested.append(environment_id)

    if not requested:
        raise ValueError(
            "DEFAULT_ENVIRONMENT_IDS is empty. Use None to show every environment."
        )

    available = {path.stem: path for path in environment_files}
    missing = [
        environment_id
        for environment_id in requested
        if environment_id not in available
    ]
    if missing:
        raise FileNotFoundError(
            "Selected environment files do not exist: " + ", ".join(missing)
        )
    return [available[environment_id] for environment_id in requested]


def _resolve_result_dir(root, results_root, dynamics, scenario, method, environment_id):
    """Resolve the exact saved trial without mixing its scene with a newer JSON."""
    benchmark = results_root / dynamics / scenario / method / environment_id
    if (benchmark / "summary.json").exists() and (
        benchmark / "timeseries.npz"
    ).exists():
        return benchmark
    demo = root / "results" / "demo" / scenario / dynamics / environment_id / method
    if (demo / "summary.json").exists() and (demo / "timeseries.npz").exists():
        return demo
    return benchmark


def _load_saved_trial(result_dir):
    summary_path = result_dir / "summary.json"
    series_path = result_dir / "timeseries.npz"
    if not summary_path.exists() or not series_path.exists():
        return None, None, None, None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(series_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    final_path = None
    saved_scenario = None
    preview_path = result_dir / PREVIEW_FILENAME
    if preview_path.exists():
        preview = load_corridor_preview_data(preview_path)
        saved_scenario = preview["scenario"]
        if preview["frames"]:
            final_path = np.asarray(preview["frames"][-1]["path"], float)
    saved_path = result_dir / "final_jps_path.npy"
    if final_path is None and saved_path.exists():
        final_path = np.asarray(np.load(saved_path, allow_pickle=False), float)
    return summary, arrays, final_path, saved_scenario


def _failure_reason(summary):
    reason = str(summary.get("failure_reason", "unknown") or "unknown")
    return reason.replace("physical_collision", "collision").replace("_", " ")


def _trajectory_color(method):
    """Use the same blue executed trajectory for every method."""
    return CORRIDOR_TRAJECTORY_COLOR


def _draw_panel(ax, scenario, method, summary, arrays, final_path, row, column, rows):
    for obstacle in scenario.static_obstacles:
        _draw_obstacle(
            ax,
            obstacle,
            edgecolor=COLORS["static_edge"],
            facecolor=COLORS["static_fill"],
            alpha=0.72,
            linewidth=0.72,
            zorder=1,
        )

    if arrays is None:
        t_final = 0.0
    else:
        t_values = arrays.get("t", np.empty(0))
        t_final = float(t_values[-1]) if len(t_values) else 0.0
    _draw_dynamic_obstacles(ax, scenario, t_final)

    if final_path is not None and len(final_path):
        ax.plot(
            final_path[:, 0],
            final_path[:, 1],
            color=COLORS["gray"],
            linestyle=(0, (4, 2)),
            linewidth=PLANNED_PATH_LINEWIDTH,
            alpha=0.9,
            zorder=6,
        )
    if arrays is not None and "px" in arrays and "py" in arrays:
        ax.plot(
            arrays["px"],
            arrays["py"],
            color=_trajectory_color(method),
            linewidth=TRAJECTORY_LINEWIDTH,
            solid_capstyle="round",
            zorder=9,
        )

    ax.scatter(
        [scenario.start[0]],
        [scenario.start[1]],
        s=48,
        marker="o",
        color=COLORS["green"],
        edgecolors="white",
        linewidths=0.8,
        zorder=12,
    )
    ax.scatter(
        [scenario.goal[0]],
        [scenario.goal[1]],
        marker="*",
        s=105,
        color=COLORS["yellow"],
        edgecolors=COLORS["black"],
        linewidths=0.7,
        zorder=12,
    )

    ax.set_xlim(0.0, scenario.width)
    ax.set_ylim(0.0, scenario.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.linspace(0.0, scenario.width, 5))
    ax.set_yticks(np.linspace(0.0, scenario.height, 5))
    ax.set_xlabel(
        "x(m)" if row == rows - 1 else "",
        labelpad=AXIS_LABEL_PAD,
    )
    ax.set_ylabel(
        "y(m)" if column == 0 else "",
        fontfamily="Times New Roman",
        labelpad=AXIS_LABEL_PAD,
    )
    if row != rows - 1:
        ax.tick_params(labelbottom=False)
    if column != 0:
        ax.tick_params(labelleft=False)
    style_axes(ax, grid=True)
    ax.tick_params(which="both", width=AXIS_LINEWIDTH)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)


def _legend(method, has_dynamic, has_saved_path):
    handles = [
        Patch(
            facecolor=COLORS["static_fill"],
            edgecolor=COLORS["static_edge"],
            label="Obstacle",
        ),
        Line2D(
            [0],
            [0],
            color=_trajectory_color(method),
            linewidth=2.4,
            # label=f"{METHOD_LABELS.get(method, method)} trajectory"),
            label=f"{METHOD_LABELS.get(method, method)} trajectory",
        ),
    ]
    if has_saved_path:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["gray"],
                linestyle=(0, (4, 2)),
                linewidth=1.3,
                label="Final JPS path",
            )
        )
    if has_dynamic:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["orange"],
                linestyle=(0, (5, 2)),
                linewidth=0.9,
                label="Dynamic - obstacle motion",
            )
        )
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLORS["green"],
                markeredgecolor="white",
                markersize=12,
                label="Start",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor=COLORS["yellow"],
                markeredgecolor=COLORS["black"],
                markersize=15,
                label="Goal",
            ),
        ]
    )
    return handles


def main():
    args = _arguments()
    if args.columns < 1:
        raise ValueError("--columns must be at least 1")
    root = project_root()
    environment_dir = root / "data" / "custom_environments" / args.scenario
    environment_files = sorted(
        environment_dir.glob("environment_*.json"), key=_natural_environment_key
    )
    if not environment_files:
        raise RuntimeError(f"No environments under {environment_dir}")
    environment_files = _select_environment_files(
        environment_files,
        DEFAULT_ENVIRONMENT_IDS,
    )

    row_spacing = float(DEFAULT_ROW_SPACING)
    if row_spacing < 0.0:
        raise ValueError("DEFAULT_ROW_SPACING must be non-negative")
    column_spacing = float(DEFAULT_COLUMN_SPACING)
    if column_spacing < 0.0:
        raise ValueError("DEFAULT_COLUMN_SPACING must be non-negative")
    if EXPORT_WIDTH_INCHES <= 0.0 or EXPORT_DPI <= 0:
        raise ValueError("EXPORT_WIDTH_INCHES and EXPORT_DPI must be positive")

    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = root / results_root
    columns = int(args.columns)
    rows = int(math.ceil(len(environment_files) / columns))

    apply_publication_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Times New Roman"],
            "font.size": 3,
            "font.weight": FONT_WEIGHT,
            "axes.labelsize": 18,
            "axes.labelweight": FONT_WEIGHT,
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "axes.linewidth": AXIS_LINEWIDTH,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.major.width": AXIS_LINEWIDTH,
            "ytick.major.width": AXIS_LINEWIDTH,
            "legend.fontsize": 18,
            "savefig.bbox": None,
        }
    )
    panel_height = EXPORT_WIDTH_INCHES / float(columns)
    figure_height = panel_height * rows + 1.10
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(EXPORT_WIDTH_INCHES, figure_height),
        squeeze=False,
        facecolor="white",
        gridspec_kw={"width_ratios": [1.0] * columns},
    )

    has_dynamic = False
    has_saved_path = False
    missing = []
    for index, environment_file in enumerate(environment_files):
        row, column = divmod(index, columns)
        environment_id = environment_file.stem
        result_dir = _resolve_result_dir(
            root,
            results_root,
            args.dynamics,
            args.scenario,
            args.method,
            environment_id,
        )
        summary, arrays, final_path, saved_scenario = _load_saved_trial(result_dir)
        if summary is not None:
            saved_id = str(summary.get("environment_id", "") or "")
            if saved_id and saved_id != environment_id:
                raise ValueError(
                    f"Saved result {saved_id!r} does not match {environment_id!r}"
                )
        if saved_scenario is not None:
            saved_id = str(saved_scenario.environment_id or "")
            if saved_id and saved_id != environment_id:
                raise ValueError(
                    f"Saved scene {saved_id!r} does not match {environment_id!r}"
                )
            saved_scenario.environment_id = environment_id
            scenario = saved_scenario
        else:
            scenario = Scenario.load(environment_file)
        has_dynamic |= bool(scenario.dynamic_specs)
        has_saved_path |= final_path is not None and len(final_path) > 0
        if summary is None:
            missing.append(scenario.environment_id)
        _draw_panel(
            axes[row, column],
            scenario,
            args.method,
            summary,
            arrays,
            final_path,
            row,
            column,
            rows,
        )

    for index in range(len(environment_files), rows * columns):
        axes.flat[index].axis("off")

    fig.legend(
        handles=_legend(args.method, has_dynamic, has_saved_path),
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, LEGEND_Y),
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.subplots_adjust(
        left=FIGURE_LEFT,
        right=FIGURE_RIGHT,
        top=FIGURE_TOP,
        bottom=FIGURE_BOTTOM,
        hspace=row_spacing,
        wspace=column_spacing,
    )

    output_dir = results_root / "figures" / "montages"
    output_path = (
        output_dir / f"{args.scenario}_{args.dynamics}_{args.method}_montage.png"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_options = {"bbox_inches": None, "facecolor": "white"}
    fig.savefig(output_path, dpi=EXPORT_DPI, **save_options)
    print(f"Environments : {len(environment_files)}")
    print("Selected     : " + ", ".join(path.stem for path in environment_files))
    print(f"Layout       : {rows} rows x {columns} columns")
    print(f"Row spacing  : {row_spacing}")
    print(f"Column space : {column_spacing}")
    print(
        f"Missing      : {len(missing)}"
        + (f" ({', '.join(missing)})" if missing else "")
    )
    print(
        f"Export width : {EXPORT_WIDTH_INCHES} in ({EXPORT_WIDTH_INCHES * EXPORT_DPI:.0f} px)"
    )
    print(f"PNG          : {output_path}")
    if DEFAULT_SHOW and not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
