from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from core.constants import METHOD_LABELS
from visualization.style import (
    COLORS,
    METHOD_COLORS,
    STATUS_COLORS,
    apply_publication_style,
    save_publication_figure,
    style_axes,
)


apply_publication_style()


def _obstacle_polygon(obstacle, spacing=0.05):
    points = obstacle.boundary_points(spacing, 0.0)
    return np.vstack((points, points[0]))


def _draw_obstacle(
    ax,
    obstacle,
    edgecolor,
    facecolor="none",
    alpha=1.0,
    linestyle="-",
    linewidth=1.2,
    zorder=2,
):
    polygon = _obstacle_polygon(obstacle)
    ax.plot(
        polygon[:, 0],
        polygon[:, 1],
        color=edgecolor,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder + 1,
    )
    if facecolor != "none":
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=facecolor,
            edgecolor="none",
            alpha=alpha,
            zorder=zorder,
        )


def _trajectory_vertices(trajectory):
    if hasattr(trajectory, "positions"):
        return np.asarray(trajectory.positions, float)
    if hasattr(trajectory, "a") and hasattr(trajectory, "b"):
        return np.vstack((trajectory.a, trajectory.b))
    return np.empty((0, 2), float)


def _draw_dynamic_obstacles(ax, scenario, t_final):
    """Draw motion path, t=0 footprint, and method-specific final footprint."""
    for spec in scenario.dynamic_specs:
        vertices = _trajectory_vertices(spec.trajectory)
        if len(vertices) >= 2:
            ax.plot(
                vertices[:, 0],
                vertices[:, 1],
                color=COLORS["orange"],
                linestyle=(0, (5, 2)),
                linewidth=1.25,
                alpha=0.9,
                zorder=3,
            )
        _draw_obstacle(
            ax,
            spec.obstacle_at(0.0),
            edgecolor=COLORS["orange"],
            facecolor=COLORS["yellow"],
            alpha=0.35,
            linestyle="-",
            linewidth=1.25,
            zorder=4,
        )
        _draw_obstacle(
            ax,
            spec.obstacle_at(t_final),
            edgecolor=COLORS["vermillion"],
            facecolor="none",
            alpha=1.0,
            linestyle=(0, (2, 1.5)),
            linewidth=1.55,
            zorder=5,
        )


def _mark_failure(ax, result):
    """Use a visible frame and concise reason without relying on colour alone."""
    if result.summary["terminal_status"] == "success":
        return
    color = STATUS_COLORS.get(result.summary["terminal_status"], COLORS["vermillion"])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.2)
    reason = result.summary.get("failure_reason", "")
    ax.text(
        0.5,
        0.975,
        f"FAILURE: {reason or 'unknown'}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor="white",
            edgecolor=color,
            linewidth=1.1,
            alpha=0.94,
        ),
        zorder=50,
    )


def _legend_handles(has_dynamic, show_safety_geometry=True):
    handles = [
        Patch(
            facecolor=COLORS["static_fill"],
            edgecolor=COLORS["static_edge"],
            label="Static obstacle",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["black"],
            linewidth=2.1,
            label="Executed trajectory (method colour)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["gray"],
            linestyle=(0, (3, 2)),
            linewidth=1.15,
            label="Final JPS path",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["purple"],
            linestyle="-.",
            linewidth=1.0,
            label="SCP boundary",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["green"],
            markeredgecolor="white",
            markersize=7,
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=COLORS["yellow"],
            markeredgecolor=COLORS["black"],
            markersize=10,
            label="Goal",
        ),
    ]
    if not show_safety_geometry:
        handles = [item for item in handles if item.get_label() != "SCP boundary"]
    if has_dynamic:
        handles[1:1] = [
            Line2D(
                [0],
                [0],
                color=COLORS["orange"],
                linestyle=(0, (5, 2)),
                linewidth=1.25,
                label="Dynamic-obstacle motion",
            ),
            Patch(
                facecolor=COLORS["yellow"],
                edgecolor=COLORS["orange"],
                alpha=0.55,
                label="Dynamic obstacle at $t=0$",
            ),
            Patch(
                facecolor="none",
                edgecolor=COLORS["vermillion"],
                linestyle=(0, (2, 1.5)),
                label="Dynamic obstacle at $t=t_f$",
            ),
        ]
    return handles


def plot_comparison(results, show=True, save_path=None, show_safety_geometry=True):
    sc = results[0].scenario
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.4, 9.5),
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    panel_labels = ("a", "b", "c", "d")
    for panel, ax, res in zip(panel_labels, axes.flat, results):
        for o in sc.static_obstacles:
            _draw_obstacle(
                ax,
                o,
                edgecolor=COLORS["static_edge"],
                facecolor=COLORS["static_fill"],
                alpha=0.72,
                linewidth=1.1,
                zorder=1,
            )
        arr = res.logger.arrays()
        t_final = float(arr["t"][-1]) if "t" in arr and len(arr["t"]) else 0.0
        _draw_dynamic_obstacles(ax, sc, t_final)
        if "px" in arr:
            ax.plot(
                arr["px"],
                arr["py"],
                color=METHOD_COLORS.get(res.method, COLORS["blue"]),
                linewidth=2.25,
                zorder=9,
            )
        if res.snapshots:
            snap = res.snapshots[-1]
            pth = np.asarray(snap["path"])
            ax.plot(
                pth[:, 0],
                pth[:, 1],
                color=COLORS["gray"],
                linestyle=(0, (3, 2)),
                linewidth=1.15,
                alpha=0.9,
                zorder=6,
            )
            for poly in snap["polygons"] if show_safety_geometry else []:
                if poly is not None and len(poly):
                    q = np.vstack((poly, poly[0]))
                    ax.plot(
                        q[:, 0],
                        q[:, 1],
                        color=COLORS["purple"],
                        linestyle="-.",
                        linewidth=1.0,
                        alpha=0.65,
                        zorder=7,
                    )
            if (
                show_safety_geometry
                and snap["seeds"] is not None
                and len(snap["seeds"])
            ):
                q = np.asarray(snap["seeds"])
                ax.scatter(
                    q[:, 0],
                    q[:, 1],
                    s=22,
                    marker="x",
                    color=COLORS["purple"],
                    linewidths=1.0,
                    zorder=8,
                )
        ax.scatter(
            [sc.start[0]],
            [sc.start[1]],
            s=52,
            marker="o",
            color=COLORS["green"],
            edgecolors="white",
            linewidths=0.8,
            zorder=12,
        )
        ax.scatter(
            [sc.goal[0]],
            [sc.goal[1]],
            marker="*",
            s=115,
            color=COLORS["yellow"],
            edgecolors=COLORS["black"],
            linewidths=0.7,
            zorder=12,
        )
        ax.set(xlim=(0, sc.width), ylim=(0, sc.height), xlabel="x(m)", ylabel="y(m)")
        ax.set_aspect("equal", adjustable="box")
        style_axes(ax, grid=True)
        status = res.summary["terminal_status"]
        status_color = STATUS_COLORS.get(status, COLORS["black"])
        ax.set_title(
            f"({panel}) {METHOD_LABELS.get(res.method, res.method)}\n"
            f"{status.replace('_', ' ').upper()}",
            color=COLORS["black"] if status == "success" else status_color,
        )
        _mark_failure(ax, res)
    env_id = getattr(sc, "environment_id", "") or f"seed={sc.seed}"
    fig.suptitle(
        f"{sc.scenario_type} | {env_id} | {results[0].dynamics} dynamics",
        fontsize=11.5,
        fontweight="bold",
        y=0.995,
    )
    has_dynamic = bool(sc.dynamic_specs)
    legend_handles = _legend_handles(has_dynamic, show_safety_geometry)
    ncol = 5 if has_dynamic else 6
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.4,
        columnspacing=1.2,
    )
    bottom = 0.145 if has_dynamic else 0.085
    fig.tight_layout(rect=(0, bottom, 1, 0.97), h_pad=1.3, w_pad=1.0)
    if save_path is not None:
        save_publication_figure(fig, Path(save_path), dpi=300)
    if show:
        plt.show()
    return fig, axes
