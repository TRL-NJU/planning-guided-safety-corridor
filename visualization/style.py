from __future__ import annotations

from pathlib import Path
import matplotlib as mpl


# Okabe-Ito colour-blind-safe palette plus neutral tones for geometry.
COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#222222",
    "gray": "#7A7A7A",
    "light_gray": "#D9D9D9",
    "static_edge": "#4D4D4D",
    "static_fill": "#D9D9D9",
    "grid": "#C8C8C8",
}

METHOD_COLORS = {
    "jps": COLORS["blue"],
    "jps_bas": COLORS["orange"],
    "single_scp_bas": COLORS["green"],
    "corridor_bas": COLORS["purple"],
}

STATUS_COLORS = {
    "success": COLORS["green"],
    "collision": COLORS["vermillion"],
    "navigation_failure": COLORS["orange"],
    "missing": COLORS["gray"],
}


def apply_publication_style():
    """Apply a journal-ready, colour-blind-safe Matplotlib baseline."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["black"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.7,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "axes.unicode_minus": False,
        }
    )


def style_axes(ax, grid=True):
    ax.tick_params(which="both", top=False, right=False)
    if grid:
        ax.grid(
            True,
            color=COLORS["grid"],
            linewidth=0.55,
            linestyle=(0, (2, 2)),
            alpha=0.55,
            zorder=0,
        )
    ax.set_axisbelow(True)


def save_publication_figure(fig, path, dpi=300):
    """Save one high-resolution PNG figure."""
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=int(dpi), bbox_inches="tight", facecolor="white")
    return path
