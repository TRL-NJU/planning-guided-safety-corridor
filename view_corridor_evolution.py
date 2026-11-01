from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from core.config import load_run_config, project_root, resolve_project_path
from planning.path_utils import PolylinePath
from visualization.corridor_preview_data import (
    PREVIEW_FILENAME,
    discover_corridor_previews,
    load_corridor_preview_data,
)
from visualization.scene import _draw_dynamic_obstacles, _draw_obstacle
from visualization.style import COLORS, apply_publication_style, style_axes


apply_publication_style()
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.5,
        "font.weight": "normal",
        "axes.labelsize": 25,
        "axes.labelweight": "normal",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 18,
        "savefig.bbox": None,
    }
)

METHOD = "corridor_bas"
PANEL_LABELS = ("a", "b", "c", "d", "e", "f")
FIELD_GRID_SIZE = 120
# Shared six-column geometry. Keep these values identical to
# export_trajectory_montage.py so the two PNG figures stack exactly.
EXPORT_WIDTH_INCHES = 22.0
EXPORT_HEIGHT_INCHES = 4.55
EXPORT_DPI = 300
AXIS_LINEWIDTH = 0.8
AXIS_LABEL_PAD = 2.0
COLUMN_SPACING = 0.08
FIGURE_LEFT = 0.045
FIGURE_RIGHT = 0.995
FIGURE_TOP = 0.880
FIGURE_BOTTOM = 0.105
LEGEND_Y = 0.94

DEFAULT_STYLE = {
    "trajectory": "#0072B2",
    "planned_path": "#4D4D4D",
    "current_point": "#D55E00",
    "scp_start": "#56B4E9",
    "scp_end": "#CC79A7",
    "corridor_face": "#F4E7B2",
    "corridor_edge": "#A06B00",
    "scp_alpha": 0.24,
    "corridor_alpha": 1.0,
}

COLOR_LABELS = (
    ("trajectory", "执行轨迹"),
    ("planned_path", "未执行规划路径"),
    ("current_point", "当前点"),
    ("scp_start", "近端SCP"),
    ("scp_end", "远端SCP"),
    ("corridor_face", "拟合走廊内部"),
    ("corridor_edge", "拟合走廊边界"),
)


def _result_roots():
    roots = []
    try:
        opt = load_run_config()["benchmark"]
        roots.append(resolve_project_path(opt["results_root"]))
    except Exception:
        roots.append(project_root() / "results" / "custom")
    roots.append(project_root() / "results" / "demo")
    return roots


def _remaining_path(path_points, current):
    path = PolylinePath(np.asarray(path_points, float))
    s0, _, _ = path.project(current)
    samples = np.arange(s0, path.length, 0.10)
    if not len(samples) or path.length - samples[-1] > 1.0e-9:
        samples = np.r_[samples, path.length]
    return np.asarray([path.point_at(float(s)) for s in samples])


def _scp_colours(count, style):
    if count <= 0:
        return []
    cmap = LinearSegmentedColormap.from_list(
        "scp_along_path", [style["scp_start"], style["scp_end"]]
    )
    return [cmap(x) for x in np.linspace(0.0, 1.0, count)]


def _field_grid(frame, scenario):
    polygons = [p for p in frame["polygons"] if p is not None and len(p) >= 3]
    if not polygons:
        return None
    all_points = np.vstack(polygons)
    lo = np.maximum(np.min(all_points, axis=0) - 0.35, 0.0)
    hi = np.minimum(
        np.max(all_points, axis=0) + 0.35,
        [float(scenario.width), float(scenario.height)],
    )
    xs = np.linspace(lo[0], hi[0], FIELD_GRID_SIZE)
    ys = np.linspace(lo[1], hi[1], FIELD_GRID_SIZE)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    points = np.c_[X.ravel(), Y.ravel()]
    H = np.asarray([frame["field"].value(point) for point in points]).reshape(X.shape)
    return X, Y, H


def _draw_fitted_corridor(ax, cache, style):
    if cache is None:
        return
    X, Y, H = cache
    hmax = float(np.nanmax(H))
    hmin = float(np.nanmin(H))
    if hmax <= 0.0:
        return
    ax.contourf(
        X,
        Y,
        H,
        levels=[0.0, hmax + 1.0e-9],
        colors=[style["corridor_face"]],
        alpha=float(style["corridor_alpha"]),
        antialiased=True,
        zorder=1,
    )
    if hmin < 0.0 < hmax:
        ax.contour(
            X,
            Y,
            H,
            levels=[0.0],
            colors=[style["corridor_edge"]],
            linewidths=1.25,
            zorder=2,
        )


def _draw_local_scps(ax, frame, style):
    colours = _scp_colours(len(frame["polygons"]), style)
    for colour, polygon in zip(colours, frame["polygons"]):
        if polygon is None or len(polygon) < 3:
            continue
        closed = np.vstack((polygon, polygon[0]))
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=colour,
            edgecolor="none",
            alpha=float(style["scp_alpha"]),
            zorder=3,
        )
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color=colour,
            linewidth=0.9,
            alpha=0.92,
            zorder=4,
        )
    if len(frame["seeds"]):
        ax.scatter(
            frame["seeds"][:, 0],
            frame["seeds"][:, 1],
            s=20,
            marker="x",
            c=_scp_colours(len(frame["seeds"]), style),
            linewidths=1.0,
            zorder=8,
        )


def _draw_scene(ax, scenario, t):
    for obstacle in scenario.static_obstacles:
        _draw_obstacle(
            ax,
            obstacle,
            edgecolor=COLORS["static_edge"],
            facecolor=COLORS["static_fill"],
            alpha=0.72,
            linewidth=0.72,
            zorder=5,
        )
    _draw_dynamic_obstacles(ax, scenario, t)


def _legend_handles(style, has_dynamic):
    handles = [
        Patch(
            facecolor=style["corridor_face"],
            edgecolor=style["corridor_edge"],
            alpha=float(style["corridor_alpha"]),
            label="Corridor",
        ),
        Patch(
            facecolor=to_rgba(style["scp_start"], float(style["scp_alpha"])),
            edgecolor=style["scp_end"],
            label="Local SCPs",
        ),
        Line2D(
            [0],
            [0],
            color=style["trajectory"],
            linewidth=2.4,
            label="Executed trajectory",
        ),
        Line2D(
            [0],
            [0],
            color=style["planned_path"],
            linewidth=1.3,
            linestyle=(0, (4, 2)),
            label="Remaining JPS path",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6.5,
            markerfacecolor=COLORS["green"],
            markeredgecolor="white",
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6.5,
            markerfacecolor=style["current_point"],
            markeredgecolor="white",
            label="Current",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markersize=9.0,
            markerfacecolor=COLORS["yellow"],
            markeredgecolor=COLORS["black"],
            label="Goal",
        ),
    ]
    if has_dynamic:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["orange"],
                linewidth=1.2,
                linestyle=(0, (5, 2)),
                label="Dynamic-obstacle motion",
            )
        )
    return handles


class PngNavigationToolbar(NavigationToolbar2Tk):
    """Route the toolbar Save button through the PNG-only export dialog."""

    def __init__(self, canvas, master, save_callback):
        self.save_callback = save_callback
        super().__init__(canvas, master, pack_toolbar=False)

    def save_figure(self, *args):
        self.save_callback()


class CorridorPreviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Corridor preview - saved Monte-Carlo result")
        self.root.geometry("1680x1000")
        self.style = dict(DEFAULT_STYLE)
        self.bundle = None
        self.field_cache = []
        self.paths = {}

        controls = ttk.Frame(root, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        plot_host = ttk.Frame(root)
        plot_host.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            controls,
            text="已保存的 corridor_bas 结果",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(anchor="w")
        self.trial_var = tk.StringVar()
        self.trial_box = ttk.Combobox(
            controls, textvariable=self.trial_var, width=42, state="readonly"
        )
        self.trial_box.pack(fill=tk.X, pady=(5, 4))
        self.trial_box.bind("<<ComboboxSelected>>", lambda _event: self.load_selected())
        ttk.Button(controls, text="刷新结果列表", command=self.refresh_files).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(controls, text="打开预览数据文件…", command=self.browse_file).pack(
            fill=tk.X, pady=2
        )

        ttk.Separator(controls).pack(fill=tk.X, pady=10)
        ttk.Label(
            controls,
            text="颜色（点击色块修改）",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(anchor="w")
        self.color_buttons = {}
        for key, label in COLOR_LABELS:
            row = ttk.Frame(controls)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
            button = tk.Button(
                row,
                width=8,
                relief=tk.RIDGE,
                background=self.style[key],
                command=lambda name=key: self.pick_colour(name),
            )
            button.pack(side=tk.RIGHT)
            self.color_buttons[key] = button

        ttk.Separator(controls).pack(fill=tk.X, pady=10)
        ttk.Label(
            controls, text="透明度", font=("Microsoft YaHei UI", 10, "bold")
        ).pack(anchor="w")
        self.scp_alpha = tk.DoubleVar(value=self.style["scp_alpha"])
        self.corridor_alpha = tk.DoubleVar(value=self.style["corridor_alpha"])
        ttk.Label(controls, text="局部SCP").pack(anchor="w")
        ttk.Scale(
            controls,
            from_=0.0,
            to=1.0,
            variable=self.scp_alpha,
            command=lambda _value: self.alpha_changed(),
        ).pack(fill=tk.X)
        ttk.Label(controls, text="拟合走廊内部").pack(anchor="w", pady=(5, 0))
        ttk.Scale(
            controls,
            from_=0.0,
            to=1.0,
            variable=self.corridor_alpha,
            command=lambda _value: self.alpha_changed(),
        ).pack(fill=tk.X)

        ttk.Separator(controls).pack(fill=tk.X, pady=10)
        ttk.Button(controls, text="恢复顶刊默认配色", command=self.reset_style).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(controls, text="导出一行六列 PNG…", command=self.export_png).pack(
            fill=tk.X, pady=2
        )
        controls.winfo_children()[-1].pack_forget()
        self.status_var = tk.StringVar(value="正在查找已保存结果…")
        ttk.Label(
            controls, textvariable=self.status_var, wraplength=310, foreground="#4D4D4D"
        ).pack(anchor="w", pady=(12, 0))

        self.figure = Figure(
            figsize=(EXPORT_WIDTH_INCHES, EXPORT_HEIGHT_INCHES),
            facecolor="white",
        )
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_host)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = PngNavigationToolbar(self.canvas, plot_host, self.save_current_view)
        toolbar.update()
        toolbar.pack(fill=tk.X)
        self.save_view_button = ttk.Button(
            plot_host,
            text="\u4fdd\u5b58\u5f53\u524d\u9875\u9762...",
            command=self.save_current_view,
        )
        self.save_view_button.place(relx=0.992, rely=0.012, anchor="ne")
        self.save_view_button.lift()
        self.refresh_files()

    def refresh_files(self):
        files = discover_corridor_previews(_result_roots())
        self.paths = {}
        for path in files:
            try:
                label = str(path.relative_to(project_root() / "results"))
            except ValueError:
                label = str(path)
            self.paths[label] = path
        labels = list(self.paths)
        self.trial_box["values"] = labels
        if labels:
            current = self.trial_var.get()
            self.trial_var.set(current if current in self.paths else labels[0])
            self.load_selected()
        else:
            self.status_var.set(
                f"未找到 {PREVIEW_FILENAME}。重新运行 run_benchmark.py；已有其他方法会继续复用，只补跑缺少预览数据的 corridor_bas。"
            )
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.axis("off")
            ax.text(
                0.5, 0.5, self.status_var.get(), ha="center", va="center", wrap=True
            )
            self.canvas.draw_idle()

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="选择已保存的走廊预览数据",
            filetypes=[("Corridor preview", "*.json.gz"), ("All files", "*.*")],
        )
        if filename:
            path = Path(filename)
            self.paths[str(path)] = path
            self.trial_box["values"] = list(self.paths)
            self.trial_var.set(str(path))
            self.load_selected()

    def load_selected(self):
        path = self.paths.get(self.trial_var.get())
        if path is None:
            return
        try:
            self.status_var.set("正在载入场景并缓存六帧拟合场…")
            self.root.update_idletasks()
            self.bundle = load_corridor_preview_data(path)
            scenario = self.bundle["scenario"]
            self.field_cache = [
                _field_grid(frame, scenario) for frame in self.bundle["frames"]
            ]
            summary = self.bundle.get("summary", {})
            self.status_var.set(
                f"{scenario.scenario_type}/{scenario.environment_id} | {self.bundle['dynamics']} | "
                f"{summary.get('terminal_status', 'unknown')}\n数据：{path}"
            )
            self.redraw()
        except Exception as exc:
            messagebox.showerror("预览数据读取失败", str(exc))

    def pick_colour(self, key):
        colour = colorchooser.askcolor(color=self.style[key], title="选择颜色")[1]
        if colour:
            self.style[key] = colour
            self.color_buttons[key].configure(background=colour)
            self.redraw()

    def alpha_changed(self):
        self.style["scp_alpha"] = float(self.scp_alpha.get())
        self.style["corridor_alpha"] = float(self.corridor_alpha.get())
        self.redraw()

    def reset_style(self):
        self.style = dict(DEFAULT_STYLE)
        self.scp_alpha.set(self.style["scp_alpha"])
        self.corridor_alpha.set(self.style["corridor_alpha"])
        for key, _label in COLOR_LABELS:
            self.color_buttons[key].configure(background=self.style[key])
        self.redraw()

    def redraw(self):
        if self.bundle is None:
            return
        scenario = self.bundle["scenario"]
        frames = self.bundle["frames"]
        trajectory = self.bundle["trajectory"]
        self.figure.clear()
        axes = self.figure.subplots(1, 6, sharex=True, sharey=True)
        for panel, frame, cache, ax in zip(
            PANEL_LABELS, frames, self.field_cache, axes
        ):
            _draw_fitted_corridor(ax, cache, self.style)
            _draw_local_scps(ax, frame, self.style)
            _draw_scene(ax, scenario, float(frame["t"]))
            remaining = _remaining_path(frame["path"], frame["current"])
            ax.plot(
                remaining[:, 0],
                remaining[:, 1],
                color=self.style["planned_path"],
                linestyle=(0, (4, 2)),
                linewidth=1.25,
                alpha=0.95,
                zorder=9,
            )
            index = int(frame["trajectory_index"])
            ax.plot(
                trajectory[: index + 1, 0],
                trajectory[: index + 1, 1],
                color=self.style["trajectory"],
                linewidth=2.35,
                solid_capstyle="round",
                zorder=10,
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
                [frame["current"][0]],
                [frame["current"][1]],
                s=54,
                marker="o",
                color=self.style["current_point"],
                edgecolors="white",
                linewidths=0.9,
                zorder=13,
            )
            ax.scatter(
                [scenario.goal[0]],
                [scenario.goal[1]],
                s=105,
                marker="*",
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
            ax.set_xlabel("x(m)", fontsize=10.5, labelpad=AXIS_LABEL_PAD)
            if ax is axes[0]:
                ax.set_ylabel("y(m)", fontsize=10.5, labelpad=AXIS_LABEL_PAD)
            style_axes(ax, grid=True)
            ax.tick_params(which="both", width=AXIS_LINEWIDTH)
            for spine in ax.spines.values():
                spine.set_linewidth(AXIS_LINEWIDTH)
            if ax is axes[0]:
                ax.tick_params(axis="y", pad=5.0)

        handles = _legend_handles(self.style, bool(scenario.dynamic_specs))
        self.figure.legend(
            handles=handles,
            loc="upper center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, LEGEND_Y),
            fontsize=10.2,
            handlelength=2.2,
            columnspacing=1.05,
        )
        self.figure.subplots_adjust(
            left=FIGURE_LEFT,
            right=FIGURE_RIGHT,
            top=FIGURE_TOP,
            bottom=FIGURE_BOTTOM,
            wspace=COLUMN_SPACING,
        )
        self.canvas.draw_idle()

    def save_current_view(self):
        """Save the current canvas as PNG without changing its geometry."""
        if self.bundle is None:
            messagebox.showwarning("No result", "Select a saved result before exporting.")
            return
        scenario = self.bundle["scenario"]
        default_name = (
            f"{scenario.scenario_type}_{scenario.environment_id}_"
            f"{self.bundle['dynamics']}_{METHOD}_current_view.png"
        )
        output_dir = project_root() / "results" / "previews"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = filedialog.asksaveasfilename(
            title="Save current view as PNG",
            initialdir=output_dir,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if not filename:
            return
        output_path = Path(filename).with_suffix(".png")
        self.figure.savefig(
            output_path,
            format="png",
            dpi=EXPORT_DPI,
            bbox_inches=None,
            facecolor=self.figure.get_facecolor(),
        )
        self.status_var.set(f"Saved current view: {output_path}")

    def export_png(self):
        """Export the six-stage figure at the configured publication size."""
        if self.bundle is None:
            messagebox.showwarning("没有结果", "请先选择一个已保存结果。")
            return
        scenario = self.bundle["scenario"]
        default_name = (
            f"{scenario.scenario_type}_{scenario.environment_id}_"
            f"{self.bundle['dynamics']}_{METHOD}_six_stage.png"
        )
        output_dir = project_root() / "results" / "previews"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = filedialog.asksaveasfilename(
            title="导出一行六列 PNG",
            initialdir=output_dir,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if not filename:
            return
        output_path = Path(filename).with_suffix(".png")
        display_size = self.figure.get_size_inches().copy()
        try:
            self.figure.set_size_inches(
                EXPORT_WIDTH_INCHES, EXPORT_HEIGHT_INCHES, forward=False,
            )
            self.figure.savefig(
                output_path, format="png", dpi=EXPORT_DPI,
                bbox_inches=None, facecolor="white",
            )
        finally:
            self.figure.set_size_inches(display_size, forward=False)
            self.canvas.draw_idle()
        self.status_var.set(f"PNG已导出：{output_path}")


def main():
    root = tk.Tk()
    CorridorPreviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
