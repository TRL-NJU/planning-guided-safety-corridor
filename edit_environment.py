from __future__ import annotations



from copy import deepcopy
from pathlib import Path
import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle as MplCircle

from core.config import (
    load_config,
    load_run_config,
    project_root,
    resolve_project_path,
    rho_from_config,
)
from geometry.obstacles import CircleObstacle, SuperellipseObstacle
from geometry.grid import rasterize_obstacles
from environments.feasibility import grid_connectivity_feasible
from environments.scenario import Scenario, DynamicObstacleSpec
from core.constants import BENCHMARK_SCENARIOS, DYNAMICS, METHODS
from environments.trajectory import PingPongTrajectory, PiecewiseLinearTrajectory
from planning.jps_wrapper import JPSWrapper
from simulation.workspace import wall_clearance, apply_wall_occupancy
from visualization.style import COLORS, apply_publication_style

apply_publication_style()


SCENARIO_DESCRIPTIONS = {
    "S1": "Static-Obstacle Navigation",
    "S2": "Mixed Static-Dynamic Navigation",
}


class EnvironmentEditor:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Experiment I Environment Editor")
        self.root.minsize(1180, 760)

        run_cfg = load_run_config()
        editor_cfg = run_cfg.get("editor", {})
        self.cfg = load_config(
            resolve_project_path(editor_cfg.get("config", "config/benchmark.yaml"))
        )
        self.save_root = resolve_project_path(
            editor_cfg.get("save_root", "data/custom_environments")
        )
        for s in ("S1", "S2"):
            (self.save_root / s).mkdir(parents=True, exist_ok=True)

        self.width = float(self.cfg["workspace"]["width"])
        self.height = float(self.cfg["workspace"]["height"])
        self.robot_radius = float(self.cfg["robot"]["radius"])
        self.safe_margin = float(self.cfg["robot"]["safety_margin"])
        self.rho = rho_from_config(self.cfg)

        # Editable state.
        self.start: np.ndarray | None = None
        self.goal: np.ndarray | None = None
        self.static_obstacles: list = []
        self.dynamic_items: list[dict] = []  # {a,b,radius,speed}
        self.current_file: Path | None = None
        self.loaded_scenario_type: str | None = None

        self.selected = None
        self.drag_target = None
        self.pending_dynamic_a: np.ndarray | None = None
        self.history: list[dict] = []

        # Dynamic preview state.
        self.preview_running = False
        self.preview_time = 0.0
        self.preview_after_id = None

        self._build_vars()
        self._build_ui()
        self._connect_canvas_events()
        self._refresh_all()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_vars(self):
        self.scenario_var = tk.StringVar(value="S1")
        self.mode_var = tk.StringVar(value="select")
        self.shape_var = tk.StringVar(value="Circle")
        self.motion_var = tk.StringVar(value="Static")

        self.x_var = tk.StringVar(value="5.0")
        self.y_var = tk.StringVar(value="5.0")
        self.radius_var = tk.StringVar(value="0.60")
        self.a_var = tk.StringVar(value="1.20")
        self.b_var = tk.StringVar(value="0.60")
        self.angle_var = tk.StringVar(value="0.0")
        self.bx_var = tk.StringVar(value="8.0")
        self.by_var = tk.StringVar(value="5.0")
        self.speed_var = tk.StringVar(value="0.30")

        self.status_var = tk.StringVar(value="Ready")
        self.count_var = tk.StringVar(value="")
        self.selection_var = tk.StringVar(value="No selection")
        self.file_var = tk.StringVar(value="Unsaved environment")
        self.scenario_desc_var = tk.StringVar(value=SCENARIO_DESCRIPTIONS["S1"])

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # ---------------- Left control panel ----------------
        left = ttk.Frame(outer, padding=(0, 0, 8, 0))
        left.grid(row=0, column=0, sticky="ns")

        scenario_box = ttk.LabelFrame(left, text="Scenario", padding=8)
        scenario_box.pack(fill="x", pady=(0, 8))
        ttk.Combobox(
            scenario_box,
            textvariable=self.scenario_var,
            values=["S1", "S2"],
            state="readonly",
            width=8,
        ).pack(side="left")
        ttk.Label(
            scenario_box, textvariable=self.scenario_desc_var, wraplength=195
        ).pack(side="left", padx=(8, 0))
        self.scenario_var.trace_add("write", lambda *_: self._scenario_changed())

        params = ttk.LabelFrame(left, text="Fixed experiment parameters", padding=8)
        params.pack(fill="x", pady=(0, 8))
        fixed_lines = [
            f"Workspace: {self.width:.1f} x {self.height:.1f} m",
            f"Robot radius: {self.robot_radius:.2f} m",
            f"Safety margin: {self.safe_margin:.2f} m",
            f"rho = r_robot + d_safe: {self.rho:.2f} m",
            f"Grid resolution: {float(self.cfg['workspace']['grid_resolution']):.2f} m",
        ]
        for line in fixed_lines:
            ttk.Label(params, text=line).pack(anchor="w")

        modes = ttk.LabelFrame(left, text="Canvas mode", padding=8)
        modes.pack(fill="x", pady=(0, 8))
        for text, value in [
            ("Select / Move", "select"),
            ("Set Start", "start"),
            ("Set Goal", "goal"),
            ("Add Obstacle", "add"),
        ]:
            ttk.Radiobutton(modes, text=text, variable=self.mode_var, value=value).pack(
                anchor="w"
            )

        create = ttk.LabelFrame(left, text="New obstacle", padding=8)
        create.pack(fill="x", pady=(0, 8))
        ttk.Label(create, text="Shape").grid(row=0, column=0, sticky="w")
        self.shape_combo = ttk.Combobox(
            create,
            textvariable=self.shape_var,
            values=["Circle", "Rectangle"],
            state="readonly",
            width=14,
        )
        self.shape_combo.grid(row=0, column=1, padx=(6, 0), pady=2)
        ttk.Label(create, text="Motion").grid(row=1, column=0, sticky="w")
        self.motion_combo = ttk.Combobox(
            create,
            textvariable=self.motion_var,
            values=["Static", "Dynamic"],
            state="readonly",
            width=14,
        )
        self.motion_combo.grid(row=1, column=1, padx=(6, 0), pady=2)
        self.shape_var.trace_add("write", lambda *_: self._shape_changed())

        ttk.Label(
            create,
            text=("Dynamic Circle: click A, then click B.\nRectangle is static only."),
            wraplength=210,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        objects = ttk.LabelFrame(left, text="Objects", padding=8)
        objects.pack(fill="both", expand=True, pady=(0, 8))
        self.object_list = tk.Listbox(objects, height=12, exportselection=False)
        self.object_list.pack(fill="both", expand=True)
        self.object_list.bind("<<ListboxSelect>>", self._listbox_selected)

        row = ttk.Frame(left)
        row.pack(fill="x", pady=(0, 6))
        ttk.Button(row, text="Delete", command=self.delete_selected).pack(
            side="left", expand=True, fill="x"
        )
        ttk.Button(row, text="Undo", command=self.undo).pack(
            side="left", expand=True, fill="x", padx=4
        )
        ttk.Button(row, text="Clear All", command=self.clear_all).pack(
            side="left", expand=True, fill="x"
        )

        preview_row = ttk.Frame(left)
        preview_row.pack(fill="x", pady=(0, 8))
        self.preview_button = ttk.Button(
            preview_row, text="Preview Dynamic Motion", command=self.toggle_preview
        )
        self.preview_button.pack(fill="x")

        # ---------------- Center matplotlib canvas ----------------
        center = ttk.Frame(outer)
        center.grid(row=0, column=1, sticky="nsew")
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(8.0, 7.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # ---------------- Right properties panel ----------------
        right = ttk.Frame(outer, padding=(8, 0, 0, 0))
        right.grid(row=0, column=2, sticky="ns")

        prop = ttk.LabelFrame(right, text="Selected object properties", padding=8)
        prop.pack(fill="x", pady=(0, 8))
        ttk.Label(
            prop, textvariable=self.selection_var, font=("TkDefaultFont", 10, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

        fields = [
            ("x / Ax", self.x_var),
            ("y / Ay", self.y_var),
            ("Circle radius", self.radius_var),
            ("Rect half-length a", self.a_var),
            ("Rect half-width b", self.b_var),
            ("Rect angle [deg]", self.angle_var),
            ("Dynamic Bx", self.bx_var),
            ("Dynamic By", self.by_var),
            ("Dynamic speed [m/s]", self.speed_var),
        ]
        self.property_entries = {}
        for r, (label, var) in enumerate(fields, start=1):
            ttk.Label(prop, text=label).grid(row=r, column=0, sticky="w", pady=2)
            e = ttk.Entry(prop, textvariable=var, width=13)
            e.grid(row=r, column=1, sticky="ew", padx=(8, 0), pady=2)
            self.property_entries[label] = e

        ttk.Button(prop, text="Apply Properties", command=self.apply_properties).grid(
            row=len(fields) + 1, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        validate_box = ttk.LabelFrame(right, text="Validation / Saving", padding=8)
        validate_box.pack(fill="x", pady=(0, 8))
        ttk.Button(
            validate_box, text="Validate Environment", command=self.validate_from_ui
        ).pack(fill="x", pady=2)
        ttk.Button(
            validate_box, text="Save New Environment", command=self.save_new
        ).pack(fill="x", pady=2)
        ttk.Button(
            validate_box, text="Overwrite Current", command=self.overwrite_current
        ).pack(fill="x", pady=2)
        ttk.Button(
            validate_box, text="Open Environment...", command=self.open_environment
        ).pack(fill="x", pady=2)
        ttk.Button(
            validate_box, text="New Blank Environment", command=self.new_blank
        ).pack(fill="x", pady=2)

        info = ttk.LabelFrame(right, text="Current file / experiment count", padding=8)
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.file_var, wraplength=245).pack(anchor="w")
        ttk.Separator(info).pack(fill="x", pady=6)
        ttk.Label(info, textvariable=self.count_var, wraplength=245).pack(anchor="w")
        ttk.Label(
            info,
            text=f"Formal benchmark: S1/S2 only, {len(DYNAMICS)} dynamics x {len(METHODS)} methods.",
            wraplength=245,
        ).pack(anchor="w", pady=(5, 0))

        status = ttk.Label(
            self.root, textvariable=self.status_var, relief="sunken", anchor="w"
        )
        status.pack(fill="x", side="bottom")

    def _connect_canvas_events(self):
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    # ------------------------------------------------------------------
    # State snapshots / undo
    # ------------------------------------------------------------------
    def _snapshot(self):
        return {
            "start": None if self.start is None else self.start.copy(),
            "goal": None if self.goal is None else self.goal.copy(),
            "static": [deepcopy(o) for o in self.static_obstacles],
            "dynamic": [
                {
                    "a": x["a"].copy(),
                    "b": x["b"].copy(),
                    "radius": float(x["radius"]),
                    "speed": float(x["speed"]),
                }
                for x in self.dynamic_items
            ],
            "scenario": self.scenario_var.get(),
        }

    def _restore(self, snap):
        self.start = None if snap["start"] is None else snap["start"].copy()
        self.goal = None if snap["goal"] is None else snap["goal"].copy()
        self.static_obstacles = [deepcopy(o) for o in snap["static"]]
        self.dynamic_items = [
            {
                "a": x["a"].copy(),
                "b": x["b"].copy(),
                "radius": float(x["radius"]),
                "speed": float(x["speed"]),
            }
            for x in snap["dynamic"]
        ]
        self.scenario_var.set(snap["scenario"])
        self.selected = None
        self.pending_dynamic_a = None
        self._refresh_all()

    def _push_history(self):
        self.history.append(self._snapshot())
        if len(self.history) > 60:
            self.history.pop(0)

    def undo(self):
        if not self.history:
            self.status_var.set("Nothing to undo.")
            return
        snap = self.history.pop()
        self._restore(snap)
        self.status_var.set("Undo complete.")

    # ------------------------------------------------------------------
    # Canvas interactions
    # ------------------------------------------------------------------
    def _point_from_event(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        return np.array(
            [
                np.clip(event.xdata, 0.0, self.width),
                np.clip(event.ydata, 0.0, self.height),
            ],
            float,
        )

    def _on_press(self, event):
        if event.button != 1:
            return
        p = self._point_from_event(event)
        if p is None:
            return
        mode = self.mode_var.get()

        if mode == "start":
            self._push_history()
            self.start = p
            self.selected = ("start", None)
            self._set_properties_from_selection()
            self._refresh_all()
            return

        if mode == "goal":
            self._push_history()
            self.goal = p
            self.selected = ("goal", None)
            self._set_properties_from_selection()
            self._refresh_all()
            return

        if mode == "add":
            self._add_click(p)
            return

        # select / move
        target = self._nearest_target(p)
        self.selected = target
        if target is not None:
            self._push_history()
            self.drag_target = target
            self._set_properties_from_selection()
        self._refresh_all()

    def _on_motion(self, event):
        if self.drag_target is None:
            return
        p = self._point_from_event(event)
        if p is None:
            return
        typ, idx = self.drag_target
        if typ == "start":
            self.start = p
        elif typ == "goal":
            self.goal = p
        elif typ == "static":
            self.static_obstacles[idx].center = p
        elif typ == "dynamic_a":
            self.dynamic_items[idx]["a"] = p
        elif typ == "dynamic_b":
            self.dynamic_items[idx]["b"] = p
        self._set_properties_from_selection()
        self._draw()

    def _on_release(self, event):
        if self.drag_target is not None:
            self.drag_target = None
            self._refresh_object_list()

    def _add_click(self, p):
        shape = self.shape_var.get()
        motion = self.motion_var.get()

        try:
            radius = float(self.radius_var.get())
            a = float(self.a_var.get())
            b = float(self.b_var.get())
            angle = math.radians(float(self.angle_var.get()))
            speed = float(self.speed_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid parameter", "Obstacle properties must be numeric."
            )
            return

        if shape == "Rectangle":
            if a <= 0 or b <= 0:
                messagebox.showerror(
                    "Invalid parameter", "Rectangle half sizes must be positive."
                )
                return
            self._push_history()
            self.static_obstacles.append(SuperellipseObstacle(p, a, b, 8, angle))
            self.selected = ("static", len(self.static_obstacles) - 1)
            self._set_properties_from_selection()
            self._refresh_all()
            return

        if radius <= 0:
            messagebox.showerror("Invalid parameter", "Circle radius must be positive.")
            return

        if motion == "Static":
            self._push_history()
            self.static_obstacles.append(CircleObstacle(p, radius))
            self.selected = ("static", len(self.static_obstacles) - 1)
            self._set_properties_from_selection()
            self._refresh_all()
            return

        # Dynamic circle: two clicks A then B.
        if self.pending_dynamic_a is None:
            self.pending_dynamic_a = p
            self.status_var.set("Dynamic Circle: endpoint A set. Click endpoint B.")
            self._draw()
            return

        if speed <= 0:
            messagebox.showerror("Invalid parameter", "Dynamic speed must be positive.")
            return
        if np.linalg.norm(p - self.pending_dynamic_a) < 0.05:
            messagebox.showerror(
                "Invalid trajectory", "Dynamic endpoints A and B must be different."
            )
            return

        self._push_history()
        self.dynamic_items.append(
            {
                "a": self.pending_dynamic_a.copy(),
                "b": p.copy(),
                "radius": radius,
                "speed": speed,
            }
        )
        self.pending_dynamic_a = None
        self.selected = ("dynamic_a", len(self.dynamic_items) - 1)
        self._set_properties_from_selection()
        self._refresh_all()

    def _nearest_target(self, p):
        candidates = []
        if self.start is not None:
            candidates.append((np.linalg.norm(p - self.start), ("start", None)))
        if self.goal is not None:
            candidates.append((np.linalg.norm(p - self.goal), ("goal", None)))
        for i, o in enumerate(self.static_obstacles):
            candidates.append((np.linalg.norm(p - o.center), ("static", i)))
        for i, d in enumerate(self.dynamic_items):
            candidates.append((np.linalg.norm(p - d["a"]), ("dynamic_a", i)))
            candidates.append((np.linalg.norm(p - d["b"]), ("dynamic_b", i)))
        if not candidates:
            return None
        dist, target = min(candidates, key=lambda x: x[0])
        threshold = max(0.45, 0.025 * max(self.width, self.height))
        return target if dist <= threshold else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def _set_properties_from_selection(self):
        if self.selected is None:
            self.selection_var.set("No selection")
            return
        typ, idx = self.selected
        if typ == "start":
            self.selection_var.set("Start")
            self.x_var.set(f"{self.start[0]:.3f}")
            self.y_var.set(f"{self.start[1]:.3f}")
        elif typ == "goal":
            self.selection_var.set("Goal")
            self.x_var.set(f"{self.goal[0]:.3f}")
            self.y_var.set(f"{self.goal[1]:.3f}")
        elif typ == "static":
            o = self.static_obstacles[idx]
            self.x_var.set(f"{o.center[0]:.3f}")
            self.y_var.set(f"{o.center[1]:.3f}")
            if o.name == "circle":
                self.selection_var.set(f"Static Circle #{idx + 1}")
                self.radius_var.set(f"{o.radius:.3f}")
            else:
                self.selection_var.set(f"Static Rectangle #{idx + 1}")
                self.a_var.set(f"{o.a:.3f}")
                self.b_var.set(f"{o.b:.3f}")
                self.angle_var.set(f"{math.degrees(o.angle):.2f}")
        elif typ in ("dynamic_a", "dynamic_b"):
            d = self.dynamic_items[idx]
            self.selection_var.set(
                f"Dynamic Circle #{idx + 1} ({'A' if typ == 'dynamic_a' else 'B'})"
            )
            self.x_var.set(f"{d['a'][0]:.3f}")
            self.y_var.set(f"{d['a'][1]:.3f}")
            self.bx_var.set(f"{d['b'][0]:.3f}")
            self.by_var.set(f"{d['b'][1]:.3f}")
            self.radius_var.set(f"{d['radius']:.3f}")
            self.speed_var.set(f"{d['speed']:.3f}")

    def apply_properties(self):
        if self.selected is None:
            messagebox.showinfo("No selection", "Select an object first.")
            return
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            radius = float(self.radius_var.get())
            a = float(self.a_var.get())
            b = float(self.b_var.get())
            angle = math.radians(float(self.angle_var.get()))
            bx = float(self.bx_var.get())
            by = float(self.by_var.get())
            speed = float(self.speed_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid parameter", "All property values must be numeric."
            )
            return

        if radius <= 0 or a <= 0 or b <= 0 or speed <= 0:
            messagebox.showerror(
                "Invalid parameter",
                "Radius, rectangle sizes and speed must be positive.",
            )
            return

        self._push_history()
        p = np.array([x, y], float)
        typ, idx = self.selected
        if typ == "start":
            self.start = p
        elif typ == "goal":
            self.goal = p
        elif typ == "static":
            o = self.static_obstacles[idx]
            o.center = p
            if o.name == "circle":
                o.radius = radius
            else:
                o.a, o.b, o.angle = a, b, angle
        elif typ in ("dynamic_a", "dynamic_b"):
            d = self.dynamic_items[idx]
            d["a"] = p
            d["b"] = np.array([bx, by], float)
            d["radius"] = radius
            d["speed"] = speed
        self._refresh_all()

    # ------------------------------------------------------------------
    # Object list / deletion
    # ------------------------------------------------------------------
    def _refresh_object_list(self):
        self.object_list.delete(0, tk.END)
        self._list_targets = []
        if self.start is not None:
            self.object_list.insert(
                tk.END, f"Start ({self.start[0]:.2f}, {self.start[1]:.2f})"
            )
            self._list_targets.append(("start", None))
        if self.goal is not None:
            self.object_list.insert(
                tk.END, f"Goal ({self.goal[0]:.2f}, {self.goal[1]:.2f})"
            )
            self._list_targets.append(("goal", None))
        for i, o in enumerate(self.static_obstacles):
            label = "Circle" if o.name == "circle" else "Rectangle"
            self.object_list.insert(tk.END, f"Static {label} #{i + 1}")
            self._list_targets.append(("static", i))
        for i, d in enumerate(self.dynamic_items):
            self.object_list.insert(
                tk.END,
                f"Dynamic Circle #{i + 1}  v={d['speed']:.2f} m/s",
            )
            self._list_targets.append(("dynamic_a", i))

    def _listbox_selected(self, _event=None):
        sel = self.object_list.curselection()
        if not sel:
            return
        self.selected = self._list_targets[sel[0]]
        self.mode_var.set("select")
        self._set_properties_from_selection()
        self._draw()

    def delete_selected(self):
        if self.selected is None:
            return
        self._push_history()
        typ, idx = self.selected
        if typ == "start":
            self.start = None
        elif typ == "goal":
            self.goal = None
        elif typ == "static":
            del self.static_obstacles[idx]
        elif typ in ("dynamic_a", "dynamic_b"):
            del self.dynamic_items[idx]
        self.selected = None
        self._refresh_all()

    def clear_all(self):
        if not messagebox.askyesno(
            "Clear all", "Delete Start, Goal and all obstacles?"
        ):
            return
        self._push_history()
        self.start = None
        self.goal = None
        self.static_obstacles = []
        self.dynamic_items = []
        self.selected = None
        self.pending_dynamic_a = None
        self._refresh_all()

    # ------------------------------------------------------------------
    # Drawing / dynamic preview
    # ------------------------------------------------------------------
    def _draw_boundary(
        self, obs, clearance, linestyle, linewidth=1.2, alpha=0.8, color=None
    ):
        pts = obs.boundary_points(0.06, clearance=clearance)
        q = np.vstack((pts, pts[0]))
        self.ax.plot(
            q[:, 0],
            q[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

    def _dynamic_position(self, item, t):
        tr = PingPongTrajectory(item["a"], item["b"], item["speed"])
        return tr.position(t)

    def _draw(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_title(
            f"{self.scenario_var.get()} - {SCENARIO_DESCRIPTIONS[self.scenario_var.get()]}"
        )
        self.ax.set_facecolor("white")
        self.ax.grid(
            True,
            color=COLORS["grid"],
            linewidth=0.5,
            linestyle=(0, (2, 2)),
            alpha=0.55,
        )

        # Static obstacles: physical + rho-expanded safety boundary.
        for i, o in enumerate(self.static_obstacles):
            pts = o.boundary_points(0.06, 0.0)
            q = np.vstack((pts, pts[0]))
            self.ax.fill(q[:, 0], q[:, 1], color=COLORS["static_fill"], alpha=0.75)
            self.ax.plot(q[:, 0], q[:, 1], color=COLORS["static_edge"], linewidth=1.3)
            self._draw_boundary(
                o, self.rho, "--", linewidth=1.0, alpha=0.75, color=COLORS["sky"]
            )
            self.ax.text(
                o.center[0],
                o.center[1],
                f"S{i + 1}",
                ha="center",
                va="center",
                fontsize=8,
            )

        # Dynamic circles: A-B trajectory plus physical/safety circle at preview t.
        for i, d in enumerate(self.dynamic_items):
            a, b = d["a"], d["b"]
            self.ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color=COLORS["orange"],
                linestyle=(0, (5, 2)),
                linewidth=1.2,
                alpha=0.9,
            )
            self.ax.scatter(
                [a[0]],
                [a[1]],
                marker="o",
                s=35,
                color=COLORS["yellow"],
                edgecolors=COLORS["orange"],
            )
            self.ax.scatter(
                [b[0]], [b[1]], marker="x", s=45, color=COLORS["vermillion"]
            )
            self.ax.text(a[0], a[1], f" A{i + 1}", fontsize=8)
            self.ax.text(b[0], b[1], f" B{i + 1}", fontsize=8)
            pos = self._dynamic_position(
                d, self.preview_time if self.preview_running else 0.0
            )
            physical = MplCircle(
                pos,
                d["radius"],
                facecolor=COLORS["yellow"],
                edgecolor=COLORS["orange"],
                linewidth=1.2,
                alpha=0.45,
            )
            safe = MplCircle(
                pos,
                d["radius"] + self.rho,
                fill=False,
                edgecolor=COLORS["sky"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )
            self.ax.add_patch(physical)
            self.ax.add_patch(safe)

        if self.pending_dynamic_a is not None:
            a = self.pending_dynamic_a
            self.ax.scatter([a[0]], [a[1]], s=80, marker="o")
            self.ax.text(a[0], a[1], " Dynamic A (click B)", fontsize=9)

        if self.start is not None:
            self.ax.scatter(
                [self.start[0]],
                [self.start[1]],
                marker="o",
                s=90,
                color=COLORS["green"],
                edgecolors="white",
                linewidths=0.8,
                zorder=10,
            )
            self.ax.add_patch(
                MplCircle(self.start, self.robot_radius, fill=False, linewidth=1.5)
            )
            self.ax.add_patch(
                MplCircle(
                    self.start, self.rho, fill=False, linestyle=":", linewidth=1.0
                )
            )
            self.ax.text(
                self.start[0], self.start[1], " Start", fontsize=9, va="bottom"
            )
        if self.goal is not None:
            self.ax.scatter(
                [self.goal[0]],
                [self.goal[1]],
                marker="*",
                s=130,
                color=COLORS["yellow"],
                edgecolors=COLORS["black"],
                linewidths=0.7,
                zorder=10,
            )
            self.ax.text(self.goal[0], self.goal[1], " Goal", fontsize=9, va="bottom")

        # Selection ring / marker.
        if self.selected is not None:
            typ, idx = self.selected
            p = None
            if typ == "start" and self.start is not None:
                p = self.start
            elif typ == "goal" and self.goal is not None:
                p = self.goal
            elif typ == "static" and idx < len(self.static_obstacles):
                p = self.static_obstacles[idx].center
            elif typ == "dynamic_a" and idx < len(self.dynamic_items):
                p = self.dynamic_items[idx]["a"]
            elif typ == "dynamic_b" and idx < len(self.dynamic_items):
                p = self.dynamic_items[idx]["b"]
            if p is not None:
                self.ax.scatter(
                    [p[0]],
                    [p[1]],
                    s=180,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=20,
                )

        self.canvas.draw_idle()

    def toggle_preview(self):
        if not self.dynamic_items:
            messagebox.showinfo(
                "No dynamic obstacles", "There are no dynamic circles to preview."
            )
            return
        self.preview_running = not self.preview_running
        if self.preview_running:
            self.preview_time = 0.0
            self.preview_button.configure(text="Stop Dynamic Preview")
            self._preview_tick()
        else:
            self._stop_preview()

    def _preview_tick(self):
        if not self.preview_running:
            return
        self.preview_time += 0.05
        self._draw()
        self.preview_after_id = self.root.after(50, self._preview_tick)

    def _stop_preview(self):
        self.preview_running = False
        self.preview_time = 0.0
        self.preview_button.configure(text="Preview Dynamic Motion")
        if self.preview_after_id is not None:
            try:
                self.root.after_cancel(self.preview_after_id)
            except tk.TclError:
                pass
            self.preview_after_id = None
        self._draw()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _obstacle_inside_workspace(self, o):
        if o.name == "circle":
            r = o.radius
            return (
                r <= o.center[0] <= self.width - r
                and r <= o.center[1] <= self.height - r
            )
        # Conservative rotated rectangle bound for a high-power superellipse.
        c, s = abs(math.cos(o.angle)), abs(math.sin(o.angle))
        ex = c * o.a + s * o.b
        ey = s * o.a + c * o.b
        return (
            ex <= o.center[0] <= self.width - ex
            and ey <= o.center[1] <= self.height - ey
        )

    def _true_collision_at_point(self, p):
        # Robot is a disk: true collision is obstacle geometry inflated by robot radius.
        if (
            self.cfg["workspace"].get("boundary_is_wall", False)
            and wall_clearance(p, self.width, self.height, self.robot_radius) <= 0
        ):
            return True
        for o in self.static_obstacles:
            if o.contains(p, clearance=self.robot_radius):
                return True
        for d in self.dynamic_items:
            o = CircleObstacle(d["a"], d["radius"])
            if o.contains(p, clearance=self.robot_radius):
                return True
        return False

    def _segment_point_distance(self, p, a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        p = np.asarray(p, float)
        d = b - a
        den = float(d @ d)
        if den <= 1e-15:
            return float(np.linalg.norm(p - a))
        u = float(np.clip((p - a) @ d / den, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + u * d)))

    def _segments_distance(self, a0, a1, b0, b1):
        # 2-D segment intersection test; otherwise endpoint-to-segment minimum.
        def orient(a, b, c):
            return float(np.cross(b - a, c - a))

        a0 = np.asarray(a0, float)
        a1 = np.asarray(a1, float)
        b0 = np.asarray(b0, float)
        b1 = np.asarray(b1, float)
        o1, o2 = orient(a0, a1, b0), orient(a0, a1, b1)
        o3, o4 = orient(b0, b1, a0), orient(b0, b1, a1)
        if o1 * o2 <= 0 and o3 * o4 <= 0:
            return 0.0
        return min(
            self._segment_point_distance(a0, b0, b1),
            self._segment_point_distance(a1, b0, b1),
            self._segment_point_distance(b0, a0, a1),
            self._segment_point_distance(b1, a0, a1),
        )

    def _jps_crossing_warnings(self):
        warnings = []
        if not self.dynamic_items or self.start is None or self.goal is None:
            return warnings
        try:
            grid = rasterize_obstacles(
                self.static_obstacles,
                self.width,
                self.height,
                float(self.cfg["workspace"]["grid_resolution"]),
                self.rho,
            )
            if self.cfg["workspace"].get("boundary_is_wall", False):
                apply_wall_occupancy(
                    grid,
                    self.width,
                    self.height,
                    float(self.cfg["workspace"]["grid_resolution"]),
                    self.rho,
                )
            planner = JPSWrapper(
                self.width,
                self.height,
                float(self.cfg["workspace"]["grid_resolution"]),
                verbose=False,
            )
            result = planner.plan(grid, self.start, self.goal)
            if not result.success:
                return warnings
            path = np.asarray(result.path, float)
            for i, d in enumerate(self.dynamic_items):
                min_dist = np.inf
                for j in range(len(path) - 1):
                    min_dist = min(
                        min_dist,
                        self._segments_distance(d["a"], d["b"], path[j], path[j + 1]),
                    )
                # Warning only; this is not a feasibility filter.
                if min_dist <= d["radius"] + self.rho:
                    warnings.append(
                        f"Dynamic Circle #{i + 1} trajectory intersects or closely crosses the initial JPS route."
                    )
        except Exception as exc:
            warnings.append(
                "JPS crossing warning could not be evaluated; saving is still allowed. "
                f"Reason: {type(exc).__name__}: {exc}"
            )
        return warnings

    def validate_environment(self):
        errors = []
        warnings = []

        if self.start is None:
            errors.append("Start has not been set.")
        if self.goal is None:
            errors.append("Goal has not been set.")

        if self.start is not None:
            if not (
                self.robot_radius <= self.start[0] <= self.width - self.robot_radius
                and self.robot_radius
                <= self.start[1]
                <= self.height - self.robot_radius
            ):
                errors.append(
                    "Start is too close to or outside the workspace boundary."
                )
        if self.goal is not None:
            if not (
                self.robot_radius <= self.goal[0] <= self.width - self.robot_radius
                and self.robot_radius <= self.goal[1] <= self.height - self.robot_radius
            ):
                errors.append("Goal is too close to or outside the workspace boundary.")

        for i, o in enumerate(self.static_obstacles):
            if not self._obstacle_inside_workspace(o):
                errors.append(
                    f"Static obstacle #{i + 1} extends outside the workspace."
                )

        for i, d in enumerate(self.dynamic_items):
            r = float(d["radius"])
            if r <= 0:
                errors.append(f"Dynamic Circle #{i + 1} radius must be positive.")
            if float(d["speed"]) <= 0:
                errors.append(f"Dynamic Circle #{i + 1} speed must be positive.")
            if np.linalg.norm(d["b"] - d["a"]) <= 1e-8:
                errors.append(f"Dynamic Circle #{i + 1} endpoints A and B must differ.")
            for name, p in (("A", d["a"]), ("B", d["b"])):
                if not (r <= p[0] <= self.width - r and r <= p[1] <= self.height - r):
                    errors.append(
                        f"Dynamic Circle #{i + 1} endpoint {name} places the obstacle outside the workspace."
                    )

        if self.start is not None and self._true_collision_at_point(self.start):
            errors.append("Start is in physical collision with an obstacle.")
        if self.goal is not None and self._true_collision_at_point(self.goal):
            errors.append("Goal is in physical collision with an obstacle.")

        # Independent static feasibility filter with rho inflation.
        if self.start is not None and self.goal is not None and not errors:
            feasible = grid_connectivity_feasible(
                self.static_obstacles,
                self.start,
                self.goal,
                self.width,
                self.height,
                float(self.cfg["workspace"]["feasibility_resolution"]),
                self.rho,
            )
            if feasible and self.cfg["workspace"].get("boundary_is_wall", False):
                resolution = float(self.cfg["workspace"]["feasibility_resolution"])
                grid = rasterize_obstacles(
                    self.static_obstacles, self.width, self.height, resolution, self.rho
                )
                apply_wall_occupancy(
                    grid, self.width, self.height, resolution, self.rho
                )
                feasible = (
                    JPSWrapper(
                        self.width,
                        self.height,
                        resolution,
                        verbose=False,
                        extra_dilation_cells=0,
                    )
                    .plan(grid, self.start, self.goal)
                    .success
                )
            if not feasible:
                errors.append(
                    "After inflating obstacles and physical walls by rho, Start and Goal are not connected."
                )

        scenario = self.scenario_var.get()
        if scenario == "S1" and self.dynamic_items:
            warnings.append(
                "S1 contains dynamic obstacles. This is semantically inconsistent with Static-Obstacle Navigation."
            )
        if scenario == "S2" and not self.dynamic_items:
            warnings.append("S2 contains no dynamic obstacle.")
        warnings.extend(self._jps_crossing_warnings())

        return errors, warnings

    def validate_from_ui(self):
        errors, warnings = self.validate_environment()
        if errors:
            messagebox.showerror(
                "Environment invalid", "\n".join(f"- {e}" for e in errors)
            )
            self.status_var.set(f"Validation failed: {len(errors)} error(s).")
            return False
        msg = "Environment passes mandatory validation."
        if warnings:
            msg += "\n\nWarnings:\n" + "\n".join(f"- {w}" for w in warnings)
        messagebox.showinfo("Validation", msg)
        self.status_var.set(f"Validation passed with {len(warnings)} warning(s).")
        return True

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def _next_path(self, scenario_type):
        folder = self.save_root / scenario_type
        folder.mkdir(parents=True, exist_ok=True)
        nums = []
        for p in folder.glob("environment_*.json"):
            try:
                nums.append(int(p.stem.split("_")[-1]))
            except ValueError:
                pass
        n = max(nums, default=0) + 1
        return folder / f"environment_{n:03d}.json"

    def _scenario_for_path(self, path, seed):
        env_id = Path(path).stem
        dyn_specs = []
        for d in self.dynamic_items:
            tr = PingPongTrajectory(d["a"], d["b"], d["speed"])
            dyn_specs.append(
                DynamicObstacleSpec(
                    shape=CircleObstacle(d["a"], d["radius"]),
                    trajectory=tr,
                )
            )
        return Scenario(
            scenario_type=self.scenario_var.get(),
            seed=int(seed),
            width=self.width,
            height=self.height,
            start=self.start.copy(),
            goal=self.goal.copy(),
            static_obstacles=[deepcopy(o) for o in self.static_obstacles],
            dynamic_specs=dyn_specs,
            subtype="manual",
            environment_id=env_id,
        )

    def _confirm_warnings(self, warnings):
        if not warnings:
            return True
        text = "The environment has semantic/non-blocking warnings:\n\n" + "\n".join(
            f"- {w}" for w in warnings
        )
        text += "\n\nSave anyway?"
        return messagebox.askyesno("Warnings", text)

    def save_new(self):
        errors, warnings = self.validate_environment()
        if errors:
            messagebox.showerror("Cannot save", "\n".join(f"- {e}" for e in errors))
            return
        if not self._confirm_warnings(warnings):
            return

        path = self._next_path(self.scenario_var.get())
        seed = int(path.stem.split("_")[-1])
        scenario = self._scenario_for_path(path, seed)
        scenario.save(path)
        self.current_file = path
        self.loaded_scenario_type = self.scenario_var.get()
        self.file_var.set(str(path.relative_to(project_root())))
        self.status_var.set(f"Saved: {path}")
        self._refresh_counts()
        messagebox.showinfo("Saved", f"Environment saved to:\n{path}")

    def overwrite_current(self):
        if self.current_file is None:
            messagebox.showinfo(
                "No current file",
                "Open an existing environment or use Save New Environment.",
            )
            return
        if self.loaded_scenario_type != self.scenario_var.get():
            messagebox.showwarning(
                "Scenario type changed",
                "The loaded environment's S1/S2 label was changed. Use Save New Environment so the file is placed in the correct category folder.",
            )
            return
        errors, warnings = self.validate_environment()
        if errors:
            messagebox.showerror("Cannot save", "\n".join(f"- {e}" for e in errors))
            return
        if not self._confirm_warnings(warnings):
            return

        try:
            seed = int(self.current_file.stem.split("_")[-1])
        except ValueError:
            seed = 0
        scenario = self._scenario_for_path(self.current_file, seed)
        scenario.save(self.current_file)
        self.status_var.set(f"Overwritten: {self.current_file}")
        self._refresh_counts()
        messagebox.showinfo("Saved", f"Environment overwritten:\n{self.current_file}")

    def open_environment(self):
        path = filedialog.askopenfilename(
            title="Open saved environment",
            initialdir=str(self.save_root),
            filetypes=[("Environment JSON", "*.json")],
        )
        if not path:
            return
        path = Path(path)
        try:
            sc = Scenario.load(path)
        except Exception as exc:
            messagebox.showerror("Open failed", f"{type(exc).__name__}: {exc}")
            return

        if abs(sc.width - self.width) > 1e-9 or abs(sc.height - self.height) > 1e-9:
            messagebox.showerror(
                "Workspace mismatch",
                "This environment was created with a different workspace size. Common workspace parameters are fixed by config/benchmark.yaml.",
            )
            return

        self._push_history()
        self.start = sc.start.copy()
        self.goal = sc.goal.copy()
        self.static_obstacles = [deepcopy(o) for o in sc.static_obstacles]
        self.dynamic_items = []

        converted_legacy = False
        for spec in sc.dynamic_specs:
            if spec.shape.name != "circle":
                messagebox.showerror(
                    "Unsupported dynamic shape",
                    "The editor supports dynamic circles only.",
                )
                return
            tr = spec.trajectory
            if isinstance(tr, PingPongTrajectory):
                a, b, speed = tr.a.copy(), tr.b.copy(), float(tr.speed)
            elif isinstance(tr, PiecewiseLinearTrajectory):
                converted_legacy = True
                a = tr.positions[0].copy()
                b = tr.positions[-1].copy()
                distance = np.linalg.norm(b - a)
                duration = max(float(tr.times[-1] - tr.times[0]), 1e-6)
                speed = max(distance / duration, 0.1)
            else:
                raise TypeError(type(tr))
            self.dynamic_items.append(
                {"a": a, "b": b, "radius": float(spec.shape.radius), "speed": speed}
            )

        self.scenario_var.set(sc.scenario_type)
        self.current_file = path
        self.loaded_scenario_type = sc.scenario_type
        self.file_var.set(str(path))
        self.selected = None
        self.pending_dynamic_a = None
        self.history.clear()
        self._refresh_all()
        if converted_legacy:
            messagebox.showwarning(
                "Legacy trajectory converted",
                "A legacy piecewise dynamic trajectory was converted to an A-B ping-pong trajectory for editing. Use Save New if you want to keep the original file unchanged.",
            )

    def new_blank(self):
        if not messagebox.askyesno(
            "New environment", "Clear the editor and start a new unsaved environment?"
        ):
            return
        self._stop_preview()
        self.start = None
        self.goal = None
        self.static_obstacles = []
        self.dynamic_items = []
        self.current_file = None
        self.loaded_scenario_type = None
        self.selected = None
        self.pending_dynamic_a = None
        self.history.clear()
        self.file_var.set("Unsaved environment")
        self._refresh_all()

    # ------------------------------------------------------------------
    # Misc UI updates
    # ------------------------------------------------------------------
    def _scenario_changed(self):
        self.scenario_desc_var.set(SCENARIO_DESCRIPTIONS[self.scenario_var.get()])
        self._draw()

    def _shape_changed(self):
        if self.shape_var.get() == "Rectangle":
            self.motion_var.set("Static")
            self.motion_combo.configure(state="disabled")
        else:
            self.motion_combo.configure(state="readonly")

    def _refresh_counts(self):
        counts = {}
        total = 0
        for s in ("S1", "S2"):
            n = len(list((self.save_root / s).glob("environment_*.json")))
            counts[s] = n
            total += n
        formal_total = sum(counts[scenario] for scenario in BENCHMARK_SCENARIOS)
        trials = formal_total * len(DYNAMICS) * len(METHODS)
        self.count_var.set(
            f"Saved environments: N={total}\n"
            f"S1={counts['S1']}, S2={counts['S2']}\n"
            f"Formal paired trials: {formal_total} x {len(DYNAMICS)} x {len(METHODS)} = {trials}"
        )

    def _refresh_all(self):
        self._refresh_object_list()
        self._refresh_counts()
        self._set_properties_from_selection()
        self._draw()


def main():
    root = tk.Tk()
    app = EnvironmentEditor(root)

    def on_close():
        app._stop_preview()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
