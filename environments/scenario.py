from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import warnings
import numpy as np
from geometry.obstacles import obstacle_from_dict
from .trajectory import trajectory_from_dict


@dataclass
class DynamicObstacleSpec:
    shape: object
    trajectory: object

    def obstacle_at(self, t):
        return self.shape.moved_to(self.trajectory.position(t))

    def to_dict(self):
        return {
            "shape": self.shape.to_dict(),
            "trajectory": self.trajectory.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            obstacle_from_dict(d["shape"]),
            trajectory_from_dict(d["trajectory"]),
        )


@dataclass
class Scenario:
    scenario_type: str
    seed: int
    width: float
    height: float
    start: np.ndarray
    goal: np.ndarray
    static_obstacles: list
    dynamic_specs: list
    subtype: str = ""
    environment_id: str = ""

    def dynamic_obstacles(self, t):
        return [x.obstacle_at(t) for x in self.dynamic_specs]

    def to_dict(self):
        return {
            "scenario_type": self.scenario_type,
            "subtype": self.subtype,
            "environment_id": self.environment_id,
            "seed": int(self.seed),
            "width": float(self.width),
            "height": float(self.height),
            "start": np.asarray(self.start).tolist(),
            "goal": np.asarray(self.goal).tolist(),
            "static_obstacles": [o.to_dict() for o in self.static_obstacles],
            "dynamic_obstacles": [d.to_dict() for d in self.dynamic_specs],
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, d):
        return cls(
            scenario_type=d["scenario_type"],
            seed=int(d.get("seed", 0)),
            width=float(d["width"]),
            height=float(d["height"]),
            start=np.asarray(d["start"], float),
            goal=np.asarray(d["goal"], float),
            static_obstacles=[obstacle_from_dict(x) for x in d["static_obstacles"]],
            dynamic_specs=[
                DynamicObstacleSpec.from_dict(x) for x in d.get("dynamic_obstacles", [])
            ],
            subtype=d.get("subtype", ""),
            environment_id=d.get("environment_id", ""),
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        obj = cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        file_id = path.stem
        if obj.environment_id and obj.environment_id != file_id:
            warnings.warn(
                f"environment_id={obj.environment_id!r} does not match "
                f"file name {file_id!r}; using the file name for result provenance.",
                RuntimeWarning,
                stacklevel=2,
            )
        obj.environment_id = file_id
        return obj
