from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os

import numpy as np


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_save_npz(path: Path, arrays: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


@dataclass
class TrialLogger:
    rows: list = field(default_factory=list)
    events: list = field(default_factory=list)
    high_level: list = field(default_factory=list)

    def log_step(self, **kwargs):
        self.rows.append(kwargs)

    def event(self, t, kind, **kwargs):
        self.events.append({"t": float(t), "kind": kind, **kwargs})

    def log_high_level(self, **kwargs):
        self.high_level.append(kwargs)

    def arrays(self):
        if not self.rows:
            return {}
        return {
            key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]
        }

    def save(self, out_dir, summary, save_timeseries=True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            out / "events.json",
            json.dumps(self.events, ensure_ascii=False, indent=2),
        )
        _atomic_write_text(
            out / "high_level.json",
            json.dumps(self.high_level, ensure_ascii=False, indent=2),
        )
        if save_timeseries:
            _atomic_save_npz(out / "timeseries.npz", self.arrays())
        # The summary is the completion marker used by resume. Publish it last.
        _atomic_write_text(
            out / "summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2),
        )
