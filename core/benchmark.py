"""Inventory and input provenance for the saved S1/S2 paired benchmark."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import json

from core.config import project_root
from core.constants import BENCHMARK_SCENARIOS, DYNAMICS, METHODS


def discover_environments(root):
    root = Path(root)
    return [
        path
        for scenario in BENCHMARK_SCENARIOS
        for path in sorted((root / scenario).glob("environment_*.json"))
    ]


def environment_inventory(root):
    """Validate labels, not navigation outcomes; never discard difficult maps."""
    from environments.scenario import Scenario

    inventory = {scenario: [] for scenario in BENCHMARK_SCENARIOS}
    for path in discover_environments(root):
        scenario = Scenario.load(path)
        if scenario.scenario_type != path.parent.name:
            raise ValueError(f"Scenario label does not match folder: {path}")
        if scenario.scenario_type == "S1" and scenario.dynamic_specs:
            raise ValueError(f"Static benchmark S1 contains moving obstacles: {path}")
        if scenario.scenario_type == "S2" and not scenario.dynamic_specs:
            raise ValueError(f"Dynamic benchmark S2 has no moving obstacles: {path}")
        inventory[scenario.scenario_type].append(path.stem)
    if any(not ids for ids in inventory.values()):
        raise ValueError(
            "The formal benchmark requires saved environments in both S1 and S2."
        )
    return inventory


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def simulation_source_hash():
    digest = hashlib.sha256()
    for folder in (
        "analysis",
        "controllers",
        "core",
        "dynamics",
        "environments",
        "experiments",
        "geometry",
        "planning",
        "safety",
        "simulation",
    ):
        for path in sorted((project_root() / folder).glob("*.py")):
            digest.update(path.relative_to(project_root()).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def trial_provenance(config_path, environment_path):
    return {
        "config_sha256": sha256_file(config_path),
        "environment_sha256": sha256_file(environment_path),
        "simulation_source_sha256": simulation_source_hash(),
    }


def benchmark_manifest(environment_root, config_path):
    inventory = environment_inventory(environment_root)
    return {
        "version": "V2.0",
        "environments": inventory,
        "environment_counts": {key: len(ids) for key, ids in inventory.items()},
        "total_environments": sum(map(len, inventory.values())),
        "expected_trials": sum(map(len, inventory.values()))
        * len(DYNAMICS)
        * len(METHODS),
        "dynamics": list(DYNAMICS),
        "methods": list(METHODS),
        "config_sha256": sha256_file(config_path),
        "simulation_source_sha256": simulation_source_hash(),
        "environment_sha256": {
            f"{path.parent.name}/{path.stem}": sha256_file(path)
            for path in discover_environments(environment_root)
        },
    }


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
