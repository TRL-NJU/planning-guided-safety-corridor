# Planning-Guided Safety Corridor Navigation

## Code layout

```text
config/          Experiment parameters and run/output settings
data/           25 saved S1 and 25 saved S2 environments
planning/        JPS planner and reference-path utilities
geometry/        Obstacle geometry, distances, and convex operations
safety/          SCP construction, corridor fusion, acceptance, Poisson fields, and BaS
controllers/     Navigation feedback and input-barrier velocity controller
dynamics/        Robot state and motion integration
simulation/      Closed-loop simulation and result logging
experiments/     Trial execution and result collection
analysis/        Evaluation metrics
visualization/   Shared plotting style, scenes, and PNG previews
core/            Configuration, constants, and reproducibility metadata
requirements.txt Runtime Python dependencies
```

## Entry scripts

| Script | Purpose |
|---|---|
| `run_benchmark.py` | Run all four methods in all 50 saved environments and export the results table. |
| `run_trial.py` | Run one method in one selected environment. |
| `edit_environment.py` | Create and edit environments through the graphical editor. |
| `view_trial_result.py` | Inspect an individual result, print its status, and export its trajectory as PNG. |
| `export_trajectory_montage.py` | Export selected or all environments as PNG trajectory montages. |
| `view_corridor_evolution.py` | Select and export stages of the evolving corridor through the graphical viewer. |
| `export_benchmark_table.py` | Export success rates and mean start-to-goal times as CSV and PNG. |
| `audit_input_constraints.py` | Check recorded velocity limits, input-barrier consistency, and integration residuals. |

`config/benchmark.yaml` defines the method and simulation parameters.
`config/run.yaml` selects the environment and result directories.
Results are stored in `results/benchmark/` and excluded from Git.
The graphical editor and viewer use Tkinter.
All exported figures use PNG format. CSV, JSON, and NPZ files retain numerical results.
