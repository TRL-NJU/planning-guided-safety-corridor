

from pathlib import Path
import argparse
import csv
import sys

DEFAULT_SCENARIO = "S1"
DEFAULT_METHOD = "jps_bas"
DEFAULT_ENVIRONMENT = 24
DEFAULT_RESULTS_ROOT = "results/benchmark"
DEFAULT_SHOW = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("S1", "S2"), default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--method",
        choices=("jps", "jps_bas", "single_scp_bas", "corridor_bas"),
        default=DEFAULT_METHOD,
    )
    parser.add_argument(
        "--environment",
        "--env",
        default=DEFAULT_ENVIRONMENT,
        help="Environment number or environment_XXX",
    )
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    try:
        number = int(str(args.environment).removeprefix("environment_"))
        if number < 1:
            raise ValueError()
    except ValueError:
        parser.error("--environment must be a positive number or environment_XXX")
    if args.no_show or not DEFAULT_SHOW:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from export_trajectory_montage import (
        _resolve_result_dir,
        _load_saved_trial,
        _draw_panel,
        _legend,
        apply_publication_style,
        METHOD_LABELS,
    )
    from environments.scenario import Scenario

    root = Path(__file__).resolve().parent
    results = Path(args.results_root)
    if not results.is_absolute():
        results = root / results
    env_id = f"environment_{number:03d}"
    trial = _resolve_result_dir(
        root, results, "single", args.scenario, args.method, env_id
    )
    summary, arrays, final_path, saved_scene = _load_saved_trial(trial)
    print(
        f"场景: {args.scenario} | 方法: {METHOD_LABELS[args.method]} | 环境: {env_id}"
    )
    if summary is None:
        # A summary alone cannot reconstruct an executed trajectory.
        aggregate = results / "summary_all.csv"
        if aggregate.exists():
            with aggregate.open(encoding="utf-8-sig", newline="") as f:
                matches = [
                    r
                    for r in csv.DictReader(f)
                    if r.get("scenario") == args.scenario
                    and r.get("method") == args.method
                    and r.get("dynamics") == "single"
                    and r.get("environment_id") == env_id
                ]
            if len(matches) == 1:
                report(matches[0])
                print(f"状态来源: {aggregate}")
        raise FileNotFoundError(
            f"缺少完整轨迹结果，无法绘制实际轨迹: {trial}\n需要 summary.json 和 timeseries.npz；不会自动重新运行仿真。"
        )
    for key, expected in (
        ("environment_id", env_id),
        ("method", args.method),
        ("scenario", args.scenario),
        ("dynamics", "single"),
    ):
        if summary.get(key) and str(summary[key]) != expected:
            raise ValueError(
                f"结果标识不匹配: {key}={summary[key]}, expected {expected}"
            )
    report(summary)
    print(f"结果来源: {trial}")
    if saved_scene is not None:
        scene = saved_scene
        if scene.environment_id and scene.environment_id != env_id:
            raise ValueError("保存的场景与环境编号不一致")
    else:
        scene = Scenario.load(
            root / "data" / "custom_environments" / args.scenario / f"{env_id}.json"
        )
        print("场景来源: 当前环境 JSON（结果未保存场景快照）")
    apply_publication_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Times New Roman"],
            "font.size": 3,
            "font.weight": "normal",
            "axes.labelsize": 20,
            "axes.labelweight": "normal",
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 15,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    _draw_panel(ax, scene, args.method, summary, arrays, final_path, 0, 0, 1)
    ax.set_title("")
    legend_handles = _legend(
        args.method,
        bool(scene.dynamic_specs),
        final_path is not None and len(final_path) > 0,
    )
    for handle in legend_handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(2.5)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        markerscale=1.8,
        handlelength=3.0,
        handleheight=1.5,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.15, top=0.89)
    output = results / "figures" / "individual"
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"{args.scenario}_single_{args.method}_{env_id}"
    path = stem.with_suffix(".png")
    fig.savefig(path, format="png", dpi=300, facecolor="white")
    print(f"图片: {path}")
    if DEFAULT_SHOW and not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def report(summary):
    status = str(summary.get("terminal_status", "unknown"))
    reason = str(summary.get("failure_reason", "") or "")
    if status == "success":
        print("状态: 成功")
    else:
        meanings = {
            "physical_collision": "碰撞",
            "collision": "碰撞",
            "timeout": "超时",
            "deadlock": "死锁/停滞",
        }
        translated = meanings.get(reason, meanings.get(status, reason or status))
        print(f"状态: 失败 | 原因: {translated} | 原始记录: {status} / {reason}")
    if summary.get("travel_time") is not None:
        print(f"运行时长: {float(summary['travel_time']):.2f} s")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)
