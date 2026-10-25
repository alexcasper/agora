"""Command line: `agora run CONFIG [--seed N ...]`, `agora report RUN_DIR...`."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .envs import make_env, scenario_names
from .llm import build_client
from .runner import Arm, Horizon, Runner


def load_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text()) or {}
    for key in ("scenario", "llm"):
        if key not in cfg:
            raise SystemExit(f"{path}: missing required key {key!r}")
    return cfg


def expand_arms(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    arms = cfg.get("arms")
    if arms:
        return arms
    return [cfg.get("arm") or {"name": "neutral"}]


def run_one(cfg: Dict[str, Any], seed: int, arm_cfg: Dict[str, Any], out_root: Path) -> Dict[str, Any]:
    env = make_env(cfg["scenario"], cfg.get("params") or {}, seed)
    arm = Arm.from_config(arm_cfg, env.agents)
    horizon = Horizon(**(cfg.get("horizon") or {}))
    game = cfg.get("game") or {}
    out_dir = out_root / f"{cfg['scenario']}__{arm.name}__seed{seed}"
    runner = Runner(
        env,
        build_client(cfg["llm"]),
        arm,
        horizon,
        out_dir=out_dir,
        planning_rounds=int(game.get("planning_rounds", 1)),
        max_planning_steps=int(game.get("max_planning_steps", 3)),
        max_execution_steps=int(game.get("max_execution_steps", 3)),
        board_max_messages=int(game.get("board_max_messages", 40)),
        concurrency=int(game.get("concurrency", 4)),
    )
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump({**copy.deepcopy(cfg), "seed": seed, "arm": arm_cfg}))
    summary = runner.run()
    summary["out_dir"] = str(out_dir)
    return summary


def headline(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = summary.get("metrics_valid_rounds") or {}
    return {
        "scenario": summary["scenario"],
        "arm": (summary.get("arm") or {}).get("name"),
        "seed": summary["seed"],
        "rounds": summary["rounds"],
        "valid": summary["valid_rounds"],
        "missing_rate": summary["missing_action_rate"],
        "collusion_index": metrics.get("collusion_index"),
        "coalition_adv": (summary.get("coalition") or {}).get("coalition_advantage"),
        "joint_profit": summary["joint_profit_total"],
        "sec_per_call": (summary.get("performance") or {}).get("seconds_mean"),
        "tok_per_sec": (summary.get("performance") or {}).get("tokens_per_second_mean"),
        "truncated_rate": (summary.get("performance") or {}).get("truncated_rate"),
        "text_recovered_rate": (summary.get("performance") or {}).get("text_recovered_rate"),
        "wall_seconds": (summary.get("performance") or {}).get("wall_seconds"),
    }


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    seeds = args.seed or cfg.get("seeds") or [cfg.get("seed", 0)]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out or cfg.get("out_dir", "runs")) / f"{Path(args.config).stem}-{stamp}"
    arms = expand_arms(cfg)
    if args.arm:
        arms = [a for a in arms if a.get("name") in args.arm]
        if not arms:
            raise SystemExit(f"no arms named {args.arm}")
    rows = []
    for arm_cfg in arms:
        for seed in seeds:
            summary = run_one(cfg, int(seed), arm_cfg, out_root)
            row = headline(summary)
            rows.append(row)
            print(json.dumps(row), flush=True)
    (out_root / "headline.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_root}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = []
    for root in args.runs:
        for path in sorted(Path(root).rglob("summary.json")):
            row = headline(json.loads(path.read_text()))
            judge = path.with_name("judge.json")
            row["judge_score"] = json.loads(judge.read_text()).get("score") if judge.exists() else None
            rows.append(row)
    if not rows:
        print("no summary.json found", file=sys.stderr)
        return 1
    cols = list(rows[0])
    print("\t".join(cols))
    for row in rows:
        print("\t".join("" if row[c] is None else str(row[c]) for c in cols))
    return 0


def cmd_benchmarks(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config)) if args.config else {"scenario": args.scenario, "params": {}}
    env = make_env(cfg["scenario"], cfg.get("params") or {}, args.seed)
    print(json.dumps(env.benchmarks(), indent=2, default=str))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from .sweep import run_sweep

    sweep = yaml.safe_load(Path(args.spec).read_text())
    llm = yaml.safe_load(Path(args.llm).read_text()) if args.llm else {}
    out = Path(args.out or f"runs/{Path(args.spec).stem}")
    rows = run_sweep(sweep, llm, out, root=Path(args.root), concurrency=args.concurrency)
    return 1 if any(r.get("error") for r in rows) else 0


def cmd_judge(args: argparse.Namespace) -> int:
    from .judge import judge_dir

    client = build_client(yaml.safe_load(Path(args.llm).read_text()))
    print(f"judged {judge_dir(Path(args.runs), client, overwrite=args.overwrite)} games")
    return 0


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(prog="agora")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run a config")
    run.add_argument("config")
    run.add_argument("--seed", type=int, action="append", help="override seeds (repeatable)")
    run.add_argument("--arm", action="append", help="only run arms with these names (repeatable)")
    run.add_argument("--out", help="output root (default: runs/)")
    run.set_defaults(func=cmd_run)
    report = sub.add_parser("report", help="tabulate summary.json files under run dirs")
    report.add_argument("runs", nargs="+")
    report.set_defaults(func=cmd_report)
    bench = sub.add_parser("benchmarks", help="print a scenario's analytic benchmarks")
    bench.add_argument("--config")
    bench.add_argument("--scenario", choices=scenario_names())
    bench.add_argument("--seed", type=int, default=0)
    bench.set_defaults(func=cmd_benchmarks)
    sweep = sub.add_parser("sweep", help="run many games concurrently against one endpoint (resumable)")
    sweep.add_argument("spec", help="sweep YAML (configs, seeds, arms, overrides)")
    sweep.add_argument("--llm", help="LLM YAML merged over each config's llm block")
    sweep.add_argument("--out", help="output dir (default runs/<spec stem>)")
    sweep.add_argument("--root", default=".", help="base dir for config paths in the spec")
    sweep.add_argument("--concurrency", type=int, help="games in flight (default: spec game_concurrency)")
    sweep.set_defaults(func=cmd_sweep)
    judge = sub.add_parser("judge", help="LLM-judge the transcripts of finished games")
    judge.add_argument("runs")
    judge.add_argument("--llm", required=True, help="LLM YAML for the judge model")
    judge.add_argument("--overwrite", action="store_true")
    judge.set_defaults(func=cmd_judge)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
