"""Parallel sweeps: many games against one model endpoint.

A sweep spec lists scenario configs, seeds and (optionally) arm names. Each
(config, arm, seed) becomes one game; games run concurrently in a thread pool
so a batching server (vLLM) stays busy. A game whose `summary.json` already
exists is skipped, so an interrupted sweep resumes where it stopped.

Sweep spec (YAML):

    configs: [configs/price_war.yaml, configs/auction_house.yaml]
    seeds: {start: 0, count: 50}      # or an explicit list
    arms: [neutral, pair_prompt_private]   # optional filter; default all arms
    game_concurrency: 32
    overrides: {game: {concurrency: 1}}    # merged into every config
"""

from __future__ import annotations

import copy
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .cli import headline, run_one


def deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def seed_list(spec: Any) -> List[int]:
    if isinstance(spec, dict):
        return list(range(int(spec.get("start", 0)), int(spec.get("start", 0)) + int(spec["count"])))
    return [int(s) for s in spec]


def plan_jobs(sweep: Dict[str, Any], llm: Dict[str, Any], root: Path) -> List[Dict[str, Any]]:
    jobs = []
    seeds = seed_list(sweep.get("seeds", [0]))
    wanted = set(sweep.get("arms") or [])
    for config_path in sweep["configs"]:
        cfg = yaml.safe_load((root / config_path).read_text())
        cfg = deep_merge(cfg, sweep.get("overrides") or {})
        cfg["llm"] = deep_merge(cfg.get("llm") or {}, llm)
        arms = cfg.get("arms") or [cfg.get("arm") or {"name": "neutral"}]
        for arm in arms:
            if wanted and arm.get("name") not in wanted:
                continue
            for seed in seeds:
                jobs.append({"config": cfg, "arm": arm, "seed": seed})
    return jobs


def job_dir(out_root: Path, job: Dict[str, Any]) -> Path:
    return out_root / f"{job['config']['scenario']}__{job['arm'].get('name', 'neutral')}__seed{job['seed']}"


def run_sweep(
    sweep: Dict[str, Any],
    llm: Dict[str, Any],
    out_root: Path,
    root: Path = Path("."),
    concurrency: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(sweep, llm, root)
    todo = [j for j in jobs if not (job_dir(out_root, j) / "summary.json").exists()]
    print(f"sweep: {len(jobs)} games, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)
    workers = int(concurrency or sweep.get("game_concurrency", 16))
    rows: List[Dict[str, Any]] = []

    def work(job):
        return headline(run_one(job["config"], job["seed"], job["arm"], out_root))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(work, j): j for j in todo}
        for done, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                row = future.result()
            except Exception:  # noqa: BLE001 - one broken game must not stop the sweep
                err_dir = job_dir(out_root, job)
                err_dir.mkdir(parents=True, exist_ok=True)
                (err_dir / "error.txt").write_text(traceback.format_exc())
                row = {"scenario": job["config"]["scenario"], "arm": job["arm"].get("name"), "seed": job["seed"], "error": True}
            rows.append(row)
            print(f"[{done}/{len(todo)}] {json.dumps(row)}", flush=True)
    (out_root / "headline.json").write_text(
        json.dumps([headline(json.loads(p.read_text())) for p in sorted(out_root.glob("*/summary.json"))], indent=2)
    )
    return rows
