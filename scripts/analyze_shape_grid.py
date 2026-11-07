"""Aggregate the SwarmShape grid (runs/shape_grid) into games.csv + report.md.

Joins each finished game's summary.json with graph features recomputed from the
arm's graph_edges (configs/grid/n<N>.yaml via the run's own config.yaml), so no
census file is needed at analysis time. Usage:
    uv run python scripts/analyze_shape_grid.py [--run runs/shape_grid]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agora.graphs import features_from_edges  # noqa: E402


def load_games(run: Path):
    rows = []
    for d in sorted(run.iterdir()):
        summary = d / "summary.json"
        if not summary.exists():
            continue
        s = json.load(open(summary))
        edges = (s.get("arm") or {}).get("graph_edges")
        if edges is None:  # named-shape arms (smoke runs) have no edges
            continue
        n_agents = max(max(e) for e in edges) + 1
        f = features_from_edges(n_agents, edges)
        rogue = s.get("rogue") or {}
        infl = s.get("proposal_influence") or {}
        name = s.get("arm", {}).get("name") or d.name
        parts = name.split("_")
        arm_kind = parts[-1] if parts[-1] in ("clean", "rogue") else "?"
        family = "_".join(parts[2:-1])
        rows.append({
            "arm": name,
            "n": f["n"],
            "family": family,
            "kind": arm_kind,
            "seed": s.get("seed"),
            "rounds": s.get("rounds"),
            "valid_rate": round(s.get("valid_rounds", 0) / max(1, s.get("rounds", 1)), 3),
            # topology features
            "edges": len(edges),
            "density": f["density"],
            "diameter": f["diameter"],
            "max_degree": f["max_degree"],
            "degree_var": f["degree_var"],
            "clustering": f["clustering"],
            # outcomes
            "rogue_violations": rogue.get("violations"),
            "rogue_rate": rogue.get("violation_rate"),
            "rogue_onset": min(map(int, (rogue.get("by_round") or {}).keys()), default=None),
            "follows": infl.get("follows"),
            "takeover_hhi": infl.get("takeover_hhi"),
            "uniform_hhi": infl.get("uniform_hhi"),
            "joint_profit": s.get("joint_profit_total"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/shape_grid")
    args = ap.parse_args()
    run = Path(args.run)
    rows = load_games(run)
    if not rows:
        print(f"no finished games with graph_edges under {run}")
        return
    out_csv = run / "games.csv"
    keys = list(rows[0].keys())
    with open(out_csv, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    # aggregate: mean rogue rate / takeover ratio per (n, kind)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        a = agg[r["n"]]
        a[f"{r['kind']}_rogue_rate"].append(r["rogue_rate"] or 0.0)
        if r["takeover_hhi"] is not None and r["uniform_hhi"]:
            a[f"{r['kind']}_takeover_ratio"].append(r["takeover_hhi"] / r["uniform_hhi"])
    report = ["# SwarmShape grid results", "",
              f"{len(rows)} games finished (of the 114-game grid).", "",
              "| n | clean rogue-rate | rogue-arm rogue-rate | clean takeover/uniform | rogue takeover/uniform |",
              "|---|---|---|---|---|"]
    for n in sorted(agg):
        a = agg[n]
        mean = lambda k: (round(sum(a[k]) / len(a[k]), 3) if a.get(k) else None)  # noqa: E731
        report.append(
            f"| {n} | {mean('clean_rogue_rate')} | {mean('rogue_rogue_rate')} "
            f"| {mean('clean_takeover_ratio')} | {mean('rogue_takeover_ratio')} |")
    (run / "report.md").write_text("\n".join(report) + "\n")
    print(f"wrote {out_csv} and {run/'report.md'} for {len(rows)} games")


if __name__ == "__main__":
    main()
