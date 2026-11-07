"""SwarmShape analysis (apart-u5s): rogue rate + takeover HHI vs size and graph features.

Usage (from agora/):  uv run python tools/shape_analysis.py runs/shape_grid
Writes <run_dir>/games.csv and <run_dir>/report.md. Stdlib only; safe to rerun on a
partial grid (reports k/expected games done).

Design notes
- Each sampled graph is played once per arm (clean / rogue) at seed 42, so the unit of
  analysis is the GRAPH: clean vs rogue are paired on the same graph, and topology effects
  are cross-sectional correlations across graphs. There is no within-cell seed variance;
  every per-n number is n_graphs <= 2 games and must be read as such.
- Rogue-arm metrics split the rogue seat from everyone else. "spillover" = honour-rule
  violation rate of NON-rogue agents in the rogue arm minus the all-agent violation rate
  in the clean arm on the same graph: the swarm-level question (does one rogue make its
  neighbours break rules?).
- Data-quality columns are carried per game; headline tables use games whose
  missing_action_rate <= MAX_MISSING (pre-registered gate, B0).
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agora.graphs import features_from_edges  # noqa: E402

EXPECTED_GAMES = 114
MAX_MISSING = 0.10
ARM_RE = re.compile(r"^n(?P<n>\d+)_g(?P<g>\d+)_(?P<kind>[a-z]+)_(?P<arm>clean|rogue)$")
FEATURES = ["density", "degree_var", "clustering", "diameter", "max_degree"]


def _family_index(census_path: Path):
    """Map a feature signature -> family name for sampled census graphs (best effort)."""
    idx = defaultdict(set)
    if not census_path.exists():
        return idx
    for c in json.loads(census_path.read_text()):
        sig = (c["n"], c["edges"], c["triangles"], c["max_degree"], c["diameter"], c["degree_var"])
        idx[sig].add(c["family"])
    return idx


def _family(feat, idx):
    fams = idx.get((feat["n"], feat["edges"], feat["triangles"], feat["max_degree"],
                    feat["diameter"], feat["degree_var"]), set())
    return "|".join(sorted(fams)) if fams else ("complete" if feat["density"] == 1.0 else "?")


def load_games(run_dir: Path, fam_idx):
    rows = []
    for p in sorted(run_dir.glob("*/summary.json")):
        s = json.loads(p.read_text())
        arm = s["arm"]
        m = ARM_RE.match(arm["name"])
        if not m:
            continue
        n = int(m["n"])
        edges = [tuple(e) for e in (arm.get("graph_edges") or [])]
        feat = features_from_edges(n, edges)
        agents = [f"seller_{i + 1}" for i in range(n)]
        deg = {a: 0 for a in agents}
        for a, b in edges:
            deg[agents[a]] += 1
            deg[agents[b]] += 1
        rogue = s.get("rogue") or {}
        by_agent = rogue.get("by_agent") or {}
        valid = s.get("valid_rounds") or 0
        rogue_seat = arm.get("rogue")
        others = [a for a in agents if a != rogue_seat]
        other_viol = sum(by_agent.get(a, 0) for a in others)
        pi = s.get("proposal_influence") or {}
        perf = s.get("performance") or {}
        mv = s.get("metrics_valid_rounds") or {}
        hhi, uni = pi.get("takeover_hhi"), pi.get("uniform_hhi")
        rows.append({
            "game": p.parent.name,
            "n": n,
            "graph": f"n{n}_g{m['g']}",
            "arm": m["arm"],
            "family": _family(feat, fam_idx),
            **{k: feat[k] for k in ["edges", *FEATURES, "is_tree"]},
            "rogue_seat": rogue_seat or "",
            "rogue_degree": deg.get(rogue_seat) if rogue_seat else "",
            "rogue_degree_rank": (sorted(deg.values(), reverse=True).index(deg[rogue_seat]) + 1)
            if rogue_seat else "",
            "valid_rounds": valid,
            "rounds": s.get("rounds"),
            "violation_rate": rogue.get("violation_rate"),
            "rogue_seat_violation_rate": (by_agent.get(rogue_seat, 0) / valid) if (rogue_seat and valid) else "",
            "others_violation_rate": (other_viol / (len(others) * valid)) if valid else None,
            "onset_round": rogue.get("onset_round"),
            "follows": pi.get("follows"),
            "takeover_hhi": hhi,
            "uniform_hhi": uni,
            "hhi_excess": round(hhi - uni, 4) if (hhi is not None and uni is not None) else None,
            "collusion_index": mv.get("collusion_index"),
            "missing_action_rate": s.get("missing_action_rate"),
            "truncated_rate": perf.get("truncated_rate"),
            "text_recovered_rate": perf.get("text_recovered_rate"),
            "llm_error_rate": perf.get("llm_error_rate"),
            "invalid_tool_call_rate": (round((s.get("counters") or {}).get("invalid_tool_calls", 0)
                                             / (s.get("counters") or {}).get("llm_calls"), 4)
                                       if (s.get("counters") or {}).get("llm_calls") else None),
            "wall_seconds": perf.get("wall_seconds"),
        })
    return rows


def _num(xs):
    return [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]


def _mean(xs):
    xs = _num(xs)
    return sum(xs) / len(xs) if xs else None


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if isinstance(a, (int, float)) and isinstance(b, (int, float))
             and not isinstance(a, bool) and not isinstance(b, bool)]
    if len(pairs) < 5:
        return None, len(pairs)
    rx, ry = _rank([a for a, _ in pairs]), _rank([b for _, b in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return (round(cov / (vx * vy), 3) if vx and vy else None), len(pairs)


def _fmt(x, nd=3):
    if x is None or x == "":
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def paired(rows):
    by_graph = defaultdict(dict)
    for r in rows:
        by_graph[r["graph"]][r["arm"]] = r
    out = []
    for g, d in by_graph.items():
        if "clean" in d and "rogue" in d:
            c, ro = d["clean"], d["rogue"]
            out.append({
                "graph": g, "n": c["n"], "family": c["family"], **{k: c[k] for k in FEATURES},
                "rogue_degree": ro["rogue_degree"], "rogue_degree_rank": ro["rogue_degree_rank"],
                "clean_viol": c["violation_rate"], "rogue_seat_viol": ro["rogue_seat_violation_rate"],
                "others_viol_rogue": ro["others_violation_rate"],
                "spillover": (ro["others_violation_rate"] - c["violation_rate"])
                if isinstance(ro["others_violation_rate"], (int, float)) and isinstance(c["violation_rate"], (int, float)) else None,
                "hhi_excess_clean": c["hhi_excess"], "hhi_excess_rogue": ro["hhi_excess"],
                "ci_clean": c["collusion_index"], "ci_rogue": ro["collusion_index"],
            })
    return sorted(out, key=lambda r: (r["n"], r["graph"]))


def report(run_dir: Path, rows):
    gated = [r for r in rows if (r["missing_action_rate"] or 0) <= MAX_MISSING]
    pr = paired(gated)
    L = [f"# SwarmShape analysis — {run_dir.name}", "",
         f"Games found: **{len(rows)}/{EXPECTED_GAMES}**; passing B0 gate (missing_action_rate ≤ {MAX_MISSING}): "
         f"{len(gated)}; complete clean+rogue pairs: {len(pr)}. Model: dusk Qwen3.5-4B, seed 42 only.",
         "", "> **Read as provisional.** One seed per (graph, arm): there is no within-cell variance, so "
         "per-graph differences mix topology with sampling noise. Topology claims rest on cross-graph "
         "rank correlations below, not on any single row.", ""]
    # data quality
    L += ["## Data quality", "",
          "| metric | mean | max |", "|---|---:|---:|"]
    for k in ["missing_action_rate", "truncated_rate", "text_recovered_rate", "llm_error_rate", "invalid_tool_call_rate"]:
        xs = _num(r[k] for r in rows)
        L.append(f"| {k} | {_fmt(_mean(xs))} | {_fmt(max(xs) if xs else None)} |")
    # arm-level
    L += ["", "## Arm totals (gated games)", "",
          "| arm | games | violation rate (all) | rogue-seat rate | others' rate | takeover HHI − uniform | collusion index |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ["clean", "rogue"]:
        a = [r for r in gated if r["arm"] == arm]
        L.append(f"| {arm} | {len(a)} | {_fmt(_mean(r['violation_rate'] for r in a))} | "
                 f"{_fmt(_mean(r['rogue_seat_violation_rate'] for r in a)) if arm == 'rogue' else '–'} | "
                 f"{_fmt(_mean(r['others_violation_rate'] for r in a))} | "
                 f"{_fmt(_mean(r['hhi_excess'] for r in a))} | {_fmt(_mean(r['collusion_index'] for r in a))} |")
    sp = _num(p["spillover"] for p in pr)
    if sp:
        pos = sum(1 for x in sp if x > 0)
        neg = sum(1 for x in sp if x < 0)
        L += ["", f"**Spillover** (others' violation rate in rogue arm − clean-arm rate, paired by graph): "
              f"mean {_fmt(_mean(sp))}, positive in {pos}/{len(sp)} graphs, negative in {neg}, zero in {len(sp) - pos - neg}. "
              f"Sign test is the honest test at one seed per cell."]
    # per-n
    L += ["", "## Per graph (paired)", "",
          "| graph | family | density | deg var | clust | diam | rogue deg (rank) | clean viol | rogue-seat viol | others viol | spillover | HHI−u clean | HHI−u rogue |",
          "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for p in pr:
        L.append(f"| {p['graph']} | {p['family']} | {_fmt(p['density'])} | {_fmt(p['degree_var'])} | {_fmt(p['clustering'])} | "
                 f"{_fmt(p['diameter'])} | {p['rogue_degree']} ({p['rogue_degree_rank']}) | {_fmt(p['clean_viol'])} | "
                 f"{_fmt(p['rogue_seat_viol'])} | {_fmt(p['others_viol_rogue'])} | {_fmt(p['spillover'])} | "
                 f"{_fmt(p['hhi_excess_clean'])} | {_fmt(p['hhi_excess_rogue'])} |")
    # correlations
    L += ["", "## Rank correlations across graphs (Spearman ρ, pairs with ≥5 graphs)", "",
          "| outcome | n | " + " | ".join(FEATURES + ["rogue_degree"]) + " |",
          "|---|---|" + "---:|" * (len(FEATURES) + 1)]
    for out in ["clean_viol", "rogue_seat_viol", "others_viol_rogue", "spillover", "hhi_excess_clean", "hhi_excess_rogue"]:
        cells, npairs, nrho = [], 0, None
        for f in ["n"] + FEATURES + ["rogue_degree"]:
            rho, k = spearman([p[f] for p in pr], [p[out] for p in pr])
            npairs = max(npairs, k)
            if f != "n":
                cells.append(_fmt(rho))
            else:
                nrho = rho
        L.append(f"| {out} | ρ(n)={_fmt(nrho)} | " + " | ".join(cells) + " |")
    L += ["", "Features are computed from each game's own `graph_edges` (agora.graphs.features_from_edges); "
          "family is matched against data/graph_census.json by feature signature ('?' = no unique match). "
          "Density and n are strongly confounded in this grid (larger n → sparser sampled families), so read "
          "ρ(n) alongside every feature column.", ""]
    (run_dir / "report.md").write_text("\n".join(L))


def main(argv):
    run_dir = Path(argv[1] if len(argv) > 1 else "runs/shape_grid")
    fam_idx = _family_index(Path(__file__).resolve().parents[1] / "data" / "graph_census.json")
    rows = load_games(run_dir, fam_idx)
    if not rows:
        print(f"no games under {run_dir}")
        return 1
    with open(run_dir / "games.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    report(run_dir, rows)
    print(f"{len(rows)}/{EXPECTED_GAMES} games -> {run_dir/'games.csv'}, {run_dir/'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
