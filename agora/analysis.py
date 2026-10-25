"""Aggregate finished games into tidy rows and per-cell tables.

    agora analyze runs-modal/main-20261025 runs-modal/main-20261025-phi-r2 \
        --exclude k2-horizon-3.7b --prefer phi-4-mini=main-20261025-phi-r2 --out analysis/

Expects <root>/<tag>/<model>/<game>/summary.json (the Modal layout). One row
per game; tables are mean/sd/n over seeds for each model x scenario x arm.

Headline index per scenario is `collusion_index` (0 = competitive benchmark,
1 = collusive benchmark), except RideHailing, which uses
`collusion_index_vs_welfare`: its Nash outcome is itself rider-harming, so the
plain index overstates collusion there.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# The everyone-colludes arm has a scenario-specific name.
ARM_ALIASES = {"ring_all": "all_prompt_private", "chain_prompt_private": "all_prompt_private"}
ARM_ORDER = ["neutral", "pair_channel_only", "pair_prompt_only", "pair_prompt_private", "all_prompt_private"]
SCENARIO_ORDER = ["price_war", "auction_house", "supply_chain", "grid_storage", "ride_hailing", "ad_auction_gsp"]
HEADLINE = {"ride_hailing": "collusion_index_vs_welfare"}

# Secondary, scenario-specific readouts carried into the rows.
EXTRAS = {
    "price_war": ["mean_market_price", "full_tie_rounds", "proposal_commit_divergence_rate"],
    "auction_house": ["revenue_vs_bne", "mean_bid_over_bne_bid", "efficiency_rate", "winner_turnover"],
    "supply_chain": ["mean_input_price", "consumer_surplus_loss_vs_competitive", "shutdown_rounds"],
    "grid_storage": ["peak_withholding_index", "mean_peak_net_discharge", "complete_days"],
    "ride_hailing": ["collusion_index", "offline_rate", "served_fraction"],
    "ad_auction_gsp": ["mean_revenue_per_round", "allocative_efficiency", "truthful_bid_rate"],
}


def find_games(roots: Iterable[Path]) -> List[Path]:
    paths: List[Path] = []
    for root in roots:
        paths += sorted(Path(root).glob("**/summary.json"))
    return paths


def row_from_summary(path: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    model, tag = path.parent.parent.name, path.parent.parent.parent.name
    scenario = summary["scenario"]
    metrics = summary.get("metrics_valid_rounds") or {}
    perf = summary.get("performance") or {}
    arm = (summary.get("arm") or {}).get("name", "neutral")
    row = {
        "tag": tag,
        "model": model,
        "scenario": scenario,
        "arm": ARM_ALIASES.get(arm, arm),
        "seed": summary["seed"],
        "rounds": summary["rounds"],
        "valid_rounds": summary["valid_rounds"],
        "missing_action_rate": summary["missing_action_rate"],
        "index": metrics.get(HEADLINE.get(scenario, "collusion_index")),
        "coalition_advantage": (summary.get("coalition") or {}).get("coalition_advantage"),
        "joint_profit_total": summary.get("joint_profit_total"),
        "seconds_per_call": perf.get("seconds_mean"),
        "tokens_per_call": perf.get("completion_tokens_mean"),
        "truncated_rate": perf.get("truncated_rate"),
        "text_recovered_rate": perf.get("text_recovered_rate"),
        "wall_seconds": perf.get("wall_seconds"),
    }
    for key in EXTRAS.get(scenario, []):
        row[f"x_{key}"] = metrics.get(key)
    return row


def load_rows(roots: Iterable[Path], exclude: Iterable[str] = (), prefer: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Rows for every readable game. `prefer` maps model -> tag to use for it
    when the same model appears under several tags (e.g. a rerun)."""
    exclude, prefer = set(exclude), dict(prefer or {})
    rows = []
    for path in find_games(roots):
        try:
            summary = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # partially downloaded; pull again
        row = row_from_summary(path, summary)
        if row["model"] in exclude:
            continue
        if row["model"] in prefer and row["tag"] != prefer[row["model"]]:
            continue
        rows.append(row)
    return rows


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": None, "sd": None, "n": 0}
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else None
    return {"mean": round(mean, 4), "sd": round(sd, 4) if sd is not None else None, "n": len(vals)}


def cell_table(rows: List[Dict[str, Any]], value: str = "index") -> Dict[tuple, Dict[str, Any]]:
    cells: Dict[tuple, List[float]] = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["scenario"], r["arm"])].append(r[value])
    return {k: _stats(v) for k, v in cells.items()}


def quality_table(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    out = {}
    for model, rs in sorted(by_model.items()):
        out[model] = {
            "games": len(rs),
            "missing_action_rate": _stats([r["missing_action_rate"] for r in rs])["mean"],
            "games_without_valid_rounds": sum(1 for r in rs if not r["valid_rounds"]),
            "truncated_rate": _stats([r["truncated_rate"] for r in rs])["mean"],
            "text_recovered_rate": _stats([r["text_recovered_rate"] for r in rs])["mean"],
            "seconds_per_call": _stats([r["seconds_per_call"] for r in rs])["mean"],
            "tokens_per_call": _stats([r["tokens_per_call"] for r in rs])["mean"],
        }
    return out


def _fmt(stat: Dict[str, Any]) -> str:
    if not stat or stat["mean"] is None:
        return "—"
    sd = f" ± {stat['sd']:.2f}" if stat["sd"] is not None else ""
    return f"{stat['mean']:.2f}{sd} (n={stat['n']})"


def markdown_report(rows: List[Dict[str, Any]]) -> str:
    models = sorted({r["model"] for r in rows})
    lines = ["# agora results", "", f"{len(rows)} games, models: {', '.join(models)}", ""]
    lines += ["## Run quality by model", "", "| model | games | missing | no-valid games | truncated | text-recovered | s/call | tokens/call |", "|---|---|---|---|---|---|---|---|"]
    for model, q in quality_table(rows).items():
        lines.append(
            f"| {model} | {q['games']} | {q['missing_action_rate']:.3f} | {q['games_without_valid_rounds']} | "
            f"{(q['truncated_rate'] or 0):.3f} | {(q['text_recovered_rate'] or 0):.3f} | {(q['seconds_per_call'] or 0):.1f} | {(q['tokens_per_call'] or 0):.0f} |"
        )
    table = cell_table(rows)
    for scenario in SCENARIO_ORDER:
        metric = HEADLINE.get(scenario, "collusion_index")
        lines += ["", f"## {scenario} — `{metric}` (mean ± sd over seeds)", "", "| model | " + " | ".join(ARM_ORDER) + " |", "|---" * (len(ARM_ORDER) + 1) + "|"]
        for model in models:
            lines.append(f"| {model} | " + " | ".join(_fmt(table.get((model, scenario, arm))) for arm in ARM_ORDER) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(rows: List[Dict[str, Any]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("x_"), k))
    with open(out / "games.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out / "report.md").write_text(markdown_report(rows))
