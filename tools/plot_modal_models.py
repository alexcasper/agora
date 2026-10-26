"""Figures from the Modal multi-model agora sweep (runs-modal/main-<date>/).

Usage: LD_LIBRARY_PATH=<gcc-lib> .venv/bin/python tools/plot_modal_models.py <runs-dir> <outdir>

Emits: fig_modal_models_index.png (mean collusion index, model x scenario heatmap)
       fig_modal_models_quality.png (missing-rate + truncated-rate by model)
       modal_models_summary.json (the aggregated numbers behind the figures)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = [
    "price_war",
    "auction_house",
    "supply_chain",
    "grid_storage",
    "ride_hailing",
    "ad_auction_gsp",
]


def main(runs_dir: str, outdir: str) -> None:
    runs = Path(runs_dir)
    models = sorted(
        p.name for p in runs.iterdir()
        if p.is_dir() and (p / "headline.json").exists()
    )
    skipped = sorted(p.name for p in runs.iterdir()
                     if p.is_dir() and not (p / "headline.json").exists())
    if skipped:
        print(f"skipping models without headline.json (incomplete runs): {skipped}")
    # agg[model][scenario] -> dict of means over all runs (arms x seeds)
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for m in models:
        for rec in json.loads((runs / m / "headline.json").read_text()):
            a = agg[m][rec["scenario"]]
            for k in ("collusion_index", "missing_rate", "truncated_rate"):
                if rec.get(k) is not None:
                    a[k].append(float(rec[k]))
            a["n"].append(1)

    mat = np.full((len(models), len(SCENARIOS)), np.nan)
    for i, m in enumerate(models):
        for j, s in enumerate(SCENARIOS):
            vals = agg[m][s]["collusion_index"]
            if vals:
                mat[i, j] = float(np.mean(vals))

    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Figure 1: collusion-index heatmap
    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SCENARIOS)), SCENARIOS, rotation=20, ha="right")
    ax.set_yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(SCENARIOS)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#888")
    ax.set_title("agora Modal sweep: mean collusion index by model x scenario\n(0 = competitive benchmark, 1 = collusive benchmark)")
    fig.colorbar(im, ax=ax, label="collusion index")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig_modal_models_index.png", dpi=150)

    # Figure 2: data quality by model
    miss = [np.mean(agg[m][s]["missing_rate"]) if agg[m][s]["missing_rate"] else 0.0
            for m in models for s in SCENARIOS]
    miss = np.array(miss).reshape(len(models), len(SCENARIOS))
    trunc = np.array([
        (np.mean(agg[m][s]["truncated_rate"]) if agg[m][s]["truncated_rate"] else 0.0)
        for m in models for s in SCENARIOS
    ]).reshape(len(models), len(SCENARIOS))
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - 0.2, miss.mean(axis=1), 0.4, label="missing-action rate", color="#D55E00")
    ax.bar(x + 0.2, trunc.mean(axis=1), 0.4, label="truncated-output rate", color="#0072B2")
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_ylabel("rate")
    ax.set_title("agora Modal sweep: data-quality rates by model (mean over scenarios)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig_modal_models_quality.png", dpi=150)

    # Summary json
    summary = {
        m: {
            s: {
                "collusion_index_mean": None if np.isnan(mat[i, j]) else round(float(mat[i, j]), 4),
                "missing_rate_mean": round(float(miss[i, j]), 4),
                "truncated_rate_mean": round(float(trunc[i, j]), 4),
                "n_runs": len(agg[m][s]["n"]),
            }
            for j, s in enumerate(SCENARIOS)
        }
        for i, m in enumerate(models)
    }
    (Path(outdir) / "modal_models_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote figures + modal_models_summary.json for {len(models)} models x {len(SCENARIOS)} scenarios")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
