import json

from agora.analysis import load_rows, markdown_report


def _game(root, tag, model, scenario, arm, seed, metrics):
    d = root / tag / model / f"{scenario}__{arm}__seed{seed}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({
        "scenario": scenario, "seed": seed, "rounds": 4, "valid_rounds": 4, "missing_action_rate": 0.0,
        "joint_profit_total": 1.0, "metrics_valid_rounds": metrics, "arm": {"name": arm},
        "performance": {"seconds_mean": 1.0, "completion_tokens_mean": 10, "truncated_rate": 0.0, "text_recovered_rate": 0.0},
    }))


def test_rows_aliases_prefer_and_headline(tmp_path):
    _game(tmp_path, "main", "m1", "auction_house", "ring_all", 0, {"collusion_index": 0.5})
    _game(tmp_path, "main", "m1", "ride_hailing", "neutral", 0, {"collusion_index": 0.9, "collusion_index_vs_welfare": 0.2})
    _game(tmp_path, "main", "phi", "price_war", "neutral", 0, {"collusion_index": 0.0})
    _game(tmp_path, "r2", "phi", "price_war", "neutral", 0, {"collusion_index": 0.3})
    _game(tmp_path, "main", "k2", "price_war", "neutral", 0, {"collusion_index": 1.0})
    (tmp_path / "main" / "m1" / "broken").mkdir()
    (tmp_path / "main" / "m1" / "broken" / "summary.json").write_text("")
    rows = load_rows([tmp_path], exclude=["k2"], prefer={"phi": "r2"})
    by = {(r["model"], r["scenario"]): r for r in rows}
    assert len(rows) == 3
    assert by[("m1", "auction_house")]["arm"] == "all_prompt_private"
    assert by[("m1", "ride_hailing")]["index"] == 0.2
    assert by[("phi", "price_war")]["index"] == 0.3
    assert "| m1 |" in markdown_report(rows)
