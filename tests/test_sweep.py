import json

import yaml

from agora import cli, sweep
from agora.llm import ScriptedClient
from helpers import call, respond


def test_sweep_runs_resumes_and_isolates_errors(tmp_path, monkeypatch):
    cfg = yaml.safe_load(open("configs/price_war.yaml"))
    cfg["arms"] = [{"name": "neutral"}, {"name": "broken", "coalition": ["nobody", "else"]}]
    (tmp_path / "pw.yaml").write_text(yaml.safe_dump(cfg))
    spec = {"configs": ["pw.yaml"], "seeds": {"start": 0, "count": 3}, "game_concurrency": 4,
            "overrides": {"horizon": {"min_units": 1, "continue_prob": 0.0, "max_units": 1}}}
    client = ScriptedClient(lambda m, t: respond(call("set_price", price=60)) if "EXECUTION" in m[1]["content"] else respond())
    monkeypatch.setattr(cli, "build_client", lambda llm: client)
    out = tmp_path / "out"
    rows = sweep.run_sweep(spec, {"base_url": "http://x", "model": "m"}, out, root=tmp_path)
    assert sum(1 for r in rows if r.get("error")) == 3 and sum(1 for r in rows if not r.get("error")) == 3
    assert len(json.loads((out / "headline.json").read_text())) == 3
    calls = client.calls
    rows2 = sweep.run_sweep(spec, {"base_url": "http://x", "model": "m"}, out, root=tmp_path)
    assert all(r.get("error") for r in rows2) and client.calls == calls  # finished games skipped
