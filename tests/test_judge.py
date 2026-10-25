import json

from agora.judge import judge_dir
from agora.llm import ScriptedClient, respond


def test_judge_parses_json_and_skips_done(tmp_path):
    run = tmp_path / "g"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"scenario": "price_war", "boards": {"public": [{"round": 1, "agent": "seller_1", "text": "let's all hold 90"}]}}))
    seen = []
    client = ScriptedClient(lambda m, t: (seen.append(m[0]["content"]), respond(content='ok {"score": 8, "collusive": true, "evidence": "hold 90"}'))[1])
    assert judge_dir(tmp_path, client) == 1
    assert json.loads((run / "judge.json").read_text())["score"] == 8
    assert "hold 90" in seen[0]
    assert judge_dir(tmp_path, client) == 0
