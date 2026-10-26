import json
import random

import pytest

from agora.envs import make_env
from agora.llm import parse_text_tool_calls
from agora.runner import Arm, Horizon, Runner
from helpers import call, respond, scripted


def test_horizon_bounds_and_determinism():
    h = Horizon(min_units=4, continue_prob=0.75, max_units=12)
    draws = [h.draw(random.Random(i)) for i in range(500)]
    assert min(draws) == 4 and max(draws) <= 12
    assert h.draw(random.Random(7)) == h.draw(random.Random(7))
    assert "not announced" in h.describe("market day")
    assert "exactly 5" in Horizon(5, 0.0, 5).describe("auction")


def test_arm_validation_and_text():
    agents = ["a", "b", "c"]
    arm = Arm.from_config({"name": "x", "coalition": 2, "coalition_prompt": True, "private_channel": True}, agents)
    assert arm.coalition == ["a", "b"]
    assert "PRIVATE ARRANGEMENT" in arm.text("a", "seller") and arm.text("c", "seller") == ""
    assert "team" in arm.objective("a") and "own" in arm.objective("c")
    with pytest.raises(ValueError):
        Arm.from_config({"coalition": ["a"], "private_channel": True}, agents)
    with pytest.raises(ValueError):
        Arm.from_config({"coalition": ["z", "a"]}, agents)


def test_text_tool_call_recovery():
    text = 'thinking... <tool_call>{"name": "set_price", "arguments": {"price": 50}}</tool_call>'
    calls = parse_text_tool_calls(text, ["set_price"])
    assert calls[0].name == "set_price" and calls[0].arguments == {"price": 50} and calls[0].parsed_from_text
    assert parse_text_tool_calls(text, ["other"]) == []


def test_missing_commit_abstains_and_flags_round(tmp_path):
    env = make_env("price_war", {"costs": [40, 45.5, 52.8]}, seed=1)
    # seller_3 never commits; others post 60.
    client = scripted(lambda a, m: respond() if a == "seller_3" else respond(call("set_price", price=60)))
    summary = Runner(env, client, Arm(), Horizon(3, 0.0, 3), out_dir=tmp_path, concurrency=1).run()
    assert summary["rounds"] == 3 and summary["valid_rounds"] == 0
    assert summary["missing_action_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert all(r["missing"] == ["seller_3"] for r in summary["history"])
    assert "seller_3" not in summary["history"][0]["prices"]  # no fallback price
    assert summary["metrics_valid_rounds"] == {}
    assert summary["profit_totals"]["seller_1"] == pytest.approx(3 * 30 * 20)
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "settle" for e in events)


def test_private_channel_visibility_and_proposals(tmp_path):
    env = make_env("price_war", {"costs": [40, 45.5, 52.8]}, seed=1)

    def planning(agent, messages):
        if agent == "seller_1":
            return respond(call("post_message", text="psst", channel="private"), call("announce_price", price=70))
        if agent == "seller_3":
            return respond(call("post_message", text="hello", channel="private"))
        return respond()

    client = scripted(lambda a, m: respond(call("set_price", price=70)), planning)
    arm = Arm.from_config({"name": "pair", "coalition": ["seller_1", "seller_2"], "private_channel": True}, env.agents)
    runner = Runner(env, client, arm, Horizon(1, 0.0, 1), out_dir=tmp_path, concurrency=1)
    summary = runner.run()
    assert [p["text"] for p in summary["boards"]["private"]] == ["psst"]  # seller_3 cannot post there
    assert runner.counters["invalid_tool_calls"] == 1
    assert "PRIVATE BOARD" in runner.boards.render("seller_2") and "PRIVATE BOARD" not in runner.boards.render("seller_3")
    assert any(p["kind"] == "announcement" for p in summary["boards"]["public"])
    m = summary["metrics_valid_rounds"]
    assert m["proposal_commit_pairs"] == 1 and m["proposal_commit_divergence_rate"] == 0
    assert summary["coalition"]["coalition_advantage"] is not None


def test_execution_nudge_after_text_only_reply():
    env = make_env("price_war", {"costs": [40, 45.5, 52.8]}, seed=1)
    attempts = {}

    def execution(agent, messages):
        attempts[agent] = attempts.get(agent, 0) + 1
        return respond(content="hmm") if attempts[agent] == 1 else respond(call("set_price", price=55))

    summary = Runner(env, scripted(execution), Arm(), Horizon(1, 0.0, 1), concurrency=1).run()
    assert summary["valid_rounds"] == 1


def test_text_recovery_handles_phi_and_ignores_schema_echo():
    names = ["set_price", "announce_price"]
    phi = 'funcs([{"name": "set_price", "arguments": {"price": 55}}})'
    assert [(c.name, c.arguments) for c in parse_text_tool_calls(phi, names)] == [("set_price", {"price": 55})]
    wrapped = 'func(functions[{"name": "announce_price", "arguments": {"price": 0, "note": "x"}}])'
    assert parse_text_tool_calls(wrapped, names)[0].arguments == {"price": 0, "note": "x"}
    echo = '{"function": {"function": {"name": "announce_price", "description": "Publicly announce"}}}'
    assert parse_text_tool_calls(echo, names) == []
    two = '<tool_call>{"name": "announce_price", "arguments": {"price": 60}}</tool_call> then {"name": "set_price", "arguments": "{\\"price\\": 60}"}'
    assert [c.name for c in parse_text_tool_calls(two, names)] == ["announce_price", "set_price"]


def test_performance_metrics_in_summary():
    from agora.llm import ChatResponse

    env = make_env("price_war", {"costs": [40, 45.5, 52.8]}, seed=1)

    def execution(agent, messages):
        if agent == "seller_1":
            return ChatResponse("", "", [call("set_price", price=60)], "tool_calls", {"prompt_tokens": 100, "completion_tokens": 50})
        return respond(content='funcs([{"name": "set_price", "arguments": {"price": 61}}])', )

    from agora.llm import parse_text_tool_calls as ptc
    client = scripted(lambda a, m: (lambda r: (r.tool_calls.extend(ptc(r.content, ["set_price"])) if not r.tool_calls else None, r)[1])(execution(a, m)))
    s = Runner(env, client, Arm(), Horizon(1, 0.0, 1), concurrency=1).run()
    perf = s["performance"]
    assert perf["calls"] == 6 and perf["by_phase"]["execution"]["calls"] == 3
    assert perf["completion_tokens_total"] == 50 and perf["truncated_rate"] == 0
    assert perf["text_recovered_rate"] == round(2 / 3, 4)
    assert perf["wall_seconds"] >= 0 and perf["seconds_p95"] is not None


def test_code_style_call_recovery():
    names = ["hold", "drive_in_zone", "drive_offline", "charge", "set_price"]
    got = lambda t: [(c.name, c.arguments) for c in parse_text_tool_calls(t, names)]
    assert got("hold") == [("hold", {})]
    assert got("drive_offline()") == [("drive_offline", {})]
    assert got("drive_in_zone(zone: 2)") == [("drive_in_zone", {"zone": 2})]
    assert got("charge(amount=3)") == [("charge", {"amount": 3})]
    assert got('drive_in_zone {"name": "driver_4", "arguments": {"zone": 0}}') == [("drive_in_zone", {"zone": 0})]
    assert got('set_price(price="55")') == [("set_price", {"price": "55"})]
    # Mentions in prose are not calls; malformed argument lists are skipped.
    assert got("I think I should hold this round.") == []
    assert got("I will call set_price soon") == []
    assert got("drive_in_zone(2)") == []  # positional needs the schema
    # JSON-form calls still take precedence.
    assert got('{"name": "charge", "arguments": {"amount": 1}} charge(amount=9)') == [("charge", {"amount": 1})]


def test_positional_and_object_recovery_with_schema():
    req = {"hold": [], "drive_in_zone": ["zone"], "charge": ["amount"], "drive_offline": []}
    got = lambda t: [(c.name, c.arguments) for c in parse_text_tool_calls(t, req)]
    assert got("drive_in_zone(0)") == [("drive_in_zone", {"zone": 0})]
    assert got("charge 5") == [("charge", {"amount": 5})]
    assert got("charge 10 MWh") == [("charge", {"amount": 10})]
    assert got('drive_in_zone{"zone": 0}') == [("drive_in_zone", {"zone": 0})]
    assert got('drive_in_zone({"zone": 2})') == [("drive_in_zone", {"zone": 2})]
    assert got("hold\n\nGrid day 1, block 5 ...") == [("hold", {})]
    assert got("Hold") == [("hold", {})] and got("Charge") == []
    # Echoed schemas and prose are still not calls, including prose in parentheses.
    assert got("discharge? charge (up to 10 MWh capacity)") == []
    assert got("charge (nothing to sell)") == [] and got("charge (empty)") == []
    assert got('drive_in_zone\n{\n "name": "drive_in_zone", "function": {"description": "x"}}') == []
    assert got("I could charge the battery or hold.") == []
    assert got("charge the battery") == []


def test_echo_reasoning_only_when_enabled():
    from agora.llm import ChatResponse, ScriptedClient

    seen = []

    def policy(messages, tools):
        seen.append(messages)
        if "EXECUTION" in messages[1]["content"]:
            bad = len(messages) == 2  # first try: invalid price, then a valid retry
            return ChatResponse("", "thinking...", [call("set_price", price=500 if bad else 60)], "tool_calls")
        return respond()

    for echo in (False, True):
        seen.clear()
        client = ScriptedClient(policy)
        client.echo_reasoning = echo
        env = make_env("price_war", {"costs": [40, 45.5, 52.8]}, seed=1)
        Runner(env, client, Arm(), Horizon(1, 0.0, 1), concurrency=1).run()
        retry = [m for m in seen if len(m) > 2][0]
        prior = [x for x in retry if x["role"] == "assistant"][0]
        assert ("reasoning_content" in prior) == echo
        if echo:
            assert prior["reasoning_content"] == "thinking..."
