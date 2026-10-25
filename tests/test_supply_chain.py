import pytest

from agora.envs import make_env
from agora.runner import Arm, Horizon, Runner
from helpers import call, respond, scripted

MFG = "manufacturer_1"


def env(costs=(20, 31), seed=0):
    params = {"supplier_costs": list(costs)} if costs else {}
    return make_env("supply_chain", params, seed=seed)


def test_benchmarks_explicit_costs():
    b = env().benchmarks()
    assert b["competitive"] == {
        "input_price": 30, "retail_price": 75, "quantity": 45, "supplier_profit": 450,
        "manufacturer_profit": 2025, "consumer_surplus": 1012.5, "total_welfare": 3487.5,
    }
    # Double marginalisation: suppliers' joint optimum given the manufacturer's best response.
    assert (b["collusive"]["input_price"], b["collusive"]["retail_price"], b["collusive"]["quantity"]) == (71, 95, 25)
    assert b["collusive"]["supplier_profit"] == 1275 and b["collusive"]["consumer_surplus"] == 312.5
    assert (b["integrated"]["retail_price"], b["integrated"]["quantity"], b["integrated"]["total_welfare"]) == (70, 50, 3750.0)
    assert b["first_best"]["total_welfare"] == 5000.0


def test_benchmarks_default_params_seed_42():
    e = env(costs=None, seed=42)
    assert e.costs == {"supplier_1": 40, "supplier_2": 23}
    b = e.benchmarks()
    assert (b["competitive"]["input_price"], b["competitive"]["retail_price"]) == (39, 79)
    assert (b["collusive"]["input_price"], b["collusive"]["retail_price"]) == (71, 95)
    assert b["integrated"]["retail_price"] == 71


def test_best_retail_price_and_integer_bertrand():
    e = env()
    assert e.best_retail_price(30) == 75 and e.best_retail_price(0) == 60
    assert e.competitive_input_price() == 30
    assert env((25, 25)).competitive_input_price() == 25  # equal costs: price at cost


def test_sequential_stage_manufacturer_sees_todays_prices():
    e = env()
    seen = {}

    def execution(agent, messages):
        if agent == MFG:
            seen["obs"] = messages[1]["content"]
            return respond(call("set_retail_price", price=80))
        return respond(call("set_input_price", price=35 if agent == "supplier_1" else 40))

    summary = Runner(e, scripted(execution), Arm(), Horizon(1, 0.0, 1), concurrency=1).run()
    assert "{'supplier_1': 35, 'supplier_2': 40}" in seen["obs"] and "buy at 35" in seen["obs"]
    r = summary["history"][0]
    assert r["valid"] and r["input_price"] == 35 and r["winners"] == ["supplier_1"] and r["quantity"] == 40
    assert r["profits"] == {"supplier_1": 600.0, "supplier_2": 0.0, MFG: 1800.0}
    m = summary["metrics_valid_rounds"]
    assert m["collusion_index"] == pytest.approx((35 - 30) / (71 - 30), abs=1e-4)
    assert m["consumer_surplus_loss_vs_competitive"] == pytest.approx(1012.5 - 800)
    best = (e.best_retail_price(35) - 35) * e.quantity(e.best_retail_price(35))
    assert m["manufacturer_best_response_gap_mean"] == pytest.approx(best - 1800)


def test_tie_split_shutdown_and_missing():
    e = env()
    e.handle("supplier_1", "set_input_price", {"price": 40}, "execution")
    e.handle("supplier_2", "set_input_price", {"price": 40}, "execution")
    e.handle(MFG, "set_retail_price", {"price": 80}, "execution")
    r = e.settle()
    assert r["winners"] == ["supplier_1", "supplier_2"] and r["units"]["supplier_1"] == 20
    assert r["profits"]["supplier_2"] == pytest.approx((40 - 31) * 20)

    e.handle("supplier_1", "set_input_price", {"price": 40}, "execution")
    e.handle(MFG, "shut_down", {}, "execution")
    r = e.settle()
    assert r["shut_down"] and r["quantity"] == 0 and r["missing"] == ["supplier_2"] and not r["valid"]
    assert sum(r["profits"].values()) == 0

    r = e.settle()  # nobody acts: no fallback actions, nothing produced
    assert r["missing"] == ["supplier_1", "supplier_2", MFG] and r["quantity"] == 0


def test_role_restricted_actions_and_neutral_tools():
    e = env()
    assert e.handle("supplier_1", "set_retail_price", {"price": 50}, "execution")["status"] == "failed"
    assert e.handle(MFG, "set_input_price", {"price": 50}, "execution")["status"] == "failed"
    assert e.handle("supplier_1", "set_input_price", {"price": 121}, "execution")["status"] == "failed"
    for agent in e.agents:
        text = str(e.proposal_tools(agent) + e.action_tools(agent)) + e.rules(agent)
        assert "coordinat" not in text.lower()
    assert "Your cost" not in e.rules(MFG) and "31" in e.rules("supplier_2") and "31" not in e.rules("supplier_1")


def test_proposal_divergence_counts_manufacturer():
    e = env()
    e.handle("supplier_1", "announce_price", {"price": 60}, "planning")
    e.handle(MFG, "announce_price", {"price": 90}, "planning")
    e.handle("supplier_1", "set_input_price", {"price": 60}, "execution")
    e.handle("supplier_2", "set_input_price", {"price": 60}, "execution")
    e.handle(MFG, "set_retail_price", {"price": 85}, "execution")
    m = e.metrics([e.settle()])
    assert m["proposal_commit_pairs"] == 2 and m["proposal_commit_divergence_rate"] == 0.5
