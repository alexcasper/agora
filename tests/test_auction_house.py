import pytest

from agora.envs import make_env
from agora.runner import Arm, Horizon, Runner
from helpers import call, respond, scripted


def bne_bidding(env):
    return lambda a, m: respond(call("submit_bid", bid=round(env.bne_bid(env.value(env.round, a)))))


def test_bne_rule():
    e = make_env("auction_house", {}, seed=1)
    assert e.bne_bid(50) == 50 and e.bne_bid(150) == pytest.approx(50 + 2 / 3 * 100)
    assert e.benchmarks()["expected_bne_revenue_per_auction"] == pytest.approx(50 + 2 / 3 * 75)


def test_equilibrium_bidding_scores_near_zero():
    e = make_env("auction_house", {}, seed=3)
    s = Runner(e, scripted(bne_bidding(e)), Arm(), Horizon(6, 0.0, 6), concurrency=1).run()
    m = s["metrics_valid_rounds"]
    assert abs(m["collusion_index"]) < 0.02
    assert m["mean_bid_over_bne_bid"] == pytest.approx(1.0, abs=0.02)
    assert m["efficiency_rate"] == 1.0


def test_ring_at_floor_scores_one_and_bids_are_published():
    e = make_env("auction_house", {}, seed=3)
    s = Runner(e, scripted(lambda a, m: respond(call("submit_bid", bid=10))), Arm(), Horizon(3, 0.0, 3), concurrency=1).run()
    assert s["metrics_valid_rounds"]["collusion_index"] == pytest.approx(1.0)
    assert "bids {'bidder_1': 10" in e.observation("bidder_1", "planning")


def test_values_independent_of_tie_breaks():
    a = make_env("auction_house", {}, seed=9)
    b = make_env("auction_house", {}, seed=9)
    for e, bid in ((a, 10), (b, 60)):
        for _ in range(4):
            for agent in e.agents:
                e.value(e.round, agent)
                e.handle(agent, "submit_bid", {"bid": bid}, "execution")
            e.settle()
    assert a.values == b.values


def test_missing_bidder_does_not_take_part():
    e = make_env("auction_house", {}, seed=2)
    s = Runner(e, scripted(lambda a, m: respond() if a == "bidder_2" else respond(call("submit_bid", bid=40))), Arm(), Horizon(1, 0.0, 1), concurrency=1).run()
    r = s["history"][0]
    assert "bidder_2" not in r["bids"] and r["missing"] == ["bidder_2"] and not r["valid"]


def test_ring_arm_is_symmetric_and_no_tipoff():
    e = make_env("auction_house", {}, seed=2)
    arm = Arm.from_config({"name": "ring", "coalition": 3, "coalition_prompt": True, "private_channel": True}, e.agents)
    r = Runner(e, scripted(bne_bidding(e)), arm, Horizon(1, 0.0, 1))
    assert all("PRIVATE ARRANGEMENT" in r.system_prompt(a) for a in e.agents)
    outsider_arm = Arm.from_config({"name": "pair", "coalition": 2, "coalition_prompt": True}, e.agents)
    r2 = Runner(e, scripted(bne_bidding(e)), outsider_arm, Horizon(1, 0.0, 1))
    assert "arrangement" not in r2.system_prompt("bidder_3").lower()
