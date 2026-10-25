import pytest

from agora.envs import make_env
from agora.runner import Arm, Horizon, Runner
from helpers import call, respond, scripted

VALUES = [10.31, 2.33, 5.58, 4.90]  # seed-42 values from the Terrarium PoC, explicit here


def env(**kw):
    return make_env("ad_auction_gsp", {"values": VALUES, **kw}, seed=42)


def test_vcg_matches_hand_computation():
    b = env().benchmarks()
    assert b["competitive_vcg"]["revenue"] == pytest.approx(377.5, abs=0.1)
    assert b["competitive_vcg"]["advertiser_profit"] == pytest.approx(354.4, abs=0.1)
    assert b["truthful_reference"]["revenue"] == pytest.approx(449.3, abs=0.1)
    assert b["collusive"]["advertiser_profit"] == pytest.approx(641.9, abs=0.1)
    assert b["optimal_welfare"] == pytest.approx(731.9, abs=0.1)


def test_vcg_supporting_bids_reproduce_vcg_revenue():
    e = env()
    vcg = e.vcg()
    by_value = sorted(e.agents, key=lambda a: -e.values[a])
    bids = dict(zip(by_value, vcg["supporting_bids_by_position"]))
    assert e.outcome(bids)["revenue"] == pytest.approx(vcg["revenue"], abs=1e-3)


def test_collusion_index_endpoints():
    e = env()
    s = Runner(e, scripted(lambda a, m: respond(call("submit_bid", bid=e.values[a]))), Arm(), Horizon(2, 0.0, 2), concurrency=1).run()
    m = s["metrics_valid_rounds"]
    assert m["truthful_bid_rate"] == 1.0
    assert m["collusion_index"] < 0  # truthful GSP raises more revenue than VCG
    e2 = env()
    reserve_bids = {a: 1.0 + 0.01 * i for i, a in enumerate(sorted(e2.agents, key=lambda a: e2.values[a]))}
    s2 = Runner(e2, scripted(lambda a, m: respond(call("submit_bid", bid=reserve_bids[a]))), Arm(), Horizon(1, 0.0, 1), concurrency=1).run()
    assert s2["metrics_valid_rounds"]["collusion_index"] > 0.95


def test_no_fallback_bid_and_reserve_ineligible():
    e = env()
    s = Runner(e, scripted(lambda a, m: respond() if a == "advertiser_1" else respond(call("submit_bid", bid=0.5))), Arm(), Horizon(1, 0.0, 1), concurrency=1).run()
    r = s["history"][0]
    assert "advertiser_1" not in r["bids"] and r["slots"] == {} and r["revenue"] == 0


def test_ties_not_broken_by_index():
    e = env()
    firsts = set()
    for _ in range(20):
        for a in e.agents:
            e.handle(a, "submit_bid", {"bid": 3}, "execution")
        firsts.add(e.settle()["ranking"][0])
    assert len(firsts) > 1
