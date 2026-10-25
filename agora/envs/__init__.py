"""Scenario registry. Keys are the names used in configs."""

from __future__ import annotations

import importlib
from typing import Dict, Type

from ..env import MarketEnv

_SCENARIOS: Dict[str, str] = {
    "price_war": "agora.envs.price_war:PriceWar",
    "auction_house": "agora.envs.auction_house:AuctionHouse",
    "supply_chain": "agora.envs.supply_chain:SupplyChain",
    "grid_storage": "agora.envs.grid_storage:GridStorage",
    "ride_hailing": "agora.envs.ride_hailing:RideHailing",
    "ad_auction_gsp": "agora.envs.ad_auction_gsp:AdAuctionGSP",
}


def scenario_names():
    return sorted(_SCENARIOS)


def load_scenario(name: str) -> Type[MarketEnv]:
    if name not in _SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {scenario_names()}")
    module, cls = _SCENARIOS[name].split(":")
    return getattr(importlib.import_module(module), cls)


def make_env(name: str, params: dict, seed: int) -> MarketEnv:
    return load_scenario(name)(params or {}, seed)
