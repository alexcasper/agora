"""Base class for agora market scenarios.

A scenario owns the economics only: rules text, per-agent observations, tool
schemas, action validation, settlement and analytic benchmarks. Boards, arms,
the horizon and the LLM loop live in the runner.

Conventions every scenario follows:
- Each round has a planning phase (talk + non-binding public proposals) and an
  execution phase (exactly one binding commit per agent).
- A missing commit is never filled with a default action. The agent abstains
  for that round (sells nothing, bids nothing, holds, stays offline) and the
  round record lists it under `missing`. Summaries report the missing-action
  rate and compute headline metrics over valid rounds only.
- Rewards are reported per round and cumulatively; nothing reports the last
  round as if it were the whole game.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Set


def tool(name: str, description: str, properties: Dict[str, Any], required: Sequence[str] = ()) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required)},
        },
    }


def as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number != int(number):
        return None
    return int(number)


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fail(reason: str) -> Dict[str, Any]:
    return {"status": "failed", "reason": reason}


def ok(**fields: Any) -> Dict[str, Any]:
    return {"status": "success", **fields}


class MarketEnv:
    """Subclasses set `name`, `role` and `commit_tools`, and implement the hooks below."""

    name = "market"
    role = "participant"  # noun used in arm prompts ("seller", "bidder", ...)
    unit = "round"  # what one horizon unit is called in prompts ("market day", "auction", ...)
    rounds_per_unit = 1  # rounds per horizon unit (e.g. 4 pricing blocks per grid day)
    commit_tools: Set[str] = set()

    def __init__(self, params: Dict[str, Any], seed: int):
        self.params = dict(params or {})
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.agents: List[str] = []
        self.round = 0  # index of the round currently being played
        self.history: List[Dict[str, Any]] = []
        self.pending: Dict[str, Dict[str, Any]] = {}
        self.proposals: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}

    # ---- hooks -------------------------------------------------------- #

    def rules(self, agent: str) -> str:
        raise NotImplementedError

    def observation(self, agent: str, phase: str) -> str:
        raise NotImplementedError

    def proposal_tools(self, agent: str) -> List[Dict[str, Any]]:
        return []

    def action_tools(self, agent: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def propose(self, agent: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return fail(f"unknown tool {name}")

    def commit(self, agent: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and store a binding action in `self.pending[agent]`."""
        raise NotImplementedError

    def settle_round(self, missing: List[str]) -> Dict[str, Any]:
        """Settle `self.pending` for the current round. Must return a record with
        `profits` ({agent: float}); the base class adds round/missing fields."""
        raise NotImplementedError

    def benchmarks(self) -> Dict[str, Any]:
        return {}

    def metrics(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {}

    def execution_stages(self) -> List[List[str]]:
        """Groups of agents that commit in sequence; later groups may observe
        earlier groups' commits through `observation`. Default: simultaneous."""
        return [list(self.agents)]

    # ---- shared machinery -------------------------------------------- #

    def tools(self, agent: str, phase: str) -> List[Dict[str, Any]]:
        return self.proposal_tools(agent) if phase == "planning" else self.action_tools(agent)

    def handle(self, agent: str, name: str, args: Dict[str, Any], phase: str) -> Dict[str, Any]:
        if phase == "planning":
            result = self.propose(agent, name, args)
            if result.get("status") == "success":
                self.proposals.setdefault(self.round, {}).setdefault(agent, []).append(
                    {"tool": name, **{k: v for k, v in result.items() if k not in ("status", "post")}}
                )
            return result
        allowed = {t["function"]["name"] for t in self.action_tools(agent)}
        if name not in allowed:
            return fail(f"{name} is not one of your actions; use one of {sorted(allowed)}")
        if agent in self.pending:
            return fail("you already committed an action this round")
        result = self.commit(agent, name, args)
        if result.get("status") == "success":
            result["committed"] = True
        return result

    def last_proposal(self, round_index: int, agent: str) -> Optional[Dict[str, Any]]:
        entries = self.proposals.get(round_index, {}).get(agent) or []
        return entries[-1] if entries else None

    def settle(self) -> Dict[str, Any]:
        missing = [a for a in self.agents if a not in self.pending]
        record = self.settle_round(missing)
        record = {
            "round": self.round + 1,
            "missing": missing,
            "valid": not missing,
            "actions": {a: dict(self.pending[a]) for a in self.agents if a in self.pending},
            "proposals": {a: list(v) for a, v in self.proposals.get(self.round, {}).items()},
            **record,
        }
        record["profits"] = {a: float(record["profits"].get(a, 0.0)) for a in self.agents}
        record["joint_profit"] = round(sum(record["profits"].values()), 6)
        self.history.append(record)
        self.pending = {}
        self.round += 1
        return record

    def totals(self) -> Dict[str, float]:
        return {a: round(sum(r["profits"][a] for r in self.history), 6) for a in self.agents}

    def summary(self) -> Dict[str, Any]:
        n = len(self.history)
        slots = n * len(self.agents)
        missing = sum(len(r["missing"]) for r in self.history)
        valid = [r for r in self.history if r["valid"]]
        return {
            "scenario": self.name,
            "seed": self.seed,
            "params": self.params,
            "rounds": n,
            "valid_rounds": len(valid),
            "missing_actions": missing,
            "missing_action_rate": round(missing / slots, 4) if slots else None,
            "profit_totals": self.totals(),
            "joint_profit_total": round(sum(self.totals().values()), 6),
            "benchmarks": self.benchmarks(),
            "metrics_valid_rounds": self.metrics(valid),
            "metrics_all_rounds": self.metrics(self.history),
            "history": self.history,
        }


def mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 6) if values else None


def collusion_index(realized: Optional[float], competitive: float, collusive: float) -> Optional[float]:
    """0 at the competitive benchmark, 1 at the collusive one (Calvano-style)."""
    if realized is None or collusive == competitive:
        return None
    return round((realized - competitive) / (collusive - competitive), 4)
