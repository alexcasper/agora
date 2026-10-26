"""Replay replies that produced no tool call through the current text-call parser.

Runs recorded with an older parser (see docs/MODELS.md, caveat 5) may have
missed calls the current one would recover. For each model this counts
no-call replies, how many the current parser would turn into a call to a tool
the agent actually had, and how many missed execution turns that would have
rescued. It is an estimate: a rescued turn would have changed the rest of the
game, which a replay cannot show.

    python tools/replay_parser.py runs-modal/main-20261025 runs-modal/main-20261025-phi-r2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agora.envs import make_env  # noqa: E402
from agora.llm import parse_text_tool_calls  # noqa: E402
from agora.runner import post_message_tool  # noqa: E402


def required_params(tools: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    return {t["function"]["name"]: t["function"].get("parameters", {}).get("required", []) for t in tools}


def tool_sets(summary: Dict[str, Any], start: Dict[str, Any]) -> Dict[tuple, Dict[str, List[str]]]:
    """Tools per (agent, phase) as offered at the start of the game."""
    params = (summary.get("config") or {}).get("params") or summary.get("params") or {}
    env = make_env(summary["scenario"], params, summary["seed"])
    out = {}
    for agent in env.agents:
        out[(agent, "planning")] = required_params(env.tools(agent, "planning") + [post_message_tool(["public", "private"])])
        out[(agent, "execution")] = required_params(env.tools(agent, "execution"))
    return out


def replay_game(game: Path, stats: Counter) -> None:
    summary = json.loads((game / "summary.json").read_text())
    events = [json.loads(line) for line in (game / "events.jsonl").read_text().splitlines() if line.strip()]
    start = next((e for e in events if e.get("event") == "start"), {})
    tools = tool_sets(summary, start)
    turns: Dict[tuple, Dict[str, bool]] = defaultdict(lambda: {"committed": False, "rescuable": False})
    for e in events:
        if e.get("event") == "tool_result" and e.get("phase") == "execution":
            if (e.get("result") or {}).get("committed"):
                turns[(e["round"], e["agent"])]["committed"] = True
        if e.get("event") != "llm_call":
            continue
        phase = e["phase"]
        stats[f"{phase}_calls"] += 1
        calls = e.get("tool_calls") or []
        if calls:
            stats[f"{phase}_text_recovered_old"] += sum(1 for c in calls if c.get("parsed_from_text"))
            continue
        stats[f"{phase}_no_call"] += 1
        if e.get("finish_reason") == "length":
            stats[f"{phase}_no_call_truncated"] += 1
        if phase == "execution":
            turns[(e["round"], e["agent"])]  # a turn with at least one no-call reply
        allowed = tools.get((e["agent"], phase), {})
        recovered = [c for c in parse_text_tool_calls(e.get("content") or "", allowed) if c.name in allowed]
        if recovered:
            stats[f"{phase}_recovered_new"] += 1
            if phase == "execution":
                turns[(e["round"], e["agent"])]["rescuable"] = True
    for turn in turns.values():
        if not turn["committed"]:
            stats["missed_turns_with_no_call"] += 1
            stats["missed_turns_rescuable"] += turn["rescuable"]
    stats["games"] += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path, help="<tag> directories holding <model>/<game>/")
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    by_model: Dict[str, Counter] = defaultdict(Counter)
    for root in args.roots:
        for events in sorted(root.glob("*/*/events.jsonl")):
            model = events.parent.parent.name
            if model in args.exclude or not (events.parent / "summary.json").exists():
                continue
            try:
                replay_game(events.parent, by_model[model])
            except (OSError, ValueError, KeyError) as exc:
                by_model[model]["unreadable_games"] += 1
                print(f"skip {events.parent}: {exc}", file=sys.stderr)
    print(json.dumps({m: dict(c) for m, c in sorted(by_model.items())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
