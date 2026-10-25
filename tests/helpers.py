"""Shared test helpers: scripted LLM policies that drive the real runner."""

from __future__ import annotations

import re
from typing import Callable, Dict, List

from agora.llm import ChatResponse, ScriptedClient, call, respond

_AGENT = re.compile(r"You are (\w+)")


def agent_of(messages: List[Dict]) -> str:
    return _AGENT.search(messages[0]["content"]).group(1)


def phase_of(messages: List[Dict]) -> str:
    return "planning" if "PLANNING PHASE" in messages[1]["content"] else "execution"


def scripted(execution: Callable[[str, List[Dict]], ChatResponse], planning=None) -> ScriptedClient:
    """Planning: optional `planning(agent, messages)` once, then end turn.
    Execution: `execution(agent, messages)` until a commit lands."""

    def policy(messages, tools):
        agent = agent_of(messages)
        if phase_of(messages) == "planning":
            already_acted = any(m["role"] == "tool" for m in messages)
            if planning is None or already_acted:
                return respond()
            return planning(agent, messages)
        return execution(agent, messages)

    return ScriptedClient(policy)


__all__ = ["agent_of", "phase_of", "scripted", "call", "respond"]
