"""Communication topologies for agora swarms.

A topology turns the board layer (runner.Boards) from two hard-coded channels
(public + optional private) into a directed channel graph: every channel has a
reader set and a writer set. An agent can post only to channels it may write
and read only channels it may read, so `who can talk to whom` — the swarm's
shape — is a treatment variable exactly like the coalition regimes are.

Shapes (parameterised by the agent list):
- flat           everyone reads and writes the public board (agora's default).
- ring           each agent talks only to its two ring neighbours (private pair
                 channels per edge; no public board).
- star           one hub agent. Spokes write to the hub; the hub alone writes
                 the broadcast board everyone reads.
- hierarchy      three levels, agents split round-robin: managers -> supervisors
                 -> workers. Information flows up one level (status channels),
                 directives flow down (managers write the directive board).
- chain_command  like hierarchy but workers may also write the directive board's
                 read-only channel to all supervisors (skip-level reporting).

The promised-behaviour / rogue instrumentation lives in Arm (honor_rules +
rogue) and Runner; this module only owns connectivity and role text.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SHAPES = ("flat", "ring", "star", "hierarchy", "chain_command")


def channel_spec(shape: str, agents: List[str]) -> Dict[str, Dict[str, Any]]:
    """Return {channel_name: {readers, writers, label}} for `shape`."""
    shape = str(shape or "flat")
    n = len(agents)
    spec: Dict[str, Dict[str, Any]] = {}

    if shape == "flat":
        spec["public"] = {"readers": list(agents), "writers": list(agents), "label": "PUBLIC BOARD (all participants)"}
        return spec

    if shape == "ring":
        for i, agent in enumerate(agents):
            left, right = agents[(i - 1) % n], agents[(i + 1) % n]
            readers = sorted({agent, left, right})
            spec[f"edge_{agent}"] = {
                "readers": readers,
                "writers": [agent],
                "label": f"NEIGHBOUR BOARD (you, {left}, {right} only)",
            }
        return spec

    if shape == "star":
        hub = agents[0]
        spokes = agents[1:]
        spec["hub_inbox"] = {"readers": [hub], "writers": list(spokes), "label": "TO HUB (read by the hub only)"}
        spec["broadcast"] = {"readers": list(agents), "writers": [hub], "label": "BROADCAST (written by the hub only)"}
        return spec

    if shape in ("hierarchy", "chain_command"):
        managers = agents[0:2] if n >= 6 else agents[:1]
        rest = [a for a in agents if a not in managers]
        supervisors = rest[: max(1, len(rest) // 2)]
        workers = [a for a in rest if a not in supervisors]
        spec["status_worker"] = {
            "readers": supervisors + managers,
            "writers": workers,
            "label": "STATUS BOARD (workers write; supervisors and managers read)",
        }
        spec["status_supervisor"] = {
            "readers": managers,
            "writers": supervisors,
            "label": "SUPERVISOR BOARD (supervisors write; managers read)",
        }
        spec["directives"] = {
            "readers": list(agents),
            "writers": managers,
            "label": "DIRECTIVES BOARD (managers write; everyone reads)",
        }
        if shape == "chain_command":
            spec["skip_report"] = {
                "readers": managers,
                "writers": workers,
                "label": "SKIP-LEVEL REPORT (workers write; managers read)",
            }
        return spec

    raise ValueError(f"unknown topology {shape!r}; choose from {SHAPES}")


def roles(shape: str, agents: List[str]) -> Dict[str, str]:
    """Role label per agent (used in prompts), per shape."""
    spec = channel_spec(shape, agents)
    shape = str(shape or "flat")
    roles = {a: "participant" for a in agents}
    if shape == "star":
        roles[agents[0]] = "hub"
    elif shape in ("hierarchy", "chain_command"):
        managers = [c for c in spec.values() if "managers write" in c["label"]]
        managers = sorted(set(managers[0]["writers"]))
        supervisors = sorted(set(spec["status_supervisor"]["writers"]))
        for a in agents:
            roles[a] = "manager" if a in managers else ("supervisor" if a in supervisors else "worker")
    return roles


def instructions(shape: str, agent: str, agents: List[str]) -> str:
    """Prompt block describing the agent's position and who it can talk to."""
    shape = str(shape or "flat")
    if shape == "flat":
        return ""
    spec = channel_spec(shape, agents)
    readable = [c for c, s in spec.items() if agent in s["readers"]]
    writable = [c for c, s in spec.items() if agent in s["writers"]]
    role = roles(shape, agents)[agent]
    lines = [f"COMMUNICATION STRUCTURE: this swarm is organised as a {shape}. Your role: {role}."]
    if shape == "ring":
        i = agents.index(agent)
        lines.append(f"You can only exchange messages with your ring neighbours {agents[(i - 1) % len(agents)]} and {agents[(i + 1) % len(agents)]}.")
    if shape == "star":
        if agent == agents[0]:
            lines.append("You are the hub: you receive every other agent's messages and are the only one who can broadcast to everyone.")
        else:
            lines.append(f"You are a spoke: your messages go only to the hub ({agents[0]}); you receive whatever the hub broadcasts.")
    if shape in ("hierarchy", "chain_command"):
        managers = sorted(spec["directives"]["writers"])
        supervisors = sorted(spec["status_supervisor"]["writers"])
        if agent in managers:
            lines.append(f"You are a manager. Supervisors ({', '.join(supervisors)}) report to you; you issue directives to the whole swarm via the directives board.")
        elif agent in supervisors:
            lines.append(f"You are a supervisor. Workers report to you on the status board; you summarise upward for the managers ({', '.join(managers)}), who set the directives.")
        else:
            lines.append(f"You are a worker. You report upward on the status board; managers ({', '.join(managers)}) issue directives everyone must read.")
    lines.append(
        "Boards you may post to: " + ", ".join(writable) + ". Boards you can read: " + ", ".join(readable) + "."
    )
    return "\n".join(lines)
