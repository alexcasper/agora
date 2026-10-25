"""Minimal OpenAI-compatible chat client with tool calling.

Works against llama.cpp, vLLM, OpenRouter, Nebius or any server exposing
`POST {base_url}/chat/completions`. Stdlib only, so the harness has no HTTP
dependency. `ScriptedClient` replays canned responses for tests.
"""

from __future__ import annotations

import json
import re
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    parsed_from_text: bool = False


@dataclass
class ChatResponse:
    content: str
    reasoning: str
    tool_calls: List[ToolCall]
    finish_reason: str
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {"_unparsed": raw}
    return value if isinstance(value, dict) else {"_unparsed": raw}


def parse_text_tool_calls(content: str, tools: Any) -> List[ToolCall]:
    """Recover tool calls that a small model wrote as text instead of using the API.

    Scans the text for JSON objects of the form `{"name": <known tool>,
    "arguments": {...}}`, wherever they appear: inside `<tool_call>` tags
    (Qwen/Hermes), in a `funcs([...])` wrapper (Phi-4-mini), or bare. Trailing
    junk such as an extra brace is tolerated. Objects without an `arguments`
    dict (e.g. an echoed tool schema) and calls to unknown tools are ignored.
    Every recovered call is flagged so analysis can separate it from native calls.

    `tools` is a list of tool names, or a dict mapping each name to its required
    parameter names (which lets a single positional argument be recovered).
    """
    required = dict(tools) if isinstance(tools, dict) else {name: [] for name in tools}
    tool_names = list(required)
    calls: List[ToolCall] = []
    text = content or ""
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except ValueError:
            pos = start + 1
            continue
        if isinstance(obj, dict) and obj.get("name") in tool_names and isinstance(obj.get("arguments"), (dict, str)):
            calls.append(
                ToolCall(
                    id=f"text_{len(calls)}",
                    name=obj["name"],
                    arguments=_parse_arguments(obj["arguments"]),
                    parsed_from_text=True,
                )
            )
            pos = end
        else:
            pos = start + 1
    return calls or _parse_code_style_calls(text, required)


def _literal(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except ValueError:
        return raw.strip("'\"")


def _split_top_level(text: str) -> List[str]:
    parts, depth, quote, current = [], 0, None, []
    for ch in text:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return [p for p in parts if p.strip()]


def _parse_code_style_calls(text: str, required: Dict[str, List[str]]) -> List[ToolCall]:
    """Fallback for calls written as code rather than JSON (seen from Phi-4-mini):

    - `name(k=v, ...)` / `name(k: v, ...)` with an exact tool name
    - `name {...}` / `name({...})`, using the object's `arguments` if present,
      otherwise the object itself as the arguments
    - `name(v)` / `name v` when the tool has exactly one required parameter
    - a reply whose first line is just a tool name with no required parameters

    Only exact tool names count, and a name merely mentioned in prose ("I will
    use set_price") is not a call: it must be followed by `(`, `{`, a single
    value, or stand alone on the first line.
    """
    calls: List[ToolCall] = []
    if not required:
        return calls
    names = "|".join(re.escape(n) for n in sorted(required, key=len, reverse=True))
    decoder = json.JSONDecoder()

    def object_args(start: int) -> Optional[Dict[str, Any]]:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except ValueError:
            return None
        if not isinstance(obj, dict):
            return None
        if isinstance(obj.get("arguments"), dict):
            return obj["arguments"]
        if "function" in obj or "name" in obj or "description" in obj:
            return None  # an echoed tool schema, not arguments
        return obj

    def single(name: str, raw: str) -> Optional[Dict[str, Any]]:
        params = required.get(name) or []
        return {params[0]: _literal(raw)} if len(params) == 1 and raw.strip() else None

    for match in re.finditer(r"(?<![\w.])(" + names + r")\s*(\(\s*\{|\{|\(|[ \t]+(?=[-\w\"'.]))", text):
        name, opener = match.group(1), match.group(2)
        args: Optional[Dict[str, Any]]
        if "{" in opener:
            args = object_args(match.end(2) - 1)
        elif opener.startswith("("):
            close = text.find(")", match.end(2))
            if close < 0:
                continue
            inner = text[match.end(2):close]
            parts = _split_top_level(inner)
            if len(parts) == 1 and not re.search(r"^\s*['\"]?\w+['\"]?\s*[=:]", parts[0]):
                args = single(name, parts[0])
            else:
                args = {}
                for part in parts:
                    pair = re.split(r"\s*[=:]\s*", part.strip(), maxsplit=1)
                    key = pair[0].strip("'\"")
                    if len(pair) != 2 or not re.fullmatch(r"\w+", key):
                        args = None
                        break
                    args[key] = _literal(pair[1])
        else:
            # `name value` at the start of a line with a numeric value, e.g. `charge 5`
            # or `charge 10 MWh`; prose such as "charge the battery" never matches.
            line_start = match.start() == 0 or text[match.start() - 1] == "\n"
            token = re.match(r"-?\d+(?:\.\d+)?(?![\w.])", text[match.end(2):])
            args = single(name, token.group(0)) if (token and line_start) else None
        if args is not None:
            calls.append(ToolCall(id=f"text_{len(calls)}", name=name, arguments=args, parsed_from_text=True))
    if not calls:
        first = text.strip().splitlines()[0].strip().strip("`.()").strip().lower() if text.strip() else ""
        bare = {n.lower(): n for n, params in required.items() if not params}
        if first in bare:
            calls.append(ToolCall(id="text_0", name=bare[first], arguments={}, parsed_from_text=True))
    return calls

class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        params: Optional[Dict[str, Any]] = None,
        echo_reasoning: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self.timeout = timeout
        self.max_retries = max_retries
        self.params = dict(params or {})
        # Send each prior assistant turn's reasoning back as `reasoning_content`.
        self.echo_reasoning = echo_reasoning

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ChatResponse:
        body: Dict[str, Any] = {"model": self.model, "messages": messages, **self.params}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=data, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read())
                required = {t["function"]["name"]: t["function"].get("parameters", {}).get("required", []) for t in tools}
                return self._to_response(payload, required)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:500]
                last_error = LLMError(f"HTTP {exc.code}: {detail}")
                # Context overflow and malformed requests will not succeed on retry.
                if exc.code == 400:
                    break
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
            time.sleep(2 ** attempt)
        raise LLMError(f"chat request failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _to_response(payload: Dict[str, Any], tool_names: Any) -> ChatResponse:
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        calls = [
            ToolCall(
                id=tc.get("id") or f"call_{idx}",
                name=(tc.get("function") or {}).get("name", ""),
                arguments=_parse_arguments((tc.get("function") or {}).get("arguments")),
            )
            for idx, tc in enumerate(message.get("tool_calls") or [])
        ]
        if not calls:
            calls = parse_text_tool_calls(content, tool_names)
        return ChatResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=calls,
            finish_reason=choice.get("finish_reason") or "",
            usage=payload.get("usage") or {},
            raw=payload,
        )


class ScriptedClient:
    """Test double: `policy(agent, phase, messages, tools)` returns a ChatResponse."""

    def __init__(self, policy: Callable[..., ChatResponse]):
        self.policy = policy
        self.calls = 0

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ChatResponse:
        self.calls += 1
        return self.policy(messages, tools)


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"scripted_{name}", name=name, arguments=arguments)


def respond(*calls: ToolCall, content: str = "") -> ChatResponse:
    return ChatResponse(content=content, reasoning="", tool_calls=list(calls), finish_reason="stop")


def build_client(cfg: Dict[str, Any]) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=cfg["base_url"],
        model=cfg["model"],
        api_key=cfg.get("api_key"),
        api_key_env=cfg.get("api_key_env"),
        timeout=float(cfg.get("timeout", 600)),
        max_retries=int(cfg.get("max_retries", 3)),
        params=cfg.get("params"),
        echo_reasoning=bool(cfg.get("echo_reasoning", False)),
    )
