"""Minimal OpenAI-compatible chat client with tool calling.

Works against llama.cpp, vLLM, OpenRouter, Nebius or any server exposing
`POST {base_url}/chat/completions`. Stdlib only, so the harness has no HTTP
dependency. `ScriptedClient` replays canned responses for tests.
"""

from __future__ import annotations

import json
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


def parse_text_tool_calls(content: str, tool_names: List[str]) -> List[ToolCall]:
    """Recover tool calls that a small model wrote as text instead of using the API.

    Scans the text for JSON objects of the form `{"name": <known tool>,
    "arguments": {...}}`, wherever they appear: inside `<tool_call>` tags
    (Qwen/Hermes), in a `funcs([...])` wrapper (Phi-4-mini), or bare. Trailing
    junk such as an extra brace is tolerated. Objects without an `arguments`
    dict (e.g. an echoed tool schema) and calls to unknown tools are ignored.
    Every recovered call is flagged so analysis can separate it from native calls.
    """
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
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self.timeout = timeout
        self.max_retries = max_retries
        self.params = dict(params or {})

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
                return self._to_response(payload, [t["function"]["name"] for t in tools])
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
    def _to_response(payload: Dict[str, Any], tool_names: List[str]) -> ChatResponse:
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
    )
