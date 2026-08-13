#!/usr/bin/env python3
"""A deliberately thin external agent, for the Sprint 1 external-agent proof.

    python scripts/agent_reference_client.py \
        --base-url http://localhost:8000 \
        --resource ERE/source_portfolio/direct_001 \
        --provider anthropic

What this program is
--------------------
An *outside* caller. It talks to Trakt over HTTP and nothing else. There is not
one Trakt import in this file — no ``trakt_core``, no ``trakt_tools``, no
``mi_agent``, no dataframe, no configuration, no database. Only the standard
library and (optionally) a model SDK.
``tests/test_agent_reference_client.py::test_the_reference_client_imports_nothing_from_trakt``
asserts that, because the claim is the entire point: if this file could reach
inside Trakt, the demonstration would prove nothing about an external agent.

What the model does and does not do
-----------------------------------
The model decides **which tool to call, and with what arguments**. That is the
only inference in the loop. It does not compute a covenant utilisation, a
headroom or a breach amount, and it is instructed not to state a number Trakt
did not return. Every figure in the final narrative came from a tool result.

Model-agnostic by construction
------------------------------
Trakt publishes JSON Schema. Each provider adapter below translates that into
its own tool format at *this* boundary — nothing provider-specific ever reaches
Trakt, and adding a provider is one class here.

    --provider anthropic   Claude, via the `anthropic` SDK
    --provider openai      GPT, via the `openai` SDK
    --provider scripted    no model at all: a fixed tool sequence

``scripted`` exists to prove the point the other two cannot. If the same
workflow completes with no model in the loop, then the permissions, the
calculation, the evidence and the audit trail are properties of *Trakt* — not of
whichever model happened to be driving. It is also what runs in CI, where there
is no API key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

SYSTEM_INSTRUCTION = """\
You are a credit analyst reviewing a loan portfolio through Trakt.

Trakt is the system of record. It holds the governed portfolio data, applies the
operator-approved test definitions and performs every calculation. You decide
what to ask it.

Rules, in order of importance:

1. NEVER state a figure Trakt did not return. Do not compute, estimate,
   annualise, convert, average or infer any number yourself. If you want a
   number, call a tool.
2. Every tool result carries a `lineage` block naming the approved configuration
   version and the person who approved it, and a `snapshot` naming the dataset
   and its content hash. Cite those when you report a finding.
3. Check `source` on a covenant result. Only "approved_configuration" is
   operator-approved. "legacy_extracted" is indicative and must be labelled as
   such.
4. If a call is refused, read `error.code` and stop asking for that thing. A
   refusal is a permission decision, not an obstacle to work around. Report what
   you could not see rather than filling the gap.
5. If data is unavailable, say so plainly. An honest gap is a finding.

Your task: review the portfolio for covenant and concentration issues. Report
breaches first, then anything approaching its limit, then anything you could not
determine. Be brief and specific."""


# --------------------------------------------------------------------------- #
# The Trakt client — plain HTTP, no Trakt imports
# --------------------------------------------------------------------------- #
class TraktAgentAPI:
    """The whole of this agent's access to Trakt."""

    def __init__(self, base_url: str, token: Optional[str] = None,
                 correlation_id: Optional[str] = None, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.correlation_id = correlation_id or f"agent-run-{uuid.uuid4().hex[:12]}"
        self.timeout = timeout
        #: Every call made, in order. This is the client-side half of the audit
        #: trail; Trakt keeps the authoritative one, joined by correlation id.
        self.calls: List[Dict[str, Any]] = []

    def _request(self, method: str, path: str,
                 body: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json",
                   "X-Correlation-Id": self.correlation_id}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, data=data, headers=headers,
                                         method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            # A governed refusal arrives as a non-2xx with a full envelope. It is
            # a normal outcome for an agent, not a transport failure.
            raw = exc.read() or b"{}"
            try:
                return exc.code, json.loads(raw)
            except json.JSONDecodeError:
                return exc.code, {"error": {"code": "TRANSPORT_ERROR",
                                            "message": raw.decode("utf-8", "replace")}}

    def list_tools(self) -> Dict[str, Any]:
        status, payload = self._request("GET", "/v1/agent/tools")
        if status != 200:
            raise SystemExit(
                f"Could not read the tool catalogue ({status}): "
                f"{payload.get('error', {}).get('message', payload)}")
        return payload

    def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        status, payload = self._request(
            "POST", f"/v1/agent/tools/{tool_name}", arguments)
        self.calls.append({"tool": tool_name, "arguments": arguments,
                           "http_status": status,
                           "status": payload.get("status"),
                           "request_id": payload.get("request_id"),
                           "error_code": (payload.get("error") or {}).get("code")})
        return payload


# --------------------------------------------------------------------------- #
# Provider adapters — JSON Schema in, provider tool format out
# --------------------------------------------------------------------------- #
class ScriptedProvider:
    """No model. A fixed tool sequence over whatever the catalogue offers.

    Proves the workflow is Trakt's: same permissions, same calculations, same
    evidence, same audit trail, with nothing reasoning in the middle.
    """

    name = "scripted"

    def __init__(self, resource: str, tool: Optional[str] = None):
        self.resource = resource
        #: Name one tool, or leave it unset to take the first the catalogue
        #: offers that this provider can satisfy.
        self.tool = tool
        self._done = False

    def next_call(self, tools: List[Dict[str, Any]],
                  history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self._done or not tools:
            return None
        # The first tool the catalogue offers that this provider can actually
        # satisfy — one whose only required argument is the resource. Reading
        # that off the PUBLISHED schema rather than hard-coding a tool name is
        # the point: a scripted client with no model still discovers what it may
        # call, and adding a tool with more required arguments cannot break it.
        callable_tools = [
            t for t in tools
            if set((t.get("input_schema") or {}).get("required") or ())
            <= {t.get("resource_argument") or "resource"}]
        if self.tool:
            callable_tools = [t for t in callable_tools if t["name"] == self.tool]
        if not callable_tools:
            return None
        self._done = True
        return {"tool": callable_tools[0]["name"],
                "arguments": {"resource": self.resource}}

    def summarise(self, history: List[Dict[str, Any]]) -> str:
        lines = ["Scripted run — no model was involved.", ""]
        for entry in history:
            result = entry["result"]
            if result.get("status") != "success":
                code = (result.get("error") or {}).get("code")
                lines.append(f"- {entry['tool']}: refused ({code})")
                continue
            payload = result.get("result") or {}
            summary = payload.get("summary") or {}
            lines.append(
                f"- {entry['tool']}: {summary.get('active_tests')} active test(s), "
                f"{summary.get('breaches')} breach(es), "
                f"{summary.get('warnings')} warning(s); "
                f"overall {summary.get('overall_status')}")
            for test in payload.get("tests") or []:
                if test.get("status") in ("breach", "warning"):
                    lines.append(
                        f"    {test.get('display_name')}: "
                        f"{test.get('current_value')} vs limit "
                        f"{test.get('threshold')} ({test.get('status')})")
        return "\n".join(lines)


class AnthropicProvider:
    """Claude. Trakt's JSON Schema becomes an Anthropic tool definition here."""

    name = "anthropic"

    def __init__(self, model: str, max_turns: int = 8):
        import anthropic  # imported lazily so the SDK is optional

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_turns = max_turns
        self.messages: List[Dict[str, Any]] = []
        self._tools: List[Dict[str, Any]] = []
        self._final = ""

    @staticmethod
    def _translate(tool: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": tool["name"],
            "description": f"{tool['description']}\n\n{tool.get('agent_guidance', '')}".strip(),
            "input_schema": tool["input_schema"],
        }

    def start(self, tools: List[Dict[str, Any]], task: str) -> None:
        self._tools = [self._translate(t) for t in tools]
        self.messages = [{"role": "user", "content": task}]

    def next_call(self, tools, history) -> Optional[Dict[str, Any]]:
        if len(history) >= self.max_turns:
            return None
        response = self.client.messages.create(
            model=self.model, max_tokens=2048, system=SYSTEM_INSTRUCTION,
            tools=self._tools, messages=self.messages)
        self.messages.append({"role": "assistant", "content": response.content})
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                self._pending_id = block.id
                return {"tool": block.name, "arguments": dict(block.input)}
        self._final = "".join(getattr(b, "text", "") for b in response.content)
        return None

    def record_result(self, result: Dict[str, Any]) -> None:
        self.messages.append({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": self._pending_id,
            "content": json.dumps(result)}]})

    def summarise(self, history) -> str:
        return self._final or "(the model produced no closing narrative)"


class OpenAIProvider:
    """GPT. The same JSON Schema becomes an OpenAI function definition."""

    name = "openai"

    def __init__(self, model: str, max_turns: int = 8):
        import openai  # imported lazily so the SDK is optional

        self.client = openai.OpenAI()
        self.model = model
        self.max_turns = max_turns
        self.messages: List[Dict[str, Any]] = []
        self._tools: List[Dict[str, Any]] = []
        self._final = ""
        self._pending_id = ""

    @staticmethod
    def _translate(tool: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": tool["name"],
            "description": f"{tool['description']}\n\n{tool.get('agent_guidance', '')}".strip(),
            "parameters": tool["input_schema"]}}

    def start(self, tools: List[Dict[str, Any]], task: str) -> None:
        self._tools = [self._translate(t) for t in tools]
        self.messages = [{"role": "system", "content": SYSTEM_INSTRUCTION},
                         {"role": "user", "content": task}]

    def next_call(self, tools, history) -> Optional[Dict[str, Any]]:
        if len(history) >= self.max_turns:
            return None
        response = self.client.chat.completions.create(
            model=self.model, messages=self.messages, tools=self._tools)
        message = response.choices[0].message
        self.messages.append(message.model_dump(exclude_none=True))
        for call in (message.tool_calls or []):
            self._pending_id = call.id
            return {"tool": call.function.name,
                    "arguments": json.loads(call.function.arguments or "{}")}
        self._final = message.content or ""
        return None

    def record_result(self, result: Dict[str, Any]) -> None:
        self.messages.append({"role": "tool", "tool_call_id": self._pending_id,
                              "content": json.dumps(result)})

    def summarise(self, history) -> str:
        return self._final or "(the model produced no closing narrative)"


def build_provider(name: str, *, model: Optional[str], resource: str,
                   tool: Optional[str] = None):
    if name == "scripted":
        return ScriptedProvider(resource, tool=tool)
    if name == "anthropic":
        return AnthropicProvider(model or "claude-sonnet-4-20250514")
    if name == "openai":
        return OpenAIProvider(model or "gpt-4o")
    raise SystemExit(f"unknown provider {name!r}")


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def run(api: TraktAgentAPI, provider: Any, resource: str) -> Dict[str, Any]:
    catalogue = api.list_tools()
    tools = catalogue.get("tools") or []
    permitted = catalogue.get("resources") or {}

    print(f"Trakt published {len(tools)} tool(s): "
          f"{', '.join(t['name'] for t in tools) or '(none)'}")
    for capability, refs in sorted(permitted.items()):
        print(f"  {capability}: {', '.join(refs) or '(no resources)'}")
    if not tools:
        raise SystemExit(
            "This identity holds no tools. Check its grants in the entitlement "
            "configuration — an agent's scopes are derived from them.")

    task = (f"Review portfolio {resource} for covenant and concentration issues. "
            f"Use the Trakt tools; state no figure Trakt did not return.")
    if hasattr(provider, "start"):
        provider.start(tools, task)

    history: List[Dict[str, Any]] = []
    while True:
        decision = provider.next_call(tools, history)
        if decision is None:
            break
        print(f"\n→ {decision['tool']}({json.dumps(decision['arguments'])})")
        result = api.invoke(decision["tool"], decision["arguments"])

        status = result.get("status")
        if status == "success":
            snapshot = result.get("snapshot") or {}
            lineage = (result.get("result") or {}).get("lineage") or {}
            print(f"← {status}  snapshot={snapshot.get('snapshot_id')} "
                  f"config_version={lineage.get('configuration_version')}")
        else:
            error = result.get("error") or {}
            print(f"← {status}  {error.get('code')}: {error.get('message')}")

        history.append({"tool": decision["tool"], "arguments": decision["arguments"],
                        "result": result})
        if hasattr(provider, "record_result"):
            provider.record_result(result)

    return {"correlation_id": api.correlation_id, "calls": api.calls,
            "history": history, "narrative": provider.summarise(history)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default=os.environ.get(
        "TRAKT_AGENT_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--resource", required=True,
                        help="e.g. ERE/source_portfolio/direct_001")
    parser.add_argument("--provider", default="scripted",
                        choices=("scripted", "anthropic", "openai"))
    parser.add_argument("--model", default=None,
                        help="Provider model id. Defaults per provider.")
    parser.add_argument("--tool", default=None,
                        help="scripted provider only: call this tool by name "
                             "instead of the first one the catalogue offers.")
    parser.add_argument("--token", default=os.environ.get("TRAKT_AGENT_TOKEN"),
                        help="Entra bearer token. Omit when the deployment runs "
                             "TRAKT_AGENT_AUTH_MODE=disabled.")
    parser.add_argument("--json", action="store_true",
                        help="Emit the full run record as JSON on stdout.")
    args = parser.parse_args(argv)

    api = TraktAgentAPI(args.base_url, token=args.token)
    provider = build_provider(args.provider, model=args.model,
                              resource=args.resource, tool=args.tool)
    record = run(api, provider, args.resource)

    if args.json:
        print(json.dumps(record, indent=2))
    else:
        print("\n" + "=" * 72)
        print(record["narrative"])
        print("=" * 72)
        print(f"\n{len(record['calls'])} tool call(s), correlation id "
              f"{record['correlation_id']}")
        print("Trakt holds the authoritative audit trail for these calls; this "
              "correlation id joins them up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
