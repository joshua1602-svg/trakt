"""trakt_tools.registry — the ONE place a Trakt agent tool is declared.

Everything an external agent can do is in here, and nothing in here holds
business logic. The registry is the single source for:

    GET /v1/agent/tools          the published catalogue + JSON Schemas
    deploy/agent-api/*.yaml      the generated OpenAPI document
    a future MCP server          tool declarations, translated 1:1
    the synthetic agents         the tool list they are given

Keeping those four generated from one declaration is what stops a tool existing
on one surface and not another, or drifting in its schema between them.

Registration is by import side effect from :mod:`trakt_tools.handlers`, and each
handler module imports its heavy dependencies *inside* the handler function. So
importing this registry — which the OpenAPI generator and the catalogue route
both do — costs nothing beyond the standard library.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from trakt_core.context import ExecutionContext

from .spec import ToolSpec

logger = logging.getLogger("trakt_tools.registry")

#: name -> spec. Module-level because the tool surface is a property of the
#: deployment, not of a request.
_REGISTRY: Dict[str, ToolSpec] = {}

#: Set once :mod:`trakt_tools.handlers` has been imported, so a lookup can
#: trigger registration exactly once rather than every call.
_LOADED = False


def register(spec: ToolSpec) -> ToolSpec:
    """Add a tool. A duplicate name is a startup failure, never a silent
    replacement — two tools answering to one name is precisely the ambiguity a
    published contract must not have."""
    if spec.name in _REGISTRY:
        raise ValueError(f"a tool named {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    return spec


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    # Marked BEFORE the import so a handler module importing the registry (for
    # `register`) cannot recurse into loading itself.
    _LOADED = True
    from . import handlers  # noqa: F401  - registration by import side effect


def get(name: Optional[str]) -> Optional[ToolSpec]:
    _ensure_loaded()
    return _REGISTRY.get(str(name or "").strip())


def all_tools() -> Tuple[ToolSpec, ...]:
    _ensure_loaded()
    return tuple(_REGISTRY[k] for k in sorted(_REGISTRY))


def tool_names() -> Tuple[str, ...]:
    _ensure_loaded()
    return tuple(sorted(_REGISTRY))


def catalogue(context: Optional[ExecutionContext] = None) -> Dict[str, Any]:
    """The published tool catalogue, narrowed to what ``context`` may call.

    Narrowing matters for an agent, not just for security: an agent given a tool
    it will always be refused wastes a call and then reasons about a refusal.
    Telling it only what it can actually use is the difference between a tool
    list and a capability statement.

    With no context the full catalogue is returned — that form is for generating
    the OpenAPI document and for operator inspection, never for a request.
    """
    _ensure_loaded()
    tools: List[Dict[str, Any]] = []
    for spec in all_tools():
        if context is not None and not context.has_scope(spec.required_capability):
            continue
        tools.append(spec.to_catalogue_entry())

    payload: Dict[str, Any] = {"tools": tools, "tool_count": len(tools)}

    if context is not None:
        # The closed set of resources this caller may name. Published for the
        # same reason: an agent should not have to guess an identifier and be
        # refused. `permitted_resources_for` returns () when no entitlements were
        # resolved, which is a real answer meaning "nothing", never a wildcard.
        from trakt_core.entitlement import permitted_resources_for

        by_capability: Dict[str, List[str]] = {}
        for spec in all_tools():
            if not context.has_scope(spec.required_capability):
                continue
            refs = permitted_resources_for(
                context, capability=spec.required_capability)
            by_capability[spec.required_capability] = sorted(r.key for r in refs)
        payload["resources"] = by_capability
        payload["organisation_id"] = context.organisation_id
        payload["tenant_id"] = context.tenant_id
    return payload


def reset_for_tests() -> None:
    """Clear the registry. Tests that register a fixture tool use this to avoid
    leaking it into another test's published catalogue."""
    global _LOADED
    _REGISTRY.clear()
    _LOADED = False
