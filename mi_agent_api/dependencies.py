"""mi_agent_api/dependencies.py — what a governed capability needs to run.

A capability takes ``(request, context, dependencies)``. The first two are data;
this is the third — the collaborators it must not reach out and construct for
itself. Injecting them is what lets a test exercise the real capability against
a fixture resolver, and what will let a future adapter run the same capability
against a different store without touching capability code.

Deliberately small. This is not a DI container: it is the three things the MI
capability actually needs, plus the runtime mode it must not decide for itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from trakt_core.runtime import runtime_mode
from trakt_core.tenancy import TenantRegistry, load_tenant_registry

from . import datasets as _datasets


def default_tenant_id() -> str:
    """The tenant this deployment serves.

    Deployment-per-tenant is the current production model, so the tenant is
    configuration, not request data: ``MI_AGENT_CLIENT_ID`` first (what the auth
    runbook sets), else the client embedded in ``MI_AGENT_PLATFORM_URI``
    (``blob://…/platform/{client}/latest/…``), else the historical default.
    """
    explicit = (os.environ.get("MI_AGENT_CLIENT_ID") or "").strip()
    if explicit:
        return explicit
    from_uri = _datasets._client_from_platform_uri()
    return from_uri or "client_001"


@dataclass(frozen=True)
class CapabilityDependencies:
    """Collaborators for one governed capability invocation."""

    tenant_registry: TenantRegistry
    #: Anything exposing ``describe_active_dataset`` / ``resolve_authorised_frame``
    #: / ``dataset_snapshot_for``. Production passes ``mi_agent_api.datasets``;
    #: tests pass a stub to exercise policy branches without real data.
    datasets: Any = _datasets
    #: ``production`` | ``development`` | ``test``. Resolved once per invocation
    #: by the factory below so a capability never reads the environment mid-flight.
    runtime_mode: str = "production"
    #: Optional override for the analytical engine, used by tests. A custom
    #: runner must accept the same keyword arguments as
    #: ``mi_agent.mi_agent_workflow.run_mi_agent_query`` — including ``parsed``,
    #: which carries the single parse of the question.
    query_runner: Optional[Callable[..., Any]] = None
    #: **Business Semantics Registry seam — now wired.**
    #:
    #: ``f(question, spec, semantics) -> Mapping[str, Any]``. Invoked at the
    #: single parse site; its result is carried on
    #: ``ParsedQuestion.semantics_context`` → ``RouteRequest.semantics_context``,
    #: reaching every recogniser and handler with no further plumbing.
    #:
    #: Defaulted by :func:`build_dependencies` to
    #: ``mi_workflows.semantics.semantics_context_resolver`` (the registry
    #: exists and has governed consumers). Injectable as before: tests and
    #: alternative deployments may pass their own, and a resolver fault always
    #: yields an empty context, never a failed query.
    semantics_resolver: Optional[Callable[..., Any]] = None


def build_dependencies(
    *,
    tenant_registry: Optional[TenantRegistry] = None,
    datasets: Any = None,
    mode: Optional[str] = None,
    semantics_resolver: Optional[Callable[..., Any]] = None,
) -> CapabilityDependencies:
    """Construct the dependency set for one invocation.

    Not cached: the tenancy config and runtime mode are cheap to read and a
    long-lived cache would hide a config change from a running worker.

    ``semantics_resolver`` is the Business Semantics Registry seam. Defaulting
    it here (rather than at each call site) makes governed business semantics
    available to every channel at once — the wiring the seam was built for.
    """
    from mi_workflows.semantics import semantics_context_resolver

    return CapabilityDependencies(
        tenant_registry=tenant_registry
        or load_tenant_registry(default_tenant_id=default_tenant_id()),
        datasets=datasets or _datasets,
        runtime_mode=mode or runtime_mode(),
        semantics_resolver=semantics_resolver or semantics_context_resolver,
    )
