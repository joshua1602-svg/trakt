"""trakt_tools.execution — the one governed path an agent tool call takes.

    execute_governed_tool(tool_name, arguments, context) -> GovernedResult

Every external agent call goes through this function, in this order, and no
handler can reorder or skip a step:

    1. the tool exists                     registry
    2. the arguments satisfy its schema    trakt_tools.schema
    3. the caller holds the capability     ExecutionContext.require_scope
    4. the organisation is entitled to
       the named resource                  trakt_core.entitlement
    5. the dataset is approved to answer   trakt_core.policy
    6. the deterministic handler runs      existing Trakt implementations
    7. the result is enveloped             trakt_core.envelope.GovernedResult
    8. the execution is audited            trakt_core.audit

Steps 1-5 touch no data. That is the same property ``mi_agent_api.mi_service``
holds and it is stated the same way here so it can be tested the same way: no
dataframe is resolved for a caller who is not entitled to it, or for a dataset
that is not approved to answer.

What a handler can and cannot reach
-----------------------------------
A handler receives typed arguments and a :class:`~trakt_tools.spec.ToolInvocation`
carrying the *already authorised* resource. It never sees a token, a header, a
connection string or a storage path, and it has no way to ask for a resource
other than the one authorised here. An external model, further out, sees only
the serialised :class:`~trakt_core.envelope.GovernedResult` — never a dataframe,
never a database handle, never a file.

Failure is always a governed result, never an exception across the boundary. A
typed :class:`~trakt_core.errors.TraktError` becomes ``status=blocked`` for a
governance refusal and ``status=error`` for everything else, so an agent can
tell "I am not allowed to ask that" from "that could not be computed" without
parsing prose.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from trakt_core.audit import emit_audit_event
from trakt_core.context import ExecutionContext
from trakt_core.entitlement import AuthorisedResource, authorise_resource_access
from trakt_core.envelope import (
    STATUS_BLOCKED,
    STATUS_ERROR,
    STATUS_SUCCESS,
    AuditMetadata,
    GovernedResult,
    PolicyState,
    ProvenanceRef,
    SnapshotRef,
)
from trakt_core.errors import ErrorCategory, ErrorCode, TraktError
from trakt_core.policy import evaluate_source_approval
from trakt_core.resource import ResourceCatalogue

from . import registry as registry_mod
from .schema import validate
from .spec import ToolInvocation, ToolSpec
from .telemetry import ToolTelemetry

logger = logging.getLogger("trakt_tools.execution")

#: Error categories that mean "governance refused" rather than "the capability
#: could not answer". They map to STATUS_BLOCKED so an autonomous caller can
#: distinguish a permission decision from a data outcome and stop retrying.
_BLOCKING_CATEGORIES = frozenset({
    ErrorCategory.AUTHENTICATION, ErrorCategory.AUTHORISATION, ErrorCategory.POLICY,
})


@dataclass(frozen=True)
class ToolDependencies:
    """Collaborators one tool execution needs.

    Small and explicit, exactly like ``mi_agent_api.dependencies``: this is not a
    container, it is the handful of things a tool must not construct for itself
    so that a test can exercise the real governed path against fixtures.
    """

    #: Anything exposing ``describe_active_dataset``. Production passes
    #: ``mi_agent_api.datasets``; tests pass a stub to drive the policy branches.
    datasets: Any = None
    #: ``production`` | ``development`` | ``test``. Resolved once per execution
    #: so a handler never reads the environment mid-flight.
    runtime_mode: str = ""
    #: The resource catalogue the entitlement check expands against.
    catalogue: Optional[ResourceCatalogue] = None
    #: Local onboarding output root for run/snapshot discovery. Defaults to the
    #: same resolver every other governed surface uses.
    output_root: Any = None
    #: Test seam: replaces the covenant evaluation with a stub so the governance
    #: ordering can be proven without a portfolio on disk. Production leaves it
    #: unset and the handler calls the real implementation.
    covenant_evaluator: Any = None
    #: Test seam: ``() -> DataFrame`` for the governed loan frame. Production
    #: leaves it unset and the handler resolves through
    #: ``mi_agent_api.datasets._resolve_query_frame`` — the SAME resolver the MI
    #: Query path uses, so an agent and the workspace read one frame.
    loan_frame_resolver: Any = None
    #: Test seam: an explicit :class:`trakt_core.lineage.LineageIndex`, or a path
    #: to one. Production leaves both unset and the index is discovered beside
    #: the governed output.
    lineage_index: Any = None
    lineage_index_path: Any = None
    #: Test seams for the two engine-wrapping tools, for the same reason as
    #: ``covenant_evaluator``: the governance ordering must be provable without a
    #: portfolio on disk. Production leaves both unset.
    drillthrough_evaluator: Any = None
    period_change_analyser: Any = None


def build_dependencies(
    *,
    datasets: Any = None,
    mode: Optional[str] = None,
    catalogue: Optional[ResourceCatalogue] = None,
    output_root: Any = None,
    covenant_evaluator: Any = None,
    loan_frame_resolver: Any = None,
    lineage_index: Any = None,
    lineage_index_path: Any = None,
    drillthrough_evaluator: Any = None,
    period_change_analyser: Any = None,
) -> ToolDependencies:
    """Construct the dependency set for one execution.

    Imports ``mi_agent_api.datasets`` lazily so that importing the registry — for
    the catalogue route, the OpenAPI generator or a schema test — does not pull
    in pandas or storage.
    """
    from trakt_core.runtime import runtime_mode

    if datasets is None:
        from mi_agent_api import datasets as _datasets
        datasets = _datasets
    if output_root is None:
        output_root = datasets._onboarding_output_root()
    return ToolDependencies(
        datasets=datasets,
        runtime_mode=mode or runtime_mode(),
        catalogue=catalogue,
        output_root=output_root,
        covenant_evaluator=covenant_evaluator,
        loan_frame_resolver=loan_frame_resolver,
        lineage_index=lineage_index,
        lineage_index_path=lineage_index_path,
        drillthrough_evaluator=drillthrough_evaluator,
        period_change_analyser=period_change_analyser,
    )


# --------------------------------------------------------------------------- #
# Envelope helpers
# --------------------------------------------------------------------------- #
def _snapshot_ref(descriptor: Any, approval_state: Optional[str]) -> SnapshotRef:
    """The dataset identity, read from the descriptor the resolver produced.

    Deliberately identical to ``mi_agent_api.mi_service._snapshot_ref``: the
    agent surface and the UI surface must describe the same dataset the same way,
    or an agent could cite a snapshot the UI does not recognise.
    """
    return SnapshotRef(
        source_kind=getattr(descriptor, "source_kind", None),
        source_base=getattr(descriptor, "source_base", None),
        dataset_label=getattr(descriptor, "label", None),
        reporting_date=getattr(descriptor, "reporting_date", None),
        snapshot_id=getattr(descriptor, "snapshot_id", None),
        content_hash=getattr(descriptor, "content_hash", None),
        row_count=getattr(descriptor, "row_count", None),
        approval_state=approval_state,
        source_portfolios=tuple(getattr(descriptor, "source_portfolios", ()) or ()),
    )


def _audit(context: ExecutionContext, capability: str, *, outcome: str,
           started_at: str, t0: float, resource_key: Optional[str],
           snapshot_id: Optional[str], error_code: Optional[str]) -> AuditMetadata:
    return AuditMetadata(
        capability=capability,
        request_id=context.request_id,
        correlation_id=context.correlation_id,
        tenant_id=context.tenant_id,
        organisation_id=context.organisation_id,
        microsoft_tenant_id=context.microsoft_tenant_id,
        actor_id=context.actor_id,
        actor_type=context.actor_type,
        channel=context.channel,
        # The resource is what an agent call is actually about, so it is what the
        # audit line records in the portfolio slot. A tool call names a resource;
        # it does not name a portfolio selector.
        portfolio_id=resource_key,
        snapshot_id=snapshot_id,
        outcome=outcome,
        started_at=started_at,
        duration_ms=int((time.monotonic() - t0) * 1000),
        error_code=error_code,
    )


def _failure(context: ExecutionContext, capability: str, error: TraktError, *,
             started_at: str, t0: float, resource_key: Optional[str],
             policy: PolicyState,
             snapshot: Optional[SnapshotRef] = None) -> GovernedResult[Dict[str, Any]]:
    status = STATUS_BLOCKED if error.category in _BLOCKING_CATEGORIES else STATUS_ERROR
    return GovernedResult(
        capability=capability,
        status=status,
        request_id=context.request_id,
        correlation_id=context.correlation_id,
        tenant_id=context.tenant_id,
        portfolio_id=resource_key,
        snapshot=snapshot,
        result=None,
        policy=policy,
        error=error,
        audit=_audit(context, capability, outcome=status, started_at=started_at,
                     t0=t0, resource_key=resource_key,
                     snapshot_id=snapshot.snapshot_id if snapshot else None,
                     error_code=error.code),
    )


# --------------------------------------------------------------------------- #
# The governed execution
# --------------------------------------------------------------------------- #
def execute_governed_tool(
    tool_name: str,
    arguments: Optional[Mapping[str, Any]],
    context: ExecutionContext,
    *,
    dependencies: Optional[ToolDependencies] = None,
    registry: Any = registry_mod,
) -> GovernedResult[Dict[str, Any]]:
    """Run one agent tool call under full governance.

    Never raises for a caller-caused or data-caused failure: the outcome is
    always a :class:`~trakt_core.envelope.GovernedResult` carrying a typed error,
    and every path — including every refusal — emits an audit event before
    returning.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    capability = f"tool.{str(tool_name or '').strip() or 'unknown'}"
    policy = PolicyState()

    # ---- 1. does the tool exist? ----------------------------------------- #
    spec: Optional[ToolSpec] = registry.get(tool_name)
    if spec is None:
        result = _failure(
            context, capability,
            TraktError(ErrorCode.TOOL_NOT_FOUND,
                       "That tool is not available on this deployment.",
                       request_id=context.request_id,
                       details={"tool": str(tool_name),
                                "available_tools": list(registry.tool_names())}),
            started_at=started_at, t0=t0, resource_key=None, policy=policy)
        emit_audit_event(result)
        return result

    capability = spec.capability_id
    deps = dependencies or build_dependencies()
    policy = PolicyState(runtime_mode=deps.runtime_mode)

    # ---- 2. do the arguments satisfy the published schema? ---------------- #
    args, schema_errors = validate(dict(arguments or {}), spec.input_schema)
    if schema_errors:
        result = _failure(
            context, capability,
            TraktError(ErrorCode.TOOL_INPUT_INVALID,
                       "The arguments do not satisfy this tool's input schema.",
                       request_id=context.request_id,
                       # The failing paths, so an agent can correct and retry
                       # rather than guess. Argument VALUES are deliberately not
                       # echoed: they would land in an error body and, through it,
                       # in a client-side log.
                       details={"tool": spec.name, "violations": schema_errors}),
            started_at=started_at, t0=t0, resource_key=None, policy=policy)
        emit_audit_event(result)
        return result

    resource_key = str(args.get(spec.resource_argument) or "") or None

    # ---- 3 + 4. capability, then entitlement over the named resource ------ #
    # `authorise_resource_access` checks the scope itself as its first step; it
    # is called explicitly first so the two axes are visible here as two
    # decisions rather than one, which is what the ordering contract promises.
    try:
        context.require_scope(spec.required_capability)
        authorised: AuthorisedResource = authorise_resource_access(
            context, resource_key or "", spec.required_capability,
            catalogue=deps.catalogue)
    except TraktError as err:
        result = _failure(context, capability, err, started_at=started_at, t0=t0,
                          resource_key=resource_key, policy=policy)
        emit_audit_event(result)
        return result

    policy = PolicyState(runtime_mode=deps.runtime_mode, tenant_authorised=True,
                         capability_allowed=True)

    # ---- 5. is this dataset allowed to answer at all? --------------------- #
    try:
        descriptor = deps.datasets.describe_active_dataset()
        approval = evaluate_source_approval(
            descriptor.source_base, mode=deps.runtime_mode,
            dataset_available=descriptor.available)
    except Exception as exc:  # noqa: BLE001 - a storage fault is not a raw 500
        logger.exception("dataset description failed for tenant=%s tool=%s",
                         context.tenant_id, spec.name)
        result = _failure(
            context, capability,
            TraktError(ErrorCode.STORAGE_UNAVAILABLE,
                       "The governed data store is currently unavailable.",
                       request_id=context.request_id, cause=exc),
            started_at=started_at, t0=t0, resource_key=resource_key, policy=policy)
        emit_audit_event(result)
        return result

    snapshot = _snapshot_ref(descriptor, approval.state)
    if not approval.approved:
        result = _failure(
            context, capability,
            TraktError(approval.error_code or ErrorCode.DATA_SOURCE_NOT_APPROVED,
                       approval.reason, request_id=context.request_id),
            started_at=started_at, t0=t0, resource_key=resource_key,
            policy=PolicyState(runtime_mode=deps.runtime_mode,
                               tenant_authorised=True, capability_allowed=True,
                               data_approved=False, notes=(approval.reason,)),
            snapshot=snapshot)
        emit_audit_event(result)
        return result

    policy = PolicyState(
        runtime_mode=deps.runtime_mode, tenant_authorised=True,
        capability_allowed=True, data_approved=True, fixture_source=approval.fixture,
        notes=(approval.reason,) if approval.fixture else ())

    # ---- 6. the deterministic handler ------------------------------------- #
    telemetry = ToolTelemetry(tool=spec.name)
    invocation = ToolInvocation(context=context, authorised=authorised,
                                snapshot=snapshot, dependencies=deps,
                                telemetry=telemetry)
    try:
        payload = spec.handler(args, invocation)
    except TraktError as err:
        # A handler that knows why it cannot answer says so in the taxonomy.
        result = _failure(context, capability, err, started_at=started_at, t0=t0,
                          resource_key=resource_key, policy=policy,
                          snapshot=snapshot)
        emit_audit_event(result)
        return result
    except Exception as exc:  # noqa: BLE001 - an unclassified fault is a defect
        logger.exception("tool %s failed for tenant=%s resource=%s",
                         spec.name, context.tenant_id, resource_key)
        result = _failure(
            context, capability,
            # The exception type and message are logged, never returned: an
            # internal class name is not something a client agent should read.
            TraktError(ErrorCode.TOOL_EXECUTION_FAILED,
                       "The tool could not complete. The failure has been logged.",
                       request_id=context.request_id, details={"tool": spec.name},
                       cause=exc),
            started_at=started_at, t0=t0, resource_key=resource_key, policy=policy,
            snapshot=snapshot)
        emit_audit_event(result)
        return result

    # ---- 7. envelope ------------------------------------------------------ #
    warnings = tuple(str(w) for w in (payload.get("warnings") or ()))
    # What the call cost, published rather than only logged: a client agent
    # cannot read Trakt's logs, and an agent that cannot see what its last call
    # cost has no way to choose a cheaper next one.
    payload["_telemetry"] = telemetry.emit(
        duration_ms=int((time.monotonic() - t0) * 1000),
        tenant_id=context.tenant_id, resource=resource_key or "",
        request_id=context.request_id)
    result = GovernedResult(
        capability=capability,
        status=STATUS_SUCCESS,
        request_id=context.request_id,
        correlation_id=context.correlation_id,
        tenant_id=context.tenant_id,
        portfolio_id=resource_key,
        snapshot=snapshot,
        result=payload,
        warnings=warnings,
        policy=policy,
        provenance=ProvenanceRef(
            source_notes=tuple(payload.get("source_notes") or ()),
            reconciliation=payload.get("reconciliation"),
            snapshot=snapshot),
        audit=_audit(context, capability, outcome=STATUS_SUCCESS,
                     started_at=started_at, t0=t0, resource_key=resource_key,
                     snapshot_id=snapshot.snapshot_id, error_code=None),
    )
    # ---- 8. audit --------------------------------------------------------- #
    emit_audit_event(result)
    return result
