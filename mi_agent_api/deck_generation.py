"""mi_agent_api/deck_generation.py — the governed artefact-PRODUCTION capability.

    capability id:  artefact.investor_pack.generate

``artefacts.get_investor_pack`` serves a deck the orchestration already
published. This is its counterpart: it lets an authorised user ASK for one to be
produced, so the investor pack is no longer reachable only as a side effect of a
nightly pipeline run.

Everything that makes a deck is already governed and lives elsewhere. This module
adds no reporting logic whatsoever — it authorises the request, resolves which
completed run the deck should be built from, and hands off to the SAME
orchestration stage the pipeline uses
(:func:`apps.blob_trigger_app.pptx_stage.generate_investor_pptx`), which invokes
the same generator, applies the same publication gates and publishes to the same
durable store. A deck a user asks for and a deck the pipeline produces are
therefore the same artefact built the same way, not two implementations that
drift.

Three properties the API contract depends on:

* **The tenant is never the caller's parameter.** It comes from the execution
  context, exactly as it does for reading.
* **Requests are single-flight.** Concurrent requests for the same scope and
  period join the running job rather than starting a second generation over the
  same data.
* **A blocked deck is a distinct outcome, not a failure.** When the publication
  gates refuse, the job completes as ``blocked`` and carries the failed gates —
  the pack exists for an operator to diagnose, and no stale deck is replaced.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from trakt_core.audit import emit_audit_event
from trakt_core.context import (
    SCOPE_ARTEFACT_GENERATE,
    ExecutionContext,
)
from trakt_core.envelope import (
    STATUS_BLOCKED,
    STATUS_ERROR,
    STATUS_SUCCESS,
    AuditMetadata,
    GovernedResult,
    PolicyState,
)
from trakt_core.errors import ErrorCategory, ErrorCode, TraktError
from trakt_core.tenancy import authorise_portfolio_access

from .dependencies import CapabilityDependencies, build_dependencies

logger = logging.getLogger("mi_agent_api.deck_generation")

CAPABILITY = "artefact.investor_pack.generate"

#: Job lifecycle. ``blocked`` means the deck was built and the publication gates
#: refused to release it — deliberately distinct from ``failed``, which means no
#: deck exists at all.
STATE_QUEUED = "queued"
STATE_GENERATING = "generating"
STATE_COMPLETED = "completed"
STATE_BLOCKED = "blocked"
STATE_FAILED = "failed"

TERMINAL_STATES = frozenset({STATE_COMPLETED, STATE_BLOCKED, STATE_FAILED})

#: How many finished jobs to remember. Enough for a user to poll a result and for
#: an idempotent retry to find it; small enough to be a cache, not a store.
MAX_REMEMBERED_JOBS = 64

_BLOCKING_CATEGORIES = frozenset({
    ErrorCategory.AUTHENTICATION, ErrorCategory.AUTHORISATION, ErrorCategory.POLICY,
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# The job.
# --------------------------------------------------------------------------- #

@dataclass
class GenerationJob:
    """One requested generation, and what became of it.

    Mutated only under :data:`_LOCK` by the worker; readers take a snapshot via
    :meth:`to_payload`.
    """

    job_id: str
    tenant_id: str
    portfolio_id: str
    portfolio_context: Optional[str]
    period: Optional[str]
    requested_by: str
    channel: str
    requested_at: str
    state: str = STATE_QUEUED
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    #: Reporting period the deck was actually built for, once resolved.
    reporting_period: Optional[str] = None
    run_id: Optional[str] = None
    #: Publication-gate verdict for a finished build.
    failed_gates: List[str] = field(default_factory=list)
    gate_count: Optional[int] = None
    #: Investor-safe failure text. Never an exception repr, never a path.
    message: Optional[str] = None
    error_code: Optional[str] = None
    #: Correlation of the ORIGINAL request, so a polled status ties to the audit
    #: trail of the request that started the work.
    request_id: str = ""
    correlation_id: Optional[str] = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_payload(self) -> Dict[str, Any]:
        """Client-safe view. Carries no filesystem or blob location, and no
        internal identifiers beyond the run the deck reports on."""
        return {
            "jobId": self.job_id,
            "state": self.state,
            "portfolioId": self.portfolio_id,
            "portfolioContext": self.portfolio_context,
            "reportingPeriod": self.reporting_period or self.period,
            "runId": self.run_id,
            "requestedAt": self.requested_at,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "failedGates": list(self.failed_gates),
            "gateCount": self.gate_count,
            "message": self.message,
            "errorCode": self.error_code,
        }


# --------------------------------------------------------------------------- #
# Registry. In-process by design: this deployment runs one API instance, and a
# job that outlives the process is a queue, not a cache. The seam is the two
# module functions below, so a future durable queue replaces them alone.
# --------------------------------------------------------------------------- #

_LOCK = threading.RLock()
_JOBS: "OrderedDict[str, GenerationJob]" = OrderedDict()
#: scope key -> job id, for single-flight and idempotency.
_BY_KEY: Dict[Tuple[str, ...], str] = {}


def _scope_key(tenant_id: str, portfolio_id: str, portfolio_context: Optional[str],
               period: Optional[str], idempotency_key: Optional[str]) -> Tuple[str, ...]:
    return (tenant_id, portfolio_id, portfolio_context or "total",
            period or "latest", idempotency_key or "")


def _remember(job: GenerationJob, key: Tuple[str, ...]) -> None:
    with _LOCK:
        _JOBS[job.job_id] = job
        _BY_KEY[key] = job.job_id
        while len(_JOBS) > MAX_REMEMBERED_JOBS:
            evicted, _ = _JOBS.popitem(last=False)
            for k, v in list(_BY_KEY.items()):
                if v == evicted:
                    _BY_KEY.pop(k, None)


def _existing(key: Tuple[str, ...], *, reuse_terminal: bool) -> Optional[GenerationJob]:
    """A job to join instead of starting a new one.

    An in-flight job for the same scope is always joined — two identical
    generations over the same data would race for the same output file. A
    FINISHED job is reused only when the caller supplied an idempotency key, so
    an ordinary "generate again" after new data still generates.
    """
    with _LOCK:
        job = _JOBS.get(_BY_KEY.get(key, ""))
        if job is None:
            return None
        if not job.terminal:
            return job
        return job if reuse_terminal else None


def reset_jobs() -> None:
    """Forget every job. For tests only."""
    with _LOCK:
        _JOBS.clear()
        _BY_KEY.clear()


# --------------------------------------------------------------------------- #
# Run resolution — which completed run does this deck report on?
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ResolvedRun:
    run_dir: Path
    run_id: str
    reporting_date: Optional[str]
    #: Root of the client's runs — the multi-period surfaces (evolution,
    #: movement, cohort progression) need it and cannot infer it from one run.
    output_root: str


def _period_matches(reporting_date: Optional[str], run_id: str,
                    period: Optional[str]) -> bool:
    if not period or period == "latest":
        return True
    want = str(period).strip()
    for candidate in (reporting_date, run_id):
        if candidate and str(candidate).startswith(want):
            return True
    return False


def resolve_run(client_id: str, period: Optional[str]) -> Optional[ResolvedRun]:
    """The completed run a deck for ``period`` should be built from.

    ``None`` / ``latest`` resolves the newest run. Discovery is the same walk the
    reporting-date selector is built from, so the periods a user can ask to
    generate are exactly the periods the UI offers.
    """
    from . import datasets as datasets_mod
    from . import snapshots as snapshots_mod

    root = datasets_mod._onboarding_output_root()
    if not root:
        return None
    try:
        discovered = snapshots_mod.discover_snapshots(root)
    except Exception as exc:  # noqa: BLE001 — discovery is never fatal here
        logger.warning("run discovery failed for %s: %s", client_id, exc)
        return None

    runs = next((p.get("runs") or [] for p in discovered.get("portfolios") or ()
                 if p.get("client_id") == client_id), [])
    # discover_snapshots orders oldest → newest; the newest match wins.
    for run in reversed(runs):
        run_id = str(run.get("run_id") or "")
        if not run_id or not _period_matches(run.get("reporting_date"), run_id, period):
            continue
        tape = snapshots_mod.resolve_tape_path(root, client_id, run_id)
        if tape is None:
            continue
        # …/<run>/central/<tape>  or  …/<run>/output/central/<tape>
        return ResolvedRun(run_dir=tape.parent.parent, run_id=run_id,
                           reporting_date=run.get("reporting_date"),
                           output_root=str(root))
    return None


# --------------------------------------------------------------------------- #
# The worker.
# --------------------------------------------------------------------------- #

def _run_generation(job: GenerationJob, run: ResolvedRun) -> None:
    """Build the deck for ``run`` and record the outcome on ``job``.

    Runs off-request. Every failure path is caught: a generation fault must leave
    a job the caller can read, never an unobservable dead thread.
    """
    with _LOCK:
        job.state = STATE_GENERATING
        job.started_at = _now()
        job.run_id = run.run_id
        job.reporting_period = run.reporting_date or job.period

    try:
        from apps.blob_trigger_app import pptx_stage

        artifact = pptx_stage.generate_investor_pptx(
            run.run_dir,
            client_name=job.tenant_id,
            as_of_date=run.reporting_date or "",
            source_run_id=run.run_id,
            portfolio_context=job.portfolio_context,
            client_id=job.tenant_id,
            tenant_id=job.tenant_id,
            output_root=run.output_root,
        )
        preflight = artifact.get("preflight") or {}
        gates = preflight.get("gates")
        with _LOCK:
            job.completed_at = _now()
            job.gate_count = len(gates) if isinstance(gates, list) else None
            job.failed_gates = list(preflight.get("failed_gates") or [])
            status = artifact.get("status")
            if status == "generated_not_published":
                job.state = STATE_BLOCKED
                job.message = (
                    "The pack was generated but did not pass its publication "
                    "checks, so it has not been released. The previously "
                    "published deck is unchanged.")
            elif status == "available":
                job.state = STATE_COMPLETED
                job.message = None
            else:
                job.state = STATE_FAILED
                job.error_code = ErrorCode.ARTEFACT_INCOMPLETE
                job.message = "The pack could not be generated for this period."
    except Exception:  # noqa: BLE001 — the caller gets a job, not a traceback
        logger.exception("investor pack generation failed for tenant=%s run=%s",
                         job.tenant_id, run.run_id)
        with _LOCK:
            job.state = STATE_FAILED
            job.completed_at = _now()
            job.error_code = ErrorCode.INTERNAL_ERROR
            job.message = "Generation failed. The published deck is unchanged."


def _dispatch(job: GenerationJob, run: ResolvedRun,
              executor: Optional[Callable[[Callable[[], None]], None]]) -> None:
    work = lambda: _run_generation(job, run)  # noqa: E731
    if executor is not None:
        executor(work)
        return
    threading.Thread(target=work, name=f"deck-gen-{job.job_id}",
                     daemon=True).start()


# --------------------------------------------------------------------------- #
# Capability.
# --------------------------------------------------------------------------- #

def _audit(context: ExecutionContext, *, outcome: str, portfolio_id: Optional[str],
           error_code: Optional[str]) -> AuditMetadata:
    return AuditMetadata(
        capability=CAPABILITY, request_id=context.request_id,
        correlation_id=context.correlation_id, tenant_id=context.tenant_id,
        actor_id=context.actor_id, actor_type=context.actor_type,
        channel=context.channel, portfolio_id=portfolio_id, outcome=outcome,
        started_at=_now(), error_code=error_code)


def _failed(context: ExecutionContext, deps: CapabilityDependencies,
            err: TraktError, portfolio_id: Optional[str]
            ) -> GovernedResult[GenerationJob]:
    status = STATUS_BLOCKED if err.category in _BLOCKING_CATEGORIES else STATUS_ERROR
    result: GovernedResult[GenerationJob] = GovernedResult(
        capability=CAPABILITY, status=status, request_id=context.request_id,
        correlation_id=context.correlation_id, tenant_id=context.tenant_id,
        portfolio_id=portfolio_id, result=None, error=err,
        policy=PolicyState(runtime_mode=deps.runtime_mode),
        audit=_audit(context, outcome=status, portfolio_id=portfolio_id,
                     error_code=err.code))
    emit_audit_event(result)
    return result


def _succeeded(context: ExecutionContext, deps: CapabilityDependencies,
               job: GenerationJob) -> GovernedResult[GenerationJob]:
    result: GovernedResult[GenerationJob] = GovernedResult(
        capability=CAPABILITY, status=STATUS_SUCCESS, request_id=context.request_id,
        correlation_id=context.correlation_id, tenant_id=context.tenant_id,
        portfolio_id=job.portfolio_id, result=job,
        policy=PolicyState(runtime_mode=deps.runtime_mode, tenant_authorised=True,
                           capability_allowed=True, data_approved=True),
        audit=_audit(context, outcome=STATUS_SUCCESS, portfolio_id=job.portfolio_id,
                     error_code=None))
    emit_audit_event(result)
    return result


def generation_enabled() -> bool:
    """Whether on-demand generation is offered by this deployment.

    A deployment that produces decks only from the nightly pipeline sets this
    off, and the control disables itself rather than failing on use.
    """
    val = os.environ.get("TRAKT_INVESTOR_PPTX_ON_DEMAND", "true").strip().lower()
    return val not in ("0", "false", "no", "off")


def request_generation(
    context: ExecutionContext,
    *,
    portfolio_id: Optional[str] = None,
    portfolio_context: Optional[str] = None,
    period: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    dependencies: Optional[CapabilityDependencies] = None,
    executor: Optional[Callable[[Callable[[], None]], None]] = None,
) -> GovernedResult[GenerationJob]:
    """Ask for an investor pack to be generated for the authenticated tenant.

    Returns as soon as the work is accepted — generation takes seconds, which is
    too long to hold a request open and too short to justify a durable queue. The
    caller polls :func:`get_generation`.

    ``executor`` runs the work; tests pass a synchronous one so a generation can
    be asserted without a thread.
    """
    deps = dependencies or build_dependencies()
    try:
        context.require_scope(SCOPE_ARTEFACT_GENERATE)

        if not generation_enabled():
            raise TraktError(
                ErrorCode.PERMISSION_DENIED,
                "On-demand investor packs are not enabled for this deployment.",
                request_id=context.request_id)

        authorised = authorise_portfolio_access(
            context, portfolio_id, registry=deps.tenant_registry)

        if period is not None and not _valid_period(period):
            raise TraktError(ErrorCode.INVALID_INPUT,
                             "The reporting period is not a valid period label.",
                             request_id=context.request_id)
        if portfolio_context is not None and not _valid_context(portfolio_context):
            raise TraktError(ErrorCode.INVALID_INPUT,
                             "The portfolio context is not a valid scope label.",
                             request_id=context.request_id)
        if idempotency_key is not None and not _valid_idempotency_key(idempotency_key):
            raise TraktError(ErrorCode.INVALID_INPUT,
                             "The idempotency key is not valid.",
                             request_id=context.request_id)

        key = _scope_key(context.tenant_id, authorised.portfolio_id,
                         portfolio_context, period, idempotency_key)
        joined = _existing(key, reuse_terminal=bool(idempotency_key))
        if joined is not None:
            return _succeeded(context, deps, joined)

        # The run is resolved BEFORE the job is accepted, so "there is no data
        # for that period" is an answer to the request rather than a job that
        # fails a moment later.
        run = resolve_run(context.tenant_id, period)
        if run is None:
            raise TraktError(
                ErrorCode.DATA_SOURCE_UNAVAILABLE,
                f"No completed reporting run is available for "
                f"{period or 'the latest period'}.",
                request_id=context.request_id)

        job = GenerationJob(
            job_id=f"deckgen_{uuid.uuid4().hex[:16]}",
            tenant_id=context.tenant_id, portfolio_id=authorised.portfolio_id,
            portfolio_context=portfolio_context, period=period,
            requested_by=context.actor_id, channel=context.channel,
            requested_at=_now(), request_id=context.request_id,
            correlation_id=context.correlation_id,
            # Recorded at acceptance, not on completion: the caller is told which
            # period it will get while the job is still queued, so the UI never
            # has to guess what "latest" resolved to.
            run_id=run.run_id, reporting_period=run.reporting_date or period)
        _remember(job, key)
        _dispatch(job, run, executor)
        return _succeeded(context, deps, job)

    except TraktError as err:
        return _failed(context, deps, err, portfolio_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("generation request failed for tenant=%s", context.tenant_id)
        return _failed(context, deps, TraktError(
            ErrorCode.INTERNAL_ERROR,
            "The investor pack could not be requested.",
            request_id=context.request_id, cause=exc), portfolio_id)


def get_generation(context: ExecutionContext, job_id: str, *,
                   dependencies: Optional[CapabilityDependencies] = None
                   ) -> GovernedResult[GenerationJob]:
    """The state of a previously requested generation.

    A job belonging to another tenant is reported as NOT FOUND rather than
    forbidden: whether another tenant has a job in flight is itself information.
    """
    deps = dependencies or build_dependencies()
    try:
        context.require_scope(SCOPE_ARTEFACT_GENERATE)
        with _LOCK:
            job = _JOBS.get(str(job_id or ""))
        if job is None or job.tenant_id != context.tenant_id:
            raise TraktError(ErrorCode.ARTEFACT_NOT_FOUND,
                             "That generation request is not known.",
                             request_id=context.request_id)
        return _succeeded(context, deps, job)
    except TraktError as err:
        return _failed(context, deps, err, None)


# --------------------------------------------------------------------------- #
# Input validation. Every one of these values reaches a path or a scope
# resolver, so each is checked against a shape rather than sanitised.
# --------------------------------------------------------------------------- #

def _valid_period(period: str) -> bool:
    import re
    return bool(re.fullmatch(r"latest|\d{4}-\d{2}(-\d{2})?", str(period).strip()))


def _valid_context(value: str) -> bool:
    import re
    return bool(re.fullmatch(r"[A-Za-z0-9_.\-]{1,64}", str(value).strip()))


def _valid_idempotency_key(value: str) -> bool:
    import re
    return bool(re.fullmatch(r"[A-Za-z0-9_.\-]{1,128}", str(value).strip()))
