"""operations_control.mi_query_telemetry — the governed record of one MI question.

This lives in ``operations_control`` because the RECORD is an OCC document. OCC
owns the store it lands in, the layout it lands under, the review vocabulary an
operator classifies it with, and the routes that read it back. The MI API is
merely the writer, and calls in here to write.

Putting it the other way round — the projection living beside the MI service —
coupled two independently deployed App Services in both directions: the OCC API
imported ``mi_agent_api`` at module scope for the review vocabulary, which its
deployment package does not ship (and should not: the MI service is a separate
deployment with its own image and requirements), and the OCC API then failed to
import at all. The dependency that remains runs one way only, MI API → OCC, and
is declared in ``deploy/trakt-mi-api/package_contents.txt``.

Every MI Query execution already ends at a :class:`~trakt_core.envelope.
GovernedResult` carrying who asked, of which snapshot, what happened and how
long it took, and an analytical envelope carrying the question, the answer, the
structured interpretation and the route that produced it. This module projects
those two into one durable record in the Operations Control Centre's existing
client-scoped store, so an operator can review live usage.

The split, deliberately:

* **Application Insights** keeps what it already keeps — the ``trakt.audit``
  line: identifiers, outcome, error code, snapshot, latency. Unchanged, and
  still carrying no question and no answer.
* **The governed OCC store** keeps the question and the answer, client-scoped,
  behind the same tenancy rules as every other document in that container.

Nothing here computes, interprets or judges. It records what the pipeline
already decided, and it never raises: telemetry must not be able to fail a
query that has already been answered.

What is NEVER recorded: model reasoning, prompts, tokens, credentials, stack
traces, or any field the audit event already forbids.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trakt_core.envelope import GovernedResult
from trakt_core.errors import ErrorCategory, ErrorCode, category_for

logger = logging.getLogger("operations_control.mi_query_telemetry")

SCHEMA_VERSION = "1.0.0"

# -- outcome, from the governed status and the existing error vocabulary ---- #
ANSWERED = "ANSWERED"
REFUSED = "REFUSED"
ERROR = "ERROR"

#: A capability-level non-delivery is a governed REFUSAL, not a fault: the
#: request was well formed and correctly authorised, and the governed answer is
#: "we did not compute one". Only a broken calculation or failed infrastructure
#: is an ERROR. Both readings come from the existing code vocabulary — nothing
#: is invented here.
_ERROR_CODES = frozenset({ErrorCode.CALCULATION_FAILED})
_ERROR_CATEGORIES = frozenset({ErrorCategory.INFRASTRUCTURE})

# -- operator quality review ------------------------------------------------ #
UNREVIEWED = "UNREVIEWED"
REVIEW_CLASSIFICATIONS = (
    "CORRECT",
    "WRONG_INTERPRETATION",
    "WRONG_CALCULATION",
    "RENDERING_ERROR",
    "PARTIALLY_CORRECT",
    "APPROPRIATE_REFUSAL",
    "SHOULD_HAVE_ANSWERED",
    "NEEDS_INVESTIGATION",
)
#: Classifications that mean the response was not what it should have been.
#: Used for counting only; it never changes what the client was shown.
PROBLEMATIC = frozenset({
    "WRONG_INTERPRETATION", "WRONG_CALCULATION", "RENDERING_ERROR",
    "PARTIALLY_CORRECT", "SHOULD_HAVE_ANSWERED",
})


def enabled() -> bool:
    """Is a governed store actually configured to record into?

    Telemetry WRITES, where the audit line only logs, so it records only where a
    storage backend has been deliberately chosen — Azure in a deployment, or an
    explicit local root. Without one, the filesystem backend would resolve to a
    path relative to the working directory and quietly create a store nobody
    asked for, which is not a place governed client records may land.
    """
    if os.environ.get("TRAKT_MI_QUERY_TELEMETRY", "").strip().lower() in ("0", "off",
                                                                          "false"):
        return False
    try:
        from apps.blob_trigger_app.storage import decide_backend
        decision = decide_backend()
    except Exception:  # noqa: BLE001
        return False
    if decision["backend"] == "azure_blob":
        return True
    # Filesystem is only a governed store when someone named its root.
    return bool(os.environ.get("TRAKT_LOCAL_BLOB_ROOT"))


def outcome_for(result: GovernedResult) -> str:
    """ANSWERED / REFUSED / ERROR, from the governed status alone."""
    if result.status == "success":
        return ANSWERED
    code = result.error_code
    if not code:
        return ERROR if result.status == "error" else REFUSED
    if code in _ERROR_CODES or category_for(code) in _ERROR_CATEGORIES:
        return ERROR
    return REFUSED


def _spec_interpretation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The structured interpretation, exactly as the parser produced it.

    Only keys the spec actually carries are recorded. A route that exposes no
    structured spec records an empty interpretation rather than a fabricated
    one — the absence is itself the finding an operator needs to see.
    """
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        return {}
    keys = (
        # what population was read
        "portfolio_lens", "segment", "state_filters", "filters",
        # what was measured
        "metric", "measures", "aggregation", "weight_field",
        # how it was cut
        "dimension", "dimensions", "hierarchy", "bucket_field",
        "concentration_dimension", "ranking_mode", "top_n", "sort_by",
        # over what period
        "as_of_date", "reporting_date", "start_date", "end_date",
        "baseline_date", "current_date", "compare_periods", "temporal_mode",
        "trend_grain", "cohort_grain",
        # what shape of answer was asked for
        "intent", "output_type", "chart_type", "execution_mode",
        # what the parser could not honour
        "unavailable_filters", "metric_defaulted",
    )
    return {k: spec[k] for k in keys if k in spec and spec[k] not in (None, "", [], {})}


def _artifact_kinds(payload: Dict[str, Any]) -> List[str]:
    arts = payload.get("artifacts")
    if not isinstance(arts, list):
        return []
    return sorted({str(a.get("type")) for a in arts
                   if isinstance(a, dict) and a.get("type")})


def build_record(result: GovernedResult, *, question: str,
                 requested_portfolio: Optional[str] = None,
                 query_id: Optional[str] = None,
                 asked_at: Optional[str] = None) -> Dict[str, Any]:
    """Project one governed execution into the telemetry record.

    Pure: no I/O, no computation over portfolio data, and no judgement about
    whether the answer was any good — that is the operator's, later.
    """
    payload = result.result if isinstance(result.result, dict) else {}
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else {}
    audit = result.audit
    now = asked_at or (audit.started_at if audit else None) \
        or datetime.now(timezone.utc).isoformat()
    qid = query_id or f"miq_{uuid.uuid4().hex[:16]}"
    outcome = outcome_for(result)

    return {
        "schema_version": SCHEMA_VERSION,
        # -- identity ------------------------------------------------------ #
        "query_id": qid,
        "asked_at": now,
        "day": str(now)[:10],
        "client_id": result.tenant_id,
        "portfolio_id": result.portfolio_id or requested_portfolio,
        "user_id": audit.actor_id if audit else None,
        "user_type": audit.actor_type if audit else None,
        "channel": audit.channel if audit else None,
        "organisation_id": audit.organisation_id if audit else None,
        "request_id": result.request_id,
        "correlation_id": result.correlation_id,
        # -- the question -------------------------------------------------- #
        "question": question,
        # -- data context -------------------------------------------------- #
        "snapshot_id": result.snapshot.snapshot_id if result.snapshot else None,
        "content_hash": result.snapshot.content_hash if result.snapshot else None,
        "source_kind": result.snapshot.source_kind if result.snapshot else None,
        "reporting_period": (meta.get("asOfDate") or meta.get("reportingDate")
                             or spec.get("as_of_date")
                             or spec.get("reporting_date")),
        "dataset_view": (meta.get("view") or meta.get("dataset")
                         or meta.get("datasetContext")),
        "data_source_kind": meta.get("dataSourceKind") or None,
        "data_source_label": meta.get("dataSourceLabel") or None,
        # -- interpretation ------------------------------------------------ #
        "interpretation": _spec_interpretation(payload),
        "parser": {k: v for k, v in (meta.get("parserProvenance") or {}).items()
                   if k in ("parser_used", "llm_failure", "parser_mode_detail")}
                  or ({"parser_used": meta["parserMode"]} if meta.get("parserMode")
                      else {}),
        # -- execution ----------------------------------------------------- #
        # ``route_id`` is the routed governed capability where one ran; the
        # workflow path leaves it unset, and its deterministic capability is
        # named by the interpretation's metric + intent instead. Recorded as it
        # is rather than back-filled, so "no named route" stays visible.
        "route": (spec.get("route_id") if isinstance(spec, dict) else None) \
            or meta.get("route") or None,
        "capability": result.capability,
        "engine": meta.get("engine") or None,
        "execution_mode": spec.get("execution_mode") if isinstance(spec, dict) else None,
        "result_type": meta.get("resultType") or None,
        "row_count": meta.get("rowCount"),
        "lens_applied": meta.get("lensApplied"),
        "artifact_kinds": _artifact_kinds(payload),
        # -- the answer, exactly as the user saw it ------------------------ #
        "answer": payload.get("answer") or "",
        # -- outcome ------------------------------------------------------- #
        "outcome": outcome,
        "governed_status": result.status,
        "refusal_reason": result.error_code if outcome == REFUSED else None,
        "error_code": result.error_code if outcome == ERROR else None,
        "error_category": (category_for(result.error_code)
                           if result.error_code else None),
        "message": (payload.get("error") or None) if outcome != ANSWERED else None,
        "warnings": [str(w) for w in (result.warnings or ())],
        # -- performance --------------------------------------------------- #
        "duration_ms": audit.duration_ms if audit else None,
        # -- operator quality review (never set by the system) ------------- #
        "review": {"classification": UNREVIEWED, "reviewer": None,
                   "reviewed_at": None, "note": None},
    }


def record(store, result: GovernedResult, *, question: str,
           requested_portfolio: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Persist the telemetry record. Never raises, never blocks the answer.

    A query without a tenant is not recorded: the store is client-scoped, and a
    record with no client could not be isolated.
    """
    try:
        if not result.tenant_id or not enabled():
            return None
        doc = build_record(result, question=question,
                           requested_portfolio=requested_portfolio)
        store.save_mi_query(doc)
        # A client that asks MI questions may have no OCC workflow yet, and the
        # operator's client list is the OCC index. Without this its questions
        # are recorded and then invisible to anyone browsing by client.
        # Idempotent, and the same registration every other OCC document makes.
        try:
            store.register_client(doc["client_id"])
        except Exception:  # noqa: BLE001 — indexing must not lose the record
            logger.warning("mi query telemetry recorded but client %s not "
                           "indexed", doc["client_id"], exc_info=True)
        return doc
    except Exception:  # noqa: BLE001 — telemetry must never fail a query
        logger.warning("mi query telemetry not recorded for request_id=%s",
                       getattr(result, "request_id", None), exc_info=True)
        return None
