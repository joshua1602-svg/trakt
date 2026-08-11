"""trakt_tools.handlers.provenance — ``explain_values`` (primitive), ``explain_value`` (wrapper).

    capability: loan:read

The difference between a number and a fact:

    "the balance is £183,450"
    "the balance is £183,450, read from OUT_PRIN in the July servicer tape,
     snapshot snap_7f3a91 (sha256:9c1e…), mapped at alias tier 3, validated
     against BAL001/BAL101 with no errors"

Assembled on demand, not stored per cell
----------------------------------------
Mapping, transformation and validation are properties of a **field within a
snapshot**, not of an individual cell, so the evidence is composed at request
time from three things that already exist: the compact per-field lineage index
written at ingestion (:mod:`trakt_core.lineage`), the canonical row itself, and
the snapshot identity governance already resolved. A 130-column tape therefore
has a 130-entry index rather than 130-million cell envelopes.

Bulk by default
---------------
``explain_values`` is the primitive because due diligence asks about tens of
values at once, and the alternative shape — one call per value — pays the
governance chain, the frame resolution and the index load every time.
``explain_value`` calls it with a single request.

Snapshot and tenant binding is enforced, not assumed
----------------------------------------------------
:meth:`trakt_core.lineage.LineageIndex.assert_binding` refuses an index that
names a different tenant or snapshot from the one the value was read out of.
Provenance from the wrong snapshot is worse than no provenance, because it is
confidently wrong — so the failure is an error rather than a degradation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from trakt_core.errors import ErrorCode, TraktError
from trakt_core.lineage import LINEAGE_INDEX_NAME, LineageIndex, load_lineage_index

from ..schema import object_schema
from ..spec import ToolInvocation
from .loans import LOAN_ID_FIELD, resolve_governed_frame

#: One DD sweep's worth. Same ceiling as get_loans, for the same reason.
MAX_REQUESTS = 500


# --------------------------------------------------------------------------- #
# Published contract
# --------------------------------------------------------------------------- #
_RESOURCE_PROPERTY = {
    "type": "string",
    "description": "The resource the values were read from.",
    "pattern": r"^[^/]+/[^/]+/[^/]+$",
}

EXPLAIN_VALUES_INPUT = object_schema(
    description=("Return the evidence behind SEVERAL values in one call. This is "
                 "the primitive: to justify more than one figure, call this once "
                 "rather than explain_value repeatedly."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "requests": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_REQUESTS,
            "items": {"type": "object"},
            "description": ("Objects of {loan_id, canonical_field}, at most "
                            f"{MAX_REQUESTS} per call. Answers are returned in "
                            "the order requested."),
        },
    },
    required=["resource", "requests"],
)

EXPLAIN_VALUES_OUTPUT = object_schema(
    description="One evidence envelope per requested value.",
    properties={
        "resource": {"type": "string"},
        "explanations": {"type": "array", "items": {"type": "object"}},
        "unresolved": {
            "type": "array", "items": {"type": "object"},
            "description": ("Requests that could not be answered, each with a "
                            "reason. Never silently dropped."),
        },
        "lineage_available": {
            "type": "boolean",
            "description": ("False when this snapshot carries no lineage index — "
                            "values are still returned, with their source "
                            "unstated rather than guessed."),
        },
        "snapshot": {"type": "object"},
        "requested_count": {"type": "integer"},
        "explained_count": {"type": "integer"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "explanations", "unresolved", "lineage_available",
              "snapshot", "requested_count", "explained_count"],
)

EXPLAIN_VALUE_INPUT = object_schema(
    description=("Return the evidence behind ONE value. For several values call "
                 "explain_values once instead."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "loan_id": {"type": "string"},
        "canonical_field": {"type": "string"},
    },
    required=["resource", "loan_id", "canonical_field"],
)

EXPLAIN_VALUE_OUTPUT = object_schema(
    description="The evidence behind one value.",
    properties={
        "resource": {"type": "string"},
        "explanation": {"type": ["object", "null"]},
        "resolved": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
        "lineage_available": {"type": "boolean"},
        "snapshot": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "explanation", "resolved", "lineage_available",
              "snapshot"],
)


# --------------------------------------------------------------------------- #
# Index resolution
# --------------------------------------------------------------------------- #
def _index_for(inv: ToolInvocation) -> Optional[LineageIndex]:
    """The lineage index for this resource's snapshot, or ``None``.

    ``None`` is a legitimate answer — a snapshot produced before the index
    existed — and is reported as ``lineage_available: false`` rather than
    invented.
    """
    deps = inv.dependencies
    override = getattr(deps, "lineage_index", None)
    if override is not None:
        index = override
    else:
        path = getattr(deps, "lineage_index_path", None)
        if path is None:
            from mi_agent_api import datasets as datasets_mod
            from pathlib import Path

            root = datasets_mod._onboarding_output_root()
            if not root:
                return None
            path = Path(root) / LINEAGE_INDEX_NAME
        index = load_lineage_index(path, request_id=inv.request_id)

    if index is None:
        return None
    # Binding is checked here, once, rather than per value — and it RAISES.
    index.assert_binding(tenant_id=inv.tenant_id,
                         snapshot_id=getattr(inv.snapshot, "snapshot_id", None),
                         request_id=inv.request_id)
    return index


def _snapshot_block(inv: ToolInvocation) -> Dict[str, Any]:
    snapshot = inv.snapshot
    return {
        "snapshot_id": getattr(snapshot, "snapshot_id", None),
        "content_hash": getattr(snapshot, "content_hash", None),
        "dataset_label": getattr(snapshot, "dataset_label", None),
        "reporting_date": getattr(snapshot, "reporting_date", None),
        "source_base": getattr(snapshot, "source_base", None),
    }


def _envelope(loan_id: str, canonical_field: str, value: Any,
              entry: Any, snapshot: Dict[str, Any],
              resource_key: str) -> Dict[str, Any]:
    """The minimum provenance envelope: value, source, dates, transformation,
    validation, calculation, evidence."""
    return {
        "loan_id": loan_id,
        "canonical_field": canonical_field,
        "value": value,
        "unit": getattr(entry, "unit", None),
        "format": getattr(entry, "format", None),
        "source": {
            "resource": resource_key,
            "dataset_label": snapshot.get("dataset_label"),
            "source_dataset": getattr(entry, "source_dataset", None),
            "snapshot_id": snapshot.get("snapshot_id"),
            "content_hash": snapshot.get("content_hash"),
        },
        "source_field": getattr(entry, "source_field", None),
        "origin": getattr(entry, "origin", None),
        "effective_date": snapshot.get("reporting_date"),
        "transformation": {
            "mapping_method": getattr(entry, "mapping_method", None),
            "mapping_confidence": getattr(entry, "mapping_confidence", None),
            "mapping_version": getattr(entry, "mapping_version", None),
            "derivation_rule": getattr(entry, "derivation_rule", None),
        },
        "validation": {
            "status": getattr(entry, "validation_status", "not_validated"),
            "rules_applied": list(getattr(entry, "validation_rules", ()) or ()),
            "error_count": getattr(entry, "validation_error_count", None),
        },
        "calculation": {
            "method": getattr(entry, "calculation_method", None),
            "version": getattr(entry, "calculation_version", None),
        },
    }


# --------------------------------------------------------------------------- #
# The primitive
# --------------------------------------------------------------------------- #
def explain_values(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """Evidence for a bounded batch of values, in ONE frame access and ONE index load."""
    raw_requests = args.get("requests") or []
    if not raw_requests:
        raise TraktError(ErrorCode.TOOL_INPUT_INVALID,
                         "At least one {loan_id, canonical_field} request is required.",
                         request_id=inv.request_id)
    if len(raw_requests) > MAX_REQUESTS:
        raise TraktError(
            ErrorCode.TOOL_INPUT_INVALID,
            f"At most {MAX_REQUESTS} values may be explained in one call.",
            request_id=inv.request_id,
            details={"requested": len(raw_requests), "maximum": MAX_REQUESTS})

    warnings: List[str] = []
    unresolved: List[Dict[str, Any]] = []
    pairs: List[tuple] = []
    for item in raw_requests:
        if not isinstance(item, dict):
            unresolved.append({"request": str(item),
                               "reason": "not an object of {loan_id, canonical_field}"})
            continue
        loan_id = str(item.get("loan_id") or "").strip()
        canonical_field = str(item.get("canonical_field") or "").strip()
        if not loan_id or not canonical_field:
            unresolved.append({"loan_id": loan_id or None,
                               "canonical_field": canonical_field or None,
                               "reason": "loan_id and canonical_field are both required"})
            continue
        pairs.append((loan_id, canonical_field))

    snapshot = _snapshot_block(inv)
    resource_key = inv.authorised.resource.ref.key
    if not pairs:
        return {"resource": resource_key, "explanations": [], "unresolved": unresolved,
                "lineage_available": False, "snapshot": snapshot,
                "requested_count": len(raw_requests), "explained_count": 0,
                "warnings": warnings}

    index = _index_for(inv)
    if index is None:
        warnings.append(
            "This snapshot carries no lineage index, so mapping, transformation "
            "and validation are unstated. Values are still returned from the "
            "governed frame; their source is unknown rather than assumed.")

    # ---- ONE frame access for the whole batch ---------------------------- #
    df = resolve_governed_frame(inv)
    wanted_loans = {loan_id for loan_id, _ in pairs}
    wanted_fields = sorted({f for _, f in pairs if f in df.columns})
    missing_fields = sorted({f for _, f in pairs if f not in df.columns})

    columns = [LOAN_ID_FIELD] + [f for f in wanted_fields if f != LOAN_ID_FIELD]
    ids = df[LOAN_ID_FIELD].astype(str)
    matched = df.loc[ids.isin(wanted_loans), columns]

    from .loans import _jsonable

    rows: Dict[str, Dict[str, Any]] = {}
    for record in matched.to_dict(orient="records"):
        key = str(record.get(LOAN_ID_FIELD))
        rows.setdefault(key, {k: _jsonable(v) for k, v in record.items()})

    explanations: List[Dict[str, Any]] = []
    for loan_id, canonical_field in pairs:
        if canonical_field in missing_fields:
            unresolved.append({"loan_id": loan_id, "canonical_field": canonical_field,
                               "reason": "this tape does not carry that canonical field"})
            continue
        row = rows.get(loan_id)
        if row is None:
            unresolved.append({"loan_id": loan_id, "canonical_field": canonical_field,
                               "reason": "no such loan in this resource"})
            continue
        entry = index.get(canonical_field) if index is not None else None
        explanations.append(_envelope(loan_id, canonical_field,
                                      row.get(canonical_field), entry,
                                      snapshot, resource_key))

    return {
        "resource": resource_key,
        "explanations": explanations,
        "unresolved": unresolved,
        "lineage_available": index is not None,
        "snapshot": snapshot,
        "requested_count": len(raw_requests),
        "explained_count": len(explanations),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #
def explain_value(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """One value's evidence. A thin wrapper over :func:`explain_values`."""
    batch = explain_values(
        {"resource": args.get("resource"),
         "requests": [{"loan_id": args.get("loan_id"),
                       "canonical_field": args.get("canonical_field")}]}, inv)
    explanations = batch["explanations"]
    unresolved = batch["unresolved"]
    return {
        "resource": batch["resource"],
        "explanation": explanations[0] if explanations else None,
        "resolved": bool(explanations),
        "reason": (unresolved[0].get("reason") if unresolved and not explanations
                   else None),
        "lineage_available": batch["lineage_available"],
        "snapshot": batch["snapshot"],
        "warnings": batch["warnings"],
    }
