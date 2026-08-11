"""trakt_tools.handlers.loans — ``get_loans`` (primitive) and ``get_loan`` (wrapper).

    capability: loan:read
    source:     the SAME governed canonical frame MI Query and the workspace read

**Batch-first, deliberately.** Measured on a canonical-shaped frame, retrieving
twenty loans one at a time costs 1,617 ms at one million rows against 76 ms for a
single batched call — a 21× penalty (37× at 100k rows), before HTTP and the
governance chain are counted at all. An agent handed only a single-loan tool will
call it once per loan, so the batch form has to be the primitive rather than an
optimisation offered later.

``get_loan`` is therefore a thin wrapper that calls ``get_loans`` with a
one-element list. There is one implementation, and
``tests/test_agent_loan_retrieval.py`` asserts the wrapper adds no lookup of its
own.

Two shapes, one retrieval
-------------------------
``shape="flat"`` returns canonical fields as one object per loan — the original
contract, unchanged. ``shape="structured"`` groups the SAME values into the
credit objects they belong to (loan / borrower / collateral+valuations /
contract / performance) using ``trakt_core.entity``. Assembly reads the row
already in hand, so a structured response costs a dictionary walk rather than
another query, and no value is recomputed on the way.

What this module does NOT do
---------------------------
* **No calculation.** Every value is carried through from the canonical frame
  unchanged. A derived figure belongs in the canonical transform, where MI and
  the agent both read it.
* **No second data path.** The frame comes from the same governed resolver the MI
  Query path uses, so a loan an agent reads and a loan the workspace shows are
  the same row of the same snapshot.
* **No object store.** The structured shape is assembled per request from the
  canonical row; there is no materialised rich-object database to keep in step
  and no second answer to "what is the balance".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from trakt_core.errors import ErrorCode, TraktError
from trakt_core.portfolio import FIELD_PORTFOLIO_ID

from ..schema import object_schema
from ..spec import ToolInvocation

#: The canonical identifier a loan is addressed by.
LOAN_ID_FIELD = "loan_identifier"

#: Hard ceiling on one batch. Above this an agent is doing analysis rather than
#: investigation and should be using stratify / concentration / rank_loans, which
#: return aggregates instead of rows.
MAX_LOAN_IDS = 500

#: What a caller gets when it names no fields. Curated rather than "everything":
#: a canonical tape carries 66-130 columns and an agent needs a fraction of them,
#: which is also what makes a future columnar projection worth having. Fields
#: absent from a given tape are simply omitted — this is a superset across asset
#: classes, not a required contract.
DEFAULT_FIELDS: tuple = (
    LOAN_ID_FIELD, "borrower_identifier", FIELD_PORTFOLIO_ID, "portfolio_cohort",
    "account_status", "current_principal_balance", "current_outstanding_balance",
    "original_principal_balance", "current_loan_to_value", "indexed_loan_to_value",
    "current_interest_rate", "arrears_balance", "number_of_days_in_arrears",
    "ifrs9_stage", "internal_risk_stage", "origination_date", "maturity_date",
    "current_valuation_amount", "current_valuation_date", "current_valuation_method",
    "original_valuation_amount", "collateral_type", "geographic_region_obligor",
    "geographic_region_collateral", "exposure_currency_denomination",
    "amortisation_type", "reporting_date",
)


# --------------------------------------------------------------------------- #
# Published contract
# --------------------------------------------------------------------------- #
_RESOURCE_PROPERTY = {
    "type": "string",
    "description": ("The resource holding the loans, as "
                    "'{tenant}/{kind}/{resource_id}'. Call GET /v1/agent/tools "
                    "to see the resources you may name."),
    "pattern": r"^[^/]+/[^/]+/[^/]+$",
}

_FIELDS_PROPERTY = {
    "type": ["array", "null"],
    "items": {"type": "string"},
    "description": ("Canonical field names to return. Omit for a curated default "
                    "projection; ask for only what you need — a canonical tape "
                    "carries over a hundred columns."),
    "default": None,
}

GET_LOANS_INPUT = object_schema(
    description=("Retrieve several loans in ONE call. This is the primitive: to "
                 "look at more than one loan, call this once rather than "
                 "get_loan repeatedly."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "loan_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": MAX_LOAN_IDS,
            "description": (f"Loan identifiers, at most {MAX_LOAN_IDS} per call. "
                            "Results are returned in the order requested."),
        },
        "fields": _FIELDS_PROPERTY,
        "shape": {
            "type": "string",
            "enum": ["flat", "structured"],
            "description": (
                "'flat' returns canonical fields as one object per loan. "
                "'structured' groups them into the credit objects they belong "
                "to — loan, borrower, collateral (with valuations), contract, "
                "performance — which is usually what you want to reason about."),
            "default": "flat",
        },
        "include": {
            "type": ["array", "null"],
            "items": {"type": "string", "enum": ["valuations"]},
            "description": ("Repeating children to include. Only meaningful with "
                            "shape='structured'. Omit for scalar sections only."),
            "default": None,
        },
    },
    required=["resource", "loan_ids"],
)

GET_LOANS_OUTPUT = object_schema(
    description="The requested loans, plus an explicit account of any not found.",
    properties={
        "resource": {"type": "string"},
        "loans": {"type": "array", "items": {"type": "object"}},
        "not_found": {
            "type": "array", "items": {"type": "string"},
            "description": ("Identifiers that matched no loan in this resource. "
                            "Never silently omitted: an absent loan is a finding, "
                            "not a smaller answer."),
        },
        "fields_returned": {"type": "array", "items": {"type": "string"}},
        "fields_unavailable": {
            "type": "array", "items": {"type": "string"},
            "description": "Requested fields this tape does not carry.",
        },
        "requested_count": {"type": "integer"},
        "returned_count": {"type": "integer"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "loans", "not_found", "fields_returned",
              "requested_count", "returned_count", "scope"],
)

GET_LOAN_INPUT = object_schema(
    description=("Retrieve ONE loan you are already investigating. For several "
                 "loans call get_loans once instead — it is the same work in one "
                 "request."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "loan_id": {"type": "string", "description": "The loan identifier."},
        "fields": _FIELDS_PROPERTY,
        "shape": {"type": "string", "enum": ["flat", "structured"],
                  "description": "See get_loans.", "default": "flat"},
        "include": {"type": ["array", "null"],
                    "items": {"type": "string", "enum": ["valuations"]},
                    "description": "See get_loans.", "default": None},
    },
    required=["resource", "loan_id"],
)

GET_LOAN_OUTPUT = object_schema(
    description="One loan, or an explicit not-found.",
    properties={
        "resource": {"type": "string"},
        "loan": {"type": ["object", "null"]},
        "found": {"type": "boolean"},
        "fields_returned": {"type": "array", "items": {"type": "string"}},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "loan", "found", "fields_returned", "scope"],
)


# --------------------------------------------------------------------------- #
# Shared frame resolution
# --------------------------------------------------------------------------- #
def resolve_governed_frame(inv: ToolInvocation):
    """The governed canonical frame, narrowed to what this authorisation permits.

    Raises rather than widening. The narrowing here is the *authorisation*, not a
    presentation lens, so an unenforceable boundary must refuse: applying a
    book scope to a tape that carries no ``source_portfolio_id`` would return the
    whole book under a narrower label.
    """
    deps = inv.dependencies
    resolved = inv.authorised.resource

    resolver = getattr(deps, "loan_frame_resolver", None)
    if resolver is None:
        from mi_agent_api import datasets as datasets_mod

        def resolver():
            frame, error = datasets_mod._resolve_query_frame("funded", None)
            if error:
                raise TraktError(ErrorCode.DATA_SOURCE_UNAVAILABLE, error,
                                 request_id=inv.request_id)
            return frame

    df = resolver()
    if df is None or not len(df):
        raise TraktError(
            ErrorCode.DATA_SOURCE_UNAVAILABLE,
            "No governed loan data is available for this resource.",
            request_id=inv.request_id)

    if resolved.spv_id:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            "This resource is defined by an SPV boundary, which the canonical "
            "loan frame cannot currently apply. Answering it would return the "
            "enclosing book rather than the SPV.",
            request_id=inv.request_id, details={"resource": resolved.ref.key})

    if resolved.source_portfolio_ids:
        if FIELD_PORTFOLIO_ID not in df.columns:
            raise TraktError(
                ErrorCode.RESOURCE_NOT_PARTITIONABLE,
                f"The governed tape carries no {FIELD_PORTFOLIO_ID} column, so a "
                "book boundary cannot be enforced on it.",
                request_id=inv.request_id,
                details={"resource": resolved.ref.key,
                         "required_fields": sorted(resolved.required_fields)})
        df = df[df[FIELD_PORTFOLIO_ID].astype(str)
                .isin([str(p) for p in resolved.source_portfolio_ids])]
    elif not resolved.whole_tenant_book:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            "This resource declares no book narrowing and is not registered as "
            "the tenant's whole book, so the population it means is undefined.",
            request_id=inv.request_id, details={"resource": resolved.ref.key})

    if LOAN_ID_FIELD not in df.columns:
        raise TraktError(
            ErrorCode.CALCULATION_FAILED,
            f"The governed tape carries no {LOAN_ID_FIELD} column, so loans "
            "cannot be addressed individually.",
            request_id=inv.request_id)
    return df


def _collateral_identity_columns() -> List[str]:
    """The columns the collateral identity may be resolved from.

    They have to survive the projection. A tape carrying ``collateral_id`` and
    no ``collateral_identifier`` would otherwise assemble against the loan-id
    fallback while ``explain_values`` — which pulls the identity columns for its
    own derivation — resolved the real one, and the SAME valuation would carry
    two ids in the two responses.
    """
    from trakt_core.entity import load_entity_model

    model = load_entity_model()
    spec = model.entity("collateral") if model.configured else None
    if spec is None:
        return ["collateral_identifier", "collateral_id"]
    return [c for c in (spec.key, "collateral_id", spec.key_fallback) if c]


def _valuation_observation_columns() -> List[str]:
    """Canonical columns the declared valuation observations are read from."""
    from trakt_core.entity import load_entity_model

    model = load_entity_model()
    spec = model.entity("valuation") if model.configured else None
    if spec is None:
        return []
    out: List[str] = ["collateral_currency", "exposure_currency_denomination"]
    for columns in spec.observation_fields.values():
        out.extend(str(v) for v in columns.values())
    return out


def _scope_block(inv: ToolInvocation) -> Dict[str, Any]:
    resolved = inv.authorised.resource
    return {
        "resource": resolved.ref.key,
        "display_label": resolved.display_label,
        "source_portfolio_ids": list(resolved.source_portfolio_ids),
        "whole_tenant_book": resolved.whole_tenant_book,
        "attribution": resolved.attribution,
    }


def _jsonable(value: Any) -> Any:
    """A pandas cell as JSON. Nulls become ``None``; nothing is reformatted."""
    try:
        import pandas as pd

        if value is None or (not isinstance(value, (list, dict, tuple))
                             and pd.isna(value)):
            return None
    except Exception:  # noqa: BLE001 - a value we cannot test is passed through
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):          # numpy scalar
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return value


# --------------------------------------------------------------------------- #
# The primitive
# --------------------------------------------------------------------------- #
def get_loans(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """Retrieve a bounded batch of loans in ONE dataset access.

    The whole point is the absence of a loop: identifiers are matched with a
    single vectorised membership test, not one lookup per loan.
    """
    loan_ids = [str(x) for x in (args.get("loan_ids") or [])]
    if not loan_ids:
        raise TraktError(ErrorCode.TOOL_INPUT_INVALID,
                         "At least one loan identifier is required.",
                         request_id=inv.request_id)
    if len(loan_ids) > MAX_LOAN_IDS:
        # The schema also bounds this; the handler refuses independently so the
        # limit holds for an in-process caller that bypassed the schema.
        raise TraktError(
            ErrorCode.TOOL_INPUT_INVALID,
            f"At most {MAX_LOAN_IDS} loans may be requested in one call. For a "
            "larger population use the aggregate tools (stratify, concentration, "
            "rank_loans) rather than retrieving rows.",
            request_id=inv.request_id,
            details={"requested": len(loan_ids), "maximum": MAX_LOAN_IDS})

    warnings: List[str] = []
    # Duplicates are collapsed rather than served twice — the caller gets one
    # record per distinct loan and the order it asked in.
    seen: set = set()
    ordered_ids: List[str] = []
    for loan_id in loan_ids:
        if loan_id not in seen:
            seen.add(loan_id)
            ordered_ids.append(loan_id)
    if len(ordered_ids) != len(loan_ids):
        warnings.append(
            f"{len(loan_ids) - len(ordered_ids)} duplicate identifier(s) were "
            "collapsed; one record is returned per distinct loan.")

    df = resolve_governed_frame(inv)

    requested_fields = args.get("fields")
    wanted = ([str(f) for f in requested_fields] if requested_fields
              else list(DEFAULT_FIELDS))
    if LOAN_ID_FIELD not in wanted:
        wanted.insert(0, LOAN_ID_FIELD)
    # Assembling into entities means the identity columns must survive the
    # projection, and including valuations means the observation columns must
    # too — otherwise the history is silently truncated to whatever the default
    # field list happened to carry, and the collateral resolves to a fallback.
    # Both are read from the entity model rather than hard-coded, so declaring a
    # new observation type does not also require editing this projection.
    extra: List[str] = []
    if str(args.get("shape") or "flat") == "structured":
        extra.extend(_collateral_identity_columns())
    if "valuations" in (args.get("include") or ()):
        extra.extend(_valuation_observation_columns())
    if extra:
        wanted = list(wanted) + [f for f in extra if f not in wanted]

    present = [f for f in wanted if f in df.columns]
    unavailable = [f for f in wanted if f not in df.columns]
    # Only an EXPLICIT request for a missing field is worth a warning; the
    # default projection is a cross-asset-class superset and is expected to be
    # partially absent on any one tape.
    explicit_unavailable = ([f for f in unavailable if f in (requested_fields or ())]
                            if requested_fields else [])
    if explicit_unavailable:
        warnings.append(
            f"This tape does not carry: {', '.join(sorted(explicit_unavailable))}.")

    # ---- ONE vectorised selection, no per-loan lookup -------------------- #
    ids = df[LOAN_ID_FIELD].astype(str)
    matched = df.loc[ids.isin(ordered_ids), present]
    by_id: Dict[str, Dict[str, Any]] = {}
    for record in matched.to_dict(orient="records"):
        key = str(record.get(LOAN_ID_FIELD))
        if key not in by_id:            # first row wins; duplicates flagged below
            by_id[key] = {k: _jsonable(v) for k, v in record.items()}

    if len(matched) > len(by_id):
        warnings.append(
            f"{len(matched) - len(by_id)} duplicate loan row(s) were found in the "
            "governed frame; the first occurrence of each identifier is returned.")

    # Deterministic ordering: exactly the order requested, so a caller can zip
    # results against its own list without matching on a key.
    loans = [by_id[i] for i in ordered_ids if i in by_id]
    not_found = [i for i in ordered_ids if i not in by_id]

    # ---- optional structured assembly ------------------------------------ #
    # The SAME records, grouped into the credit objects they belong to. Nothing
    # is recomputed and nothing is fetched again: assembly reads the row already
    # in hand, so a structured response costs a dictionary walk, not a query.
    shape = str(args.get("shape") or "flat")
    include = tuple(args.get("include") or ())
    if shape == "structured":
        from trakt_core.entity import assemble_loan, load_entity_model

        model = load_entity_model()
        if not model.configured:
            warnings.append(
                "No entity model is configured, so loans are returned flat. "
                "The values are unchanged.")
        else:
            loans = [assemble_loan(loan, model, include=include) for loan in loans]

    snapshot_id = getattr(inv.snapshot, "snapshot_id", None)
    for loan in loans:
        # A pointer, not an envelope. The detail is what explain_values is for;
        # inlining full provenance for every field of every loan would make the
        # response unusable.
        loan_id = (loan.get(LOAN_ID_FIELD)
                   or (loan.get("_entity_model") or {}).get("loan_id"))
        loan["_provenance_ref"] = {
            "resource": inv.authorised.resource.ref.key,
            "snapshot_id": snapshot_id,
            "loan_id": loan_id,
        }

    return {
        "resource": inv.authorised.resource.ref.key,
        "loans": loans,
        "not_found": not_found,
        "fields_returned": present,
        "fields_unavailable": sorted(unavailable),
        "requested_count": len(ordered_ids),
        "returned_count": len(loans),
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #
def get_loan(args: Dict[str, Any], inv: ToolInvocation) -> Dict[str, Any]:
    """One loan. A thin wrapper over :func:`get_loans` — no second lookup path."""
    batch = get_loans({"resource": args.get("resource"),
                       "loan_ids": [args.get("loan_id")],
                       "fields": args.get("fields"),
                       "shape": args.get("shape"),
                       "include": args.get("include")}, inv)
    loans = batch["loans"]
    return {
        "resource": batch["resource"],
        "loan": loans[0] if loans else None,
        "found": bool(loans),
        "fields_returned": batch["fields_returned"],
        "fields_unavailable": batch["fields_unavailable"],
        "scope": batch["scope"],
        "warnings": batch["warnings"],
    }
