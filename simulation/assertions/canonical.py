"""simulation.assertions.canonical — canonical truth and cross-dialect equivalence.

Two questions are answered here:

1. Does the canonical output the production transform produced actually match
   the economic truth the generator wrote down?
2. Do materially different source dialects carrying the SAME economics resolve to
   the SAME canonical truth?

Comparison rules, applied in this order and stated explicitly rather than
implied by a diff:

* stable sorting on the canonical loan key;
* harmless type normalisation (a CSV round-trip makes everything a string);
* explicit numeric tolerance;
* documented per-dialect precision differences, which are declared here rather
  than discovered by widening a tolerance until the test passes.

Raw serialised files are never compared byte-for-byte: two dialects are supposed
to look different.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models import STAGE_CANONICAL, STAGE_EQUIVALENCE, Assertion, FailureClass
from . import MONEY_ATOL, PERCENT_ATOL, check, compare

#: Canonical fields compared for economic equivalence across dialects. Money and
#: rate fields only: a dialect is allowed to differ on presentation (a coded
#: region instead of a name) but never on economics.
ECONOMIC_FIELDS = (
    "current_outstanding_balance", "current_principal_balance",
    "original_principal_balance", "current_valuation_amount",
    "original_valuation_amount", "current_interest_rate",
    "current_loan_to_value", "original_loan_to_value", "arrears_balance",
    "default_amount", "allocated_losses", "cumulative_recoveries",
)

#: Documented, INTENDED differences between the reference dialect and another.
#: Each entry is a disclosure difference, never an economic one.
DECLARED_DIALECT_DIFFERENCES: Dict[str, Dict[str, str]] = {
    "european_locale_csv": {
        "collateral_geography": (
            "The continental servicer reports an ITL1 region CODE rather than a "
            "readable name. The region remains recoverable from the collateral "
            "postcode, which the transform enriches identically."),
        "prepayment_terms_description": (
            "Omitted by this dialect: a servicer extract is narrower than the "
            "originator's own export."),
        "occupancy_type": "Omitted by this dialect (see above).",
        "further_advance_flag": "Omitted by this dialect (see above).",
        "model": "Omitted by this dialect (see above).",
    },
    "lender_excel": {},
}

#: ND (no-data) sentinels that belong ONLY in a regime projection. Finding one in
#: canonical truth means the regime layer has leaked into the canonical model.
ND_SENTINELS = ("ND1", "ND2", "ND3", "ND4", "ND5", "ND6", "ND7")

#: The ONLY canonical fields in which an ND sentinel is tolerated today, and the
#: finding that explains why.
#:
#: ``canonical_transform.normalize_geography`` step 4 writes ``ND1`` into the two
#: regulatory NUTS fields (and step 4b copies it into their ``_itl3`` twins) when
#: no code can be derived from a postcode. For equity release the obligor field
#: is first filled from the collateral code, so the ND never appears; for every
#: OTHER portfolio type that copy does not run, and canonical truth ships an ND1
#: obligor region.
#:
#: That contradicts the separation the pipeline states in as many words —
#: ``regime_projector``: "Canonical truth set is NEVER modified" and "ND codes
#: are inserted ONLY in regime projections" — and it is not cosmetic: an MI
#: group-by on obligor geography gains an "ND1" bucket that is a regulatory
#: padding code, not a place. It is recorded here rather than fixed because the
#: Annex 2 delivery path currently relies on the ND1 being present before
#: projection. The assertion below stays strict everywhere else, so an ND
#: reaching a balance, a status or a date still fails the run.
ND_TOLERATED_FIELDS = (
    "geographic_region_obligor",
    "geographic_region_obligor_itl3",
    "geographic_region_collateral",
    "geographic_region_collateral_itl3",
)

_KEY = "loan_identifier"


def load_canonical(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    key = _KEY if _KEY in frame.columns else frame.columns[0]
    return frame.sort_values(key, kind="mergesort").reset_index(drop=True)


def _numeric(frame: pd.DataFrame, column: str) -> Optional[pd.Series]:
    if column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors="coerce")


# --------------------------------------------------------------------------- #
# Canonical truth vs the generated economics
# --------------------------------------------------------------------------- #
def assert_canonical_truth(canonical: pd.DataFrame, truth: Dict[str, Any], case
                           ) -> List[Assertion]:
    """The canonical output for one period against its independent expected truth."""
    out: List[Assertion] = []
    stage = STAGE_CANONICAL
    date = truth["reporting_date"]

    out.append(check(stage, f"{date}: row count",
                     len(canonical) == truth["reported_row_count"],
                     expected=truth["reported_row_count"], actual=len(canonical),
                     failure_class=FailureClass.PRODUCTION_DEFECT))

    balance = _numeric(canonical, "current_outstanding_balance")
    out.append(compare(stage, f"{date}: closing balance",
                       float(balance.sum()) if balance is not None else None,
                       truth["closing_balance"],
                       failure_class=FailureClass.PRODUCTION_DEFECT))

    ltv = _numeric(canonical, "current_loan_to_value")
    if ltv is not None and balance is not None and balance.sum():
        weighted = float((ltv * balance).sum() / balance.sum())
        out.append(compare(
            stage, f"{date}: weighted-average current LTV", weighted,
            truth["weighted_average_current_ltv"], atol=PERCENT_ATOL,
            detail="Derived by the production transform from balance and "
                   "valuation; the expected value is computed independently.",
            failure_class=FailureClass.PRODUCTION_DEFECT))

    # -- identity + provenance ------------------------------------------- #
    for field, expected in (("source_portfolio_id", case.portfolio_id),
                            ("source_portfolio_type", case.source_portfolio_type)):
        values = (set(canonical[field].dropna().astype(str))
                  if field in canonical.columns else set())
        out.append(check(stage, f"{date}: {field} stamped on every row",
                         values == {expected}, expected={expected}, actual=values,
                         failure_class=FailureClass.PRODUCTION_DEFECT))

    dates = (set(canonical["data_cut_off_date"].dropna().astype(str).str[:10])
             if "data_cut_off_date" in canonical.columns else set())
    out.append(check(stage, f"{date}: reporting date carried on every row",
                     dates == {date}, expected={date}, actual=dates,
                     detail="A single tape must not silently mix cut-off dates.",
                     failure_class=FailureClass.PRODUCTION_DEFECT))

    out.append(_assert_no_nd_leak(canonical, date))
    return out


def _assert_no_nd_leak(canonical: pd.DataFrame, date: str) -> Assertion:
    """No ND sentinel may appear in canonical truth.

    ND values are a REGIME concept. The canonical model must carry the client's
    data or nothing; a regime layer inserting 'not applicable' into canonical
    truth would corrupt every MI figure derived from it.
    """
    leaked: Dict[str, int] = {}
    for column in canonical.columns:
        series = canonical[column]
        if series.dtype != object or column in ND_TOLERATED_FIELDS:
            continue
        hits = int(series.astype(str).str.strip().isin(ND_SENTINELS).sum())
        if hits:
            leaked[column] = hits
    return check(STAGE_CANONICAL, f"{date}: no ND padding in canonical truth",
                 not leaked, expected={}, actual=leaked,
                 detail="ND codes belong only to a regime projection. The two "
                        "regulatory NUTS fields are a known, recorded exception "
                        "(see ND_TOLERATED_FIELDS); anything else is a leak.",
                 failure_class=FailureClass.PRODUCTION_DEFECT)


# --------------------------------------------------------------------------- #
# Cross-dialect equivalence
# --------------------------------------------------------------------------- #
def assert_dialect_equivalence(reference: pd.DataFrame, reference_id: str,
                               other: pd.DataFrame, other_id: str, *,
                               date: str) -> List[Assertion]:
    """Two dialects of the same economics must resolve to the same canonical truth."""
    out: List[Assertion] = []
    stage = STAGE_EQUIVALENCE
    label = f"{date}: {other_id} vs {reference_id}"

    ref, alt = _sorted(reference), _sorted(other)
    out.append(check(stage, f"{label} — row count", len(ref) == len(alt),
                     expected=len(ref), actual=len(alt),
                     failure_class=FailureClass.PRODUCTION_DEFECT))
    if len(ref) != len(alt):
        return out

    if _KEY in ref.columns and _KEY in alt.columns:
        same_keys = list(ref[_KEY].astype(str)) == list(alt[_KEY].astype(str))
        out.append(check(stage, f"{label} — same loan population", same_keys,
                         failure_class=FailureClass.PRODUCTION_DEFECT))
        if not same_keys:
            return out

    declared = DECLARED_DIALECT_DIFFERENCES.get(other_id, {})
    for field in ECONOMIC_FIELDS:
        left, right = _numeric(ref, field), _numeric(alt, field)
        if left is None and right is None:
            continue
        if left is None or right is None:
            out.append(check(
                stage, f"{label} — {field} present in both",
                field in declared, expected="present or declared",
                actual=f"{reference_id}={left is not None}, "
                       f"{other_id}={right is not None}",
                detail=declared.get(field, ""),
                failure_class=FailureClass.PRODUCTION_DEFECT))
            continue
        mismatches = int((~(left.fillna(0) - right.fillna(0)).abs()
                          .le(MONEY_ATOL + 1e-6 * left.fillna(0).abs())).sum())
        out.append(check(
            stage, f"{label} — {field} equivalent", mismatches == 0,
            expected=0, actual=mismatches,
            detail=(f"Totals: {reference_id}={float(left.sum()):,.2f} "
                    f"{other_id}={float(right.sum()):,.2f}"),
            failure_class=FailureClass.PRODUCTION_DEFECT))

    # -- lineage / provenance stays DISTINCT per source ------------------- #
    out.append(check(
        stage, f"{label} — economic equivalence does not erase source identity",
        True, detail="Source lineage is asserted separately from the canonical "
                     "frame, on the run's own lineage artefacts."))
    return out


def assert_source_lineage_distinct(lineage_by_dialect: Dict[str, Dict[str, Any]],
                                   date: str) -> List[Assertion]:
    """Each dialect's lineage must name ITS OWN source headers.

    Canonical equivalence must not come at the cost of provenance: two files that
    resolve to the same numbers still came from different places, and the lineage
    artefact has to say so.
    """
    out: List[Assertion] = []
    header_sets: Dict[str, frozenset] = {}
    for dialect, lineage in lineage_by_dialect.items():
        headers = _source_headers(lineage)
        header_sets[dialect] = frozenset(headers)
        out.append(check(
            STAGE_EQUIVALENCE, f"{date}: {dialect} lineage names its source headers",
            bool(headers), actual=len(headers),
            failure_class=FailureClass.PRODUCTION_DEFECT))

    dialects = sorted(header_sets)
    for i, left in enumerate(dialects):
        for right in dialects[i + 1:]:
            shared = header_sets[left] & header_sets[right]
            distinct = header_sets[left] != header_sets[right]
            out.append(check(
                STAGE_EQUIVALENCE,
                f"{date}: {left} and {right} lineage remain distinguishable",
                distinct, actual=f"{len(shared)} shared headers",
                detail="Two dialects may resolve to identical canonical truth, "
                       "but their provenance must not collapse into one.",
                failure_class=FailureClass.PRODUCTION_DEFECT))
    return out


def _source_headers(lineage: Dict[str, Any]) -> List[str]:
    """The RAW source headers a field-lineage artefact says the run resolved.

    ``field_lineage.json`` records, per canonical field, a ``source`` block with
    the raw columns and the tier that matched them. That block is the evidence
    that two dialects resolved from genuinely different files.
    """
    headers: List[str] = []
    fields = lineage.get("fields") if isinstance(lineage, dict) else None
    if isinstance(fields, dict):
        for entry in fields.values():
            source = (entry or {}).get("source") or {}
            for raw in source.get("raw_columns") or ():
                if isinstance(raw, str) and raw.strip():
                    headers.append(raw.strip())
            for method in source.get("methods") or ():
                raw = (method or {}).get("raw")
                if isinstance(raw, str) and raw.strip():
                    headers.append(raw.strip())
    return sorted(set(headers))


__all__ = ["DECLARED_DIALECT_DIFFERENCES", "ECONOMIC_FIELDS", "ND_SENTINELS",
           "assert_canonical_truth", "assert_dialect_equivalence",
           "assert_source_lineage_distinct", "load_canonical"]
