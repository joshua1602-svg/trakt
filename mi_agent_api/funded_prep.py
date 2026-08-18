"""Funded MI data preparation — promoted central lender tape -> analytics-ready.

The promoted ``18_central_lender_tape.csv`` carries per-loan canonical fields but
not all the derived MI **dimensions** the semantic layer strat-charts on. This
step derives the bucket *source* fields the tape supports and then runs the
EXISTING, canonical bucketing engine (``analytics_lib.buckets`` over
``config/mi/buckets.yaml``) — the same engine Streamlit uses — so Streamlit /
React / API stay consistent. No bucketing lives in React.

LTV derivation (product rule — LTV is NOT "missing" just because no raw LTV column
is supplied):
  * ``current_loan_to_value``  : prefer an explicit, valid source value; otherwise
    derive ``current_outstanding_balance / current_valuation_amount``.
  * ``original_loan_to_value`` : prefer explicit; otherwise derive
    ``original_principal_balance / original_valuation_amount``.
  Derivation is divide-by-zero / non-numeric safe, normalised to a 0..1 ratio
  (the bucket engine then maps to ltv_bucket / original_ltv_bucket), and records a
  derivation basis (source fields, numerator, denominator, method, confidence).

Other derivations: ``vintage_year`` + ``months_on_book`` from ``origination_date``.

Dimensions that need a source the funded tape lacks are reported in
``missing_dimensions`` with an exact reason code (``derivation_inputs_missing`` |
``not_in_central_tape`` | ``bucket_error``). Finer upstream reasons
(``raw_not_found`` | ``mapped_but_out_of_scope`` | ``source_period_ineligible`` |
``join_failed``) are produced by ``funded_mi_trace`` from the run artefacts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics_lib.dates import coerce_dates
from analytics_lib.numeric import coerce_numeric
from trakt_core import perf as _perf

from engine import region_taxonomy as _region
from mi_agent import seasoning as _seasoning

logger = logging.getLogger("mi_agent_api.funded_prep")

# Percent-vs-fraction detector (mirrors analytics_lib.buckets median heuristic):
# an LTV column whose median exceeds this is treated as a percentage (÷100).
_PERCENT_MEDIAN = 1.5

# Core funded stratification dimensions and how each is sourced.
#   kind: bucket_ltv | bucket | bucket_derived | derived | group
# A ``group`` dimension is satisfied by ANY of several interchangeable source
# fields (region may arrive as obligor / collateral / collateral_geography;
# channel as origination_channel / broker_channel). The first present is aliased
# onto the catalogue ``primary`` so MI queries resolve regardless of which arrived.
_DIM_SPEC: Dict[str, Dict[str, Any]] = {
    "ltv_bucket": {"kind": "bucket_ltv", "target": "current_loan_to_value"},
    "original_ltv_bucket": {"kind": "bucket_ltv", "target": "original_loan_to_value"},
    "interest_rate_bucket": {"kind": "bucket", "source": "current_interest_rate"},
    "ticket_bucket": {"kind": "bucket", "source": "current_outstanding_balance"},
    "age_bucket": {"kind": "bucket", "source": "youngest_borrower_age"},
    "time_on_book_bucket": {"kind": "bucket_derived", "source": "months_on_book"},
    "vintage_year": {"kind": "derived", "source": "vintage_year"},
    "borrower_type": {"kind": "derived", "source": "borrower_type"},
    "geographic_region_obligor": {
        "kind": "group", "primary": "geographic_region_obligor",
        "sources": ["geographic_region_obligor", "geographic_region_collateral",
                    "collateral_geography"]},
    "origination_channel": {
        "kind": "group", "primary": "origination_channel",
        "sources": ["origination_channel", "broker_channel"]},
}
CORE_FUNDED_DIMENSIONS = list(_DIM_SPEC.keys())

# LTV (target -> (balance/numerator field, valuation/denominator field)).
_LTV_INPUTS = {
    "current_loan_to_value": ("current_outstanding_balance", "current_valuation_amount"),
    "original_loan_to_value": ("original_principal_balance", "original_valuation_amount"),
}


def _to_num(s: pd.Series) -> pd.Series:
    # Shared deterministic parser (commas / £ / accounting negatives), so LTV
    # and valuation inputs that arrive as formatted strings parse correctly.
    return coerce_numeric(s)


def _numeric_mi_fields() -> List[str]:
    """Numeric MI columns to normalise up-front, sourced from config (no hard
    client values): every ``source_field`` in ``config/mi/buckets.yaml`` plus the
    LTV derivation inputs. Normalising these once means balance aggregation,
    ticket_bucket, LTV derivation and valuation parsing all see clean floats."""
    fields = set()
    try:
        from analytics_lib.buckets import load_bucket_config
        for spec in (load_bucket_config().get("buckets") or {}).values():
            sf = spec.get("source_field")
            if sf:
                fields.add(sf)
    except Exception:
        pass
    for tgt, (num, den) in _LTV_INPUTS.items():
        fields.update((tgt, num, den))
    return sorted(fields)


def _normalise_numeric_columns(out: pd.DataFrame) -> pd.DataFrame:
    """Coerce the configured numeric MI columns in place (only those present)."""
    for col in _numeric_mi_fields():
        if col in out.columns:
            out[col] = coerce_numeric(out[col])
    return out


def _to_ratio(s: pd.Series) -> pd.Series:
    """Normalise an LTV series to a 0..1 ratio (percent inputs ÷100, col-level)."""
    valid = s.dropna()
    if not valid.empty and float(valid.median()) > _PERCENT_MEDIAN:
        return s / 100.0
    return s


def _derive_ltv(out: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Prefer an explicit, valid source LTV; else derive balance/valuation.

    Mutates ``out[target]`` to a 0..1 ratio when resolvable. Returns the basis.
    """
    cols = set(out.columns)
    num_field, den_field = _LTV_INPUTS[target]
    basis: Dict[str, Any] = {"target": target}

    # 1. explicit source field present and valid (>0)?
    if target in cols:
        s = _to_num(out[target])
        if (s.notna() & (s > 0)).any():
            out[target] = _to_ratio(s)
            basis.update(method="source_field", source_fields=[target],
                         numerator=None, denominator=None, confidence=1.0,
                         valid_rows=int((s.notna() & (s > 0)).sum()))
            return basis

    # 2. derive numerator / denominator (divide-by-zero / non-numeric safe).
    if {num_field, den_field} <= cols:
        num = _to_num(out[num_field])
        den = _to_num(out[den_field])
        ratio = num / den.where(den > 0)
        if ratio.notna().any():
            out[target] = _to_ratio(ratio)
            basis.update(method="derived_ratio", source_fields=[num_field, den_field],
                         numerator=num_field, denominator=den_field, confidence=0.9,
                         valid_rows=int(ratio.notna().sum()))
            return basis

    # 3. inputs missing -> do NOT fabricate an LTV.
    have = [f for f in (target, num_field, den_field) if f in cols]
    basis.update(method="unavailable", reason="derivation_inputs_missing",
                 source_fields=have,
                 detail=f"need explicit {target}, or both {num_field} and {den_field}")
    return basis


# Borrower DOB fields (mapped from "Customer 1/2 DOB" etc.) used to derive the
# youngest-borrower age when no explicit age field is supplied.
_DOB_FIELDS = ("borrower_1_DOB", "borrower_2_DOB", "borrower_1_dob", "borrower_2_dob",
               "customer_1_dob", "customer_2_dob")
# Second-applicant fields whose presence means a JOINT borrower.
_SECOND_BORROWER_FIELDS = ("borrower_2_DOB", "borrower_2_dob", "customer_2_dob",
                           "borrower_2_gender", "customer_2_gender", "borrower_2_name",
                           "customer_2_name")
# Postcode fields preserved for region derivation / reason-coding.
_POSTCODE_FIELDS = ("property_post_code", "postcode", "post_code")


def _reporting_date(out: pd.DataFrame) -> Optional[pd.Series]:
    """The run reporting date series (reporting_date / cut-off), or None."""
    rep_col = next((c for c in ("reporting_date", "data_cut_off_date", "cut_off_date")
                    if c in out.columns), None)
    if not rep_col:
        return None
    rd = pd.to_datetime(out[rep_col], errors="coerce")
    return rd if rd.notna().any() else None


def _derive_vintage_and_seasoning(out: pd.DataFrame, derived: List[str]) -> None:
    """origination_date → vintage_year, months_on_book, seasoning band + segment.

    THE single derivation for the vintage/seasoning axis, shared by the funded
    tape prep and the platform-canonical read path. It previously lived inline in
    ``_derive_source_fields`` only, so the MI Agent — which reads the platform
    canonical — saw no vintage dimension at all and refused every vintage
    question on a book that carried an origination date for every loan.

    Seasoning is measured against the frame's own governed REPORTING / CUT-OFF
    date, not wall-clock "today", so a historical snapshot yields the seasoning
    the book had at that reporting date.
    """
    if "origination_date" not in out.columns:
        return
    od = coerce_dates(out["origination_date"])
    if not od.notna().any():
        return

    if "vintage_year" not in out.columns:
        out["vintage_year"] = od.dt.year.astype("Int64")
        derived.append("vintage_year")

    rd = _reporting_date(out)
    if rd is None:
        return
    mob = _seasoning.months_between(od, rd)
    if not mob.notna().any():
        return
    if "months_on_book" not in out.columns:
        out["months_on_book"] = mob.astype("Int64")
        derived.append("months_on_book")
    derived.extend(_seasoning.derive_seasoning(out, months=mob))


def _derive_youngest_age(out: pd.DataFrame, derived: List[str]) -> None:
    """Derive ``youngest_borrower_age`` from borrower DOBs as of the reporting date.

        borrower age        = completed years between DOB and the reporting date
        youngest borrower   = MINIMUM valid age across all borrower DOB fields
                              (the youngest borrower has the latest DOB)

    Applied ROW-WISE, filling only the rows that have no explicit age. That
    matters on a combined platform canonical: the frame carries every source
    portfolio at once, so a column-level "already populated?" test is answered by
    whichever book happens to supply an age, and every other book is then skipped.
    That is precisely why an acquired book delivering ``Borrower 1/2 DOB`` showed
    a null borrower age while the direct book — which supplies an explicit age —
    rendered fine.

    One derivation serves every funded portfolio. Nothing here reads
    ``source_portfolio_type``: a book gets an age because it supplied a DOB and a
    reporting date, not because of what kind of book it is. Rows with no valid
    DOB keep NULL — a missing date of birth is never age zero.
    """
    dob_cols = [c for c in _DOB_FIELDS if c in out.columns]
    rd = _reporting_date(out)
    if not dob_cols or rd is None:
        return

    ages = pd.DataFrame(index=out.index)
    for c in dob_cols:
        dob = coerce_dates(out[c])
        # Age is undefined for a DOB at/after the reporting date (a bad parse or
        # a placeholder); leave those rows null rather than emitting a negative.
        years = (rd - dob).dt.days / 365.25
        ages[c] = years.where(years >= 0)
    youngest = ages.min(axis=1, skipna=True)  # youngest borrower => minimum age
    if not youngest.notna().any():
        return
    # Completed years (floor) — 77.8 years of age is 77, not 78.
    computed = np.floor(youngest).astype("Int64")

    existing = (_to_num(out["youngest_borrower_age"]).astype("Int64")
                if "youngest_borrower_age" in out.columns
                else pd.Series(pd.NA, index=out.index, dtype="Int64"))
    filled = existing.where(existing.notna(), computed)
    if filled.notna().sum() > existing.notna().sum():
        out["youngest_borrower_age"] = filled
        if "youngest_borrower_age" not in derived:
            derived.append("youngest_borrower_age")


def _coalesce_group_dimensions(out: pd.DataFrame) -> List[str]:
    """Fill each group dimension's catalogue primary from its alternatives, ROW-WISE.

    A ``group`` dimension is satisfied by any of several interchangeable source
    fields — region may arrive as obligor / collateral / collateral_geography,
    channel as origination_channel / broker_channel. Which one a book supplies is
    a delivery detail, so the first populated alternative is coalesced onto the
    catalogue primary and MI queries ("by region") resolve either way.

    Row-wise, not column-wise. On a combined platform canonical the primary
    column exists as soon as ONE portfolio supplies it, and a column-level "is
    the primary already here?" test then skips every other book — which is why an
    acquired tape carrying a region under a different canonical name reported
    "region not supplied" while the direct book rendered.

    Returns the alias notes for the preparation report, so the coalescing is
    visible in provenance rather than silent.
    """
    notes: List[str] = []
    for spec in _DIM_SPEC.values():
        if spec.get("kind") != "group":
            continue
        primary = spec["primary"]
        sources = [s for s in spec["sources"] if s in out.columns]
        if not sources:
            continue
        filled = (out[primary].copy() if primary in out.columns
                  else pd.Series(pd.NA, index=out.index, dtype="object"))
        blank = _blank_mask(filled)
        # Per-dimension, so one dimension's coalescing never decides whether
        # another's result is written back.
        changed = False
        for alt in sources:
            if alt == primary or not blank.any():
                continue
            take = blank & ~_blank_mask(out[alt])
            if not take.any():
                continue
            filled = filled.mask(take, out[alt])
            blank = _blank_mask(filled)
            changed = True
            notes.append(f"{primary}<-{alt} ({int(take.sum())} rows)")
        if changed:
            out[primary] = filled
    return notes


def _apply_region_taxonomy(out: pd.DataFrame, client_id: Optional[str] = None
                           ) -> Dict[str, Any]:
    """Stamp the governed region detail / reporting columns onto a prepared frame.

    Harmonisation happens HERE, in the canonical preparation layer, so every
    channel — charts, MI queries, geography, exports — reads one already-resolved
    vocabulary. Nothing downstream renames, cases or consolidates a region, and
    the runtime never asks an LLM to classify one. A deployment with no taxonomy
    configured gets a no-op and behaves exactly as before.
    """
    try:
        taxonomy = _region.resolve_taxonomy(client_id or _client_hint(out))
        return _region.apply(out, taxonomy)
    except Exception as exc:  # noqa: BLE001 - harmonisation must never break prep
        logger.warning("region harmonisation skipped: %s", exc)
        return {}


def _client_hint(out: pd.DataFrame) -> Optional[str]:
    """The client this frame belongs to, for per-client taxonomy selection."""
    import os
    for col in ("client_id", "trakt_client_id"):
        if col in out.columns:
            values = [v for v in out[col].dropna().astype(str).unique() if v.strip()]
            if len(values) == 1:
                return values[0]
    return os.environ.get("MI_AGENT_CLIENT_ID") or None


def _blank_mask(series: pd.Series) -> pd.Series:
    """True where a categorical value is absent (null, empty or a null token)."""
    text = series.astype(str).str.strip().str.lower()
    return series.isna() | text.isin(["", "nan", "none", "nat", "<na>", "null"])


def _derive_borrower_type(out: pd.DataFrame, derived: List[str]) -> None:
    """Single vs joint borrower — JOINT iff ANY second-applicant field
    (borrower_2 DOB / gender / name) is populated, else SINGLE. A first-class
    categorical dimension so the MI Agent can run single-vs-joint cohort analysis
    and stratifications (e.g. LTV by borrower_type). NNEG exposure is joint-life,
    so the split matters for lifetime-mortgage risk. No second-applicant column
    present ⇒ not derivable (skipped)."""
    cols = [c for c in _SECOND_BORROWER_FIELDS if c in out.columns]
    if not cols:
        return
    present = pd.Series(False, index=out.index)
    for c in cols:
        present = present | ~_blank_mask(out[c])
    computed = pd.Series(np.where(present.to_numpy(), "joint", "single"),
                         index=out.index, dtype="object")

    # Row-wise, for the same reason as the age derivation: on a combined
    # platform canonical one book supplying borrower_type must not suppress the
    # derivation for every other book.
    existing = (out["borrower_type"] if "borrower_type" in out.columns
                else pd.Series(pd.NA, index=out.index, dtype="object"))
    gaps = _blank_mask(existing)
    if not gaps.any():
        return
    out["borrower_type"] = existing.mask(gaps, computed)
    if "borrower_type" not in derived:
        derived.append("borrower_type")


def _dedupe_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Collapse duplicate-named columns into one, coalescing values row-wise.

    A promoted tape can carry the same canonical name twice (e.g. a field both
    in the loan domain and promoted as MI enrichment). Selecting that name then
    returns a DataFrame, not a Series, which crashes numeric coercion. Keep the
    first occurrence's position and fill blanks from the later duplicates so no
    populated value is lost; record what was collapsed."""
    if not df.columns.duplicated().any():
        return df, []
    collapsed: List[Dict[str, Any]] = []
    new: Dict[str, pd.Series] = {}
    for name in list(dict.fromkeys(df.columns)):  # first-seen order, unique
        sub = df.loc[:, df.columns == name]
        if sub.shape[1] == 1:
            new[name] = sub.iloc[:, 0]
        else:
            # Treat blank/whitespace as missing, then take first non-null per row.
            filled = sub.replace(r"^\s*$", pd.NA, regex=True).bfill(axis=1).iloc[:, 0]
            new[name] = filled
            collapsed.append({"column": str(name), "occurrences": int(sub.shape[1])})
    return pd.DataFrame(new, index=df.index), collapsed


def augment_platform_canonical_dimensions(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Additively derive the MI dimensions the platform assembler does NOT emit —
    ``borrower_type`` (single vs joint) and ``youngest_borrower_age`` (NNEG) — onto
    an already-typed platform canonical, WITHOUT the full funded-tape prep (no LTV
    re-derivation, no dedup, no numeric coercion of existing columns).

    Purely additive and idempotent, and gap-filling ROW-WISE — the platform
    canonical is the combined tape for every source portfolio, so a field one
    book already supplies must not suppress the derivation for the others. Used
    by the MI API's platform canonical path so the MI Agent sees these dimensions
    without an onboarding re-run — the derivation happens at read time, not at
    onboarding time."""
    out = df.copy()
    derived: List[str] = []
    # origination_date → vintage_year / months_on_book / seasoning. The platform
    # assembler does not emit these, so without this the MI Agent refused every
    # vintage question on a book that carries an origination date for every loan.
    _derive_vintage_and_seasoning(out, derived)
    _derive_youngest_age(out, derived)   # DOBs → youngest_borrower_age (age_bucket)
    _derive_borrower_type(out, derived)  # second-applicant presence → single/joint
    # Region / channel arrive under different canonical names per book; coalesce
    # them onto the catalogue primary so a query for "region" resolves for EVERY
    # portfolio in the combined tape, not only the one that named it the primary.
    for note in _coalesce_group_dimensions(out):
        primary = note.split("<-", 1)[0]
        if primary not in derived:
            derived.append(primary)
    # Governed region harmonisation: one shared vocabulary across books, with the
    # source granularity retained alongside it. Deterministic and persisted — no
    # LLM is consulted on this path.
    if _apply_region_taxonomy(out).get("applied"):
        for f in (_region.FIELD_DETAIL, _region.FIELD_REPORTING):
            if f not in derived:
                derived.append(f)
    return out, derived


def _derive_source_fields(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    out = df.copy()
    out, dedup = _dedupe_columns(out)
    _normalise_numeric_columns(out)
    derived: List[str] = []
    basis: List[Dict[str, Any]] = []

    for target in ("current_loan_to_value", "original_loan_to_value"):
        b = _derive_ltv(out, target)
        basis.append(b)
        if b.get("method") in ("source_field", "derived_ratio") and target not in df.columns:
            derived.append(target)

    _derive_vintage_and_seasoning(out, derived)
    _derive_youngest_age(out, derived)
    _derive_borrower_type(out, derived)

    return out, derived, basis, dedup


@_perf.stage_fn("funded_prep")
def prepare_funded_mi_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return ``(analytics_ready_df, report)`` for a funded central lender tape."""
    out, derived, ltv_basis, dedup = _derive_source_fields(df)

    applied: Dict[str, Any] = {}
    issues: List[Dict[str, Any]] = []
    try:
        from analytics_lib.buckets import load_bucket_config, materialise_buckets
        out, issues, applied = materialise_buckets(
            out, load_bucket_config(), target="semantic_field")
    except Exception as exc:  # bucketing is additive; never block the dataset
        issues = [{"bucket": "*", "code": "engine_error", "severity": "error",
                   "detail": str(exc)}]

    # Group dimensions: alias the first present interchangeable source onto the
    # catalogue primary so MI queries (e.g. "by region") resolve regardless of
    # which region/channel field the source supplied (obligor vs collateral, etc.).
    group_aliases = _coalesce_group_dimensions(out)

    cols = set(out.columns)
    available: List[str] = []
    missing: List[Dict[str, Any]] = []
    basis_by_target = {b["target"]: b for b in ltv_basis}

    def _has_values(col: str) -> bool:
        # A dimension is only usable if its column exists AND has at least one
        # non-blank value (a bucket can exist but be all-NaN when values fall
        # outside the configured edges — that is NOT a usable stratification).
        return col in out.columns and out[col].notna().any() and (
            out[col].astype(str).str.strip() != "").any()

    for dim, spec in _DIM_SPEC.items():
        if _has_values(dim):
            available.append(dim)
            continue
        kind = spec["kind"]
        if dim in cols:
            # Column was produced but is entirely empty (e.g. LTV out of bucket
            # range / a scale mismatch) — report it honestly, not as available.
            detail = "column present but all rows blank after preparation"
            if kind == "bucket_ltv":
                b = basis_by_target.get(spec["target"], {})
                detail = (f"{spec['target']} bucketed to no value for any row "
                          f"(check scale/edges; basis={b.get('method')})")
            missing.append({"dimension": dim, "reason": "no_values_after_preparation",
                            "detail": detail})
        elif kind == "bucket_ltv":
            b = basis_by_target.get(spec["target"], {})
            missing.append({"dimension": dim, "reason": "derivation_inputs_missing",
                            "detail": b.get("detail", "LTV inputs unavailable")})
        elif kind in ("bucket", "bucket_derived", "derived"):
            src = spec["source"]
            reason = ("derivation_inputs_missing" if kind in ("bucket_derived", "derived")
                      else "not_in_central_tape")
            detail = (f"derivation source {src!r} not present"
                      if reason == "derivation_inputs_missing"
                      else f"source field {src!r} not in dataset")
            missing.append({"dimension": dim, "reason": reason, "detail": detail})
        else:  # group (region / channel) — none of the interchangeable sources present
            # Region: if only a postcode is available (no region/geography field
            # and no postcode->region map), say so exactly rather than "absent".
            if dim == "geographic_region_obligor" and any(_has_values(p) for p in _POSTCODE_FIELDS):
                pc = next(p for p in _POSTCODE_FIELDS if _has_values(p))
                missing.append({"dimension": dim, "reason": "postcode_available_region_not_derived",
                                "detail": f"{pc!r} present but no region field and no "
                                          "postcode->region mapping configured"})
            else:
                missing.append({"dimension": dim, "reason": "not_in_central_tape",
                                "detail": f"none of {spec['sources']} in dataset"})

    report = {
        "preparation_applied": True,
        "derived_fields": derived,
        "ltv_derivation_basis": ltv_basis,
        "buckets_applied": {k: v for k, v in applied.items() if v},
        "group_aliases": group_aliases,
        "duplicate_columns_collapsed": dedup,
        "dimensions_available": sorted(available),
        "missing_dimensions": missing,
        "bucket_errors": [i for i in (issues or []) if i.get("severity") == "error"][:20],
    }
    return out, report


def missing_dimension_names(report: Dict[str, Any]) -> List[str]:
    """Convenience: the dimension names from a report's ``missing_dimensions``."""
    return [m["dimension"] if isinstance(m, dict) else m
            for m in report.get("missing_dimensions", [])]
