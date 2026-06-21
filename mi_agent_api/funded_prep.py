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

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analytics_lib.numeric import coerce_numeric

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


def _derive_source_fields(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[Dict[str, Any]]]:
    out = df.copy()
    _normalise_numeric_columns(out)
    derived: List[str] = []
    basis: List[Dict[str, Any]] = []

    for target in ("current_loan_to_value", "original_loan_to_value"):
        b = _derive_ltv(out, target)
        basis.append(b)
        if b.get("method") in ("source_field", "derived_ratio") and target not in df.columns:
            derived.append(target)

    if "origination_date" in out.columns:
        od = pd.to_datetime(out["origination_date"], errors="coerce", dayfirst=True)
        if od.notna().any():
            if "vintage_year" not in out.columns:
                out["vintage_year"] = od.dt.year.astype("Int64")
                derived.append("vintage_year")
            rep_col = next((c for c in ("reporting_date", "data_cut_off_date", "cut_off_date")
                            if c in out.columns), None)
            if rep_col and "months_on_book" not in out.columns:
                rd = pd.to_datetime(out[rep_col], errors="coerce")
                mob = (rd.dt.year - od.dt.year) * 12 + (rd.dt.month - od.dt.month)
                if mob.notna().any():
                    out["months_on_book"] = mob.astype("Int64")
                    derived.append("months_on_book")

    return out, derived, basis


def prepare_funded_mi_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return ``(analytics_ready_df, report)`` for a funded central lender tape."""
    out, derived, ltv_basis = _derive_source_fields(df)

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
    group_aliases: List[str] = []
    for dim, spec in _DIM_SPEC.items():
        if spec.get("kind") != "group" or spec["primary"] in out.columns:
            continue
        alt = next((s for s in spec["sources"] if s in out.columns), None)
        if alt:
            out[spec["primary"]] = out[alt]
            group_aliases.append(f"{spec['primary']}<-{alt}")

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
            missing.append({"dimension": dim, "reason": "not_in_central_tape",
                            "detail": f"none of {spec['sources']} in dataset"})

    report = {
        "preparation_applied": True,
        "derived_fields": derived,
        "ltv_derivation_basis": ltv_basis,
        "buckets_applied": {k: v for k, v in applied.items() if v},
        "group_aliases": group_aliases,
        "dimensions_available": sorted(available),
        "missing_dimensions": missing,
        "bucket_errors": [i for i in (issues or []) if i.get("severity") == "error"][:20],
    }
    return out, report


def missing_dimension_names(report: Dict[str, Any]) -> List[str]:
    """Convenience: the dimension names from a report's ``missing_dimensions``."""
    return [m["dimension"] if isinstance(m, dict) else m
            for m in report.get("missing_dimensions", [])]
