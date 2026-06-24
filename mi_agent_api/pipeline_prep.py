"""Pipeline MI data preparation — governed pipeline source -> analytics-ready.

The pre-funded form of the SAME lending asset as the funded book. This is the
pipeline analogue of ``funded_prep.prepare_funded_mi_dataset``: it turns a raw
M2L KFI / pipeline extract (or the onboarding ``18a_central_pipeline_tape.csv``)
into an analytics-ready **prepared pipeline MI dataset** by

  * resolving source columns to canonical fields via the governed pipeline field
    contract (``config/mi/pipeline_field_contract.yaml``) — economic fields reuse
    the funded MI canonical names so funded + pipeline correlate for forecasting;
  * normalising funnel stage / status against the existing stage vocabulary
    (``mi_agent.states.models``);
  * parsing amounts / valuations / rates / LTVs / ages with the SAME deterministic
    numeric parser the funded dataset uses (``analytics_lib.numeric``);
  * deriving LTV, youngest-borrower age, expected completion date, completion
    probability (from the EXISTING ``config/client/pipeline_expected_funding.yaml``
    stage assumptions — never invented), expected / weighted expected funded
    amount, and case-age / days-to-completion;
  * materialising buckets with the EXISTING bucket engine
    (``analytics_lib.buckets`` over ``config/mi/buckets.yaml``) plus the
    pipeline-derived ``expected_completion_month`` / ``pipeline_stage_bucket``.

Separation guarantee: this NEVER reads or writes the funded central lender tape.
Every prepared row carries ``record_type == "pipeline"`` and the output is a
separate artefact (``20_prepared_pipeline_mi.csv``).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from analytics_lib.numeric import coerce_numeric

from mi_agent.states import models as _states

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = _REPO_ROOT / "config" / "mi" / "pipeline_field_contract.yaml"
_FORECAST_CONFIG_PATH = _REPO_ROOT / "config" / "client" / "pipeline_expected_funding.yaml"

RECORD_TYPE = "pipeline"
_PERCENT_MEDIAN = 1.5

# Funnel stage normalisation -> canonical stage token (config stages are these).
_STAGE_CANON = {
    "kfi": "KFI", "kfi issued": "KFI", "illustration": "KFI", "quote": "KFI",
    "application": "APPLICATION", "applied": "APPLICATION", "app": "APPLICATION",
    "offer": "OFFER", "offered": "OFFER", "offer issued": "OFFER",
    "completed": "COMPLETED", "complete": "COMPLETED", "completion": "COMPLETED",
    "funds released": "COMPLETED", "funded": "COMPLETED", "drawn": "COMPLETED",
    "drawdown": "COMPLETED", "live": "COMPLETED",
    "withdrawn": "WITHDRAWN", "declined": "WITHDRAWN", "rejected": "WITHDRAWN",
    "cancelled": "WITHDRAWN", "lapsed": "WITHDRAWN", "abandoned": "WITHDRAWN",
}
# Coarse funnel grouping for pipeline_stage_bucket.
_STAGE_BUCKET = {
    "KFI": "early", "APPLICATION": "mid", "OFFER": "late",
    "COMPLETED": "completed", "WITHDRAWN": "withdrawn",
}

# Pipeline status (reuses funded-status semantics): funded / pipeline / withdrawn.
_OPEN_PIPELINE_STAGES = {"KFI", "APPLICATION", "OFFER"}

_DOB_ALIASES_DEFAULT = ["dob app 1", "dob app 2", "date of birth"]


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_pipeline_contract() -> Dict[str, Any]:
    """The governed pipeline field contract (cached)."""
    if not _CONTRACT_PATH.exists():
        return {}
    return yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def _forecast_config() -> Dict[str, Any]:
    if not _FORECAST_CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(_FORECAST_CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _stage_probabilities() -> Dict[str, float]:
    """``{stage_upper: probability}`` from the existing forecast config (never invented)."""
    raw = (_forecast_config().get("stage_probabilities") or {})
    out: Dict[str, float] = {}
    for stage, prob in raw.items():
        try:
            out[str(stage).strip().upper()] = float(prob)
        except (TypeError, ValueError):
            continue
    return out


def _stage_days_to_fund() -> Dict[str, int]:
    raw = (_forecast_config().get("stage_days_to_fund") or {})
    out: Dict[str, int] = {}
    for stage, days in raw.items():
        try:
            out[str(stage).strip().upper()] = int(days)
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
# Column resolution (source headers -> canonical fields via contract aliases)
# --------------------------------------------------------------------------- #
def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).strip().lower()).strip()


def _field_specs() -> Dict[str, Dict[str, Any]]:
    """Flatten the contract's funded-correlated + pipeline-specific field specs."""
    contract = load_pipeline_contract()
    specs: Dict[str, Dict[str, Any]] = {}
    specs.update(contract.get("funded_correlated_fields", {}) or {})
    specs.update(contract.get("pipeline_specific_fields", {}) or {})
    return specs


def resolve_source_columns(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    """Map ``{canonical_field: source_column}`` using the contract aliases.

    A canonical field already present by name is kept as-is. Otherwise the first
    source column whose normalised header matches an alias wins. Returns the
    mapping and the list of canonical fields that had no source.
    """
    specs = _field_specs()
    norm_to_col = {_norm(c): c for c in df.columns}
    mapping: Dict[str, str] = {}
    unmatched: List[str] = []
    for canonical, spec in specs.items():
        if canonical in df.columns:
            mapping[canonical] = canonical
            continue
        aliases = spec.get("source_aliases", []) or []
        found = next((norm_to_col[_norm(a)] for a in aliases if _norm(a) in norm_to_col), None)
        if found is not None:
            mapping[canonical] = found
        else:
            unmatched.append(canonical)
    return mapping, unmatched


def _dob_columns(df: pd.DataFrame) -> List[str]:
    contract = load_pipeline_contract()
    aliases = ((contract.get("funded_correlated_fields", {}) or {})
               .get("youngest_borrower_age", {}) or {}).get(
        "dob_source_aliases", _DOB_ALIASES_DEFAULT)
    norm_to_col = {_norm(c): c for c in df.columns}
    return [norm_to_col[_norm(a)] for a in aliases if _norm(a) in norm_to_col]


# --------------------------------------------------------------------------- #
# Numeric / date / ratio helpers (shared with funded prep semantics)
# --------------------------------------------------------------------------- #
def _to_ratio(s: pd.Series) -> pd.Series:
    valid = s.dropna()
    if not valid.empty and float(valid.median()) > _PERCENT_MEDIAN:
        return s / 100.0
    return s


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=False)


def _as_of_date(df: pd.DataFrame, explicit: Optional[str],
                source_file: Optional[str]) -> Optional[pd.Timestamp]:
    """The operational as-of date for the pipeline snapshot: the explicit value
    (the selected weekly extract date) if given, else parsed from the source file
    name. This is the pipeline's own operational date — NOT the funded reporting
    cut-off."""
    if explicit:
        ts = pd.to_datetime(explicit, errors="coerce")
        if pd.notna(ts):
            return ts
    if source_file:
        m = re.search(r"(\d{4})[_\-.](\d{2})[_\-.](\d{2})", str(source_file))
        if m:
            ts = pd.to_datetime("-".join(m.groups()), errors="coerce")
            if pd.notna(ts):
                return ts
        m = re.search(r"(\d{4})[_\-.](\d{2})", str(source_file))
        if m:
            ts = pd.to_datetime(f"{m.group(1)}-{m.group(2)}-01", errors="coerce")
            if pd.notna(ts):
                return ts
    return None


# --------------------------------------------------------------------------- #
# Core preparation
# --------------------------------------------------------------------------- #
def prepare_pipeline_mi_dataset(
    df: pd.DataFrame,
    *,
    as_of_date: Optional[str] = None,
    source_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return ``(prepared_pipeline_df, report)`` for a raw pipeline extract.

    ``as_of_date`` is the operational as-of date of this weekly pipeline extract
    (e.g. the selected file's date). It is deliberately distinct from the funded
    reporting cut-off and is used for case-age / days-to-completion derivations.
    """
    mapping, unmatched = resolve_source_columns(df)
    out = pd.DataFrame(index=df.index)
    out["record_type"] = RECORD_TYPE
    derived: List[str] = []

    # 1. Lift mapped source columns onto their canonical names.
    for canonical, src in mapping.items():
        out[canonical] = df[src].values

    # 2. Numeric economic fields.
    for fld in ("current_outstanding_balance", "current_valuation_amount",
                "current_interest_rate", "current_loan_to_value"):
        if fld in out.columns:
            out[fld] = coerce_numeric(out[fld])

    # 3. LTV: prefer explicit, else derive balance / valuation (funded prep rule).
    ltv_basis = _derive_ltv(out, derived)

    # 4. Youngest borrower age from DOBs, as of the operational as-of date.
    rep_ts = _as_of_date(df, as_of_date, source_file)
    _derive_youngest_age(df, out, rep_ts, derived)

    # 5. Normalise funnel stage + status.
    _normalise_stage(out, derived)

    # 6. Timing fields (dates) + derived expected completion.
    days_to_fund = _stage_days_to_fund()
    _parse_pipeline_dates(out, derived)
    _derive_expected_completion(out, rep_ts, days_to_fund, derived)

    # 7. Completion probability + expected / weighted funded amount.
    stage_probs = _stage_probabilities()
    _derive_probabilities_and_amounts(out, stage_probs, derived)

    # 8. Case-age / days-to-completion.
    _derive_durations(out, rep_ts, derived)

    # 9. Region / channel group aliases (mirror funded prep) + provenance.
    group_aliases = _apply_group_aliases(out)
    out["pipeline_source_file"] = source_file or ""
    if rep_ts is not None:
        out["pipeline_as_of_date"] = rep_ts.date().isoformat()
    if "pipeline_as_of_date" not in derived and "pipeline_as_of_date" in out.columns:
        derived.append("pipeline_as_of_date")

    # 10. Buckets via the existing engine + pipeline-derived dimensions.
    bucket_issues = _materialise_buckets(out)
    _derive_pipeline_buckets(out, derived)

    report = _build_report(out, mapping, unmatched, derived, ltv_basis,
                           group_aliases, bucket_issues, rep_ts)
    return out, report


def _derive_ltv(out: pd.DataFrame, derived: List[str]) -> Dict[str, Any]:
    basis: Dict[str, Any] = {"target": "current_loan_to_value"}
    if "current_loan_to_value" in out.columns:
        s = coerce_numeric(out["current_loan_to_value"])
        if (s.notna() & (s > 0)).any():
            out["current_loan_to_value"] = _to_ratio(s)
            basis.update(method="source_field", source_fields=["current_loan_to_value"],
                         confidence=1.0)
            return basis
    if {"current_outstanding_balance", "current_valuation_amount"} <= set(out.columns):
        num = coerce_numeric(out["current_outstanding_balance"])
        den = coerce_numeric(out["current_valuation_amount"])
        ratio = num / den.where(den > 0)
        if ratio.notna().any():
            out["current_loan_to_value"] = _to_ratio(ratio)
            if "current_loan_to_value" not in derived:
                derived.append("current_loan_to_value")
            basis.update(method="derived_ratio",
                         source_fields=["current_outstanding_balance", "current_valuation_amount"],
                         confidence=0.9)
            return basis
    basis.update(method="unavailable", reason="derivation_inputs_missing",
                 detail="need explicit current_loan_to_value, or both loan amount and valuation")
    return basis


def _derive_youngest_age(src: pd.DataFrame, out: pd.DataFrame,
                         rep_ts: Optional[pd.Timestamp], derived: List[str]) -> None:
    if "youngest_borrower_age" in out.columns and coerce_numeric(out["youngest_borrower_age"]).notna().any():
        out["youngest_borrower_age"] = coerce_numeric(out["youngest_borrower_age"])
        return
    dob_cols = _dob_columns(src)
    if not dob_cols or rep_ts is None:
        return
    ages = pd.DataFrame(index=src.index)
    for c in dob_cols:
        dob = pd.to_datetime(src[c], errors="coerce", dayfirst=False)
        ages[c] = (rep_ts - dob).dt.days / 365.25
    youngest = ages.min(axis=1, skipna=True)  # youngest borrower = minimum age
    if youngest.notna().any():
        out["youngest_borrower_age"] = np.floor(youngest).astype("Int64")
        if "youngest_borrower_age" not in derived:
            derived.append("youngest_borrower_age")


def _normalise_stage(out: pd.DataFrame, derived: List[str]) -> None:
    if "pipeline_stage" not in out.columns:
        return
    raw = out["pipeline_stage"].astype(str).map(_norm)
    canon = raw.map(lambda v: _STAGE_CANON.get(v, "UNKNOWN" if v else "UNKNOWN"))
    out["pipeline_stage"] = canon
    # Pipeline status reuses funded-status semantics.
    def _status(stage: str) -> str:
        if stage == "COMPLETED":
            return "funded"
        if stage == "WITHDRAWN":
            return "withdrawn"
        if stage in _OPEN_PIPELINE_STAGES:
            return "pipeline"
        return "unknown"
    out["pipeline_status"] = canon.map(_status)
    if "pipeline_status" not in derived:
        derived.append("pipeline_status")


def _parse_pipeline_dates(out: pd.DataFrame, derived: List[str]) -> None:
    for fld in ("kfi_date", "application_date", "offer_date", "expected_completion_date"):
        if fld in out.columns:
            out[fld] = _parse_date(out[fld])
    # pipeline_stage_date = latest available stage date.
    date_cols = [c for c in ("kfi_date", "application_date", "offer_date") if c in out.columns]
    if date_cols:
        out["pipeline_stage_date"] = out[date_cols].max(axis=1)
        if "pipeline_stage_date" not in derived:
            derived.append("pipeline_stage_date")


def _derive_expected_completion(out: pd.DataFrame, rep_ts: Optional[pd.Timestamp],
                                days_to_fund: Dict[str, int], derived: List[str]) -> None:
    have_explicit = ("expected_completion_date" in out.columns
                     and out["expected_completion_date"].notna().any())
    if "expected_completion_date" not in out.columns:
        out["expected_completion_date"] = pd.NaT
    if not days_to_fund or "pipeline_stage" not in out.columns:
        return
    base = rep_ts if rep_ts is not None else out.get("pipeline_stage_date")
    if base is None:
        return
    filled = 0
    for idx in out.index:
        if pd.notna(out.at[idx, "expected_completion_date"]):
            continue
        stage = str(out.at[idx, "pipeline_stage"])
        if stage not in days_to_fund:
            continue
        base_ts = base if isinstance(base, pd.Timestamp) else base.get(idx)
        if base_ts is None or pd.isna(base_ts):
            continue
        out.at[idx, "expected_completion_date"] = base_ts + pd.Timedelta(days=days_to_fund[stage])
        filled += 1
    if filled and not have_explicit and "expected_completion_date" not in derived:
        derived.append("expected_completion_date")


def _derive_probabilities_and_amounts(out: pd.DataFrame, stage_probs: Dict[str, float],
                                      derived: List[str]) -> None:
    # Completion probability: row-level where present, else config stage lookup.
    if "completion_probability" in out.columns:
        prob = coerce_numeric(out["completion_probability"])
    else:
        prob = pd.Series(np.nan, index=out.index)
    if stage_probs and "pipeline_stage" in out.columns:
        stage_prob = out["pipeline_stage"].map(stage_probs)
        prob = prob.where(prob.notna(), stage_prob)
        out["stage_conversion_probability"] = out["pipeline_stage"].map(stage_probs)
        if "stage_conversion_probability" not in derived:
            derived.append("stage_conversion_probability")
    out["completion_probability"] = prob
    if "completion_probability" not in derived:
        derived.append("completion_probability")

    # Expected funded amount == the economic amount; weighted by probability.
    if "current_outstanding_balance" in out.columns:
        amount = coerce_numeric(out["current_outstanding_balance"])
        out["expected_funded_amount"] = amount
        out["weighted_expected_funded_amount"] = amount * prob
        for f in ("expected_funded_amount", "weighted_expected_funded_amount"):
            if f not in derived:
                derived.append(f)


def _derive_durations(out: pd.DataFrame, rep_ts: Optional[pd.Timestamp],
                      derived: List[str]) -> None:
    if rep_ts is None:
        return
    start_cols = [c for c in ("application_date", "kfi_date") if c in out.columns]
    if start_cols:
        start = out[start_cols].min(axis=1)
        out["pipeline_case_age_days"] = (rep_ts - start).dt.days
        if "pipeline_case_age_days" not in derived:
            derived.append("pipeline_case_age_days")
    if "expected_completion_date" in out.columns:
        out["days_to_expected_completion"] = (out["expected_completion_date"] - rep_ts).dt.days
        if "days_to_expected_completion" not in derived:
            derived.append("days_to_expected_completion")


def _apply_group_aliases(out: pd.DataFrame) -> List[str]:
    aliases: List[str] = []
    if "collateral_geography" in out.columns and "geographic_region_obligor" not in out.columns:
        out["geographic_region_obligor"] = out["collateral_geography"]
        aliases.append("geographic_region_obligor<-collateral_geography")
    if "broker_channel" in out.columns and "origination_channel" not in out.columns:
        out["origination_channel"] = out["broker_channel"]
        aliases.append("origination_channel<-broker_channel")
    return aliases


def _materialise_buckets(out: pd.DataFrame) -> List[Dict[str, Any]]:
    try:
        from analytics_lib.buckets import load_bucket_config, materialise_buckets
        # Same engine + config the funded dataset uses; ``target="semantic_field"``
        # writes ltv_bucket / age_bucket / ticket_bucket / interest_rate_bucket
        # (buckets whose source column is absent simply produce no column).
        prepared, issues, _applied = materialise_buckets(
            out, load_bucket_config(), target="semantic_field")
        for col in prepared.columns:
            if col not in out.columns:
                out[col] = prepared[col].values
        return [i for i in (issues or []) if i.get("severity") == "error"][:20]
    except Exception as exc:  # bucketing is additive; never block the dataset
        return [{"bucket": "*", "code": "engine_error", "severity": "error", "detail": str(exc)}]


def _derive_pipeline_buckets(out: pd.DataFrame, derived: List[str]) -> None:
    if "expected_completion_date" in out.columns and out["expected_completion_date"].notna().any():
        out["expected_completion_month"] = (
            out["expected_completion_date"].dt.strftime("%Y-%m"))
        if "expected_completion_month" not in derived:
            derived.append("expected_completion_month")
    if "pipeline_stage" in out.columns:
        out["pipeline_stage_bucket"] = out["pipeline_stage"].map(
            lambda s: _STAGE_BUCKET.get(str(s), "unknown"))
        if "pipeline_stage_bucket" not in derived:
            derived.append("pipeline_stage_bucket")


# --------------------------------------------------------------------------- #
# Report (metadata + data-quality diagnostics)
# --------------------------------------------------------------------------- #
_DIMENSION_FIELDS = [
    "pipeline_stage", "pipeline_status", "pipeline_stage_bucket",
    "geographic_region_obligor", "collateral_geography", "origination_channel",
    "broker_channel", "product_type", "ltv_bucket", "age_bucket", "ticket_bucket",
    "interest_rate_bucket", "expected_completion_month",
]
_METRIC_FIELDS = [
    "current_outstanding_balance", "expected_funded_amount",
    "weighted_expected_funded_amount", "completion_probability",
    "current_valuation_amount", "current_loan_to_value", "current_interest_rate",
    "youngest_borrower_age", "pipeline_case_age_days", "days_to_expected_completion",
]


def _has_values(out: pd.DataFrame, col: str) -> bool:
    return col in out.columns and out[col].notna().any() and (
        out[col].astype(str).str.strip() != "").any()


def _numeric_coverage(out: pd.DataFrame, col: str) -> Tuple[int, int]:
    if col not in out.columns:
        return 0, len(out)
    parsed = coerce_numeric(out[col]).notna().sum()
    return int(parsed), int(len(out))


def field_correlation_to_funded(out: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """``{pipeline_field: {funded_correlation, available}}`` from the contract."""
    contract = load_pipeline_contract()
    fc = contract.get("funded_correlated_fields", {}) or {}
    result: Dict[str, Any] = {}
    for fld, spec in fc.items():
        result[fld] = {
            "funded_correlation": spec.get("funded_correlation", []),
            "available": bool(out is not None and _has_values(out, fld)),
        }
    return result


def forecast_readiness(out: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """The contract's forecast-readiness block, annotated with availability."""
    contract = load_pipeline_contract()
    fr = dict(contract.get("forecast_readiness", {}) or {})
    if out is not None:
        required = {
            "expected_amount": fr.get("economic_amount_field"),
            "completion_probability": fr.get("baseline_completion_probability_field"),
            "expected_completion_date": fr.get("expected_completion_date_field"),
            "expected_completion_month": fr.get("expected_completion_month_field"),
        }
        fr["fields_available"] = {
            k: bool(v and _has_values(out, v)) for k, v in required.items()
        }
        corr = fr.get("correlation_fields", {}) or {}
        fr["correlation_fields_available"] = {
            k: bool(_has_values(out, v)) for k, v in corr.items()
        }
        fr["forecast_ready"] = all(fr["fields_available"].values())
    return fr


def _build_report(out: pd.DataFrame, mapping: Dict[str, str], unmatched: List[str],
                  derived: List[str], ltv_basis: Dict[str, Any],
                  group_aliases: List[str], bucket_issues: List[Dict[str, Any]],
                  rep_ts: Optional[pd.Timestamp]) -> Dict[str, Any]:
    n = int(len(out))
    dims_available = sorted({d for d in _DIMENSION_FIELDS if _has_values(out, d)})
    metrics_available = sorted(
        {m for m in _METRIC_FIELDS if _numeric_coverage(out, m)[0] > 0})

    missing: List[Dict[str, str]] = []
    for d in _DIMENSION_FIELDS:
        if d in out.columns and not _has_values(out, d):
            missing.append({"dimension": d, "reason": "no_values_after_preparation",
                            "detail": f"{d!r} present but blank for all rows"})

    # Stage breakdown + economic totals.
    stage_counts: Dict[str, int] = {}
    if "pipeline_stage" in out.columns:
        stage_counts = {str(k): int(v) for k, v in
                        out["pipeline_stage"].value_counts(dropna=False).items()}
    total_amount = (float(coerce_numeric(out["current_outstanding_balance"]).sum())
                    if "current_outstanding_balance" in out.columns else 0.0)
    expected_funded = (float(coerce_numeric(out["expected_funded_amount"]).sum())
                       if "expected_funded_amount" in out.columns else 0.0)
    weighted_expected = (float(coerce_numeric(out["weighted_expected_funded_amount"]).sum())
                         if "weighted_expected_funded_amount" in out.columns else None)

    return {
        "preparation_applied": True,
        "record_type": RECORD_TYPE,
        "row_count": n,
        "source_columns_mapped": mapping,
        "unmatched_canonical_fields": unmatched,
        "derived_fields": derived,
        "ltv_derivation_basis": ltv_basis,
        "group_aliases": group_aliases,
        "pipeline_as_of_date": rep_ts.date().isoformat() if rep_ts is not None else None,
        "dimensions_available": dims_available,
        "metrics_available": metrics_available,
        "missing_dimensions": missing,
        "stage_counts": stage_counts,
        "total_pipeline_amount": round(total_amount, 2),
        "expected_funded_amount": round(expected_funded, 2),
        "weighted_expected_funded_amount": (round(weighted_expected, 2)
                                            if weighted_expected is not None else None),
        "bucket_errors": bucket_issues,
        "data_quality": validate_pipeline_dataset(out),
        "field_correlation_to_funded": field_correlation_to_funded(out),
        "forecast_readiness": forecast_readiness(out),
    }


# --------------------------------------------------------------------------- #
# Part 5 — data-aware validation / diagnostics (blocker / warning / info)
# --------------------------------------------------------------------------- #
def _diag(check: str, severity: str, detail: str, **extra: Any) -> Dict[str, Any]:
    d = {"check": check, "severity": severity, "detail": detail}
    d.update(extra)
    return d


def validate_pipeline_dataset(out: pd.DataFrame) -> List[Dict[str, Any]]:
    """Diagnostics partitioned into ``blocker`` / ``warning`` / ``info``."""
    n = int(len(out))
    diags: List[Dict[str, Any]] = []
    if n == 0:
        return [_diag("empty_dataset", "blocker", "pipeline dataset has no rows")]

    # Required identifier availability.
    if not _has_values(out, "pipeline_case_identifier"):
        diags.append(_diag("missing_case_identifier", "blocker",
                           "pipeline_case_identifier absent or blank for all rows"))
    else:
        ids = out["pipeline_case_identifier"].astype(str).str.strip()
        dupes = int(ids[ids != ""].duplicated().sum())
        if dupes:
            diags.append(_diag("duplicate_case_identifiers", "warning",
                               f"{dupes} duplicate pipeline_case_identifier value(s)",
                               count=dupes))

    # Stage / status availability.
    if not _has_values(out, "pipeline_stage"):
        diags.append(_diag("missing_stage", "blocker",
                           "pipeline_stage absent or blank for all rows"))
    else:
        unknown = int((out["pipeline_stage"].astype(str) == "UNKNOWN").sum())
        if unknown:
            diags.append(_diag("unrecognised_stage_values", "warning",
                               f"{unknown} row(s) have an unrecognised stage",
                               count=unknown))

    # Economic amount parse coverage.
    parsed, total = _numeric_coverage(out, "current_outstanding_balance")
    if parsed == 0:
        diags.append(_diag("missing_economic_amount", "blocker",
                           "no parseable pipeline economic amount (current_outstanding_balance)"))
    elif parsed < total:
        diags.append(_diag("amount_parse_partial", "warning",
                           f"economic amount parsed for {parsed}/{total} rows",
                           parsed=parsed, total=total))

    # Optional coverage checks (informational when partial, never blocking).
    for fld, label in (("expected_completion_date", "expected completion date"),
                       ("current_loan_to_value", "LTV"),
                       ("broker_channel", "broker/channel"),
                       ("geographic_region_obligor", "region")):
        if fld in out.columns:
            present = int(out[fld].notna().sum()) if out[fld].dtype.kind in "Mf" \
                else int((out[fld].astype(str).str.strip() != "").sum())
            if present == 0:
                diags.append(_diag(f"missing_{fld}", "info",
                                   f"{label} present as a column but empty for all rows"))
            elif present < n:
                diags.append(_diag(f"{fld}_partial", "info",
                                   f"{label} populated for {present}/{n} rows",
                                   present=present, total=n))
        else:
            diags.append(_diag(f"absent_{fld}", "info",
                               f"{label} not present in this pipeline source"))

    # Completion probability coverage (forecast input).
    if "completion_probability" in out.columns:
        prob_cov, _ = _numeric_coverage(out, "completion_probability")
        if prob_cov == 0:
            diags.append(_diag("missing_completion_probability", "warning",
                               "no completion probability available (no row value and no "
                               "config stage probability) — forecast not yet weightable"))
    return diags


def diagnostics_by_severity(report: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Group a report's ``data_quality`` diagnostics by severity."""
    out: Dict[str, List[Dict[str, Any]]] = {"blocker": [], "warning": [], "info": []}
    for d in report.get("data_quality", []) or []:
        out.setdefault(d.get("severity", "info"), []).append(d)
    return out
