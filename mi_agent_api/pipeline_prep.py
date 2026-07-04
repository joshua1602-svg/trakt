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


#: Second-applicant source columns whose presence means a JOINT borrower.
_SECOND_BORROWER_ALIASES_DEFAULT = ["dob app 2", "gender app 2", "date of birth 2",
                                    "borrower 2 dob", "customer 2 dob"]


def _second_borrower_columns(df: pd.DataFrame) -> List[str]:
    contract = load_pipeline_contract()
    aliases = ((contract.get("funded_correlated_fields", {}) or {})
               .get("borrower_type", {}) or {}).get(
        "second_borrower_source_aliases", _SECOND_BORROWER_ALIASES_DEFAULT)
    norm_to_col = {_norm(c): c for c in df.columns}
    return [norm_to_col[_norm(a)] for a in aliases if _norm(a) in norm_to_col]


def _derive_borrower_type(src: pd.DataFrame, out: pd.DataFrame, derived: List[str]) -> None:
    """Single vs joint borrower — JOINT iff ANY second-applicant field (DOB App 2 /
    Gender App 2) is populated, else SINGLE. A first-class categorical dimension so
    the MI Agent can run single-vs-joint cohort analysis and stratifications
    (e.g. LTV by borrower_type). No second-applicant column ⇒ not derivable."""
    cols = _second_borrower_columns(src)
    if not cols:
        return
    present = pd.Series(False, index=src.index)
    for c in cols:
        v = src[c].astype(str).str.strip().str.lower()
        present = present | (src[c].notna() & ~v.isin(["", "nan", "none"]))
    out["borrower_type"] = np.where(present.to_numpy(), "joint", "single")
    if "borrower_type" not in derived:
        derived.append("borrower_type")


# --------------------------------------------------------------------------- #
# Numeric / date / ratio helpers (shared with funded prep semantics)
# --------------------------------------------------------------------------- #
def _to_ratio(s: pd.Series) -> pd.Series:
    valid = s.dropna()
    if not valid.empty and float(valid.median()) > _PERCENT_MEDIAN:
        return s / 100.0
    return s


def _parse_date(series: pd.Series) -> pd.Series:
    # UK day-first dates (dd/mm/yyyy) — matches funded_prep. dayfirst=False silently
    # dropped every DOB / date whose day > 12 to NaT and month/day-swapped the rest,
    # which broke youngest-borrower age (NNEG) and all pipeline timing.
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


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
    historical_model: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return ``(prepared_pipeline_df, report)`` for a raw pipeline extract.

    ``as_of_date`` is the operational as-of date of this weekly pipeline extract
    (e.g. the selected file's date). It is deliberately distinct from the funded
    reporting cut-off and is used for case-age / days-to-completion derivations.

    ``historical_model`` (from ``pipeline_history``) supplies empirically-observed
    stage completion rates; when a stage has sufficient history its rate is used
    in preference to the configured stage probability.
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

    # 4b. Borrower type (single vs joint) from second-applicant field presence.
    _derive_borrower_type(df, out, derived)

    # 5. Normalise funnel stage + status.
    _normalise_stage(out, derived)

    # 6. Timing fields (dates) + derived expected completion.
    days_to_fund = _stage_days_to_fund()
    _parse_pipeline_dates(out, derived)
    _derive_expected_completion(out, rep_ts, days_to_fund, derived)

    # 7. Completion probability (governed hierarchy) + weighted funded amount.
    stage_probs = _stage_probabilities()
    historical_rates = dict((historical_model or {}).get("stage_rates", {}) or {})
    prob_basis = _derive_probabilities_and_amounts(
        out, stage_probs, historical_rates, derived)

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
                           group_aliases, bucket_issues, rep_ts,
                           prob_basis, historical_model)
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
        dob = pd.to_datetime(src[c], errors="coerce", dayfirst=True)  # UK dd/mm/yyyy DOBs
        ages[c] = (rep_ts - dob).dt.days / 365.25
    youngest = ages.min(axis=1, skipna=True)  # youngest borrower = minimum age
    if youngest.notna().any():
        out["youngest_borrower_age"] = np.floor(youngest).astype("Int64")
        if "youngest_borrower_age" not in derived:
            derived.append("youngest_borrower_age")


def canonical_stage(value: Any) -> str:
    """Normalise a raw stage/status value to the canonical funnel token
    (KFI / APPLICATION / OFFER / COMPLETED / WITHDRAWN / UNKNOWN)."""
    return _STAGE_CANON.get(_norm(value), "UNKNOWN")


# Stages that are open and forecast-relevant (vs COMPLETED / WITHDRAWN / UNKNOWN).
ACTIVE_STAGES = ("KFI", "APPLICATION", "OFFER")


def case_stage_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve ``[case_id, application_id, stage, completion_date]`` from a raw
    weekly pipeline extract, reusing the contract aliases + stage normalisation.

    Used by the historical completion-rate model to track a case across weekly
    snapshots. Case identity prefers the pipeline case id, falling back to the
    application/KFI reference."""
    mapping, _ = resolve_source_columns(df)
    out = pd.DataFrame(index=df.index)
    cid_col = mapping.get("pipeline_case_identifier") or mapping.get("application_identifier")
    app_col = mapping.get("application_identifier") or mapping.get("pipeline_case_identifier")
    stage_col = mapping.get("pipeline_stage")
    comp_col = mapping.get("expected_completion_date")  # funds-released date in M2L KFI
    out["case_id"] = (df[cid_col].astype(str).str.strip() if cid_col
                      else pd.Series("", index=df.index))
    out["application_id"] = (df[app_col].astype(str).str.strip() if app_col
                             else out["case_id"])
    out["stage"] = (df[stage_col].map(canonical_stage) if stage_col
                    else pd.Series("UNKNOWN", index=df.index))
    out["completion_date"] = (_parse_date(df[comp_col]) if comp_col
                              else pd.Series(pd.NaT, index=df.index))
    return out


def _normalise_stage(out: pd.DataFrame, derived: List[str]) -> None:
    if "pipeline_stage" not in out.columns:
        return
    canon = out["pipeline_stage"].map(canonical_stage)
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
                                      historical_rates: Dict[str, float],
                                      derived: List[str]) -> str:
    """Assign ``completion_probability`` per the governed hierarchy and record the
    row-level ``completion_probability_source``. Returns the overall basis.

    Hierarchy (highest first):
      1. row-level explicit probability (a real source value)  -> ``row_level``
      2. empirical historical stage rate (sufficient history)  -> ``historical_stage_rate``
      3. configured stage probability                          -> ``configured_stage_rate``
      4. WITHDRAWN / inactive  -> excluded from weighting       -> ``excluded_withdrawn``
      5. UNKNOWN / unmapped stage -> no probability             -> ``missing_stage``
      6. otherwise no probability                               -> ``unavailable``
    """
    n = len(out)
    explicit = (coerce_numeric(out["completion_probability"])
                if "completion_probability" in out.columns
                else pd.Series(np.nan, index=out.index))
    stage = (out["pipeline_stage"].astype(str) if "pipeline_stage" in out.columns
             else pd.Series("UNKNOWN", index=out.index))

    prob = pd.Series(np.nan, index=out.index, dtype="float64")
    source = pd.Series("unavailable", index=out.index, dtype="object")
    for idx in out.index:
        s = stage.loc[idx]
        if pd.notna(explicit.loc[idx]):
            prob.loc[idx] = float(explicit.loc[idx]); source.loc[idx] = "row_level"
        elif s == "WITHDRAWN":
            source.loc[idx] = "excluded_withdrawn"          # NaN -> not weighted
        elif s in historical_rates:
            prob.loc[idx] = float(historical_rates[s]); source.loc[idx] = "historical_stage_rate"
        elif s in stage_probs:
            prob.loc[idx] = float(stage_probs[s]); source.loc[idx] = "configured_stage_rate"
        elif s in ("UNKNOWN", "", "nan", "None"):
            source.loc[idx] = "missing_stage"
        else:
            source.loc[idx] = "unavailable"

    out["completion_probability"] = prob
    out["completion_probability_source"] = source
    out["stage_conversion_probability"] = stage.map(stage_probs)
    for f in ("completion_probability", "completion_probability_source",
              "stage_conversion_probability"):
        if f not in derived:
            derived.append(f)

    # Expected funded amount == the economic amount; weighted by probability.
    if "current_outstanding_balance" in out.columns:
        amount = coerce_numeric(out["current_outstanding_balance"])
        out["expected_funded_amount"] = amount
        out["weighted_expected_funded_amount"] = amount * prob
        for f in ("expected_funded_amount", "weighted_expected_funded_amount"):
            if f not in derived:
                derived.append(f)

    used = set(source.unique())
    has_hist = "historical_stage_rate" in used
    has_cfg = "configured_stage_rate" in used or "row_level" in used
    if has_hist and has_cfg:
        return "mixed_historical_and_config"
    if has_hist:
        return "historical_observed"
    if has_cfg:
        return "stage_config"
    return "unavailable"


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
    "interest_rate_bucket", "expected_completion_month", "borrower_type",
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


def completion_probability_summary(out: pd.DataFrame) -> Dict[str, Any]:
    """Per-source counts + amounts for ``completion_probability_source``, plus the
    gross / excluded / weighted totals used by the forecast disclosure."""
    if "completion_probability_source" not in out.columns:
        return {}
    src = out["completion_probability_source"].astype(str)
    amount = (coerce_numeric(out["current_outstanding_balance"])
              if "current_outstanding_balance" in out.columns
              else pd.Series(0.0, index=out.index))
    weighted = (coerce_numeric(out["weighted_expected_funded_amount"])
                if "weighted_expected_funded_amount" in out.columns
                else pd.Series(np.nan, index=out.index))
    by_source: Dict[str, Any] = {}
    for s in sorted(src.unique()):
        mask = src == s
        by_source[s] = {"count": int(mask.sum()),
                        "amount": round(float(amount[mask].sum()), 2)}
    excluded_sources = {"excluded_withdrawn", "missing_stage", "unavailable"}
    excluded_mask = src.isin(excluded_sources)
    gross = float(amount.sum())
    excluded_amount = float(amount[excluded_mask].sum())
    active_gross = gross - excluded_amount
    weighted_total = float(weighted.sum())
    return {
        "by_source": by_source,
        "gross_pipeline_amount": round(gross, 2),
        "excluded_amount": round(excluded_amount, 2),
        "active_gross_amount": round(active_gross, 2),
        "weighted_expected_funded_amount": round(weighted_total, 2),
        "amount_weighted_historical": round(float(
            amount[src == "historical_stage_rate"].sum()), 2),
        "amount_weighted_config": round(float(
            amount[src.isin({"configured_stage_rate", "row_level"})].sum()), 2),
        "blended_weighted_conversion": (round(weighted_total / active_gross, 4)
                                        if active_gross > 0 else None),
        "excluded_count": int(excluded_mask.sum()),
    }


def _build_report(out: pd.DataFrame, mapping: Dict[str, str], unmatched: List[str],
                  derived: List[str], ltv_basis: Dict[str, Any],
                  group_aliases: List[str], bucket_issues: List[Dict[str, Any]],
                  rep_ts: Optional[pd.Timestamp],
                  prob_basis: str = "stage_config",
                  historical_model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        "completion_probability_basis": prob_basis,
        "completion_probability_summary": completion_probability_summary(out),
        "historical_completion_model": historical_model or {"available": False},
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


def _stage_breakdown_of(mask: pd.Series, stage: pd.Series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in stage[mask].value_counts().items()}


def classify_forecast_gaps(out: pd.DataFrame) -> List[Dict[str, Any]]:
    """Stage-classified forecast gaps (shared by validation + watchlist).

    Distinguishes INTENTIONAL exclusions (withdrawn/inactive — INFO, not weighted)
    from ACTIVE cases genuinely missing a probability or expected completion date
    (WARNING), and UNKNOWN/unmapped stages (WARNING with the offending values).
    Each item carries ``count``, ``by_stage``, ``excluded`` and ``weighted`` so the
    UI can explain it. ``check`` doubles as the watchlist ``category``.
    """
    items: List[Dict[str, Any]] = []
    if "pipeline_stage" not in out.columns or len(out) == 0:
        return items
    stage = out["pipeline_stage"].astype(str)
    active = stage.isin(ACTIVE_STAGES)

    # --- completion probability gaps ----------------------------------------- #
    if "completion_probability" in out.columns:
        prob_na = coerce_numeric(out["completion_probability"]).isna()
        withdrawn = prob_na & (stage == "WITHDRAWN")
        if withdrawn.any():
            c = int(withdrawn.sum())
            items.append(_diag(
                "withdrawn_excluded_from_weighting", "info",
                f"{c} withdrawn/inactive case(s) excluded from forecast probability weighting",
                count=c, by_stage=_stage_breakdown_of(withdrawn, stage),
                excluded=True, weighted=False))
        active_missing = prob_na & active
        if active_missing.any():
            c = int(active_missing.sum())
            items.append(_diag(
                "active_missing_completion_probability", "warning",
                f"{c} active case(s) missing completion probability",
                count=c, by_stage=_stage_breakdown_of(active_missing, stage),
                excluded=False, weighted=False))
        unknown = prob_na & ~stage.isin(list(ACTIVE_STAGES) + ["WITHDRAWN", "COMPLETED"])
        if unknown.any():
            c = int(unknown.sum())
            items.append(_diag(
                "unknown_stage_no_probability", "warning",
                f"{c} case(s) with an unknown/unmapped stage have no probability",
                count=c, by_stage=_stage_breakdown_of(unknown, stage),
                excluded=True, weighted=False))

    # --- expected completion date gaps --------------------------------------- #
    if "expected_completion_date" in out.columns:
        ecd_na = out["expected_completion_date"].isna()
        inactive_na = ecd_na & ~active
        active_na = ecd_na & active
        if inactive_na.any():
            c = int(inactive_na.sum())
            items.append(_diag(
                "expected_completion_date_not_required", "info",
                f"{c} withdrawn/inactive case(s) have no expected completion date",
                count=c, by_stage=_stage_breakdown_of(inactive_na, stage),
                excluded=True, weighted=False))
        if active_na.any():
            c = int(active_na.sum())
            items.append(_diag(
                "active_missing_expected_completion_date", "warning",
                f"{c} active case(s) missing expected completion date",
                count=c, by_stage=_stage_breakdown_of(active_na, stage),
                excluded=False, weighted=True))
    return items


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
    for fld, label in (("current_loan_to_value", "LTV"),
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

    # Stage-classified forecast gaps (probability + expected completion date):
    # withdrawn/inactive -> INFO (intentionally excluded); active -> WARNING.
    diags.extend(classify_forecast_gaps(out))

    # Completion probability coverage (only a real WARNING when NO active case is
    # weightable at all — withdrawn-only exclusions are handled above as INFO).
    if "completion_probability" in out.columns:
        prob_cov, _ = _numeric_coverage(out, "completion_probability")
        if prob_cov == 0:
            diags.append(_diag("no_weightable_probability", "warning",
                               "no completion probability available (no row value and no "
                               "config stage probability) — forecast not yet weightable"))
    return diags


def diagnostics_by_severity(report: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Group a report's ``data_quality`` diagnostics by severity."""
    out: Dict[str, List[Dict[str, Any]]] = {"blocker": [], "warning": [], "info": []}
    for d in report.get("data_quality", []) or []:
        out.setdefault(d.get("severity", "info"), []).append(d)
    return out
