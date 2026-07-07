"""mi_agent_pptx.pipeline_prep — config-driven pipeline-tape canonicalisation.

A pipeline run may land as the *raw* central pipeline tape
(``18a_central_pipeline_tape.csv``) whose columns carry source-alias headers
("Loan Amount", "Status", "Broker", "Property Region", "Product Rate", …) rather
than the canonical funded field names. This module maps those raw columns onto
the canonical names the deck resolves against, so pipeline / forecast charts
render from real data instead of degrading to placeholders.

Everything here is **registry/config-driven** — the alias map comes from
``config/mi/pipeline_field_contract.yaml`` (``source_aliases`` /
``funded_correlation``) and the stage vocabulary + completion probabilities /
days-to-fund come from ``config/client/pipeline_expected_funding.yaml``. No new
economics are invented: the forecast weighting and expected-completion timing
are exactly the registry-declared bridge inputs.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from .registry_loader import REPO_ROOT

_CONTRACT_PATH = REPO_ROOT / "config" / "mi" / "pipeline_field_contract.yaml"
_FORECAST_CFG_PATH = REPO_ROOT / "config" / "client" / "pipeline_expected_funding.yaml"

# Canonical stage vocabulary (mirrors analytics/pipeline_prep._map_stage).
_STAGE_PATTERNS = [
    ("KFI", "KFI"),
    ("APPLICATION", "APPLICATION"),
    ("OFFER", "OFFER"),
    ("COMPLET", "COMPLETED"),
    ("WITHDRAW", "WITHDRAWN"),
    ("LEGAL", "OFFER"),
]
_STAGE_LABEL = {
    "KFI": "KFI Issued", "APPLICATION": "Application", "OFFER": "Offer Issued",
    "COMPLETED": "Completed", "WITHDRAWN": "Withdrawn", "OTHER": "Other",
}

# Extra intermediate aliases (from the MI Agent's own normalize_pipeline_snapshot
# rename map) so both the raw M2L headers and the lightly-normalised central tape
# resolve identically.
_EXTRA_ALIASES: Dict[str, List[str]] = {
    "current_outstanding_balance": ["loan_amount", "facility", "expected_funded_amount"],
    "current_valuation_amount": ["estimated_value", "property_value"],
    "current_interest_rate": ["product_rate"],
    "broker_channel": ["broker"],
    "collateral_geography": ["property_region"],
    "pipeline_stage": ["status", "status_raw", "dpr_status", "dpr_status_raw", "stage"],
    "erm_product_type": ["product", "loan_plan"],
    "youngest_borrower_age": ["borrower_age", "youngest_borrower_age"],
    "expected_completion_date": ["expected_completion_date", "offer_date"],
    "pipeline_reference_date": ["snapshot_date", "application_submitted_date",
                                "kfi_submitted_date"],
    "completion_probability": ["completion_probability", "stage_conversion_probability"],
    "weighted_expected_funded_amount": ["weighted_expected_funded_amount"],
}
_DOB_ALIASES = ["dob_app_1", "dob_app_2", "date_of_birth"]


def _norm(name: str) -> str:
    return re.sub(r"[\s_]+", " ", str(name).strip().lower())


@lru_cache(maxsize=1)
def _alias_map() -> Dict[str, str]:
    """Build ``{normalised_alias: canonical_field}`` from the contract + extras."""
    out: Dict[str, str] = {}

    def add(canonical: str, aliases: List[str]) -> None:
        for a in aliases or []:
            out.setdefault(_norm(a), canonical)
        out.setdefault(_norm(canonical), canonical)

    try:
        cfg = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}
    for section in ("funded_correlated_fields", "pipeline_specific_fields"):
        for canonical, spec in (cfg.get(section, {}) or {}).items():
            if not isinstance(spec, dict):
                continue
            target = spec.get("semantic_registry_field", canonical)
            add(target, spec.get("source_aliases", []))
            add(target, spec.get("funded_correlation", []))
    for canonical, aliases in _EXTRA_ALIASES.items():
        add(canonical, aliases)
    return out


@lru_cache(maxsize=1)
def _forecast_cfg() -> Dict[str, Any]:
    try:
        return yaml.safe_load(_FORECAST_CFG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _map_stage(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.upper().fillna("")
    out = pd.Series("OTHER", index=s.index, dtype="object")
    for pat, label in _STAGE_PATTERNS:
        out[s.str.contains(pat, na=False)] = label
    return out


def is_raw_pipeline(df: pd.DataFrame) -> bool:
    """True when the tape lacks the canonical balance/stage names (needs prep)."""
    canon = {"current_outstanding_balance", "current_principal_balance"}
    return not (canon & set(df.columns)) or "pipeline_stage" not in df.columns


def canonicalise_pipeline(df: pd.DataFrame, *, as_of: Optional[str] = None) -> pd.DataFrame:
    """Return a copy of *df* with canonical pipeline field names + forecast inputs.

    Idempotent: a tape already carrying canonical names passes through with only
    the derived forecast columns added where missing.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    amap = _alias_map()

    # 1) rename recognised source columns -> canonical (first alias wins).
    rename: Dict[str, str] = {}
    for col in out.columns:
        canon = amap.get(_norm(col))
        if canon and canon not in out.columns and canon not in rename.values():
            rename[col] = canon
    out = out.rename(columns=rename)

    # 2) normalise the pipeline stage vocabulary + readable label.
    if "pipeline_stage" in out.columns:
        mapped = _map_stage(out["pipeline_stage"])
        out["pipeline_stage"] = mapped.map(lambda s: _STAGE_LABEL.get(s, s))
        out["_stage_code"] = mapped

    # 3) youngest borrower age from DOB columns when age absent.
    if "youngest_borrower_age" not in out.columns:
        dob_norm = {_norm(a) for a in _DOB_ALIASES}
        dob_col = next((c for c in out.columns if _norm(c) in dob_norm), None)
        if dob_col is not None and as_of:
            dob = pd.to_datetime(out[dob_col], errors="coerce")
            asof_ts = pd.to_datetime(as_of, errors="coerce")
            if pd.notna(asof_ts):
                out["youngest_borrower_age"] = ((asof_ts - dob).dt.days / 365.25).round()

    # 4) derive current LTV = balance / valuation when not explicit.
    if ("current_loan_to_value" not in out.columns
            and "current_outstanding_balance" in out.columns
            and "current_valuation_amount" in out.columns):
        bal = pd.to_numeric(out["current_outstanding_balance"], errors="coerce")
        val = pd.to_numeric(out["current_valuation_amount"], errors="coerce")
        out["current_loan_to_value"] = (bal / val).where(val > 0)

    # 5) registry forecast inputs: completion probability + weighted amount.
    cfg = _forecast_cfg()
    probs = {str(k).upper(): float(v)
             for k, v in (cfg.get("stage_probabilities", {}) or {}).items()}
    days = {str(k).upper(): float(v)
            for k, v in (cfg.get("stage_days_to_fund", {}) or {}).items()}
    if "_stage_code" in out.columns:
        if "completion_probability" not in out.columns and probs:
            out["completion_probability"] = out["_stage_code"].map(probs).fillna(0.0)
        if ("weighted_expected_funded_amount" not in out.columns
                and "current_outstanding_balance" in out.columns
                and "completion_probability" in out.columns):
            out["weighted_expected_funded_amount"] = (
                pd.to_numeric(out["current_outstanding_balance"], errors="coerce")
                * pd.to_numeric(out["completion_probability"], errors="coerce"))
        # 6) expected completion date = reference date + stage days-to-fund.
        if "expected_completion_date" not in out.columns and days:
            ref = None
            for c in ("pipeline_reference_date",):
                if c in out.columns:
                    ref = pd.to_datetime(out[c], errors="coerce")
                    break
            if ref is None and as_of:
                ref = pd.Series(pd.to_datetime(as_of, errors="coerce"), index=out.index)
            if ref is not None:
                lag = out["_stage_code"].map(days).fillna(45.0)
                out["expected_completion_date"] = ref + pd.to_timedelta(lag, unit="D")

    return out
