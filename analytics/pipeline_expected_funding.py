"""Expected funding / forward exposure modelling for normalized pipeline snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExpectedFundingConfig:
    model_version: str = "pipeline_expected_v1"
    stage_probabilities: dict[str, float] = field(
        default_factory=lambda: {"KFI": 0.20, "APPLICATION": 0.45, "OFFER": 0.75, "COMPLETED": 1.0}
    )
    stage_days_to_fund: dict[str, int] = field(
        default_factory=lambda: {"KFI": 90, "APPLICATION": 60, "OFFER": 30, "COMPLETED": 0}
    )
    broker_adjustments: dict[str, float] = field(default_factory=dict)
    product_adjustments: dict[str, float] = field(default_factory=dict)
    probability_floor: float = 0.0
    probability_cap: float = 1.0
    include_stages: list[str] = field(default_factory=lambda: ["KFI", "APPLICATION", "OFFER"])
    exclude_stages: list[str] = field(default_factory=lambda: ["WITHDRAWN", "COMPLETED"])
    exclude_reconciled_rows: bool = True
    high_confidence_threshold: float = 0.7


DEFAULT_CONFIG = ExpectedFundingConfig()


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def load_expected_funding_config(raw_config: dict[str, Any] | None) -> ExpectedFundingConfig:
    if not raw_config:
        return DEFAULT_CONFIG

    params = {
        "model_version": raw_config.get("model_version", DEFAULT_CONFIG.model_version),
        "stage_probabilities": raw_config.get("stage_probabilities", DEFAULT_CONFIG.stage_probabilities),
        "stage_days_to_fund": raw_config.get("stage_days_to_fund", DEFAULT_CONFIG.stage_days_to_fund),
        "broker_adjustments": raw_config.get("broker_adjustments", DEFAULT_CONFIG.broker_adjustments),
        "product_adjustments": raw_config.get("product_adjustments", DEFAULT_CONFIG.product_adjustments),
        "probability_floor": raw_config.get("probability_floor", DEFAULT_CONFIG.probability_floor),
        "probability_cap": raw_config.get("probability_cap", DEFAULT_CONFIG.probability_cap),
        "include_stages": raw_config.get("include_stages", DEFAULT_CONFIG.include_stages),
        "exclude_stages": raw_config.get("exclude_stages", DEFAULT_CONFIG.exclude_stages),
        "exclude_reconciled_rows": raw_config.get("exclude_reconciled_rows", DEFAULT_CONFIG.exclude_reconciled_rows),
        "high_confidence_threshold": raw_config.get(
            "high_confidence_threshold", DEFAULT_CONFIG.high_confidence_threshold
        ),
    }
    return ExpectedFundingConfig(**params)


def build_expected_funding_dataset(
    pipeline_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
    config: ExpectedFundingConfig | None = None,
) -> pd.DataFrame:
    config = config or DEFAULT_CONFIG
    if pipeline_df is None or pipeline_df.empty:
        return pd.DataFrame()

    out = pipeline_df.copy()

    stage_col = "stage" if "stage" in out.columns else "pipeline_stage"
    out["stage"] = _clean_text(out.get(stage_col, "")).str.upper()

    reconciled_cols = [
        "pipeline_opportunity_id",
        "is_reconciled_to_funded",
        "reconciliation_match_rule",
    ]
    if reconciliation_df is not None and not reconciliation_df.empty:
        bridge = reconciliation_df[[c for c in reconciled_cols if c in reconciliation_df.columns]].drop_duplicates(
            subset=["pipeline_opportunity_id"]
        )
        out = out.merge(bridge, on="pipeline_opportunity_id", how="left")
    if "is_reconciled_to_funded" not in out.columns:
        out["is_reconciled_to_funded"] = False
    out["is_reconciled_to_funded"] = out["is_reconciled_to_funded"].fillna(False)
    if "reconciliation_match_rule" not in out.columns:
        out["reconciliation_match_rule"] = ""
    out["reconciliation_match_rule"] = _clean_text(out["reconciliation_match_rule"])

    include_stages = {s.upper() for s in config.include_stages}
    exclude_stages = {s.upper() for s in config.exclude_stages}
    eligible = out["stage"].isin(include_stages) & ~out["stage"].isin(exclude_stages)
    eligible &= ~out.get("is_terminal_stage", False)
    if config.exclude_reconciled_rows:
        eligible &= ~out["is_reconciled_to_funded"]
    out = out.loc[eligible].copy()
    if out.empty:
        return pd.DataFrame()

    out["pipeline_amount"] = pd.to_numeric(out.get("loan_amount", pd.NA), errors="coerce").fillna(0.0)
    out["estimated_value"] = pd.to_numeric(out.get("estimated_value", pd.NA), errors="coerce")
    out["current_ltv"] = pd.to_numeric(out.get("current_ltv", pd.NA), errors="coerce")

    out["base_probability"] = out["stage"].map(config.stage_probabilities).fillna(0.0)
    out["expected_days_to_fund"] = out["stage"].map(config.stage_days_to_fund).fillna(0).astype(int)

    broker_key = _clean_text(out.get("broker", "")).str.upper()
    product_key = _clean_text(out.get("product", "")).str.upper()

    out["broker_adjustment"] = broker_key.map({k.upper(): v for k, v in config.broker_adjustments.items()}).fillna(0.0)
    out["product_adjustment"] = product_key.map({k.upper(): v for k, v in config.product_adjustments.items()}).fillna(0.0)

    out["final_probability"] = (
        out["base_probability"] + out["broker_adjustment"] + out["product_adjustment"]
    ).clip(lower=config.probability_floor, upper=config.probability_cap)

    out["expected_funded_amount"] = out["pipeline_amount"] * out["final_probability"]
    out["snapshot_date"] = pd.to_datetime(out.get("snapshot_date"), errors="coerce")
    out["expected_funded_date"] = out["snapshot_date"] + pd.to_timedelta(out["expected_days_to_fund"], unit="D")

    out["high_confidence_flag"] = out["final_probability"] >= config.high_confidence_threshold
    out["high_confidence_expected_amount"] = out["expected_funded_amount"].where(out["high_confidence_flag"], 0.0)

    out["model_version"] = config.model_version

    keep_cols = [
        "pipeline_opportunity_id",
        "snapshot_date",
        "stage",
        "broker",
        "product",
        "property_region",
        "property_region_code",
        "youngest_borrower_age",
        "borrower_age",
        "pipeline_amount",
        "estimated_value",
        "current_ltv",
        "base_probability",
        "broker_adjustment",
        "product_adjustment",
        "final_probability",
        "expected_funded_amount",
        "expected_days_to_fund",
        "expected_funded_date",
        "is_reconciled_to_funded",
        "reconciliation_match_rule",
        "high_confidence_expected_amount",
        "high_confidence_flag",
        "model_version",
    ]
    return out[[c for c in keep_cols if c in out.columns]].copy()
