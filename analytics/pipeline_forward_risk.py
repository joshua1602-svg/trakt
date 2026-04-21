"""Forward exposure aggregations bridging funded and expected pipeline views."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analytics.portfolio_semantics import normalize_region_labels


@dataclass(frozen=True)
class ForwardRiskSchemaConfig:
    funded_region_column: str | None = None
    funded_exposure_column: str | None = None


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _resolve_funded_column(
    funded_df: pd.DataFrame,
    explicit_col: str | None,
    default_candidates: list[str],
    fallback_candidates: list[str],
    column_type: str,
) -> str:
    if explicit_col:
        if explicit_col in funded_df.columns:
            return explicit_col
        raise KeyError(f"Configured funded {column_type} column not found: {explicit_col}")

    for c in default_candidates:
        if c in funded_df.columns:
            return c

    for c in fallback_candidates:
        if c in funded_df.columns:
            return c

    raise KeyError(f"Could not find funded {column_type} column.")


def aggregate_forward_region_exposure(
    funded_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    region_limit_df: pd.DataFrame | None = None,
    schema_config: ForwardRiskSchemaConfig | None = None,
) -> pd.DataFrame:
    schema_config = schema_config or ForwardRiskSchemaConfig()

    funded_region = pd.DataFrame(columns=["property_region", "funded_current_exposure"])
    expected_region = pd.DataFrame(columns=["property_region", "expected_pipeline_exposure"])

    if funded_df is not None and not funded_df.empty:
        region_col = _resolve_funded_column(
            funded_df,
            schema_config.funded_region_column,
            default_candidates=["geographic_region", "geographic_region_classification"],
            fallback_candidates=["property_region", "region"],
            column_type="region",
        )
        exposure_col = _resolve_funded_column(
            funded_df,
            schema_config.funded_exposure_column,
            default_candidates=["current_principal_balance", "total_balance", "current_outstanding_balance"],
            fallback_candidates=["current_balance", "balance", "loan_balance", "exposure"],
            column_type="exposure",
        )

        funded_region = (
            funded_df.assign(property_region=normalize_region_labels(funded_df[region_col]))
            .groupby("property_region", dropna=False, observed=True)[exposure_col]
            .sum(min_count=1)
            .reset_index(name="funded_current_exposure")
        )

    if expected_df is not None and not expected_df.empty:
        expected_region = (
            expected_df.assign(property_region=normalize_region_labels(expected_df.get("property_region", "")))
            .groupby("property_region", dropna=False, observed=True)["expected_funded_amount"]
            .sum(min_count=1)
            .reset_index(name="expected_pipeline_exposure")
        )

    combined = funded_region.merge(expected_region, on="property_region", how="outer")
    combined["property_region"] = _clean_text(combined["property_region"])
    combined["funded_current_exposure"] = pd.to_numeric(combined["funded_current_exposure"], errors="coerce").fillna(0.0)
    combined["expected_pipeline_exposure"] = pd.to_numeric(combined["expected_pipeline_exposure"], errors="coerce").fillna(0.0)
    combined["combined_forward_exposure"] = (
        combined["funded_current_exposure"] + combined["expected_pipeline_exposure"]
    )

    total_combined = combined["combined_forward_exposure"].sum()
    combined["combined_exposure_pct"] = 0.0
    if total_combined > 0:
        combined["combined_exposure_pct"] = combined["combined_forward_exposure"] / total_combined

    if region_limit_df is not None and not region_limit_df.empty:
        limits = region_limit_df.copy()
        limits = limits.rename(columns={"region": "property_region", "limit_pct": "region_limit_pct"})
        if "property_region" in limits.columns:
            limits["property_region"] = normalize_region_labels(limits["property_region"])
            combined = combined.merge(
                limits[[c for c in ["property_region", "region_limit_pct"] if c in limits.columns]],
                on="property_region",
                how="left",
            )

    return combined.sort_values("combined_forward_exposure", ascending=False).reset_index(drop=True)
