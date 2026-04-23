"""Shared semantics helpers for region mapping and LTV calculations."""

from __future__ import annotations

import pandas as pd


REGION_NAME_TO_CODE: dict[str, str] = {
    "LONDON": "UKI",
    "SOUTH EAST": "UKJ",
    "EAST ANGLIA": "UKH",
    "EAST OF ENGLAND": "UKH",
    "SOUTH WEST": "UKK",
    "WEST MIDLANDS": "UKG",
    "EAST MIDLANDS": "UKF",
    "NORTH WEST": "UKD",
    "YORKSHIRE AND HUMBERSIDE": "UKE",
    "YORKSHIRE & HUMBERSIDE": "UKE",
    "NORTH EAST": "UKC",
    "WALES": "UKL",
    "SCOTLAND": "UKM",
    "NORTHERN IRELAND": "UKN",
}

REGION_CANONICAL_LABEL: dict[str, str] = {
    "LONDON": "London",
    "SOUTH EAST": "South East",
    "EAST ANGLIA": "East Anglia",
    "EAST OF ENGLAND": "East Anglia",
    "SOUTH WEST": "South West",
    "WEST MIDLANDS": "West Midlands",
    "EAST MIDLANDS": "East Midlands",
    "NORTH WEST": "North West",
    "YORKSHIRE AND HUMBERSIDE": "Yorkshire and Humberside",
    "YORKSHIRE & HUMBERSIDE": "Yorkshire and Humberside",
    "NORTH EAST": "North East",
    "WALES": "Wales",
    "SCOTLAND": "Scotland",
    "NORTHERN IRELAND": "Northern Ireland",
}

VALID_REGION_CODES = set(REGION_NAME_TO_CODE.values())


def _clean_region_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def normalize_region_labels(series: pd.Series) -> pd.Series:
    cleaned = _clean_region_text(series)
    upper = cleaned.str.upper()
    canonical = upper.map(REGION_CANONICAL_LABEL)
    return canonical.fillna(cleaned)


def region_codes_from_labels(series: pd.Series) -> pd.Series:
    cleaned = _clean_region_text(series)
    upper = cleaned.str.upper()
    mapped = upper.map(REGION_NAME_TO_CODE)
    is_code = upper.isin(VALID_REGION_CODES)
    mapped.loc[is_code] = upper.loc[is_code]
    return mapped.astype("string")


def safe_ltv_percent(balance: pd.Series, valuation: pd.Series) -> pd.Series:
    bal = pd.to_numeric(balance, errors="coerce")
    val = pd.to_numeric(valuation, errors="coerce")
    out = pd.Series(pd.NA, index=bal.index, dtype="Float64")
    valid = bal.notna() & val.notna() & (val > 0)
    out.loc[valid] = (bal.loc[valid] / val.loc[valid]) * 100.0
    return out
