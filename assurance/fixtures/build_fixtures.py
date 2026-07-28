"""Deterministic edge-case fixtures for the assurance programme.

Each fixture is a small, explicit mutation of the committed multibook platform
canonical, built for one pathology. Fixtures are written to
``assurance/fixtures/generated/`` and are byte-deterministic: no randomness, no
timestamps. A manifest records a checksum and the salient facts of each.

Run:  python assurance/fixtures/build_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "synthetic_demo/output/multibook/platform_2026-06-30_canonical_typed.csv"
PRIOR = ROOT / "synthetic_demo/output/multibook/platform_2026-05-31_canonical_typed.csv"
OUT = Path(__file__).resolve().parent / "generated"

BAL = "current_outstanding_balance"
LTV = "current_loan_to_value"
PID = "source_portfolio_id"
PTYPE = "source_portfolio_type"
LOAN = "loan_identifier"
CUR = "exposure_currency_denomination"
PROP = "property_type"


def _base() -> pd.DataFrame:
    return pd.read_csv(BASE)


def single_portfolio(df):
    return df[df[PID] == "alp_origination"].copy()


def two_portfolios(df):
    return df[df[PID].isin(["alp_origination", "alp_acquired"])].copy()


def three_portfolios(df):
    return df.copy()


def one_loan(df):
    return df.iloc[:1].copy()


def empty_portfolio(df):
    return df.iloc[0:0].copy()


def mixed_currency(df):
    out = df.copy()
    # Flip one portfolio to EUR so the scope is internally mixed.
    out.loc[out[PID] == "alp_acquired", CUR] = "EUR"
    if "collateral_currency" in out.columns:
        out.loc[out[PID] == "alp_acquired", "collateral_currency"] = "EUR"
    return out


def missing_currency(df):
    out = df.copy()
    for c in (CUR, "collateral_currency"):
        if c in out.columns:
            out = out.drop(columns=[c])
    return out


def null_currency(df):
    out = df.copy()
    out.loc[out.index[:10], CUR] = None
    return out


def duplicate_loan_ids_within(df):
    out = single_portfolio(df).copy()
    # Force the first two loan ids to collide within one portfolio.
    ids = out[LOAN].tolist()
    if len(ids) >= 2:
        out.iloc[1, out.columns.get_loc(LOAN)] = ids[0]
    return out


def duplicate_ids_across_portfolios(df):
    out = two_portfolios(df).copy()
    # Reuse an origination id inside the acquired book (no provenance suffix).
    orig = out[out[PID] == "alp_origination"][LOAN].iloc[0]
    acq_idx = out[out[PID] == "alp_acquired"].index[0]
    out.loc[acq_idx, LOAN] = orig
    return out


def missing_provenance(df):
    out = df.copy()
    return out.drop(columns=[c for c in (PID, PTYPE, "source_portfolio_label",
                                         "portfolio_cohort") if c in out.columns])


def two_reporting_dates(df):
    return pd.concat([pd.read_csv(PRIOR), df], ignore_index=True)


def stale_only(df):
    # Only a prior cut exists; a request for the current date must adjust+disclose.
    return pd.read_csv(PRIOR).copy()


def zero_balances(df):
    out = df.copy()
    out[BAL] = 0.0
    if "current_principal_balance" in out.columns:
        out["current_principal_balance"] = 0.0
    return out


def zero_weights(df):
    # Weight column (balance) all zero, LTV populated -> WA must not fall back.
    out = df.copy()
    out[BAL] = 0.0
    return out


def negative_weights(df):
    out = df.copy()
    out.loc[out.index[:5], BAL] = -1000.0
    return out


def all_null_measure(df):
    out = df.copy()
    out[BAL] = None
    return out


def all_null_dimension(df):
    out = df.copy()
    out[PROP] = None
    return out


def unknown_categories(df):
    out = df.copy()
    out.loc[out.index[:8], PROP] = None  # bucketed as (unknown)
    return out


def high_cardinality_dimension(df):
    out = df.copy()
    out[PROP] = [f"cat_{i}" for i in range(len(out))]
    return out


def conflicting_asset_class(df):
    out = df.copy()
    if "asset_class" not in out.columns:
        out["asset_class"] = "equity_release"
    out.loc[out[PID] == "alp_acquired", "asset_class"] = "buy_to_let"
    return out


def missing_asset_class(df):
    out = df.copy()
    return out.drop(columns=[c for c in ("asset_class",) if c in out.columns])


MUTATIONS: Dict[str, Callable] = {
    "single_portfolio": single_portfolio,
    "two_portfolios": two_portfolios,
    "three_portfolios": three_portfolios,
    "one_loan": one_loan,
    "empty_portfolio": empty_portfolio,
    "mixed_currency": mixed_currency,
    "missing_currency": missing_currency,
    "null_currency": null_currency,
    "duplicate_loan_ids_within": duplicate_loan_ids_within,
    "duplicate_ids_across_portfolios": duplicate_ids_across_portfolios,
    "missing_provenance": missing_provenance,
    "two_reporting_dates": two_reporting_dates,
    "stale_only": stale_only,
    "zero_balances": zero_balances,
    "zero_weights": zero_weights,
    "negative_weights": negative_weights,
    "all_null_measure": all_null_measure,
    "all_null_dimension": all_null_dimension,
    "unknown_categories": unknown_categories,
    "high_cardinality_dimension": high_cardinality_dimension,
    "conflicting_asset_class": conflicting_asset_class,
    "missing_asset_class": missing_asset_class,
}

PURPOSES = {
    "single_portfolio": "Total == the one book; scope resolution baseline.",
    "two_portfolios": "Disjoint sub-scopes sum to the two-book total.",
    "three_portfolios": "Full multibook; base numerical substrate.",
    "one_loan": "Single-name concentration == 100%; averages == that loan.",
    "empty_portfolio": "No rows: controlled empty result, never a fabricated zero KPI.",
    "mixed_currency": "Monetary totals suppressed; count shares still valid.",
    "missing_currency": "No currency field: totals allowed but flagged as unverified currency.",
    "null_currency": "Null currency values must not be treated as a distinct currency.",
    "duplicate_loan_ids_within": "Dedup / identity integrity within one book.",
    "duplicate_ids_across_portfolios": "Composite key needed; must not collapse two loans.",
    "missing_provenance": "No source_portfolio_id: lens scoping degrades to Total with disclosure.",
    "two_reporting_dates": "Point-in-time resolves as-of-latest, never aggregates both.",
    "stale_only": "Requested date unavailable: adjust to prior and disclose the gap.",
    "zero_balances": "Zero exposure: shares undefined, not divide-by-zero.",
    "zero_weights": "Weighted average must be None, never a simple-mean fallback.",
    "negative_weights": "Negative weights excluded from the positive-weight denominator.",
    "all_null_measure": "All-null measure: no fabricated total.",
    "all_null_dimension": "All-null dimension: single (unknown) bucket == full population.",
    "unknown_categories": "Unknown bucket kept in the denominator; shares still sum to 1.",
    "high_cardinality_dimension": "One category per loan: deterministic ranking, no crash.",
    "conflicting_asset_class": "Conflicting asset classes must not be silently merged.",
    "missing_asset_class": "No asset-class declaration: asset-specific fields not applied.",
}


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build() -> Dict[str, Dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    base = _base()
    manifest: Dict[str, Dict] = {}
    for name, fn in MUTATIONS.items():
        frame = fn(base)
        path = OUT / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest[name] = {
            "file": f"generated/{name}.csv",
            "rows": int(len(frame)),
            "checksum": _checksum(path),
            "purpose": PURPOSES[name],
        }
    manifest_path = Path(__file__).resolve().parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = build()
    print(f"built {len(m)} fixtures under {OUT}")
    for name, meta in m.items():
        print(f"  {name:34s} rows={meta['rows']:4d} sha={meta['checksum']}")
