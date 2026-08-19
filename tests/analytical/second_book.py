"""tests/analytical/second_book.py — a second, deliberately different book.

The analytical capability layer is accepted only if it works on a book whose
economics are not the demonstration book's. This builds one: a synthetic
equity-release platform with inverted provenance mix, a far wider direct/acquired
LTV gap, a much larger front book, a different geographic centre of gravity,
smaller and more skewed tickets, and realistic missing values.

It writes the SAME governed artefact the platform assembler writes — a dated
``platform_canonical_typed.csv`` plus its manifest, and a ``latest/`` pointer —
so the MI read path, the preparation layer, the derived seasoning and vintage
fields and every capability under test run exactly as they do on the first book.
No canonical field is invented and no code path is special-cased.

Deterministic: seeded once, no wall-clock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SYNTHETIC_NOTICE = "SYNTHETIC TEST DATA — NOT A REAL CUSTOMER"

CLIENT_ID = "kestrelmoor"
DISPLAY_NAME = "Kestrelmoor Lifetime Lending (synthetic)"
SELLER_NAME = "Thornwood Assurance (synthetic)"

#: Reporting dates, oldest first. Three retained snapshots, as the first book has.
REPORTING_DATES: Tuple[str, ...] = ("2026-04-30", "2026-05-31", "2026-06-30")

#: Regions and their share of ORIGINATIONS. Deliberately North-West/Scotland led,
#: the inverse of the first book's South East concentration.
_REGION_WEIGHTS: Dict[str, float] = {
    "North West": 0.24, "Scotland": 0.16, "Yorkshire and The Humber": 0.12,
    "North East": 0.10, "West Midlands": 0.09, "Wales": 0.08,
    "East Midlands": 0.07, "South West": 0.06, "London": 0.05,
    "South East": 0.03,
}

#: ITL3 codes per region, so the geographic fields carry a plausible NUTS value
#: rather than a region name in a code column.
_ITL3 = {
    "North West": "TLD33", "Scotland": "TLM75", "Yorkshire and The Humber": "TLE31",
    "North East": "TLC22", "West Midlands": "TLG31", "Wales": "TLL16",
    "East Midlands": "TLF21", "South West": "TLK14", "London": "TLI43",
    "South East": "TLJ25",
}

_POSTCODE_PREFIX = {
    "North West": "M", "Scotland": "EH", "Yorkshire and The Humber": "LS",
    "North East": "NE", "West Midlands": "B", "Wales": "CF",
    "East Midlands": "NG", "South West": "BS", "London": "SE", "South East": "GU",
}


@dataclass(frozen=True)
class PortfolioSpec:
    portfolio_id: str
    portfolio_type: str
    label: str
    loans: int
    #: Mean initial advance and its spread — the ticket profile.
    mean_advance: float
    advance_spread: float
    #: Mean origination LTV in POINTS, and its spread.
    mean_origination_ltv: float
    ltv_spread: float
    #: Mean roll-up rate in points.
    mean_rate: float
    #: Mean borrower age at origination.
    mean_age: float
    #: Share of the portfolio originated within the last 12 months.
    front_book_share: float
    #: Earliest and latest origination year.
    first_vintage: int
    last_vintage: int
    acquisition_date: str = ""


#: Acquired is 67% of the book by balance here, against 30% on the first book,
#: and the direct/acquired LTV gap is ~14 points against ~0.7.
PORTFOLIOS: Tuple[PortfolioSpec, ...] = (
    PortfolioSpec("kmr_direct_north", "direct", "Kestrelmoor Direct — North",
                  loans=3160, mean_advance=78_000.0, advance_spread=42_000.0,
                  mean_origination_ltv=22.0, ltv_spread=7.0, mean_rate=6.9,
                  mean_age=68.5, front_book_share=0.34,
                  first_vintage=2016, last_vintage=2026),
    PortfolioSpec("kmr_direct_retail", "direct", "Kestrelmoor Direct — Retail",
                  loans=2452, mean_advance=71_000.0, advance_spread=38_000.0,
                  mean_origination_ltv=24.5, ltv_spread=7.5, mean_rate=7.1,
                  mean_age=69.5, front_book_share=0.30,
                  first_vintage=2017, last_vintage=2026),
    PortfolioSpec("kmr_acq_meridian", "acquired", "Kestrelmoor Acquired — Meridian",
                  loans=4041, mean_advance=126_000.0, advance_spread=74_000.0,
                  mean_origination_ltv=33.0, ltv_spread=9.0, mean_rate=6.2,
                  mean_age=76.0, front_book_share=0.19,
                  first_vintage=2013, last_vintage=2025,
                  acquisition_date="2024-03-31"),
    PortfolioSpec("kmr_acq_thornwood", "acquired", "Kestrelmoor Acquired — Thornwood",
                  loans=2602, mean_advance=118_000.0, advance_spread=68_000.0,
                  mean_origination_ltv=35.5, ltv_spread=9.5, mean_rate=6.05,
                  mean_age=77.5, front_book_share=0.17,
                  first_vintage=2013, last_vintage=2025,
                  acquisition_date="2025-01-31"),
)

SEED = 20260419


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def _origination_dates(rng: np.random.Generator, spec: PortfolioSpec,
                       latest: date) -> np.ndarray:
    """Origination dates with a deliberately front-loaded seasoning profile."""
    n = spec.loans
    front = int(round(n * spec.front_book_share))
    # Front book: within the last 12 months of the latest reporting date.
    front_months = rng.integers(0, 12, front)
    # Back book: spread across the portfolio's vintage range.
    span_months = max(13, (latest.year - spec.first_vintage) * 12)
    back_months = rng.integers(13, span_months, n - front)
    months = np.concatenate([front_months, back_months])
    rng.shuffle(months)
    out = []
    for offset in months:
        year = latest.year
        month = latest.month - int(offset)
        while month <= 0:
            month += 12
            year -= 1
        day = min(28, 1 + int(rng.integers(0, 28)))
        out.append(date(year, month, day).isoformat())
    return np.array(out, dtype=object)


def _build_portfolio(spec: PortfolioSpec, rng: np.random.Generator,
                     latest: date) -> pd.DataFrame:
    n = spec.loans
    origination = _origination_dates(rng, spec, latest)

    # A log-normal advance produces the skew a real book has: a median well
    # below the mean and a long right tail, which is what makes the largest
    # exposure meaningful.
    sigma = 0.62
    mu = np.log(max(spec.mean_advance, 1.0)) - (sigma ** 2) / 2
    advance = np.round(rng.lognormal(mu, sigma, n), 2)
    advance = np.clip(advance, 18_000.0, 900_000.0)

    origination_ltv = np.clip(
        rng.normal(spec.mean_origination_ltv, spec.ltv_spread, n), 6.0, 62.0)
    valuation_at_origination = np.round(advance / (origination_ltv / 100.0), 2)
    rate = np.round(np.clip(rng.normal(spec.mean_rate, 0.55, n), 3.2, 9.4), 3)
    age_at_origination = np.clip(
        np.round(rng.normal(spec.mean_age, 6.2, n)), 55, 94).astype(int)

    regions = rng.choice(list(_REGION_WEIGHTS),
                         size=n, p=list(_REGION_WEIGHTS.values()))
    channel = rng.choice(["Broker", "Direct"], size=n,
                         p=([0.72, 0.28] if spec.portfolio_type == "direct"
                            else [0.88, 0.12]))
    purpose = rng.choice(["RMRT", "RENV", "EQRE"], size=n, p=[0.45, 0.34, 0.21])
    property_type = rng.choice(["RHOS", "RFLT", "RBGL"], size=n, p=[0.74, 0.14, 0.12])

    return pd.DataFrame({
        "loan_identifier": [f"{spec.portfolio_id.upper()}-{i:06d}"
                            for i in range(1, n + 1)],
        "origination_date": origination,
        "advance": advance,
        "origination_ltv": origination_ltv,
        "valuation_at_origination": valuation_at_origination,
        "rate": rate,
        "age_at_origination": age_at_origination,
        "region": regions,
        "origination_channel": channel,
        "purpose": purpose,
        "property_type": property_type,
        "source_portfolio_id": spec.portfolio_id,
        "source_portfolio_type": spec.portfolio_type,
        "source_portfolio_label": spec.label,
        "acquisition_date": spec.acquisition_date or "",
    })


#: Annual house-price index growth, applied monthly.
_HPI_ANNUAL = 0.031


def _snapshot(base: pd.DataFrame, reporting: date) -> pd.DataFrame:
    """One reporting date's canonical rows, DERIVED from each loan's economics.

    Balance rolls up at the loan's own rate and the valuation indexes, so LTV
    always reconciles to the two of them and month-on-month movement is a
    property of the data rather than a manufactured delta. A loan originated
    after the reporting date is simply not in that snapshot, which is what makes
    the loan count grow month on month.
    """
    origination = pd.to_datetime(base["origination_date"])
    months = ((reporting.year - origination.dt.year) * 12
              + (reporting.month - origination.dt.month))
    live = months >= 0
    frame = base[live].copy()
    months = months[live].astype(int)

    monthly_rate = frame["rate"].to_numpy() / 100.0 / 12.0
    balance = np.round(frame["advance"].to_numpy()
                       * np.power(1.0 + monthly_rate, months.to_numpy()), 2)
    hpi_monthly = (1.0 + _HPI_ANNUAL) ** (1.0 / 12.0)
    valuation = np.round(frame["valuation_at_origination"].to_numpy()
                         * np.power(hpi_monthly, months.to_numpy()), 2)
    ltv = balance / valuation * 100.0
    age = frame["age_at_origination"].to_numpy() + (months.to_numpy() // 12)

    reporting_iso = reporting.isoformat()
    regions = frame["region"].to_numpy()
    out = pd.DataFrame({
        "account_status": "Active",
        "allocated_losses": 0.0,
        "amortisation_type": "Interest roll-up",
        "arrears_balance": 0.0,
        "borrower_identifier": "",
        "collateral_geography": regions,
        "collateral_type": "Residential property",
        "credit_impaired_obligor": "N",
        "cumulative_recoveries": 0.0,
        "current_interest_rate": frame["rate"].to_numpy(),
        "current_principal_balance": balance,
        "current_valuation_amount": valuation,
        "current_valuation_method": "Full survey",
        "data_cut_off_date": reporting_iso,
        "default_amount": 0.0,
        "exposure_currency_denomination": "GBP",
        "interest_rate_type": "FXRL",
        "litigation": "N",
        "loan_identifier": frame["loan_identifier"].to_numpy(),
        "maturity_date": "",
        "new_collateral_identifier": "",
        "number_of_days_in_arrears": 0,
        "occupancy_type": "Owner Occupied",
        "original_collateral_identifier": "",
        "original_obligor_identifier": "",
        "original_principal_balance": frame["advance"].to_numpy(),
        "original_underlying_exposure_identifier": "",
        "original_valuation_amount": frame["valuation_at_origination"].to_numpy(),
        "origination_channel": frame["origination_channel"].to_numpy(),
        "origination_date": frame["origination_date"].to_numpy(),
        "originator_legal_entity_identifier": "SYNTHETICTEST0000000",
        "originator_name": DISPLAY_NAME,
        "payment_due": 0.0,
        "pool_addition_date": frame["acquisition_date"].to_numpy(),
        "prepayment_fee": 0.0,
        "property_post_code": [f"{_POSTCODE_PREFIX[r]}{(i % 40) + 1} {(i % 9) + 1}"
                               f"{'ABDEFGHJ'[i % 8]}{'LNPQRSTU'[i % 8]}"
                               for i, r in enumerate(regions)],
        "property_type": frame["property_type"].to_numpy(),
        "purpose": frame["purpose"].to_numpy(),
        "redemption_date": "",
        "resident": "Y",
        "total_credit_limit": np.round(frame["advance"].to_numpy() * 1.1, 2),
        "youngest_borrower_age": age,
        "current_outstanding_balance": balance,
        "accrued_interest": "",
        "current_loan_to_value": ltv,
        "original_loan_to_value": frame["origination_ltv"].to_numpy(),
        "geographic_region_obligor": [_ITL3[r] for r in regions],
        "geographic_region_collateral": [_ITL3[r] for r in regions],
        "geographic_region_classification": "2021",
        "geographic_region_obligor_itl3": [_ITL3[r] for r in regions],
        "geographic_region_collateral_itl3": [_ITL3[r] for r in regions],
        "geographic_region_classification_source":
            "configured_nuts_classification_year",
        "originator_establishment_country": "GB",
        "source_portfolio_id": frame["source_portfolio_id"].to_numpy(),
        "source_portfolio_type": frame["source_portfolio_type"].to_numpy(),
        "source_portfolio_label": frame["source_portfolio_label"].to_numpy(),
        "acquisition_date": frame["acquisition_date"].to_numpy(),
        "seller_name": np.where(
            frame["source_portfolio_type"].to_numpy() == "acquired",
            SELLER_NAME, ""),
        "portfolio_cohort": frame["source_portfolio_id"].to_numpy(),
        "platform_loan_key": (frame["source_portfolio_id"].astype(str) + "/"
                              + frame["loan_identifier"].astype(str)).to_numpy(),
        "postcode": "",
    })
    return out.reset_index(drop=True)


def _inject_nulls(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Realistic missing values — a book without them tests nothing about them."""
    n = len(frame)
    missing_valuation = rng.choice(n, size=min(207, n // 40), replace=False)
    frame.loc[missing_valuation, "current_valuation_amount"] = np.nan
    frame.loc[missing_valuation, "current_loan_to_value"] = np.nan
    missing_region = rng.choice(n, size=min(72, n // 120), replace=False)
    frame.loc[missing_region, "collateral_geography"] = ""
    return frame


def build(root: Path) -> Dict[str, object]:
    """Write the whole second book under ``root``. Returns its truth manifest.

    ``root`` becomes ``TRAKT_LOCAL_BLOB_ROOT``; the platform canonical lands at
    ``processed/platform/<client>/<date>/`` exactly as the assembler writes it.
    """
    rng = np.random.default_rng(SEED)
    latest = date.fromisoformat(REPORTING_DATES[-1])
    base = pd.concat([_build_portfolio(spec, rng, latest) for spec in PORTFOLIOS],
                     ignore_index=True)

    platform_dir = Path(root) / "processed" / "platform" / CLIENT_ID
    truth: Dict[str, object] = {"client_id": CLIENT_ID, "snapshots": {}}
    latest_frame = None
    for iso in REPORTING_DATES:
        frame = _snapshot(base, date.fromisoformat(iso))
        frame = _inject_nulls(frame, np.random.default_rng(SEED + int(iso[:4])))
        folder = platform_dir / iso
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / "platform_canonical_typed.csv", index=False)
        total = float(pd.to_numeric(frame["current_outstanding_balance"],
                                    errors="coerce").sum())
        (folder / "platform_canonical_manifest.json").write_text(json.dumps({
            "artifact": "platform_canonical",
            "synthetic_notice": SYNTHETIC_NOTICE,
            "composite_key": "source_portfolio_id + loan_identifier",
            "composite_key_column": "platform_loan_key",
            "portfolio_count": len(PORTFOLIOS),
            "total_rows": int(len(frame)),
            "output_total_balance": round(total, 2),
            "reporting_date": iso,
            "portfolios": [{
                "source_portfolio_id": spec.portfolio_id,
                "source_portfolio_type": spec.portfolio_type,
                "source_portfolio_label": spec.label,
                "snapshot_date": iso,
                "row_count": int((frame["source_portfolio_id"]
                                  == spec.portfolio_id).sum()),
            } for spec in PORTFOLIOS],
        }, indent=2), encoding="utf-8")
        truth["snapshots"][iso] = {"rows": int(len(frame)),
                                   "balance": round(total, 2)}
        latest_frame = frame

    latest_dir = platform_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    latest_frame.to_csv(latest_dir / "platform_canonical_typed.csv", index=False)
    (latest_dir / "platform_canonical_manifest.json").write_text(
        (platform_dir / REPORTING_DATES[-1] / "platform_canonical_manifest.json")
        .read_text(encoding="utf-8"), encoding="utf-8")
    (latest_dir / "latest_platform_canonical.json").write_text(json.dumps({
        "synthetic_notice": SYNTHETIC_NOTICE,
        "client_id": CLIENT_ID,
        "reporting_date": REPORTING_DATES[-1],
        "period": REPORTING_DATES[-1][:7],
        "run_id": REPORTING_DATES[-1],
        "platform_canonical": "platform_canonical_typed.csv",
        "source_dated_cut": f"blob://processed/platform/{CLIENT_ID}/{REPORTING_DATES[-1]}",
    }, indent=2), encoding="utf-8")

    registry = Path(root) / "portfolio_registry.yaml"
    registry.write_text(
        "portfolios:\n" + "".join(
            f"- source_portfolio_id: {spec.portfolio_id}\n"
            f"  asset_class: equity_release\n" for spec in PORTFOLIOS),
        encoding="utf-8")
    truth["registry"] = str(registry)
    truth["platform_prefix"] = f"blob://processed/platform/{CLIENT_ID}"
    return truth


def mi_env(root: Path) -> Dict[str, str]:
    """The environment that points the MI read path at this book."""
    return {
        "TRAKT_STORAGE_BACKEND": "file",
        "TRAKT_LOCAL_BLOB_ROOT": str(root),
        "MI_AGENT_PLATFORM_URI":
            f"blob://processed/platform/{CLIENT_ID}/latest",
        "MI_AGENT_ONBOARDING_OUTPUT_ROOT": f"blob://processed/platform/{CLIENT_ID}",
        "MI_AGENT_CLIENT_ID": CLIENT_ID,
        "MI_AGENT_RUN_ID": REPORTING_DATES[-1],
        "MI_AGENT_REPORTING_DATE": REPORTING_DATES[-1],
        "TRAKT_PORTFOLIO_REGISTRY": str(Path(root) / "portfolio_registry.yaml"),
        "MI_AGENT_LLM_ENABLED": "0",
        "MI_AGENT_AUTH_ENABLED": "false",
        "MI_AGENT_DATA_CACHE_TTL": "0",
    }
