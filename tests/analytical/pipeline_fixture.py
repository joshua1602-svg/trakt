"""tests/analytical/pipeline_fixture.py — a governed weekly pipeline pack.

Three of the nine CFO acceptance questions read the PIPELINE (offer stock,
expected completions, forecast funded balance). The demonstration platform
builds a funded book only, so without a pipeline those three could only ever be
observed refusing.

This builds a governed weekly extract pack in the raw M2L KFI / pipeline source
format — the same shape the production discovery layer already reads — so the
real preparation, probability, timing and forecast engines run end to end.
Nothing here computes a completion probability, an expected amount or a
completion month: those all come from ``mi_agent_api.pipeline_prep`` and
``config/client/pipeline_expected_funding.yaml``, exactly as they do in
production.

Deterministic by construction: a fixed seed, no wall-clock, no randomness that
is not reproducible from ``(seed, week)``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HEADER: Tuple[str, ...] = (
    "Company", "Pool", "Account Number", "KFI Number", "Broker",
    "KFI Submitted Date", "DOB App 1", "Gender APP 1", "DOB App 2",
    "Gender APP 2", "Loan Amount", "Estimated Value", "Product", "Product Rate",
    "Loan Plan", "Facility", "Max Facility", "Max Entitlement",
    "Property Region", "PEG Percentage", "Fees Added", "Property Value",
    "Loan Purpose", "Loan Purpose Detail", "Status", "DPR Status",
    "Application Submitted Date", "Offer Date", "Date Funds Released",
    "Rejection Reason A", "Rejection Reason B", "KFI Used For App",
    "Contracted Payment Period", "Interest Payment Percentage",
)

#: The funnel a case walks, and how many weeks it typically spends in each.
_STAGES = ("KFI", "APPLICATION", "OFFER", "COMPLETED")


@dataclass(frozen=True)
class BookProfile:
    """The economics of the book the pipeline feeds.

    Deliberately parameterised: a pipeline that only makes sense beside one
    fixture would prove nothing about generalisation.
    """

    client: str
    company: str
    #: Cases created per week.
    weekly_cases: int
    #: Mean advance, and the spread around it.
    mean_advance: float
    advance_spread: float
    #: Mean property value multiple of the advance.
    value_multiple: float
    regions: Sequence[str]
    brokers: Sequence[str]
    products: Sequence[Tuple[str, float]]
    seed: int


ALDERBRIDGE = BookProfile(
    client="alderbridge", company="Alderbridge Lending Platform (synthetic)",
    weekly_cases=42, mean_advance=185_000.0, advance_spread=95_000.0,
    value_multiple=2.6,
    regions=("South East", "London", "South West", "East of England",
             "North West", "West Midlands", "Yorkshire and The Humber",
             "East Midlands", "Scotland", "Wales"),
    brokers=("Aldermoor Advisory", "Bramwell Financial", "Coldharbour Wealth",
             "Dunsfold Later Life", "Eastgate Retirement"),
    products=(("Lifetime Mortgage Lump Sum", 6.45),
              ("Lifetime Mortgage Drawdown", 6.72),
              ("Lifetime Mortgage Enhanced", 6.18)),
    seed=20260630)

#: A deliberately different pipeline: smaller tickets, a broker-heavy mix, a
#: slower funnel and a different geography. A capability that only works on the
#: first profile is not accepted.
KESTRELMOOR = BookProfile(
    client="kestrelmoor", company="Kestrelmoor Lifetime Lending (synthetic)",
    weekly_cases=61, mean_advance=104_000.0, advance_spread=58_000.0,
    value_multiple=3.1,
    regions=("North West", "Scotland", "Yorkshire and The Humber",
             "North East", "West Midlands", "Wales", "East Midlands",
             "South West", "London", "South East"),
    brokers=("Kestrel Later Life", "Moorgate Equity", "Thornwood Advisory",
             "Meridian Retirement"),
    products=(("Lifetime Mortgage Drawdown", 7.05),
              ("Lifetime Mortgage Lump Sum", 6.88),
              ("Lifetime Mortgage Enhanced", 7.34)),
    seed=20260628)


def _lcg(state: int) -> Tuple[int, float]:
    """A tiny deterministic generator — reproducible without importing RNG state."""
    state = (1103515245 * state + 12345) % (2 ** 31)
    return state, state / float(2 ** 31)


def _cases(profile: BookProfile, weeks: Sequence[date]) -> List[Dict[str, str]]:
    """Every case the pack contains, with the week it entered the funnel."""
    state = profile.seed
    rows: List[Dict[str, str]] = []
    case_no = 1
    for cohort_index, week in enumerate(weeks):
        for _ in range(profile.weekly_cases):
            state, r1 = _lcg(state)
            state, r2 = _lcg(state)
            state, r3 = _lcg(state)
            state, r4 = _lcg(state)
            state, r5 = _lcg(state)
            advance = round(profile.mean_advance
                            + (r1 - 0.5) * 2 * profile.advance_spread, 2)
            advance = max(28_000.0, advance)
            value = round(advance * (profile.value_multiple * (0.85 + r2 * 0.4)), 2)
            product, base_rate = profile.products[int(r3 * len(profile.products))
                                                  % len(profile.products)]
            rows.append({
                "case": f"{profile.client[:3].upper()}P{case_no:05d}",
                "kfi": f"KFI{case_no:06d}",
                "cohort": cohort_index,
                "advance": advance,
                "value": value,
                "product": product,
                "rate": round(base_rate + (r4 - 0.5) * 0.6, 3),
                "region": profile.regions[int(r5 * len(profile.regions))
                                          % len(profile.regions)],
                "broker": profile.brokers[case_no % len(profile.brokers)],
                "dob": f"{1946 + (case_no % 18)}-{1 + (case_no % 12):02d}-"
                       f"{1 + (case_no % 27):02d}",
                "kfi_date": week,
            })
            case_no += 1
    return rows


#: Weeks a case spends in each stage before advancing. A case that has not yet
#: accumulated the weeks for the next stage is still in the current one.
_WEEKS_TO_STAGE = {"APPLICATION": 3, "OFFER": 6, "COMPLETED": 10}


def _stage_at(case: Dict[str, str], week_index: int) -> str:
    """The funnel stage a case has reached by ``week_index``.

    A pure function of the case's cohort and a per-case pace, so the same case
    is at the same stage in every extract that observes the same week — which is
    what lets the historical completion model track it across snapshots.
    """
    elapsed = week_index - int(case["cohort"])
    pace = int(case["case"][-2:]) % 4        # 0-3 extra weeks, stable per case
    if elapsed >= _WEEKS_TO_STAGE["COMPLETED"] + pace:
        return "COMPLETED"
    if elapsed >= _WEEKS_TO_STAGE["OFFER"] + pace:
        return "OFFER"
    if elapsed >= _WEEKS_TO_STAGE["APPLICATION"] + pace:
        return "APPLICATION"
    return "KFI"


_STATUS_TEXT = {"KFI": "KFI Issued", "APPLICATION": "Application",
                "OFFER": "Offer Issued", "COMPLETED": "Completed"}


def write_pack(root: Path, profile: BookProfile, *, weeks: int = 12,
               last_week: date) -> List[Path]:
    """Write ``weeks`` weekly extracts ending at ``last_week``. Returns the paths.

    Each extract is a full snapshot of every case that had entered the funnel by
    that week — the operational shape the production discovery layer expects, and
    what the historical completion model replays to derive empirical stage rates.
    """
    week_dates = [last_week - timedelta(weeks=(weeks - 1 - i))
                  for i in range(weeks)]
    cases = _cases(profile, week_dates)
    written: List[Path] = []
    for index, week in enumerate(week_dates):
        folder = root / profile.client / "pipeline" / week.isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ("M2L_KFI_and_Pipeline_"
                         f"{week.strftime('%Y_%m_%d')}.csv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            for case in cases:
                if int(case["cohort"]) > index:
                    continue                      # not yet in the funnel
                stage = _stage_at(case, index)
                kfi_date = case["kfi_date"]
                app_date = (kfi_date + timedelta(weeks=3)
                            if stage in ("APPLICATION", "OFFER", "COMPLETED") else "")
                offer_date = (kfi_date + timedelta(weeks=6)
                              if stage in ("OFFER", "COMPLETED") else "")
                funds_date = (kfi_date + timedelta(weeks=10)
                              if stage == "COMPLETED" else "")
                writer.writerow([
                    profile.company, "POOL_A", case["case"], case["kfi"],
                    case["broker"], kfi_date.isoformat(), case["dob"], "F",
                    "", "", f"{case['advance']:.2f}", f"{case['value']:.2f}",
                    case["product"], f"{case['rate']:.3f}", "Plan A",
                    f"{case['advance']:.2f}", f"{case['advance'] * 1.3:.2f}",
                    f"{case['advance'] * 1.2:.2f}", case["region"], "25.0",
                    "1195.00", f"{case['value']:.2f}", "Home Improvements",
                    "Property works", _STATUS_TEXT[stage], "Approved",
                    app_date.isoformat() if app_date else "",
                    offer_date.isoformat() if offer_date else "",
                    funds_date.isoformat() if funds_date else "",
                    "", "", "Yes", "12", "0",
                ])
        written.append(path)
    return written
