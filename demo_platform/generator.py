"""demo_platform.generator — the synthetic UK lifetime-mortgage source generator.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

Builds two internally consistent loan books for the fictional Alderbridge
Lending Platform and writes each one out, per reporting month, in its OWN raw
source schema (:mod:`demo_platform.schemas`).

Design
------
The books are modelled, not sampled. Each loan is generated once with its
completion date, region, property valuation, initial advance, fixed roll-up rate
and youngest-borrower age; every reported figure at a given month-end is then
*derived* from that loan's own economics:

    balance(d)    = advance · (1 + r/12)^m   [+ drawdown · (1 + r/12)^m_d]
    valuation(d)  = original valuation · (1 + hpi)^m
    LTV(d)        = balance(d) / valuation(d)
    age(d)        = age at completion + whole years elapsed

so LTV always reconciles to balance and valuation, month-on-month balances
always reconcile to the engineered movement, and a loan's age moves
directionally with the reporting date. Nothing is written twice from two
different assumptions.

Calibration
-----------
The narrative the film states (a ~£18.2m monthly increase, ~43.1% weighted LTV,
~71.4 years average age, the South East as the primary driver) is produced by
*engineering the source data*, never by patching an output. A deterministic
calibration solves four scalars — an initial-LTV shift, an origination-age
shift, Portfolio A's new-business volume and Portfolio B's drawdown volume —
against the same formulas the pipeline applies, and records the solved values in
the demo manifest. The real pipeline output is then independently re-verified by
:mod:`demo_platform.assertions`, which fails the run on any drift.

Everything is seeded from :data:`demo_platform.config.RANDOM_SEED`, so two runs
of the same code produce byte-identical files.
"""

from __future__ import annotations

import calendar
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config as cfg
from . import schemas

# --------------------------------------------------------------------------- #
# Small date helpers (month-end arithmetic only — no timezone concerns)
# --------------------------------------------------------------------------- #
def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _months_between(start: date, end: date) -> int:
    """Whole months from ``start`` to ``end`` (never negative)."""
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))


def _add_months(d: date, months: int) -> date:
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


# --------------------------------------------------------------------------- #
# Postcode pool, drawn from the in-repo ITL master lookup
# --------------------------------------------------------------------------- #
_POSTCODE_POOL: Optional[Dict[str, List[str]]] = None


def postcode_pool() -> Dict[str, List[str]]:
    """``{clean ITL1 region label: [postcode districts]}``.

    Districts are read from the repository's own ``uk_itl_master_lookup_v2.csv``
    and filtered to the ITL1 region, so a loan's postcode is genuinely inside the
    region it reports. The transform's postcode → ITL3 enrichment then derives an
    ITL3 code that agrees with the readable label — the two geography views
    reconcile by construction rather than by assertion.
    """
    global _POSTCODE_POOL
    if _POSTCODE_POOL is not None:
        return _POSTCODE_POOL
    lookup = cfg.REPO_ROOT / "uk_itl_master_lookup_v2.csv"
    df = pd.read_csv(lookup, dtype=str).fillna("")
    pool: Dict[str, List[str]] = {}
    for clean, lookup_name in cfg.ITL1_LOOKUP_NAMES.items():
        sub = df[df["itl1_name"].str.strip() == lookup_name]
        districts = sorted({str(p).strip().upper() for p in sub["postcode_prefix"] if str(p).strip()})
        if not districts:  # pragma: no cover - configuration error
            raise ValueError(
                f"No postcode districts found for ITL1 region {lookup_name!r} in "
                f"{lookup.name}; check config.ITL1_LOOKUP_NAMES."
            )
        pool[clean] = districts
    _POSTCODE_POOL = pool
    return pool


def _postcode(rng: np.random.Generator, region: str) -> str:
    """A plausible full UK postcode inside ``region``."""
    districts = postcode_pool()[region]
    district = districts[int(rng.integers(0, len(districts)))]
    sector = int(rng.integers(1, 10))
    letters = "ABDEFGHJLNPQRSTUWXYZ"
    a = letters[int(rng.integers(0, len(letters)))]
    b = letters[int(rng.integers(0, len(letters)))]
    return f"{district} {sector}{a}{b}"


# --------------------------------------------------------------------------- #
# Calibration parameters
# --------------------------------------------------------------------------- #
@dataclass
class Calibration:
    """The scalars the calibration solves. All start at a neutral value."""

    #: Added to both portfolios' initial-LTV mean (fraction of valuation).
    ltv_shift: float = 0.0
    #: Added to both portfolios' origination-age mean (years).
    age_shift: float = 0.0
    #: Multiplies Portfolio A's current-month new-business VOLUME (loan count).
    #: Volume is the lever a lender's month-on-month movement actually reflects,
    #: so calibrating it keeps every individual loan's economics untouched.
    completion_scale_a: float = 1.0
    #: Multiplies Portfolio B's current-month drawdown tranche size.
    drawdown_scale_b: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {k: round(float(v), 8) for k, v in asdict(self).items()}


# --------------------------------------------------------------------------- #
# The loan universe
# --------------------------------------------------------------------------- #
@dataclass
class LoanBook:
    """One portfolio's complete generated loan universe (all history)."""

    portfolio: cfg.PortfolioSpec
    shape: cfg.PortfolioShape
    loans: pd.DataFrame

    @property
    def total_loans(self) -> int:
        return int(len(self.loans))


def _region_draw(rng: np.random.Generator, n: int, weights: Dict[str, float]) -> np.ndarray:
    regions = list(weights)
    p = np.array([weights[r] for r in regions], dtype=float)
    p = p / p.sum()
    return rng.choice(regions, size=n, p=p)


def _build_loans(
    portfolio: cfg.PortfolioSpec,
    shape: cfg.PortfolioShape,
    calib: Calibration,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate the full loan universe for one portfolio.

    Composition, oldest → newest:
      * a legacy cohort completed before the first reporting period;
      * one new-completion cohort per reporting period.

    The legacy cohort is sized so that, after the engineered redemptions across
    the three periods, the active loan count at the CURRENT reporting date lands
    on ``portfolio.target_current_loans``.
    """
    completions = list(shape.monthly_completions)
    redemptions = list(shape.monthly_redemptions)
    # Calibrated new-business volume applies to the CURRENT month only; earlier
    # months keep their configured volume so the prior-period comparison the film
    # makes is a genuine month-on-month change, not a calibration artefact.
    current_idx = next(i for i, p in enumerate(cfg.PERIODS) if p.role == "current")
    if completions[current_idx] > 0:
        completions[current_idx] = max(
            1, int(round(completions[current_idx] * calib.completion_scale_a)))
    n_legacy = portfolio.target_current_loans + sum(redemptions) - sum(completions)
    if n_legacy <= 0:  # pragma: no cover - configuration error
        raise ValueError(f"{portfolio.display_id}: legacy cohort would be empty.")

    rows: List[Dict[str, Any]] = []
    seq = 0

    def _emit(n: int, completion_dates: np.ndarray, region_weights: Dict[str, float],
              advance_scale: float) -> None:
        nonlocal seq
        if n <= 0:
            return
        regions = _region_draw(rng, n, region_weights)
        # Property valuation at completion: national base × regional index,
        # lognormal-ish spread so the tail is plausible rather than symmetric.
        base = rng.normal(shape.base_valuation, shape.base_valuation_sd, n)
        index = np.array([cfg.REGION_VALUE_INDEX[r] for r in regions])
        valuation = np.clip(base * index, 95_000.0, 4_500_000.0)
        valuation = np.round(valuation / 1_000.0) * 1_000.0
        # Initial advance as a fraction of valuation, bounded by the product's
        # age-related maximum release (older borrower -> larger release).
        ages = rng.normal(shape.origination_age_mean + calib.age_shift,
                          shape.origination_age_sd, n)
        ages = np.clip(np.round(ages), 55, 92).astype(int)
        # Age-related maximum release, mirroring how UK lifetime-mortgage product
        # tables work: a 55-year-old can release ~18% of the property value, an
        # 85-year-old ~57%.
        max_release = np.clip(0.180 + (ages - 55) * 0.0145, 0.180, 0.600)
        ltv = rng.normal(shape.initial_ltv_mean + calib.ltv_shift, shape.initial_ltv_sd, n)
        ltv = np.clip(ltv, 0.075, 1.0) * advance_scale
        ltv = np.minimum(ltv, max_release)
        advance = np.round(valuation * ltv / 100.0) * 100.0
        advance = np.clip(advance, 12_000.0, None)
        rates = np.clip(rng.normal(shape.rate_mean, shape.rate_sd, n), 2.60, 9.40)
        rates = np.round(rates, 3)
        # Property-type vocabulary drawn from the values the repository's own enum
        # library already recognises (config/system/enum_synonyms.yaml →
        # property_type), so the canonical enum check passes without touching
        # production enum configuration.
        prop_types = rng.choice(
            ["Detached house", "Semi-detached house", "House", "Bungalow", "Flat"],
            size=n, p=[0.32, 0.28, 0.19, 0.15, 0.06])
        for i in range(n):
            seq += 1
            rows.append({
                "loan_seq": seq,
                "loan_id": f"{portfolio.display_id}-{seq:06d}",
                "region": str(regions[i]),
                "postcode": _postcode(rng, str(regions[i])),
                "origination_date": completion_dates[i],
                "original_valuation": float(valuation[i]),
                "original_principal": float(advance[i]),
                "rate": float(rates[i]),
                "age_at_origination": int(ages[i]),
                "property_type": str(prop_types[i]),
                "redeem_period": None,
                "drawdown_period": None,
                "drawdown_amount": 0.0,
            })

    # -- Legacy cohort ---------------------------------------------------- #
    lo, hi = shape.vintage_range
    first_period_start = date(cfg.PERIODS[0].reporting_date and int(cfg.PERIODS[0].reporting_date[:4]),
                              int(cfg.PERIODS[0].reporting_date[5:7]), 1)
    legacy_dates = []
    span_months = max(1, (first_period_start.year - lo) * 12 + first_period_start.month - 1)
    offsets = rng.integers(0, span_months, n_legacy)
    for off in offsets:
        d = _add_months(date(lo, 1, 1), int(off))
        day = int(rng.integers(1, calendar.monthrange(d.year, d.month)[1] + 1))
        legacy_dates.append(date(d.year, d.month, day))
    _emit(n_legacy, np.array(legacy_dates, dtype=object), cfg.REGION_WEIGHTS, 1.0)

    # -- New-completion cohorts, one per reporting period ------------------ #
    for idx, prd in enumerate(cfg.PERIODS):
        n = completions[idx]
        if n <= 0:
            continue
        year, month = int(prd.reporting_date[:4]), int(prd.reporting_date[5:7])
        last = calendar.monthrange(year, month)[1]
        days = rng.integers(1, last + 1, n)
        dates = np.array([date(year, month, int(d)) for d in days], dtype=object)
        # Only the CURRENT month's volume is calibrated; earlier months keep the
        # configured shape so the film's prior-period comparison is untouched.
        scale = calib.completion_scale_a if prd.role == "current" else 1.0
        _emit(n, dates, cfg.COMPLETION_REGION_WEIGHTS, scale)

    loans = pd.DataFrame(rows)

    # -- Redemptions: assign a redeeming month per period ------------------ #
    # Redemptions come from the seasoned part of the book (mortality / long-term
    # care / voluntary repayment), so they are drawn from loans completed before
    # the reporting periods began, weighted toward the oldest borrowers.
    eligible = loans.index[loans["origination_date"] < date(
        int(cfg.PERIODS[0].reporting_date[:4]), int(cfg.PERIODS[0].reporting_date[5:7]), 1)]
    age_weights = loans.loc[eligible, "age_at_origination"].to_numpy(dtype=float)
    age_weights = np.maximum(age_weights - 52.0, 1.0) ** 2.2
    taken: set = set()
    for idx, prd in enumerate(cfg.PERIODS):
        n = redemptions[idx]
        if n <= 0:
            continue
        mask = np.array([i not in taken for i in eligible])
        pool = eligible[mask]
        w = age_weights[mask]
        w = w / w.sum()
        chosen = rng.choice(pool, size=min(n, len(pool)), replace=False, p=w)
        for i in chosen:
            taken.add(int(i))
            loans.at[i, "redeem_period"] = prd.period

    # -- Reserve-facility drawdowns (further advances) --------------------- #
    if any(shape.monthly_drawdown_loans):
        drawn: set = set()
        for idx, prd in enumerate(cfg.PERIODS):
            n = shape.monthly_drawdown_loans[idx]
            if n <= 0:
                continue
            # A loan can draw only if it is still on book at that reporting date
            # and has not already drawn in an earlier month of the demo window.
            avail = [
                int(i) for i in loans.index
                if int(i) not in drawn
                and loans.at[i, "origination_date"] < date(
                    int(prd.reporting_date[:4]), int(prd.reporting_date[5:7]), 1)
                and (loans.at[i, "redeem_period"] is None
                     or str(loans.at[i, "redeem_period"]) > prd.period)
            ]
            chosen = rng.choice(np.array(avail), size=min(n, len(avail)), replace=False)
            scale = calib.drawdown_scale_b if prd.role == "current" else 1.0
            amounts = np.clip(
                rng.normal(shape.drawdown_mean * scale, shape.drawdown_sd, len(chosen)),
                5_000.0, 250_000.0)
            amounts = np.round(amounts / 100.0) * 100.0
            for i, amt in zip(chosen, amounts):
                drawn.add(int(i))
                loans.at[int(i), "drawdown_period"] = prd.period
                loans.at[int(i), "drawdown_amount"] = float(amt)

    return loans


# --------------------------------------------------------------------------- #
# Point-in-time projection
# --------------------------------------------------------------------------- #
def period_frame(book: LoanBook, prd: cfg.PeriodSpec) -> pd.DataFrame:
    """The model-space view of ``book`` as at ``prd``'s month-end.

    Contains every loan on the servicer's book at the reporting date: loans that
    are still active, plus loans that redeemed *during that reporting month*
    (status Redeemed, nil balance, redemption date populated) — which is how a
    real monthly extract behaves. Loans that redeemed in an earlier month have
    left the book and do not appear.
    """
    loans = book.loans
    shape = book.shape
    rd = date(int(prd.reporting_date[:4]), int(prd.reporting_date[5:7]),
              int(prd.reporting_date[8:10]))

    orig = loans["origination_date"]
    on_book = np.array([o <= rd for o in orig])
    redeem = loans["redeem_period"].astype(object)
    not_gone = np.array([r is None or str(r) >= prd.period for r in redeem])
    sub = loans[on_book & not_gone].copy()

    months = np.array([_months_between(o, rd) for o in sub["origination_date"]], dtype=float)
    monthly_factor = 1.0 + sub["rate"].to_numpy(dtype=float) / 1200.0

    # Capital plus capitalised roll-up interest.
    balance = sub["original_principal"].to_numpy(dtype=float) * monthly_factor ** months

    # A reserve drawdown taken in month D also rolls up from D to the cut-off.
    dd_period = sub["drawdown_period"].astype(object).to_numpy()
    dd_amount = sub["drawdown_amount"].to_numpy(dtype=float)
    for i in range(len(sub)):
        p = dd_period[i]
        if p is None or float(dd_amount[i]) <= 0.0 or str(p) > prd.period:
            continue
        dd_end = _month_end(int(str(p)[:4]), int(str(p)[5:7]))
        m_d = float(_months_between(dd_end, rd))
        balance[i] += float(dd_amount[i]) * monthly_factor[i] ** m_d

    valuation = (sub["original_valuation"].to_numpy(dtype=float)
                 * (1.0 + shape.monthly_hpi) ** months)

    redeemed_now = np.array([r is not None and str(r) == prd.period
                             for r in sub["redeem_period"]])

    loan_ids = sub["loan_id"].to_numpy()
    book_prefix = book.portfolio.display_id
    # Use of proceeds, deterministic per loan (no RNG here — period_frame must be
    # a pure projection of the universe, so two runs agree row for row).
    _PURPOSES = ("Equity Release", "Home improvement", "Refinance")
    purposes = [_PURPOSES[int(str(x)[-4:]) % len(_PURPOSES)] for x in loan_ids]

    # --- Regulatory-reporting attributes -----------------------------------
    # A lender that reports under ESMA Annex 2 carries these on its own tape;
    # without them the exposure-level submission cannot be produced. They are
    # modelled from the same loan economics as everything else, so the
    # submission reconciles to the MI.
    #
    # Lifetime mortgages have no contractual repayment date: they redeem on the
    # last life's death or move into long-term care. The reported maturity is
    # therefore the product long-stop — the date the youngest borrower reaches
    # 110, which is how the book's own duration model bounds each loan.
    ages_at_origination = sub["age_at_origination"].to_numpy(dtype=int)
    maturity_dates = [
        _month_end(o.year + int(110 - a), o.month).isoformat()
        for o, a in zip(sub["origination_date"], ages_at_origination)
    ]
    # The date the loan entered the reported pool. For the origination book that
    # is completion; for the acquired book it is the later of completion and the
    # date the acquisition itself completed.
    acquired_on = book.portfolio.acquisition_date
    pool_addition_dates = [
        o.isoformat() if acquired_on is None or o.isoformat() >= acquired_on
        else acquired_on
        for o in sub["origination_date"]
    ]
    # Approved facility: the initial advance plus the loan's reserve facility.
    # The reserve is the larger of the drawdowns the loan actually took and a
    # 10% headroom, rounded up to the nearest £100, so the reported limit is
    # never below what was drawn.
    original_principal = sub["original_principal"].to_numpy(dtype=float)
    reserve = np.maximum(dd_amount, original_principal * 0.10)
    credit_limit = np.ceil((original_principal + reserve) / 100.0) * 100.0
    # ESMA Annex 2 asks for the collateral identifier as it stood at
    # securitisation (RREC3) as well as the current one (RREC4). The origination
    # book has never re-keyed its collateral, so the two agree. The acquired book
    # was re-keyed onto the servicer's numbering when it transferred, so it
    # carries the vendor's original key alongside the current one.
    collateral_ids = [str(x).replace(book_prefix, "COL", 1) for x in loan_ids]
    if book.portfolio.source_portfolio_type == "acquired":
        collateral_ids_original = [str(x).replace(book_prefix, "VND", 1)
                                   for x in loan_ids]
    else:
        collateral_ids_original = collateral_ids

    out = pd.DataFrame({
        "loan_id": loan_ids,
        "reporting_date": prd.reporting_date,
        "origination_date": [o.isoformat() for o in sub["origination_date"]],
        "original_principal": np.round(sub["original_principal"].to_numpy(dtype=float), 2),
        "current_balance": np.round(np.where(redeemed_now, 0.0, balance), 2),
        "original_valuation": np.round(sub["original_valuation"].to_numpy(dtype=float), 2),
        "current_valuation": np.round(valuation, 2),
        "rate": sub["rate"].to_numpy(),
        "borrower_age": (sub["age_at_origination"].to_numpy(dtype=int)
                         + (months // 12).astype(int)),
        "region": sub["region"].to_numpy(),
        "postcode": sub["postcode"].to_numpy(),
        "property_type": sub["property_type"].to_numpy(),
        "status": np.where(redeemed_now, "Redeemed", "Active"),
        "redemption_date": [
            _month_end(int(str(r)[:4]), int(str(r)[5:7])).isoformat() if flag else ""
            for r, flag in zip(sub["redeem_period"], redeemed_now)
        ],
        "drawdown_amount": np.round(
            np.where([p is not None and str(p) == prd.period for p in sub["drawdown_period"]],
                     dd_amount, 0.0), 2),
        # Enum values the repository's enum library already recognises.
        # ESMA Annex 2 requires an exposure and an obligor identifier distinct
        # from the servicing loan key (RREL2/RREL4, which back-fill the mandatory
        # RREL3/RREL5/RREC2). A lender preparing for regulatory reporting carries
        # these on its tape, so the generator supplies them.
        "exposure_id": [str(x).replace(book_prefix, "EXP", 1) for x in loan_ids],
        "obligor_id": [str(x).replace(book_prefix, "OBL", 1) for x in loan_ids],
        # Use of proceeds. Drawn from the values the repository's own enum library
        # already recognises (config/system/enum_synonyms.yaml -> purpose), so the
        # Annex 2 enum layer resolves them without touching production config.
        "purpose": purposes,
        "rate_type": "Fixed",
        "amortisation_type": "Interest roll-up",
        "occupancy_type": "Owner Occupied",
        "currency": "GBP",
        # Regulatory-reporting attributes (see above).
        "maturity_date": maturity_dates,
        "pool_addition_date": pool_addition_dates,
        "credit_limit": np.round(credit_limit, 2),
        "collateral_id": collateral_ids,
        "collateral_id_original": collateral_ids_original,
        "collateral_type": "Residential property",
        "valuation_method": "Full survey",
        "origination_channel": shape.origination_channel,
        "resident": "Yes",
        "credit_impaired": "No",
        "litigation": "No",
        # This book is performing: nothing is due, nothing is in arrears, nothing
        # has defaulted, and no loss has been allocated. Reported as measured
        # zeroes rather than as "no data", because zero is the fact.
        "payment_due": 0.0,
        "arrears_balance": 0.0,
        "days_in_arrears": 0,
        "default_amount": 0.0,
        "allocated_losses": 0.0,
        "cumulative_recoveries": 0.0,
        "prepayment_fee": 0.0,
        "synthetic_notice": cfg.SYNTHETIC_BANNER,
    })
    # Redeemed accounts carry no live valuation-based exposure; keep the last
    # known valuation so the tape remains auditable, but the LTV the pipeline
    # derives is nil because the balance is nil.
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Deliberate, bounded data-quality exceptions
# --------------------------------------------------------------------------- #
def _inject_exceptions(frame: pd.DataFrame, prd: cfg.PeriodSpec,
                       portfolio: cfg.PortfolioSpec) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Introduce a small, fixed set of genuine data-quality defects.

    Without these the validation gate has nothing to report, and the film would
    be showing a validation step that cannot fail. Each defect is deterministic
    (index-selected, not random), bounded to a handful of rows, and recorded so
    the demo manifest can state exactly what was injected.
    """
    out = frame.copy()
    n = len(out)
    if n < 200:
        return out, []
    # Blanks are written into otherwise-numeric columns; widen them to object so
    # pandas stores the empty string rather than coercing it.
    for col in ("current_valuation", "original_valuation", "borrower_age"):
        out[col] = out[col].astype(object)
    injected: List[Dict[str, Any]] = []
    # Deterministic, well-separated row positions that differ per period so the
    # exception report changes month to month like a real one does.
    offset = {"opening": 0, "prior": 7, "current": 13}[prd.role]
    picks = [(n // 7) + offset, (2 * n // 7) + offset, (3 * n // 7) + offset,
             (4 * n // 7) + offset, (5 * n // 7) + offset, (6 * n // 7) + offset]
    picks = [p % n for p in picks][:cfg.INJECTED_EXCEPTIONS_PER_PERIOD]

    def _record(row: int, rule: str, detail: str) -> None:
        injected.append({
            "portfolio": portfolio.display_id, "period": prd.period,
            "loan_id": str(out.at[row, "loan_id"]), "rule": rule, "detail": detail,
        })

    r = picks[0]
    out.at[r, "postcode"] = ""
    _record(r, "missing_property_postcode",
            "Property postcode absent — ITL3 geography cannot be derived for this loan.")

    r = picks[1]
    out.at[r, "current_valuation"] = ""
    _record(r, "missing_current_valuation",
            "Current valuation absent — current LTV cannot be derived for this loan.")

    r = picks[2]
    out.at[r, "borrower_age"] = 52
    _record(r, "borrower_age_below_product_minimum",
            "Youngest borrower age 52 is below the product minimum of 55.")

    r = picks[3]
    out.at[r, "region"] = ""
    _record(r, "missing_region_label",
            "Region label absent — the loan falls into the unattributed geography bucket.")

    r = picks[4]
    out.at[r, "original_valuation"] = ""
    _record(r, "missing_original_valuation",
            "Valuation at completion absent — original LTV cannot be derived.")

    return out, injected


# --------------------------------------------------------------------------- #
# Model-space metrics (the same formulas the pipeline applies)
# --------------------------------------------------------------------------- #
def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def model_metrics(frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Metrics computed on the MODEL frames, used only to steer calibration.

    Mirrors the production definitions exactly:
      * funded balance   = Σ current balance
      * WA current LTV   = Σ(LTV·balance)/Σ balance, LTV = balance/valuation
      * average age      = simple mean of the youngest-borrower age
    The authoritative figures always come from the pipeline output via
    :mod:`demo_platform.metrics`; these exist so the generator can converge.
    """
    combined = pd.concat(frames.values(), ignore_index=True)
    bal = _num(combined["current_balance"]).fillna(0.0)
    val = _num(combined["current_valuation"])
    ltv = (bal / val).where(val > 0)
    mask = ltv.notna() & (bal > 0)
    wa_ltv = float((ltv[mask] * bal[mask]).sum() / bal[mask].sum()) * 100.0 if mask.any() else None
    return {
        "funded_balance": float(bal.sum()),
        "loan_count": int(len(combined)),
        "wa_current_ltv_points": wa_ltv,
        "avg_borrower_age": float(_num(combined["borrower_age"]).dropna().mean()),
    }


def _movement(frames_by_period: Dict[str, pd.DataFrame]) -> float:
    cur = _num(frames_by_period[cfg.CURRENT_PERIOD.period]["current_balance"]).fillna(0.0).sum()
    pri = _num(frames_by_period[cfg.PRIOR_PERIOD.period]["current_balance"]).fillna(0.0).sum()
    return float(cur - pri)


# --------------------------------------------------------------------------- #
# Build + calibrate
# --------------------------------------------------------------------------- #
@dataclass
class GeneratedBooks:
    """The calibrated result: books, per-period frames and the solved scalars."""

    books: Dict[str, LoanBook]
    #: ``{source_portfolio_id: {period: frame}}`` after exception injection.
    frames: Dict[str, Dict[str, pd.DataFrame]]
    calibration: Calibration
    injected_exceptions: List[Dict[str, Any]]
    model_metrics: Dict[str, Any]
    iterations: int


def _build_all(calib: Calibration) -> Tuple[Dict[str, LoanBook], Dict[str, Dict[str, pd.DataFrame]]]:
    books: Dict[str, LoanBook] = {}
    frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    for p in cfg.PORTFOLIOS:
        shape = cfg.SHAPES[p.source_portfolio_id]
        # Per-portfolio seed derived from the global seed and the portfolio id, so
        # each book is independent yet fully reproducible.
        seed = cfg.RANDOM_SEED + (sum(ord(c) for c in p.source_portfolio_id) * 7919)
        rng = np.random.default_rng(seed)
        loans = _build_loans(p, shape, calib, rng)
        book = LoanBook(p, shape, loans)
        books[p.source_portfolio_id] = book
        frames[p.source_portfolio_id] = {
            prd.period: period_frame(book, prd) for prd in cfg.PERIODS
        }
    return books, frames


def _build_sponsored(valuation_scale: float) -> Tuple[LoanBook, Dict[str, pd.DataFrame]]:
    """Build SPV1, the sold securitisation, on its own.

    Deliberately outside :func:`_build_all` and outside the platform's calibration.
    SPV1 is derecognised — it is not in the assembled platform canonical and takes no
    part in the movement narrative — so it must not be able to perturb a single figure
    the platform produces. Its own seed, its own solve, one scalar.
    """
    spec = cfg.PORTFOLIO_S
    shape = cfg.SHAPES[spec.source_portfolio_id]
    seed = cfg.RANDOM_SEED + (sum(ord(c) for c in spec.source_portfolio_id) * 7919)
    rng = np.random.default_rng(seed)
    # A neutral calibration: the platform's solved scalars are none of SPV1's business.
    loans = _build_loans(spec, shape, Calibration(), rng)
    loans = loans.copy()
    # The one lever. Scaling valuations scales the advances written against them, and
    # so the balances that rolled up from those advances, without touching the loan
    # count, the age profile or the LTV distribution.
    for column in ("original_valuation",):
        loans[column] = loans[column].to_numpy(dtype=float) * valuation_scale
    loans["original_principal"] = np.round(
        loans["original_principal"].to_numpy(dtype=float) * valuation_scale, 2)
    book = LoanBook(spec, shape, loans)
    frames = {prd.period: period_frame(book, prd) for prd in cfg.PERIODS}
    return book, frames


def _funded(frame: pd.DataFrame) -> float:
    """Funded balance of a period frame: live accounts only."""
    return float(_num(frame["current_balance"]).fillna(0.0).sum())


def generate_sponsored(*, max_iterations: int = 24, verbose: bool = True):
    """Solve SPV1's one scalar onto its own balance target."""
    cur = cfg.CURRENT_PERIOD.period
    target = cfg.TARGET_BALANCE_S

    x0 = 1.0
    book, frames = _build_sponsored(x0)
    y0 = _funded(frames[cur]) - target.target
    x1 = 1.0 + (0.06 if y0 < 0 else -0.06)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        book, frames = _build_sponsored(x1)
        y1 = _funded(frames[cur]) - target.target
        if verbose:
            print(f"  [calibrate] SPV1 iter {iterations:2d} "
                  f"balance{y1 / 1e6:+7.3f}m  scale {x1:.6f}")
        if target.ok(target.target + y1):
            break
        x_next = _secant_step(x0, y0, x1, y1, lo=0.30, hi=3.0, fallback=0.02)
        x0, y0, x1 = x1, y1, x_next

    injected: List[Dict[str, Any]] = []
    final: Dict[str, pd.DataFrame] = {}
    for prd in cfg.PERIODS:
        frame, inj = _inject_exceptions(frames[prd.period], prd, cfg.PORTFOLIO_S)
        final[prd.period] = frame
        injected.extend(inj)
    return book, final, injected, round(x1, 8), iterations


def _secant_step(x0: float, y0: float, x1: float, y1: float, *,
                 lo: float, hi: float, fallback: float) -> float:
    """One damped secant step toward ``y = 0``, clamped to ``[lo, hi]``."""
    if y1 == y0:
        return float(np.clip(x1 + fallback, lo, hi))
    nxt = x1 - y1 * (x1 - x0) / (y1 - y0)
    if not np.isfinite(nxt):
        return float(np.clip(x1 + fallback, lo, hi))
    # Damp so a steep local slope cannot throw the solve out of range.
    nxt = x1 + 0.85 * (nxt - x1)
    return float(np.clip(nxt, lo, hi))


def generate(*, max_iterations: int = 44, verbose: bool = True) -> GeneratedBooks:
    """Build both books and calibrate them onto the narrative targets."""
    calib = Calibration()
    history: List[Dict[str, float]] = []
    books, frames = _build_all(calib)

    a_id = cfg.PORTFOLIO_A.source_portfolio_id
    b_id = cfg.PORTFOLIO_B.source_portfolio_id
    cur = cfg.CURRENT_PERIOD.period

    def _errors(frames_: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, float]:
        current = {pid: fr[cur] for pid, fr in frames_.items()}
        m = model_metrics(current)
        return {
            "ltv": (m["wa_current_ltv_points"] or 0.0) - cfg.TARGET_WA_LTV_CURRENT.target,
            "age": m["avg_borrower_age"] - cfg.TARGET_BORROWER_AGE_CURRENT.target,
            "mov_a": _movement(frames_[a_id]) - cfg.TARGET_MOVEMENT_A.target,
            "mov_b": _movement(frames_[b_id]) - cfg.TARGET_MOVEMENT_B.target,
        }

    err = _errors(frames)
    # Previous (x, y) pairs per scalar, seeded with a nudge so the first secant
    # step has a gradient to work from.
    prev = {
        "ltv": (calib.ltv_shift, err["ltv"]),
        "age": (calib.age_shift, err["age"]),
        "mov_a": (calib.completion_scale_a, err["mov_a"]),
        "mov_b": (calib.drawdown_scale_b, err["mov_b"]),
    }
    nudged = Calibration(
        ltv_shift=calib.ltv_shift + 0.02,
        age_shift=calib.age_shift - 0.5,
        completion_scale_a=calib.completion_scale_a + 0.12,
        drawdown_scale_b=calib.drawdown_scale_b + 0.12,
    )
    calib = nudged
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        books, frames = _build_all(calib)
        err = _errors(frames)
        history.append({"iteration": iterations, **calib.to_dict(),
                        **{f"err_{k}": round(v, 4) for k, v in err.items()}})
        if verbose:
            print(f"  [calibrate] iter {iterations:2d} "
                  f"ltv{err['ltv']:+7.3f}pp  age{err['age']:+6.3f}y  "
                  f"movA{err['mov_a'] / 1e6:+7.3f}m  movB{err['mov_b'] / 1e6:+7.3f}m")
        converged = (
            cfg.TARGET_WA_LTV_CURRENT.ok(cfg.TARGET_WA_LTV_CURRENT.target + err["ltv"])
            and cfg.TARGET_BORROWER_AGE_CURRENT.ok(
                cfg.TARGET_BORROWER_AGE_CURRENT.target + err["age"])
            and cfg.TARGET_MOVEMENT_A.ok(cfg.TARGET_MOVEMENT_A.target + err["mov_a"])
            and cfg.TARGET_MOVEMENT_B.ok(cfg.TARGET_MOVEMENT_B.target + err["mov_b"])
        )
        if converged:
            break

        nxt = Calibration(
            ltv_shift=_secant_step(prev["ltv"][0], prev["ltv"][1], calib.ltv_shift,
                                   err["ltv"], lo=-0.20, hi=0.35, fallback=-0.01),
            age_shift=_secant_step(prev["age"][0], prev["age"][1], calib.age_shift,
                                   err["age"], lo=-12.0, hi=12.0, fallback=-0.25),
            completion_scale_a=_secant_step(
                prev["mov_a"][0], prev["mov_a"][1], calib.completion_scale_a,
                err["mov_a"], lo=0.15, hi=4.0, fallback=0.05),
            drawdown_scale_b=_secant_step(
                prev["mov_b"][0], prev["mov_b"][1], calib.drawdown_scale_b,
                err["mov_b"], lo=0.05, hi=6.0, fallback=0.05),
        )
        prev = {
            "ltv": (calib.ltv_shift, err["ltv"]),
            "age": (calib.age_shift, err["age"]),
            "mov_a": (calib.completion_scale_a, err["mov_a"]),
            "mov_b": (calib.drawdown_scale_b, err["mov_b"]),
        }
        calib = nxt

    # Inject the bounded data-quality exceptions AFTER calibration converges, so
    # the reported metrics include them (they are immaterial by construction, but
    # the exported figures must be the ones the pipeline actually produces).
    injected: List[Dict[str, Any]] = []
    final_frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    for pid, per_period in frames.items():
        portfolio = cfg.portfolio(pid)
        final_frames[pid] = {}
        for prd in cfg.PERIODS:
            frame, inj = _inject_exceptions(per_period[prd.period], prd, portfolio)
            final_frames[pid][prd.period] = frame
            injected.extend(inj)

    # SPV1 last, and separately: the platform is already solved and frozen by here, so
    # nothing about the sold deal can reach back into it.
    spv_book, spv_frames, spv_injected, spv_scale, spv_iterations = generate_sponsored(
        verbose=verbose)
    books[cfg.PORTFOLIO_S.source_portfolio_id] = spv_book
    final_frames[cfg.PORTFOLIO_S.source_portfolio_id] = spv_frames
    injected.extend(spv_injected)

    # Platform metrics are computed over the PLATFORM portfolios only. SPV1 is reported
    # beside them, never inside them.
    platform_ids = [p.source_portfolio_id for p in cfg.PORTFOLIOS]
    metrics = model_metrics({pid: final_frames[pid][cur] for pid in platform_ids})
    metrics["calibration_history"] = history
    metrics["sponsored"] = {
        "source_portfolio_id": cfg.PORTFOLIO_S.source_portfolio_id,
        "display_id": cfg.PORTFOLIO_S.display_id,
        "valuation_scale": spv_scale,
        "iterations": spv_iterations,
        "funded_balance": round(_funded(spv_frames[cur]), 2),
        "loan_count": int(
            (_num(spv_frames[cur]["current_balance"]).fillna(0.0) > 0).sum()),
    }
    metrics["movement_by_portfolio"] = {
        pid: round(_movement(final_frames[pid]), 2) for pid in platform_ids
    }
    metrics["consolidated_movement"] = round(
        sum(metrics["movement_by_portfolio"].values()), 2)

    return GeneratedBooks(
        books=books, frames=final_frames, calibration=calib,
        injected_exceptions=injected, model_metrics=metrics, iterations=iterations,
    )


# --------------------------------------------------------------------------- #
# Write the raw source extracts
# --------------------------------------------------------------------------- #
def write_source_files(generated: GeneratedBooks) -> List[Dict[str, Any]]:
    """Write each portfolio/period frame out in that portfolio's OWN schema.

    Returns one descriptor per written file for the demo manifest.
    """
    written: List[Dict[str, Any]] = []
    for portfolio in cfg.ALL_PORTFOLIOS:
        schema = schemas.schema_for(portfolio.schema_name)
        for prd in cfg.PERIODS:
            frame = generated.frames[portfolio.source_portfolio_id][prd.period]
            out = pd.DataFrame({
                col.header: frame[col.model_field] for col in schema.columns
            })
            # Portfolio B's status column is the servicer's ("Account State", not
            # "Loan Status") but its VALUES stay on the canonical enumeration. The
            # repository's enum library carries no account_status synonyms, so an
            # invented "Open" would reach the ESMA Annex 2 delivery rules unmapped
            # and block the regulatory submission. The schema difference the
            # onboarding scene shows is the header, which is genuine; inventing a
            # value vocabulary nothing maps would be a demonstration artefact.
            path = cfg.source_file(portfolio, prd)
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
            balance_col = next(
                c.header for c in schema.columns if c.model_field == "current_balance")
            written.append({
                "portfolio": portfolio.display_id,
                "source_portfolio_id": portfolio.source_portfolio_id,
                "period": prd.period,
                "reporting_date": prd.reporting_date,
                "run_id": prd.run_id,
                "schema": schema.name,
                "path": str(path),
                "rows": int(len(out)),
                "columns": list(out.columns),
                "total_balance": round(float(_num(out[balance_col]).fillna(0.0).sum()), 2),
                "synthetic_notice": cfg.SYNTHETIC_BANNER,
            })
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse
    ap = argparse.ArgumentParser(
        description="Generate the Alderbridge synthetic source extracts.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    generated = generate(verbose=not args.quiet)
    written = write_source_files(generated)
    print(json.dumps({
        "calibration": generated.calibration.to_dict(),
        "iterations": generated.iterations,
        "model_metrics": {k: v for k, v in generated.model_metrics.items()
                          if k != "calibration_history"},
        "files": [{k: f[k] for k in ("portfolio", "period", "rows", "total_balance")}
                  for f in written],
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
