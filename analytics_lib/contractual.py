"""analytics_lib.contractual — cash flows the contract itself determines.

The distinction this module exists to hold
------------------------------------------
``analytics_lib.history`` measures what *happened*: SMM, CPR, arrears, losses,
recoveries. This module derives what the contract *says will happen*, and
nothing else. The two must not blur, so the vocabulary is separate:

    OBSERVED             measured from snapshots            history.py
    CONTRACTUAL          derived from contractual terms      here
    ASSUMPTION_REQUIRED  needs an unknown future variable    refused here
    MODEL_REQUIRED       needs behavioural modelling         refused here

Nothing in this file forecasts. There is no rate curve, no prepayment vector,
no mortality table, and no default assumption. Where the contract stops
determining the answer, the answer is a refusal that says which field ran out.

Why the amortisation type decides everything
--------------------------------------------
ESMA gives the amortisation structure as a governed field (RREL35), so the
contract tells Trakt what happens to principal. That turns "can we compute a
WAL?" from one question into several, and they have different answers:

* **BLLT** — "full principal amount is repaid in the last instalment". The
  principal series is a single flow at maturity. It does not depend on the
  interest rate at all, so **WAL is deterministic even on a floating-rate
  loan**. YTM is not, because YTM needs the interest series too.
* **FIXE** — "the principal amount repaid in each instalment is the same".
  Constant principal, again rate-independent. Same split as BLLT.
* **FRXX** — "the total amount — principal plus interest — repaid in each
  instalment is the same". Here the *total* is fixed, so
  ``principal_t = instalment − rate × balance_(t−1)`` and the principal series
  is a function of the rate. WAL inherits whatever uncertainty the rate has.
* **DEXX** — as FRXX after an interest-only first instalment.
* **OTHR** — the regime conveys no principal profile. For Trakt's equity
  release book this is not a gap to be filled: repayment is contingent on
  death, sale or long-term care, so a contractual WAL does not exist. A legal
  long-stop maturity is **not** a WAL and is never used as one.

Payment dates
-------------
Annex 2 carries no payment-date field — ``next_payment_date`` and
``start_date_of_amortisation`` are CREL104 and CREL16, Annex 3 only. Where an
explicit date exists it is used. Where it does not, dates are counted **back
from maturity** at the contractual frequency, and the reason is in the regime's
own wording rather than convenience:

* RREL41 defines the balloon as "principal repayment to be paid **at the
  maturity date**", so the terminal flow must land exactly on maturity.
  Counting back guarantees that; counting forward from origination leaves a
  short final period and misplaces the balloon.
* RREL58 counts "payments made prior to the exposure being transferred to
  **the securitisation**" — it anchors to the transfer, not to the reporting
  date, so it cannot locate the next payment relative to the data cut-off at
  all.

Any stub therefore falls at the front of the schedule, and the result declares
which anchor it used.

Day count
---------
``day_count_convention`` exists only as CREL122, Annex 3. This module therefore
computes a **periodic** yield — discounting at the payment frequency and
annualising by it — and says so in the methodology identifier. It does not
invent an ACT/360 or 30/360 basis for a tape that never supplied one. The WAL
time axis is a measure rather than an accrual basis, and is stated explicitly
as ACT/365F.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

# --------------------------------------------------------------------------- #
# Methodology identifiers. Versioned for the same reason the observed ones are:
# a finding that cited one must keep meaning what it meant.
# --------------------------------------------------------------------------- #
SCHEDULE_METHOD = "CONTRACTUAL_SCHEDULE@v1"
WAL_METHOD = "CONTRACTUAL_WAL@v1"
YTM_METHOD = "CONTRACTUAL_YTM_PERIODIC@v1"

#: The WAL time axis. Named because "do not invent a day-count convention"
#: applies to measures too, even when the measure is not an accrual basis.
WAL_TIME_BASIS = "ACT/365F"
DAYS_PER_YEAR = 365.0

# -- outcome statuses ------------------------------------------------------- #
SUPPORTED = "SUPPORTED"
#: An unknown future variable — a floating reference rate. Not a field gap.
ASSUMPTION_REQUIRED = "ASSUMPTION_REQUIRED"
#: Economically undefined for this product, not merely unimplemented.
NOT_APPLICABLE = "NOT_APPLICABLE"
#: A contractual input the regime should carry is absent for this loan.
FIELD_GAP = "FIELD_GAP"
STATUSES = (SUPPORTED, ASSUMPTION_REQUIRED, NOT_APPLICABLE, FIELD_GAP)

# -- canonical fields, every one checked against its regime definition ------ #
LOAN_ID_FIELD = "loan_identifier"
BALANCE_FIELD = "current_principal_balance"                 # RREL30
MATURITY_FIELD = "maturity_date"                            # RREL24
ORIGINATION_FIELD = "origination_date"                      # RREL23
AMORTISATION_FIELD = "amortisation_type"                    # RREL35
RATE_TYPE_FIELD = "interest_rate_type"                      # RREL42
RATE_FIELD = "current_interest_rate"                        # RREL43
PRINCIPAL_FREQUENCY_FIELD = "scheduled_principal_payment_frequency"   # RREL37
INTEREST_FREQUENCY_FIELD = "scheduled_interest_payment_frequency"     # RREL38
PAYMENT_DUE_FIELD = "payment_due"                           # RREL39 — P&I
BALLOON_FIELD = "balloon_amount"                            # RREL41
PRICE_FIELD = "purchase_price"                              # RREL34 — % of par
CUT_OFF_FIELD = "data_cut_off_date"                         # RREL6
REVISION_DATE_FIELD = "interest_revision_date_1"            # RREL51
NEXT_PAYMENT_DATE_FIELD = "next_payment_date"               # CREL104, A3 only
AMORTISATION_START_FIELD = "start_date_of_amortisation"     # CREL16,  A3 only

#: Per-period principal. Analytics-only — no regime code on any annex — so it
#: is used when present and derived when not. See ``_fixe_principal``.
REGULAR_PRINCIPAL_FIELD = "regular_principal_instalment"

# -- amortisation and rate-type vocabularies -------------------------------- #
BULLET, FIXED_AMORT, FRENCH, GERMAN, OTHER = "BLLT", "FIXE", "FRXX", "DEXX", "OTHR"

#: Rate types whose future path is contractually fixed for the whole life.
FIXED_FOR_LIFE = frozenset({"FXRL"})
#: Fixed only until a contractual revision date (RREL51 dates it, RREL50 gives
#: the new margin but not the new index level).
FIXED_UNTIL_REVISION = frozenset({"FXPR", "FLCF"})
#: Floating in some form. RREL48/49 bound the rate; nothing supplies its level.
FLOATING = frozenset({"FLIF", "FINX", "FLFL", "CAPP", "FLCA"})

#: RREL37/RREL38 frequency codes → months between payments.
FREQUENCY_MONTHS = {"MNTH": 1, "QUTR": 3, "SEMI": 6, "YEAR": 12}

#: Bound on enumerated periods. A 50-year monthly schedule is 600 payments;
#: beyond that the input is more likely wrong than the loan long.
MAX_PERIODS = 1200

#: Source-label synonyms for the ESMA enums this module normalises, keyed by
#: field code and lower-cased label.
#:
#: Restated here rather than read from the regulatory delivery
#: configuration. ``analytics_lib`` is
#: MI runtime, and MI must not depend on the regulatory delivery chain — an
#: invariant ``tests/test_mi_regulatory_separation.py`` enforces, and one this
#: module previously broke.
#:
#: The previous version read the delivery rules file to avoid restating
#: the mapping, and caught ``OSError`` by falling back to an **empty** map. That
#: made the coupling worse than the duplication it avoided: a deployment that
#: runs MI without the regulatory configuration would silently stop recognising
#: "French" and "Bullet", and would then produce *fewer contractual schedules*
#: with no error and no warning — a quiet reduction in coverage, which is the
#: hardest kind of defect to notice. An explicit map cannot fail that way.
#:
#: Drift against the delivery rules is caught by
#: ``tests/test_contractual_enum_synonyms.py``, which compares the two whenever
#: the delivery configuration is present. A test that fails loudly is a better
#: guarantee than a coupling that degrades silently.
_ENUM_SYNONYMS: Dict[str, Dict[str, str]] = {
    "RREL35": {
        "french": "FRXX", "german": "DEXX",
        "fixed amortisation schedule": "FIXE", "bullet": "BLLT",
        "other": "OTHR", "interest roll-up": "OTHR",
        "frxx": "FRXX", "dexx": "DEXX", "fixe": "FIXE", "bllt": "BLLT",
        "othr": "OTHR",
        # A level-instalment loan, however the lender labels it.
        "annuity": "FRXX", "french amortisation": "FRXX",
        "capital_repayment": "FRXX", "constant_installment": "FRXX",
        "level_payment": "FRXX", "standard": "FRXX",
        # Constant principal with declining interest.
        "tilgung": "DEXX", "tilgungsdarlehen": "DEXX",
        "interest_then_level": "DEXX",
        "fixed": "FIXE", "fixed_amortisation": "FIXE",
        "constant_principal": "FIXE", "linear": "FIXE",
        # Principal repaid in one final instalment.
        "interest_only_bullet": "BLLT", "io_bullet": "BLLT",
        "full_repay_at_maturity": "BLLT", "redemption_at_maturity": "BLLT",
        # Not a schedule this module can enumerate — OTHR, and OTHR yields no
        # cashflows rather than a guessed one.
        "part_and_part": "OTHR", "mixed": "OTHR", "hybrid": "OTHR",
        "unknown": "OTHR", "n_a": "OTHR",
    },
    "RREL42": {
        "fixed": "FXRL",
        "fxrl": "FXRL", "flif": "FLIF", "disc": "DISC", "flcf": "FLCF",
        "fxpr": "FXPR", "finx": "FINX", "capp": "CAPP", "flfl": "FLFL",
        "flca": "FLCA", "mode": "MODE", "obls": "OBLS", "swic": "SWIC",
        "othr": "OTHR",
        "fixed rate": "FXRL", "fix": "FXRL",
        "variable": "FLIF", "variable rate": "FLIF", "floating": "FLIF",
        "var": "FLIF", "libor": "FLIF", "euribor": "FLIF",
        "base_rate": "FLIF",
        "teaser": "DISC", "introductory": "DISC", "discounted": "DISC",
        "mixed": "OTHR", "unknown": "OTHR", "other": "OTHR",
    },
    "RREL37": {
        "monthly": "MNTH", "quarterly": "QUTR",
        "mnth": "MNTH", "qutr": "QUTR",
        "semi": "SEMI", "year": "YEAR", "othr": "OTHR",
    },
}


def _delivery_enum_map(code: str) -> Dict[str, str]:
    """The enum synonyms for one ESMA field code."""
    return _ENUM_SYNONYMS.get(code, {})


def normalise_amortisation(value: Any) -> Optional[str]:
    """RREL35, as a code. Unrecognised labels return None rather than OTHR —
    "we do not recognise this" and "the contract says other" are different
    statements and only one of them is a fact about the loan."""
    return _normalise(value, "RREL35",
                      {BULLET, FIXED_AMORT, FRENCH, GERMAN, OTHER})


def normalise_rate_type(value: Any) -> Optional[str]:
    """RREL42, as a code."""
    known = FIXED_FOR_LIFE | FIXED_UNTIL_REVISION | FLOATING | {
        "DISC", "SWIC", "OBLS", "MODE", OTHER}
    return _normalise(value, "RREL42", known)


def _normalise(value: Any, code: str, known: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    if not text:
        return None
    mapped = _delivery_enum_map(code).get(text.lower())
    if mapped:
        return mapped
    upper = text.upper()
    return upper if upper in known else None


def frequency_months(value: Any) -> Optional[int]:
    """RREL37/RREL38 → months. ``OTHR`` returns None: an unstated interval
    cannot be enumerated, and guessing monthly would be an invention."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    if not text:
        return None
    code = _delivery_enum_map("RREL37").get(text.lower()) or text.upper()
    return FREQUENCY_MONTHS.get(code)


# --------------------------------------------------------------------------- #
# The schedule
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContractualSchedule:
    """Contractual cash flows for one exposure, or a reasoned refusal.

    ``principal_deterministic`` and ``interest_deterministic`` are separate on
    purpose. A floating-rate bullet loan has a fully determined principal
    series and an undetermined interest series, so its WAL is contractual and
    its YTM is not. Collapsing them into one flag would lose exactly the
    capability this sprint exists to expose.
    """

    loan_id: str
    status: str
    method: str = SCHEDULE_METHOD
    reason: str = ""
    amortisation_type: Optional[str] = None
    rate_type: Optional[str] = None
    principal_deterministic: bool = False
    interest_deterministic: bool = False
    anchor_basis: str = ""
    frequency_months: Optional[int] = None
    as_of: Optional[date] = None
    dates: Tuple[date, ...] = ()
    principal: Tuple[float, ...] = ()
    interest: Tuple[float, ...] = ()
    opening_balance: float = 0.0
    balloon: float = 0.0
    #: ``purchase_price`` (RREL34) for THIS exposure, carried on the schedule
    #: so a caller working from schedules alone can still price it. Sprint 3
    #: found the governed tool silently dropping it: the schedule did not
    #: retain the price, so ``contractual_analytics`` could only produce a
    #: yield when the caller passed an override — and a portfolio whose every
    #: loan carried RREL34 reported ASSUMPTION_REQUIRED for all of them.
    price_percent_of_par: Optional[float] = None
    notes: Tuple[str, ...] = ()

    @property
    def total(self) -> Tuple[float, ...]:
        return tuple(p + i for p, i in zip(self.principal, self.interest))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loan_id": self.loan_id,
            "status": self.status,
            "methodology_version": self.method,
            "reason": self.reason or None,
            "amortisation_type": self.amortisation_type,
            "rate_type": self.rate_type,
            "principal_deterministic": self.principal_deterministic,
            "interest_deterministic": self.interest_deterministic,
            "anchor_basis": self.anchor_basis or None,
            "frequency_months": self.frequency_months,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "opening_balance": round(self.opening_balance, 2),
            "balloon": round(self.balloon, 2),
            "price_percent_of_par": self.price_percent_of_par,
            "cashflows": [
                {"date": d.isoformat(), "principal": round(p, 2),
                 "interest": round(i, 2), "total": round(p + i, 2)}
                for d, p, i in zip(self.dates, self.principal, self.interest)
            ],
            "notes": list(self.notes),
        }


def _refusal(loan_id: str, status: str, reason: str, **extra: Any
             ) -> ContractualSchedule:
    return ContractualSchedule(loan_id=loan_id, status=status, reason=reason,
                               **extra)


def _as_date(value: Any) -> Optional[date]:
    """One date, cheaply.

    ``pd.to_datetime`` on a *scalar* re-runs its format guesser on every call,
    which profiling showed to be 49% of the whole enumeration at 10k loans —
    3.0 of 10.7 seconds in ``_guess_datetime_format_for_array`` alone. ISO
    strings are the overwhelming majority and parse directly; pandas remains
    the fallback for everything else, so nothing stops being accepted.
    """
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            pass
    try:
        stamp = pd.to_datetime(value, errors="coerce")
    except (ValueError, TypeError):
        return None
    if stamp is None or pd.isna(stamp):
        return None
    return stamp.date()


def _as_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result


def _add_months(anchor: date, months: int) -> date:
    """Calendar month arithmetic, clamping the day to the target month's end so
    a 31st does not roll into the following month."""
    total = (anchor.year * 12 + anchor.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    day = min(anchor.day, _days_in_month(year, month))
    return date(year, month, day)


#: Month lengths, non-leap. February is corrected below. Constructing two
#: ``date`` objects per call to subtract ordinals cost 1.0s per 10k loans in
#: the profile, for arithmetic that is a table lookup.
_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _days_in_month(year: int, month: int) -> int:
    if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        return 29
    return _MONTH_LENGTHS[month - 1]


def _payment_dates(as_of: date, maturity: date, step_months: int,
                   explicit_next: Optional[date]) -> Tuple[List[date], str]:
    """Every remaining contractual payment date, and how it was anchored.

    Counting back from maturity is preferred where no explicit date exists —
    see the module docstring for why the regime's own wording forces it.
    """
    if explicit_next is not None and explicit_next <= maturity:
        dates, cursor = [], explicit_next
        while cursor <= maturity and len(dates) < MAX_PERIODS:
            dates.append(cursor)
            cursor = _add_months(cursor, step_months)
        if dates and dates[-1] != maturity:
            dates.append(maturity)
        return dates, "explicit_next_payment_date"

    backwards: List[date] = []
    cursor = maturity
    while cursor > as_of and len(backwards) < MAX_PERIODS:
        backwards.append(cursor)
        cursor = _add_months(cursor, -step_months)
    backwards.reverse()
    return backwards, "counted_back_from_maturity"


def _rate_determinism(rate_type: Optional[str], last_date: Optional[date],
                      revision_date: Optional[date]) -> Tuple[bool, str]:
    """Is the interest series contractually determined for the whole schedule?"""
    if rate_type is None:
        return False, ("the tape does not state interest_rate_type (RREL42), "
                       "so whether the rate is contractually fixed is unknown. "
                       "Interest is not assumed to be fixed.")
    if rate_type in FIXED_FOR_LIFE:
        return True, ""
    if rate_type in FIXED_UNTIL_REVISION:
        if revision_date is None:
            return False, (f"rate type {rate_type} is fixed only until a "
                           "contractual revision, and interest_revision_date_1 "
                           "(RREL51) is not populated, so the deterministic "
                           "horizon is unknown.")
        if last_date is not None and revision_date >= last_date:
            return True, ""
        return False, (f"rate type {rate_type} is fixed only to "
                       f"{revision_date.isoformat()} (RREL51); the schedule "
                       "runs beyond it and the rate after that revision is not "
                       "a contractual fact. RREL50 gives the new margin, not "
                       "the index level.")
    if rate_type in FLOATING:
        return False, (f"rate type {rate_type} is floating: the future "
                       "reference index level is unknown. RREL48/RREL49 bound "
                       "the rate; they do not determine it.")
    return False, (f"rate type {rate_type} does not describe a contractually "
                   "determined rate path.")


def build_schedule(loan: Mapping[str, Any], *, as_of: Optional[date] = None
                   ) -> ContractualSchedule:
    """Contractual cash flows for one exposure.

    Returns a refusal — never a partial schedule labelled contractual — where
    the contract does not determine the answer.
    """
    loan_id = str(loan.get(LOAN_ID_FIELD) or loan.get("loan_id") or "")

    amortisation = normalise_amortisation(loan.get(AMORTISATION_FIELD))
    rate_type = normalise_rate_type(loan.get(RATE_TYPE_FIELD))

    if amortisation is None:
        return _refusal(loan_id, FIELD_GAP,
                        "the tape does not state amortisation_type (RREL35), "
                        "so what the contract does with principal is unknown.",
                        rate_type=rate_type)

    if amortisation == OTHER:
        return _refusal(
            loan_id, NOT_APPLICABLE,
            "amortisation_type is OTHR: the regime conveys no principal "
            "profile. For a lifetime mortgage this is not a data gap — "
            "repayment is contingent on death, sale or long-term care, so no "
            "contractual repayment date exists. A legal maturity is not a "
            "contractual WAL and is not used as one. An expected WAL would "
            "require mortality and redemption assumptions, which Trakt does "
            "not hold and does not model.",
            amortisation_type=amortisation, rate_type=rate_type)

    balance = _as_float(loan.get(BALANCE_FIELD))
    if balance is None or balance <= 0:
        return _refusal(loan_id, FIELD_GAP,
                        f"no positive {BALANCE_FIELD} (RREL30) to amortise.",
                        amortisation_type=amortisation, rate_type=rate_type)

    maturity = _as_date(loan.get(MATURITY_FIELD))
    if maturity is None:
        return _refusal(loan_id, FIELD_GAP,
                        "the tape does not state maturity_date (RREL24), so "
                        "the schedule has no end point.",
                        amortisation_type=amortisation, rate_type=rate_type)

    cut_off = as_of or _as_date(loan.get(CUT_OFF_FIELD))
    if cut_off is None:
        return _refusal(loan_id, FIELD_GAP,
                        "no as-of date: neither an explicit one nor "
                        "data_cut_off_date (RREL6).",
                        amortisation_type=amortisation, rate_type=rate_type)

    if maturity <= cut_off:
        return _refusal(loan_id, FIELD_GAP,
                        f"maturity {maturity.isoformat()} is not after the "
                        f"as-of date {cut_off.isoformat()}: there are no "
                        "remaining contractual cash flows.",
                        amortisation_type=amortisation, rate_type=rate_type,
                        as_of=cut_off)

    step = frequency_months(loan.get(PRINCIPAL_FREQUENCY_FIELD))
    if amortisation == BULLET and step is None:
        # A bullet has one principal flow whatever the interest frequency, so
        # the interest frequency alone can carry the schedule.
        step = frequency_months(loan.get(INTEREST_FREQUENCY_FIELD))
    if step is None:
        return _refusal(loan_id, FIELD_GAP,
                        "the payment frequency (RREL37/RREL38) is absent or "
                        "OTHR, so the payment interval itself is unknown. It "
                        "is not assumed to be monthly.",
                        amortisation_type=amortisation, rate_type=rate_type,
                        as_of=cut_off)

    dates, anchor = _payment_dates(
        cut_off, maturity, step, _as_date(loan.get(NEXT_PAYMENT_DATE_FIELD)))
    if not dates:
        return _refusal(loan_id, FIELD_GAP,
                        "no remaining payment dates could be constructed.",
                        amortisation_type=amortisation, rate_type=rate_type,
                        as_of=cut_off)

    interest_ok, interest_reason = _rate_determinism(
        rate_type, dates[-1], _as_date(loan.get(REVISION_DATE_FIELD)))
    annual_rate = _as_float(loan.get(RATE_FIELD))
    if annual_rate is None:
        interest_ok = False
        interest_reason = ("the tape does not state current_interest_rate "
                           "(RREL43).")
    periodic_rate = (annual_rate / 100.0) * (step / 12.0) if annual_rate else 0.0

    balloon = max(0.0, _as_float(loan.get(BALLOON_FIELD)) or 0.0)
    notes: List[str] = []
    if anchor == "counted_back_from_maturity":
        notes.append(
            "Payment dates counted back from maturity_date (RREL24) at the "
            "contractual frequency: RREL41 requires the balloon to fall on "
            "the maturity date, and Annex 2 carries no payment-date field. "
            "Any stub period is therefore at the front of the schedule.")

    # -- the principal series ------------------------------------------------ #
    if amortisation == BULLET:
        principal = [0.0] * (len(dates) - 1) + [balance]
        principal_ok = True
    elif amortisation == FIXED_AMORT:
        principal, principal_ok, fixe_note = _fixe_principal(
            loan, balance, balloon, len(dates))
        if fixe_note:
            notes.append(fixe_note)
    else:  # FRXX / DEXX
        if not interest_ok:
            return _refusal(
                loan_id, ASSUMPTION_REQUIRED,
                f"amortisation_type {amortisation} fixes the TOTAL instalment, "
                "so the principal share is what is left after interest — which "
                f"makes the principal series rate-dependent. {interest_reason} "
                "No rate path is assumed.",
                amortisation_type=amortisation, rate_type=rate_type,
                as_of=cut_off, frequency_months=step, anchor_basis=anchor)
        principal, interest_series, annuity_note = _annuity_principal(
            loan, balance, balloon, periodic_rate, len(dates),
            interest_only_first=(amortisation == GERMAN))
        if annuity_note:
            notes.append(annuity_note)
        return ContractualSchedule(
            loan_id=loan_id, status=SUPPORTED, reason="",
            amortisation_type=amortisation, rate_type=rate_type,
            principal_deterministic=True, interest_deterministic=True,
            anchor_basis=anchor, frequency_months=step, as_of=cut_off,
            dates=tuple(dates), principal=tuple(principal),
            interest=tuple(interest_series), opening_balance=balance,
            balloon=balloon, price_percent_of_par=_as_float(loan.get(PRICE_FIELD)),
            notes=tuple(notes))

    if not principal_ok:
        return _refusal(loan_id, FIELD_GAP,
                        "constant principal could not be established: "
                        f"{REGULAR_PRINCIPAL_FIELD} is absent and the balance "
                        "net of the balloon does not divide into the remaining "
                        "periods.",
                        amortisation_type=amortisation, rate_type=rate_type,
                        as_of=cut_off, frequency_months=step)

    # -- the interest series, which BLLT and FIXE do not need for WAL -------- #
    if interest_ok:
        interest_series = _interest_on_declining_balance(
            balance, principal, periodic_rate)
    else:
        interest_series = [0.0] * len(principal)
        notes.append(
            "Interest is NOT contractually determined and is reported as zero "
            f"rather than assumed: {interest_reason} The PRINCIPAL series is "
            "unaffected — under this amortisation type the contract fixes "
            "principal directly — so contractual WAL remains available while "
            "contractual YTM does not.")

    return ContractualSchedule(
        loan_id=loan_id, status=SUPPORTED, reason="",
        amortisation_type=amortisation, rate_type=rate_type,
        principal_deterministic=True, interest_deterministic=interest_ok,
        anchor_basis=anchor, frequency_months=step, as_of=cut_off,
        dates=tuple(dates), principal=tuple(principal),
        interest=tuple(interest_series), opening_balance=balance,
        balloon=balloon, price_percent_of_par=_as_float(loan.get(PRICE_FIELD)),
        notes=tuple(notes))


def _fixe_principal(loan: Mapping[str, Any], balance: float, balloon: float,
                    periods: int) -> Tuple[List[float], bool, str]:
    """Constant principal per instalment, per RREL35 = FIXE.

    ``regular_principal_instalment`` is used where the tape carries it. It has
    no regime code on any annex, so where it is absent the constant is
    recovered as ``(balance − balloon) ÷ amortising periods`` — an inference
    made **only** because RREL35 has already asserted that the principal
    repaid in each instalment is the same. Outside FIXE the same arithmetic
    would be a fabrication.
    """
    amortising = max(0, periods - 1) if balloon > 0 else periods
    stated = _as_float(loan.get(REGULAR_PRINCIPAL_FIELD))
    note = ""

    if stated is not None and stated > 0:
        per_period = stated
    elif amortising > 0:
        per_period = max(0.0, balance - balloon) / amortising
        note = (f"{REGULAR_PRINCIPAL_FIELD} is not on the tape and carries no "
                "regime code on any annex, so the constant was derived as "
                "(balance − balloon) ÷ remaining periods. This is sound only "
                "because RREL35 = FIXE states the per-instalment principal is "
                "constant.")
    else:
        return [], False, ""

    series = [per_period] * periods
    # The contract cannot repay more than is owed: the final instalment trues
    # up so the schedule closes on the balance rather than overshooting it.
    running = 0.0
    for index in range(periods):
        remaining = balance - running
        series[index] = min(series[index], max(0.0, remaining))
        running += series[index]
    shortfall = balance - running
    if shortfall > 0.005:
        series[-1] += shortfall
    return series, True, note


def _annuity_principal(loan: Mapping[str, Any], balance: float, balloon: float,
                       periodic_rate: float, periods: int, *,
                       interest_only_first: bool
                       ) -> Tuple[List[float], List[float], str]:
    """FRXX and DEXX: the contract fixes the TOTAL instalment.

    ``payment_due`` (RREL39) is "the next contractual payment due by the
    obligor according to the payment frequency", which for a constant-total
    structure *is* that constant, and it is carried on all five annexes. Where
    it is absent the annuity factor supplies it from balance, rate and term —
    still contractual arithmetic, with no behavioural input.

    DEXX differs by one instalment: "the first instalment is interest-only and
    the remaining instalments are constant". No principal is allocated to that
    first period.
    """
    amortising = periods - 1 if interest_only_first else periods
    note = ""

    stated = _as_float(loan.get(PAYMENT_DUE_FIELD))
    if stated is not None and stated > 0:
        instalment = stated
    else:
        instalment = _annuity_instalment(balance, balloon, periodic_rate,
                                         amortising)
        note = ("payment_due (RREL39) is not on the tape, so the constant "
                "instalment was derived from the annuity factor over balance, "
                "rate and remaining term. No behavioural input is involved.")

    principal: List[float] = []
    interest: List[float] = []
    running = balance
    for index in range(periods):
        accrued = running * periodic_rate
        if interest_only_first and index == 0:
            principal.append(0.0)
            interest.append(accrued)
            continue
        share = instalment - accrued
        if index == periods - 1:
            # Final instalment closes the loan: whatever principal remains,
            # which is where a balloon lands under RREL41.
            share = running
        share = max(0.0, min(share, running))
        principal.append(share)
        interest.append(accrued)
        running -= share
    return principal, interest, note


def _annuity_instalment(balance: float, balloon: float, periodic_rate: float,
                        periods: int) -> float:
    if periods <= 0:
        return balance
    if abs(periodic_rate) < 1e-12:
        return max(0.0, balance - balloon) / periods + balloon / periods
    factor = (1.0 + periodic_rate) ** periods
    amortising_part = (balance - balloon / factor)
    return amortising_part * periodic_rate * factor / (factor - 1.0)


def _interest_on_declining_balance(balance: float, principal: Sequence[float],
                                   periodic_rate: float) -> List[float]:
    interest: List[float] = []
    running = balance
    for amount in principal:
        interest.append(running * periodic_rate)
        running -= amount
    return interest


# --------------------------------------------------------------------------- #
# Weighted average life
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WalResult:
    status: str
    method: str = WAL_METHOD
    wal_years: Optional[float] = None
    reason: str = ""
    explanation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"metric": "contractual_wal", "status": self.status,
                "methodology_version": self.method,
                "wal_years": self.wal_years,
                "reason": self.reason or None,
                "explanation": self.explanation or None}


def contractual_wal(schedule: ContractualSchedule) -> WalResult:
    """WAL = Σ(tᵢ × Pᵢ) ÷ ΣPᵢ, over contractual principal only.

    Interest is not in the numerator and not in the denominator: WAL is a
    principal-weighted measure, which is exactly why a floating-rate bullet
    loan has a contractual WAL and no contractual YTM.
    """
    if schedule.status != SUPPORTED:
        return WalResult(status=schedule.status, reason=schedule.reason,
                         explanation={
                             "amortisation_type": schedule.amortisation_type,
                             "rate_type": schedule.rate_type})
    if not schedule.principal_deterministic:
        return WalResult(status=ASSUMPTION_REQUIRED,
                         reason="the principal series is not contractually "
                                "determined.")

    total = sum(schedule.principal)
    if total <= 0:
        return WalResult(status=FIELD_GAP,
                         reason="the contractual principal series sums to zero.")

    weighted = 0.0
    for when, amount in zip(schedule.dates, schedule.principal):
        years = (when - schedule.as_of).days / DAYS_PER_YEAR
        weighted += years * amount

    return WalResult(
        status=SUPPORTED, wal_years=round(weighted / total, 6),
        explanation={
            "amortisation_type": schedule.amortisation_type,
            "rate_type": schedule.rate_type,
            "schedule_method": schedule.method,
            "anchor_basis": schedule.anchor_basis,
            "payment_frequency_months": schedule.frequency_months,
            "as_of": schedule.as_of.isoformat(),
            "final_payment_date": schedule.dates[-1].isoformat(),
            "time_basis": WAL_TIME_BASIS,
            "principal_payments": len([p for p in schedule.principal if p > 0]),
            "total_principal": round(total, 2),
            "balloon": round(schedule.balloon, 2),
            "balloon_treatment": (
                "paid at maturity per RREL41" if schedule.balloon > 0
                else "no balloon on this exposure"),
            "interest_deterministic": schedule.interest_deterministic,
            "notes": list(schedule.notes),
        })


def portfolio_wal(schedules: Sequence[ContractualSchedule]) -> WalResult:
    """Portfolio WAL by aggregating the contractual principal cash flows.

    Aggregation rather than a weighted average of loan WALs. The two are
    mathematically equivalent when the weights are total principal, and
    aggregation is the one that cannot be got wrong by weighting on balance
    instead — which is the mistake that would silently understate a book with
    balloons.
    """
    buckets: Dict[date, float] = {}
    supported = 0
    excluded: Dict[str, int] = {}
    as_of: Optional[date] = None

    for schedule in schedules:
        if schedule.status != SUPPORTED or not schedule.principal_deterministic:
            excluded[schedule.status] = excluded.get(schedule.status, 0) + 1
            continue
        supported += 1
        as_of = as_of or schedule.as_of
        for when, amount in zip(schedule.dates, schedule.principal):
            if amount:
                buckets[when] = buckets.get(when, 0.0) + amount

    if not buckets or as_of is None:
        return WalResult(status=NOT_APPLICABLE,
                         reason="no exposure in this population has a "
                                "contractually determined principal series.",
                         explanation={"excluded_by_status": excluded})

    total = sum(buckets.values())
    weighted = sum((when - as_of).days / DAYS_PER_YEAR * amount
                   for when, amount in buckets.items())
    return WalResult(
        status=SUPPORTED, wal_years=round(weighted / total, 6),
        explanation={
            "basis": "aggregated contractual principal cash flows",
            "loans_included": supported,
            "excluded_by_status": excluded,
            "as_of": as_of.isoformat(),
            "time_basis": WAL_TIME_BASIS,
            "total_principal": round(total, 2),
            "payment_dates": len(buckets),
        })


# --------------------------------------------------------------------------- #
# Yield to maturity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class YtmResult:
    status: str
    method: str = YTM_METHOD
    ytm_pct: Optional[float] = None
    periodic_yield_pct: Optional[float] = None
    reason: str = ""
    explanation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"metric": "contractual_ytm_periodic", "status": self.status,
                "methodology_version": self.method,
                "ytm_pct": self.ytm_pct,
                "periodic_yield_pct": self.periodic_yield_pct,
                "reason": self.reason or None,
                "explanation": self.explanation or None}


def contractual_ytm(schedule: ContractualSchedule, *,
                    price_percent_of_par: Optional[float] = None,
                    loan: Optional[Mapping[str, Any]] = None) -> YtmResult:
    """The periodic contractual yield, annualised by the payment frequency.

    **Periodic, and named as such.** ``day_count_convention`` exists only as
    CREL122 on Annex 3, so a day-count-exact yield cannot be computed for a
    residential tape without inventing a convention. Discounting at the
    contractual payment frequency needs no such invention and is exact on a
    regular schedule; the first period may be a stub, which the explanation
    discloses.

    Price is ``purchase_price`` (RREL34) — "the price, relative to par, at
    which the underlying exposure was purchased by the SSPE. Enter 100 if no
    discounting was applied" — so it is already a percentage and needs no cash
    consideration figure.
    """
    if schedule.status != SUPPORTED:
        return YtmResult(status=schedule.status, reason=schedule.reason,
                         explanation={
                             "amortisation_type": schedule.amortisation_type,
                             "rate_type": schedule.rate_type})
    if not schedule.interest_deterministic:
        return YtmResult(
            status=ASSUMPTION_REQUIRED,
            reason=("the contractual interest series is not determined, so a "
                    "yield would require a rate assumption. The principal "
                    "series is determined, so contractual WAL is available."),
            explanation={"amortisation_type": schedule.amortisation_type,
                         "rate_type": schedule.rate_type,
                         "notes": list(schedule.notes)})

    price_pct = price_percent_of_par
    if price_pct is None and loan is not None:
        price_pct = _as_float(loan.get(PRICE_FIELD))
    if price_pct is None:
        # The schedule carries the exposure's own RREL34 where the tape had
        # one, so a caller holding only schedules still prices correctly.
        price_pct = schedule.price_percent_of_par
    if price_pct is None:
        return YtmResult(status=FIELD_GAP,
                         reason=("no purchase_price (RREL34). A price is not "
                                 "assumed to be par: that would turn every "
                                 "yield into the coupon."))
    if price_pct <= 0:
        return YtmResult(status=FIELD_GAP,
                         reason="purchase_price (RREL34) is not positive.")

    consideration = schedule.opening_balance * price_pct / 100.0
    flows = list(schedule.total)
    if not flows or consideration <= 0:
        return YtmResult(status=FIELD_GAP,
                         reason="no contractual cash flows to discount.")

    periodic = _solve_irr(consideration, flows)
    if periodic is None:
        return YtmResult(status=FIELD_GAP,
                         reason=("no yield solves these cash flows within the "
                                 "search bracket; the inputs are inconsistent."))

    per_year = 12.0 / float(schedule.frequency_months or 12)
    annualised = ((1.0 + periodic) ** per_year - 1.0) * 100.0

    return YtmResult(
        status=SUPPORTED, ytm_pct=round(annualised, 6),
        periodic_yield_pct=round(periodic * 100.0, 6),
        explanation={
            "price_basis": (f"purchase_price (RREL34) = {price_pct}% of par; "
                            f"consideration {round(consideration, 2)} on an "
                            f"opening balance of "
                            f"{round(schedule.opening_balance, 2)}"),
            "amortisation_type": schedule.amortisation_type,
            "rate_type": schedule.rate_type,
            "interest_rate_basis": (
                "current_interest_rate (RREL43), contractually fixed for the "
                "life of the schedule"),
            "schedule_method": schedule.method,
            "anchor_basis": schedule.anchor_basis,
            "date_frequency_basis": (
                f"periodic: {schedule.frequency_months} month(s) per period, "
                f"{per_year:g} periods per year, annualised as "
                "(1+y)^periods_per_year − 1"),
            "day_count": ("NOT APPLIED — this is a periodic yield. "
                          "day_count_convention exists only as CREL122 "
                          "(Annex 3) and is not invented for tapes without it"),
            "irr_method": "bisection on the discounted cash flows",
            "periods": len(flows),
            "as_of": schedule.as_of.isoformat(),
            "notes": list(schedule.notes),
        })


def _solve_irr(consideration: float, flows: Sequence[float],
               tolerance: float = 1e-10, iterations: int = 200
               ) -> Optional[float]:
    """Bisection. Chosen over Newton deliberately: it cannot diverge on a
    schedule with an awkward first period, and the bracket makes a failure to
    converge visible rather than silent."""
    def present_value(rate: float) -> float:
        # Accumulated iteratively rather than as `flow / (1+rate)**n`. At the
        # bottom of the bracket the discount base is 1e-4, and 1e-4**600
        # underflows to exactly 0.0 — which the power form then divides by.
        # Accumulating forwards, an underflowed factor simply makes the
        # remaining terms zero, which is their correct limit.
        base = 1.0 + rate
        factor, total = 1.0, 0.0
        for flow in flows:
            factor /= base
            if factor == 0.0:
                break
            total += flow * factor
        return total - consideration

    low, high = -0.9999, 10.0
    low_value, high_value = present_value(low), present_value(high)
    if low_value * high_value > 0:
        return None
    for _ in range(iterations):
        mid = (low + high) / 2.0
        value = present_value(mid)
        if abs(value) < tolerance or (high - low) < tolerance:
            return mid
        if low_value * value <= 0:
            high = mid
        else:
            low, low_value = mid, value
    return (low + high) / 2.0


# --------------------------------------------------------------------------- #
# Frame-level entry point
# --------------------------------------------------------------------------- #
def schedule_frame(frame: pd.DataFrame, *, as_of: Optional[date] = None
                   ) -> List[ContractualSchedule]:
    """One schedule per row. Columns absent from the frame are simply absent —
    the per-loan refusals name which field ran out."""
    columns = [c for c in (
        LOAN_ID_FIELD, BALANCE_FIELD, MATURITY_FIELD, AMORTISATION_FIELD,
        RATE_TYPE_FIELD, RATE_FIELD, PRINCIPAL_FREQUENCY_FIELD,
        INTEREST_FREQUENCY_FIELD, PAYMENT_DUE_FIELD, BALLOON_FIELD,
        PRICE_FIELD, CUT_OFF_FIELD, REVISION_DATE_FIELD,
        NEXT_PAYMENT_DATE_FIELD, REGULAR_PRINCIPAL_FIELD, ORIGINATION_FIELD,
    ) if c in frame.columns]
    projected = frame[columns] if columns else frame.iloc[:, :0]
    return [build_schedule(row, as_of=as_of)
            for row in projected.to_dict(orient="records")]
