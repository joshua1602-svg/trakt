"""analytics_lib.valuation_age — how old, and how good, is the collateral evidence.

Every LTV in a portfolio rests on a valuation observation, and the reported LTV
is only as reliable as the observation behind it. This module measures that
evidence: how old the selected valuations are, what methods produced them, and
how much balance has no usable valuation at all.

The distinction it exists to support
------------------------------------
A stale valuation does not make a loan bad. It makes the reported LTV
*unreliable*, which is a data-quality finding rather than a credit one — and the
two call for different remediation. A book flagged for high LTV where the
valuations are five years old has a measurement problem; the same book with
valuations from last quarter has an economic one. Reporting either as the other
sends the reader to the wrong place.

One definition of "which valuation"
-----------------------------------
Selection runs through :mod:`trakt_core.valuation`, the same versioned policy
that backs ``current_loan_to_value`` in ``explain_values``. That is deliberate
and load-bearing: if this module picked its own valuation, a loan could be
"stale" here while its LTV was derived from a different observation there, and
no reviewer could reconcile the two. There is one selection policy.

Shared, not agent-specific
--------------------------
This lives in ``analytics_lib`` beside ``stratify`` and ``concentration``
because MI, the workspace and the agent tools should all be able to ask it. A
securitisation-specific copy is exactly what the readiness work must not create.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

#: Age buckets in months, as (label, lower_inclusive, upper_exclusive).
#: Chosen to straddle the 24-month limit the valuation selection policy already
#: uses, so the buckets and the policy agree about where "stale" begins.
DEFAULT_AGE_BUCKETS = (
    ("0-6 months", 0, 6),
    ("6-12 months", 6, 12),
    ("12-24 months", 12, 24),
    ("24-36 months", 24, 36),
    ("36-60 months", 36, 60),
    ("60+ months", 60, None),
)

#: Methods that rest on an inspection of the asset, as opposed to an index or a
#: model. Matched case-insensitively against a substring of the recorded method,
#: because tapes spell these many ways ("Full valuation", "full_survey", …).
PHYSICAL_METHOD_MARKERS = ("full", "drive", "desktop", "survey", "inspect",
                           "physical", "valuer")
INDEXED_METHOD_MARKERS = ("index", "hpi", "avm", "automated", "model")


def _months_between(observed: Optional[str], as_of: Optional[str]) -> Optional[int]:
    """Whole months from ``observed`` to ``as_of``, on (year, month) only.

    Day-of-month is deliberately ignored: a valuation dated the 1st and one
    dated the 28th of the same month are the same age for this purpose, and
    counting days would make the bucket a loan lands in depend on when in the
    month the reporting date happens to fall.
    """
    if not observed or not as_of:
        return None
    try:
        left = pd.to_datetime(str(observed)[:10], errors="coerce")
        right = pd.to_datetime(str(as_of)[:10], errors="coerce")
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(left) or pd.isna(right):
        return None
    return (right.year - left.year) * 12 + (right.month - left.month)


def _classify_method(method: Optional[str]) -> str:
    """``physical`` | ``indexed`` | ``other`` | ``unknown``."""
    if method is None or (isinstance(method, float) and method != method):
        return "unknown"
    text = str(method).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return "unknown"
    if any(marker in text for marker in INDEXED_METHOD_MARKERS):
        return "indexed"
    if any(marker in text for marker in PHYSICAL_METHOD_MARKERS):
        return "physical"
    return "other"


def valuation_age_profile(
    df: pd.DataFrame,
    *,
    balance_col: str = "current_principal_balance",
    valuation_date_col: str = "current_valuation_date",
    valuation_amount_col: str = "current_valuation_amount",
    valuation_method_col: str = "current_valuation_method",
    as_of: Optional[str] = None,
    reporting_date_col: str = "reporting_date",
    stale_after_months: int = 24,
    buckets: Sequence = DEFAULT_AGE_BUCKETS,
) -> Dict[str, Any]:
    """Age, method and completeness of the collateral evidence, by balance.

    Balance-weighted throughout, because a stale valuation on a large loan
    matters more than one on a small loan and a count would hide that.

    Returns the three facts the readiness framework screens on —
    ``stale_balance_pct``, ``indexed_balance_pct``, ``missing_balance_pct`` —
    alongside the distribution they come from, so a reviewer sees the shape and
    not only the headline.
    """
    if balance_col not in df.columns:
        return {"available": False,
                "reason": f"the tape carries no {balance_col!r} column, so a "
                          "balance-weighted valuation profile cannot be built."}
    if valuation_date_col not in df.columns:
        return {"available": False,
                "reason": f"the tape carries no {valuation_date_col!r} column, "
                          "so valuation age cannot be measured."}

    balances = pd.to_numeric(df[balance_col], errors="coerce").fillna(0.0)
    total_balance = float(balances.sum())

    # As-at date: explicit, else the tape's own reporting date. Never "today" —
    # a profile that moved with the wall clock could not be reproduced.
    anchor = as_of
    if not anchor and reporting_date_col in df.columns:
        values = df[reporting_date_col].dropna()
        if len(values):
            anchor = str(values.iloc[0])[:10]
    if not anchor:
        return {"available": False,
                "reason": "no as-at date: pass `as_of`, or supply a "
                          f"{reporting_date_col!r} column. Ages are never "
                          "measured against the current wall clock, because a "
                          "profile that moved on its own could not be "
                          "reproduced."}

    amounts = (pd.to_numeric(df[valuation_amount_col], errors="coerce")
               if valuation_amount_col in df.columns else None)
    dates = df[valuation_date_col]
    ages = [_months_between(d, anchor) for d in dates]

    # A valuation is USABLE when it has both an amount and a date. Either alone
    # cannot support an LTV that anyone can age.
    has_amount = (amounts.notna() if amounts is not None
                  else pd.Series(True, index=df.index))
    has_date = pd.Series([a is not None for a in ages], index=df.index)
    usable = has_amount & has_date

    missing_balance = float(balances[~usable].sum())
    usable_balance = float(balances[usable].sum())

    # ---- age distribution, over the usable population -------------------- #
    age_series = pd.Series(ages, index=df.index)
    distribution: List[Dict[str, Any]] = []
    for label, low, high in buckets:
        in_bucket = usable & (age_series >= low)
        if high is not None:
            in_bucket = in_bucket & (age_series < high)
        bucket_balance = float(balances[in_bucket].sum())
        distribution.append({
            "bucket": label,
            "lower_months": low,
            "upper_months": high,
            "loan_count": int(in_bucket.sum()),
            "balance": bucket_balance,
            "balance_pct_of_portfolio": (round(bucket_balance / total_balance * 100, 6)
                                         if total_balance else None),
        })

    stale = usable & (age_series >= stale_after_months)
    stale_balance = float(balances[stale].sum())

    # ---- method mix, over the usable population -------------------------- #
    methods = (df[valuation_method_col] if valuation_method_col in df.columns
               else pd.Series([None] * len(df), index=df.index))
    classes = methods.map(_classify_method)
    method_mix: Dict[str, Any] = {}
    for name in ("physical", "indexed", "other", "unknown"):
        selected = usable & (classes == name)
        balance = float(balances[selected].sum())
        method_mix[name] = {
            "loan_count": int(selected.sum()),
            "balance": balance,
            "balance_pct_of_portfolio": (round(balance / total_balance * 100, 6)
                                         if total_balance else None),
        }

    def _pct(value: float) -> Optional[float]:
        return round(value / total_balance * 100, 6) if total_balance else None

    oldest = max((a for a in ages if a is not None), default=None)
    weighted_age = None
    if usable_balance:
        paired = [(a, b) for a, b in zip(ages, balances) if a is not None]
        if paired:
            weighted_age = round(
                sum(a * b for a, b in paired) / sum(b for _a, b in paired), 4)

    return {
        "available": True,
        "as_of": anchor,
        "stale_after_months": stale_after_months,
        "population": {
            "loan_count": int(len(df)),
            "total_balance": total_balance,
            "loans_with_usable_valuation": int(usable.sum()),
            "balance_with_usable_valuation": usable_balance,
        },
        # The three facts the screening pack reads. Named, so a rule references
        # them by key rather than by position.
        "stale_balance_pct": _pct(stale_balance),
        "indexed_balance_pct": method_mix["indexed"]["balance_pct_of_portfolio"],
        "missing_balance_pct": _pct(missing_balance),
        "stale_balance": stale_balance,
        "stale_loan_count": int(stale.sum()),
        "missing_balance": missing_balance,
        "missing_loan_count": int((~usable).sum()),
        "weighted_average_age_months": weighted_age,
        "oldest_valuation_age_months": oldest,
        "age_distribution": distribution,
        "method_mix": method_mix,
        "notes": [
            "Balance-weighted throughout: a stale valuation on a large loan "
            "matters more than one on a small loan.",
            "A stale valuation makes the reported LTV unreliable; it does not "
            "by itself make the loan bad. Treat it as a data-quality finding "
            "unless the underlying value is independently in doubt.",
            f"'Stale' means {stale_after_months} months or older, matching the "
            "age limit in the governed valuation selection policy.",
        ],
    }
