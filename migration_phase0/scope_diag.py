#!/usr/bin/env python3
"""Why did every filtered question return "no loans match"? — a shape-only probe.

WHY THIS EXISTS. Against the live book at `--scope direct`, four questions of the
accepted bank refused with "No loans in this book match that filter (Borrower
Age, Current LTV, Source Portfolio)". On an equity-release tape, *borrowers over
55* is very close to the whole population, so an empty result is not a fact about
the book — it is a symptom. Three independent things could produce it and they
need opposite fixes:

  * the provenance filter selects a `source_portfolio_id` the tape does not
    carry, so the scope is empty before any other predicate runs;
  * `current_loan_to_value` is held as a FRACTION (0.62) while the question
    means PERCENT (> 50), so no row can ever exceed it;
  * the age column is absent or null, so every comparison is False.

This prints BOOLEANS AND COLUMN NAMES ONLY. No counts, no balances, no ages, no
LTVs, no identifiers beyond the governed portfolio ids the scope resolver itself
publishes. The output is safe to paste into a support thread; that is the point.
"""
from __future__ import annotations

import os
import sys
import glob


def _locate_app_root() -> None:
    if os.path.isdir("mi_agent_api"):
        sys.path.insert(0, os.getcwd())
        return
    for p in glob.glob("/tmp/*/mi_agent_api"):
        root = os.path.dirname(p)
        if os.path.isfile(os.path.join(root, "mi_agent_api", "app.py")):
            sys.path.insert(0, root)
            return
    raise SystemExit("could not find mi_agent_api; run from the extracted app")


def _yes(value) -> str:
    return "yes" if bool(value) else "NO"


def main() -> int:
    _locate_app_root()
    from mi_agent_api import data_source

    df = data_source.get_dataframe()
    cols = set(df.columns)
    print("source :", data_source.data_source_kind())
    print()

    AGE = "youngest_borrower_age"
    LTV = "current_loan_to_value"
    PID = "source_portfolio_id"

    print("COLUMNS PRESENT")
    for name in (AGE, LTV, PID):
        print("  %-26s %s" % (name, _yes(name in cols)))
    print()

    if PID in cols:
        ids = sorted(str(v) for v in df[PID].dropna().unique())
        # Portfolio ids are governed provenance labels published by the scope
        # resolver itself ("portfolio scope applied: Direct (direct_001)"), not
        # client data — and which ids the tape actually carries IS the question.
        print("PORTFOLIO IDS ON THE TAPE")
        print(" ", ids if len(ids) <= 12 else "%d ids" % len(ids))
        print("  contains 'direct_001'    ", _yes("direct_001" in ids))
        print()
        direct = df[df[PID].astype(str) == "direct_001"]
    else:
        direct = df.iloc[0:0]

    print("THE DIRECT SCOPE")
    print("  has any rows at all      ", _yes(len(direct)))
    print()

    if AGE in cols:
        age = df[AGE]
        print("BORROWER AGE (whole tape)")
        print("  every value is null      ", _yes(age.isna().all()))
        print("  any value > 55           ", _yes((age > 55).any()))
        print("  any value > 75           ", _yes((age > 75).any()))
        print()

    if LTV in cols:
        ltv = df[LTV].dropna()
        print("CURRENT LTV (whole tape) — scale check")
        print("  every value is null      ", _yes(df[LTV].isna().all()))
        print("  any value > 1.5          ", _yes((ltv > 1.5).any()),
              " <- 'NO' means LTV is a FRACTION, so '> 50' can never match")
        print("  any value > 50           ", _yes((ltv > 50).any()))
        print()

    if len(direct) and AGE in cols and LTV in cols:
        print("THE FAILING PREDICATE, ONE TERM AT A TIME (direct book)")
        d_age = direct[AGE] > 55
        d_ltv = direct[LTV] > 50
        print("  direct & age > 55        ", _yes(d_age.any()))
        print("  direct & ltv > 50        ", _yes(d_ltv.any()))
        print("  direct & both            ", _yes((d_age & d_ltv).any()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
