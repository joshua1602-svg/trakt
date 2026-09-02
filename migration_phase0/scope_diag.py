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

    PRODUCT = "erm_product_type"
    GEO = "collateral_geography"
    RATE = "current_interest_rate"

    if RATE in cols:
        rate = df[RATE].dropna()
        print("INTEREST RATE — the numeric CONTROL")
        print("  any value > 7            ", _yes((rate > 7).any()),
              " <- a rate filter DID answer, so numeric thresholds work")
        print()

    # A value the question names must match a value the BOOK carries. Product
    # type answers fine as a GROUP-BY and never as a FILTER, so the failure is
    # in value matching, not in the column. These print booleans about how the
    # book spells its values, never the spellings themselves.
    def _norm(v):
        return str(v).strip().lower().replace("_", " ").replace("-", " ")

    def _match(col, wanted):
        """Whole tape AND direct book, because those are different questions.

        The LTV result taught this: "> 50 exists on the tape" and "> 50 exists
        in the DIRECT book" had opposite answers, and only the second one is
        what the questions ran against. A value carried solely by the acquired
        book is a correct refusal at scope=direct, not a matching bug.
        """
        if col not in cols:
            print("  %-24s column absent" % wanted)
            return
        vals = [str(v) for v in df[col].dropna().unique()]
        exact = any(v == wanted for v in vals)
        loose = any(_norm(v) == wanted for v in vals)
        if len(direct):
            dvals = [str(v) for v in direct[col].dropna().unique()]
            in_direct = "yes" if any(_norm(v) == wanted for v in dvals) else "NO"
            n_direct = len(dvals)
        else:
            in_direct, n_direct = "?", 0
        print("  %-12s tape: exact %-4s normalised %-4s (%d values)   "
              "DIRECT book: present %-4s (%d values)"
              % (wanted, "yes" if exact else "NO", "yes" if loose else "NO",
                 len(vals), in_direct, n_direct))

    print("VALUE MATCHING — does the book carry the value the question names?")
    print("  (exact = character-for-character; normalised = ignoring case,")
    print("   underscores and hyphens. 'NO / yes' means a normalisation bug;")
    print("   'NO / NO' means the book spells it differently altogether.)")
    for col, wanted in ((PRODUCT, "drawdown"), (PRODUCT, "lump sum"),
                        (GEO, "london")):
        _match(col, wanted)
    print()

    if len(direct) and AGE in cols and LTV in cols:
        print("THE FAILING PREDICATE, ONE TERM AT A TIME (direct book)")
        d_age = direct[AGE] > 55
        d_ltv = direct[LTV] > 50
        print("  direct & age > 55        ", _yes(d_age.any()))
        print("  direct & ltv > 50        ", _yes(d_ltv.any()))
        print("  direct & both            ", _yes((d_age & d_ltv).any()))
        print()
        # WHERE the high-LTV loans actually are. "> 50 on the tape but not in
        # direct" makes 18 refusals correct answers rather than defects, so it
        # is worth stating rather than inferring.
        print("  ltv > 50 anywhere        ", _yes((df[LTV] > 50).any()))
        print("  ltv > 40 in direct       ", _yes((direct[LTV] > 40).any()))
        print("  ltv > 30 in direct       ", _yes((direct[LTV] > 30).any()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
