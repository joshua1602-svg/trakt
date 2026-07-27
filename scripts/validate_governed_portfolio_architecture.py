#!/usr/bin/env python3
"""Manual product validation of the governed multi-portfolio MI architecture.

Drives the LIVE API (the same FastAPI app the React workspace calls) against an
assembled ERE platform tape and prints, in business terms, what a user would see:

    Total / Direct / Acquired / each individual portfolio
    · loan counts and balances, and whether they reconcile
    · which analyses each scope may open, and why not when it may not
    · the pipeline's originating-only participation, in Total context
    · the forecast treatment and runoff disclosure per portfolio
    · an MI Agent answer with partial field coverage

Usage:
    python scripts/validate_governed_portfolio_architecture.py            # ERE today
    python scripts/validate_governed_portfolio_architecture.py --expanded # + _002 books
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("TRAKT_RUNTIME_MODE", "test")
os.environ["MI_AGENT_AUTH_ENABLED"] = "false"

from tests.test_platform_assembly_all_portfolios import (  # noqa: E402
    ACQUIRED_ROWS, CLIENT, DIRECT_ROWS, _canonical, _Ctx, _DIRECT_ONLY)

TODAY = [("direct_001", "direct", DIRECT_ROWS),
         ("acquired_001", "acquired", ACQUIRED_ROWS)]
EXPANDED = TODAY + [("direct_002", "direct", 41), ("acquired_002", "acquired", 120)]

RULE = "─" * 78


def _heading(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def _assemble(portfolios):
    tmp = tempfile.TemporaryDirectory(prefix="ere_validation_")
    ctx = _Ctx(Path(tmp.name))
    last_local = last_pid = None
    for pid, ptype, rows in portfolios:
        extra = _DIRECT_ONLY if ptype == "direct" else ()
        last_local = ctx.accept(
            pid, _canonical(pid, rows, portfolio_type=ptype, extra_columns=extra))
        ctx.register(pid)
        last_pid = pid
    ctx.refresh(last_pid, last_local)
    frame = ctx.platform()
    tmp.cleanup()
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", action="store_true",
                        help="add direct_002 + acquired_002 (no code change)")
    args = parser.parse_args()

    portfolios = EXPANDED if args.expanded else TODAY
    platform = _assemble(portfolios)

    from engine import platform_assembler as pa
    workdir = tempfile.TemporaryDirectory()
    path = Path(workdir.name) / pa.PLATFORM_CANONICAL_NAME
    platform.to_csv(path, index=False)
    os.environ["MI_AGENT_PLATFORM_CANONICAL"] = str(path)

    from fastapi.testclient import TestClient
    from mi_agent_api import data_source as ds
    ds.reset_cache()
    from mi_agent_api.app import app
    client = TestClient(app)

    _heading(f"PLATFORM TAPE — {len(platform):,} loans across "
             f"{platform['source_portfolio_id'].nunique()} source portfolios")
    counts = platform.groupby("source_portfolio_id").size()
    for pid, n in counts.items():
        ptype = platform.loc[platform["source_portfolio_id"] == pid,
                             "source_portfolio_type"].iloc[0]
        print(f"  {pid:16} {ptype:9} {n:>6,} loans")

    # ---- 1. the governed hierarchy --------------------------------------- #
    index = client.get("/mi/portfolio-context").json()
    _heading("1. GOVERNED HIERARCHY (from /mi/portfolio-context)")
    for ctx in index["contexts"]:
        indent = "  " + "    " * ctx["depth"]
        print(f"{indent}{ctx['label']:<16} [{ctx['context_id']}] "
              f"→ {', '.join(ctx['portfolio_ids'])}")

    # ---- 2. funded reconciliation ---------------------------------------- #
    def snapshot(context):
        return client.get("/mi/snapshot",
                          params={"portfolioId": f"{CLIENT}/latest",
                                  "portfolioContext": context}).json()

    _heading("2. FUNDED MI BY SCOPE (from /mi/snapshot)")
    print(f"  {'scope':<16} {'loans':>8} {'balance':>18}   portfolios in scope")
    contexts = [c["context_id"] for c in index["contexts"]]
    totals = {}
    for context in contexts:
        body = snapshot(context)
        totals[context] = (body["loan_count"], body["current_outstanding_balance"])
        in_scope = ", ".join(body["portfolioScope"]["portfolios_in_scope"])
        print(f"  {context:<16} {body['loan_count']:>8,} "
              f"{body['current_outstanding_balance']:>18,.2f}   {in_scope}")

    print()
    direct_n, direct_bal = totals["direct"]
    acquired_n, acquired_bal = totals["acquired"]
    total_n, total_bal = totals["total"]
    ok_n = total_n == direct_n + acquired_n
    ok_bal = abs(total_bal - (direct_bal + acquired_bal)) < 0.01
    print(f"  Total = Direct + Acquired (loans)   {total_n:,} = {direct_n:,} + "
          f"{acquired_n:,}   {'OK' if ok_n else 'MISMATCH'}")
    print(f"  Total = Direct + Acquired (balance) {'OK' if ok_bal else 'MISMATCH'}")
    pair = sum(totals[c][0] for c in ("direct_001", "acquired_001") if c in totals)
    print(f"  Total is NOT direct_001 + acquired_001: {total_n:,} vs {pair:,}   "
          f"{'OK' if total_n >= pair else 'MISMATCH'}"
          f"{' (equal — only two books onboarded)' if total_n == pair else ''}")

    # ---- 3. capability matrix -------------------------------------------- #
    _heading("3. CAPABILITY MATRIX (what each scope may open)")
    caps = ["funded", "risk", "evolution", "cohorts", "pipeline",
            "origination_forecast", "runoff_forecast", "consolidated_forecast"]
    print(f"  {'scope':<16} " + " ".join(f"{c[:6]:>7}" for c in caps))
    for ctx in index["contexts"]:
        marks = " ".join(
            f"{'yes' if ctx['capabilities'][c]['enabled'] else 'NO':>7}" for c in caps)
        print(f"  {ctx['context_id']:<16} {marks}")

    print("\n  Reasons where an analysis is refused:")
    for ctx in index["contexts"]:
        for cap in ("pipeline", "origination_forecast", "runoff_forecast"):
            state = ctx["capabilities"][cap]
            if not state["enabled"]:
                print(f"    {ctx['context_id']:<16} {cap:<22} "
                      f"{state['reason_code']}: {state['detail']}")

    # ---- 4. pipeline in Total context ------------------------------------ #
    _heading("4. PIPELINE PARTICIPATION IN THE TOTAL CONTEXT")
    total_pipeline = next(c for c in index["contexts"]
                          if c["context_id"] == "total")["capabilities"]["pipeline"]
    print(f"  Total workspace selected — context preserved, never switched.")
    print(f"  Pipeline enabled: {total_pipeline['enabled']}")
    print(f"  Pipeline includes: "
          f"{', '.join(total_pipeline['contributing_portfolios']) or '(none)'}")
    print(f"  Excluded from pipeline: "
          f"{', '.join(total_pipeline['excluded_portfolios']) or '(none)'}")
    print(f"  Reason: {total_pipeline['detail'] or total_pipeline['reason_code']}")

    refusal = client.get("/mi/pipeline/snapshot",
                         params={"portfolioId": f"{CLIENT}/latest",
                                 "portfolioContext": "acquired"}).json()
    print(f"\n  Acquired scope → /mi/pipeline/snapshot")
    print(f"    applicable: {refusal.get('applicable')}   "
          f"reasonCode: {refusal.get('reasonCode')}")
    print(f"    context preserved as: "
          f"{(refusal.get('portfolioScope') or {}).get('context_id')}")

    # ---- 5. forecast treatment ------------------------------------------- #
    _heading("5. CONSOLIDATED FORECAST BY PORTFOLIO")
    body = client.get("/mi/forecast/snapshot",
                      params={"portfolioId": f"{CLIENT}/latest",
                              "portfolioContext": "total"}).json()
    projections = body.get("portfolioProjections") or {}
    print(f"  {'portfolio':<16} {'treatment':<16} {'current':>16} "
          f"{'new orig':>12} {'projected':>16}  runoff")
    for entry in projections.get("portfolios", []):
        print(f"  {entry['portfolioId']:<16} {entry['forecastTreatment']:<16} "
              f"{entry['currentBalance']:>16,.2f} "
              f"{entry['expectedNewOriginations']:>12,.2f} "
              f"{entry['projectedBalance']:>16,.2f}  "
              f"{'modelled' if entry['runoffModelled'] else 'NOT MODELLED'}")
    print(f"\n  Total projected: {projections.get('totalProjectedBalance', 0):,.2f}")
    print(f"  Runoff modelled:     {projections.get('runoffModelled') or '(none)'}")
    print(f"  Runoff NOT modelled: {projections.get('runoffNotModelled') or '(none)'}")
    print(f"  Basis: {projections.get('basis', '')}")

    # ---- 6. MI Agent depth + coverage disclosure ------------------------- #
    _heading("6. MI AGENT — SAME DEPTH FOR EVERY BOOK, COVERAGE DISCLOSED")
    questions = ["What is the total current balance?",
                 "How many loans are in the book?"]
    for context in contexts:
        for question in questions:
            answer = client.post("/mi/query", json={
                "question": question, "sourcePortfolioLens": context}).json()
            coverage = answer.get("portfolioCoverage") or {}
            print(f"  {context:<16} {question:<36} ok={answer.get('ok')!s:<6} "
                  f"used={','.join(coverage.get('portfolios_used', [])) or '-'}")

    print("\n  Coverage disclosure for a Total answer:")
    answer = client.post("/mi/query", json={
        "question": "What is the total current balance?",
        "sourcePortfolioLens": "total"}).json()
    coverage = answer["portfolioCoverage"]
    print(f"    fully consolidated: {coverage['is_fully_consolidated']}")
    print(f"    {coverage['disclosure']}")

    # ---- 7. channel parity ------------------------------------------------ #
    _heading("7. CHANNEL PARITY — REACT vs COPILOT")
    os.environ["TRAKT_COPILOT_AUTH_MODE"] = "disabled"
    copilot = client.post("/v1/copilot/mi/query", json={
        "question": "What is the total current balance?"}).json()
    react_scope = answer["governance"]["scope"]
    same = copilot.get("portfolioCoverage") == react_scope
    print(f"  React governance.scope == Copilot portfolioCoverage: "
          f"{'IDENTICAL' if same else 'DIVERGED'}")
    print(f"  portfolios_used: {react_scope['portfolios_used']}")

    workdir.cleanup()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
