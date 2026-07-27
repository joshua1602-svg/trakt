#!/usr/bin/env python3
"""Manual product validation of the LOCAL MI Agent workspace (ports 8000 / 5173).

Exercises the running stack the way the browser does and prints, in business
terms, the state of every defect manual testing found:

    1  CLIENT is the tenant, and contains no source portfolio ids
    2  PORTFOLIO carries Total / Direct / Acquired / each source portfolio
    3  the two cannot conflict, and portfolio scope controls the numbers
    4  borrower age renders for the acquired book, from its DOBs
    5  region renders for the acquired book, harmonised for consolidation
    6  pipeline loads, is eligible only where a book originates, and says so
    7  the MI Agent chat answers 200 with governed scope + field coverage

Usage:
    python scripts/validate_local_mi_workspace.py                  # via :8000
    python scripts/validate_local_mi_workspace.py --base http://127.0.0.1:5173
        (the second form goes through the Vite dev proxy — i.e. exactly the
         path the browser takes, which is where the 404 was seen)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

RULE = "─" * 78


def _heading(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def _get(base: str, path: str, **params):
    url = f"{base}{path}"
    if params:
        from urllib.parse import urlencode
        url += "?" + urlencode({k: v for k, v in params.items() if v is not None})
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.status, json.loads(response.read().decode())


def _post(base: str, path: str, body: dict):
    request = urllib.request.Request(
        f"{base}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.status, json.loads(response.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000",
                        help="API base, or the Vite dev server to test the proxy")
    parser.add_argument("--run", default=None, help="reporting run id")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    try:
        _, health = _get(base, "/health")
    except (urllib.error.URLError, OSError) as exc:
        print(f"Cannot reach {base}: {exc}")
        return 2
    print(f"API: {base}  ·  source: {health.get('dataSource')}  "
          f"·  tenant: {(health.get('governance') or {}).get('tenantId')}")

    # ---- 1 + 2: the two axes ------------------------------------------- #
    _heading("1. CLIENT axis (the tenant) — /mi/snapshots")
    _, snapshots = _get(base, "/mi/snapshots")
    clients = []
    run_id = args.run
    for pf in snapshots.get("portfolios", []):
        runs = [(r["run_id"], r.get("reporting_date"), r.get("loan_count"))
                for r in pf.get("runs", [])]
        clients.append(pf["client_id"])
        print(f"  Client: {pf['client_id']:10} label={pf['label']:10} runs={runs}")
        if run_id is None and runs:
            run_id = runs[-1][0]

    _heading("2. PORTFOLIO axis (source scope) — /mi/portfolio-context")
    _, index = _get(base, "/mi/portfolio-context")
    contexts = index.get("contexts", [])
    for ctx in contexts:
        print(f"  {'    ' * ctx['depth']}{ctx['label']:<16} [{ctx['context_id']}] "
              f"→ {', '.join(ctx['portfolio_ids'])}")

    portfolio_ids = {p["portfolio_id"] for p in index.get("portfolios", [])}
    overlap = set(clients) & portfolio_ids
    print(f"\n  Source portfolios in the CLIENT selector: "
          f"{sorted(overlap) or 'none'}   {'OK' if not overlap else 'CONFLICT'}")

    # ---- 3: scope controls the numbers ---------------------------------- #
    _heading("3. PORTFOLIO scope controls the workspace — /mi/snapshot")
    pid = f"{clients[0]}/{run_id}" if clients else run_id
    counts = {}
    print(f"  {'portfolio':<16} {'loans':>8} {'balance':>18}   in scope")
    for ctx in contexts:
        _, body = _get(base, "/mi/snapshot", portfolioId=pid,
                       portfolioContext=ctx["context_id"])
        counts[ctx["context_id"]] = body.get("loan_count")
        scope = (body.get("portfolioScope") or {}).get("portfolios_in_scope", [])
        print(f"  {ctx['context_id']:<16} {body.get('loan_count', 0):>8,} "
              f"{body.get('current_outstanding_balance', 0):>18,.2f}   "
              f"{', '.join(scope)}")
    if {"total", "direct", "acquired"} <= counts.keys():
        ok = counts["total"] == counts["direct"] + counts["acquired"]
        print(f"\n  Total = Direct + Acquired: {counts['total']:,} = "
              f"{counts['direct']:,} + {counts['acquired']:,}   "
              f"{'OK' if ok else 'MISMATCH'}")

    # ---- 4 + 5: acquired dimensions ------------------------------------- #
    _heading("4/5. BORROWER AGE and REGION by portfolio — funded stratifications")
    for context in [c["context_id"] for c in contexts]:
        _, body = _get(base, "/mi/snapshot", portfolioId=pid, portfolioContext=context)
        strat = {s["key"]: s for s in body.get("stratifications", [])}
        age = strat.get("age", {})
        region = strat.get("region", {})
        print(f"  {context:<16} age={age.get('availability', '-'):<20} "
              f"region={region.get('availability', '-'):<20} "
              f"regions={[b['label'] for b in region.get('bars', [])]}")

    # ---- 6: pipeline ----------------------------------------------------- #
    _heading("6. PIPELINE eligibility and attribution")
    print(f"  pipeline portfolios: {index.get('pipeline_portfolios')}   "
          f"attributed: {index.get('pipeline_attributed')}")
    for ctx in contexts:
        state = ctx["capabilities"]["pipeline"]
        print(f"  {ctx['context_id']:<16} enabled={str(state['enabled']):<6}"
              f" contributing={state['contributing_portfolios']}"
              f" excluded={state['excluded_portfolios']}")
        if state.get("detail"):
            print(f"      {state['detail']}")
    status, pipeline = _get(base, "/mi/pipeline/snapshot", portfolioId=pid,
                            portfolioContext="total")
    print(f"\n  /mi/pipeline/snapshot (Total)  HTTP {status}  ok={pipeline.get('ok')}"
          f"  rows={pipeline.get('pipelineRowCount')}"
          f"  context preserved as "
          f"{(pipeline.get('portfolioScope') or {}).get('context_id')}")
    status, refused = _get(base, "/mi/pipeline/snapshot", portfolioId=pid,
                           portfolioContext="acquired")
    print(f"  /mi/pipeline/snapshot (Acquired)  applicable={refused.get('applicable')}"
          f"  reason={refused.get('reasonCode')}"
          f"  context preserved as "
          f"{(refused.get('portfolioScope') or {}).get('context_id')}")

    # ---- 7: chat ---------------------------------------------------------- #
    _heading("7. MI AGENT CHAT — no 404, governed scope + coverage")
    questions = ["What is the total current balance?", "Show average borrower age."]
    for context in [c["context_id"] for c in contexts]:
        for question in questions:
            try:
                status, answer = _post(base, "/mi/query", {
                    "question": question, "sourcePortfolioLens": context,
                    "portfolioId": pid})
            except urllib.error.HTTPError as exc:
                print(f"  {context:<16} {question:<36} HTTP {exc.code}")
                continue
            coverage = answer.get("portfolioCoverage") or {}
            print(f"  {context:<16} {question:<36} HTTP {status} ok={str(answer.get('ok')):<5}"
                  f" used={','.join(coverage.get('portfolios_used', [])) or '-'}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
