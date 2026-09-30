"""Run a bank of questions through the governed MI Query Agent, in process.

For the capability audit: does the agent answer what the dashboard shows,
across funded, pipeline, forecast and limits? Run it on the MI App Service
(SSH) so it reads the live published data through exactly the path
``POST /mi/query`` uses — ``mi_service.execute_governed_mi_query`` — with no
bearer token and no gateway timeout.

    python -m mi_agent_api.question_bank                       # default audit set
    python -m mi_agent_api.question_bank --categories pipeline,forecast
    python -m mi_agent_api.question_bank --ids pipeline_003,forecast_004
    python -m mi_agent_api.question_bank --lens direct_001 --out /home/qb.jsonl

One line per question (ANSWERED / REFUSED / ERROR, route, seconds, the first
words of the answer), a summary by category, and a JSONL file with the full
answer text for review.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_BANK_DIR = _ROOT / "config" / "mi" / "golden_questions"
DEFAULT_BANKS = [_BANK_DIR / "ere_mi_questions.yaml",
                 _BANK_DIR / "ere_capability_supplement.yaml"]
#: Funded, pipeline, forecast and limits (current and forward).
DEFAULT_CATEGORIES = ["funded_kpi", "funded_breakdown_1d", "pipeline",
                      "pipeline_evolution", "forecast", "forecast_scale",
                      "risk_limits", "risk_limits_forward"]


def load_bank(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text) or {}
            items = data.get("questions", data) if isinstance(data, dict) else data
        elif path.suffix == ".json":
            data = json.loads(text)
            items = data.get("rows", data) if isinstance(data, dict) else data
        else:
            items = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, item in enumerate(items):
            if isinstance(item, str):
                item = {"id": f"{path.stem}_{i + 1:03d}", "category": "adhoc",
                        "question": item}
            rows.append(item)
    return rows


#: The governed-plan serving decision for the question being asked — captured
#: from the canary's evidence record (NEW = the plan path answered;
#: LEGACY_FALLBACK = the legacy path did, with the reason).
_SERVING: Dict[str, Any] = {}


def _capture_serving() -> None:
    """Wrap the plan-serving evidence writer so each run records which path
    answered. A no-op on a build without the governed-plan architecture."""
    try:
        from mi_agent import plan_shadow_evidence as evidence
    except Exception:  # noqa: BLE001
        return
    if getattr(evidence.write, "_question_bank", False):
        return
    original = evidence.write

    def write(body, *a, **kw):
        if isinstance(body, dict) and isinstance(body.get("serving"), dict):
            _SERVING.update(body["serving"])
        return original(body, *a, **kw)
    write._question_bank = True  # type: ignore[attr-defined]
    evidence.write = write


def _ask(question: str, *, portfolio: Optional[str], lens: Optional[str],
         principal: str):
    from mi_agent_api.dependencies import default_tenant_id
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext
    ctx = ExecutionContext.for_internal(default_tenant_id(), actor_id=principal)
    return execute_governed_mi_query(
        MiQueryRequest(question=question, portfolio_id=portfolio,
                       source_portfolio_lens=lens), ctx)


def run_one(row: Dict[str, Any], *, portfolio: Optional[str],
            lens: Optional[str], principal: str = "question-bank") -> Dict[str, Any]:
    from trakt_core import perf as _perf

    t0 = time.monotonic()
    _SERVING.clear()
    timing: Dict[str, Any] = {}
    try:
        # WHERE THE TIME GOES, per question: the same stage collector the HTTP
        # layer opens for every request (and reports as `Server-Timing`),
        # opened here because the bank calls the service in-process.
        with _perf.collect(route="question_bank") as collector:
            result = _ask(row["question"], portfolio=portfolio, lens=lens,
                          principal=principal)
            if collector is not None:
                snap = collector.snapshot()
                timing = {"total_ms": snap["total_ms"], "stages": snap["stages"]}
        res = result.result or {}
        meta = res.get("metadata") or {}
        ok = bool(res.get("ok"))
        answer = str(res.get("answer") or res.get("error") or "")
        outcome = "ANSWERED" if ok else "REFUSED"
        route = meta.get("route")
        view = meta.get("datasetContext")
    except Exception as exc:  # noqa: BLE001 - an audit records, never stops
        outcome, answer, route, view = "ERROR", f"{type(exc).__name__}: {exc}", None, None
    served = _SERVING.get("response_served_from") or "-"
    return {"id": row.get("id"), "category": row.get("category"),
            "question": row["question"], "outcome": outcome, "route": route,
            "view": view, "seconds": round(time.monotonic() - t0, 1),
            "served": served, "serving_reason": _SERVING.get("reason") or "",
            "timing": timing, "answer": answer}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", action="append", type=Path,
                    help="bank file (yaml / json / txt); repeatable")
    ap.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                    help="comma-separated categories, or 'all'")
    ap.add_argument("--ids", default="", help="comma-separated question ids")
    ap.add_argument("--portfolio", default=None, help="portfolioId, e.g. ERE/2026-08-31")
    ap.add_argument("--lens", default=None, help="source-portfolio lens, e.g. direct_001")
    ap.add_argument("--principal", default="question-bank",
                    help="actor id to ask as; name one listed in "
                         "MI_AGENT_PLAN_SERVE_PRINCIPALS to exercise the "
                         "governed-plan path (MI_AGENT_PLAN_SERVE=canary)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("question_bank_results.jsonl"))
    args = ap.parse_args(argv)

    import logging
    import os
    logging.disable(logging.WARNING)
    _capture_serving()
    mode = os.environ.get("MI_AGENT_PLAN_SERVE", "off")
    listed = args.principal.strip().lower() in {
        p.strip().lower() for p in
        os.environ.get("MI_AGENT_PLAN_SERVE_PRINCIPALS", "").split(",")}
    print(f"MI_AGENT_PLAN_SERVE={mode}; principal {args.principal!r} "
          f"{'IS' if listed else 'is NOT'} on the allow-list", flush=True)
    rows = load_bank(args.bank or DEFAULT_BANKS)
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        rows = [r for r in rows if r.get("id") in wanted]
    elif args.categories != "all":
        cats = {s.strip() for s in args.categories.split(",") if s.strip()}
        rows = [r for r in rows if r.get("category") in cats]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} question(s)\n", flush=True)

    tally: Dict[str, Counter] = defaultdict(Counter)
    routes: Counter = Counter()
    served: Counter = Counter()
    stage_ms: Dict[str, List[float]] = defaultdict(list)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            rec = run_one(row, portfolio=args.portfolio, lens=args.lens,
                          principal=args.principal)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            tally[rec["category"]][rec["outcome"]] += 1
            routes[rec["route"] or "(point-in-time)"] += 1
            served[rec["served"] + (f" ({rec['serving_reason']})"
                                    if rec["serving_reason"] else "")] += 1
            snippet = " ".join(rec["answer"].split())[:110]
            print(f"{rec['outcome']:<8} {rec['id']:<22} {rec['served']:<6} "
                  f"{str(rec['route'] or '-'):<26} "
                  f"{rec['seconds']:>5}s  {row['question'][:60]!r}\n"
                  f"{'':8} -> {snippet}", flush=True)
            stages = (rec.get("timing") or {}).get("stages") or {}
            for name, ms in stages.items():
                stage_ms[name].append(float(ms))
            if stages:
                print(f"{'':8} time: {_heaviest(stages)}", flush=True)

    print("\nSUMMARY  (answered / refused / error)")
    for cat in sorted(tally):
        c = tally[cat]
        print(f"  {cat:<22} {c['ANSWERED']:>3} / {c['REFUSED']:>3} / {c['ERROR']:>3}")
    print("\nROUTES")
    for route, n in routes.most_common():
        print(f"  {route:<28} {n}")
    print("\nSERVED BY  (NEW = governed plan; '-' = canary not engaged)")
    for how, n in served.most_common():
        print(f"  {how:<60} {n}")
    if stage_ms:
        print("\nTIME BY STAGE  (median / max seconds, questions that ran it)")
        for name, values in sorted(stage_ms.items(),
                                   key=lambda kv: -statistics.median(kv[1])):
            print(f"  {name:<44} {statistics.median(values) / 1000:>6.1f} "
                  f"{max(values) / 1000:>7.1f}  {len(values):>3}")
    print(f"\nfull answers: {args.out.resolve()}")
    return 0


def _heaviest(stages: Dict[str, Any], limit: int = 4) -> str:
    """The stages that took longest, in seconds — names only, never data."""
    top = sorted(stages.items(), key=lambda kv: -float(kv[1]))[:limit]
    return ", ".join(f"{name} {float(ms) / 1000:.1f}s" for name, ms in top)


if __name__ == "__main__":
    sys.exit(main())
