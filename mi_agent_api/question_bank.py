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
#: The held-out variants (P0 design §25): each asks what one bank question
#: asks, in other words, and is compared with it on the same deploy.
HOLDOUT_BANK = _BANK_DIR / "holdout_variants_20260930b.yaml"
#: The conversation bank (P0 design §34, D24-D25): each answered follow-up is
#: scored against a stand-alone twin. The twins that are not production-bank
#: questions are asked on their own first, so each twin's outcome is known.
CONVERSATION_BANK = _BANK_DIR / "conversation_bank_20261001.yaml"
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
#: from the canary's evidence record (NEW = the plan path answered; DECLINED =
#: the plan path declined in words, with the reason, and legacy was not asked
#: (owner decision D18); LEGACY_FALLBACK = the legacy path answered).
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
         principal: str, continuation: Optional[str] = None,
         conversation_id: Optional[str] = None):
    from mi_agent_api.dependencies import default_tenant_id
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext
    ctx = ExecutionContext.for_internal(default_tenant_id(), actor_id=principal)
    return execute_governed_mi_query(
        MiQueryRequest(question=question, portfolio_id=portfolio,
                       source_portfolio_lens=lens, continuation=continuation,
                       conversation_id=conversation_id), ctx)


def holdout_rows(mode: str) -> List[Dict[str, Any]]:
    """The held-out variants to ask.

    ``all``: every variant — compared with a whole-bank run on the same
    deploy. ``recent``: the variants of the questions changed for since
    2026-09-29, each after the bank question it varies, so one run holds both
    sides of every comparison. ``unspent``: the variants not yet used to fix
    anything — what a sign-off asks beside the whole bank (D13).
    """
    variants = load_bank([HOLDOUT_BANK])
    if mode == "all":
        return variants
    if mode == "unspent":
        # The variants no fix was made against: every row but the
        # `holdout_recent` ones, whose findings were fixed (D15-D17 and since)
        # — a fixed finding spends its row (§25).
        return [v for v in variants if v.get("category") != "holdout_recent"]
    recent = [v for v in variants if v.get("category") == "holdout_recent"]
    bank = {r.get("id"): r for r in load_bank(DEFAULT_BANKS)}
    rows: List[Dict[str, Any]] = []
    for original in dict.fromkeys(v["variant_of"] for v in recent):
        rows.append(bank[original])
        rows.extend(v for v in recent if v["variant_of"] == original)
    return rows


def twin_rows() -> List[Dict[str, Any]]:
    """The conversation bank's NEW stand-alone twins, each once.

    A twin that names a production-bank question (``bank_id``) is that
    question, already measured on every full-bank run; only the others are
    asked here. Each keeps the id of the first turn that names it.
    """
    data = yaml.safe_load(CONVERSATION_BANK.read_text(encoding="utf-8")) or {}
    rows: List[Dict[str, Any]] = []
    seen = set()
    for conversation in data.get("conversations") or ():
        for i, turn in enumerate(conversation.get("turns") or ()):
            twin = turn.get("twin") or {}
            question = twin.get("question")
            if not question or twin.get("bank_id") or question in seen:
                continue
            seen.add(question)
            rows.append({"id": f"{conversation['id']}_t{i}_twin",
                         "category": "conversation_twin", "question": question})
    return rows


def run_one(row: Dict[str, Any], *, portfolio: Optional[str],
            lens: Optional[str], principal: str = "question-bank",
            continuation: Optional[str] = None,
            conversation_id: Optional[str] = None) -> Dict[str, Any]:
    from trakt_core import perf as _perf
    from mi_agent_api import request_scope as _request_scope

    t0 = time.monotonic()
    _SERVING.clear()
    timing: Dict[str, Any] = {}
    try:
        # WHERE THE TIME GOES, per question: the same stage collector the HTTP
        # layer opens for every request (and reports as `Server-Timing`),
        # opened here because the bank calls the service in-process — and the
        # same REQUEST SCOPE beside it, so a storage revalidation is made once
        # per question as it is once per HTTP request. Without it the bank
        # measured several revalidations a question that no user pays.
        with _perf.collect(route="question_bank") as collector, \
                _request_scope.scope():
            result = _ask(row["question"], portfolio=portfolio, lens=lens,
                          principal=principal, continuation=continuation,
                          conversation_id=conversation_id)
            if collector is not None:
                snap = collector.snapshot()
                timing = {"total_ms": snap["total_ms"], "stages": snap["stages"],
                          "stage_calls": snap.get("stage_calls", {}),
                          "storage_calls": {
                              k: v for k, v in (snap.get("counters") or {}).items()
                              if k.startswith("storage.")}}
        res = result.result or {}
        meta = res.get("metadata") or {}
        ok = bool(res.get("ok"))
        answer = str(res.get("answer") or res.get("error") or "")
        outcome = "ANSWERED" if ok else "REFUSED"
        route = meta.get("route")
        view = meta.get("datasetContext")
        talk = res.get("conversation") or {}
        handed = talk.get("continuation")
    except Exception as exc:  # noqa: BLE001 - an audit records, never stops
        outcome, answer, route, view = "ERROR", f"{type(exc).__name__}: {exc}", None, None
        handed, talk = None, {}
    served = _SERVING.get("response_served_from") or "-"
    rec = {"id": row.get("id"), "category": row.get("category"),
           "question": row["question"], "outcome": outcome, "route": route,
           "view": view, "seconds": round(time.monotonic() - t0, 1),
           "served": served, "serving_reason": _SERVING.get("reason") or "",
           # The governed plan's identity: a follow-up and its stand-alone
           # twin that compiled the same plan asked the same question (§34).
           "plan_id": _SERVING.get("plan_id"),
           "timing": timing, "answer": answer}
    if continuation is not None or handed:
        # The token itself is not recorded: what the run needs is whether one
        # was handed back, what the message was read as, and the next turn
        # holds the token in memory.
        rec["conversation"] = {"replied_with_continuation": continuation is not None,
                               "continuation_issued": bool(handed),
                               "kind": talk.get("kind"),
                               "read_as": talk.get("readAs")}
        rec["_continuation"] = handed
    return rec


def conversation_rows(groups: str = "C") -> List[Dict[str, Any]]:
    """The conversation bank's conversations in `groups` (`all` for every
    group), as the turns a run plays in order (§34, §39): every turn the
    model reads. The `run: code` turns — the memory's mechanics — need no
    model and are enforced on every build
    (`test_conversation_bank_mechanics.py`)."""
    data = yaml.safe_load(CONVERSATION_BANK.read_text(encoding="utf-8")) or {}
    wanted = {g.strip().upper() for g in groups.split(",") if g.strip()}
    out: List[Dict[str, Any]] = []
    for conversation in data.get("conversations") or ():
        group = str(conversation.get("group") or "").upper()
        if "ALL" not in wanted and group not in wanted:
            continue
        for i, turn in enumerate(conversation.get("turns") or ()):
            if turn.get("run") != "live":
                continue
            out.append({"id": f"{conversation['id']}_t{i}",
                        "conversation": conversation["id"], "turn": i,
                        "category": f"conversation_{conversation['group']}",
                        "question": turn["question"], "expect": turn["expect"],
                        "twin": (turn.get("twin") or {}).get("question")})
    return out


def play_conversations(rows: List[Dict[str, Any]], *, portfolio: Optional[str],
                       lens: Optional[str], principal: str, stamp: str):
    """Each conversation turn by turn, every message sent with the
    continuation the agent's previous message handed back, then each turn's
    stand-alone twin — so a follow-up or a reply can be scored against the
    question it should equal (D25). A twin asked earlier in the run is not
    asked again: it is a stand-alone question on the same book, and its
    record is reused (marked). Yields each record as it is made."""
    held: Dict[str, Optional[str]] = {}
    twins: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cid = row["conversation"]
        rec = run_one(row, portfolio=portfolio, lens=lens, principal=principal,
                      continuation=held.get(cid) if row["turn"] else None,
                      conversation_id=f"{stamp}-{cid}")
        held[cid] = rec.pop("_continuation", None)
        rec.update(expect=row["expect"], conversation_turn=row["turn"],
                   conversation_id=cid)
        yield rec
        if row.get("twin"):
            if row["twin"] in twins:
                twin = dict(twins[row["twin"]], id=f"{row['id']}_twin",
                            reused_from=twins[row["twin"]]["id"], seconds=0.0,
                            timing={})
            else:
                twin = run_one({"id": f"{row['id']}_twin",
                                "category": "conversation_twin",
                                "question": row["twin"]},
                               portfolio=portfolio, lens=lens, principal=principal)
                twin.pop("_continuation", None)
                twin.pop("conversation", None)
                twins[row["twin"]] = twin
            twin = dict(twin, twin_of=row["id"])
            yield twin


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
    ap.add_argument("--signoff", action="store_true",
                    help="the D13 sign-off: the bank (by --categories), then "
                         "the unspent held-out variants, in one run")
    ap.add_argument("--conversations", default="",
                    help="play the conversation bank's conversations in these "
                         "groups (e.g. C, or all), each message sent with the "
                         "continuation the previous one handed back, then each "
                         "turn's stand-alone twin (§34, §39)")
    ap.add_argument("--holdout", choices=("all", "recent", "unspent"), default=None,
                    help="ask the held-out variants (all), or the recently "
                         "changed questions' variants each after its bank "
                         "question (recent), instead of a bank")
    ap.add_argument("--twins", action="store_true",
                    help="ask the conversation bank's new stand-alone twins "
                         "(those that are not production-bank questions)")
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
    if args.conversations:
        rows = conversation_rows(args.conversations)
        _switch_conversation_on_for_this_run()
    elif args.twins:
        rows = twin_rows()
    elif args.signoff:
        cats = {c.strip() for c in args.categories.split(",") if c.strip()}
        rows = [r for r in load_bank(args.bank or DEFAULT_BANKS)
                if args.categories == "all" or r.get("category") in cats]
        rows += holdout_rows("unspent")
    elif args.holdout:
        rows = holdout_rows(args.holdout)
    else:
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
    records = (play_conversations(rows, portfolio=args.portfolio, lens=args.lens,
                                  principal=args.principal,
                                  stamp=time.strftime("%Y%m%dT%H%M%S"))
               if args.conversations else
               (run_one(row, portfolio=args.portfolio, lens=args.lens,
                        principal=args.principal) for row in rows))
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in records:
            rec.pop("_continuation", None)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            tally[rec["category"]][rec["outcome"]] += 1
            routes[rec["route"] or "(point-in-time)"] += 1
            served[rec["served"] + (f" ({rec['serving_reason']})"
                                    if rec["serving_reason"] else "")] += 1
            snippet = " ".join(rec["answer"].split())[:110]
            print(f"{rec['outcome']:<8} {rec['id']:<22} {rec['served']:<6} "
                  f"{str(rec['route'] or '-'):<26} "
                  f"{rec['seconds']:>5}s  {rec['question'][:60]!r}\n"
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


def _switch_conversation_on_for_this_run() -> None:
    """The conversation is switched on in THIS process only, with a key made
    for the run, unless the deployment already has it on: a run measures the
    conversation without changing what any user of the service is served."""
    import os
    import secrets
    from mi_agent import conversation as convo
    if convo.enabled():
        print("conversation: on (the deployment's own setting)", flush=True)
        return
    os.environ[convo.SWITCH_ENV] = "on"
    os.environ[convo.KEY_ENV] = secrets.token_urlsafe(48)
    print("conversation: switched on for this run only", flush=True)


def _heaviest(stages: Dict[str, Any], limit: int = 4) -> str:
    """The stages that took longest, in seconds — names only, never data."""
    top = sorted(stages.items(), key=lambda kv: -float(kv[1]))[:limit]
    return ", ".join(f"{name} {float(ms) / 1000:.1f}s" for name, ms in top)


if __name__ == "__main__":
    sys.exit(main())
