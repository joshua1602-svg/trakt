#!/usr/bin/env python3
"""migration_phase0/intent_vocabulary_census.py

Blast-radius census for the governed-intent vocabulary alignment that makes a
bare ``case``/``cases`` stop asserting PIPELINE dataset intent.

Two passes, and the method matters:

  STATIC   every one of the 882 distinct corpus questions, through
           `intent.classify` and `workspace.resolve_dataset`. Cheap, so the
           denominator is the whole corpus and nothing is sampled.
  EXECUTED only the questions whose STATIC reading moved, run end to end for
           route / ok / answer. Executing all 882 twice is hours; executing the
           moved set is minutes and is the only set that CAN move downstream —
           a question whose family, requirements and dataset are all unchanged
           cannot route or answer differently because of this change.

    python -m migration_phase0.intent_vocabulary_census before|after
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")

#: The corpus contains bare-`case` questions, but every one of them ALSO says
#: "pipeline", so a census confined to it would be blind to the exact shape this
#: change exists for. These probes carry that shape: the P1C golden-bank
#: question that started it, the brief's worked examples, and a bare
#: stratification. Executed alongside the corpus hits, before and after.
PROBES = (
    "Which region gained the most cases since last month?",
    "How many cases are there?",
    "cases by region",
    "How many pipeline cases are there?",
    "pipeline cases by stage",
    "open pipeline cases",
    "How many applications are there?",
    "How many KFIs are there?",
    "How many offers are outstanding?",
    "Forecast application volumes next month",
    "Forecast case completions next quarter",
)


def _questions() -> List[str]:
    out: List[str] = []
    seen = set()
    for f in CORPORA:
        for row in json.loads((_REPO / f).read_text())["rows"]:
            q = row.get("question") or ""
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def _env() -> str:
    import logging
    import warnings
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def _classify_one(q: str) -> Dict[str, Any]:
    from mi_workflows.analytical import intent as it
    from mi_agent_api.workspace import resolve_dataset
    r = it.classify(q, spec=None)
    return {"families": list(r.families),
            "requirements": sorted(set(r.requirements)),
            "materiallyAnalytical": bool(r.materially_analytical),
            "dataset": resolve_dataset(q)}


def static_pass() -> Dict[str, Any]:
    from mi_workflows.analytical import intent as it
    from mi_agent_api.workspace import resolve_dataset

    rows = {}
    for q in _questions():
        r = it.classify(q, spec=None)
        rows[q] = {
            "families": list(r.families),
            "requirements": sorted(set(r.requirements)),
            "materiallyAnalytical": bool(r.materially_analytical),
            "dataset": resolve_dataset(q),
        }
    return rows


def executed(questions: List[str], client_id: str) -> Dict[str, Any]:
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(client_id)
    out = {}
    for q in questions:
        env = execute_governed_mi_query(MiQueryRequest(question=q), ctx).result or {}
        md = env.get("metadata") or {}
        out[q] = {"route": md.get("route"), "ok": env.get("ok"),
                  "dataset": md.get("datasetContext"),
                  "controlledRefusal": env.get("controlledRefusal"),
                  "answer": (env.get("answer") or env.get("error") or "")[:200]}
    return out


def main(argv: List[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("before", "after"):
        print(__doc__)
        return 2
    phase = argv[1]
    client_id = _env()
    out_dir = _REPO / "migration_phase0"

    rows = static_pass()
    print(f"STATIC pass ({phase}): {len(rows)} distinct corpus questions")
    probe_static = {q: v for q, v in
                    ((q, _classify_one(q)) for q in PROBES)}

    if phase == "before":
        (out_dir / "INTENT_VOCAB_BEFORE.json").write_text(
            json.dumps({**rows, **probe_static}, indent=2, default=str))
        # Everything a bare `case`/`cases` could possibly reach, so the
        # executed comparison set is fixed BEFORE the change rather than
        # derived from it.
        import re
        hits = [q for q in rows if re.search(r"\bcases?\b", q, flags=re.I)]
        touched = hits + [q for q in PROBES if q not in hits]
        print(f"corpus questions containing `case`/`cases` : {len(hits)}")
        print(f"probes added                               : {len(touched) - len(hits)}")
        (out_dir / "INTENT_VOCAB_EXEC_BEFORE.json").write_text(
            json.dumps(executed(touched, client_id), indent=2, default=str))
        print(f"executed and recorded                      : {len(touched)}")
        return 0

    before = json.loads((out_dir / "INTENT_VOCAB_BEFORE.json").read_text())
    exec_before = json.loads((out_dir / "INTENT_VOCAB_EXEC_BEFORE.json").read_text())
    now = {**rows, **probe_static}
    assert set(before) == set(now), "the census population moved; census invalid"

    rows = now
    moved = {q: (before[q], rows[q]) for q in rows if before[q] != rows[q]}
    fam = [q for q in moved if moved[q][0]["families"] != moved[q][1]["families"]]
    req = [q for q in moved if moved[q][0]["requirements"] != moved[q][1]["requirements"]]
    ds = [q for q in moved if moved[q][0]["dataset"] != moved[q][1]["dataset"]]

    print(f"\nSTATIC movement over {len(rows)} questions")
    print(f"  intent FAMILY changed       : {len(fam)}")
    print(f"  intent REQUIREMENT changed  : {len(req)}")
    print(f"  DATASET changed             : {len(ds)}")
    for q in sorted(moved)[:40]:
        b, a = moved[q]
        print(f"    {b['families']} -> {a['families']}  req {b['requirements']} -> "
              f"{a['requirements']}  ds {b['dataset']} -> {a['dataset']}\n"
              f"      :: {q[:82]}")

    exec_after = executed(sorted(exec_before), client_id)
    assert set(exec_after) == set(exec_before), "the executed set moved"
    print(f"\nEXECUTED comparison set (fixed before the change): {len(exec_after)}")
    route_moved = [q for q in exec_after
                   if exec_before[q]["route"] != exec_after[q]["route"]]
    ok_moved = [q for q in exec_after if exec_before[q]["ok"] != exec_after[q]["ok"]]
    ans_moved = [q for q in exec_after
                 if exec_before[q]["answer"] != exec_after[q]["answer"]]
    print(f"  ROUTE changed               : {len(route_moved)}")
    print(f"  ok/refusal changed          : {len(ok_moved)}")
    print(f"  ANSWER TEXT changed         : {len(ans_moved)}")
    for q in ok_moved + [q for q in route_moved if q not in ok_moved]:
        b, a = exec_before[q], exec_after[q]
        print(f"    route {b['route']} -> {a['route']}  ok {b['ok']} -> {a['ok']}\n"
              f"      :: {q[:82]}\n"
              f"      before: {b['answer'][:110]}\n"
              f"      after : {a['answer'][:110]}")

    (out_dir / "INTENT_VOCAB_AFTER.json").write_text(
        json.dumps({"static": rows, "executed": exec_after,
                    "staticMoved": sorted(moved),
                    "routeMoved": route_moved, "okMoved": ok_moved,
                    "answerMoved": ans_moved}, indent=2, default=str))
    print(f"\nwritten : migration_phase0/INTENT_VOCAB_AFTER.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
