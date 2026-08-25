#!/usr/bin/env python3
"""migration_phase0/c6_fixture_equivalence.py — C6 pipeline-family equivalence.

READ-ONLY. Drives `_route_evolution` DIRECTLY against the deterministic
five-week fixture, because the configured production discovery root carries zero
weekly extracts and every pipeline / stage / funnel question there answers "No
weekly pipeline extracts are available".

Everything this instrument proves is therefore **fixture-proven,
production-data-unexercised**. It is not production-delivered evidence and must
never be reported as such.

Run it once before the conversion and once after, then `--diff`.

    python -m migration_phase0.c6_fixture_equivalence --out before.json
    python -m migration_phase0.c6_fixture_equivalence --diff before.json after.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

FIXTURE = _REPO / "tests" / "fixtures" / "pipeline_history_5w"
CLIENT = "client_001"

#: The pipeline-family surface, covering all three route identities and every
#: governed stage. Written from the governed vocabulary, not from what the
#: implementation happens to answer.
QUESTIONS = (
    "Show pipeline amount by stage over time.",
    "Show pipeline cases by stage over time.",
    "Show stage migration over time.",
    "Show the KFI trend.",
    "Show the application trend.",
    "Show the offer trend.",
    "Show the completion trend.",
    "Show pipeline amount evolution by month.",
    "Show pipeline case count evolution by month.",
    # Governed spellings BEYOND the retired five-substring map. These are the
    # authorised H4 activation surface: if any of them delivers after the
    # conversion and refused before, it is an activation, not an equivalence.
    "Show the illustration trend.",
    "Show the quote trend.",
    "Show the offer issued trend.",
    "Show the drawdown trend.",
    "Show the withdrawn trend.",
    "Show the cancelled trend.",
)

_VOLATILE = re.compile(r"(run[_-]?id|generated[_-]?at|timestamp|duration|"
                       r"query[_-]?id|request[_-]?id|trace[_-]?id)", re.I)


def _strip(node):
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if not _VOLATILE.search(str(k))}
    if isinstance(node, (list, tuple)):
        return [_strip(v) for v in node]
    if isinstance(node, float):
        return round(node, 2)
    return node


def capture() -> List[Dict[str, Any]]:
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)

    if not FIXTURE.is_dir():
        raise SystemExit("ASSURANCE INVALID - the five-week fixture is missing")

    from migration_phase0.assurance_semantics import load_assurance_semantics
    from mi_agent import execution_receipt as R
    from mi_agent.parsed_question import ParsedQuestion
    from mi_agent_api import chat_routing
    from question_interpretation import projection as proj

    semantics = load_assurance_semantics()
    out: List[Dict[str, Any]] = []
    for question in QUESTIONS:
        spec = ParsedQuestion.parse(question, semantics).spec
        terms = R.requested_dimension_terms(question, semantics, None)
        facets = R.detect_requested_facets(question, semantics, frame=None,
                                           requested_dimensions=terms)
        qi = proj.from_parts(question, spec=spec, facets=facets, dim_terms=terms,
                             semantics=semantics)
        # A real funded root as well as the fixture: a pipeline-family question
        # the route decides is FUNDED must still be able to build its series,
        # or the comparison degrades into "both sides crashed".
        kwargs = dict(client_id=CLIENT, run_id=None,
                      output_root=os.environ.get("MI_AGENT_ONBOARDING_OUTPUT_ROOT"),
                      pipeline_root=str(FIXTURE), portfolio_id=None, as_of=None,
                      semantics=semantics)
        try:
            envelope = chat_routing._route_evolution(
                question, spec, spec.to_dict(), interpretation=qi, **kwargs)
        except TypeError as exc:
            if "interpretation" not in str(exc):
                envelope = {"__raised__": f"TypeError: {exc}"}
            else:
                # The pre-conversion signature has no `interpretation` parameter.
                try:
                    envelope = chat_routing._route_evolution(
                        question, spec, spec.to_dict(), **kwargs)
                except Exception as inner:  # noqa: BLE001 - a crash IS the finding
                    envelope = {"__raised__": f"{type(inner).__name__}: {inner}"}
        except Exception as exc:  # noqa: BLE001 - a crash IS the finding
            envelope = {"__raised__": f"{type(exc).__name__}: {exc}"}

        rows: List[Any] = []
        if isinstance(envelope, dict):
            for artifact in envelope.get("artifacts") or []:
                rows.extend(artifact.get("rows") or [])
        insufficient = any("insufficient-data" in str(w)
                           for w in ((envelope or {}).get("warnings") or []))
        out.append({
            "question": question,
            "route": ((envelope or {}).get("metadata") or {}).get("route")
                     if isinstance(envelope, dict) else None,
            "deferred": envelope is None,
            "answer": (envelope or {}).get("answer") if isinstance(envelope, dict) else None,
            "rows": _strip(rows),
            "row_count": len(rows),
            # THE PERMANENT NON-VACUITY RULE.
            "delivered": bool(isinstance(envelope, dict)
                              and envelope.get("ok", True) and rows
                              and not insufficient),
            "grain": ((envelope or {}).get("metadata") or {}).get("seriesGrain")
                     if isinstance(envelope, dict) else None,
        })
    return out


#: Routes that answer a question ABOUT A STAGE.
_STAGE_ROUTES = ("evolution_funnel", "evolution_pipeline_stage")


def _names_a_governed_stage(question: str) -> bool:
    from question_interpretation.lexical import pipeline_stage_request
    stage, _axis = pipeline_stage_request(question)
    return bool(stage)


def diff(before, after) -> int:
    index = {r["question"]: r for r in before}
    same, moved, activated = 0, [], []
    for row in after:
        was = index.get(row["question"])
        if was is None:
            continue
        if json.dumps(was, sort_keys=True, default=str) == \
                json.dumps(row, sort_keys=True, default=str):
            same += 1
            continue
        # AUTHORISED H4 — LEGACY WRONG-DELIVERY CORRECTION.
        #
        # NOT "refused before, delivered after", and not equivalence either.
        # Measured, the refused/delivered test would have called every one of
        # these a regression:
        # the retired five-substring map did not refuse a stage spelling it
        # failed to recognise, it fell through and answered the WHOLE FUNDED
        # BOOK. "Show the illustration trend" returned £1.96bn of funded
        # balance — a silent substitution of a different question's answer.
        #
        # So an activation is: the question names a governed stage, and the
        # conversion moved it onto a stage route. What happens there — a real
        # series, or an honest "no weekly Withdrawn extracts" — is the governed
        # outcome either way, and both are better than the wrong number.
        if (_names_a_governed_stage(row["question"])
                and row["route"] in _STAGE_ROUTES
                and was["route"] not in _STAGE_ROUTES):
            activated.append((was, row))
        else:
            moved.append((was, row))

    print("=" * 92)
    print("C6 PIPELINE-FAMILY EQUIVALENCE — fixture-proven, "
          "production-data-unexercised")
    print("=" * 92)
    print(f"questions            : {len(after)}")
    print(f"delivered before/after: {sum(1 for r in before if r['delivered'])}"
          f" / {sum(1 for r in after if r['delivered'])}")
    print(f"IDENTICAL            : {same}")
    print(f"AUTHORISED H4 - LEGACY WRONG-DELIVERY CORRECTION: {len(activated)}")
    print(f"UNEXPLAINED movements: {len(moved)}")
    for was, now in activated:
        print(f"\n   WRONG-DELIVERY CORRECTED  {was['question']}")
        print(f"      before: {was['route']} delivered={was['delivered']} "
              f"rows={was['row_count']}  {str(was['answer'])[:66]}")
        print(f"      after : {now['route']} delivered={now['delivered']} "
              f"rows={now['row_count']}  {str(now['answer'])[:66]}")
    for was, now in moved:
        print(f"\n   MOVED  {was['question']}")
        print(f"      before: route={was['route']} delivered={was['delivered']} "
              f"rows={was['row_count']} answer={str(was['answer'])[:80]}")
        print(f"      after : route={now['route']} delivered={now['delivered']} "
              f"rows={now['row_count']} answer={str(now['answer'])[:80]}")
    print("\n" + "=" * 92)
    print("VERDICT: " + ("NO UNEXPLAINED MOVEMENT" if not moved
                         else "UNEXPLAINED MOVEMENT PRESENT"))
    if activated:
        print("         (equivalence is NOT claimed for the corrections above)")
    print("=" * 92)
    return 0 if not moved else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"), type=Path)
    args = parser.parse_args(argv)
    if args.diff:
        return diff(json.loads(args.diff[0].read_text()),
                    json.loads(args.diff[1].read_text()))
    rows = capture()
    delivered = sum(1 for r in rows if r["delivered"])
    print(f"captured {len(rows)} pipeline-family questions; delivered {delivered}")
    for r in rows:
        print(f"   {'DELIV' if r['delivered'] else '  -  '} {str(r['route']):<26}"
              f" rows={r['row_count']:<4} {r['question'][:44]}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2, default=str))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
