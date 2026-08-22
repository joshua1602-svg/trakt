#!/usr/bin/env python3
"""question_interpretation/llm_arm_comparison.py

THE FIFTEEN SHIPPED SHAPES, DETERMINISTIC ARM vs LLM ARM.

Why this exists
---------------
Everything fixed in items 1-4 was measured with `MI_AGENT_LLM_PARSER=off`. Users
do not reach the system that way. If a fix only lands when the deterministic
parser owns the question, the user sees none of it.

A CORRECTION TO THE PREMISE, measured rather than assumed. The brief said the
model "proposes and the deterministic layer checks". It is the other way round:
`parse_with_repair` runs the DETERMINISTIC parser first and hands off to the
model only when `zero_cost_first`'s gate fails — that is, when the deterministic
spec does not validate, or its confidence is not "high", or the question is
"layered". The LLM is the fallback, not the proposer.

That correction does not make the concern smaller. Measured on this bank,
**14 of the 15 shapes fail the gate and reach the model**, so the deterministic
results this programme has been quoting all week are the FALLBACK path for
almost every shape it fixed.

REPEATS. The LLM arm disagrees with itself run to run, and that instability is
larger than any single change — so each question is run `--repeats` times
(default 5) and the report states, per question, how many distinct outcomes
appeared. A shape with two outcomes in five runs is reported as unstable rather
than as one result.

    python -m question_interpretation.llm_arm_comparison --repeats 5
    python -m question_interpretation.llm_arm_comparison --gate-only

`--gate-only` needs no API key and reports which shapes reach the model at all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BANK = _REPO_ROOT / "config" / "mi" / "golden_questions" / "shipped_shapes.yaml"

#: What a divergence MEANS. The brief asks for this attribution per question.
PROPOSAL_NOT_REACHED = "the model proposed something the fix does not reach"
GUARD_REFUSED_PROPOSAL = "the guard refused a proposal the fix would have handled"
SAME = "same outcome"
UNSTABLE = "the LLM arm disagreed with itself"


def cases() -> List[Tuple[str, str]]:
    doc = yaml.safe_load(BANK.read_text(encoding="utf-8"))
    return [(c["id"], c["question"])
            for shape in doc["shapes"] for c in shape["questions"]]


def _env(llm: bool) -> None:
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "on" if llm else "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "1" if llm else "0"
    import logging
    logging.disable(logging.WARNING)


def llm_available() -> Tuple[bool, str]:
    """Whether the LLM arm can actually run. Reported, never assumed."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, ("ANTHROPIC_API_KEY is not set; the LLM arm cannot run "
                       "and no result is reported for it")
    try:
        import anthropic  # noqa: F401
    except Exception:  # noqa: BLE001
        return False, "the anthropic package is not installed"
    return True, "available"


# --------------------------------------------------------------------------- #
# The gate — measurable with no key at all
# --------------------------------------------------------------------------- #
def gate_report() -> List[Dict[str, Any]]:
    """Per question: does the DETERMINISTIC parser own it outright?

    Reproduces `parse_with_repair`'s `zero_cost_first` gate. A question that
    passes it never reaches the model, so a fix to the deterministic path is all
    the user ever sees. A question that fails it is handed over, and the fix's
    visibility depends on what comes back.
    """
    _env(llm=False)
    from mi_agent import llm_query_parser as P
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api import data_source

    semantics = load_mi_semantics(
        str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
    columns = set(data_source.get_dataframe().columns)

    rows = []
    for cid, question in cases():
        _spec, meta = P.parse_with_repair(question, semantics,
                                          available_columns=columns,
                                          llm_enabled=False)
        layered = P._is_layered_question(question)
        confidence = meta.get("parser_confidence")
        note = meta.get("note")
        owned = (meta.get("parser_mode") == "deterministic" and not layered
                 and confidence == "high" and note != "unmapped")
        rows.append({"id": cid, "question": question, "owned": owned,
                     "layered": layered, "confidence": confidence, "note": note,
                     "why": ("owned" if owned else
                             "layered" if layered else
                             "unmapped" if note == "unmapped" else
                             "confidence=%s" % confidence)})
    return rows


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #
def _observe(question: str) -> Dict[str, Any]:
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext
    from demo_platform import config as cfg
    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    res = execute_governed_mi_query(MiQueryRequest(question=question), ctx).result or {}
    meta = res.get("metadata") or {}
    guard = meta.get("semantic_guard") or {}
    summary = res.get("executionSummary") or {}
    return {"ok": bool(res.get("ok")), "route": meta.get("route"),
            "verdict": guard.get("verdict"), "measure": summary.get("measure"),
            "population": summary.get("population"),
            "filters": sorted(str(f) for f in (summary.get("filtersApplied") or ())),
            "dimensions": sorted(str(d) for d in (summary.get("dimensionsApplied") or ())),
            "parser_mode": meta.get("parser_mode")}


def _key(seen: Dict[str, Any]) -> str:
    return json.dumps({k: seen[k] for k in
                       ("ok", "measure", "population", "filters", "dimensions")},
                      sort_keys=True, default=str)


def attribute(det: Dict[str, Any], llm: Dict[str, Any]) -> str:
    """Which KIND of divergence this is — the brief's question.

    * the model proposed something the fix does not reach — it answered, but
      over a different measure, population or grouping than the deterministic
      arm now produces;
    * the guard refused a proposal the fix would have handled — the
      deterministic arm answers and the LLM arm is blocked.
    """
    if _key(det) == _key(llm):
        return SAME
    if det["ok"] and not llm["ok"]:
        return GUARD_REFUSED_PROPOSAL
    return PROPOSAL_NOT_REACHED


def run(repeats: int) -> Dict[str, Any]:
    available, why = llm_available()
    gate = gate_report()
    if not available:
        return {"llm_available": False, "reason": why, "gate": gate,
                "repeats": repeats, "rows": []}

    _env(llm=False)
    det = {cid: _observe(q) for cid, q in cases()}
    _env(llm=True)
    rows = []
    for cid, question in cases():
        runs = [_observe(question) for _ in range(repeats)]
        counts = Counter(_key(r) for r in runs)
        stable = len(counts) == 1
        modal = counts.most_common(1)[0]
        modal_run = next(r for r in runs if _key(r) == modal[0])
        rows.append({"id": cid, "question": question,
                     "deterministic": det[cid], "llm": modal_run,
                     "distinct_outcomes": len(counts),
                     "modal_share": "%d/%d" % (modal[1], repeats),
                     "stable": stable,
                     "attribution": (UNSTABLE if not stable
                                     else attribute(det[cid], modal_run))})
    return {"llm_available": True, "gate": gate, "repeats": repeats, "rows": rows}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(result: Dict[str, Any]) -> int:
    gate = result["gate"]
    owned = [g for g in gate if g["owned"]]
    print("=" * 78)
    print("THE FIFTEEN SHIPPED SHAPES — deterministic arm vs LLM arm")
    print("=" * 78)
    print()
    print("WHICH ARM ACTUALLY ANSWERS  (the `zero_cost_first` gate)")
    print("-" * 78)
    print("%-5s %-50s %-7s %s" % ("id", "question", "owned", "why not"))
    for g in gate:
        print("%-5s %-50s %-7s %s"
              % (g["id"], g["question"][:50], g["owned"],
                 "" if g["owned"] else g["why"]))
    print("-" * 78)
    print("deterministically OWNED — the model is never called : %d of %d"
          % (len(owned), len(gate)))
    print("HANDED to the model when a key is present           : %d of %d"
          % (len(gate) - len(owned), len(gate)))
    print()

    if not result["llm_available"]:
        print("=" * 78)
        print("LLM ARM: NOT RUN")
        print("=" * 78)
        print("  %s." % result["reason"])
        print()
        print("  No LLM outcome is reported for any question, and none is")
        print("  estimated. The gate table above is what this environment can")
        print("  establish, and it is the part that answers whether a")
        print("  deterministic fix reaches a user at all.")
        return 2

    print("PER QUESTION  (repeats: %d)" % result["repeats"])
    print("-" * 78)
    for r in result["rows"]:
        print("%-5s %-46s det=%-6s llm=%-6s %s"
              % (r["id"], r["question"][:46], r["deterministic"]["ok"],
                 r["llm"]["ok"], r["modal_share"]))
        if r["attribution"] != SAME:
            print("      %s" % r["attribution"])
            print("      det: measure=%s pop=%s filters=%s dims=%s"
                  % (r["deterministic"]["measure"], r["deterministic"]["population"],
                     r["deterministic"]["filters"], r["deterministic"]["dimensions"]))
            print("      llm: measure=%s pop=%s filters=%s dims=%s"
                  % (r["llm"]["measure"], r["llm"]["population"],
                     r["llm"]["filters"], r["llm"]["dimensions"]))
    print("-" * 78)
    tally = Counter(r["attribution"] for r in result["rows"])
    for k in (SAME, PROPOSAL_NOT_REACHED, GUARD_REFUSED_PROPOSAL, UNSTABLE):
        print("   %-52s %d" % (k, tally.get(k, 0)))
    unstable = [r["id"] for r in result["rows"] if not r["stable"]]
    if unstable:
        print()
        print("   self-disagreement across %d repeats: %s"
              % (result["repeats"], ", ".join(unstable)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5,
                    help="LLM runs per question; the arm disagrees with itself")
    ap.add_argument("--gate-only", action="store_true",
                    help="report which shapes reach the model; needs no API key")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args(argv)

    result = ({"llm_available": False, "reason": "--gate-only requested",
               "gate": gate_report(), "repeats": 0, "rows": []}
              if args.gate_only else run(args.repeats))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str),
                                   encoding="utf-8")
    return report(result)


if __name__ == "__main__":
    raise SystemExit(main())
