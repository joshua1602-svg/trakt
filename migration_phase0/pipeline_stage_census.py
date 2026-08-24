#!/usr/bin/env python3
"""migration_phase0/pipeline_stage_census.py — is PIPELINE_STAGE representable?

READ-ONLY. The last unclosed C6 prerequisite is that the interpretation contract
cannot structurally express Pipeline Stage semantics, so `_route_evolution`
rereads the raw question to choose between ordinary evolution, stage evolution
and the per-stage funnel series.

This measures, for the 882-question corpus plus targeted stage probes, what the
CONTRACT can say about a stage today, beside what the SHIPPED HANDLER decides
from the same question. Run before and after any contract change: the two
columns are the owner-agreement proof, and the corpus half is the blast radius.

    python -m migration_phase0.pipeline_stage_census [--json out.json] [--probes-only]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")

#: Targeted probes. Every canonical stage is named at least once, in Pipeline
#: wording and in wording that does NOT contain the word "pipeline", plus the
#: generic uses of "stage" that must NOT acquire Pipeline semantics.
PROBES: Tuple[Tuple[str, str, Optional[str], str], ...] = (
    # id, question, expected canonical stage (None = stage dimension, no specific stage), note
    ("S-KFI-1", "Show the KFI trend by week.", "KFI", "funnel, explicit stage"),
    ("S-KFI-2", "How many KFIs have we issued?", "KFI", "stage word, no 'pipeline'"),
    ("S-APP-1", "How have application-stage cases changed?", "APPLICATION", "stage evolution"),
    ("S-APP-2", "Show the application trend over time.", "APPLICATION", "stage word, no 'pipeline'"),
    ("S-OFF-1", "How have offer-stage cases changed?", "OFFER", "the brief's worked example"),
    ("S-OFF-2", "Show the offer trend over time.", "OFFER", "stage word, no 'pipeline'"),
    ("S-COM-1", "Show the completion trend by week.", "COMPLETED", "completion -> COMPLETED"),
    ("S-COM-2", "How many cases completed this month?", "COMPLETED", "past-tense spelling"),
    ("S-WDR-1", "Show withdrawn cases over time.", "WITHDRAWN", "THE STAGE THE ROUTE CANNOT NAME"),
    ("S-WDR-2", "How many pipeline cases were withdrawn?", "WITHDRAWN", "explicit pipeline + withdrawn"),
    ("S-WDR-3", "Show declined cases by week.", "WITHDRAWN", "governed alias of WITHDRAWN"),
    # Stage as a BREAKDOWN, not a specific stage.
    ("D-BY-1", "Show pipeline amount by stage over time.", None, "by-stage breakdown"),
    ("D-BY-2", "How has the pipeline changed by stage?", None, "the brief's worked example"),
    ("D-BY-3", "Show pipeline stage balances over time.", None, "the brief's worked example"),
    ("D-BY-4", "Show pipeline stage migration.", None, "stage migration phrase"),
    ("D-BY-5", "What is the pipeline stage distribution?", None, "stage distribution"),
    # Pipeline, but NOT a stage question — must not acquire a stage claim.
    ("N-PIPE-1", "How has the pipeline changed over time?", None, "pipeline, no stage"),
    ("N-PIPE-2", "Show pipeline amount evolution by week.", None, "pipeline, no stage"),
    # NOT pipeline at all — must not acquire pipeline semantics.
    ("N-FUND-1", "Show funded balance evolution by month.", None, "funded control"),
    ("N-FUND-2", "Which region gained the most cases since last month?", None, "the golden-bank regression"),
    ("N-GEN-1", "What stage is the securitisation at?", None, "generic 'stage', not pipeline"),
    ("N-GEN-2", "Show the offer price distribution.", None, "'offer' as a non-stage word"),
)


def _corpus() -> List[str]:
    out: List[str] = []
    seen = set()
    for f in CORPORA:
        p = _REPO / f
        if not p.exists():
            continue
        for row in json.loads(p.read_text(encoding="utf-8"))["rows"]:
            q = row.get("question") or ""
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def _env() -> Tuple[str, Dict[str, Any]]:
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    from mi_agent_api import mi_service
    sem: Dict[str, Any] = {}
    for name in ("load_semantics", "_load_semantics", "semantics_for"):
        fn = getattr(mi_service, name, None)
        if callable(fn):
            try:
                sem = fn(cfg.CLIENT_ID) or {}
                break
            except TypeError:
                try:
                    sem = fn() or {}
                    break
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
    return cfg.CLIENT_ID, sem


def _shipped_decision(question: str) -> Dict[str, Any]:
    """What the SHIPPED handler decides from the raw question, reproduced from
    its own conditions rather than from a copy of them."""
    from mi_agent_api.chat_routing import _FUNNEL_KEYWORDS
    from mi_agent_api.workspace import resolve_dataset
    low = question.lower()
    funnel = next((st for kw, st in _FUNNEL_KEYWORDS.items() if kw in low), None)
    by_stage = any(w in low for w in ("by stage", "stage over time", "stage migration"))
    dataset = resolve_dataset(question)
    if funnel:
        sub = "funnel"
    elif dataset == "pipeline" and by_stage:
        sub = "by_stage"
    else:
        sub = "ordinary"
    return {"sub_route": sub, "shipped_stage": funnel, "dataset": dataset}


def _contract_reading(question: str, semantics: Dict[str, Any]) -> Dict[str, Any]:
    """What the CONTRACT can say about a stage, with no second owner consulted."""
    from question_interpretation import projection as proj

    # `project` owns its own parse and facet detection; handing it a spec was
    # the bug that made the first run of this census vacuous — the TypeError was
    # caught into the row and every contract column read "-" for the wrong
    # reason. Errors now propagate.
    qi = proj.project(question, semantics=semantics, frame=None)

    dims = [(getattr(d, "candidate_concept", None), getattr(d, "role", None))
            for d in (getattr(qi, "dimensions", None) or [])]
    filters = [(getattr(f, "categorical_value", None), getattr(f, "source", None))
               for f in (getattr(qi, "filters", None) or [])]
    # Anything the contract already carries that could mean a pipeline stage.
    stage_dim = any(str(k or "").lower() in ("pipeline_stage", "stage") for k, _ in dims)
    stage_filter = [v for v, _ in filters
                    if str(v or "").upper() in ("KFI", "APPLICATION", "OFFER",
                                                "COMPLETED", "WITHDRAWN")]
    claim = getattr(qi, "pipeline_stage", None)  # exists only after the change
    return {
        "dims": dims,
        "stage_dim": stage_dim,
        "stage_filter": stage_filter or None,
        "contract_stage": (getattr(claim, "stage", None) if claim is not None else None),
        "contract_state": (getattr(claim, "state", None) if claim is not None else None),
        "contract_is_breakdown": (getattr(claim, "breakdown", None)
                                  if claim is not None else None),
        "claim_exists": claim is not None,
    }


def run(probes_only: bool = False) -> List[Dict[str, Any]]:
    client_id, semantics = _env()
    questions: List[Tuple[str, str, Optional[str], str]] = list(PROBES)
    if not probes_only:
        questions += [(f"C{i:03d}", q, None, "corpus")
                      for i, q in enumerate(_corpus())]
    rows: List[Dict[str, Any]] = []
    for cid, q, expect, note in questions:
        row: Dict[str, Any] = {"id": cid, "question": q, "expect": expect, "note": note}
        row.update(_shipped_decision(q))
        # Deliberately unguarded: a census that swallows its own failure reports
        # "the contract says nothing" when what happened is that it never asked.
        row.update(_contract_reading(q, semantics))
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--probes-only", action="store_true")
    args = ap.parse_args(argv)

    rows = run(args.probes_only)
    probes = [r for r in rows if r["note"] != "corpus"]
    corpus = [r for r in rows if r["note"] == "corpus"]

    print("=" * 96)
    print(f"PIPELINE STAGE CENSUS — {len(probes)} probes, {len(corpus)} corpus questions")
    print("=" * 96)
    print(f"\n{'id':<10}{'shipped':<11}{'shipStage':<13}{'dataset':<10}"
          f"{'contract':<12}{'expect':<13}question")
    print("-" * 96)
    for r in probes:
        contract = (r.get("contract_stage") or
                    ("BREAKDOWN" if r.get("contract_is_breakdown") else None) or
                    (r.get("stage_filter") and r["stage_filter"][0]) or
                    ("dim" if r.get("stage_dim") else "-"))
        print(f"{r['id']:<10}{str(r.get('sub_route')):<11}"
              f"{str(r.get('shipped_stage') or '-'):<13}{str(r.get('dataset')):<10}"
              f"{str(contract):<12}{str(r.get('expect') or '-'):<13}{r['question'][:34]}")

    print(f"\nCONTRACT CAN NAME A STAGE AT ALL: "
          f"{'YES' if any(r.get('claim_exists') for r in rows) else 'NO'}")
    print(f"contract carries a stage DIMENSION on any probe: "
          f"{sum(1 for r in probes if r.get('stage_dim'))}")
    print(f"contract carries a stage FILTER VALUE on any probe: "
          f"{sum(1 for r in probes if r.get('stage_filter'))}")

    if corpus:
        sub = {}
        for r in corpus:
            sub[r.get("sub_route")] = sub.get(r.get("sub_route"), 0) + 1
        print(f"\nCORPUS shipped sub-route distribution: {sub}")
        print(f"corpus questions the shipped handler sends to funnel  : "
              f"{sum(1 for r in corpus if r.get('sub_route') == 'funnel')}")
        print(f"corpus questions the shipped handler sends to by-stage: "
              f"{sum(1 for r in corpus if r.get('sub_route') == 'by_stage')}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
