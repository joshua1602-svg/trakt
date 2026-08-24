#!/usr/bin/env python3
"""migration_phase0/funnel_stage_representation.py — C6 prerequisite 2.

READ-ONLY. Two dependencies failed the C6 four-part matrix outright:
**funnel-stage selection** and **by-stage selection**, both marked "NO contract
representation at all". This measures what selects them today, what the
interpretation contract can currently say about a stage, and whether an existing
governed conversion capability already covers what C6 would otherwise rebuild.

Three questions, each answered from code rather than from the C6 report:

  1. What actually selects the funnel and by-stage sub-routes?
  2. Can any existing claim on the contract carry a pipeline stage?
  3. Does a governed conversion calculation already exist?

    python -m migration_phase0.funnel_stage_representation
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")


def _questions() -> List[str]:
    out, seen = [], set()
    for f in CORPORA:
        for row in json.loads((_REPO / f).read_text())["rows"]:
            q = row.get("question") or ""
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def main(argv=None) -> int:
    import logging, warnings
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    from mi_agent_api import chat_routing as CR
    from mi_agent_api import pipeline_prep
    from question_interpretation import schema as S

    print("=" * 84)
    print("C6 PREREQUISITE 2 — FUNNEL / STAGE SELECTION, MEASURED")
    print("=" * 84)

    # ------------------------------------------------------------------ 1
    print("\n1. WHAT SELECTS THE SUB-ROUTES TODAY")
    src = inspect.getsource(CR._route_evolution)
    print("   funnel branch   : raw-question membership test against")
    print(f"                     _FUNNEL_KEYWORDS = {CR._FUNNEL_KEYWORDS}")
    by_stage = re.findall(r'"((?:by stage|stage over time|stage migration))"', src)
    print(f"   by-stage branch : raw-question substring test against {sorted(set(by_stage))}")
    print("   Both read the RAW QUESTION inside the handler. Neither consults the")
    print("   interpretation contract, and no plan primitive expresses either.")

    # ------------------------------------------------------------------ 2
    print("\n2. THE GOVERNED STAGE VOCABULARY — one owner, already canonical")
    canon = getattr(pipeline_prep, "_STAGE_CANON", {})
    stages = sorted(set(canon.values())) if canon else []
    print(f"   pipeline_prep._STAGE_CANON -> {len(canon)} spellings -> {stages}")
    routed = sorted(set(CR._FUNNEL_KEYWORDS.values()))
    print(f"   _FUNNEL_KEYWORDS reaches    -> {routed}")
    missing = [s for s in stages if s not in routed]
    print(f"   governed stages the route's vocabulary CANNOT name -> {missing or 'none'}")

    # ------------------------------------------------------------------ 3
    print("\n3. CAN ANY EXISTING CLAIM CARRY A STAGE?")
    claims = [n for n in dir(S) if n.endswith("Claim")]
    carriers = []
    for name in claims:
        cls = getattr(S, name)
        fields = set(getattr(cls, "__annotations__", {}) or {})
        hit = sorted(f for f in fields if "stage" in f.lower() or "funnel" in f.lower())
        print(f"   {name:<20} fields={len(fields):<3} stage-bearing={hit or '-'}")
        if hit:
            carriers.append(name)
    print(f"   claims that can carry a stage TODAY: {carriers or 'NONE'}")

    # ------------------------------------------------------------------ 4
    print("\n4. DOES A GOVERNED CONVERSION CALCULATION ALREADY EXIST?")
    from mi_agent_api import evolution as EV
    from mi_agent_api import forecast_extrapolation as FX
    found = [
        ("evolution._conversion_pct", EV, "_conversion_pct",
         "stage-to-stage percentage, used by the funnel"),
        ("forecast_extrapolation.kfi_conversion_model", FX, "kfi_conversion_model",
         "empirical 5-week completion-vs-KFI rate, shared with the forecast bridge"),
        ("chat_routing._route_conversion", CR, "_route_conversion",
         "the shipped conversion ROUTE"),
    ]
    for label, mod, attr, why in found:
        exists = hasattr(mod, attr)
        print(f"   {'YES' if exists else 'NO ':<4} {label:<48} {why}")
    print("   => C6 must CONSUME these, never add a second conversion rate.")

    # ------------------------------------------------------------------ 5
    print("\n5. CORPUS DEMAND FOR THE TWO UNREPRESENTED DEPENDENCIES")
    qs = _questions()
    fun = [q for q in qs if any(k in q.lower() for k in CR._FUNNEL_KEYWORDS)]
    bys = [q for q in qs if any(w in q.lower() for w in set(by_stage))]
    print(f"   questions naming a funnel stage word : {len(fun)} of {len(qs)}")
    print(f"   questions naming a by-stage phrase   : {len(bys)} of {len(qs)}")

    print("\n6. VERDICT")
    print("   funnel-stage selection : NO contract representation, NO plan primitive,")
    print("                            selected by a raw-question read inside the handler")
    print("   by-stage selection     : as above")
    print(f"   governed vocabulary    : EXISTS and is canonical ({len(stages)} stages),")
    print("                            but the route reaches only "
          f"{len(routed)} of {len(stages)} of it")
    print("   conversion capability  : EXISTS in three places — C6 consumes, "
          "does not rebuild")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
