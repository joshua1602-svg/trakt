#!/usr/bin/env python3
"""migration_phase0/dataset_rule_census.py

READ-ONLY. Censuses CANDIDATE authoritative dataset rules against today's
production readings, over the 882 distinct Stage 1 + Stage 2 corpus questions,
BEFORE one is chosen.

The point is to choose the rule on measured movement rather than on which one
reads best in a docstring. Breadth is not a virtue: every extra term is an
unauthorised movement waiting to happen, so the winner is the narrowest rule
that satisfies the target state's worked examples.

    python -m migration_phase0.dataset_rule_census
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")

#: The target state's worked examples, from the remediation brief. A candidate
#: rule that fails any of these is disqualified whatever its movement count.
WORKED: Tuple[Tuple[str, str], ...] = (
    ("What is the funded balance?", "funded"),
    ("What is the acquired funded balance?", "funded"),
    ("What is the direct funded balance?", "funded"),
    ("What is the funded loan count?", "funded"),
    ("How many applications are there?", "pipeline"),
    ("How many KFIs are there?", "pipeline"),
    ("How many cases are there?", "pipeline"),
    ("How many offers are there?", "pipeline"),
    ("What is the pipeline amount?", "pipeline"),
    ("Forecast application volumes next quarter", "forecast"),
    ("Forecast completions over the next 3 months", "forecast"),
    ("Forecast funded volumes for the next quarter", "forecast"),
    ("How much of the forecast comes from pipeline?", "forecast"),
    ("What is the balance by seasoning segment excluding pipeline cases?",
     "funded"),
    ("What is the total balance?", "funded"),
)


def _questions() -> List[str]:
    seen: List[str] = []
    known = set()
    for f in CORPORA:
        for row in json.loads((_REPO / f).read_text())["rows"]:
            q = row.get("question") or ""
            if q and q not in known:
                known.add(q)
                seen.append(q)
    return seen


def _rules() -> Dict[str, Callable[[str], str]]:
    from mi_agent.portfolio_lens import undisclaimed_mention as um
    from mi_agent_api.workspace import (DEFAULT_VIEW, resolve_active_view,
                                        view_named_by_question)
    from mi_agent_api import chat_routing as cr

    #: The tape artefacts `_dataset_for` reads and `view_named_by_question`
    #: does not. NOT a new vocabulary: this is `chat_routing._PIPELINE_WORDS`
    #: minus `pipeline`, which the view names already cover.
    ARTEFACTS = ("case", "kfi", "application", "offer")

    def today_pointintime(q: str) -> str:
        """What an ordinary (unrouted) question resolves to at NO tab."""
        return resolve_active_view(q, None)

    def today_routed(q: str) -> str:
        """What `_route_compare` / `_route_evolution` resolve at NO tab."""
        return cr._dataset_for(q, resolve_active_view(q, None))

    def narrow_union(q: str) -> str:
        """R1 — the view names, then the tape artefacts, then the default.

        Deliberately the NARROWEST union of the two existing owners: steps 1-3
        are `view_named_by_question` unchanged, so nothing it already decides
        can move. Step 4 fires only where it returned None, which is exactly
        the gap `_dataset_for` was covering alone.
        """
        named = view_named_by_question(q)
        if named is not None:
            return named
        low = (q or "").lower()
        if any(um(low, w) for w in ARTEFACTS):
            return "pipeline"
        return DEFAULT_VIEW

    def intent_requirements(q: str) -> str:
        """R2 — the governed analytical intent layer's structural requirements."""
        from mi_workflows.analytical import intent as it
        reading = it.classify(q, spec=None)
        reqs = set(getattr(reading, "requirements", ()) or ())
        if it.REQ_FORECAST in reqs:
            return "forecast"
        if it.REQ_PIPELINE_DATASET in reqs:
            return "pipeline"
        return DEFAULT_VIEW

    return {"today_pointintime": today_pointintime,
            "today_routed": today_routed,
            "R1_narrow_union": narrow_union,
            "R2_intent_requirements": intent_requirements}


def main() -> int:
    import logging
    import warnings
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    logging.disable(logging.WARNING)

    qs = _questions()
    rules = _rules()
    print("=" * 92)
    print(f"DATASET RULE CENSUS — {len(qs)} distinct corpus questions")
    print("=" * 92)

    # --- the worked examples gate ----------------------------------------- #
    print("\nTarget-state worked examples (a rule failing any of these is out):")
    gate: Dict[str, int] = {}
    for name, fn in rules.items():
        bad = [(q, want, fn(q)) for q, want in WORKED if fn(q) != want]
        gate[name] = len(bad)
        print(f"  {name:<24} failures {len(bad):>2} of {len(WORKED)}")
        for q, want, got in bad:
            print(f"        want {want:<9} got {got:<9} :: {q[:66]}")

    # --- corpus movement vs today ----------------------------------------- #
    base_pit = [rules["today_pointintime"](q) for q in qs]
    base_rtd = [rules["today_routed"](q) for q in qs]
    print(f"\nCorpus movement, against today's point-in-time reading "
          f"(the one that loads the frame):")
    moves: Dict[str, List[Tuple[str, str, str]]] = {}
    for name in ("R1_narrow_union", "R2_intent_requirements"):
        fn = rules[name]
        m = [(q, b, fn(q)) for q, b in zip(qs, base_pit) if fn(q) != b]
        moves[name] = m
        pct = 100.0 * len(m) / len(qs)
        print(f"  {name:<24} moves {len(m):>4} of {len(qs)}  ({pct:.1f}%)")

    for name, m in moves.items():
        print(f"\n--- {name}: first 20 movements ---")
        for q, before, after in m[:20]:
            print(f"  {before:<9} -> {after:<9} :: {q[:74]}")

    out = _REPO / "migration_phase0" / "DATASET_RULE_CENSUS.json"
    out.write_text(json.dumps({
        "questions": len(qs),
        "workedExampleFailures": gate,
        "movementsAgainstPointInTime": {
            k: [{"question": q, "before": b, "after": a} for q, b, a in v]
            for k, v in moves.items()},
        "routedDisagreesWithPointInTime": sum(
            1 for a, b in zip(base_pit, base_rtd) if a != b),
    }, indent=2, default=str))
    print(f"\nrouted rule disagrees with point-in-time rule TODAY, on "
          f"{sum(1 for a, b in zip(base_pit, base_rtd) if a != b)} of {len(qs)}")
    print(f"written : {out.relative_to(_REPO)}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
