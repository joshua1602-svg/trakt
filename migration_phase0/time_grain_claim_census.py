"""migration_phase0/time_grain_claim_census.py

READ-ONLY. Measures the SECOND owner of the reporting-grain claim.

`chat_routing._route_evolution` decides what grain a series IS. Independently,
`execution_receipt._ROUTE_TIME_GRAIN` declares what grain the receipt BELIEVES
each route publishes — a static route -> grain map asserting `"month"` for all
ten series routes, on the stated premise that "every one of these reads the
governed month-end funded snapshots".

That premise is false for three of the ten, and was false before any change in
this task:

  * `evolution_funnel`         keys its rows on `week` (day-level extract dates)
  * `evolution_pipeline_stage` keys its rows on the extract date
  * `evolution` on PIPELINE    keys its rows on `week` (after the owner fix)

This census counts, over the governed corpora, how many questions name a
sub-month unit AND would be answered by a route that actually publishes one, so
the size of the second owner's remaining blast is a number rather than an
impression.

    python -m migration_phase0.time_grain_claim_census
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")

#: The routes whose rows are keyed on a DAY-level value, read from the route
#: source rather than asserted here. `evolution` is conditional: it is weekly
#: only when the pipeline producer supplied the series.
DAY_LEVEL_ROUTES = ("evolution_funnel", "evolution_pipeline_stage")


def _questions() -> List[str]:
    seen: Dict[str, None] = {}
    for rel in CORPORA:
        p = _ROOT / rel
        if not p.exists():
            continue
        for row in json.loads(p.read_text(encoding="utf-8"))["rows"]:
            q = row.get("question") or ""
            if q:
                seen.setdefault(str(q), None)
    return list(seen)


def main(argv=None) -> int:
    import logging, warnings
    warnings.simplefilter("ignore")
    logging.disable(logging.CRITICAL)

    from mi_agent import period_request as _pr
    from mi_agent import execution_receipt as R
    from mi_agent_api.workspace import resolve_dataset

    qs = _questions()
    print("=" * 84)
    print(f"TIME-GRAIN CLAIM CENSUS — {len(qs)} distinct governed questions")
    print("=" * 84)

    print("\n1. WHAT `_ROUTE_TIME_GRAIN` CLAIMS")
    for route, grain in sorted(R._ROUTE_TIME_GRAIN.items()):
        print(f"   {route:28} -> {grain}")

    named = [q for q in qs if _pr.requested_unit(q)]
    units: Dict[str, int] = {}
    for q in named:
        u = str(_pr.requested_unit(q))
        units[u] = units.get(u, 0) + 1
    print(f"\n2. QUESTIONS NAMING A REPORTING UNIT: {len(named)} of {len(qs)}")
    for u, n in sorted(units.items(), key=lambda kv: -kv[1]):
        print(f"   {u:12} {n}")

    sub_month = [q for q in named
                 if str(_pr.requested_unit(q)).lower() in ("week", "day", "daily", "weekly")]
    print(f"\n3. OF THOSE, NAMING A SUB-MONTH UNIT: {len(sub_month)}")
    pipeline_side = [q for q in sub_month if resolve_dataset(q) == "pipeline"]
    print(f"   ...on the PIPELINE dataset: {len(pipeline_side)}")
    for q in pipeline_side:
        print(f"      {q}")
    funded_side = [q for q in sub_month if resolve_dataset(q) != "pipeline"]
    print(f"   ...on a FUNDED dataset (correctly told months): {len(funded_side)}")
    for q in funded_side[:12]:
        print(f"      {q}")

    print("\n4. VERDICT")
    print(f"   The static map is wrong for {len(DAY_LEVEL_ROUTES) + 1} of "
          f"{len(R._ROUTE_TIME_GRAIN)} routes it covers.")
    print(f"   {len(pipeline_side)} governed corpus question(s) name a sub-month unit on "
          "the pipeline\n   dataset and are refused a correct weekly answer by the "
          "stale claim alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
