#!/usr/bin/env python3
"""migration_phase0/plan_equivalence_temporal_compare.py — economics, C5.

READ-ONLY. Runs the SHIPPED calculation path (as `_route_compare` called it
before Conversion 5: `spec.compare_periods[0:2]`, `workspace.resolve_dataset`,
`spec.metric`, `spec.aggregation` -> `temporal_compare.run_temporal_compare`)
against the COMPOSITIONAL one (contract -> accessors -> plan -> the same
engine), and compares every economic field.

The comparison is per owned case AND per workspace tab, because the tab is the
axis this route's dataset decision used to be sensitive to. It is not any more,
and running both is how that stays true through the conversion.

    python -m migration_phase0.plan_equivalence_temporal_compare
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

#: A2. The programme's standing economic tolerance.
_TOL = 0.005

#: Every economic field the comparison result carries.
_FIELDS = ("available", "metric", "metricLabel", "format", "periodA", "periodB",
           "valueA", "valueB", "absoluteDelta", "percentageDelta", "direction",
           "reason", "status", "availablePeriods", "sourcePeriods", "dataset",
           "portfolioId", "toRunId", "reconciliation", "lineage")


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


def _close(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < _TOL
    return a == b


def main() -> int:
    client_id = _env()
    from migration_phase0.route_ownership_temporal_compare import CASES, DATASETS
    from mi_agent.llm_query_parser import parse_with_repair
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api import analytical_plan as plan_mod
    from mi_agent_api import temporal_compare as compare_mod
    from mi_agent_api import workspace as ws
    from mi_agent_api.data_source import semantics_path
    from mi_agent import execution_receipt as receipt_mod
    from question_interpretation import projection as proj

    sem = load_mi_semantics(semantics_path())
    root = os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"]

    owned = [(c, q) for c, q, other in CASES if other is None]
    expected = len(owned) * len(DATASETS)
    print("=" * 92)
    print("temporal_compare — C5 ECONOMIC EQUIVALENCE, shipped vs compositional")
    print(f"{len(owned)} owned cases x {len(DATASETS)} tabs = {expected} comparisons")
    print("=" * 92)

    rows: List[Dict[str, Any]] = []
    differing = 0
    for case, question in owned:
        spec, _meta = parse_with_repair(question, sem, llm_enabled=False)
        dim_terms = receipt_mod.requested_dimension_terms(question, sem, None)
        facets = list(receipt_mod.detect_requested_facets(
            question, sem, frame=None, requested_dimensions=dim_terms))
        for tab in DATASETS:
            # ---- SHIPPED ------------------------------------------------- #
            periods = list(getattr(spec, "compare_periods", None) or [])
            shipped = compare_mod.run_temporal_compare(
                root, root, client_id, None,
                dataset=ws.resolve_dataset(question),
                metric=getattr(spec, "metric", None),
                aggregation=getattr(spec, "aggregation", ""),
                period_a=periods[0], period_b=periods[1])

            # ---- COMPOSITIONAL ------------------------------------------- #
            qi = proj.from_parts(question, spec=spec, facets=facets,
                                 dim_terms=dim_terms, semantics=sem,
                                 registry=None, caller_scope=None,
                                 caller_dataset=tab)
            composed = plan_mod.temporal_compare(root, root, client_id, None,
                                                 interpretation=qi)

            bad = [f for f in _FIELDS
                   if not _close(shipped.get(f), composed.get(f))]
            if bad:
                differing += 1
            flag = "   " if not bad else "!! "
            print(f"{flag}{case:<4} tab={tab:<8} {shipped.get('metric')!s:<22} "
                  f"A={shipped.get('valueA')!s:<14} B={shipped.get('valueB')!s:<14} "
                  f"avail={shipped.get('available')}")
            for f in bad:
                print(f"      {f}: shipped={shipped.get(f)!r} "
                      f"composed={composed.get(f)!r}")
            rows.append({"case": case, "question": question, "tab": tab,
                         "differingFields": bad,
                         "shipped": {f: shipped.get(f) for f in _FIELDS},
                         "composed": {f: composed.get(f) for f in _FIELDS}})

    out = _REPO / "migration_phase0" / "PLAN_EQUIVALENCE_TEMPORAL_COMPARE.json"
    out.write_text(json.dumps({"rows": rows}, indent=2, default=str))

    print("\n" + "=" * 92)
    print(f"comparisons made          : {len(rows)}  (expected {expected})")
    print(f"fields compared per pair  : {len(_FIELDS)}")
    print(f"DIFFERING COMPARISONS     : {differing}")
    print(f"written                   : {out.relative_to(_REPO)}")
    print("=" * 92)
    if len(rows) != expected:
        print("!! the surface moved; this comparison proves nothing")
        return 1
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
