#!/usr/bin/env python3
"""migration_phase0/plan_equivalence_funded_bridge.py — economics, C4.

READ-ONLY. Runs the SHIPPED calculation path (as `_route_bridge` called it
before Conversion 4: `spec.bridge_dimension` -> registry -> `evolution.
funded_bridge`, population from `resolve_lens_with_default`, start period from
`spec.compare_periods[0]`) against the COMPOSITIONAL one (contract -> grouping
concept -> plan -> the same engine), and compares every economic field.

Populations are compared on ROWS, never on the lens name: Phase 1G resolves
categories through the governed registry, so the two sides legitimately carry
different filter SHAPES for the same population.

    python -m migration_phase0.plan_equivalence_funded_bridge
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

_TOL = 0.005


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
    from migration_phase0.route_ownership_funded_bridge import CASES, SCOPES
    from mi_agent import portfolio_lens as pl
    from mi_agent.llm_query_parser import parse_with_repair
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api import analytical_plan as plan_mod
    from mi_agent_api import chat_routing as routing
    from mi_agent_api import evolution as ev
    from mi_agent_api import portfolio_context as ctx_mod
    from mi_agent_api.data_source import semantics_path
    from question_interpretation import projection

    sem = load_mi_semantics(semantics_path())
    root = os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"]
    frames = ev.funded_frames(root, client_id, None)
    registry = ctx_mod.build_registry(frames[-1]["df"])

    owned = [(c, q) for c, q, other in CASES if other is None]
    expected = len(owned) * len(SCOPES)

    print("=" * 78)
    print("funded_bridge — SHIPPED calculation vs COMPOSITIONAL")
    print("=" * 78)
    import inspect
    params = list(inspect.signature(plan_mod.build_funded_bridge_plan).parameters)
    assert not any("question" in p for p in params), params
    print(f"\nguard: build_funded_bridge_plan{tuple(params)} carries no question — OK")
    print(f"expected comparisons: {len(owned)} cases x {len(SCOPES)} scopes = {expected}\n")

    compared = 0
    fields = 0
    diffs: List[str] = []
    records: List[Dict[str, Any]] = []
    for case, q in owned:
        for sc in SCOPES:
            spec, _m = parse_with_repair(q, sem, llm_enabled=False)

            # ---- SHIPPED (pre-C4 semantics, reconstructed) ----------------
            ship_key, ship_col, ship_label = routing._bridge_dimension(
                getattr(spec, "bridge_dimension", None), sem)
            dl = pl.lens_from_selection(sc) if sc is not None else None
            ship_lens = pl.resolve_lens_with_default(q, dl)
            ship_start = (getattr(spec, "compare_periods", None) or [None])[0]
            ship = ev.funded_bridge(root, client_id, ship_col,
                                    start_period=ship_start, to_run_id=None,
                                    lens_filters=ship_lens.filters or None,
                                    lens_label=ship_lens.label, top_n=8)

            # ---- COMPOSITIONAL --------------------------------------------
            qi = projection.project(q, semantics=sem, frame=frames[-1]["df"],
                                    registry=registry, caller_scope=sc)
            concept = (plan_mod.grouping_concepts(qi) or (None,))[0]
            key, col, label = routing._bridge_dimension(concept, sem)
            comp = plan_mod.funded_bridge(
                root, client_id, interpretation=qi, dimension_columns=col,
                dimension_key=key, dimension_label=label, to_run_id=None)

            case_diffs: List[str] = []
            # population, on the ROWS
            sr = ev._scope_frame_lens(frames[-1]["df"], ship_lens.filters or None)
            pr = ev._scope_frame_lens(frames[-1]["df"],
                                      plan_mod.lens_filters(
                                          plan_mod.build_funded_bridge_plan(
                                              qi, dimension_key=key,
                                              dimension_label=label)))
            fields += 1
            if (0 if sr is None else len(sr)) != (0 if pr is None else len(pr)):
                case_diffs.append(f"POPULATION rows {len(sr)} vs {len(pr)}")
            # the executed axis, and the start period
            fields += 2
            if ship_col != col:
                case_diffs.append(f"dimension columns {ship_col!r} vs {col!r}")
            if ship_start != plan_mod.comparison_period(qi):
                case_diffs.append(f"start period {ship_start!r} vs "
                                  f"{plan_mod.comparison_period(qi)!r}")
            for k in ("available", "reason", "dimensionCol", "netChange", "lens"):
                fields += 1
                if not _close(ship.get(k), comp.get(k)):
                    case_diffs.append(f"{k}: {ship.get(k)!r} vs {comp.get(k)!r}")
            for side in ("start", "end"):
                for k in ("period", "reporting_date", "total"):
                    fields += 1
                    if not _close((ship.get(side) or {}).get(k),
                                  (comp.get(side) or {}).get(k)):
                        case_diffs.append(f"{side}.{k}: "
                                          f"{(ship.get(side) or {}).get(k)!r} vs "
                                          f"{(comp.get(side) or {}).get(k)!r}")
            sc_, cc_ = ship.get("contributions") or [], comp.get("contributions") or []
            fields += 1
            if len(sc_) != len(cc_):
                case_diffs.append(f"contributions length {len(sc_)} vs {len(cc_)}")
            else:
                for i, (x, y) in enumerate(zip(sc_, cc_)):
                    for k in sorted(set(x) | set(y)):
                        fields += 1
                        if not _close(x.get(k), y.get(k)):
                            case_diffs.append(
                                f"contributions[{i}].{k}: {x.get(k)!r} vs {y.get(k)!r}")
            compared += 1
            records.append({"case": case, "scope": str(sc), "question": q,
                            "dimensionCol": comp.get("dimensionCol"),
                            "netChange": comp.get("netChange"),
                            "differences": case_diffs})
            if case_diffs:
                diffs.extend(f"{case}/{sc}: {d}" for d in case_diffs)

    (_REPO / "migration_phase0" / "PLAN_EQUIVALENCE_FUNDED_BRIDGE.json").write_text(
        json.dumps({"comparisons": records}, indent=2, default=str))

    print("=" * 78)
    print(f"comparisons expected     : {expected}")
    print(f"comparisons made         : {compared}")
    print(f"economic fields compared : {fields}")
    print(f"ECONOMIC DIFFERENCES     : {len(diffs)}")
    print(f"A2 tolerance             : {_TOL} (not widened)")
    print("=" * 78)
    if compared != expected:
        print("\nDENOMINATOR UNSOUND: the harness did not compare what it declared.")
        return 2
    for d in diffs:
        print(f"  {d}")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
