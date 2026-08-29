#!/usr/bin/env python3
"""migration_phase0/plan_equivalence_geo_exposure.py — economics, before the switch.

READ-ONLY, and run while `_route_geo` is still on the SHIPPED path. Both sides
are executed here:

  shipped        _resolve_lens(question, scope) -> _apply_lens_filter -> engine
  compositional  contract -> build_geo_exposure_plan -> scope_frame -> engine

and every economic field of the engine result is compared, area by area.

Population equivalence is compared on the ROWS, not on the lens name: Phase 1G
made the plan resolve a category through the governed registry, so it selects
`{'source_portfolio_id': [...]}` where a lens may carry a different shape for
the same population. Comparing names would report the governed correction as a
regression.

    python -m migration_phase0.plan_equivalence_geo_exposure
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: A2's tolerance. Not widened for this route.
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
    from migration_phase0.route_ownership_geo_exposure import CASES, SCOPES
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api import analytical_plan as plan_mod
    from mi_agent_api import chat_routing as routing
    from mi_agent_api import portfolio_context as ctx_mod
    from mi_agent_api.data_source import semantics_path
    from question_interpretation import projection

    semantics = load_mi_semantics(semantics_path())

    def frame_resolver(cli, rid=None):
        # The SAME governed resolver the routed path uses (`mi_service.
        # _routed_frame`). Resolving the frame any other way here would compare
        # two paths over two different books.
        from mi_agent_api import datasets as ds
        pid = f"{cli}/{rid}" if rid else (cli or None)
        frame, err = ds._resolve_query_frame("funded", pid)
        return None if err else frame

    df0 = frame_resolver(client_id, None)
    assert df0 is not None and len(df0), "no governed funded frame to compare on"
    registry = ctx_mod.build_registry(df0)

    owned = [(c, q) for c, q, other in CASES
             if other is None and routing._is_geo_exposure(q)]

    print("=" * 78)
    print("geo_exposure — SHIPPED vs COMPOSITIONAL, before the switch")
    print("=" * 78)
    print(f"\nowned cases {len(owned)} x scopes {len(SCOPES)} = "
          f"{len(owned) * len(SCOPES)} comparisons expected\n")

    # The plan must not be able to read the question.
    import inspect
    params = list(inspect.signature(plan_mod.build_geo_exposure_plan).parameters)
    assert not any("question" in p for p in params), params
    print(f"guard: build_geo_exposure_plan{tuple(params)} carries no question — OK\n")

    compared = 0
    fields_compared = 0
    differences: List[str] = []
    records: List[Dict[str, Any]] = []

    for case, question in owned:
        for scope in SCOPES:
            df = frame_resolver(client_id, None)

            # ---- SHIPPED ------------------------------------------------
            lens = routing._resolve_lens(question, scope)
            ship_df = (routing._apply_lens_filter(df, lens)
                       if lens.filters else df)
            ship_rows = 0 if ship_df is None else len(ship_df)
            from mi_agent_api import geo as geo_mod
            shipped = (dict(geo_mod.exposure_by_itl3(ship_df))
                       if ship_rows else {"available": False, "areas": []})

            # ---- COMPOSITIONAL -----------------------------------------
            interpretation = projection.project(
                question, semantics=semantics, frame=df, registry=registry,
                caller_scope=scope)
            plan = plan_mod.build_geo_exposure_plan(interpretation)
            plan_df = plan_mod.scope_frame(plan, df)
            plan_rows = 0 if plan_df is None else len(plan_df)
            composed = plan_mod.geo_exposure(df, interpretation=interpretation)

            case_diffs: List[str] = []
            if ship_rows != plan_rows:
                case_diffs.append(
                    f"POPULATION: shipped {ship_rows} rows ({lens.filters}), "
                    f"plan {plan_rows} rows ({plan_mod.lens_filters(plan)})")
            fields_compared += 1

            for key in ("available", "reason", "total", "coveragePct", "basis",
                        "areaCount", "resolvedFromItl3Field",
                        "resolvedFromPostcode"):
                fields_compared += 1
                if not _close(shipped.get(key), composed.get(key)):
                    case_diffs.append(f"{key}: shipped={shipped.get(key)!r} "
                                      f"plan={composed.get(key)!r}")

            sa = shipped.get("areas") or []
            ca = composed.get("areas") or []
            fields_compared += 1
            if len(sa) != len(ca):
                case_diffs.append(f"areas length: {len(sa)} vs {len(ca)}")
            else:
                for i, (a, b) in enumerate(zip(sa, ca)):
                    for key in sorted(set(a) | set(b)):
                        fields_compared += 1
                        if not _close(a.get(key), b.get(key)):
                            case_diffs.append(
                                f"areas[{i}].{key}: {a.get(key)!r} vs {b.get(key)!r}")
            # the scope label the ANSWER will say
            fields_compared += 1
            if composed.get("lens") != (lens.label if lens.filters else lens.label):
                case_diffs.append(f"lens label: shipped={lens.label!r} "
                                  f"plan={composed.get('lens')!r}")
            fields_compared += 1
            if bool(composed.get("narrowed")) != bool(lens.filters):
                case_diffs.append(f"narrowed: shipped={bool(lens.filters)} "
                                  f"plan={composed.get('narrowed')}")

            compared += 1
            records.append({"case": case, "scope": str(scope),
                            "question": question,
                            "shippedRows": ship_rows, "planRows": plan_rows,
                            "total": shipped.get("total"),
                            "areaCount": shipped.get("areaCount"),
                            "differences": case_diffs})
            if case_diffs:
                differences.extend(f"{case}/{scope}: {d}" for d in case_diffs)
                print(f"--- {case}/{scope}  {question!r}")
                for d in case_diffs:
                    print(f"    DIFFERENCE {d}")

    expected = len(owned) * len(SCOPES)
    out = _REPO / "migration_phase0" / "PLAN_EQUIVALENCE_GEO_EXPOSURE.json"
    out.write_text(json.dumps({"comparisons": records}, indent=2, default=str))

    print("=" * 78)
    print(f"comparisons expected      : {expected}")
    print(f"comparisons made          : {compared}")
    print(f"economic fields compared  : {fields_compared}")
    print(f"ECONOMIC DIFFERENCES      : {len(differences)}")
    print(f"A2 tolerance              : {_TOL} (not widened)")
    print("=" * 78)
    if compared != expected:
        print("\nDENOMINATOR UNSOUND: the harness did not compare what it "
              "declared. A zero here would mean nothing.")
        return 2
    if fields_compared < expected * 8:
        print(f"\nDENOMINATOR UNSOUND: only {fields_compared} fields over "
              f"{compared} comparisons — too few to be a real comparison.")
        return 2
    if differences:
        print("\nevery difference:")
        for d in differences:
            print(f"  {d}")
        return 1
    print("\n0 differences, over a denominator this harness proved rather than "
          "assumed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
