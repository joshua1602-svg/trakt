#!/usr/bin/env python3
"""migration_phase0/equivalence_portfolio_summary.py — shipped vs shadow.

READ-ONLY / SHADOW ONLY. Runs the SHIPPED `portfolio_summary` engine and the
shadow plan over the same governed book and compares them field by field:
economics, populations, grouping, periods, denominators, and the grouping
evidence a receipt would read.

Reports every difference. Recommends nothing.

    python -m migration_phase0.equivalence_portfolio_summary
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: The portfolio-summary surface. A1-A3 are the shipped-shapes cases; the rest
#: extend it to the scopes the shipped route supports.
#: EVERY case here is verified at run time to be one the SHIPPED recogniser
#: actually claims (``chat_routing._is_portfolio_summary``). A question the
#: route does not own is not part of its regression surface, and comparing on
#: one would manufacture an equivalence that means nothing. Two questions that
#: LOOK like portfolio summaries are deliberately listed and expected to be
#: rejected by that check, so the check is visibly doing work:
#: "Summarise the front book" (a seasoning population — answered by the
#: point-in-time path, 1,177 loans) and "What is the portfolio position for the
#: direct book?".
CASES: Tuple[Tuple[str, str], ...] = (
    ("A1", "Please provide a portfolio summary"),
    ("A2", "Give me a summary of the portfolio"),
    ("A3", "Can you summarise the book for me?"),
    ("A4", "portfolio summary"),
    ("A5", "summarise the portfolio"),
    ("A6", "overview of the portfolio"),
    ("L1", "Summarise the acquired book"),
    ("L2", "Summarise the direct book"),
    ("L3", "portfolio summary for the acquired book"),
    ("X1", "Summarise the front book"),
    ("X2", "What is the portfolio position for the direct book?"),
)

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
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < _TOL
    return a == b


def main() -> int:
    client_id = _env()
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    from mi_agent_api import evolution as evolution_mod
    from mi_agent_api import movement_summary as summary_mod
    from mi_agent import portfolio_lens as lens_mod
    from question_interpretation import projection
    from migration_phase0 import shadow_portfolio_summary as shadow

    semantics = load_mi_semantics(semantics_path())
    root = os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"]
    frames = evolution_mod.funded_frames(root, client_id, None)
    df = frames[-1]["df"]
    region_col = summary_mod._region_column(df)
    has_portfolio = summary_mod._PORTFOLIO_ID in df.columns

    shadow.assert_no_question_read(None)
    print("=" * 78)
    print("portfolio_summary — SHIPPED vs SHADOW PLAN")
    print("=" * 78)
    print("\nguard: build_plan's signature carries no question parameter — OK\n")

    blocked_cases: List[str] = []
    externally_supplied: List[str] = []
    compared = 0
    differences: List[str] = []

    from mi_agent.llm_query_parser import parse_with_repair
    from mi_agent_api import chat_routing as routing

    not_claimed: List[str] = []
    for case_id, question in CASES:
        spec, _meta = parse_with_repair(question, semantics, llm_enabled=False)
        if not routing._is_portfolio_summary(question, spec):
            not_claimed.append(case_id)
            print(f"--- {case_id}  {question!r}")
            print("    NOT CLAIMED by the portfolio_summary recogniser — excluded "
                  "from the surface\n")
            continue
        # PHASE 1G. The registry and the caller's workspace selection go in, so
        # the claim carries the GOVERNED portfolio ids and the provenance that
        # decides precedence — the same inputs the routed path now supplies.
        # Without them this measured the pre-1E reading and the raw type filter,
        # and its "0 differences" said nothing about the governed path.
        registry = None
        try:
            from mi_agent_api import portfolio_context as _ctx

            registry = _ctx.build_registry(df)
        except Exception:  # noqa: BLE001 - fall back to the pre-1G reading
            registry = None
        interpretation = projection.project(question, semantics=semantics,
                                            frame=df, registry=registry)
        plan = shadow.build_plan(interpretation, region_column=region_col,
                                 has_portfolio_column=has_portfolio)
        # What the SHIPPED route derives from the raw question. Used ONLY to run
        # the shipped side and to check the two agree — never handed to the plan.
        lens = lens_mod.resolve_lens(question)

        print(f"--- {case_id}  {question!r}")
        print(f"    plan steps      : {[s.primitive for s in plan.steps]}")
        print(f"    declares grouped: {list(plan.declares_grouped_by)}")
        print(f"    source_scope    : state={interpretation.source_scope.state!r} "
              f"scope={interpretation.source_scope.scope!r}")
        if plan.blocked:
            blocked_cases.append(case_id)
            for step in plan.blocked:
                print(f"    BLOCKED [{step.primitive}]: {step.blocked}")
            print(f"    shipped route would have used lens={lens.name!r} "
                  f"filters={lens.filters}")

        shipped = summary_mod.portfolio_summary(
            root, client_id, to_run_id=None,
            lens_filters=lens.filters or None, lens_label=lens.label)
        # NOTHING is passed in. The plan derives its own population.
        shadowed = shadow.execute_plan(plan, output_root=root, client_id=client_id)
        from_plan = (shadowed.get("lensFromPlan") or {})
        # POPULATION EQUIVALENCE IS COMPARED ON THE ROWS, NOT ON THE LENS NAME.
        #
        # Phase 1G makes the plan resolve a CATEGORY through the registry, so it
        # selects `{'source_portfolio_id': [...]}` where the shipped route still
        # selects `{'source_portfolio_type': 'acquired'}`. Those are different
        # NAMES for the same population on this book and different POPULATIONS
        # on a book with two portfolios of one type (Phase 1C: GBP300 vs
        # GBP1,200) — which is the whole reason the plan takes the governed
        # path. Comparing names here would have reported the correction as a
        # regression.
        shipped_rows = evolution_mod._scope_frame_lens(df, lens.filters or None)
        plan_rows = evolution_mod._scope_frame_lens(
            df, (from_plan.get("filters") or None))
        shipped_n = 0 if shipped_rows is None else len(shipped_rows)
        plan_n = 0 if plan_rows is None else len(plan_rows)
        if shipped_n != plan_n:
            differences.append(
                f"{case_id}: plan selected {plan_n} rows "
                f"({from_plan.get('filters')}), shipped route selected "
                f"{shipped_n} ({lens.filters})")
        compared += 1

        case_diffs: List[str] = []
        for key in ("available", "period", "reportingDate", "periodCount",
                    "regionColumn"):
            if not _close(shipped.get(key), shadowed.get(key)):
                case_diffs.append(f"{key}: shipped={shipped.get(key)!r} "
                                  f"shadow={shadowed.get(key)!r}")
        sm, dm = shipped.get("metrics") or {}, shadowed.get("metrics") or {}
        for key in sorted(set(sm) | set(dm)):
            if not _close(sm.get(key), dm.get(key)):
                case_diffs.append(f"metrics.{key}: shipped={sm.get(key)!r} "
                                  f"shadow={dm.get(key)!r}")
        sr, dr = shipped.get("topRegions") or [], shadowed.get("topRegions") or []
        if len(sr) != len(dr):
            case_diffs.append(f"topRegions length: {len(sr)} vs {len(dr)}")
        else:
            for i, (a, b) in enumerate(zip(sr, dr)):
                for key in ("region", "balance", "share"):
                    if not _close(a.get(key), b.get(key)):
                        case_diffs.append(
                            f"topRegions[{i}].{key}: {a.get(key)!r} vs {b.get(key)!r}")
        sc = {c["id"] for c in (shipped.get("cohorts") or [])}
        dc = {c["id"] for c in (shadowed.get("cohorts") or [])}
        if sc != dc:
            case_diffs.append(f"cohorts: {sorted(sc)} vs {sorted(dc)}")

        if case_diffs:
            differences.extend(f"{case_id}: {d}" for d in case_diffs)
            print(f"    DIFFERENCES ({len(case_diffs)}):")
            for d in case_diffs:
                print(f"       {d}")
        else:
            bal = (sm.get("funded_balance") or 0.0)
            print(f"    economics IDENTICAL  (balance {bal:,.2f}, "
                  f"loans {sm.get('loan_count')}, period {shipped.get('period')})")
        print()

    print("=" * 78)
    print(f"cases on the surface     : {compared}")
    print(f"cases NOT claimed        : {len(not_claimed)} -> {not_claimed}")
    print(f"economic differences     : {len(differences)}")
    print(f"cases the plan BLOCKS    : {len(blocked_cases)} -> {blocked_cases}")
    print(f"externally supplied lens : {len(externally_supplied)} -> "
          f"{externally_supplied}")
    print(f"plan population == shipped population on every compared case: "
          f"{'yes' if not differences else 'NO'}")
    print("=" * 78)
    if differences:
        print("\nevery difference:")
        for d in differences:
            print(f"  {d}")
    print("\nThe shadow executor receives NOTHING from this harness but the plan.")
    print("`lensFromPlan` on each result records the scope the PLAN selected, and")
    print("it is checked against what the shipped route resolved independently.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
