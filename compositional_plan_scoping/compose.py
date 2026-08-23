#!/usr/bin/env python3
"""compositional_plan_scoping/compose.py — do the routes factor?

READ-ONLY. Every callable used below ALREADY SHIPS and is used UNMODIFIED. This
instrument writes no product code, patches nothing and imports nothing into the
serving path. It exists to answer one question with a number rather than an
opinion:

    Can the shapes that hold 23 of the 27 measured capability failures —
    T3, T4, T5, T6, T7 — be expressed as compositions of primitives the
    product already owns?

It composes each shape from four existing primitives:

    stack periods    mi_agent_api.evolution.funded_frames
    select population mi_agent.seasoning.resolve_population_predicate
    group            mi_agent.mi_query_executor._grouped_aggregate   (N-ARY)
    measure          the same call's aggregation argument
    compare          mi_workflows.engine.compare_values

and then applies the migration discipline this programme uses: each composition
is summed back over the dimensions it added and **reconciled against the answer
the product ships today**. A composition that does not reconcile has not
expressed the same measure, whatever its shape.

    python -m compositional_plan_scoping.compose
    python -m compositional_plan_scoping.compose --book alderbridge
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

BALANCE = "current_outstanding_balance"
TOLERANCE = 0.005  # pence


def _book(book: str) -> str:
    """Set the governed environment for ``book`` and return its client id."""
    if book == "alderbridge":
        from demo_platform import config as cfg
        os.environ.update(cfg.mi_env(period_role="current"))
        client_id = cfg.CLIENT_ID
    elif book == "kestrelmoor":
        from question_interpretation.run_robustness_deterministic import _KESTRELMOOR_ROOT
        from tests.analytical import second_book as bk
        if not _KESTRELMOOR_ROOT.exists():
            bk.build(_KESTRELMOOR_ROOT)
        os.environ.update(bk.mi_env(_KESTRELMOOR_ROOT))
        client_id = os.environ["MI_AGENT_CLIENT_ID"]
    else:
        raise ValueError("unknown book %r" % (book,))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return client_id


def _first_present(frame, *candidates: str) -> Optional[str]:
    cols = set(frame.columns)
    return next((c for c in candidates if c in cols), None)


def compose(frames: Sequence[Dict[str, Any]], dims: Sequence[str], *,
            metric: str = BALANCE, aggregation: str = "sum",
            population=None) -> List[Dict[str, Any]]:
    """``stack periods`` x ``select population`` x ``group(dims)`` x ``measure``.

    One expression. ``dims`` is arbitrary length — the N-ARY group primitive is
    the one the point-in-time executor already uses; nothing here extends it.
    """
    from mi_agent.mi_query_executor import _grouped_aggregate

    out: List[Dict[str, Any]] = []
    for frame in frames:
        df = frame["df"]
        period = str(frame["reporting_date"])[:7]
        if population is not None:
            df = df[population(df)]
        if not dims:
            rows = [{f"{metric}_{aggregation}": float(df[metric].sum())}]
        else:
            stringified = df.assign(**{d: df[d].astype(str) for d in dims})
            grouped, _col = _grouped_aggregate(
                stringified, list(dims), metric, aggregation, None, None)
            rows = grouped.to_dict("records")
        for row in rows:
            out.append({"period": period, **row})
    return out


def reconcile(composed: List[Dict[str, Any]], shipped: List[Dict[str, Any]],
              value_key: str) -> List[Dict[str, Any]]:
    """Sum a composition back over the dimensions it added, against the shipped
    series. This is the equivalence test, not a smoke test."""
    totals: Dict[str, float] = defaultdict(float)
    for row in composed:
        totals[row["period"]] += float(row[value_key] or 0.0)
    base = {r["period"]: float(r[value_key] or 0.0) for r in shipped}
    out = []
    for period in sorted(base):
        delta = totals[period] - base[period]
        out.append({"period": period, "composed": totals[period],
                    "shipped": base[period], "delta": delta,
                    "ok": abs(delta) < TOLERANCE})
    return out


def run(book: str = "alderbridge") -> int:
    client_id = _book(book)
    from mi_agent_api import evolution as evolution_mod
    from mi_agent import seasoning as seasoning_mod
    from mi_workflows import engine

    root = os.environ.get("MI_AGENT_ONBOARDING_OUTPUT_ROOT")
    frames = evolution_mod.funded_frames(root, client_id, None)
    if len(frames) < 2:
        print(f"{book}: {len(frames)} governed snapshot(s) — a time axis needs two.")
        return 1

    value_key = f"{BALANCE}_sum"
    df0 = frames[0]["df"]
    region = _first_present(df0, "geographic_region_obligor", "collateral_geography")
    ltv = _first_present(df0, "ltv_bucket")
    seasoning_col = _first_present(df0, "seasoning_segment")

    print("=" * 78)
    print(f"COMPOSITION PROBE — {book}")
    print("=" * 78)
    print(f"\nstack periods -> {len(frames)} governed frames: "
          f"{[str(f['reporting_date'])[:7] for f in frames]}")
    print(f"dimensions    -> region={region!r}  ltv={ltv!r}  seasoning={seasoning_col!r}")

    print("\nCOMPOSITIONS (every primitive already ships, unmodified)\n")
    t1 = compose(frames, [])
    print(f"  T1  stack x measure                          {len(t1):6d} rows")

    results: Dict[str, List[Dict[str, Any]]] = {}
    if region:
        results["T3"] = compose(frames, [region])
        print(f"  T3  stack x group(region) x measure          {len(results['T3']):6d} rows")
    if region and seasoning_col:
        predicate = seasoning_mod.resolve_population_predicate("for the front book")
        wanted = str(list((predicate or {}).values())[0]).strip().casefold() if predicate else None
        if wanted:
            selector = (lambda d: d[seasoning_col].astype(str).str.strip()
                        .str.casefold() == wanted)
            results["T4"] = compose(frames, [region], population=selector)
            # Row COUNT is not evidence that the population narrowed — a book
            # whose every region survives the predicate has the same number of
            # groups either way. The balance is, so it is what is reported.
            narrowed = sum(float(r[value_key] or 0.0) for r in results["T4"])
            whole = sum(float(r[value_key] or 0.0) for r in results["T3"])
            share = (narrowed / whole * 100.0) if whole else 0.0
            print(f"  T4  stack x select x group(region) x measure  "
                  f"{len(results['T4']):6d} rows   (population {predicate}; "
                  f"{share:.1f}% of the book by balance)")
    if region and ltv:
        results["T5"] = compose(frames, [region, ltv])
        print(f"  T5  stack x group(region, ltv) x measure     {len(results['T5']):6d} rows")

    if region:
        def by_region(df):
            from mi_agent.mi_query_executor import _grouped_aggregate
            grouped, col = _grouped_aggregate(
                df.assign(**{region: df[region].astype(str)}),
                [region], BALANCE, "sum", None, None)
            return {r[region]: r[col] for r in grouped.to_dict("records")}

        prior, latest = by_region(frames[-2]["df"]), by_region(frames[-1]["df"])
        movement = []
        for key in set(prior) | set(latest):
            cmp = engine.compare_values(latest.get(key), prior.get(key))
            if cmp.get("absolute") is not None:
                movement.append({"category": key, **cmp})
        movement.sort(key=lambda m: abs(m["absolute"]), reverse=True)
        print(f"  T6  compare(T3[-1], T3[-2]) elementwise      {len(movement):6d} categories")
        print(f"  T7  rank(T6)                                 top 3 by |change|: "
              f"{[(m['category'], round(m['absolute'])) for m in movement[:3]]}")

    print("\nEQUIVALENCE — each composition summed back, against the SHIPPED T1 series\n")
    all_ok = True
    for tag in ("T3", "T5"):
        if tag not in results:
            continue
        print(f"  {tag}:")
        for row in reconcile(results[tag], t1, value_key):
            all_ok &= row["ok"]
            print(f"     {row['period']}  composed={row['composed']:>18,.2f}  "
                  f"shipped={row['shipped']:>18,.2f}  delta={row['delta']:>10,.4f}  "
                  f"{'OK' if row['ok'] else 'MISMATCH'}")

    print(f"\n  -> {'RECONCILES EXACTLY' if all_ok else 'DOES NOT RECONCILE'}: the "
          f"composition is the same measure, re-partitioned.")
    print("\n  T4 is NOT reconciled against T1 by construction — it selects a "
          "narrower\n  population, so a matching total would be the bug.")

    # ---- What arity costs, which reconciliation does not catch ------------- #
    #
    # A composition can be arithmetically exact and still be a worse answer.
    # ``_execute_grouped`` attaches the ``loan_count`` denominator and raises the
    # thin-sample warning ONLY when ``len(group_cols) == 1``. Every disclosure
    # that protects a sparse group is therefore written for arity 1 and does not
    # fire at arity 2 — which is the first arity a compositional layer unlocks.
    from mi_agent.mi_query_executor import LOW_GROUP_COUNT
    print(f"\nARITY COST — groups below the product's own thin-sample floor "
          f"({LOW_GROUP_COUNT} loans)\n")
    for dims, label in ((([region], "T3  group(region)") if region else (None, None)),
                        (([region, ltv], "T5  group(region, ltv)")
                         if region and ltv else (None, None))):
        if not dims:
            continue
        total = thin = 0
        for frame in frames:
            df = frame["df"]
            sizes = (df.assign(**{d: df[d].astype(str) for d in dims})
                     .groupby(list(dims), sort=False).size())
            total += len(sizes)
            thin += int((sizes < LOW_GROUP_COUNT).sum())
        pct = (thin / total * 100.0) if total else 0.0
        print(f"  {label:26s} {total:6d} groups   {thin:6d} thin   {pct:5.1f}%")
    print("\n  -> the thin-sample disclosure is guarded by `len(group_cols) == 1`"
          "\n     (mi_agent/mi_query_executor.py::_execute_grouped), so at arity 2"
          "\n     these groups are neither counted nor disclosed.\n")
    return 0 if all_ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m compositional_plan_scoping.compose")
    ap.add_argument("--book", default="alderbridge",
                    choices=("alderbridge", "kestrelmoor"))
    args = ap.parse_args(argv)
    return run(args.book)


if __name__ == "__main__":
    sys.exit(main())
