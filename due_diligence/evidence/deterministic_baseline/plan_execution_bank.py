#!/usr/bin/env python3
"""Deterministic execution bank: governed plan in, verified result out.

    known-correct governed plan
             -> deterministic capability / executor
             -> execution receipt
             -> independently verified result

No English parser, no ParsedQuestion, no recogniser cascade, no raw-question
routing, no Opus. Every plan is constructed directly.

Numeric truth comes from `mi_agent.tests.portfolio_truth_oracle` (which imports
nothing from the product) or is recomputed here in pandas. Where neither is
available a case is recorded as DISPOSITION-ONLY and never counted as a numeric
pass — the engine never supplies its own expected answer.

Nothing is fixed. A failing case is recorded and the bank continues.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("DET_BANK_ROOT")
            or Path(__file__).resolve().parents[3])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mi_agent.mi_query_validator import load_mi_semantics  # noqa: E402
from mi_agent.query_plan import (  # noqa: E402
    AVERAGE, COUNT, SUM, WEIGHTED_AVERAGE, AnalyticalScope, PlannedOutput,
    Predicate, QueryPlan)
from mi_agent.query_plan_execution import execute_query_plan  # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth  # noqa: E402

SEMANTICS = load_mi_semantics(str(ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
BOOK = truth.canonical_book()
BALANCE, LTV, RATE, AGE = truth.BALANCE, truth.LTV, truth.RATE, truth.AGE

CASES = []


def case(case_id, surface, *, expected, truth_source, plan_note):
    """Register one bank case. `expected` is a dict the runner interprets."""
    def wrap(fn):
        CASES.append({"case_id": case_id, "surface": surface, "fn": fn,
                      "expected": expected, "truth_source": truth_source,
                      "plan_note": plan_note})
        return fn
    return wrap


def plan_for(outputs, *, predicates=(), dimensions=(), dataset="funded",
             period="2026-06-30"):
    return QueryPlan(
        shared_scope=AnalyticalScope(
            dataset=dataset, period=period,
            filters=tuple(Predicate(*p) for p in predicates),
            dimensions=tuple(dimensions)),
        outputs=tuple(outputs))


def execute(plan, data=None):
    """Run a plan and return (envelope, receipt-ish dict).

    An engine exception is a DISPOSITION, not a harness failure: "the plan-level
    entry point raised instead of refusing" is exactly the kind of thing this
    bank exists to observe, so it is recorded rather than propagated.
    """
    try:
        env = execute_query_plan(plan, BOOK if data is None else data, SEMANTICS,
                                 validate=False)
    except Exception as exc:                                     # noqa: BLE001
        return None, {"complete": False, "raised": type(exc).__name__,
                      "reason": str(exc)[:300],
                      "outputs": [{"output_id": "a", "value": None, "ok": False,
                                   "error": str(exc)[:300],
                                   "group_field_keys": [], "filters_applied": None,
                                   "dataset": None, "period": None, "rows": 0}]}
    receipt = {
        "complete": bool(env.completeness.complete),
        "reason": env.completeness.reason() if not env.completeness.complete else "",
        "outputs": [],
    }
    for o in env.outputs:
        ref = o.execution_ref
        md = (ref.metadata or {}) if ref is not None else {}
        receipt["outputs"].append({
            "output_id": o.output_id,
            "value": None if o.value is None else float(o.value),
            "ok": ref is not None and getattr(ref, "data", None) is not None,
            "error": getattr(ref, "error", "") or "",
            "warnings": list(getattr(ref, "warnings", ()) or ()),
            "result_type": getattr(ref, "result_type", None),
            "row_count": getattr(ref, "row_count", None),
            "group_field_keys": list(md.get("group_field_keys") or ()),
            "filters_applied": (md.get("applied_predicates")
                                or md.get("applied_filter_fields")
                                or md.get("filters_applied")),
            "dataset": md.get("dataset") or md.get("source"),
            "period": (md.get("period") or md.get("reporting_date")
                       or md.get("as_of_date") or md.get("period_executed")),
            "filtered_row_count": md.get("filtered_row_count"),
            "rows": int(len(ref.data)) if getattr(ref, "data", None) is not None else 0,
        })
    return env, receipt


def cells(env, output_id, column):
    if env is None:
        return {}
    result = next(o for o in env.outputs if o.output_id == output_id).execution_ref
    if result is None or getattr(result, "data", None) is None:
        return {}
    keys = list((result.metadata or {}).get("group_field_keys") or ())
    frame = result.data
    # The executor names an aggregated column `<measure>_<agg>`; the raw measure
    # name is not a column of a grouped result.
    col = column if column in frame.columns else f"{column}_sum"
    return {tuple(str(row[k]) for k in keys): float(row[col])
            for _, row in frame.iterrows()}


# ===================== A. AGGREGATION / STATISTIC ========================== #

@case("A1-count", "AGGREGATION/STATISTIC", expected={"value": "oracle"},
      truth_source="independent_oracle", plan_note="COUNT over funded book")
def a1():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=COUNT)]))
    return {"actual": float(env.outputs[0].value),
            "expected": float(truth.row_count(BOOK)), "receipt": rc}


@case("A2-sum-balance", "AGGREGATION/STATISTIC", expected={"value": "oracle"},
      truth_source="independent_oracle", plan_note="SUM(balance)")
def a2():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)]))
    return {"actual": float(env.outputs[0].value),
            "expected": float(truth.total(BOOK, BALANCE)), "receipt": rc}


@case("A3-weighted-avg-ltv", "AGGREGATION/STATISTIC", expected={"value": "oracle"},
      truth_source="independent_oracle", plan_note="WA(LTV) weighted by balance")
def a3():
    env, rc = execute(plan_for([PlannedOutput(
        output_id="a", operation=WEIGHTED_AVERAGE, measure=LTV,
        weight_field=BALANCE)]))
    return {"actual": float(env.outputs[0].value),
            "expected": float(truth.weighted_average(BOOK, LTV, BALANCE)),
            "receipt": rc}


@case("A4-simple-avg-age", "AGGREGATION/STATISTIC", expected={"value": "oracle"},
      truth_source="independent_oracle",
      plan_note="AVERAGE(age) — an unweighted mean must not silently weight")
def a4():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=AVERAGE,
                                              measure=AGE)]))
    return {"actual": float(env.outputs[0].value),
            "expected": float(BOOK[AGE].mean()), "receipt": rc}


# ===================== B. POPULATION / FILTER ============================== #

JOINT = ("borrower_type", "eq", "Joint")
SCOTLAND = ("collateral_geography", "eq", "Scotland")

@case("B1-single-filter", "POPULATION/FILTER", expected={"value": "oracle"},
      truth_source="independent_oracle", plan_note="SUM(balance) WHERE joint")
def b1():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=[JOINT]))
    return {"actual": float(env.outputs[0].value),
            "expected": float(truth.total(BOOK, BALANCE, [JOINT])),
            "receipt": rc}


@case("B2-two-filters", "POPULATION/FILTER", expected={"value": "oracle"},
      truth_source="independent_oracle",
      plan_note="two filters must narrow together, not replace one another")
def b2():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=[JOINT, SCOTLAND]))
    return {"actual": float(env.outputs[0].value),
            "expected": float(truth.total(BOOK, BALANCE, [JOINT, SCOTLAND])),
            "receipt": rc}


@case("B3-threshold-keeps-every-row", "POPULATION/FILTER",
      expected={"value": "oracle"}, truth_source="independent_oracle",
      plan_note="a predicate true of every row must still be APPLIED and recorded")
def b3():
    pred = (LTV, "gt", 0.0)
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=[pred]))
    return {"actual": float(env.outputs[0].value),
            "expected": float(truth.total(BOOK, BALANCE, [pred])),
            "receipt": rc, "note": "filter must appear in the receipt"}


@case("B4-filter-keeps-no-row", "POPULATION/FILTER",
      expected={"disposition": "EMPTY_OR_REFUSAL_NOT_INVENTED_ZERO"},
      truth_source="recomputed",
      plan_note="an empty population must not be reported as a confident 0")
def b4():
    pred = (LTV, "gt", 1000.0)
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=[pred]))
    rows = int(truth.row_count(BOOK, [pred]))
    return {"actual": rc["outputs"][0], "expected_rows": rows, "receipt": rc}


# ===================== C. DIMENSIONS ======================================= #

@case("C1-one-dimension", "PLAN_BINDING/DIMENSIONS", expected={"cells": "oracle"},
      truth_source="independent_oracle", plan_note="SUM(balance) by product type")
def c1():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=["erm_product_type"]))
    got = cells(env, "a", BALANCE)
    want = {k: float(v) for k, v in
            truth.grouped(BOOK, ["erm_product_type"], column=BALANCE,
                          how="sum").items()}
    return {"actual": got, "expected": want, "receipt": rc}


@case("C2-two-dimension-grid", "PLAN_BINDING/DIMENSIONS",
      expected={"cells": "oracle"}, truth_source="independent_oracle",
      plan_note="a grid's mass must be in the right cells, not merely total right")
def c2():
    dims = ["erm_product_type", "collateral_geography"]
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=dims))
    got = cells(env, "a", BALANCE)
    want = {k: float(v) for k, v in truth.grouped(BOOK, dims, column=BALANCE,
                                                  how="sum").items()}
    return {"actual": got, "expected": want, "receipt": rc}


@case("C3-absent-dimension", "RECEIPT/GOVERNANCE",
      expected={"disposition": "GOVERNED_REFUSAL_NO_SUBSTITUTION"},
      truth_source="recomputed",
      plan_note="a dimension the dataset lacks must refuse, never be swapped")
def c3():
    thin = BOOK.drop(columns=["erm_product_type"])
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=["erm_product_type"]), data=thin)
    return {"actual": rc["outputs"][0], "receipt": rc,
            "dataset_columns": len(thin.columns)}


# ===================== D. GEOGRAPHY ======================================== #

@case("D1-region-grouping", "GEOGRAPHY", expected={"cells": "oracle"},
      truth_source="independent_oracle",
      plan_note="grouping on the collateral geography column the book carries")
def d1():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=["collateral_geography"]))
    got = cells(env, "a", BALANCE)
    want = {k: float(v) for k, v in
            truth.grouped(BOOK, ["collateral_geography"], column=BALANCE,
                          how="sum").items()}
    return {"actual": got, "expected": want, "receipt": rc}


@case("D2-itl3-codes", "GEOGRAPHY",
      expected={"disposition": "CODES_MUST_NOT_BE_PRESENTED_AS_PLACE_NAMES"},
      truth_source="recomputed",
      plan_note="a column of ITL3 CODES grouped as a region: codes are not names")
def d2():
    book = BOOK.copy()
    book["geographic_region_obligor_itl3"] = [
        "TLI31", "TLJ21", "TLM50", "TLD61"][: 1] * len(book)
    book.loc[: len(book) // 2, "geographic_region_obligor_itl3"] = "TLJ21"
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=["geographic_region_obligor_itl3"]),
                      data=book)
    got = cells(env, "a", BALANCE) if rc["outputs"][0]["ok"] else {}
    want = {(str(k),): float(v) for k, v in
            book.groupby("geographic_region_obligor_itl3")[BALANCE].sum().items()}
    return {"actual": rc["outputs"][0], "cells": got, "expected": want,
            "receipt": rc,
            "labels_returned": sorted({k[0] for k in got}) if got else []}


# ===================== E. BRIDGE / MOVEMENT ================================ #

def _bridge(start_map, end_map):
    from mi_agent.period_change.bridge import balance_bridge
    from mi_agent.period_change.models import SnapshotFrame

    def snap(d, sid, date):
        frame = pd.DataFrame({"loan_identifier": list(d),
                              BALANCE: [float(v) for v in d.values()]})
        return SnapshotFrame(snapshot_id=sid, reporting_date=date, frame=frame)
    return balance_bridge(snap(start_map, "s0", "2026-05-31"),
                          snap(end_map, "s1", "2026-06-30"))


def _bridge_truth(start_map, end_map):
    """Independent decomposition, written out longhand."""
    s, e = set(start_map), set(end_map)
    cont = sum(end_map[k] - start_map[k] for k in s & e)
    new = sum(end_map[k] for k in e - s)
    red = sum(start_map[k] for k in s - e)
    return {"continuing": round(cont, 6), "new": round(new, 6),
            "redeemed": round(-red, 6),
            "net": round(sum(end_map.values()) - sum(start_map.values()), 6)}


@case("E1-bridge-decomposition", "SPECIALIST_CAPABILITY",
      expected={"bridge": "recomputed"}, truth_source="recomputed",
      plan_note="continuing / new / redeemed against a longhand decomposition")
def e1():
    start = {"L1": 100.0, "L2": 200.0, "L3": 300.0}
    end = {"L1": 90.0, "L2": 250.0, "L4": 400.0}
    out = _bridge(start, end)
    return {"actual": out, "expected": _bridge_truth(start, end), "receipt": None}


@case("E2-bridge-full-redemption", "SPECIALIST_CAPABILITY",
      expected={"bridge": "recomputed"}, truth_source="recomputed",
      plan_note="every opening loan gone: redemption must be the whole opening book")
def e2():
    start = {"L1": 100.0, "L2": 50.0}
    end = {"L3": 10.0}
    out = _bridge(start, end)
    return {"actual": out, "expected": _bridge_truth(start, end), "receipt": None}


@case("E3-bridge-no-change", "SPECIALIST_CAPABILITY",
      expected={"bridge": "recomputed"}, truth_source="recomputed",
      plan_note="identical snapshots: a zero movement must read as zero, not as a change")
def e3():
    start = {"L1": 100.0, "L2": 50.0}
    out = _bridge(start, dict(start))
    return {"actual": out, "expected": _bridge_truth(start, dict(start)),
            "receipt": None}


# ===================== F. TEMPORAL ========================================= #

@case("F1-as-at-unhonourable", "TEMPORAL_RESOLUTION",
      expected={"disposition": "GOVERNED_REFUSAL_OR_DISCLOSED"},
      truth_source="recomputed",
      plan_note="a period the flat book cannot honour must not answer as if current")
def f1():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               period="1999-12-31"))
    return {"actual": rc["outputs"][0], "receipt": rc,
            "book_has_no_period_column": "reporting_date" not in BOOK.columns,
            "unfiltered_total": float(truth.total(BOOK, BALANCE))}


@case("F2-period-is-recorded", "RECEIPT/GOVERNANCE",
      expected={"disposition": "RECEIPT_STATES_THE_PERIOD_EXECUTED"},
      truth_source="recomputed",
      plan_note="the receipt must say which period the figure is for")
def f2():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               period="2026-06-30"))
    return {"actual": rc["outputs"][0], "receipt": rc}


# ===================== G. MULTI-OUTPUT / RECONCILIATION ==================== #

@case("G1-multi-output-one-population", "AGGREGATION/STATISTIC",
      expected={"multi": "oracle"}, truth_source="independent_oracle",
      plan_note="three measures over one population, each against its own truth")
def g1():
    env, rc = execute(plan_for([
        PlannedOutput(output_id="n", operation=COUNT),
        PlannedOutput(output_id="b", operation=SUM, measure=BALANCE),
        PlannedOutput(output_id="w", operation=WEIGHTED_AVERAGE, measure=LTV,
                      weight_field=BALANCE)], predicates=[JOINT]))
    got = {o.output_id: float(o.value) for o in env.outputs}
    want = {"n": float(truth.row_count(BOOK, [JOINT])),
            "b": float(truth.total(BOOK, BALANCE, [JOINT])),
            "w": float(truth.weighted_average(BOOK, LTV, BALANCE, [JOINT]))}
    return {"actual": got, "expected": want, "receipt": rc}


@case("G2-grid-reconciles-to-total", "AGGREGATION/STATISTIC",
      expected={"reconcile": "oracle"}, truth_source="independent_oracle",
      plan_note="the cells of a grid must sum to the population total")
def g2():
    dims = ["erm_product_type", "collateral_geography"]
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=dims, predicates=[JOINT]))
    got = sum(cells(env, "a", BALANCE).values())
    want = float(truth.total(BOOK, BALANCE, [JOINT]))
    return {"actual": got, "expected": want, "receipt": rc}


# =============================== runner ==================================== #

def close(a, b, places=2):
    return abs(float(a) - float(b)) < 10 ** -places


def judge(c, out):
    """PASS/FAIL plus a classification. Semantics beat plausibility."""
    exp, cid = c["expected"], c["case_id"]
    rc = out.get("receipt")

    if "value" in exp:
        if not close(out["actual"], out["expected"]):
            return "FAIL", "NUMERICAL_ERROR", (
                f"expected {out['expected']!r}, got {out['actual']!r}")
        if cid == "B3-threshold-keeps-every-row":
            applied = rc["outputs"][0].get("filters_applied")
            if not applied:
                return "FAIL", "RECEIPT/GOVERNANCE", (
                    "a predicate true of every row was applied but the receipt "
                    f"records no filter: {applied!r}")
        return "PASS", None, ""

    if "cells" in exp or "multi" in exp:
        if set(out["actual"]) != set(out["expected"]):
            return "FAIL", "PLAN_BINDING", (
                f"key sets differ: got {sorted(out['actual'])[:4]} "
                f"want {sorted(out['expected'])[:4]}")
        for k, v in out["expected"].items():
            if not close(out["actual"][k], v, 3):
                return "FAIL", "NUMERICAL_ERROR", f"cell {k}: {out['actual'][k]} != {v}"
        return "PASS", None, ""

    if "reconcile" in exp:
        return ("PASS", None, "") if close(out["actual"], out["expected"]) else (
            "FAIL", "AGGREGATION/STATISTIC",
            f"cells sum to {out['actual']}, population total is {out['expected']}")

    if "bridge" in exp:
        got, want = out["actual"], out["expected"]
        # The capability's own names for the three components. `exited_loan_balance`
        # is reported POSITIVE, so it is compared against the magnitude of the
        # oracle's signed redemption.
        checks = [
            ("continuing", float(got.continuing_movement), want["continuing"]),
            ("new", float(got.new_loan_balance), want["new"]),
            ("redeemed", -float(got.exited_loan_balance), want["redeemed"]),
            ("net", float(got.closing_balance) - float(got.opening_balance),
             want["net"]),
        ]
        miss = [f"{n}: want {w}, got {g}" for n, g, w in checks
                if not close(g, w, 4)]
        # A bridge that does not reconcile is a failure even if every component
        # happens to match: the residual is the capability's own verdict.
        if not bool(getattr(got, "reconciles", False)):
            miss.append(f"does not reconcile, residual={getattr(got,'residual',None)!r}")
        if miss:
            return "FAIL", "CALCULATION", "; ".join(miss)
        return "PASS", None, (
            f"opening={got.opening_balance} closing={got.closing_balance} "
            f"continuing={got.continuing_movement} new={got.new_loan_balance} "
            f"exited={got.exited_loan_balance} residual={got.residual}")

    d = exp.get("disposition")
    o0 = out["actual"] if isinstance(out["actual"], dict) else {}
    if d == "EMPTY_OR_REFUSAL_NOT_INVENTED_ZERO":
        if out["expected_rows"] != 0:
            return "FAIL", "OTHER", "fixture error: population is not empty"
        filtered = o0.get("filtered_row_count")
        warned = [w for w in (o0.get("warnings") or ())
                  if "empt" in str(w).lower() or "no row" in str(w).lower()
                  or "unavailable" in str(w).lower()]
        if o0.get("ok") and o0.get("value") == 0.0 and not warned:
            return "FAIL", "RECEIPT/GOVERNANCE", (
                f"an empty population (filtered_row_count={filtered!r}) returned a "
                f"confident 0.0 with no unavailability warning — an invented zero")
        return "PASS", None, (
            f"empty population -> value={o0.get('value')!r}, warnings={warned}")
    if d == "GOVERNED_REFUSAL_NO_SUBSTITUTION":
        if o0.get("ok"):
            return "FAIL", "RECEIPT/GOVERNANCE", "an absent dimension still returned ok"
        if o0.get("group_field_keys"):
            return "FAIL", "PLAN_BINDING", (
                f"substituted a grouping field: {o0['group_field_keys']}")
        return "PASS", None, f"refused: {o0.get('error','')[:110]}"
    if d == "GOVERNED_REFUSAL_OR_DISCLOSED":
        total = out["unfiltered_total"]
        if o0.get("ok") and close(o0.get("value") or 0, total):
            return "FAIL", "TEMPORAL_RESOLUTION", (
                "a period the book cannot honour returned the whole-book total "
                "as though it were that period's figure")
        return "PASS", None, f"ok={o0.get('ok')}, value={o0.get('value')!r}"
    if d == "RECEIPT_STATES_THE_PERIOD_EXECUTED":
        if not o0.get("period"):
            return "FAIL", "RECEIPT/GOVERNANCE", (
                "the receipt does not state which period was executed")
        return "PASS", None, f"period={o0.get('period')!r}"
    if d == "CODES_MUST_NOT_BE_PRESENTED_AS_PLACE_NAMES":
        labels = out.get("labels_returned") or []
        if not o0.get("ok"):
            return "PASS", None, (
                f"refused rather than mislabel: "
                f"{(o0.get('error') or 'execution produced no frame')[:100]}")
        codish = [x for x in labels if x.upper().startswith("TL")]
        if codish and len(codish) == len(labels):
            return "FAIL", "GEOGRAPHY", (
                f"returned raw ITL3 codes as region labels: {labels[:4]}")
        return "PASS", None, f"labels={labels[:4]}"
    return "FAIL", "OTHER", f"unjudgeable expectation {exp!r}"


def main():
    records = []
    for c in CASES:
        rec = {"case_id": c["case_id"], "surface": c["surface"],
               "plan": c["plan_note"], "truth_source": c["truth_source"],
               "expected_disposition": c["expected"]}
        try:
            out = c["fn"]()
            verdict, owner, detail = judge(c, out)
            rec.update({
                "verdict": verdict, "classification": owner, "detail": detail,
                "expected_result": out.get("expected", out.get("expected_rows")),
                "actual_result": out.get("actual"),
                "receipt": out.get("receipt"),
            })
        except Exception:                                        # noqa: BLE001
            rec.update({"verdict": "FAIL", "classification": "EXECUTION_ERROR",
                        "detail": traceback.format_exc(limit=3)[-500:],
                        "expected_result": None, "actual_result": None,
                        "receipt": None})
        records.append(rec)
        print(f"  {rec['verdict']:4} {rec['case_id']:32} "
              f"{rec['surface']:26} {(rec.get('classification') or '')}")
        if rec["verdict"] == "FAIL":
            print(f"        {str(rec['detail'])[:200]}")

    def jsonable(v):
        if isinstance(v, dict):
            return {(" | ".join(map(str, k)) if isinstance(k, tuple) else str(k)):
                    jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [jsonable(x) for x in v]
        if hasattr(v, "__dict__") and not isinstance(v, (str, int, float, bool)):
            return {k: jsonable(x) for k, x in vars(v).items()
                    if not k.startswith("_")}
        return v

    for r in records:
        r["expected_result"] = jsonable(r["expected_result"])
        r["actual_result"] = jsonable(r["actual_result"])
        r["receipt"] = jsonable(r["receipt"])

    out = {"base_sha": "ea8c65b592ae5d2203d924bf66afa051618641e6",
           "test_sha": "2b00172c0b033c97de8afe5fa5b60ba43e1443ec",
           "bank_size": len(records),
           "pass": sum(1 for r in records if r["verdict"] == "PASS"),
           "fail": sum(1 for r in records if r["verdict"] == "FAIL"),
           "cases": records}
    Path(__file__).with_name("plan_execution_bank_result.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\nBANK {out['pass']}/{out['bank_size']} pass, {out['fail']} fail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
