#!/usr/bin/env python3
"""The scope bank: can a governed plan's population actually be honoured?

Baseline v2 could only prove the four AnalyticalScope dimensions NEGATIVELY — it
showed a stated period was ignored, never that one was honoured, because its book
had no period, no lens and no dataset identity. These cases are the positive
fixtures, built on `scope_oracle`, whose figures are guaranteed materially
distinct so a scope-blind engine cannot pass by coincidence.

Every case runs at every phase and records the same fields, so the blast-radius
audit is a per-case before/after diff rather than an aggregate pass count.

    QueryPlan built in code -> compile_query_plan -> executor -> receipt
                            -> scope_oracle truth

No parser, no ParsedQuestion, no question_interpretation, no Opus, no
interpretation_v2, no deployed API.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BANK_ROOT") or HERE.parents[2])
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import scope_oracle as so  # noqa: E402

from mi_agent.mi_query_validator import load_mi_semantics  # noqa: E402
from mi_agent.query_plan import (  # noqa: E402
    COUNT, SUM, AnalyticalScope, PlannedOutput, Predicate, QueryPlan)
from mi_agent.query_plan_compiler import compile_query_plan  # noqa: E402
from mi_agent.query_plan_execution import execute_query_plan  # noqa: E402

SEMANTICS = load_mi_semantics(str(ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
BOOK = so.scoped_book()
DATASETS = so.dataset_pair()
BALANCE = so.BALANCE
ABSENT = "<ABSENT>"

JOINT = ("borrower_type", "eq", "Joint")

CASES = []


def case(case_id, dimension, *, requested, expect, truth, note=""):
    """`expect` is what a correct engine must do. `truth` names the oracle figure."""
    def wrap(fn):
        CASES.append({"case_id": case_id, "dimension": dimension, "fn": fn,
                      "requested_semantics": requested, "expectation": expect,
                      "truth_source": truth, "note": note})
        return fn
    return wrap


def pick(md, *names):
    for n in names:
        if isinstance(md, dict) and n in md and md[n] is not None:
            return n, md[n]
    return ABSENT, None


def plan(outputs, *, period=None, lens=None, dataset="funded", predicates=(),
         dimensions=()):
    return QueryPlan(
        shared_scope=AnalyticalScope(
            dataset=dataset, portfolio_lens=lens, period=period,
            filters=tuple(Predicate(*p) for p in predicates),
            dimensions=tuple(dimensions)),
        outputs=tuple(outputs))


def sum_balance(**kw):
    return plan([PlannedOutput(output_id="a", operation=SUM, measure=BALANCE)], **kw)


def run(p, data=None, cells_of=None):
    """Execute and return a flat, comparable observation of what happened.

    `cells_of` reads a grouped result's cells, because a grouped execution
    carries no scalar value and comparing one would measure nothing.
    """
    frame = BOOK if data is None else data
    try:
        spec = compile_query_plan(p)[0].spec
        spec_view = {f: getattr(spec, f, ABSENT) for f in
                     ("reporting_date", "as_of_date", "temporal_mode",
                      "execution_mode", "portfolio_lens", "compare_periods")}
        spec_view["dataset"] = getattr(spec, "dataset", ABSENT)
        spec_view["filters"] = len(getattr(spec, "filters", ()) or ())
    except Exception as exc:                                     # noqa: BLE001
        spec_view = {"compile_error": f"{type(exc).__name__}: {exc}"[:200]}
    try:
        env = execute_query_plan(p, frame, SEMANTICS, validate=False)
    except Exception as exc:                                     # noqa: BLE001
        return {"executed": False, "refused": True,
                "refusal": f"{type(exc).__name__}: {exc}"[:240],
                "value": None, "spec": spec_view, "warnings": [],
                "requested_period_receipt": ABSENT, "resolved_period_receipt": ABSENT,
                "lens_receipt": ABSENT, "dataset_receipt": ABSENT,
                "rows_in": None, "rows_out": None, "predicates": None}
    out = env.outputs[0]
    ref = out.execution_ref
    md = (getattr(ref, "metadata", None) or {}) if ref is not None else {}
    reqf, reqv = pick(md, "requested_period", "period_requested", "requested_scope")
    resf, resv = pick(md, "resolved_period", "period", "reporting_date",
                      "as_of_date", "snapshot_id", "period_executed")
    lensf, lensv = pick(md, "portfolio_lens", "lens", "scope_lens")
    dsf, dsv = pick(md, "dataset", "source", "dataset_label")
    prf, prv = pick(md, "applied_predicates", "applied_filter_fields")
    grouped_cells = None
    if cells_of is not None and getattr(ref, "data", None) is not None:
        frame_out = ref.data
        keys = list(md.get("group_field_keys") or ())
        col = next((c for c in (cells_of, f"{cells_of}_sum") if c in frame_out.columns),
                   None)
        grouped_cells = ({" | ".join(str(row[k]) for k in keys): float(row[col])
                          for _, row in frame_out.iterrows()} if col and keys
                         else {"<NO COLUMN>": list(frame_out.columns)})
    return {"executed": ref is not None and getattr(ref, "data", None) is not None,
            "refused": False, "refusal": "", "cells": grouped_cells,
            "value": None if out.value is None else float(out.value),
            "spec": spec_view,
            "warnings": [str(w)[:160] for w in (getattr(ref, "warnings", ()) or ())],
            "requested_period_receipt": f"{reqf}={reqv!r}",
            "resolved_period_receipt": f"{resf}={resv!r}",
            "lens_receipt": f"{lensf}={lensv!r}",
            "dataset_receipt": f"{dsf}={dsv!r}",
            "rows_in": md.get("input_row_count"),
            "rows_out": md.get("filtered_row_count"),
            "predicates": prv}


def near(a, b, places=2):
    return a is not None and b is not None and abs(float(a) - float(b)) < 10 ** -places


# =============================== PERIOD ==================================== #

@case("P01-current-period", "PERIOD",
      requested=f"period={so.CURRENT}", expect="the current-period total",
      truth="scope_oracle.total(period=CURRENT)")
def p01():
    obs = run(sum_balance(period=so.CURRENT))
    return {"obs": obs, "expected": so.total(BOOK, period=so.CURRENT),
            "mode": "value"}


@case("P02-historical-A", "PERIOD",
      requested=f"period={so.HISTORICAL_A}", expect="the March total",
      truth="scope_oracle.total(period=HISTORICAL_A)")
def p02():
    obs = run(sum_balance(period=so.HISTORICAL_A))
    return {"obs": obs, "expected": so.total(BOOK, period=so.HISTORICAL_A),
            "mode": "value"}


@case("P03-historical-B-previous", "PERIOD",
      requested=f"period={so.PREVIOUS}", expect="the May total",
      truth="scope_oracle.total(period=PREVIOUS)")
def p03():
    obs = run(sum_balance(period=so.PREVIOUS))
    return {"obs": obs, "expected": so.total(BOOK, period=so.PREVIOUS),
            "mode": "value"}


@case("P04-A-and-B-differ", "PERIOD",
      requested="two different historical periods",
      expect="two different figures, matching their own truths",
      truth="scope_oracle — the two totals are 16.5m apart by construction",
      note="the single strongest proof that a period is honoured at all")
def p04():
    a = run(sum_balance(period=so.HISTORICAL_A))
    b = run(sum_balance(period=so.PREVIOUS))
    return {"obs": {"A": a, "B": b}, "mode": "two_periods",
            "expected": {"A": so.total(BOOK, period=so.HISTORICAL_A),
                         "B": so.total(BOOK, period=so.PREVIOUS)}}


@case("P05-unavailable-period", "PERIOD",
      requested=f"period={so.UNAVAILABLE_PERIOD} (plausible, not in the book)",
      expect="a governed refusal or an explicit unavailability disclosure",
      truth="scope_oracle — the book carries no such period")
def p05():
    obs = run(sum_balance(period=so.UNAVAILABLE_PERIOD))
    return {"obs": obs, "mode": "refuse_or_disclose",
            "whole_book": so.total(BOOK),
            "current": so.total(BOOK, period=so.CURRENT)}


@case("P06-impossible-period", "PERIOD",
      requested=f"period={so.IMPOSSIBLE_PERIOD} (not a calendar date)",
      expect="a governed refusal",
      truth="scope_oracle — February has no 30th")
def p06():
    obs = run(sum_balance(period=so.IMPOSSIBLE_PERIOD))
    return {"obs": obs, "mode": "refuse_or_disclose",
            "whole_book": so.total(BOOK),
            "current": so.total(BOOK, period=so.CURRENT)}


@case("P07-future-period", "PERIOD",
      requested=f"period={so.FUTURE_PERIOD}",
      expect="a governed refusal",
      truth="scope_oracle — after every observation")
def p07():
    obs = run(sum_balance(period=so.FUTURE_PERIOD))
    return {"obs": obs, "mode": "refuse_or_disclose",
            "whole_book": so.total(BOOK),
            "current": so.total(BOOK, period=so.CURRENT)}


@case("P08-no-period-stated", "PERIOD",
      requested="period=None on a three-period book",
      expect="either one period, or an explicit disclosure that three were spanned",
      truth="scope_oracle — spanning three periods triple-counts the book",
      note="the default is a semantic decision; silence about it is not")
def p08():
    obs = run(sum_balance(period=None))
    return {"obs": obs, "mode": "no_period",
            "whole_book": so.total(BOOK),
            "current": so.total(BOOK, period=so.CURRENT)}


@case("P09-receipt-proves-requested-and-resolved", "PERIOD",
      requested=f"period={so.HISTORICAL_A}",
      expect="the receipt evidences BOTH the requested and the resolved period",
      truth="the plan object versus the receipt")
def p09():
    obs = run(sum_balance(period=so.HISTORICAL_A))
    return {"obs": obs, "mode": "period_receipt", "requested": so.HISTORICAL_A}


@case("P10-period-with-grouping", "PERIOD",
      requested=f"period={so.HISTORICAL_A} grouped by region",
      expect="March cells only",
      truth="scope_oracle.grouped(period=HISTORICAL_A)")
def p10():
    p = plan([PlannedOutput(output_id="a", operation=SUM, measure=BALANCE)],
             period=so.HISTORICAL_A, dimensions=[so.REGION])
    obs = run(p, cells_of=BALANCE)
    return {"obs": obs, "mode": "grouped_cells",
            "expected": {" | ".join(k): v for k, v in
                         so.grouped(BOOK, [so.REGION],
                                    period=so.HISTORICAL_A).items()}}


# =============================== DATASET =================================== #

@case("DS01-funded", "DATASET",
      requested="dataset=funded, funded frame supplied",
      expect="the funded total", truth="scope_oracle.total(funded)")
def ds01():
    # No period and no lens: this case must measure the DATASET dimension alone,
    # or a period defect would be reported as a dataset defect.
    obs = run(sum_balance(dataset="funded"), data=DATASETS["funded"])
    return {"obs": obs, "mode": "value",
            "expected": so.total(DATASETS["funded"])}


@case("DS02-pipeline", "DATASET",
      requested="dataset=pipeline, pipeline frame supplied",
      expect="the pipeline total, 50m clear of every funded figure",
      truth="scope_oracle.total(pipeline)")
def ds02():
    obs = run(sum_balance(dataset="pipeline"), data=DATASETS["pipeline"])
    return {"obs": obs, "mode": "value",
            "expected": so.total(DATASETS["pipeline"]),
            "note_accidental": "the pipeline frame is single-period and "
                               "single-lens, so period/lens scoping is a no-op on "
                               "it; this case is about dataset identity only"}


@case("DS03-dataset-mismatch", "DATASET",
      requested="dataset=pipeline while the FUNDED frame is supplied",
      expect="a governed refusal or a disclosed mismatch — never a silent answer",
      truth="scope_oracle — the frame is funded, the plan says pipeline",
      note="tests whether the declared dataset is verified against what arrived")
def ds03():
    obs = run(sum_balance(dataset="pipeline"), data=DATASETS["funded"])
    return {"obs": obs, "mode": "refuse_or_disclose",
            "whole_book": so.total(DATASETS["funded"]),
            "current": so.total(DATASETS["funded"], period=so.CURRENT)}


@case("DS04-receipt-names-the-dataset", "DATASET",
      requested="dataset=funded", expect="the receipt names the dataset executed",
      truth="the plan object versus the receipt")
def ds04():
    obs = run(sum_balance(dataset="funded"))
    return {"obs": obs, "mode": "dataset_receipt", "requested": "funded"}


# ============================ PORTFOLIO LENS =============================== #

@case("L01-no-lens-is-total-funded", "LENS",
      requested="lens=None", expect="the whole current book, both lenses",
      truth="scope_oracle.total(period=CURRENT)",
      note="the default must not narrow")
def l01():
    obs = run(sum_balance(period=so.CURRENT, lens=None))
    return {"obs": obs, "mode": "value",
            "expected": so.total(BOOK, period=so.CURRENT)}


@case("L02-direct", "LENS",
      requested="lens=direct", expect="the Direct population only",
      truth="scope_oracle.total(period=CURRENT, lens=Direct)")
def l02():
    obs = run(sum_balance(period=so.CURRENT, lens="direct"))
    return {"obs": obs, "mode": "value",
            "expected": so.total(BOOK, period=so.CURRENT, lens=so.DIRECT)}


@case("L03-acquired", "LENS",
      requested="lens=acquired", expect="the Acquired population only",
      truth="scope_oracle.total(period=CURRENT, lens=Acquired)")
def l03():
    obs = run(sum_balance(period=so.CURRENT, lens="acquired"))
    return {"obs": obs, "mode": "value",
            "expected": so.total(BOOK, period=so.CURRENT, lens=so.ACQUIRED)}


@case("L04-invalid-lens", "LENS",
      requested="lens=nonsense", expect="a governed refusal",
      truth="scope_oracle — no such population exists")
def l04():
    obs = run(sum_balance(period=so.CURRENT, lens="nonsense_lens"))
    return {"obs": obs, "mode": "refuse_or_disclose",
            "whole_book": so.total(BOOK),
            "current": so.total(BOOK, period=so.CURRENT)}


@case("L05-receipt-names-the-lens", "LENS",
      requested="lens=direct", expect="the receipt names the lens applied",
      truth="the plan object versus the receipt")
def l05():
    obs = run(sum_balance(period=so.CURRENT, lens="direct"))
    return {"obs": obs, "mode": "lens_receipt", "requested": "direct"}


@case("L06-direct-and-acquired-sum-to-total", "LENS",
      requested="the two lenses separately",
      expect="Direct + Acquired == the unlensed total",
      truth="scope_oracle — the two lenses partition the book")
def l06():
    d = run(sum_balance(period=so.CURRENT, lens="direct"))
    a = run(sum_balance(period=so.CURRENT, lens="acquired"))
    return {"obs": {"direct": d, "acquired": a}, "mode": "lens_partition",
            "expected": {"direct": so.total(BOOK, period=so.CURRENT, lens=so.DIRECT),
                         "acquired": so.total(BOOK, period=so.CURRENT, lens=so.ACQUIRED),
                         "total": so.total(BOOK, period=so.CURRENT)}}


# ============================= COMPOSITION ================================= #

def _composition(case_id, requested, *, period=None, lens=None, dataset="funded",
                 predicates=(), data_key=None, expected_kw=None):
    @case(case_id, "COMPOSITION", requested=requested,
          expect="every stated constraint applied together",
          truth="scope_oracle with the same constraints",
          note="adding one scope constraint must never drop another")
    def fn():
        frame = DATASETS[data_key] if data_key else BOOK
        obs = run(sum_balance(period=period, lens=lens, dataset=dataset,
                              predicates=predicates), data=frame)
        kw = dict(expected_kw or {})
        return {"obs": obs, "mode": "value",
                "expected": so.total(frame, **kw)}
    return fn


_composition("X01-period-plus-filter",
             f"period={so.CURRENT} + borrower_type=Joint",
             period=so.CURRENT, predicates=[JOINT],
             expected_kw={"period": so.CURRENT, "predicates": [JOINT]})
_composition("X02-dataset-plus-filter",
             "dataset=pipeline + borrower_type=Joint",
             dataset="pipeline", period=so.CURRENT, predicates=[JOINT],
             data_key="pipeline",
             expected_kw={"period": so.CURRENT, "predicates": [JOINT]})
_composition("X03-lens-plus-filter",
             "lens=direct + borrower_type=Joint",
             period=so.CURRENT, lens="direct", predicates=[JOINT],
             expected_kw={"period": so.CURRENT, "lens": so.DIRECT,
                          "predicates": [JOINT]})
_composition("X04-period-plus-dataset",
             f"dataset=pipeline + period={so.CURRENT}",
             dataset="pipeline", period=so.CURRENT, data_key="pipeline",
             expected_kw={"period": so.CURRENT})
_composition("X05-period-plus-lens",
             f"period={so.HISTORICAL_A} + lens=acquired",
             period=so.HISTORICAL_A, lens="acquired",
             expected_kw={"period": so.HISTORICAL_A, "lens": so.ACQUIRED})
_composition("X06-dataset-plus-lens",
             "dataset=funded + lens=direct",
             dataset="funded", period=so.CURRENT, lens="direct",
             expected_kw={"period": so.CURRENT, "lens": so.DIRECT})
_composition("X07-period-dataset-lens",
             f"dataset=funded + period={so.PREVIOUS} + lens=direct",
             dataset="funded", period=so.PREVIOUS, lens="direct",
             expected_kw={"period": so.PREVIOUS, "lens": so.DIRECT})
_composition("X08-period-dataset-lens-filter",
             f"dataset=funded + period={so.PREVIOUS} + lens=direct + Joint",
             dataset="funded", period=so.PREVIOUS, lens="direct",
             predicates=[JOINT],
             expected_kw={"period": so.PREVIOUS, "lens": so.DIRECT,
                          "predicates": [JOINT]})


# ================================ judging ================================== #

def judge(c, out):
    mode, obs = out["mode"], out["obs"]

    if mode == "value":
        o, want = obs, out["expected"]
        if o["refused"]:
            return "FAIL", "UNEXPECTED_REFUSAL", o["refusal"][:150], False
        if near(o["value"], want):
            return "PASS", None, f"{o['value']:.2f}", False
        return ("FAIL", "SCOPE_NOT_HONOURED",
                f"expected {want:.2f}, got {o['value']!r}"
                + (f" (the whole supplied frame is {out.get('whole_book', float('nan')):.2f})"
                   if "whole_book" in out else ""),
                not o["warnings"])

    if mode == "two_periods":
        a, b = obs["A"], obs["B"]
        wa, wb = out["expected"]["A"], out["expected"]["B"]
        if a["value"] is not None and b["value"] is not None \
                and near(a["value"], b["value"]):
            return ("FAIL", "SCOPE_NOT_HONOURED",
                    f"two different periods returned the identical figure "
                    f"{a['value']:.2f}; truth says {wa:.2f} and {wb:.2f}", True)
        if near(a["value"], wa) and near(b["value"], wb):
            return "PASS", None, f"A={a['value']:.2f} B={b['value']:.2f}", False
        return ("FAIL", "SCOPE_NOT_HONOURED",
                f"A got {a['value']!r} want {wa:.2f}; B got {b['value']!r} "
                f"want {wb:.2f}", True)

    if mode == "refuse_or_disclose":
        o = obs
        if o["refused"]:
            return "PASS", None, f"refused: {o['refusal'][:120]}", False
        disclosed = [w for w in o["warnings"]
                     if any(t in w.lower() for t in
                            ("unavailable", "not available", "no such", "ignored",
                             "fallback", "cannot", "unsupported", "mismatch",
                             "not honoured", "spans"))]
        if disclosed:
            return "PASS", None, f"disclosed: {disclosed[0][:120]}", False
        if near(o["value"], out.get("whole_book")) or near(o["value"], out.get("current")):
            return ("FAIL", "SILENT_FALLBACK",
                    f"answered {o['value']:.2f} — the unscoped figure — with no "
                    f"refusal and no warning", True)
        return ("FAIL", "SILENT_FALLBACK",
                f"executed with no refusal and no disclosure: {o['value']!r}", True)

    if mode == "no_period":
        o = obs
        if o["refused"]:
            return "PASS", None, f"refused: {o['refusal'][:110]}", False
        spans = [w for w in o["warnings"]
                 if any(t in w.lower() for t in ("period", "span", "snapshot"))]
        if near(o["value"], out["current"]):
            return "PASS", None, "resolved to the latest period", False
        if near(o["value"], out["whole_book"]) and spans:
            return "PASS", None, f"spanned and disclosed: {spans[0][:110]}", False
        if near(o["value"], out["whole_book"]):
            return ("FAIL", "SILENT_MULTI_PERIOD",
                    f"summed all three periods to {o['value']:.2f} with no "
                    f"disclosure that the figure is not one period", True)
        return "FAIL", "SCOPE_NOT_HONOURED", f"unexpected {o['value']!r}", True

    if mode == "period_receipt":
        o = obs
        if o["refused"]:
            return "FAIL", "UNEXPECTED_REFUSAL", o["refusal"][:120], False
        missing = [n for n, v in (("requested", o["requested_period_receipt"]),
                                  ("resolved", o["resolved_period_receipt"]))
                   if v.startswith(ABSENT)]
        if missing:
            return ("FAIL", "RECEIPT_GOVERNANCE",
                    f"the receipt evidences no {'/'.join(missing)} period "
                    f"(requested {out['requested']})", True)
        return "PASS", None, (f"{o['requested_period_receipt']} / "
                              f"{o['resolved_period_receipt']}"), False

    if mode in ("dataset_receipt", "lens_receipt"):
        key = "dataset_receipt" if mode == "dataset_receipt" else "lens_receipt"
        o = obs
        if o["refused"]:
            return "FAIL", "UNEXPECTED_REFUSAL", o["refusal"][:120], False
        if o[key].startswith(ABSENT) or "=None" in o[key]:
            return ("FAIL", "RECEIPT_GOVERNANCE",
                    f"the receipt does not name the {mode.split('_')[0]} "
                    f"(requested {out['requested']!r}); receipt says {o[key]}", True)
        return "PASS", None, o[key], False

    if mode == "grouped_cells":
        o = obs
        if o["refused"]:
            return "FAIL", "UNEXPECTED_REFUSAL", o["refusal"][:120], False
        got, want = o.get("cells") or {}, out["expected"]
        if "<NO COLUMN>" in got:
            return "FAIL", "OTHER", f"no measure column: {got['<NO COLUMN>']}", False
        if set(got) != set(want):
            return ("FAIL", "SCOPE_NOT_HONOURED",
                    f"group keys differ: got {sorted(got)[:3]} want {sorted(want)[:3]}",
                    True)
        bad = [f"{k}: {got[k]:.2f} != {want[k]:.2f}" for k in want
               if not near(got[k], want[k])]
        if bad:
            return ("FAIL", "SCOPE_NOT_HONOURED",
                    f"the period was not applied to the cells: {bad[:2]}", True)
        return "PASS", None, f"{len(got)} cells match March truth", False

    if mode == "grouped_total":
        o = obs
        if o["refused"]:
            return "FAIL", "UNEXPECTED_REFUSAL", o["refusal"][:120], False
        # A grouped result has no scalar value; compare the sum of its cells.
        return ("FAIL", "NOT_COMPARED",
                "grouped scalar unavailable at this layer; case reserved", False) \
            if o["value"] is None else (
                ("PASS", None, f"{o['value']:.2f}", False)
                if near(o["value"], out["expected"])
                else ("FAIL", "SCOPE_NOT_HONOURED",
                      f"expected {out['expected']:.2f}, got {o['value']:.2f}", True))

    if mode == "lens_partition":
        d, a = obs["direct"], obs["acquired"]
        w = out["expected"]
        if d["refused"] or a["refused"]:
            return "FAIL", "UNEXPECTED_REFUSAL", "a lens refused", False
        if near(d["value"], a["value"]):
            return ("FAIL", "SCOPE_NOT_HONOURED",
                    f"both lenses returned {d['value']!r}", True)
        if not (near(d["value"], w["direct"]) and near(a["value"], w["acquired"])):
            return ("FAIL", "SCOPE_NOT_HONOURED",
                    f"direct {d['value']!r} want {w['direct']:.2f}; acquired "
                    f"{a['value']!r} want {w['acquired']:.2f}", True)
        if not near(d["value"] + a["value"], w["total"]):
            return ("FAIL", "SCOPE_NOT_HONOURED",
                    f"the lenses do not partition: {d['value']} + {a['value']} "
                    f"!= {w['total']}", False)
        return "PASS", None, "the two lenses partition the book", False

    return "FAIL", "OTHER", f"unjudgeable mode {mode!r}", False


def main():
    records = []
    for c in CASES:
        rec = {k: c[k] for k in ("case_id", "dimension", "requested_semantics",
                                 "expectation", "truth_source", "note")}
        try:
            out = c["fn"]()
            verdict, owner, detail, silent = judge(c, out)
            obs = out["obs"]
            first = obs if "executed" in obs else list(obs.values())[0]
            rec.update({
                "verdict": verdict, "failure_class": owner, "detail": detail,
                "silent": bool(silent),
                "expected_result": out.get("expected"),
                "actual_result": (obs.get("value") if "value" in obs
                                  else {k: v.get("value") for k, v in obs.items()}),
                "disposition": ("REFUSED" if first.get("refused") else
                                "EXECUTED" if first.get("executed") else "NO_RESULT"),
                "semantic_receipt": {
                    "requested_period": first.get("requested_period_receipt"),
                    "resolved_period": first.get("resolved_period_receipt"),
                    "lens": first.get("lens_receipt"),
                    "dataset": first.get("dataset_receipt"),
                    "rows_in": first.get("rows_in"),
                    "rows_out": first.get("rows_out"),
                },
                "compiled_spec_scope": first.get("spec"),
                "warnings": first.get("warnings"),
            })
        except Exception:                                        # noqa: BLE001
            rec.update({"verdict": "HARNESS_ERROR", "failure_class": "OTHER",
                        "detail": traceback.format_exc(limit=4)[-500:],
                        "silent": False, "expected_result": None,
                        "actual_result": None, "disposition": None,
                        "semantic_receipt": None, "compiled_spec_scope": None,
                        "warnings": None})
        records.append(rec)
        print(f"  {rec['verdict']:14} {rec['case_id']:34} {rec['dimension']:12}"
              f"{' SILENT' if rec['silent'] else ''}")
        if rec["verdict"] != "PASS":
            print(f"      {str(rec['detail'])[:190]}")

    tally = {}
    for r in records:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    out = {"base_sha": "ea8c65b592ae5d2203d924bf66afa051618641e6",
           "bank": "scope_bank", "bank_size": len(records), "tally": tally,
           "fixture_figures": so.DISTINCT_FIGURES,
           "silent": [r["case_id"] for r in records if r["silent"]],
           "cases": records}
    Path(os.environ.get("SCOPE_BANK_OUT")
         or HERE / "scope_bank_result.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\nSCOPE BANK {len(records)} cases: {tally}")
    print(f"SILENT: {out['silent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
