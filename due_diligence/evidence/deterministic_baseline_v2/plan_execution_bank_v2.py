#!/usr/bin/env python3
"""Plan-level deterministic execution bank, v2 — broader, and self-auditing.

    QueryPlan built in code
        -> compile_query_plan -> MIQuerySpec
        -> deterministic executor / capability
        -> execution receipt
        -> independent oracle or longhand recomputation

No English parser, no ParsedQuestion, no question_interpretation, no recogniser
cascade, no raw-question routing, no Opus, no interpretation_v2, no deployed API.

WHY IT AUDITS ITSELF. Reconnaissance produced three harness defects that each
changed a verdict: `MIQueryResult` has no `.ok` so a getattr default made every
execution look successful; a grouped result names its column `<measure>_sum`; and
one case compared the wrong object. So here:

  * `pick()` never defaults. Reading an absent field records ABSENT, which is
    what made the missing period visible rather than invisible.
  * every case declares an `assertion` descriptor — the object, the field and the
    truth source — and the runner captures the ACTUAL runtime type it read, so
    the audit is evidence rather than a claim.
  * a judgement that cannot be made is `UNJUDGEABLE`, never a pass.

Nothing is fixed. A failing case is recorded and the bank continues.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("BANK_ROOT")
            or Path(__file__).resolve().parents[3])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mi_agent import mi_geography as geo  # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics  # noqa: E402
from mi_agent.query_plan import (  # noqa: E402
    AVERAGE, COUNT, MAX, MEDIAN, MIN, SUM, WEIGHTED_AVERAGE, AnalyticalScope,
    PlannedOutput, Predicate, QueryPlan)
from mi_agent.query_plan_compiler import compile_query_plan  # noqa: E402
from mi_agent.query_plan_execution import execute_query_plan  # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth  # noqa: E402

SEMANTICS = load_mi_semantics(str(ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
BOOK = truth.canonical_book()
BALANCE, LTV, RATE, AGE = truth.BALANCE, truth.LTV, truth.RATE, truth.AGE

# --------------------------------------------------------------------------- #
# The CONFIGURED core geography, proved from the asset configuration rather than
# assumed. `config/asset/mi_geography.yaml` keys a primary basis by asset class;
# equity_release declares its own in `config/asset/product_defaults_ERM.yaml`.
# ITL3 sits in the `code` tier, never the reporting tier, so it is not the MI
# region route — recorded here so the bank cannot drift from the contract.
# --------------------------------------------------------------------------- #
ASSET_CLASS = "equity_release"
PRIMARY_BASIS = geo.default_primary_basis(ASSET_CLASS)
BASIS_FIELDS = geo.axis_fields(PRIMARY_BASIS)
REGION_FIELD = geo.field_for_basis(PRIMARY_BASIS, frame=BOOK)
GEO_PROVENANCE = {
    "asset_class": ASSET_CLASS,
    "config_owner": "config/asset/mi_geography.yaml :: primary_basis_by_asset_class"
                    " (+ config/asset/product_defaults_ERM.yaml for this class)",
    "resolver": "mi_agent.mi_geography.default_primary_basis -> axis_fields"
                " -> field_for_basis",
    "primary_basis": PRIMARY_BASIS,
    "basis_fields_in_readability_order": list(BASIS_FIELDS),
    "region_field_used_by_this_bank": REGION_FIELD,
    "itl3_tier": [list(t) for t in geo.tiers_for_basis(PRIMARY_BASIS)],
}

# A book carrying nulls, for null-handling and missing-region cases.
NULLBOOK = BOOK.copy()
NULLBOOK.loc[NULLBOOK.index[:40], LTV] = np.nan
NULLBOOK.loc[NULLBOOK.index[40:70], REGION_FIELD] = None

CASES = []
ABSENT = "<ABSENT>"


def pick(mapping, *names):
    """First present key and its value, or (ABSENT, None). Never defaults.

    A default here is how an absent field becomes an invisible one, which is
    exactly the harness defect that made every execution look successful during
    reconnaissance.
    """
    for n in names:
        if isinstance(mapping, dict) and n in mapping and mapping[n] is not None:
            return n, mapping[n]
    return ABSENT, None


def case(case_id, surface, *, assertion, disposition="RESULT", note=""):
    def wrap(fn):
        CASES.append({"case_id": case_id, "surface": surface, "fn": fn,
                      "assertion": assertion, "expected_disposition": disposition,
                      "note": note})
        return fn
    return wrap


def plan_for(outputs, *, predicates=(), dimensions=(), dataset="funded",
             period="2026-06-30", lens=None):
    scope = AnalyticalScope(dataset=dataset, period=period,
                           filters=tuple(Predicate(*p) for p in predicates),
                           dimensions=tuple(dimensions))
    if lens is not None:
        scope = AnalyticalScope(dataset=dataset, period=period,
                               portfolio_lens=lens,
                               filters=tuple(Predicate(*p) for p in predicates),
                               dimensions=tuple(dimensions))
    return QueryPlan(shared_scope=scope, outputs=tuple(outputs))


def execute(plan, data=None):
    """Run a plan; an engine exception is a DISPOSITION, not a harness failure."""
    frame = BOOK if data is None else data
    try:
        spec = compile_query_plan(plan)[0].spec
    except Exception as exc:                                     # noqa: BLE001
        spec = None
        spec_err = f"{type(exc).__name__}: {exc}"
    else:
        spec_err = ""
    try:
        env = execute_query_plan(plan, frame, SEMANTICS, validate=False)
    except Exception as exc:                                     # noqa: BLE001
        return None, {"raised": type(exc).__name__, "reason": str(exc)[:300],
                      "complete": False, "spec_error": spec_err,
                      "spec_period_fields": _spec_period(spec),
                      "outputs": [{"executed": False, "value": None,
                                   "error": str(exc)[:300], "warnings": [],
                                   "group_keys": [], "predicates_field": ABSENT,
                                   "predicates": None, "period_field": ABSENT,
                                   "period": None, "dataset": None,
                                   "rows_in": None, "rows_out": None,
                                   "aggregation": None, "rows": 0, "columns": []}]}
    receipt = {"complete": bool(env.completeness.complete),
               "reason": "" if env.completeness.complete else env.completeness.reason(),
               "spec_error": spec_err,
               "spec_period_fields": _spec_period(spec),
               "outputs": []}
    for o in env.outputs:
        ref = o.execution_ref
        md = (getattr(ref, "metadata", None) or {}) if ref is not None else {}
        pf, pv = pick(md, "period", "reporting_date", "as_of_date",
                      "period_executed", "snapshot_id", "temporal_mode")
        prf, prv = pick(md, "applied_predicates", "applied_filter_fields",
                        "filters_applied")
        receipt["outputs"].append({
            "output_id": o.output_id,
            "executed": ref is not None and getattr(ref, "data", None) is not None,
            "value": None if o.value is None else float(o.value),
            "error": getattr(ref, "error", "") or "",
            "warnings": list(getattr(ref, "warnings", ()) or ()),
            "result_type": getattr(ref, "result_type", None),
            "group_keys": list(md.get("group_field_keys") or ()),
            "predicates_field": prf, "predicates": prv,
            "period_field": pf, "period": pv,
            "dataset": md.get("dataset"),
            "rows_in": md.get("input_row_count"),
            "rows_out": md.get("filtered_row_count"),
            "aggregation": md.get("aggregation"),
            "rows": int(len(ref.data)) if getattr(ref, "data", None) is not None else 0,
            "columns": list(ref.data.columns) if getattr(ref, "data", None) is not None else [],
        })
    return env, receipt


def _spec_period(spec):
    """Which temporal fields the compiled spec actually carries."""
    if spec is None:
        return {"spec": "<not compiled>"}
    out = {}
    for f in ("reporting_date", "as_of_date", "temporal_mode", "execution_mode",
              "compare_periods"):
        out[f] = getattr(spec, f, ABSENT)
    return out


def scalar(env):
    return None if env is None else (
        None if env.outputs[0].value is None else float(env.outputs[0].value))


def cells(env, column, output_id="a"):
    if env is None:
        return {}
    ref = next(o for o in env.outputs if o.output_id == output_id).execution_ref
    if ref is None or getattr(ref, "data", None) is None:
        return {}
    frame, md = ref.data, (ref.metadata or {})
    keys = list(md.get("group_field_keys") or ())
    col = column if column in frame.columns else None
    for suffix in ("_sum", "_avg", "_weighted_avg", "_count", "_median",
                   "_min", "_max"):
        if col is None and f"{column}{suffix}" in frame.columns:
            col = f"{column}{suffix}"
    if col is None:
        return {"<NO SUCH COLUMN>": list(frame.columns)}
    return {tuple(str(row[k]) for k in keys): float(row[col])
            for _, row in frame.iterrows()}


def num(x, y, places=2):
    if x is None or y is None:
        return False
    return abs(float(x) - float(y)) < 10 ** -places


# ======================= A. AGGREGATION / STATISTICS ======================== #

A_ORACLE = {"object": "MultiResultEnvelope.outputs[0].value",
            "field": "value", "truth": "portfolio_truth_oracle"}


@case("A01-count", "AGGREGATION_STATISTIC", assertion=A_ORACLE)
def a01():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=COUNT)]))
    return {"actual": scalar(env), "expected": float(truth.row_count(BOOK)),
            "receipt": rc}


@case("A02-sum", "AGGREGATION_STATISTIC", assertion=A_ORACLE)
def a02():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)]))
    return {"actual": scalar(env), "expected": float(truth.total(BOOK, BALANCE)),
            "receipt": rc}


@case("A03-simple-mean", "AGGREGATION_STATISTIC", assertion=A_ORACLE,
      note="an unweighted mean must not silently weight")
def a03():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=AVERAGE,
                                              measure=AGE)]))
    return {"actual": scalar(env), "expected": float(BOOK[AGE].mean()),
            "receipt": rc}


@case("A04-weighted-mean-ltv", "AGGREGATION_STATISTIC", assertion=A_ORACLE)
def a04():
    env, rc = execute(plan_for([PlannedOutput(
        output_id="a", operation=WEIGHTED_AVERAGE, measure=LTV,
        weight_field=BALANCE)]))
    return {"actual": scalar(env),
            "expected": float(truth.weighted_average(BOOK, LTV, BALANCE)),
            "receipt": rc}


@case("A05-weighted-mean-rate", "AGGREGATION_STATISTIC", assertion=A_ORACLE)
def a05():
    env, rc = execute(plan_for([PlannedOutput(
        output_id="a", operation=WEIGHTED_AVERAGE, measure=RATE,
        weight_field=BALANCE)]))
    return {"actual": scalar(env),
            "expected": float(truth.weighted_average(BOOK, RATE, BALANCE)),
            "receipt": rc}


@case("A06-weighted-differs-from-simple", "AGGREGATION_STATISTIC",
      assertion={"object": "two envelopes", "field": "value",
                 "truth": "longhand — the two means must not coincide"},
      note="guards against a weighted mean that silently ignores its weight")
def a06():
    w, _ = execute(plan_for([PlannedOutput(output_id="a",
                                           operation=WEIGHTED_AVERAGE,
                                           measure=LTV, weight_field=BALANCE)]))
    s, rc = execute(plan_for([PlannedOutput(output_id="a", operation=AVERAGE,
                                            measure=LTV)]))
    return {"actual": {"weighted": scalar(w), "simple": scalar(s)},
            "expected": {"weighted": float(truth.weighted_average(BOOK, LTV, BALANCE)),
                         "simple": float(BOOK[LTV].mean())},
            "receipt": rc, "kind": "pair"}


@case("A07-median", "AGGREGATION_STATISTIC", assertion=A_ORACLE)
def a07():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=MEDIAN,
                                              measure=BALANCE)]))
    return {"actual": scalar(env), "expected": float(BOOK[BALANCE].median()),
            "receipt": rc}


@case("A08-min", "AGGREGATION_STATISTIC", assertion=A_ORACLE)
def a08():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=MIN,
                                              measure=BALANCE)]))
    return {"actual": scalar(env), "expected": float(BOOK[BALANCE].min()),
            "receipt": rc}


@case("A09-max", "AGGREGATION_STATISTIC", assertion=A_ORACLE)
def a09():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=MAX,
                                              measure=BALANCE)]))
    return {"actual": scalar(env), "expected": float(BOOK[BALANCE].max()),
            "receipt": rc}


@case("A10-grouped-sum", "AGGREGATION_STATISTIC",
      assertion={"object": "MIQueryResult.data", "field": "<measure>_sum per group",
                 "truth": "portfolio_truth_oracle.grouped"}, )
def a10():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=["erm_product_type"]))
    return {"actual": cells(env, BALANCE),
            "expected": dict(truth.grouped(BOOK, ["erm_product_type"],
                                           column=BALANCE, how="sum")),
            "receipt": rc, "kind": "cells"}


@case("A11-grouped-count", "AGGREGATION_STATISTIC",
      assertion={"object": "MIQueryResult.data", "field": "loan_count per group",
                 "truth": "portfolio_truth_oracle.grouped(how=count)"})
def a11():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=COUNT)],
                               dimensions=["broker_channel"]))
    return {"actual": cells(env, "loan_count"),
            "expected": dict(truth.grouped(BOOK, ["broker_channel"], how="count")),
            "receipt": rc, "kind": "cells"}


@case("A12-grouped-weighted-mean", "AGGREGATION_STATISTIC",
      assertion={"object": "MIQueryResult.data",
                 "field": "<measure>_weighted_avg per group",
                 "truth": "portfolio_truth_oracle.grouped(how=weighted_avg)"})
def a12():
    env, rc = execute(plan_for([PlannedOutput(
        output_id="a", operation=WEIGHTED_AVERAGE, measure=LTV,
        weight_field=BALANCE)], dimensions=["erm_product_type"]))
    return {"actual": cells(env, LTV),
            "expected": dict(truth.grouped(BOOK, ["erm_product_type"],
                                           column=LTV, how="weighted_avg")),
            "receipt": rc, "kind": "cells"}


@case("A13-two-dimension-grid", "AGGREGATION_STATISTIC",
      assertion={"object": "MIQueryResult.data", "field": "cells of a 2-D grid",
                 "truth": "portfolio_truth_oracle.grouped on two axes"},
      note="a grid whose totals agree can still have the mass in the wrong cells")
def a13():
    dims = ["erm_product_type", REGION_FIELD]
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)], dimensions=dims))
    return {"actual": cells(env, BALANCE),
            "expected": dict(truth.grouped(BOOK, dims, column=BALANCE, how="sum")),
            "receipt": rc, "kind": "cells"}


@case("A14-grid-reconciles-to-total", "AGGREGATION_STATISTIC",
      assertion={"object": "MIQueryResult.data", "field": "sum of all cells",
                 "truth": "portfolio_truth_oracle.total"})
def a14():
    dims = ["erm_product_type", REGION_FIELD]
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)], dimensions=dims))
    got = cells(env, BALANCE)
    return {"actual": sum(got.values()) if got else None,
            "expected": float(truth.total(BOOK, BALANCE)), "receipt": rc}


@case("A15-share-column-sums-to-100", "AGGREGATION_STATISTIC",
      assertion={"object": "MIQueryResult.data", "field": "concentration_pct",
                 "truth": "longhand — shares of one population sum to 100"},
      note="the executor volunteers a share column; it must be a real share")
def a15():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=["erm_product_type"]))
    pct = cells(env, "concentration_pct")
    return {"actual": sum(pct.values()) if pct and "<NO SUCH COLUMN>" not in pct else None,
            "expected": 100.0, "receipt": rc, "places": 1}


@case("A16-share-matches-longhand", "AGGREGATION_STATISTIC",
      assertion={"object": "MIQueryResult.data", "field": "concentration_pct",
                 "truth": "longhand group total / population total * 100"})
def a16():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=["erm_product_type"]))
    pct = cells(env, "concentration_pct")
    tot = float(truth.total(BOOK, BALANCE))
    want = {k: v / tot * 100.0 for k, v in
            truth.grouped(BOOK, ["erm_product_type"], column=BALANCE,
                          how="sum").items()}
    return {"actual": pct, "expected": want, "receipt": rc, "kind": "cells",
            "places": 3}


@case("A17-null-measure-excluded-from-mean", "AGGREGATION_STATISTIC",
      assertion={"object": "MultiResultEnvelope.outputs[0].value", "field": "value",
                 "truth": "longhand pandas mean skipping nulls on NULLBOOK"},
      note="40 of 400 LTVs are null; a mean must not treat them as zero")
def a17():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=AVERAGE,
                                              measure=LTV)]), data=NULLBOOK)
    return {"actual": scalar(env), "expected": float(NULLBOOK[LTV].mean()),
            "receipt": rc, "places": 4,
            "alternative_if_zero_filled": float(NULLBOOK[LTV].fillna(0).mean())}


@case("A18-null-measure-excluded-from-sum", "AGGREGATION_STATISTIC",
      assertion={"object": "MultiResultEnvelope.outputs[0].value", "field": "value",
                 "truth": "longhand pandas sum skipping nulls on NULLBOOK"})
def a18():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=LTV)]), data=NULLBOOK)
    return {"actual": scalar(env), "expected": float(NULLBOOK[LTV].sum()),
            "receipt": rc, "places": 4}


# ========================= B. POPULATION / FILTER ========================== #

JOINT = ("borrower_type", "eq", "Joint")
SCOT = (REGION_FIELD, "eq", "Scotland")
LUMP = ("erm_product_type", "eq", "Lump Sum")


@case("B01-one-filter", "POPULATION_FILTER", assertion=A_ORACLE)
def b01():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=[JOINT]))
    return {"actual": scalar(env),
            "expected": float(truth.total(BOOK, BALANCE, [JOINT])), "receipt": rc}


@case("B02-two-filters-conjunctive", "POPULATION_FILTER", assertion=A_ORACLE,
      note="two filters must narrow together, not replace one another")
def b02():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=[JOINT, SCOT]))
    return {"actual": scalar(env),
            "expected": float(truth.total(BOOK, BALANCE, [JOINT, SCOT])),
            "receipt": rc}


@case("B03-three-filters", "POPULATION_FILTER", assertion=A_ORACLE)
def b03():
    preds = [JOINT, SCOT, LUMP]
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=preds))
    return {"actual": scalar(env),
            "expected": float(truth.total(BOOK, BALANCE, preds)), "receipt": rc}


@case("B04-filter-strictly-narrows", "POPULATION_FILTER",
      assertion={"object": "two envelopes", "field": "value",
                 "truth": "longhand — a filtered total must be < the unfiltered one"})
def b04():
    f, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                            measure=BALANCE)], predicates=[JOINT]))
    u, _ = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                           measure=BALANCE)]))
    return {"actual": {"filtered": scalar(f), "unfiltered": scalar(u)},
            "expected": "filtered < unfiltered", "receipt": rc, "kind": "narrows"}


@case("B05-filter-true-for-every-row", "POPULATION_FILTER", assertion=A_ORACLE,
      note="applied AND disclosed: a no-op predicate is still a predicate")
def b05():
    pred = (LTV, "gt", 0.0)
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=[pred]))
    return {"actual": scalar(env),
            "expected": float(truth.total(BOOK, BALANCE, [pred])),
            "receipt": rc, "require_predicate_receipt": True}


@case("B06-filter-matching-one-row", "POPULATION_FILTER", assertion=A_ORACLE)
def b06():
    top = float(BOOK[BALANCE].max())
    pred = (BALANCE, "ge", top)
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=COUNT)],
                               predicates=[pred]))
    return {"actual": scalar(env),
            "expected": float(truth.row_count(BOOK, [pred])), "receipt": rc}


@case("B07-filter-matching-zero-rows", "POPULATION_FILTER",
      assertion={"object": "receipt.outputs[0]",
                 "field": "value / warnings / rows_out",
                 "truth": "longhand — the population is empty by construction"},
      disposition="GOVERNED_UNAVAILABLE",
      note="zero matching population is NOT an economically observed zero")
def b07():
    pred = (LTV, "gt", 1000.0)
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=[pred]))
    return {"actual": rc["outputs"][0], "receipt": rc,
            "expected_rows": int(truth.row_count(BOOK, [pred]))}


@case("B08-zero-population-count", "POPULATION_FILTER",
      assertion={"object": "receipt.outputs[0]", "field": "value / warnings",
                 "truth": "longhand — an empty COUNT is genuinely 0"},
      note="the one case where 0 IS the answer; it must still say the population was empty")
def b08():
    pred = (LTV, "gt", 1000.0)
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=COUNT)],
                               predicates=[pred]))
    return {"actual": scalar(env), "expected": 0.0, "receipt": rc,
            "require_empty_disclosure": True}


@case("B09-filtered-population-with-null-measure", "POPULATION_FILTER",
      assertion={"object": "MultiResultEnvelope.outputs[0].value", "field": "value",
                 "truth": "longhand mean over the filtered NULLBOOK rows"})
def b09():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=AVERAGE,
                                              measure=LTV)],
                               predicates=[JOINT]), data=NULLBOOK)
    mask = NULLBOOK["borrower_type"] == "Joint"
    return {"actual": scalar(env), "expected": float(NULLBOOK.loc[mask, LTV].mean()),
            "receipt": rc, "places": 4}


@case("B10-population-preserved-in-receipt", "RECEIPT_GOVERNANCE",
      assertion={"object": "receipt.outputs[0]",
                 "field": "predicates / rows_in / rows_out",
                 "truth": "longhand row counts from the oracle"},
      disposition="RECEIPT_EVIDENCES_POPULATION")
def b10():
    preds = [JOINT, SCOT]
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=preds))
    return {"actual": rc["outputs"][0], "receipt": rc,
            "expected_rows_in": int(truth.row_count(BOOK)),
            "expected_rows_out": int(truth.row_count(BOOK, preds))}


@case("B11-filter-on-absent-column", "POPULATION_FILTER",
      assertion={"object": "receipt", "field": "raised / error",
                 "truth": "longhand — the column does not exist"},
      disposition="GOVERNED_REFUSAL",
      note="a predicate on a column the book lacks must refuse, never be dropped")
def b11():
    thin = BOOK.drop(columns=["borrower_type"])
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=[JOINT]), data=thin)
    return {"actual": rc["outputs"][0], "receipt": rc,
            "unfiltered_total": float(truth.total(BOOK, BALANCE))}


@case("B12-portfolio-lens-reaches-the-spec", "PLAN_BINDING",
      assertion={"object": "compile_query_plan(plan)[0].spec",
                 "field": "portfolio_lens",
                 "truth": "the plan states a lens; the spec must carry it"},
      disposition="PLAN_BINDING_CARRIES_SCOPE",
      note="the third scope field, after period and dataset")
def b12():
    plan = plan_for([PlannedOutput(output_id="a", operation=SUM, measure=BALANCE)],
                    lens="direct")
    spec = compile_query_plan(plan)[0].spec
    return {"actual": {"spec_portfolio_lens": getattr(spec, "portfolio_lens", ABSENT)},
            "expected": "direct", "receipt": None, "kind": "scope_field",
            "requested": plan.shared_scope.portfolio_lens, "field": "portfolio_lens"}


@case("B13-which-scope-fields-survive-compilation", "PLAN_BINDING",
      assertion={"object": "AnalyticalScope vs compiled MIQuerySpec",
                 "field": "dataset / portfolio_lens / period / filters",
                 "truth": "a governed plan's population must survive into the spec"},
      disposition="PLAN_BINDING_CARRIES_SCOPE",
      note="the consolidated root-cause probe for the whole scope contract")
def b13():
    plan = plan_for([PlannedOutput(output_id="a", operation=SUM, measure=BALANCE)],
                    predicates=[JOINT], period="2026-03-31", lens="direct")
    spec = compile_query_plan(plan)[0].spec
    survived = {
        "filters": bool(getattr(spec, "filters", None)),
        "dataset": getattr(spec, "dataset", ABSENT) not in (ABSENT, None, ""),
        "portfolio_lens": getattr(spec, "portfolio_lens", ABSENT) not in (ABSENT, None, ""),
        "period": any(getattr(spec, f, None) for f in
                      ("reporting_date", "as_of_date", "temporal_mode")),
    }
    return {"actual": survived,
            "expected": {k: True for k in survived}, "receipt": None,
            "kind": "scope_survival",
            "asked": {"dataset": plan.shared_scope.dataset,
                      "portfolio_lens": plan.shared_scope.portfolio_lens,
                      "period": plan.shared_scope.period,
                      "filters": len(plan.shared_scope.filters)}}


# ============================ C. CORE GEOGRAPHY ============================ #

@case("C01-region-grouping-configured-field", "CORE_GEOGRAPHY",
      assertion={"object": "MIQueryResult.data",
                 "field": f"groups of the configured region field",
                 "truth": "portfolio_truth_oracle.grouped on that same field"},
      note="the field comes from the asset config, not from a guess")
def c01():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=[REGION_FIELD]))
    return {"actual": cells(env, BALANCE),
            "expected": dict(truth.grouped(BOOK, [REGION_FIELD], column=BALANCE,
                                           how="sum")),
            "receipt": rc, "kind": "cells"}


@case("C02-region-filter", "CORE_GEOGRAPHY", assertion=A_ORACLE)
def c02():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=[SCOT]))
    return {"actual": scalar(env),
            "expected": float(truth.total(BOOK, BALANCE, [SCOT])), "receipt": rc}


@case("C03-region-cells-reconcile", "CORE_GEOGRAPHY",
      assertion={"object": "MIQueryResult.data", "field": "sum of region cells",
                 "truth": "portfolio_truth_oracle.total"})
def c03():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=[REGION_FIELD]))
    got = cells(env, BALANCE)
    return {"actual": sum(got.values()) if got else None,
            "expected": float(truth.total(BOOK, BALANCE)), "receipt": rc}


@case("C04-region-count-by-region", "CORE_GEOGRAPHY",
      assertion={"object": "MIQueryResult.data", "field": "loan_count per region",
                 "truth": "portfolio_truth_oracle.grouped(how=count)"})
def c04():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=COUNT)],
                               dimensions=[REGION_FIELD]))
    return {"actual": cells(env, "loan_count"),
            "expected": dict(truth.grouped(BOOK, [REGION_FIELD], how="count")),
            "receipt": rc, "kind": "cells"}


@case("C05-null-region-not-silently-dropped", "CORE_GEOGRAPHY",
      assertion={"object": "MIQueryResult.data + receipt",
                 "field": "region cells vs population total",
                 "truth": "longhand — 30 of 400 rows carry no region"},
      note="rows with no region must be disclosed, not quietly excluded")
def c05():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=[REGION_FIELD]), data=NULLBOOK)
    got = cells(env, BALANCE)
    return {"actual": {"cells_total": sum(got.values()) if got else None,
                       "labels": sorted(k[0] for k in got) if got else []},
            "expected": {"population_total": float(NULLBOOK[BALANCE].sum()),
                         "null_rows": int(NULLBOOK[REGION_FIELD].isna().sum()),
                         "null_balance": float(
                             NULLBOOK.loc[NULLBOOK[REGION_FIELD].isna(), BALANCE].sum())},
            "receipt": rc, "kind": "null_region"}


@case("C06-receipt-names-the-geography-binding", "RECEIPT_GOVERNANCE",
      assertion={"object": "receipt.outputs[0]", "field": "group_keys",
                 "truth": "the configured field name from the asset config"},
      disposition="RECEIPT_EVIDENCES_GEOGRAPHY")
def c06():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=[REGION_FIELD]))
    return {"actual": rc["outputs"][0]["group_keys"], "expected": [REGION_FIELD],
            "receipt": rc}


@case("C07-itl3-is-a-secondary-lens", "CORE_GEOGRAPHY",
      assertion={"object": "mi_geography tiers", "field": "tiers_for_basis",
                 "truth": "config/asset/mi_geography.yaml + region_basis levels"},
      disposition="OUTSIDE_CURRENT_SCOPE",
      note="recorded, not scored: ITL3 is a Funded/React postcode lens, not the MI route")
def c07():
    tiers = {t: list(f) for t, f in geo.tiers_for_basis(PRIMARY_BASIS)}
    itl3_fields = [f for fs in tiers.values() for f in fs if f.endswith("_itl3")]
    return {"actual": {"tiers": tiers, "itl3_fields": itl3_fields,
                       "region_field_chosen_for_mi": REGION_FIELD},
            "expected": "ITL3 present only in the non-reporting tier; MI uses the "
                        "reporting tier", "receipt": None, "kind": "itl3_contract"}


# ========================= D. TEMPORAL RESOLUTION ========================== #

@case("D01-current-period", "TEMPORAL_RESOLUTION", assertion=A_ORACLE,
      note="the flat book is the current period; this must be right")
def d01():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               period="2026-06-30"))
    return {"actual": scalar(env), "expected": float(truth.total(BOOK, BALANCE)),
            "receipt": rc}


def _temporal_case(case_id, period, note):
    @case(case_id, "TEMPORAL_RESOLUTION",
          assertion={"object": "receipt + compiled MIQuerySpec",
                     "field": "spec temporal fields / receipt period",
                     "truth": "longhand — the book carries no period column"},
          disposition="GOVERNED_REFUSAL_OR_DISCLOSED", note=note)
    def fn(period=period):
        env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                                  measure=BALANCE)],
                                   period=period))
        return {"actual": rc["outputs"][0], "receipt": rc,
                "current_total": float(truth.total(BOOK, BALANCE)),
                "requested_period": period}
    return fn


_temporal_case("D02-explicit-historical-period", "2026-03-31",
               "a valid-looking prior reporting period")
_temporal_case("D03-previous-reporting-period", "2026-05-31",
               "the immediately preceding period")
_temporal_case("D04-nonexistent-date", "2026-02-30",
               "a date that does not exist in any calendar")
_temporal_case("D05-materially-distant-period", "1999-12-31",
               "an observation no book of this kind could carry")
_temporal_case("D06-future-period", "2099-12-31",
               "a period after every observation")


@case("D07-period-reaches-the-spec", "PLAN_BINDING",
      assertion={"object": "compile_query_plan(plan)[0].spec",
                 "field": "reporting_date / as_of_date / temporal_mode",
                 "truth": "the plan states a period; the spec must carry it"},
      disposition="PLAN_BINDING_CARRIES_PERIOD",
      note="the root-cause probe: does the plan->spec compiler bind the period at all?")
def d07():
    plan = plan_for([PlannedOutput(output_id="a", operation=SUM, measure=BALANCE)],
                    period="2026-03-31")
    spec = compile_query_plan(plan)[0].spec
    return {"actual": _spec_period(spec), "expected": "2026-03-31 in some field",
            "receipt": None, "kind": "spec_period",
            "requested_period": plan.shared_scope.period}


@case("D08-two-periods-differ", "TEMPORAL_RESOLUTION",
      assertion={"object": "two envelopes", "field": "value",
                 "truth": "longhand — different periods must not be identical by default"},
      note="if two different stated periods give the same figure, neither was honoured")
def d08():
    a, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                            measure=BALANCE)], period="2026-06-30"))
    b, _ = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                           measure=BALANCE)], period="1999-12-31"))
    return {"actual": {"current": scalar(a), "ancient": scalar(b)},
            "expected": "a stated period is honoured or refused, not ignored",
            "receipt": rc, "kind": "periods_differ"}


# ======================== E. RECEIPT / GOVERNANCE ========================== #

@case("E01-receipt-dataset-identity", "RECEIPT_GOVERNANCE",
      assertion={"object": "receipt.outputs[0]", "field": "dataset",
                 "truth": "the plan states dataset='funded'"},
      disposition="RECEIPT_FIELD_PRESENT")
def e01():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)]))
    return {"actual": rc["outputs"][0]["dataset"], "expected": "funded",
            "receipt": rc, "kind": "present"}


@case("E02-receipt-aggregation", "RECEIPT_GOVERNANCE",
      assertion={"object": "receipt.outputs[0]", "field": "aggregation",
                 "truth": "the plan states operation=sum"},
      disposition="RECEIPT_FIELD_PRESENT")
def e02():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)]))
    return {"actual": rc["outputs"][0]["aggregation"], "expected": "sum",
            "receipt": rc, "kind": "present"}


@case("E03-receipt-row-counts", "RECEIPT_GOVERNANCE",
      assertion={"object": "receipt.outputs[0]", "field": "rows_in / rows_out",
                 "truth": "oracle row counts before and after the filter"},
      disposition="RECEIPT_FIELD_PRESENT")
def e03():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=[JOINT]))
    o = rc["outputs"][0]
    return {"actual": {"rows_in": o["rows_in"], "rows_out": o["rows_out"]},
            "expected": {"rows_in": int(truth.row_count(BOOK)),
                         "rows_out": int(truth.row_count(BOOK, [JOINT]))},
            "receipt": rc, "kind": "rowcounts"}


@case("E04-receipt-predicates", "RECEIPT_GOVERNANCE",
      assertion={"object": "receipt.outputs[0]", "field": "predicates",
                 "truth": "the plan states two predicates"},
      disposition="RECEIPT_FIELD_PRESENT")
def e04():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               predicates=[JOINT, SCOT]))
    o = rc["outputs"][0]
    return {"actual": {"field": o["predicates_field"], "value": o["predicates"]},
            "expected": "two predicates named", "receipt": rc,
            "kind": "predicates", "expected_count": 2}


@case("E05-receipt-grouping", "RECEIPT_GOVERNANCE",
      assertion={"object": "receipt.outputs[0]", "field": "group_keys",
                 "truth": "the plan states two dimensions"},
      disposition="RECEIPT_FIELD_PRESENT")
def e05():
    dims = ["erm_product_type", REGION_FIELD]
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)], dimensions=dims))
    return {"actual": rc["outputs"][0]["group_keys"], "expected": dims,
            "receipt": rc, "kind": "exact_list"}


@case("E06-receipt-temporal-request", "RECEIPT_GOVERNANCE",
      assertion={"object": "receipt.outputs[0]", "field": "period (any spelling)",
                 "truth": "the plan states a period; the receipt must evidence it"},
      disposition="RECEIPT_FIELD_PRESENT",
      note="pick() records ABSENT rather than defaulting, so a missing field shows")
def e06():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               period="2026-06-30"))
    o = rc["outputs"][0]
    return {"actual": {"field": o["period_field"], "value": o["period"]},
            "expected": "2026-06-30 under some key", "receipt": rc,
            "kind": "period_receipt"}


@case("E07-receipt-measure", "RECEIPT_GOVERNANCE",
      assertion={"object": "MIQueryResult.data columns", "field": "measure column",
                 "truth": "the plan states the balance measure"},
      disposition="RECEIPT_FIELD_PRESENT")
def e07():
    env, rc = execute(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                              measure=BALANCE)],
                               dimensions=["erm_product_type"]))
    cols = rc["outputs"][0]["columns"]
    return {"actual": cols,
            "expected": f"a column naming {BALANCE}", "receipt": rc,
            "kind": "measure_named", "measure": BALANCE}


# ======================= F. SPECIALIST CAPABILITIES ======================== #

def _snap(d, sid, date):
    from mi_agent.period_change.models import SnapshotFrame
    return SnapshotFrame(snapshot_id=sid, reporting_date=date,
                         frame=pd.DataFrame({"loan_identifier": list(d),
                                             BALANCE: [float(v) for v in d.values()]}))


def _bridge(start, end):
    from mi_agent.period_change.bridge import balance_bridge
    return balance_bridge(_snap(start, "s0", "2026-05-31"),
                          _snap(end, "s1", "2026-06-30"))


def _bridge_truth(start, end):
    s, e = set(start), set(end)
    return {"continuing": round(sum(end[k] - start[k] for k in s & e), 6),
            "new": round(sum(end[k] for k in e - s), 6),
            "redeemed": round(-sum(start[k] for k in s - e), 6),
            "net": round(sum(end.values()) - sum(start.values()), 6)}


BRIDGE_ASSERT = {"object": "BalanceBridge",
                 "field": "continuing_movement / new_loan_balance / "
                          "exited_loan_balance / opening / closing",
                 "truth": "longhand set decomposition of two loan-level maps"}


@case("F01-bridge-mixed-churn", "SPECIALIST_CAPABILITY", assertion=BRIDGE_ASSERT)
def f01():
    s = {"L1": 100.0, "L2": 200.0, "L3": 300.0}
    e = {"L1": 90.0, "L2": 250.0, "L4": 400.0}
    return {"actual": _bridge(s, e), "expected": _bridge_truth(s, e),
            "receipt": None, "kind": "bridge"}


@case("F02-bridge-full-redemption", "SPECIALIST_CAPABILITY", assertion=BRIDGE_ASSERT)
def f02():
    s, e = {"L1": 100.0, "L2": 50.0}, {"L3": 10.0}
    return {"actual": _bridge(s, e), "expected": _bridge_truth(s, e),
            "receipt": None, "kind": "bridge"}


@case("F03-bridge-no-change", "SPECIALIST_CAPABILITY", assertion=BRIDGE_ASSERT,
      note="identical snapshots: zero must read as zero, not as a movement")
def f03():
    s = {"L1": 100.0, "L2": 50.0}
    return {"actual": _bridge(s, dict(s)), "expected": _bridge_truth(s, dict(s)),
            "receipt": None, "kind": "bridge"}


@case("F04-bridge-all-new", "SPECIALIST_CAPABILITY", assertion=BRIDGE_ASSERT)
def f04():
    s, e = {}, {"L1": 10.0, "L2": 20.0}
    return {"actual": _bridge(s, e), "expected": _bridge_truth(s, e),
            "receipt": None, "kind": "bridge"}


@case("F05-bridge-reconciles", "SPECIALIST_CAPABILITY",
      assertion={"object": "BalanceBridge", "field": "reconciles / residual",
                 "truth": "the capability's own invariant, checked independently"})
def f05():
    s = {f"L{i}": float(i * 10) for i in range(1, 25)}
    e = {f"L{i}": float(i * 11) for i in range(3, 30)}
    b = _bridge(s, e)
    longhand = _bridge_truth(s, e)
    recomposed = (float(b.opening_balance) + float(b.continuing_movement)
                  + float(b.new_loan_balance) - float(b.exited_loan_balance))
    return {"actual": {"reconciles": bool(b.reconciles),
                       "residual": float(b.residual),
                       "recomposed_closing": recomposed,
                       "closing": float(b.closing_balance)},
            "expected": {"net": longhand["net"],
                         "closing": float(sum(e.values()))},
            "receipt": None, "kind": "bridge_reconcile"}


@case("F06-borrowing-base-no-facility", "SPECIALIST_CAPABILITY",
      assertion={"object": "borrowing_base.calculate", "field": "raised / result",
                 "truth": "no facility is configured, so no headroom exists"},
      disposition="GOVERNED_REFUSAL",
      note="must never invent a facility or a zero headroom")
def f06():
    from mi_agent.borrowing_base import calculator
    try:
        out = calculator.calculate(BOOK, None)
        return {"actual": {"returned": type(out).__name__,
                           "repr": str(out)[:200]},
                "expected": "a refusal, not a result", "receipt": None,
                "kind": "no_facility"}
    except Exception as exc:                                     # noqa: BLE001
        return {"actual": {"raised": type(exc).__name__, "message": str(exc)[:200]},
                "expected": "a refusal", "receipt": None, "kind": "no_facility"}


@case("F07-borrowing-base-configured", "SPECIALIST_CAPABILITY",
      assertion={"object": "BorrowingBaseResult",
                 "field": "eligible / advance / headroom",
                 "truth": "longhand: min(commitment, eligible x advance_rate) - drawn"},
      note="a governed facility constructed from the dataclass contract")
def f07():
    from mi_agent.borrowing_base import calculator
    from mi_agent.borrowing_base.models import FacilityConfiguration
    fac = FacilityConfiguration(
        client_id="BANKTEST", facility_id="F1", facility_label="Test Facility",
        commitment=50_000_000.0, advance_rate=0.80,
        current_drawn_amount=10_000_000.0, current_drawn_amount_as_of="2026-06-30",
        effective_date="2026-01-01", maturity_date="2030-01-01",
        prototype_assume_financing_portfolio_eligible=True)
    try:
        res = calculator.calculate(BOOK, fac, strict=False)
    except Exception as exc:                                     # noqa: BLE001
        return {"actual": {"raised": type(exc).__name__, "message": str(exc)[:250]},
                "expected": "a result or a governed refusal", "receipt": None,
                "kind": "bb_configured"}
    total = float(truth.total(BOOK, BALANCE))
    return {"actual": {"type": type(res).__name__,
                       "fields": {k: (float(v) if isinstance(v, (int, float))
                                      and not isinstance(v, bool) else str(v)[:60])
                                  for k, v in vars(res).items()
                                  if not k.startswith("_")}},
            "expected": {"whole_book_balance": total,
                         "advance_on_whole_book": total * 0.80,
                         "commitment": 50_000_000.0,
                         "drawn": 10_000_000.0},
            "receipt": None, "kind": "bb_configured"}


def _reachability(case_id, capability, probe):
    @case(case_id, "SPECIALIST_CAPABILITY",
          assertion={"object": "QueryPlan / PlannedOutput",
                     "field": "operation vocabulary",
                     "truth": "inspection of the governed-plan contract"},
          disposition="REACHABILITY_PROBE",
          note=f"is {capability} expressible in a QueryPlan at all?")
    def fn(capability=capability, probe=probe):
        return {"actual": probe(), "expected": "reachable or not",
                "receipt": None, "kind": "reachability", "capability": capability}
    return fn


def _plan_operations():
    from mi_agent import query_plan as qp
    return sorted({v for k, v in vars(qp).items()
                   if k.isupper() and isinstance(v, str) and len(v) < 20})


_reachability("F08-ranking-reachable", "ranking",
              lambda: {"plan_operations": _plan_operations(),
                       "rank_in_vocabulary": "rank" in _plan_operations()})
_reachability("F09-distribution-reachable", "distribution",
              lambda: {"plan_operations": _plan_operations(),
                       "distribution_in_vocabulary":
                           "distribution" in _plan_operations()})
_reachability("F10-portfolio-summary-reachable", "portfolio_summary",
              lambda: {"plan_operations": _plan_operations(),
                       "summary_in_vocabulary": "summary" in _plan_operations()})
_reachability("F11-movement-reachable-from-plan", "funded movement / bridge",
              lambda: {"plan_operations": _plan_operations(),
                       "movement_in_vocabulary": "movement" in _plan_operations(),
                       "bridge_entry_point":
                           "mi_agent.period_change.bridge.balance_bridge "
                           "(direct capability call, not a QueryPlan operation)"})
_reachability("F12-concentration-reachable", "concentration",
              lambda: {"plan_operations": _plan_operations(),
                       "concentration_in_vocabulary":
                           "concentration" in _plan_operations(),
                       "share_column_emitted_by_grouped_execution":
                           "concentration_pct"})


# =============================== judging =================================== #

def judge(c, out):
    cid, kind = c["case_id"], out.get("kind")
    rc = out.get("receipt")
    o0 = (rc or {}).get("outputs", [{}])[0] if rc else {}
    places = out.get("places", 2)

    if kind == "cells":
        got, want = out["actual"], out["expected"]
        if "<NO SUCH COLUMN>" in got:
            return "FAIL", "PLAN_BINDING", f"no measure column; saw {got['<NO SUCH COLUMN>']}", False
        if set(got) != set(want):
            return ("FAIL", "PLAN_BINDING",
                    f"group keys differ: got {sorted(got)[:3]} want {sorted(want)[:3]}", True)
        for k, v in want.items():
            if not num(got[k], v, places if places != 2 else 3):
                return "FAIL", "CALCULATION", f"cell {k}: {got[k]} != {v}", False
        return "PASS", None, "", False

    if kind == "pair":
        g, w = out["actual"], out["expected"]
        if not num(g["weighted"], w["weighted"], 4):
            return "FAIL", "AGGREGATION_STATISTIC", f"weighted {g} vs {w}", False
        if not num(g["simple"], w["simple"], 4):
            return "FAIL", "AGGREGATION_STATISTIC", f"simple {g} vs {w}", False
        if num(g["weighted"], g["simple"], 6):
            return ("FAIL", "AGGREGATION_STATISTIC",
                    "weighted and simple means are identical — the weight was ignored", True)
        return "PASS", None, f"weighted={g['weighted']:.4f} simple={g['simple']:.4f}", False

    if kind == "narrows":
        g = out["actual"]
        if g["filtered"] is None or g["unfiltered"] is None:
            return "FAIL", "POPULATION_FILTER", f"missing value: {g}", False
        if not g["filtered"] < g["unfiltered"]:
            return ("FAIL", "POPULATION_FILTER",
                    f"filter did not narrow: {g}", True)
        return "PASS", None, f"{g['filtered']:.0f} < {g['unfiltered']:.0f}", False

    if kind == "rowcounts":
        g, w = out["actual"], out["expected"]
        miss = [k for k in w if g.get(k) is None]
        if miss:
            return "FAIL", "RECEIPT_GOVERNANCE", f"receipt lacks {miss}", True
        bad = [f"{k}: {g[k]} != {w[k]}" for k in w if int(g[k]) != int(w[k])]
        if bad:
            return "FAIL", "RECEIPT_GOVERNANCE", "; ".join(bad), True
        return "PASS", None, f"{g}", False

    if kind == "predicates":
        g = out["actual"]
        if g["field"] == ABSENT or not g["value"]:
            return "FAIL", "RECEIPT_GOVERNANCE", "no predicates in the receipt", True
        n = len(g["value"]) if isinstance(g["value"], (list, tuple)) else 1
        if n < out["expected_count"]:
            return ("FAIL", "RECEIPT_GOVERNANCE",
                    f"receipt records {n} of {out['expected_count']} predicates", True)
        return "PASS", None, f"{g['field']} carries {n}", False

    if kind == "exact_list":
        if list(out["actual"]) != list(out["expected"]):
            return ("FAIL", "RECEIPT_GOVERNANCE",
                    f"got {out['actual']} want {out['expected']}", True)
        return "PASS", None, f"{out['actual']}", False

    if kind == "present":
        if out["actual"] in (None, ABSENT, ""):
            return "FAIL", "RECEIPT_GOVERNANCE", "field absent from the receipt", True
        if str(out["actual"]) != str(out["expected"]):
            return ("FAIL", "RECEIPT_GOVERNANCE",
                    f"got {out['actual']!r} want {out['expected']!r}", True)
        return "PASS", None, f"{out['actual']!r}", False

    if kind == "period_receipt":
        g = out["actual"]
        if g["field"] == ABSENT:
            return ("FAIL", "RECEIPT_GOVERNANCE",
                    "the receipt carries no temporal field under any spelling", True)
        return "PASS", None, f"{g['field']}={g['value']!r}", False

    if kind == "measure_named":
        cols = out["actual"]
        if not any(out["measure"] in str(c) for c in cols):
            return ("FAIL", "RECEIPT_GOVERNANCE",
                    f"no column names the measure; columns={cols}", True)
        return "PASS", None, f"{[c for c in cols if out['measure'] in str(c)]}", False

    if kind == "scope_field":
        g = out["actual"]
        val = list(g.values())[0]
        if val in (ABSENT, None, ""):
            return ("FAIL", "PLAN_BINDING",
                    f"the plan stated {out['field']}={out['requested']!r} and the "
                    f"compiled spec carries {val!r}", True)
        if str(val) != str(out["expected"]):
            return ("FAIL", "PLAN_BINDING",
                    f"spec carries {val!r}, plan asked {out['expected']!r}", True)
        return "PASS", None, f"{out['field']}={val!r}", False

    if kind == "scope_survival":
        g, asked = out["actual"], out["asked"]
        lost = sorted(k for k, ok in g.items() if not ok)
        if lost:
            return ("FAIL", "PLAN_BINDING",
                    f"of the four AnalyticalScope population facts, only "
                    f"{sorted(k for k, ok in g.items() if ok)} survive compilation; "
                    f"LOST {lost} (plan asked {asked})", True)
        return "PASS", None, "all scope fields survive", False

    if kind == "spec_period":
        g = out["actual"]
        carried = {k: v for k, v in g.items()
                   if v not in (None, ABSENT, [], "") }
        if not carried:
            return ("FAIL", "PLAN_BINDING",
                    f"the plan stated period {out['requested_period']!r} and the "
                    f"compiled spec carries no temporal value at all: {g}", True)
        return "PASS", None, f"carried {carried}", False

    if kind == "periods_differ":
        g = out["actual"]
        if g["current"] is not None and g["ancient"] is not None \
                and num(g["current"], g["ancient"], 2):
            return ("FAIL", "TEMPORAL_RESOLUTION",
                    f"two different stated periods returned the identical figure "
                    f"{g['current']} — neither was honoured", True)
        return "PASS", None, f"{g}", False

    if kind == "null_region":
        g, w = out["actual"], out["expected"]
        if g["cells_total"] is None:
            return "FAIL", "CORE_GEOGRAPHY", "no cells returned", False
        labelled_null = any(str(x).lower() in ("nan", "none", "unknown",
                                               "unresolved", "not reported", "<na>")
                            for x in g["labels"])
        # The executor may EXCLUDE unresolved grouping rows and say so. A
        # disclosed exclusion is governance working, not a silent loss — and
        # reading only the labels and the total would have missed the warning.
        disclosed = [w for w in (o0.get("warnings") or ())
                     if "missing/blank grouping" in str(w)
                     or "excluded" in str(w).lower()]
        if num(g["cells_total"], w["population_total"], 2):
            return "PASS", None, (
                f"all {w['population_total']:.0f} accounted for; "
                f"null rows labelled as {[x for x in g['labels'] if str(x).lower() in ('nan','none','unknown','unresolved','<na>')]}"), False
        missing = w["population_total"] - g["cells_total"]
        if labelled_null:
            return "PASS", None, f"nulls carried a label; gap {missing:.2f}", False
        if disclosed:
            return "PASS", None, (
                f"{w['null_rows']} unresolved rows worth {w['null_balance']:.2f} "
                f"excluded AND disclosed: {str(disclosed[0])[:120]}"), False
        if num(missing, w["null_balance"], 2):
            return ("FAIL", "CORE_GEOGRAPHY",
                    f"{w['null_rows']} rows with no region, worth "
                    f"{w['null_balance']:.2f}, dropped from the grid with no label "
                    f"and no warning", True)
        return ("FAIL", "CORE_GEOGRAPHY",
                f"cells total {g['cells_total']:.2f} != population "
                f"{w['population_total']:.2f}", True)

    if kind == "itl3_contract":
        g = out["actual"]
        reporting = dict(geo.tiers_for_basis(PRIMARY_BASIS)).get("reporting", ())
        if any(f.endswith("_itl3") for f in reporting):
            return ("FAIL", "CORE_GEOGRAPHY",
                    "ITL3 appears in the REPORTING tier — the repository "
                    "contradicts the stated product contract", True)
        if g["region_field_chosen_for_mi"] and \
                g["region_field_chosen_for_mi"].endswith("_itl3"):
            return ("FAIL", "CORE_GEOGRAPHY",
                    "MI resolved its region to an ITL3 field", True)
        return "OUTSIDE_CURRENT_SCOPE", None, (
            f"ITL3 fields {g['itl3_fields']} sit in the code tier only; MI uses "
            f"{g['region_field_chosen_for_mi']!r}"), False

    if kind == "bridge":
        got, want = out["actual"], out["expected"]
        checks = [("continuing", float(got.continuing_movement), want["continuing"]),
                  ("new", float(got.new_loan_balance), want["new"]),
                  ("redeemed", -float(got.exited_loan_balance), want["redeemed"]),
                  ("net", float(got.closing_balance) - float(got.opening_balance),
                   want["net"])]
        bad = [f"{n}: want {w}, got {g}" for n, g, w in checks if not num(g, w, 4)]
        if not bool(getattr(got, "reconciles", False)):
            bad.append(f"does not reconcile, residual={got.residual!r}")
        if bad:
            return "FAIL", "CALCULATION", "; ".join(bad), False
        return "PASS", None, (
            f"opening={got.opening_balance} closing={got.closing_balance} "
            f"continuing={got.continuing_movement} new={got.new_loan_balance} "
            f"exited={got.exited_loan_balance} residual={got.residual}"), False

    if kind == "bridge_reconcile":
        g, w = out["actual"], out["expected"]
        if not g["reconciles"]:
            return "FAIL", "CALCULATION", f"residual {g['residual']}", False
        if not num(g["closing"], w["closing"], 4):
            return "FAIL", "CALCULATION", f"closing {g['closing']} != {w['closing']}", False
        if not num(g["recomposed_closing"], g["closing"], 4):
            return ("FAIL", "CALCULATION",
                    f"components recompose to {g['recomposed_closing']} not "
                    f"{g['closing']}", False)
        return "PASS", None, f"residual={g['residual']}", False

    if kind == "no_facility":
        g = out["actual"]
        if "raised" in g:
            return "PASS", None, f"refused: {g['raised']}: {g['message'][:90]}", False
        return ("FAIL", "SPECIALIST_CAPABILITY",
                f"no facility configured yet a result was returned: {g}", True)

    if kind == "bb_configured":
        g = out["actual"]
        if "raised" in g:
            return ("UNJUDGEABLE", "DATA_CONFIG",
                    f"the capability needs configuration this bank cannot author "
                    f"honestly: {g['raised']}: {g['message'][:140]}", False)
        return ("UNJUDGEABLE", "DATA_CONFIG",
                f"result produced but no independent facility fixture exists to "
                f"verify eligibility against: {list(g['fields'])[:8]}", False)

    if kind == "reachability":
        g = out["actual"]
        present = [k for k, v in g.items() if k.endswith("_in_vocabulary") and v]
        if present:
            return "PASS", None, f"expressible: {present}", False
        return "NOT_PLAN_LEVEL_REACHABLE", None, (
            f"{out['capability']} is not an operation of the governed plan "
            f"contract; plan operations are {g['plan_operations']}"), False

    # disposition-style cases
    d = c["expected_disposition"]
    if d == "GOVERNED_UNAVAILABLE":
        if out["expected_rows"] != 0:
            return "FAIL", "OTHER", "fixture error: population is not empty", False
        warned = [w for w in (o0.get("warnings") or ())
                  if any(t in str(w).lower() for t in
                         ("empt", "no row", "no match", "unavailable", "0/"))]
        if o0.get("executed") and o0.get("value") == 0.0 and not warned:
            return ("FAIL", "RECEIPT_GOVERNANCE",
                    f"an empty population (rows_out={o0.get('rows_out')!r}) returned "
                    f"a confident 0.0 with no unavailability warning", True)
        return "PASS", None, f"value={o0.get('value')!r} warnings={warned}", False

    if d == "GOVERNED_REFUSAL":
        if cid == "B11-filter-on-absent-column":
            if o0.get("executed") and num(o0.get("value"), out["unfiltered_total"]):
                return ("FAIL", "POPULATION_FILTER",
                        "a predicate on an absent column was dropped and the whole "
                        "book answered instead", True)
            if o0.get("executed"):
                return ("FAIL", "POPULATION_FILTER",
                        f"executed despite an absent filter column: {o0.get('value')!r}", True)
            return "PASS", None, f"refused: {(rc or {}).get('raised')} {o0.get('error','')[:90]}", False
        return "PASS", None, "refused", False

    if d == "GOVERNED_REFUSAL_OR_DISCLOSED":
        if not o0.get("executed"):
            return "PASS", None, f"refused: {o0.get('error','')[:90]}", False
        if o0.get("period_field") != ABSENT:
            return "PASS", None, f"disclosed {o0['period_field']}={o0['period']!r}", False
        if num(o0.get("value"), out["current_total"]):
            return ("FAIL", "TEMPORAL_RESOLUTION",
                    f"period {out['requested_period']} was ignored: the whole-book "
                    f"current figure {out['current_total']:.2f} was returned with no "
                    f"period in the receipt and no warning", True)
        return ("FAIL", "TEMPORAL_RESOLUTION",
                f"executed with no temporal evidence: {o0.get('value')!r}", True)

    if d in ("RECEIPT_EVIDENCES_POPULATION",):
        miss = []
        if o0.get("predicates_field") == ABSENT:
            miss.append("predicates")
        if o0.get("rows_in") is None:
            miss.append("rows_in")
        if o0.get("rows_out") is None:
            miss.append("rows_out")
        if miss:
            return "FAIL", "RECEIPT_GOVERNANCE", f"receipt lacks {miss}", True
        if int(o0["rows_out"]) != out["expected_rows_out"]:
            return ("FAIL", "RECEIPT_GOVERNANCE",
                    f"rows_out {o0['rows_out']} != {out['expected_rows_out']}", True)
        return "PASS", None, f"rows {o0['rows_in']} -> {o0['rows_out']}", False

    if d == "RECEIPT_EVIDENCES_GEOGRAPHY":
        if list(out["actual"]) != list(out["expected"]):
            return ("FAIL", "RECEIPT_GOVERNANCE",
                    f"receipt names {out['actual']} not {out['expected']}", True)
        return "PASS", None, f"{out['actual']}", False

    # plain numeric
    if not num(out["actual"], out["expected"], places):
        return ("FAIL", "CALCULATION" if out.get("actual") is not None
                else "AGGREGATION_STATISTIC",
                f"expected {out['expected']!r}, got {out['actual']!r}"
                + (f" (zero-filled would be {out['alternative_if_zero_filled']!r})"
                   if "alternative_if_zero_filled" in out else ""), False)
    if out.get("require_predicate_receipt") and o0.get("predicates_field") == ABSENT:
        return ("FAIL", "RECEIPT_GOVERNANCE",
                "a predicate true of every row was applied but not recorded", True)
    if out.get("require_empty_disclosure"):
        warned = [w for w in (o0.get("warnings") or ())
                  if any(t in str(w).lower() for t in ("empt", "0/", "no row"))]
        if not warned and o0.get("rows_out") not in (0,):
            return ("FAIL", "RECEIPT_GOVERNANCE",
                    "a zero count over an empty population is not disclosed as empty", True)
    return "PASS", None, "", False


def main():
    records = []
    for c in CASES:
        rec = {"case_id": c["case_id"], "surface": c["surface"],
               "assertion": c["assertion"],
               "expected_disposition": c["expected_disposition"],
               "note": c["note"]}
        try:
            out = c["fn"]()
            verdict, owner, detail, silent = judge(c, out)
            rec.update({"verdict": verdict, "primary_owner": owner,
                        "detail": detail, "silent": bool(silent),
                        "actual_runtime_type": type(out.get("actual")).__name__,
                        "expected_result": out.get("expected"),
                        "actual_result": out.get("actual"),
                        "receipt": out.get("receipt")})
        except Exception:                                        # noqa: BLE001
            rec.update({"verdict": "HARNESS_ERROR", "primary_owner": "OTHER",
                        "detail": traceback.format_exc(limit=4)[-600:],
                        "silent": False, "actual_runtime_type": None,
                        "expected_result": None, "actual_result": None,
                        "receipt": None})
        records.append(rec)
        mark = {"PASS": "PASS", "FAIL": "FAIL"}.get(rec["verdict"], rec["verdict"])
        print(f"  {mark:26} {rec['case_id']:38} {rec['surface']:22}"
              f"{' SILENT' if rec['silent'] else ''}")
        if rec["verdict"] not in ("PASS",):
            print(f"       {str(rec['detail'])[:210]}")

    def jsonable(v):
        if isinstance(v, dict):
            return {(" | ".join(map(str, k)) if isinstance(k, tuple) else str(k)):
                    jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [jsonable(x) for x in v]
        if hasattr(v, "__dict__") and not isinstance(v, (str, int, float, bool)):
            return {k: jsonable(x) for k, x in vars(v).items()
                    if not k.startswith("_")}
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        return v

    for r in records:
        for k in ("expected_result", "actual_result", "receipt"):
            r[k] = jsonable(r[k])

    tally = {}
    for r in records:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    out = {"base_sha": "ea8c65b592ae5d2203d924bf66afa051618641e6",
           "test_sha": "2b00172c0b033c97de8afe5fa5b60ba43e1443ec",
           "geography_provenance": GEO_PROVENANCE,
           "bank_size": len(records), "tally": tally,
           "silent_semantic_errors": [r["case_id"] for r in records if r["silent"]],
           "cases": records}
    Path(os.environ.get("BANK_OUT")
         or Path(__file__).with_name("plan_execution_bank_v2_result.json")
         ).write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\nBANK {len(records)} cases: {tally}")
    print(f"SILENT SEMANTIC ERRORS: {out['silent_semantic_errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
