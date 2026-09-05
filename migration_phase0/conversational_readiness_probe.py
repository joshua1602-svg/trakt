#!/usr/bin/env python3
"""migration_phase0/conversational_readiness_probe.py

READ-ONLY characterisation harness for the conversational analytical-composition
readiness review (``MI_CONVERSATIONAL_READINESS.md``).

It changes nothing. It runs the REAL pipeline — ``run_mi_agent_query`` and
``execute_mi_query`` — over a deterministic in-memory tape and records four
things the review needs as evidence rather than as opinion:

  Phase B   SCOPE RECONSTRUCTION.  Take a successful answer, rebuild the
            population from ``spec`` + ``metadata`` ALONE (never from the
            question), execute a DIFFERENT measure over the rebuilt population,
            and check the row population is identical. This is the readiness
            criterion "can the executed population be reconstructed from the
            governed contracts".

  Phase E/F MULTI-OUTPUT COMPOSITION.  For each multi-output shape, run the
            combined request and each of its constituent atoms, and classify:
            ATOMIC_BLOCKED / COMPOSITION_FAILED / COMPOSITION_VERIFIED.
            Success requires ``metadata.measures_executed`` to name every
            requested output — prose mentioning several numbers is not evidence.

  Phase G   MULTI-TURN.  Run each dialogue's turns INDEPENDENTLY (which is what
            the stateless API does today) and record the population Q2 actually
            resolved to against the population an inheriting layer would owe.
            Classifies each turn's failure mode.

  Phase J-F NUMERIC REFERENCE SAFETY.  Check that "of the £38m" never becomes a
            numeric predicate while an explicit threshold still does.

    python migration_phase0/conversational_readiness_probe.py            # human
    python migration_phase0/conversational_readiness_probe.py --json     # machine

The tape is ``mi_agent.mi_query_harness.build_fixture`` — the estate's existing
deterministic funded tape — with one region relabelled ``London`` so the
review's own worked example is expressible. No production module is imported for
anything but reading.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SEMANTICS_PATH = str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


# --------------------------------------------------------------------------- #
# The tape
# --------------------------------------------------------------------------- #
def funded_tape(n: int = 400):
    """The estate's deterministic funded tape, with a ``London`` region.

    ``build_fixture`` materialises ``geographic_region_obligor`` over five NUTS3
    labels, none of which is London. The review's worked example is a London
    one, and a governed categorical value is resolved from THE BOOK'S OWN VALUES
    (``execution_receipt.book_values``), so relabelling one region is enough to
    make the example expressible without touching the registry or any parser.
    """
    from mi_agent.mi_query_harness import build_fixture

    df = build_fixture(n)
    df["geographic_region_obligor"] = df["geographic_region_obligor"].replace(
        {"North": "London"})
    return df


def pipeline_tape():
    """The governed pipeline frame, prepared by the ordinary production prep."""
    import pandas as pd

    from mi_agent_api.pipeline_prep import prepare_pipeline_mi_dataset

    src = (_REPO_ROOT / "tests" / "fixtures" / "pipeline_transition_2w"
           / "2026-06-12" / "M2L_KFI_and_Pipeline_2026_06_12.csv")
    prepared, _report = prepare_pipeline_mi_dataset(pd.read_csv(src),
                                                    as_of_date="2026-06-12")
    return prepared


def ask(question: str, frame, *, dataset: str = "funded") -> Dict[str, Any]:
    """One question through the REAL workflow. Deterministic parser, no network."""
    from mi_agent.mi_agent_workflow import run_mi_agent_query

    return run_mi_agent_query(question, frame, SEMANTICS_PATH,
                              parser_mode="deterministic", llm_enabled=False,
                              dataset=dataset)


def observed(result: Dict[str, Any]) -> Dict[str, Any]:
    """The machine-readable facts one answer publishes. Nothing derived."""
    qres = result.get("query_result")
    meta = (getattr(qres, "metadata", None) or {}) if qres is not None else {}
    receipt = result.get("execution_receipt") or {}
    spec = result.get("spec") or {}
    return {
        "ok": bool(result.get("ok")),
        "error": result.get("error"),
        "filters": spec.get("filters") or {},
        "metric": spec.get("metric"),
        "aggregation": spec.get("aggregation"),
        "dimension": spec.get("dimension"),
        "dimensions": list(spec.get("dimensions") or []),
        "measures": [m.get("field") for m in (spec.get("measures") or [])],
        "unavailable_filters": list(spec.get("unavailable_filters") or []),
        "measures_requested": meta.get("measures_requested"),
        "measures_executed": [m.get("field")
                              for m in (meta.get("measures_executed") or [])],
        "measures_unavailable": meta.get("measures_unavailable"),
        "applied_filter_fields": list(meta.get("applied_filter_fields") or []),
        "dataset": meta.get("dataset"),
        "population": receipt.get("population"),
        "population_total": receipt.get("populationTotal"),
        "narrowed": receipt.get("narrowed"),
        "receipt": receipt.get("receipt"),
        "guard": (result.get("semantic_guard") or {}).get("verdict"),
    }


# --------------------------------------------------------------------------- #
# PHASE B — scope reconstruction
# --------------------------------------------------------------------------- #
#: The canonical population fields a follow-up turn would have to inherit. Each
#: is named with the object that owns it TODAY, so the review's contract
#: proposal can reference existing owners rather than inventing a vocabulary.
SCOPE_SLOTS = (
    ("dataset",           "query_result.metadata['dataset']"),
    ("population_filters", "spec.filters"),
    ("grouping",          "spec.dimensions / spec.dimension"),
    ("measure",           "spec.metric / spec.measures"),
    ("aggregation",       "spec.aggregation"),
    ("weight_field",      "spec.weight_field"),
    ("reporting_period",  "spec.reporting_date / execution_receipt.period"),
    ("portfolio_lens",    "spec.portfolio_lens (request-supplied)"),
    ("comparison_basis",  "spec.comparison_basis"),
    ("applied_evidence",  "query_result.metadata['applied_filter_fields']"),
)


def reconstruct_scope(result: Dict[str, Any]) -> Dict[str, Any]:
    """The governed population an answer was calculated over, from the ANSWER.

    Reads only ``spec`` and ``query_result.metadata`` — never the question, and
    never the receipt's prose. Whether that is sufficient is the finding; this
    function exists so the sufficiency can be TESTED rather than asserted.
    """
    qres = result.get("query_result")
    meta = (getattr(qres, "metadata", None) or {}) if qres is not None else {}
    spec = result.get("spec") or {}
    receipt = result.get("execution_receipt") or {}
    return {
        "dataset": meta.get("dataset"),
        "population_filters": dict(spec.get("filters") or {}),
        "grouping": list(spec.get("dimensions") or []) or (
            [spec.get("dimension")] if spec.get("dimension") else []),
        "measure": spec.get("metric"),
        "measures": list(spec.get("measures") or []),
        "aggregation": spec.get("aggregation"),
        "weight_field": spec.get("weight_field"),
        "reporting_period": spec.get("reporting_date") or receipt.get("period"),
        "portfolio_lens": spec.get("portfolio_lens"),
        "comparison_basis": spec.get("comparison_basis"),
        "applied_filter_fields": list(meta.get("applied_filter_fields") or []),
        "population_rows": receipt.get("population"),
    }


def replay_scope(scope: Dict[str, Any], frame, *, metric: str,
                 aggregation: str, extra_filters: Optional[Dict] = None,
                 grouping: Optional[List[str]] = None) -> Dict[str, Any]:
    """Execute a NEW measure over a RECONSTRUCTED population. No question.

    This is the whole readiness question in one call: if a spec assembled from
    the reconstructed scope executes on the unchanged deterministic executor and
    lands on the same rows, the conversational layer can sit above the estate
    rather than inside it.
    """
    from mi_agent.mi_query_executor import execute_mi_query
    from mi_agent.mi_query_spec import MIQuerySpec
    from mi_agent.mi_query_validator import load_mi_semantics

    filters = dict(scope.get("population_filters") or {})
    filters.update(extra_filters or {})
    dims = list(grouping if grouping is not None else (scope.get("grouping") or []))
    spec = MIQuerySpec(
        intent="chart" if dims else "summary",
        chart_type="bar" if dims else "none",
        metric=metric, aggregation=aggregation, filters=filters,
        dimension=dims[0] if dims else None, dimensions=dims,
        title="[replay] reconstructed scope", output_format="table")
    semantics = load_mi_semantics(SEMANTICS_PATH)
    res = execute_mi_query(spec, frame, semantics,
                           dataset=scope.get("dataset") or "funded")
    row = res.data.iloc[0] if len(res.data) else None
    return {
        "ok": True,
        "filters_executed": filters,
        "grouping_executed": dims,
        "applied_filter_fields": list(res.metadata.get("applied_filter_fields") or []),
        "filtered_row_count": res.metadata.get("filtered_row_count"),
        "input_row_count": res.metadata.get("input_row_count"),
        "columns": list(res.data.columns),
        "loan_count": (int(row["loan_count"])
                       if row is not None and "loan_count" in res.data.columns
                       else None),
        "group_rows": int(len(res.data)),
    }


def phase_b(frame) -> Dict[str, Any]:
    """Reconstruct-and-replay over the review's worked example."""
    q1 = ("What is the funded balance for joint borrowers in the London region "
          "with LTV above 40%?")
    r1 = ask(q1, frame)
    scope = reconstruct_scope(r1)
    obs = observed(r1)

    cases = []
    # INHERIT — the same population, a different governed measure.
    cases.append(("INHERIT: WA LTV over the inherited population",
                  replay_scope(scope, frame, metric="current_loan_to_value",
                               aggregation="weighted_avg")))
    # ADD — the same population plus one predicate.
    cases.append(("ADD: + LTV > 80 over the inherited population",
                  replay_scope(scope, frame, metric="current_outstanding_balance",
                               aggregation="sum",
                               extra_filters={"current_loan_to_value":
                                              {"op": "gt", "value": 80.0}})))
    # MODIFY — one predicate replaced.
    modified = dict(scope["population_filters"])
    modified["borrower_type"] = "Single"
    cases.append(("MODIFY: borrower_type Joint -> Single",
                  replay_scope({**scope, "population_filters": modified}, frame,
                               metric="current_outstanding_balance",
                               aggregation="sum")))
    # RESET — every inherited narrowing dropped.
    cases.append(("RESET: whole funded book",
                  replay_scope({**scope, "population_filters": {}}, frame,
                               metric="current_outstanding_balance",
                               aggregation="sum")))
    # GROUP — the same population, cut by a dimension.
    cases.append(("GROUP: inherited population by age bucket",
                  replay_scope(scope, frame, metric="current_outstanding_balance",
                               aggregation="sum", grouping=["age_bucket"])))

    present = {name: (scope.get(name) not in (None, {}, []))
               for name, _owner in SCOPE_SLOTS if name in scope}
    return {
        "question": q1,
        "observed": obs,
        "reconstructed_scope": scope,
        "slot_owners": {name: owner for name, owner in SCOPE_SLOTS},
        "slot_populated": present,
        "replays": [{"case": name, **out} for name, out in cases],
        # The proof: the INHERIT replay must land on the ORIGINAL row population.
        "inherit_population_matches": (
            cases[0][1]["loan_count"] == obs["population"]),
    }


# --------------------------------------------------------------------------- #
# PHASE E / F — multi-output composition, isolated against its atoms
# --------------------------------------------------------------------------- #
#: Each shape: the combined request, the atoms it decomposes into, and the
#: measure fields a correct answer must prove it executed.
MULTI_SHAPES: List[Dict[str, Any]] = [
    {
        "id": "A_same_population_two_measures",
        "dataset": "funded",
        "question": "For joint borrowers, give me loan count and funded balance.",
        "expect_outputs": ["loan_count", "current_outstanding_balance"],
        "expect_population": {"borrower_type": "Joint"},
        "atoms": ["How many loans have a joint borrower type?",
                  "What is the funded balance for joint borrowers?"],
    },
    {
        "id": "B_same_population_three_measures",
        "dataset": "funded",
        "question": "For joint borrowers, give me count, balance and weighted average LTV.",
        "expect_outputs": ["loan_count", "current_outstanding_balance",
                           "current_loan_to_value"],
        "expect_population": {"borrower_type": "Joint"},
        "atoms": ["How many loans have a joint borrower type?",
                  "What is the funded balance for joint borrowers?",
                  "What is the weighted average LTV for joint borrowers?"],
    },
    {
        "id": "B2_geographic_population_three_measures",
        "dataset": "funded",
        "question": "For loans in the London region, give me count, balance and weighted average LTV.",
        "expect_outputs": ["loan_count", "current_outstanding_balance",
                           "current_loan_to_value"],
        "expect_population": {"geographic_region_obligor": "London"},
        "atoms": ["How many loans are in the London region?",
                  "What is the balance in the London region?",
                  "What is the weighted average LTV in the London region?"],
    },
    {
        "id": "C_shared_population_clause_specific_filter",
        "dataset": "funded",
        "question": ("For loans in the London region, what is the loan count, the balance, "
                     "and how much of that balance has LTV above 40%?"),
        "expect_outputs": ["loan_count", "current_outstanding_balance",
                           "current_outstanding_balance@ltv_gt_40"],
        "expect_population": {"geographic_region_obligor": "London"},
        "atoms": ["How many loans are in the London region?",
                  "What is the balance in the London region?",
                  "What is the balance in the London region with LTV above 40%?"],
    },
    {
        "id": "D_grouped_multi_measure",
        "dataset": "funded",
        "question": "By borrower type, show count, balance and WA LTV.",
        "expect_outputs": ["loan_count", "current_outstanding_balance",
                           "current_loan_to_value"],
        "expect_population": {},
        "atoms": ["How many loans by borrower type?",
                  "What is the balance by borrower type?",
                  "What is the weighted average LTV by borrower type?"],
    },
    {
        "id": "E_pipeline_multi_measure",
        "dataset": "pipeline",
        "question": ("For pipeline loans at the OFFER stage, give me case count, "
                     "balance and weighted average LTV."),
        "expect_outputs": ["loan_count", "current_outstanding_balance",
                           "current_loan_to_value"],
        "expect_population": {"pipeline_stage": "OFFER"},
        "atoms": ["How many pipeline cases are at the OFFER stage?",
                  "What is the pipeline balance at the OFFER stage?",
                  "What is the weighted average LTV at the OFFER stage?"],
    },
    {
        "id": "E2_pipeline_joint_multi_measure",
        "dataset": "pipeline",
        "question": "For pipeline joint borrowers, show case count, pipeline amount and WA LTV.",
        "expect_outputs": ["loan_count", "current_outstanding_balance",
                           "current_loan_to_value"],
        "expect_population": {"borrower_type": "Joint"},
        "atoms": ["How many pipeline cases have a joint borrower type?",
                  "What is the pipeline balance for joint borrowers?",
                  "What is the weighted average LTV for joint borrowers?"],
    },
]

ATOMIC_BLOCKED = "ATOMIC_BLOCKED"
COMPOSITION_FAILED = "COMPOSITION_FAILED"
COMPOSITION_VERIFIED = "COMPOSITION_VERIFIED"


def _population_matches(obs: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Did EXECUTION narrow on exactly the fields the shared population names?

    Read from ``applied_filter_fields`` — execution's own declaration — not from
    the spec, so a filter that was parsed and then not applied cannot pass.
    """
    applied = set(obs.get("applied_filter_fields") or ())
    want = set(expected or ())
    if want - applied:
        return False
    # A clause-scoped predicate promoted to the shared population is the silent
    # over-narrowing this check exists to catch.
    return not (applied - want)


def classify_shape(shape: Dict[str, Any], funded, pipeline) -> Dict[str, Any]:
    frame = pipeline if shape["dataset"] == "pipeline" else funded
    combined = observed(ask(shape["question"], frame, dataset=shape["dataset"]))
    atoms = [{"question": q,
              **{k: v for k, v in observed(ask(q, frame, dataset=shape["dataset"])).items()
                 if k in ("ok", "error", "filters", "applied_filter_fields",
                          "population", "receipt")}}
             for q in shape["atoms"]]
    atoms_all_green = all(a["ok"] for a in atoms)

    want = [m for m in shape["expect_outputs"] if "@" not in m]
    got = list(combined.get("measures_executed") or [])
    every_output_executed = set(want).issubset(set(got))
    clause_scoped_outputs = [m for m in shape["expect_outputs"] if "@" in m]
    population_ok = _population_matches(combined, shape["expect_population"])

    if not atoms_all_green:
        verdict = ATOMIC_BLOCKED
    elif (combined["ok"] and every_output_executed and population_ok
            and not clause_scoped_outputs):
        verdict = COMPOSITION_VERIFIED
    else:
        verdict = COMPOSITION_FAILED

    return {
        "id": shape["id"],
        "dataset": shape["dataset"],
        "question": shape["question"],
        "verdict": verdict,
        "atoms_all_green": atoms_all_green,
        "combined_ok": combined["ok"],
        "outputs_requested": shape["expect_outputs"],
        "outputs_executed": got,
        "every_output_executed": every_output_executed,
        "clause_scoped_outputs_unrepresentable": clause_scoped_outputs,
        "population_expected": shape["expect_population"],
        "population_applied": combined.get("applied_filter_fields"),
        "population_ok": population_ok,
        "silent": bool(combined["ok"] and not (every_output_executed
                                               and population_ok)),
        "combined": combined,
        "atoms": atoms,
    }


def phase_ef(funded, pipeline) -> Dict[str, Any]:
    rows = [classify_shape(s, funded, pipeline) for s in MULTI_SHAPES]
    green = [r for r in rows if r["atoms_all_green"]]
    return {
        "total": len(rows),
        "atoms_all_green": len(green),
        "composition_verified": sum(1 for r in green
                                    if r["verdict"] == COMPOSITION_VERIFIED),
        "composition_failed_despite_green_atoms": sum(
            1 for r in green if r["verdict"] == COMPOSITION_FAILED),
        "atomic_blocked": sum(1 for r in rows if r["verdict"] == ATOMIC_BLOCKED),
        "silent_partial_answers": sum(1 for r in rows if r["silent"]),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# PHASE G — multi-turn, run the way the stateless API runs it today
# --------------------------------------------------------------------------- #
DIALOGUES: List[Dict[str, Any]] = [
    {"id": "1_inherit", "dataset": "funded",
     "q1": "What is the funded balance for joint borrowers in the London region?",
     "q2": "What is their weighted average LTV?",
     "owed": {"geographic_region_obligor": "London", "borrower_type": "Joint"},
     "expectation": "same population, new measure"},
    {"id": "2_add_filter", "dataset": "funded",
     "q1": "What is the funded balance for joint borrowers in the London region?",
     "q2": "How much of that has LTV above 80%?",
     "owed": {"geographic_region_obligor": "London", "borrower_type": "Joint",
              "current_loan_to_value": {"op": "gt", "value": 80.0}},
     "expectation": "inherit population, add one predicate"},
    {"id": "3_modify_filter", "dataset": "funded",
     "q1": "What is the funded balance for joint borrowers in the London region?",
     "q2": "What about single borrowers?",
     "owed": {"geographic_region_obligor": "London", "borrower_type": "Single"},
     "expectation": "replace borrower_type, retain geography"},
    {"id": "4_numeric_reference", "dataset": "funded",
     "q1": "What is the funded balance for joint borrowers in the London region?",
     "q2": "Of the £38m, what is the weighted average LTV?",
     "owed": {"geographic_region_obligor": "London", "borrower_type": "Joint"},
     "expectation": "the money phrase is a REFERENCE, never a predicate"},
    {"id": "5_ambiguous_referent", "dataset": "funded",
     "q1": "What is the balance by borrower type?",
     "q2": "What is their weighted average LTV?",
     "owed": None,
     "expectation": "refuse / clarify; never pick one cohort silently"},
    {"id": "6_reset", "dataset": "funded",
     "q1": "What is the funded balance for joint borrowers in the London region?",
     "q2": "Now show me the whole funded book.",
     "owed": {},
     "expectation": "drop every inherited narrowing"},
    {"id": "7_dataset_boundary", "dataset": "funded",
     "q1": "What is the funded balance for joint borrowers in the London region?",
     "q2": "What about the pipeline?",
     "owed": None,
     "expectation": "explicit dataset transition, no unchecked field carry-over"},
    {"id": "8_failed_prior_turn", "dataset": "funded",
     "q1": "What is the funded balance for platinum borrowers?",
     "q2": "What about their balance?",
     "owed": {},
     "expectation": "no state may be created from an unexecuted population"},
    {"id": "9_presentation", "dataset": "funded",
     "q1": "What is the funded balance for joint borrowers in the London region?",
     "q2": "Show that by age bucket.",
     "owed": {"geographic_region_obligor": "London", "borrower_type": "Joint"},
     "expectation": "inherit population, add grouping, change renderer"},
]


def classify_turn(d: Dict[str, Any], obs1: Dict[str, Any],
                  obs2: Dict[str, Any]) -> str:
    """What today's stateless behaviour does to this follow-up."""
    if not obs2["ok"]:
        return "REFUSED (honest; no state to inherit)"
    owed = d["owed"]
    applied = set(obs2.get("applied_filter_fields") or ())
    if owed is None:
        return "ANSWERED WITHOUT CLARIFYING (ambiguity not detected)"
    want = set(owed)
    if want and not want & applied:
        return "SILENTLY BROADENED to the whole book"
    if want == applied:
        return "CORRECT"
    return "PARTIALLY INHERITED"


def phase_g(funded) -> Dict[str, Any]:
    rows = []
    for d in DIALOGUES:
        obs1 = observed(ask(d["q1"], funded, dataset=d["dataset"]))
        obs2 = observed(ask(d["q2"], funded, dataset=d["dataset"]))
        rows.append({
            "id": d["id"], "expectation": d["expectation"],
            "q1": d["q1"], "q2": d["q2"],
            "q1_population": obs1["applied_filter_fields"],
            "q1_rows": obs1["population"], "q1_ok": obs1["ok"],
            "q2_population": obs2["applied_filter_fields"],
            "q2_rows": obs2["population"], "q2_ok": obs2["ok"],
            "population_owed": d["owed"],
            "today": classify_turn(d, obs1, obs2),
            "q2_receipt": obs2["receipt"] or obs2["error"],
        })
    return {"rows": rows,
            "silently_broadened": sum(1 for r in rows
                                      if r["today"].startswith("SILENTLY")),
            "refused": sum(1 for r in rows if r["today"].startswith("REFUSED"))}


# --------------------------------------------------------------------------- #
# PHASE J-F — numeric reference safety
# --------------------------------------------------------------------------- #
NUMERIC_REFERENCE_PROBES = [
    ("Of the £38m, what is the weighted average LTV?", False),
    ("Of the 38 million, what is the weighted average LTV?", False),
    ("What is the weighted average LTV of the £38m?", False),
    ("Of the 43 loans, what is the weighted average LTV?", False),
    ("Of the £38m, how many loans are there?", False),
    # The control: an EXPLICIT threshold must still bind, or the check proves
    # nothing except that the parser is deaf to numbers.
    ("What is the balance for loans above £250,000?", True),
]


def phase_j_numeric(funded) -> Dict[str, Any]:
    rows = []
    for question, expect_predicate in NUMERIC_REFERENCE_PROBES:
        obs = observed(ask(question, funded))
        has = bool(obs["filters"])
        rows.append({"question": question, "expected_predicate": expect_predicate,
                     "filters": obs["filters"], "ok": obs["ok"],
                     "correct": has == expect_predicate})
    return {"rows": rows, "all_correct": all(r["correct"] for r in rows)}



# --------------------------------------------------------------------------- #
# ATOMIC CONFOUNDERS — defects found while ISOLATING composition, recorded here
# so the review's compositional findings are not read as covering them.
# --------------------------------------------------------------------------- #
#: A bare place name beside another predicate. ``(question, expect_geography)``.
#: The place is a governed value of ``geographic_region_obligor`` in this book,
#: so every row here is a question the book can answer.
BARE_PLACE_PROBES = [
    ("What is the balance in the London region?", True),
    ("What is the balance in London?", True),
    ("What is the funded balance for joint borrowers in the London region?", True),
    ("What is the funded balance for joint borrowers in London?", True),
    ("What is the funded balance for joint borrowers in Scotland?", True),
    ("What is the funded balance for joint borrowers in the South East?", True),
]


def phase_atomic_confounders(funded) -> Dict[str, Any]:
    """Does a bare place name survive beside a second predicate?

    Recorded because it changes what the composition rows below MEAN: a
    multi-turn row that inherits nothing because the FIRST turn never bound the
    geography is an atomic finding wearing a conversational costume.
    """
    rows = []
    for question, expect_geo in BARE_PLACE_PROBES:
        obs = observed(ask(question, funded))
        bound = "geographic_region_obligor" in (obs["filters"] or {})
        rows.append({"question": question, "geography_bound": bound,
                     "ok": obs["ok"], "filters": obs["filters"],
                     "fails_closed": (not bound) and (not obs["ok"]),
                     "correct": bound == expect_geo})
    return {"rows": rows,
            "dropped": sum(1 for r in rows if not r["geography_bound"]),
            "all_dropped_fail_closed": all(r["fails_closed"] for r in rows
                                           if not r["geography_bound"])}


# --------------------------------------------------------------------------- #
def run_all() -> Dict[str, Any]:
    funded = funded_tape()
    pipeline = pipeline_tape()
    return {
        "phase_b_scope_reconstruction": phase_b(funded),
        "phase_ef_multi_output": phase_ef(funded, pipeline),
        "phase_g_multi_turn": phase_g(funded),
        "phase_j_numeric_reference": phase_j_numeric(funded),
        "atomic_confounders": phase_atomic_confounders(funded),
    }


def _print(report: Dict[str, Any]) -> None:
    b = report["phase_b_scope_reconstruction"]
    print("=" * 78)
    print("PHASE B — SCOPE RECONSTRUCTION")
    print("=" * 78)
    print("  question : %s" % b["question"])
    print("  executed : %s" % b["observed"]["receipt"])
    print("  rebuilt  : %s" % json.dumps(b["reconstructed_scope"]["population_filters"]))
    print("  slots    : %s" % json.dumps(b["slot_populated"]))
    for r in b["replays"]:
        print("    %-52s rows=%-5s groups=%-3s" % (r["case"],
                                                    r["filtered_row_count"],
                                                    r["group_rows"]))
        print("        executed population: %s"
              % json.dumps(r["filters_executed"], default=str))
    print("  INHERIT replay lands on the original population: %s"
          % b["inherit_population_matches"])

    e = report["phase_ef_multi_output"]
    print()
    print("=" * 78)
    print("PHASE E/F — MULTI-OUTPUT COMPOSITION")
    print("=" * 78)
    print("  total multi questions ................... %d" % e["total"])
    print("  atoms all green ......................... %d" % e["atoms_all_green"])
    print("  composition verified .................... %d" % e["composition_verified"])
    print("  composition failed despite green atoms .. %d"
          % e["composition_failed_despite_green_atoms"])
    print("  blocked by a broken atom ................ %d" % e["atomic_blocked"])
    print("  answered ok but not as asked (SILENT) ... %d" % e["silent_partial_answers"])
    for r in e["rows"]:
        print("    %-42s %-21s outputs %s/%s  pop_ok=%s%s"
              % (r["id"], r["verdict"], len(r["outputs_executed"]),
                 len(r["outputs_requested"]), r["population_ok"],
                 "  << SILENT" if r["silent"] else ""))

    g = report["phase_g_multi_turn"]
    print()
    print("=" * 78)
    print("PHASE G — MULTI-TURN (stateless, as shipped)")
    print("=" * 78)
    for r in g["rows"]:
        print("    %-22s Q1 pop=%-6s -> Q2 pop=%-6s  %s"
              % (r["id"], r["q1_rows"], r["q2_rows"], r["today"]))
    print("  silently broadened: %d of %d" % (g["silently_broadened"], len(g["rows"])))

    j = report["phase_j_numeric_reference"]
    print()
    print("=" * 78)
    print("PHASE J-F — NUMERIC REFERENCE SAFETY")
    print("=" * 78)
    for r in j["rows"]:
        print("    %-58s predicate=%-5s %s"
              % (r["question"][:58], bool(r["filters"]),
                 "OK" if r["correct"] else "WRONG"))
    print("  all correct: %s" % j["all_correct"])

    a = report["atomic_confounders"]
    print()
    print("=" * 78)
    print("ATOMIC CONFOUNDERS (not compositional — recorded so they are not read as such)")
    print("=" * 78)
    for r in a["rows"]:
        print("    %-62s geo_bound=%-5s ok=%s"
              % (r["question"][:62], r["geography_bound"], r["ok"]))
    print("  place names dropped: %d; every drop fails closed: %s"
          % (a["dropped"], a["all_dropped_fail_closed"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit the raw report")
    ap.add_argument("--out", type=str, default=None, help="write the JSON here")
    args = ap.parse_args()
    report = run_all()
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
