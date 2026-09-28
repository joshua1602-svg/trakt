"""Slice 3: the population a plan asks for must be the population that ran.

TWO DEFECTS, ONE DEPENDENCY ORDER. The legacy/fallback audit found that
`population.base` was never bound, never reconciled and never enforced, so a
governed plan stating `pipeline` would have executed over the funded frame and
been served; and that the legacy parser gave "back book" and "front book" a
SEASONING meaning, so a canary question whose governed attempt fell back came
out of the stack meaning something else. The first is fixed first, because until
a plan's population is enforced there is nothing to fall back TO safely.

THE TWO SIDES, AND THEIR OWNERS.

    requested   `GovernedQueryPlan.population.base` — the plan's transcription
    executed    the governed dataset identity the runtime resolved its frame
                with, passed to `serve` as `view`

Neither is inferred from the question or from the rows. The controls below are
numbered as the brief numbers them.
"""

from __future__ import annotations

import json

import pytest

from mi_agent import plan_runtime_adapter as adapter
from mi_agent import seasoning as sz
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent

_SEASONING_COLUMNS = ["seasoning_segment", "months_on_book",
                      "current_outstanding_balance"]


def _plan(population, **over):
    body = {"schema_version": "candidate_intent/1.0",
            "capability": "generic_analysis", "operation": "point_in_time",
            "population": population,
            "measures": [{"concept": "current_outstanding_balance"}],
            "time": {"form": "current"}}
    body.update(over)
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(body))
    assert result.plan is not None, f"{population} did not compile"
    return result.plan.to_dict()


# --------------------------------------------------------------------------- #
# 1-5 — the population base gate
# --------------------------------------------------------------------------- #

def test_control_1_funded_base_on_the_funded_runtime_executes():
    ok, why, _ = adapter.check_population_base(_plan({"base": "funded"}), "funded")
    assert (ok, why) == (True, "")


def test_control_2_a_pipeline_base_never_runs_on_the_funded_runtime():
    """And zero rows are touched: the gate is BEFORE execution, not after."""
    ok, why, detail = adapter.check_population_base(
        _plan({"base": "pipeline"}), "funded")
    assert ok is False
    assert why == adapter.POPULATION_NOT_EXECUTABLE
    assert "refused rather than answered over" in detail
    # Even if a pipeline FRAME were supplied, this runtime still does not
    # execute pipeline — the refusal is about the runtime, not the frame.
    assert adapter.check_population_base(
        _plan({"base": "pipeline"}), "pipeline")[1] == adapter.POPULATION_NOT_EXECUTABLE


def test_control_3_an_unstated_base_is_the_accepted_funded_contract():
    """Slice 1 behaviour is unchanged: empty resolves to funded and executes."""
    assert adapter.requested_population_base(_plan({})) == "funded"
    assert adapter.check_population_base(_plan({}), "funded")[0] is True
    assert adapter.check_population_base(
        _plan({"base": "funded", "lens": "acquired"}), "funded")[0] is True


def test_control_4_a_runtime_that_declares_nothing_is_refused():
    for undeclared in (None, "", "   "):
        ok, why, _ = adapter.check_population_base(_plan({"base": "funded"}),
                                                   undeclared)
        assert ok is False, f"{undeclared!r} was treated as proof"
        assert why == adapter.EXECUTED_POPULATION_UNPROVEN


def test_control_5_a_base_that_mismatches_the_runtime_is_refused():
    ok, why, detail = adapter.check_population_base(
        _plan({"base": "funded"}), "pipeline")
    assert ok is False
    assert why == adapter.POPULATION_BASE_MISMATCH
    assert "'funded'" in detail and "'pipeline'" in detail


def test_whole_book_spans_two_frames_and_cannot_be_proven_against_one():
    ok, why, _ = adapter.check_population_base(_plan({"base": "whole_book"}),
                                               "funded")
    assert (ok, why) == (False, adapter.POPULATION_NOT_EXECUTABLE)


def test_a_future_runtime_registers_itself_rather_than_being_special_cased():
    """The design point: no CandidateIntent change when pipeline gets an owner."""
    assert adapter.EXECUTABLE_POPULATIONS == frozenset({"funded"})
    ok, why, _ = adapter.check_population_base(
        _plan({"base": "pipeline"}), "pipeline", executable={"pipeline"})
    assert (ok, why) == (True, "")


def test_the_gate_reads_the_plan_and_the_runtime_and_nothing_else():
    """No question, no rows. Two governed records compared."""
    import ast
    import inspect
    import textwrap

    # THE PROSE IS NOT THE CODE. The gate's docstring says, in English, that it
    # never re-reads the question — so a naive scan of the source trips on the
    # very sentence that documents the property. Docstrings are stripped first.
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(adapter.check_population_base)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body.pop(0)
    code = ast.unparse(tree)
    for forbidden in ("question", "row", "frame", "re.", "search("):
        assert forbidden not in code, (
            f"the population gate reads {forbidden!r}; it may read only the "
            f"plan and the declared runtime identity")


# --------------------------------------------------------------------------- #
# 6-10 — the legacy lexical owner
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question", [
    "What is the back book balance?",
    "What is the balance of the backbook?",
    "What is the acquired back book balance?",
])
def test_control_6_a_back_book_question_gains_no_seasoning_on_fallback(question):
    assert sz.resolve_population_predicate(question, _SEASONING_COLUMNS) is None


@pytest.mark.parametrize("question", [
    "What is in the front book?",
    "What is the front book balance?",
])
def test_control_7_a_front_book_question_gains_no_seasoning_on_fallback(question):
    assert sz.resolve_population_predicate(question, _SEASONING_COLUMNS) is None


def test_control_8_new_originations_is_not_a_seasoning_selection():
    assert sz.resolve_population_predicate(
        "What is the balance of new originations?", _SEASONING_COLUMNS) is None


@pytest.mark.parametrize("question,segment", [
    ("What is the balance of seasoned loans?", "Back Book"),
    ("What is the balance of the legacy book?", "Back Book"),
    ("What is the seasoned book balance?", "Back Book"),
    ("Show balance for recently originated loans.", "Front Book"),
    ("Show balance for newly originated loans.", "Front Book"),
    ("What is the balance of recent originations?", "Front Book"),
])
def test_control_9_legitimate_vintage_phrases_still_select(question, segment):
    assert sz.resolve_population_predicate(question, _SEASONING_COLUMNS) == {
        "seasoning_segment": segment}


def test_control_9b_a_vintage_comparison_still_narrows_nothing():
    """The signed-off NL7 canonical. Two windows named is a comparison.

    This is the regression the first cut of this repair introduced: deleting
    "back book" from the recognition table left ONE window named where there had
    been two, and the comparison silently became a narrowing.
    """
    question = ("How different is the risk profile of recent originations "
                "versus the back book?")
    assert sz.lending_windows_named(question) == ["front_book", "back_book"]
    assert sz.resolve_population_predicate(question, _SEASONING_COLUMNS) is None


def test_control_10_every_phrase_is_still_masked_before_the_place_resolver():
    """P1J-1: "Front" read as a region made a 250-loan population unreachable."""
    masked = sz.mask_segment_phrases(
        "How many acquired loans are in the front book?")
    assert "front" not in masked.lower() and "book" not in masked.lower()
    assert len(masked) == len("How many acquired loans are in the front book?")
    for phrase in ("back book", "backbook", "new originations", "seasoned loans"):
        assert phrase not in sz.mask_segment_phrases(
            f"balance of the {phrase}").lower()


def test_the_lifecycle_phrases_are_still_RECOGNISED_just_not_selected_on():
    """Recognition and selection are separate, and only selection moved."""
    assert sz.lending_windows_named("the back book") == ["back_book"]
    assert sz.segments_named("the front book") == ["Front Book"]
    assert sz.lifecycle_owned_only("the back book", "back_book") is True
    assert sz.lifecycle_owned_only("seasoned loans", "back_book") is False
    # A question using BOTH a lifecycle phrase and a vintage one has named a
    # vintage, and the vintage wins.
    assert sz.lifecycle_owned_only(
        "seasoned loans in the back book", "back_book") is False


# --------------------------------------------------------------------------- #
# 11-14 — nothing else moved
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("role", ["direct", "acquired"])
def test_control_11_the_role_axis_is_untouched(role):
    plan = _plan({"base": "funded", "lens": role})
    scope = [(p["canonical_field"], p["value"])
             for p in plan["population"]["scope_predicates"]]
    assert scope == [("source_portfolio_type", role)]
    assert adapter.check_population_base(plan, "funded")[0] is True


def test_control_12_the_named_source_axis_is_untouched():
    from trakt_core.portfolio import build_registry

    registry = build_registry(
        [{"source_portfolio_id": "alp_acquired",
          "source_portfolio_type": "acquired"}],
        metadata={"alp_acquired": {"source_portfolio_label": "ALP Acquired Back Book",
                                   "aliases": ["ALP back book"]}},
        client_id="ERE")
    body = {"schema_version": "candidate_intent/1.0",
            "capability": "generic_analysis", "operation": "point_in_time",
            "population": {"base": "funded",
                           "source_reference": "ALP Acquired Back Book"},
            "measures": [{"concept": "current_outstanding_balance"}],
            "time": {"form": "current"}}
    result = DeterministicCompiler(
        CompilerContext(source_registry=registry)).compile(
            parse_candidate_intent(body))
    scope = [(p.canonical_field, p.value)
             for p in result.plan.population.scope_predicates]
    assert scope == [("source_portfolio_id", "alp_acquired")]
    assert adapter.check_population_base(result.plan.to_dict(), "funded")[0] is True


def test_control_13_the_signed_off_corpus_is_unchanged_under_the_gate():
    """Every plan the perimeter admitted before is admitted now.

    135 recorded payloads, recompiled and re-checked. The measurement that makes
    the gate safe: of the 135, 83 ask for funded, 34 pipeline, 8 forecast and 1
    whole_book — and every one the slice 1 perimeter already admitted asks for
    funded, so refusing the rest costs no case that was being served.
    """
    from mi_agent import plan_temporal_runtime as temporal

    with open("mi_agent/interpretation_v2/evidence/"
              "run8_135_signoff_2b00172.json", encoding="utf-8") as handle:
        recorded = json.load(handle)["results"]
    compiler = DeterministicCompiler(CompilerContext())

    admitted, refused_by_base = 0, []
    for row in recorded:
        result = compiler.compile(parse_candidate_intent(row["raw_payload"]))
        plan = getattr(result, "plan", None)
        if plan is None:
            continue
        body = plan.to_dict()
        was = (temporal.check_temporal_eligibility(body)[0]
               if temporal.claims(body) else adapter.check_eligibility(body)[0])
        if not was:
            continue
        admitted += 1
        if not adapter.check_population_base(body, "funded")[0]:
            refused_by_base.append(row["question_id"])
    assert admitted > 0, "the replay admitted nothing, so it proves nothing"
    assert refused_by_base == [], (
        f"the population gate refused {len(refused_by_base)} case(s) the "
        f"perimeter already admitted: {refused_by_base}")


def test_control_14_a_temporal_plan_is_gated_on_the_same_terms():
    """Slice 2 shares the gate rather than carrying a second one."""
    from mi_agent import plan_temporal_runtime as temporal

    plan = _plan({"base": "funded"}, operation="series",
                 time={"form": "series", "grain": "monthly"})
    assert temporal.claims(plan), "this fixture stopped being a temporal plan"
    assert adapter.check_population_base(plan, "funded")[0] is True
    assert adapter.check_population_base(plan, "pipeline")[1] == (
        adapter.POPULATION_BASE_MISMATCH)


def test_the_gate_sits_above_the_runtime_dispatch():
    """One gate, both runtimes, and before either executes."""
    import inspect

    from mi_agent import plan_serving_canary as canary

    source = inspect.getsource(canary._attempt)
    gate = source.index("check_population_base")
    assert gate < source.index("temporal.claims(plan)"), (
        "the temporal runtime is dispatched before the population is proven")
    assert gate < source.index("adapter.check_eligibility(plan)")
