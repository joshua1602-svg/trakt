"""Catalogue batch 2: the forecast semantic model (P0 design §16).

`config/mi/semantic_model/forecast.yaml` declares each forecast figure once —
what the model is told it is, and the path of the figure in the output of the
owner the Forecast tab renders. These tests hold the file to that:

  - it loads, and every path it declares resolves in the owners' real output;
  - every figure the agent serves through it equals the tab's (or the scale-up
    owner's) own figure for the same book — never a number written here;
  - what the owner does not publish is refused, never recomputed;
  - the answer names the measure and its as-at, and the coverage owner proves
    the member, the axis and the region the plan asked for.
"""
from __future__ import annotations

import ast
import inspect

import pytest
import yaml

from mi_agent import plan_forecast_runtime as forecast_rt
from mi_agent import semantic_model
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import IntentParseError, parse_candidate_intent
from tests.interpretation_v2.test_specialist_runtime_forecast import (  # noqa: F401
    _CLIENT, _NEAR, _owner, _semantics, _served, _tab, estate, funded_root)

_RUN = "mi_2025_11"
_MODEL = semantic_model.load("forecast")


def _intent(**over):
    payload = {"schema_version": "candidate_intent/1.0", "capability": "forecast",
               "operation": "point_in_time", "population": {"base": "forecast"},
               "measures": [{"concept": "forecast_funded_balance"}],
               "time": {"form": "current"}}
    payload.update(over)
    return payload


def _plan(**over):
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(_intent(**over)))
    assert result.plan is not None, [(r.code, r.detail) for r in result.reasons]
    return result.plan.to_dict()


def _member(concept, value):
    return [{"concept": concept, "comparator": "eq", "value": value}]


def _run(plan, funded_root, estate):
    ok, why, detail = forecast_rt.check_eligibility(plan)
    assert ok, (why, detail)
    return _execute(plan, funded_root, estate)


def _execute(plan, funded_root, estate):
    return forecast_rt.execute(
        plan, output_root=funded_root, pipeline_root=_NEAR, client_id=_CLIENT,
        run_id=_RUN, funded_frame_resolver=estate, semantics=_semantics())


def _scale_up():
    """What the scale-up owner itself publishes for the same book and run —
    the call the tab's `/mi/forecast/extrapolation` route makes."""
    from mi_agent_api import forecast_extrapolation as fx
    import os
    return fx.build_extrapolation(os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"],
                                  _NEAR, _CLIENT, _RUN, history_model=None)


def _snapshot():
    """The Forecast tab's whole envelope for the same book."""
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    return TestClient(app).get(
        f"/mi/forecast/snapshot?portfolioId={_CLIENT}/{_RUN}").json()


# --------------------------------------------------------------------------- #
# the file
# --------------------------------------------------------------------------- #

def test_the_model_is_what_the_vocabulary_shows():
    from mi_agent.interpretation_v2.vocabulary import (
        SPECIALIST_DIMENSIONS, SPECIALIST_MEASURE_DEFINITIONS, SPECIALIST_MEASURES,
        load_governed_vocabulary)
    assert SPECIALIST_MEASURES["forecast"] == tuple(_MODEL.measures)
    assert SPECIALIST_DIMENSIONS["forecast"] == tuple(_MODEL.dimensions)
    vocabulary = load_governed_vocabulary()
    for name, measure in _MODEL.measures.items():
        assert SPECIALIST_MEASURE_DEFINITIONS[name] == measure.definition
        assert vocabulary.resolve(name).owning_capability == "forecast"
    for name, dimension in _MODEL.dimensions.items():
        assert vocabulary.resolve(name).values == dimension.values


def test_every_declared_path_resolves_in_the_owners_real_output(funded_root, estate):
    views = {"forecast_view": _snapshot(),
             "scale_up": _scale_up()}
    for name, m in _MODEL.measures.items():
        payload = views[m.view]
        paths = [m.value] if m.value else []
        paths += list(m.context.values())
        if m.series:
            paths += [m.series["rows"], m.series["horizon"]]
        for binding in m.by.values():
            paths += [spec["value"] for spec in (binding.get("members") or {}).values()]
            paths += [binding[k] for k in ("rows", "map", "basis") if binding.get(k)]
        for path in paths:
            assert semantic_model.has(payload, path), (name, path)


def test_an_invalid_model_is_refused_before_the_model_sees_it():
    good = {"version": 1, "capability": "forecast", "population": "forecast",
            "views": {"v": {"owner": "x"}},
            "dimensions": {"d": {"definition": "a d", "values": ["a"]}},
            "measures": {"m": {"definition": "an m", "unit": "gbp", "view": "v",
                               "value": "x.y", "by": {"d": {"members": {"a": {"value": "x"}}}}}}}
    semantic_model._validate(good, "forecast")
    for broken, why in (
            ({"view": "nope"}, "undeclared view"),
            ({"unit": "percent"}, "unit"),
            ({"by": {"d": {"members": {"z": {"value": "x"}}}}}, "outside its governed values"),
            ({"by": {"elsewhere": {"rows": "r"}}}, "undeclared"),
            ({"definition": ""}, "no definition")):
        doc = {**good, "measures": {"m": {**good["measures"]["m"], **broken}}}
        with pytest.raises(semantic_model.SemanticModelError, match=why):
            semantic_model._validate(doc, "forecast")


def test_the_model_reader_computes_nothing():
    tree = ast.parse(inspect.getsource(semantic_model))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.BinOp)]


# --------------------------------------------------------------------------- #
# every figure is the tab's
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("member, tab_key", [
    ("funded_book", "fundedBalance"),
    ("weighted_pipeline", "weightedExpectedFundedAmount")])
def test_each_part_of_the_forecast_is_the_tabs(member, tab_key, funded_root, estate):
    out = _run(_plan(filters=_member("forecast_component", member)),
               funded_root, estate)
    assert out.ok, (out.reason, out.detail)
    assert out.value == _tab()[tab_key]
    assert out.receipt["applied_predicates"] == [
        {"field": "forecast_component", "op": "eq", "values": [member]}]


def test_the_funded_part_reads_the_funded_book_alone(funded_root, estate):
    """P0 D1: only a figure that composes both books has a skew."""
    out = _run(_plan(filters=_member("forecast_component", "funded_book")),
               funded_root, estate)
    assert set(out.receipt["inputs"]) == {"funded"}
    assert out.receipt["input_vintage_skew_days"] is None


def test_the_bridge_is_the_tabs_two_parts(funded_root, estate):
    out = _run(_plan(operation="breakdown", dimensions=["forecast_component"]),
               funded_root, estate)
    tab = _tab()
    assert {c["forecast_component"]: c["value"] for c in out.cells} == {
        "funded_book": tab["fundedBalance"],
        "weighted_pipeline": tab["weightedExpectedFundedAmount"]}


def test_the_forecast_loan_count_is_the_tabs(funded_root, estate):
    out = _run(_plan(measures=[{"concept": "forecast_loan_count"}]),
               funded_root, estate)
    assert out.value == _tab()["forecastLoanCount"]


def test_the_forecast_by_ltv_band_is_the_tabs(funded_root, estate):
    out = _run(_plan(operation="breakdown", dimensions=["ltv_bucket"]),
               funded_root, estate)
    tab = _snapshot()["forecastBreakdowns"]["byLtvBucket"]
    assert tab
    assert {c["ltv_bucket"]: c["value"] for c in out.cells} == {
        r["key"]: r["forecastAmount"] for r in tab}


_PIPELINE = {"base": "pipeline"}


def test_the_weighting_exclusion_is_the_tabs_by_reason(funded_root, estate):
    total = _run(_plan(population=_PIPELINE,
                       measures=[{"concept": "weighting_excluded_amount"}]),
                 funded_root, estate)
    tab = _tab()
    assert total.value == tab["excludedFromWeightingAmount"]
    by = _run(_plan(operation="breakdown", population=_PIPELINE,
                    measures=[{"concept": "weighting_excluded_amount"}],
                    dimensions=["weighting_exclusion_reason"]), funded_root, estate)
    assert {c["weighting_exclusion_reason"]: c["value"] for c in by.cells} == {
        k: v["amount"] for k, v in tab["excludedByReason"].items()}
    withdrawn = _run(_plan(population=_PIPELINE,
                           measures=[{"concept": "weighting_excluded_case_count"}],
                           filters=_member("weighting_exclusion_reason", "not_forecast")),
                     funded_root, estate)
    assert withdrawn.value == tab["excludedByReason"]["not_forecast"]["count"]
    assert set(total.receipt["inputs"]) == {"pipeline"}
    # The receipt proves the population the figure is measured over.
    assert total.receipt["population_base"] == "pipeline"


def test_each_figure_is_asked_about_the_population_it_is_measured_over():
    """The 2026-09-29 spot check: 'how much pipeline is excluded because of
    missing probability' was read, correctly, as the pipeline's exclusion —
    and refused because the runtime executed `forecast` alone. The population
    is now each figure's own declaration; a plan naming another is refused,
    never answered from it."""
    assert _MODEL.measure("weighting_excluded_amount").population == "pipeline"
    assert _MODEL.measure("forecast_funded_balance").population == "forecast"
    assert forecast_rt.EXECUTABLE_POPULATIONS == {"forecast", "pipeline"}
    exclusion = _plan(population=_PIPELINE,
                      measures=[{"concept": "weighting_excluded_amount"}],
                      filters=_member("weighting_exclusion_reason",
                                      "missing_probability"))
    assert forecast_rt.check_eligibility(exclusion) == (True, "", "")
    assert forecast_rt.execution_population(exclusion) == "pipeline"
    on_the_forecast = _plan(measures=[{"concept": "weighting_excluded_amount"}])
    assert forecast_rt.check_eligibility(on_the_forecast)[1] == \
        forecast_rt.POPULATION_NOT_MEASURED
    pipeline_forecast = _plan(population=_PIPELINE)
    assert forecast_rt.check_eligibility(pipeline_forecast)[1] == \
        forecast_rt.POPULATION_NOT_MEASURED


def test_a_figure_measured_over_an_input_reads_that_input_alone():
    doc = yaml.safe_load(semantic_model._ROOT.joinpath("forecast.yaml").read_text())
    doc["measures"]["weighting_excluded_amount"]["inputs"] = ["funded", "pipeline"]
    with pytest.raises(semantic_model.SemanticModelError, match="measured over"):
        semantic_model._validate(doc, "forecast")
    doc["measures"]["weighting_excluded_amount"]["inputs"] = ["pipeline"]
    doc["measures"]["weighting_excluded_amount"]["population"] = "funded"
    with pytest.raises(semantic_model.SemanticModelError, match="measured over"):
        semantic_model._validate(doc, "forecast")
    del doc["population"]
    with pytest.raises(semantic_model.SemanticModelError, match="no population"):
        semantic_model._validate(doc, "forecast")


def test_the_excluded_pipeline_is_served_as_the_pipelines(monkeypatch, funded_root,
                                                         estate):
    payload = _served_forecast(
        _intent(population=_PIPELINE,
                measures=[{"concept": "weighting_excluded_amount"}],
                filters=_member("weighting_exclusion_reason", "not_forecast")),
        monkeypatch, funded_root, estate)
    tab = _tab()
    assert payload["metadata"]["governedPlan"]["executed"]["population_base"] == \
        "pipeline"
    assert payload["artifacts"][0]["kpis"][0]["rawValue"] == \
        tab["excludedByReason"]["not_forecast"]["amount"]
    assert tab["pipelineAsOfDate"] in payload["answer"]


def test_the_owner_publishes_every_reason_and_they_add_up():
    from mi_agent_api import pipeline_contract as pc
    from mi_agent_api.pipeline_prep import EXCLUSION_REASONS
    import glob
    extract = sorted(glob.glob(f"{_NEAR}/pipeline/*/*"))[0]
    _frame, report = pc.load_prepared_pipeline(
        {"source_file": extract, "pipeline_as_of_date": "2025-10-01"})
    summary = report["completion_probability_summary"]
    by = summary["excluded_by_reason"]
    assert set(EXCLUSION_REASONS) <= set(by)
    assert sum(r["amount"] for r in by.values()) == pytest.approx(summary["excluded_amount"])
    assert sum(r["count"] for r in by.values()) == summary["excluded_count"]


# --------------------------------------------------------------------------- #
# the scale-up forecast — every figure the owner's
# --------------------------------------------------------------------------- #

_CURVE = {"operation": "forecast_projection",
          "measures": [{"concept": "projected_funded_balance"}],
          "time": {"form": "forward_looking"}}


def test_the_curve_is_the_owners_with_all_three_bands(funded_root, estate):
    out = _run(_plan(**_CURVE), funded_root, estate)
    rows = _scale_up()["completionRunRateForecast"]["projectedBalances"]
    assert [(c["period"], c["downside"], c["base"], c["upside"]) for c in out.cells] == \
        [(r["month"], r["downside"], r["base"], r["upside"]) for r in rows]
    assert out.receipt["series_columns"] == ["downside", "base", "upside"]


def test_the_next_twelve_months_is_the_owners_curve_selected_not_projected(
        funded_root, estate):
    plan = _plan(**dict(_CURVE, time={"form": "forward_looking", "grain": "monthly",
                                      "periods_ahead": 12}))
    out = _run(plan, funded_root, estate)
    rows = _scale_up()["completionRunRateForecast"]["projectedBalances"]
    wanted = [r for r in rows if 1 <= r["offset"] <= 12]
    assert [c["period"] for c in out.cells] == [r["month"] for r in wanted]
    assert [c["base"] for c in out.cells] == [r["base"] for r in wanted]
    assert out.receipt["horizon"]["periods_ahead"] == 12


def test_a_horizon_past_the_owners_is_refused(funded_root, estate):
    plan = _plan(**dict(_CURVE, time={"form": "forward_looking", "grain": "monthly",
                                      "periods_ahead": 36}))
    out = _run(plan, funded_root, estate)
    assert (out.ok, out.reason) == (False, forecast_rt.PERIOD_NOT_SUPPORTED)


def test_a_horizon_stated_only_in_words_is_refused():
    plan = _plan(**dict(_CURVE, time={"form": "forward_looking", "grain": "monthly",
                                      "labels": ["next twelve months"]}))
    ok, why, _ = forecast_rt.check_eligibility(plan)
    assert (ok, why) == (False, forecast_rt.PERIOD_NOT_SUPPORTED)


def test_the_downside_forecast_is_the_owners_downside_line(funded_root, estate):
    out = _run(_plan(**dict(_CURVE, filters=_member("forecast_scenario", "downside"))),
               funded_root, estate)
    rows = _scale_up()["completionRunRateForecast"]["projectedBalances"]
    assert [c["downside"] for c in out.cells] == [r["downside"] for r in rows]
    assert all(set(c) == {"period", "offset", "downside"} for c in out.cells)


def test_the_annualised_run_rate_is_the_owners(funded_root, estate):
    out = _run(_plan(measures=[{"concept": "annualised_completion_run_rate"}]),
               funded_root, estate)
    rr = _scale_up()["completionRunRateForecast"]
    assert out.value == rr["annualisedRunRate"]
    # the run-rate's own input is its signal, not the funded book too
    assert set(out.receipt["inputs"]) == {"pipeline"}


def test_the_scenario_run_rates_are_declared_and_held_until_evidence(
        funded_root, estate):
    """The shape `point_in_time/forecast_completion_rate` stays held (D6 hold,
    §14.6); the declaration is ready, and reads the owner's figure."""
    plan = _plan(measures=[{"concept": "forecast_completion_rate"}],
                 filters=_member("forecast_scenario", "downside"))
    assert forecast_rt.check_eligibility(plan)[1] == forecast_rt.AMBIGUOUS_READING
    out = _execute(plan, funded_root, estate)
    rr = _scale_up()["completionRunRateForecast"]
    assert out.value == rr["scenarioMonthlyRunRate"]["downside"]


def test_the_milestone_ladder_is_the_owners_table(funded_root, estate):
    plan = _plan(operation="breakdown", measures=[{"concept": "forecast_milestone_date"}],
                 dimensions=["funding_threshold"], time={"form": "forward_looking"})
    out = _run(plan, funded_root, estate)
    rows = _scale_up()["completionRunRateForecast"]["milestones"]
    assert [(c["funding_threshold"], c["value"], c["downsideDate"]) for c in out.cells] == \
        [(r["thresholdLabel"], r["baseDate"], r.get("downsideDate")) for r in rows]


# --------------------------------------------------------------------------- #
# region: the reporting taxonomy on both books, or refused
# --------------------------------------------------------------------------- #

_BY_REGION = {"operation": "breakdown",
              "geography": {"requested": True, "group_by": True, "level": "reporting"}}


def test_forecast_by_region_on_raw_spellings_is_refused_not_substituted(
        funded_root, estate):
    """This fixture's funded tape carries no harmonised region, so the tab
    falls back to raw spellings — and the agent will not answer from them."""
    assert _snapshot()["forecastBreakdowns"]["regionBasis"]["field"] == \
        "geographic_region_obligor"
    out = _run(_plan(**_BY_REGION), funded_root, estate)
    assert (out.ok, out.reason) == (False, forecast_rt.FIELD_UNAVAILABLE)


def test_forecast_by_region_is_the_views_reporting_regions(monkeypatch, funded_root,
                                                            estate):
    """With both books harmonised (the platform path does this for the funded
    book, D12 for the pipeline) the view adds them up in reporting regions and
    the agent reads that breakdown."""
    from engine import region_taxonomy
    from mi_workflows.analytical import executors as composer

    taxonomy = region_taxonomy.resolve_taxonomy(None)
    real = composer.forecast_view

    def harmonised(ctx):
        base = ctx.base_frame().copy()
        base["geographic_region_obligor"] = [
            ("London", "SOUTH-EAST", "Yorkshire")[i % 3] for i in range(len(base))]
        region_taxonomy.apply(base, taxonomy)
        ctx._memo["base_frame"] = base
        return real(ctx)

    monkeypatch.setattr(composer, "forecast_view", harmonised)
    out = _run(_plan(**_BY_REGION), funded_root, estate)
    assert out.ok, (out.reason, out.detail)
    keys = {c["canonical_region_reporting"] for c in out.cells}
    assert {"London", "South East", "Yorkshire and The Humber"} <= keys
    assert keys <= set(taxonomy.values)
    assert out.receipt["group_field_keys"] == ["canonical_region_reporting"]
    assert out.receipt["axis_basis"]["field"] == "canonical_region_reporting"
    # What the regions rest on, per source column, over the rows the breakdown
    # adds up: every funded loan here was placed from the borrower column.
    rows = out.receipt["axis_basis"]["sourceFieldRows"]
    assert rows.get("geographic_region_obligor", 0) >= 1
    from mi_agent import answer_standard
    note = answer_standard.region_note(rows, counts=False)
    assert note.startswith("Regions are the client's reporting regions, by ")
    assert "the borrower's address" in note


# --------------------------------------------------------------------------- #
# what is not published is refused
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("over, reason", [
    ({"operation": "breakdown", "dimensions": ["broker_channel"]},
     forecast_rt.DIMENSION_NOT_SUPPORTED),
    ({"operation": "breakdown", "dimensions": ["forecast_component", "ltv_bucket"]},
     forecast_rt.DIMENSION_NOT_SUPPORTED),
    ({"filters": _member("forecast_component", "something_else")},
     forecast_rt.FILTERS_NOT_SUPPORTED),
    ({"operation": "breakdown", "dimensions": ["forecast_component"],
      "filters": _member("forecast_component", "funded_book")},
     forecast_rt.FILTERS_NOT_SUPPORTED),
    ({"measures": [{"concept": "forecast_loan_count"}],
      "filters": _member("forecast_component", "funded_book")},
     forecast_rt.FILTERS_NOT_SUPPORTED),
    ({"measures": [{"concept": "forecast_milestone_date"}], "operation": "forecast_milestone",
      "dimensions": ["funding_threshold"], "time": {"form": "forward_looking"},
      "target": {"concept": "forecast_funded_balance", "comparator": "gte",
                 "value": 50_000_000}},
     forecast_rt.DIMENSION_NOT_SUPPORTED),
])
def test_what_the_owner_does_not_publish_is_refused(over, reason):
    try:
        plan = _plan(**over)
    except AssertionError:
        return                       # the compiler already refused it
    ok, why, _ = forecast_rt.check_eligibility(plan)
    assert (ok, why) == (False, reason)


# --------------------------------------------------------------------------- #
# the forward horizon is a governed slot
# --------------------------------------------------------------------------- #

def test_periods_ahead_reaches_the_plan_and_is_part_of_its_identity():
    with_horizon = _plan(**dict(_CURVE, time={"form": "forward_looking",
                                              "periods_ahead": 12}))
    without = _plan(**_CURVE)
    assert with_horizon["period"]["periods_ahead"] == 12
    assert without["period"]["periods_ahead"] is None
    assert with_horizon["plan_id"] != without["plan_id"]


def test_periods_ahead_belongs_to_a_forward_looking_question():
    result = DeterministicCompiler(CompilerContext()).compile(parse_candidate_intent(
        _intent(time={"form": "current", "periods_ahead": 3})))
    assert result.plan is None
    assert "time.periods_ahead" in [r.subject for r in result.reasons]
    with pytest.raises(IntentParseError):
        parse_candidate_intent(_intent(time={"form": "forward_looking",
                                             "periods_ahead": 0}))


# --------------------------------------------------------------------------- #
# the answer
# --------------------------------------------------------------------------- #

def _served_forecast(intent, monkeypatch, funded_root, estate):
    from mi_agent_api.mi_service import _governed_plan_coverage
    payload, record = _served(intent, monkeypatch, funded_root, run_id=_RUN,
                              funded_frame_resolver=estate, semantics=_semantics())
    assert payload is not None, record.get("execution")
    assert _governed_plan_coverage(payload)["unaccounted"] == []
    return payload


def test_the_pipeline_part_answer_names_itself_and_both_dates(monkeypatch, funded_root,
                                                             estate):
    payload = _served_forecast(
        _intent(filters=_member("forecast_component", "weighted_pipeline")),
        monkeypatch, funded_root, estate)
    tab = _tab()
    assert payload["answer"].startswith(
        "Forecast funded balance (component: weighted pipeline): ")
    assert tab["pipelineAsOfDate"] in payload["answer"]
    assert payload["artifacts"][0]["kpis"][0]["rawValue"] == \
        tab["weightedExpectedFundedAmount"]


def test_the_curve_answer_is_a_line_chart_of_the_owners_bands(monkeypatch, funded_root,
                                                             estate):
    payload = _served_forecast(
        _intent(**dict(_CURVE, time={"form": "forward_looking", "grain": "monthly",
                                     "periods_ahead": 12})),
        monkeypatch, funded_root, estate)
    assert payload["answer"].startswith(
        "Projected funded balance over the next 12 months, ")
    chart = payload["artifacts"][0]
    assert chart["type"] == "chart" and chart["chartType"] == "line"
    assert [s["key"] for s in chart["series"]] == ["downside", "base", "upside"]
    assert any("indicative scenario bands" in w for w in payload["warnings"])


def test_the_ladder_answer_is_a_table_with_every_threshold(monkeypatch, funded_root,
                                                          estate):
    payload = _served_forecast(
        _intent(operation="breakdown", measures=[{"concept": "forecast_milestone_date"}],
                dimensions=["funding_threshold"], time={"form": "forward_looking"}),
        monkeypatch, funded_root, estate)
    rows = _scale_up()["completionRunRateForecast"]["milestones"]
    table = payload["artifacts"][0]
    assert table["type"] == "table" and len(table["rows"]) == len(rows)
    assert payload["answer"].startswith("Forecast milestone date by funding threshold: ")


def test_the_balance_answer_is_unchanged_in_substance(monkeypatch, funded_root, estate):
    payload = _served_forecast(_intent(), monkeypatch, funded_root, estate)
    assert payload["answer"].startswith("Forecast funded balance: ")
    assert "the funded balance of" in payload["answer"]
    assert "of expected completions from the open pipeline" in payload["answer"]
