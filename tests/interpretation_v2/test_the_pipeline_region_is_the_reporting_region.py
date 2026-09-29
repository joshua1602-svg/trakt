"""The pipeline "by region" is the client's REPORTING region — on the Pipeline
tab and in an agent answer alike (owner, 2026-09-29).

"By region" governs to the reporting taxonomy (`canonical_region_reporting`),
and the funded book's preparation stamps it with `engine.region_taxonomy`. The
Pipeline tab used to group the extract's raw spelling instead, so the agent
could not answer from the tab without substituting one geography for the
other. The pipeline's preparation now applies the same engine, the tab's region
chart groups by its result, and the agent reads that chart.

Each figure here is compared with the tab's own snapshot for the same frame.
"""
from __future__ import annotations

import glob

import pandas as pd
import pytest

from engine import region_taxonomy
from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent_api import pipeline_contract as pc
from mi_agent_api import pipeline_prep

_EXTRACT = sorted(glob.glob("tests/fixtures/client_001_mi_pack/pipeline/*/*"))[0]
_AS_OF = "2025-10-01"
_BY_REGION = {"schema_version": "candidate_intent/1.0", "capability": "pipeline",
              "operation": "breakdown", "population": {"base": "pipeline"},
              "measures": [{"concept": "pipeline_amount"}],
              "time": {"form": "current"},
              "geography": {"requested": True, "group_by": True,
                            "level": "reporting"}}


@pytest.fixture(scope="module")
def semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    return load_mi_semantics(semantics_path())


def _source():
    return {"source_file": _EXTRACT, "pipeline_as_of_date": _AS_OF}


def _prepared(*regions: str):
    """The fixture extract through the real preparation, with the extract's
    own region spelling of its first live cases replaced by ``regions``."""
    raw = pd.read_csv(_EXTRACT)
    live = raw.index[raw["Status"].astype(str).map(pipeline_prep.canonical_stage)
                     .isin(pipeline_prep.OPEN_STAGES)]
    for row, region in zip(live, regions):
        raw.loc[row, "Property Region"] = region
    return pipeline_prep.prepare_pipeline_mi_dataset(
        raw, as_of_date=_AS_OF, source_file=_EXTRACT)


def _tab(frame, report):
    return pc.compute_pipeline_snapshot(frame, report, {}, client_id="client_001",
                                        run_id="fixture", source=_source())


def _plan(**over):
    body = dict(_BY_REGION, **over)
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(body))
    assert result.plan is not None, [r.code for r in result.reasons]
    return result.plan.to_dict()


def _serve_frame(monkeypatch, frame, report):
    monkeypatch.setattr(pc, "load_prepared_pipeline",
                        lambda *a, **k: (frame, report))


# --------------------------------------------------------------------------- #
# the preparation and the tab
# --------------------------------------------------------------------------- #

def test_the_pipeline_is_harmonised_by_the_funded_books_engine():
    frame, report = _prepared("SOUTH-WEST")
    assert report["region_harmonisation"]["applied"] is True
    # The same engine and taxonomy as the funded book: the same answer per value.
    taxonomy = region_taxonomy.resolve_taxonomy(None)
    for raw, reporting in zip(frame["region_source_value"],
                              frame["canonical_region_reporting"]):
        assert reporting == taxonomy.to_reporting(
            taxonomy.resolve_detail(raw)[0])[0]
    # Cleaning and approved synonyms resolve; the own spelling is kept beside.
    assert {"South West", "Yorkshire and The Humber"} <= set(
        frame["canonical_region_reporting"])
    assert {"SOUTH-WEST", "Yorkshire"} <= set(frame["geographic_region_obligor"])


def test_the_tabs_region_chart_is_the_reporting_region():
    frame, report = _prepared("SOUTH-WEST", "yorkshire")
    tab = _tab(frame, report)
    assert tab["regionBasis"]["field"] == "canonical_region_reporting"
    assert tab["regionBasis"]["taxonomy"] == \
        report["region_harmonisation"]["reporting_taxonomy"]
    keys = {r["key"] for r in tab["regionBreakdownFull"]}
    assert {"South West", "Yorkshire and The Humber"} <= keys
    assert keys <= set(region_taxonomy.resolve_taxonomy(None).values)
    # The extract's own spelling stays available for audit, not charted.
    assert {"SOUTH-WEST", "yorkshire"} <= {
        r["key"] for r in tab["regionSourceBreakdownFull"]}
    assert tab["regionBasis"]["unmappedCaseCount"] == 0


def test_a_region_with_no_mapping_is_disclosed_not_placed():
    frame, report = _prepared("Atlantis")
    tab = _tab(frame, report)
    basis = tab["regionBasis"]
    assert basis["unmappedCaseCount"] == 1
    assert basis["unmappedValues"] == {"Atlantis": 1}
    assert basis["unmappedAmount"] > 0
    assert "Atlantis" not in {r["key"] for r in tab["regionBreakdownFull"]}
    # Placed + unplaced is the whole live pipeline: nothing vanished silently.
    placed = sum(r["pipelineAmount"] for r in tab["regionBreakdownFull"])
    assert placed + basis["unmappedAmount"] == pytest.approx(
        tab["pipelineAmount"])


def test_with_no_taxonomy_the_tab_is_as_before():
    frame, report = _prepared()
    bare = frame.drop(columns=["canonical_region_reporting",
                               "canonical_region_detail"])
    tab = _tab(bare, report)
    assert tab["regionBasis"]["field"] == pc.RAW_REGION_FIELD
    assert tab["regionBreakdownFull"] == tab["regionSourceBreakdownFull"]


# --------------------------------------------------------------------------- #
# the agent reads the tab's chart
# --------------------------------------------------------------------------- #

def test_the_agents_pipeline_by_region_is_the_tabs(monkeypatch, semantics):
    frame, report = _prepared("Atlantis")
    _serve_frame(monkeypatch, frame, report)
    plan = _plan()
    assert pipeline_rt.check_eligibility(plan) == (True, "", "")
    out = pipeline_rt.execute_current(plan, source=_source(), semantics=semantics)
    assert out.ok, (out.reason, out.detail)
    tab = _tab(frame, report)
    assert {c["canonical_region_reporting"]: c["value"] for c in out.cells} == \
        pytest.approx({r["key"]: r["pipelineAmount"]
                       for r in tab["regionBreakdownFull"]})
    assert out.receipt["group_field_keys"] == ["canonical_region_reporting"]
    assert out.receipt["region_basis"] == tab["regionBasis"]


def test_without_the_reporting_region_the_agent_refuses(monkeypatch, semantics):
    """Never the raw spelling in its place."""
    frame, report = _prepared()
    _serve_frame(monkeypatch, frame.drop(columns=["canonical_region_reporting"]),
                 report)
    out = pipeline_rt.execute_current(_plan(), source=_source(),
                                      semantics=semantics)
    assert (out.ok, out.reason) == (False, pipeline_rt.FIELD_UNAVAILABLE)


@pytest.mark.parametrize("geography,why", [
    ({"requested": True, "group_by": False, "level": "reporting",
      "values": ["London"]}, pipeline_rt.GEOGRAPHY_NOT_SUPPORTED),
    ({"requested": True, "group_by": True, "level": "nuts3", "basis": "obligor"},
     pipeline_rt.GEOGRAPHY_NOT_SUPPORTED),
])
def test_a_region_filter_or_another_level_is_refused(geography, why):
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(dict(_BY_REGION, geography=geography)))
    if result.plan is None:
        return                       # the compiler already refused it
    ok, reason, _ = pipeline_rt.check_eligibility(result.plan.to_dict())
    assert (ok, reason) == (False, why)


def test_region_over_time_or_with_another_axis_is_refused():
    over_time = _plan(time={"form": "series", "grain": "weekly"})
    assert pipeline_rt.check_eligibility(over_time)[1] == \
        pipeline_rt.DIMENSION_NOT_SUPPORTED
    with_broker = _plan(dimensions=["broker_channel"])
    assert pipeline_rt.check_eligibility(with_broker)[1] == \
        pipeline_rt.DIMENSION_NOT_SUPPORTED


# --------------------------------------------------------------------------- #
# the answer
# --------------------------------------------------------------------------- #

def test_the_answer_names_the_regions_and_what_it_left_out(monkeypatch):
    from mi_agent_api.mi_service import _governed_plan_coverage
    from tests.interpretation_v2.test_specialist_runtime_pipeline import _served

    frame, report = _prepared("Atlantis")
    _serve_frame(monkeypatch, frame, report)
    payload, record = _served(_BY_REGION, monkeypatch, pipeline_source=_source())
    assert payload is not None, record.get("execution")
    answer = payload["answer"]
    # The answer standard: measure, grouping, leaders, how many groups.
    assert answer.startswith("Live pipeline amount by region — largest: ")
    assert "groups), as at the weekly extract of" in answer
    # The regions are named as the client's; the taxonomy's own id is for the
    # notes, not the sentence.
    assert "Regions are the client's reporting regions;" in answer
    assert "uk_itl1" not in answer
    assert any("uk_itl1" in n["note"] for n in payload["sourceNotes"])
    assert "1 case (" in answer and "no governed mapping is in no region" in answer
    coverage = _governed_plan_coverage(payload)
    assert coverage["unaccounted"] == []
    assert any(e["kind"] == "governed_plan:geography" for e in coverage["concepts"])


def test_a_region_asked_for_and_not_grouped_is_unaccounted():
    from mi_agent_api.mi_service import _governed_plan_coverage

    envelope = {"metadata": {
        "parserMode": "governed_plan",
        "governedPlan": {
            "requested": {"capability": "pipeline",
                          "measure_concept": "pipeline_amount",
                          "population": {"base": "pipeline"},
                          "geography": {"group_by": True, "canonical_field":
                                        "canonical_region_reporting"}},
            "executed": {"capability": "pipeline",
                         "measure_concept": "pipeline_amount",
                         "population_base": "pipeline",
                         "group_field_keys": [], "applied_predicates": []}}}}
    coverage = _governed_plan_coverage(envelope)
    assert [e["kind"] for e in coverage["unaccounted"]] == [
        "governed_plan:geography"]
