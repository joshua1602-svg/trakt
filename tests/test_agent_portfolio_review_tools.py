#!/usr/bin/env python3
"""The primitives a period review needs, and the governance they keep.

Two things the existing tool surface could not reach: the weekly pipeline, which
no tool touched at all, and WHY the funded book moved — a different fact from how
much, and the one a reader acts on.

The lines this suite holds:

* **the same governance as every other tool** — capability, entitlement, SPV and
  population refusals, checked here rather than assumed from a sibling;
* **no loan rows** — a pipeline movement is dimension aggregates, and a
  composition is components; neither may leak a case identifier;
* **absence is an answer** — no prior week, no approved configuration and no
  pipeline root come back as readable refusals, not exceptions;
* **the acquisition rule survives the tool boundary** — a balance jump inside one
  portfolio must not reach an agent looking like a book arriving.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core import config_cache
from trakt_core.context import SCOPE_LOAN_READ, SCOPE_RISK_READ
from trakt_core.envelope import STATUS_BLOCKED, STATUS_SUCCESS
from trakt_core.errors import ErrorCode
from trakt_tools.execution import ToolDependencies, execute_governed_tool
from trakt_tools.handlers.portfolio_review import MAX_CONTRIBUTORS

from tests.planted_portfolio import SNAPSHOT_ID, planted_frame
from tests.test_agent_governed_execution import _Datasets, _Descriptor
from tests.test_agent_loan_retrieval import (
    PORTFOLIO_A, SPV_I, TENANT, _catalogue, _context, _CountingResolver,
)

BOTH = (SCOPE_LOAN_READ, SCOPE_RISK_READ)

REVIEW_TOOLS = (
    ("pipeline_position", {}),
    ("pipeline_movement", {}),
    ("pipeline_conversion", {}),
    ("funded_composition", {}),
    ("forward_concentration", {}),
)

PIPELINE_TOOLS = ("pipeline_position", "pipeline_movement",
                  "pipeline_conversion")
FUNDED_TOOLS = ("funded_composition", "forward_concentration")


@pytest.fixture(autouse=True)
def _clean_cache():
    config_cache.reset()
    yield
    config_cache.reset()


def _deps(**extra) -> ToolDependencies:
    base: Dict[str, Any] = dict(
        datasets=_Datasets(_Descriptor(snapshot_id=SNAPSHOT_ID)),
        runtime_mode="test", catalogue=_catalogue(), output_root="/unused",
        loan_frame_resolver=_CountingResolver(planted_frame()),
        pipeline_root="/governed/pipeline")
    base.update(extra)
    return ToolDependencies(**base)


def _run(tool: str, args: Dict[str, Any], *, deps=None, context=None):
    payload = {"resource": PORTFOLIO_A.key, **args}
    return execute_governed_tool(tool, payload,
                                 context or _context(capabilities=BOTH),
                                 dependencies=deps or _deps())


def _ok(tool: str, args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    result = _run(tool, args, **kwargs)
    assert result.status == STATUS_SUCCESS, result.error and result.error.to_dict()
    return result.result


# =========================================================================== #
# The tools are published, and published as reusable primitives
# =========================================================================== #
def test_every_review_tool_is_registered_with_a_schema():
    from trakt_tools.registry import get

    for name, _args in REVIEW_TOOLS:
        spec = get(name)
        assert spec is not None, name
        assert spec.input_schema["properties"]["resource"]
        assert spec.required_capability == SCOPE_RISK_READ
        # Every one of them is aggregate: none needs loan:read.
        assert spec.agent_guidance


def test_the_review_surface_is_five_primitives_not_a_question_list():
    """A tool per question would be a checklist wearing a tool surface."""
    from trakt_tools.registry import all_tools

    names = {s.name for s in all_tools()}
    assert {n for n, _ in REVIEW_TOOLS} <= names


# =========================================================================== #
# Governance — checked here, not assumed from a sibling tool
# =========================================================================== #
@pytest.mark.parametrize("tool, args", REVIEW_TOOLS)
def test_a_caller_without_the_capability_is_refused(tool, args):
    result = _run(tool, args, context=_context(capabilities=()))
    assert result.status == STATUS_BLOCKED


@pytest.mark.parametrize("tool, args", REVIEW_TOOLS)
def test_an_spv_boundary_is_refused_rather_than_widened(tool, args):
    """Answering from the enclosing book under an SPV's label is the fail-open
    every one of these has to refuse independently.

    The SPV is granted in the context on purpose: an entitlement refusal would
    also stop the call, and would prove nothing about whether the tool itself
    keeps the boundary.
    """
    result = execute_governed_tool(
        tool, {"resource": SPV_I.key, **args},
        _context(SPV_I, capabilities=BOTH), dependencies=_deps())
    assert result.status == STATUS_BLOCKED, tool
    assert result.error_code == ErrorCode.RESOURCE_NOT_PARTITIONABLE, tool


@pytest.mark.parametrize("tool", PIPELINE_TOOLS)
def test_a_pipeline_tool_refuses_a_resource_pinned_to_the_funded_book(tool,
                                                                     monkeypatch):
    """Symmetric with the funded tools' refusal of a pipeline resource.

    The direction differs but the rule does not: a tool may not report on a
    population its resource does not name.
    """
    from trakt_tools.handlers import portfolio_review as pr

    class _Resolved:
        population = "funded"
        spv_id = None
        ref = type("R", (), {"key": PORTFOLIO_A.key})()

    inv = type("Inv", (), {
        "authorised": type("A", (), {"resource": _Resolved()})(),
        "request_id": "r-1"})()

    with pytest.raises(Exception) as excinfo:
        pr._pipeline_scope(inv, "A pipeline position")
    assert "does not name" in str(excinfo.value)


@pytest.mark.parametrize("tool", PIPELINE_TOOLS)
def test_no_pipeline_root_is_a_readable_refusal(tool):
    """A deployment with no weekly pipeline is normal, and says so."""
    result = _run(tool, {}, deps=_deps(pipeline_root=None))
    assert result.status != STATUS_SUCCESS
    assert result.error.code == ErrorCode.DATA_SOURCE_UNAVAILABLE
    assert "absence of data, not an absence of pipeline" in result.error.message


@pytest.mark.parametrize("tool", FUNDED_TOOLS)
def test_no_funded_output_root_is_an_answer_not_an_exception(tool):
    """Returned rather than raised: a refusal an agent can read IS a finding."""
    payload = _ok(tool, {}, deps=_deps(output_root=None))
    assert payload["available"] is False
    assert "no governed funded output root" in payload["reason"]
    assert any("absence of evidence" in w for w in payload["warnings"])


# =========================================================================== #
# pipeline_movement — bounded, attributed, and carrying no loan rows
# =========================================================================== #
def _movement_payload(**over) -> Dict[str, Any]:
    payload = {
        "available": True,
        "as_of_date": "2026-08-07", "comparison_date": "2026-07-31",
        "headline_metric": {"label": "Pipeline balance", "value": 503_400_000.0,
                            "change": 23_400_000.0, "change_pct": 4.9},
        "counts": {"current": 812, "comparison": 771, "change": 41},
        "contributors": {
            "brokers": [{"name": f"Broker {i}", "amount": 1000.0 - i,
                         "share_of_change_pct": 1.0, "case_count": 3}
                        for i in range(9)],
            "regions": [{"name": "London", "amount": 9_000_000.0,
                         "share_of_change_pct": 38.5, "case_count": 40}],
            "products": [{"name": "Lump Sum", "amount": 15_000_000.0,
                          "share_of_change_pct": 64.1, "case_count": 55}],
        },
        "components": {"new": {"amount": 30_000_000.0}},
        "methodology": {"version": "1", "attribution": "…"},
    }
    payload.update(over)
    return payload


def test_pipeline_movement_carries_the_product_dimension(monkeypatch):
    from mi_agent_api import movement_detail as md

    monkeypatch.setattr(md, "resolve_movement_detail",
                        lambda *a, **k: _movement_payload())
    payload = _ok("pipeline_movement", {})

    assert payload["available"] is True
    assert payload["contributors"]["products"][0]["name"] == "Lump Sum"
    assert payload["headline"]["change"] == 23_400_000.0


def test_contributors_are_capped_per_dimension(monkeypatch):
    """A tool hands back what an agent asked for, not a league table."""
    from mi_agent_api import movement_detail as md

    monkeypatch.setattr(md, "resolve_movement_detail",
                        lambda *a, **k: _movement_payload())
    payload = _ok("pipeline_movement", {})
    assert len(payload["contributors"]["brokers"]) == MAX_CONTRIBUTORS


def test_a_pipeline_movement_never_returns_a_case_identifier(monkeypatch):
    """Dimension aggregates only — the same discipline the insight contract keeps."""
    from mi_agent_api import movement_detail as md

    monkeypatch.setattr(md, "resolve_movement_detail",
                        lambda *a, **k: _movement_payload())
    payload = _ok("pipeline_movement", {})

    for rows in payload["contributors"].values():
        for row in rows:
            assert set(row) <= {"name", "amount", "share_of_change_pct",
                                "case_count"}


def test_no_comparable_prior_week_is_reported_as_such(monkeypatch):
    from mi_agent_api import movement_detail as md

    monkeypatch.setattr(md, "resolve_movement_detail", lambda *a, **k: {
        "available": False, "reason": "only one governed weekly extract exists",
        "as_of_date": "2026-08-07", "comparison_date": None})
    payload = _ok("pipeline_movement", {})

    assert payload["available"] is False
    assert "only one governed weekly extract exists" in payload["reason"]


# =========================================================================== #
# pipeline_conversion — an insufficient window is never quoted
# =========================================================================== #
def test_an_insufficient_conversion_window_is_flagged_not_published(monkeypatch):
    from mi_agent_api import datasets as datasets_mod
    from mi_agent_api import evolution as evolution_mod

    monkeypatch.setattr(datasets_mod, "_pipeline_history", lambda cid: None)
    monkeypatch.setattr(datasets_mod, "_kfi_lag_weeks_from_model", lambda m: 6)
    monkeypatch.setattr(evolution_mod, "pipeline_funnel_evolution",
                        lambda *a, **k: {"summary": {"COMPLETED": {
                            "conversion": {"sufficient": False,
                                           "weeksInWindow": 2}}}})
    payload = _ok("pipeline_conversion", {})

    assert payload["sufficient"] is False
    assert payload["lag_weeks"] == 6
    assert any("too short to publish a conversion rate" in w
               for w in payload["warnings"])


# =========================================================================== #
# funded_composition — the acquisition rule survives the tool boundary
# =========================================================================== #
BALANCE, LOAN, PORTFOLIO = ("current_outstanding_balance", "loan_identifier",
                            "source_portfolio_id")


def _frames(current, prior):
    """Two governed reporting periods, as ``funded_frames`` returns them."""
    return [
        {"run_id": "mi_2026_06", "reporting_date": "2026-06-30", "df": prior,
         "source": "/p/prior.csv"},
        {"run_id": "mi_2026_07", "reporting_date": "2026-07-31", "df": current,
         "source": "/p/current.csv"},
    ]


def _f(rows) -> pd.DataFrame:
    return pd.DataFrame([{LOAN: l, BALANCE: b, PORTFOLIO: p, **x}
                         for l, b, p, x in rows])


ACQ_PRIOR = _f([("A1", 60_000_000.0, "portfolio_alpha", {}),
                ("A2", 52_000_000.0, "portfolio_alpha", {})])
ACQ_CURRENT = _f([
    ("A1", 61_000_000.0, "portfolio_alpha", {}),
    ("A2", 52_000_000.0, "portfolio_alpha", {}),
    ("A3", 3_000_000.0, "portfolio_alpha", {}),
    ("B1", 40_000_000.0, "portfolio_beta",
     {"source_portfolio_type": "acquired", "source_portfolio_label": "Portfolio B"}),
    ("B2", 28_000_000.0, "portfolio_beta",
     {"source_portfolio_type": "acquired", "source_portfolio_label": "Portfolio B"}),
])


@pytest.fixture
def acquisition(monkeypatch):
    from mi_agent_api import evolution as evolution_mod
    monkeypatch.setattr(evolution_mod, "funded_frames",
                        lambda *a, **k: _frames(ACQ_CURRENT, ACQ_PRIOR))


def test_an_acquisition_month_reaches_the_agent_as_components(acquisition):
    payload = _ok("funded_composition", {})

    assert payload["available"] is True
    assert payload["movement"] == 72_000_000.0
    assert payload["components"]["portfolio_additions"] == 68_000_000.0
    assert payload["components"]["organic_new_lending"] == 3_000_000.0
    assert payload["reconciliation"]["reconciles"] is True

    added = payload["portfolio_additions"][0]
    assert added["source_portfolio_id"] == "portfolio_beta"
    assert added["portfolio_type"] == "acquired"


def test_a_balance_jump_alone_never_reaches_the_agent_as_an_addition(monkeypatch):
    """The inference the agent must be structurally unable to make.

    Nothing about the SIZE of a movement may produce a portfolio addition, so a
    book that doubled inside one portfolio hands the agent an empty additions
    list — there is nothing there for a model to call an acquisition.
    """
    from mi_agent_api import evolution as evolution_mod

    prior = _f([("L1", 100_000_000.0, "portfolio_alpha", {})])
    current = _f([("L1", 100_000_000.0, "portfolio_alpha", {}),
                  ("L2", 100_000_000.0, "portfolio_alpha", {})])
    monkeypatch.setattr(evolution_mod, "funded_frames",
                        lambda *a, **k: _frames(current, prior))

    payload = _ok("funded_composition", {})
    assert payload["movement"] == 100_000_000.0
    assert payload["portfolio_additions"] == []
    assert payload["components"]["portfolio_additions"] == 0.0
    assert payload["components"]["organic_new_lending"] == 100_000_000.0


def test_the_underlying_lens_excludes_the_addition(acquisition):
    payload = _ok("funded_composition", {"underlying_only": True})

    assert payload["lens"] == "Underlying"
    assert payload["movement"] == 4_000_000.0
    assert payload["components"]["portfolio_additions"] == 0.0


def test_asking_for_an_underlying_view_of_a_month_with_no_addition_is_refused(
        monkeypatch):
    """A Total answer may never be handed back labelled 'Underlying'."""
    from mi_agent_api import evolution as evolution_mod

    prior = _f([("L1", 100.0, "portfolio_alpha", {})])
    current = _f([("L1", 150.0, "portfolio_alpha", {})])
    monkeypatch.setattr(evolution_mod, "funded_frames",
                        lambda *a, **k: _frames(current, prior))

    payload = _ok("funded_composition", {"underlying_only": True})
    assert payload["available"] is False
    assert "the underlying book is the whole book" in payload["reason"]


def test_a_composition_never_returns_a_loan_identifier(acquisition):
    payload = _ok("funded_composition", {})
    serialised = repr(payload)
    for loan_id in ("A1", "A2", "A3", "B1", "B2"):
        assert f"'{loan_id}'" not in serialised


def test_a_non_reconciling_decomposition_warns_before_it_is_quoted(monkeypatch,
                                                                   acquisition):
    from mi_agent_api import funded_composition as comp

    real = comp.decompose

    def _broken(current, prior):
        out = real(current, prior)
        out["reconciliation"]["reconciles"] = False
        return out

    monkeypatch.setattr(comp, "decompose", _broken)
    payload = _ok("funded_composition", {})
    assert any("do not attribute the movement" in w for w in payload["warnings"])


def test_a_tape_without_portfolio_identity_names_what_it_could_not_split(
        monkeypatch):
    from mi_agent_api import evolution as evolution_mod

    prior = pd.DataFrame([{LOAN: "L1", BALANCE: 100.0}])
    current = pd.DataFrame([{LOAN: "L1", BALANCE: 100.0},
                            {LOAN: "L2", BALANCE: 900.0}])
    monkeypatch.setattr(evolution_mod, "funded_frames",
                        lambda *a, **k: _frames(current, prior))

    payload = _ok("funded_composition", {})
    assert PORTFOLIO in payload["unavailable"]
    assert any("source_portfolio_id" in w for w in payload["warnings"])


# =========================================================================== #
# forward_concentration
# =========================================================================== #
def test_an_extracted_limit_set_is_labelled_indicative(monkeypatch):
    """A legacy limit is not an approved covenant and must not read as one."""
    from mi_agent_api import concentration_tests_api as conc_mod

    monkeypatch.setattr(conc_mod, "compute_concentration_tests", lambda *a, **k: {
        "available": True, "source": "legacy_extracted",
        "reportingDate": "2026-07-31", "tests": [], "emergingRisks": [],
        "states": {"available": True}, "lineage": {}})

    payload = _ok("forward_concentration", {})
    assert payload["source"] == "legacy_extracted"
    assert any("indicative" in w for w in payload["warnings"])


def test_missing_forward_states_are_not_read_as_an_immaterial_pipeline(monkeypatch):
    from mi_agent_api import concentration_tests_api as conc_mod

    monkeypatch.setattr(conc_mod, "compute_concentration_tests", lambda *a, **k: {
        "available": True, "source": "approved_configuration",
        "reportingDate": "2026-07-31", "tests": [], "emergingRisks": [],
        "states": {"available": False}, "lineage": {}})

    payload = _ok("forward_concentration", {})
    assert any("not evidence the pipeline is immaterial" in w
               for w in payload["warnings"])


def test_emerging_risks_are_carried_in_the_governed_order(monkeypatch):
    from mi_agent_api import concentration_tests_api as conc_mod

    ordered = [{"category": "current_breach", "rank": 1},
               {"category": "expected_breach", "rank": 2},
               {"category": "material_deterioration", "rank": 4}]
    monkeypatch.setattr(conc_mod, "compute_concentration_tests", lambda *a, **k: {
        "available": True, "source": "approved_configuration",
        "reportingDate": "2026-07-31", "tests": [], "emergingRisks": ordered,
        "states": {"available": True}, "lineage": {"configurationVersion": "v3"}})

    payload = _ok("forward_concentration", {})
    assert [r["rank"] for r in payload["emerging_risks"]] == [1, 2, 4]
    assert payload["lineage"]["configurationVersion"] == "v3"
