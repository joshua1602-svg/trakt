"""The Period Change Analysis route: integration, rendering and precedence.

Three things are proved here that the domain tests cannot:

* the workflow reaches real data through the EXISTING governed frame service
  (``evolution.funded_frames``), not a parallel data path;
* the route sits in the registry where it claims to, and does not steal a
  question an incumbent route already answers;
* the governed result reaches a channel intact, alongside the existing envelope
  keys rather than instead of them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mi_agent_api import chat_routing, period_change_route as pcr
from mi_agent.period_change import models as m

SEMANTICS_PATH = (Path(__file__).resolve().parents[2]
                  / "mi_agent" / "mi_semantics_field_registry.yaml")


# --------------------------------------------------------------------------- #
# Frames — supplied through the existing governed service
# --------------------------------------------------------------------------- #
def _frame(*, n: int, balance: float, ltv: float, arrears: int, region_split: dict,
           first_loan: int = 0) -> pd.DataFrame:
    regions = []
    for region, share in region_split.items():
        regions.extend([region] * int(round(n * share)))
    regions = (regions + [next(iter(region_split))] * n)[:n]
    return pd.DataFrame({
        "loan_identifier": [f"L{i:05d}" for i in range(first_loan, first_loan + n)],
        "current_outstanding_balance": [balance / n] * n,
        "current_loan_to_value": [ltv] * n,
        "current_interest_rate": [5.5] * n,
        "youngest_borrower_age": [71.0] * n,
        "interest_in_arrears": [("Y" if i < arrears else "N") for i in range(n)],
        "account_status": [("Arrears" if i < arrears else "Performing")
                           for i in range(n)],
        "collateral_geography": regions,
        "recoveries_in_period": [10.0] * n,
        "source_portfolio_id": ["alp_origination"] * n,
        "source_portfolio_label": ["ALP Book"] * n,
    })


@pytest.fixture
def frames(monkeypatch):
    prior = _frame(n=100, balance=1_000_000.0, ltv=0.40, arrears=4,
                   region_split={"South East": 0.5, "London": 0.3, "Wales": 0.2})
    current = _frame(n=110, balance=1_200_000.0, ltv=0.43, arrears=11,
                     region_split={"South East": 0.6, "London": 0.25, "Wales": 0.15})
    built = [
        {"run_id": "2026-05-31", "reporting_date": "2026-05-31", "df": prior,
         "source": "blob://x/2026-05-31/platform_canonical_typed.csv"},
        {"run_id": "2026-06-30", "reporting_date": "2026-06-30", "df": current,
         "source": "blob://x/2026-06-30/platform_canonical_typed.csv"},
    ]
    from mi_agent_api import evolution as evolution_mod
    monkeypatch.setattr(evolution_mod, "funded_frames",
                        lambda *a, **k: [dict(f) for f in built])
    return built


@pytest.fixture
def single_frame(monkeypatch, frames):
    from mi_agent_api import evolution as evolution_mod
    monkeypatch.setattr(evolution_mod, "funded_frames",
                        lambda *a, **k: [dict(frames[-1])])


def ask(question: str, **kw):
    from mi_agent.llm_query_parser import _deterministic_parse
    from mi_agent.mi_query_validator import load_mi_semantics

    semantics = load_mi_semantics(SEMANTICS_PATH)
    spec, _meta = _deterministic_parse(question, semantics)
    params = dict(client_id="client", run_id=None, output_root="blob://x",
                  portfolio_id="client/2026-06-30", as_of=None)
    params.update(kw)
    return pcr.route_period_change(question, spec, spec.to_dict(), **params)


# --------------------------------------------------------------------------- #
# Snapshot supply
# --------------------------------------------------------------------------- #
class TestSnapshotSupply:
    def test_snapshots_come_from_the_existing_governed_frame_service(self, frames):
        snapshots = pcr.build_snapshots("blob://x", "client")
        assert [s.snapshot_id for s in snapshots] == ["2026-05-31", "2026-06-30"]
        assert [s.reporting_date for s in snapshots] == ["2026-05-31", "2026-06-30"]
        assert [s.row_count for s in snapshots] == [100, 110]

    def test_a_dataset_label_is_a_name_never_a_storage_path(self, frames):
        snapshots = pcr.build_snapshots("blob://x", "client")
        assert snapshots[0].dataset_label == "platform_canonical_typed.csv"
        assert "blob://" not in str(snapshots[0].reference())

    def test_source_portfolios_are_read_through_the_governed_scope_helper(self, frames):
        snapshots = pcr.build_snapshots("blob://x", "client")
        assert snapshots[0].portfolio_ids == ("alp_origination",)

    def test_a_resolved_lens_narrows_every_snapshot(self, frames):
        """The same narrowing the funded-bridge and movement routes apply, so a
        period-change answer covers exactly the portfolios they would."""
        from mi_agent.portfolio_lens import SOURCE_ID_FIELD, PortfolioLens

        for built in frames:
            built["df"].loc[built["df"].index[:10], SOURCE_ID_FIELD] = "alp_acquired"

        total = pcr.build_snapshots("blob://x", "client")
        lens = PortfolioLens(name="cohort", label="ALP Acquired",
                             filters={SOURCE_ID_FIELD: ["alp_acquired"]},
                             cohort_id="alp_acquired")
        narrowed = pcr.build_snapshots("blob://x", "client", lens=lens)

        assert [s.row_count for s in total] == [100, 110]
        assert [s.row_count for s in narrowed] == [10, 10]
        assert narrowed[0].portfolio_ids == ("alp_acquired", "alp_origination")

    def test_a_portfolio_absent_at_one_date_fails_rather_than_reporting_nulls(
            self, frames):
        """Presence is read from the UNNARROWED snapshot, so a portfolio the
        lens selected but that a reporting date never contained is reported as
        absent instead of as a snapshot full of empty metrics."""
        from mi_agent.portfolio_lens import SOURCE_ID_FIELD, PortfolioLens

        closing = frames[-1]["df"]
        closing.loc[closing.index[:10], SOURCE_ID_FIELD] = "alp_acquired"
        lens = PortfolioLens(name="cohort", label="ALP Acquired",
                             filters={SOURCE_ID_FIELD: ["alp_acquired"]},
                             cohort_id="alp_acquired")

        with pytest.raises(m.PeriodChangeFailure) as err:
            pcr.analyse_period_change(client_id="client", output_root="blob://x",
                                      scope=lens)
        assert err.value.reason == m.FAIL_PORTFOLIO_ABSENT_AT_PERIOD
        assert err.value.detail["missing_at_start"] == ["alp_acquired"]


class TestAssetApplicability:
    def test_an_asset_class_is_used_only_when_the_caller_states_it(self):
        """§9: with no governed asset-class signal, only cross_asset entries are
        eligible — an equity-release field is never reported for an
        unidentified book."""
        assert pcr.resolve_asset_classes() == ()
        assert pcr.resolve_asset_classes(("equity_release",)) == ("equity_release",)

    def test_an_unknown_asset_class_is_dropped_not_passed_through(self):
        assert pcr.resolve_asset_classes(("not_an_asset_class",)) == ()

    def test_the_known_classes_are_the_registry_vocabulary(self):
        from mi_agent.business_semantics import load_business_semantics

        taxonomy = set(load_business_semantics().taxonomy["asset_applicability"])
        assert set(pcr.KNOWN_ASSET_CLASSES) | {"cross_asset"} == taxonomy


# --------------------------------------------------------------------------- #
# The routed answer
# --------------------------------------------------------------------------- #
class TestRoutedAnswer:
    def test_the_route_answers_a_portfolio_change_question(self, frames):
        out = ask("What changed in the portfolio this month?")
        assert out is not None
        assert out["ok"] is True
        assert out["metadata"]["route"] == "period_change_analysis"
        assert out["metadata"]["workflowId"] == m.WORKFLOW_ID
        assert out["metadata"]["lensApplied"] is True

    def test_the_route_defers_on_a_question_it_does_not_own(self, frames):
        assert ask("What is the current funded balance?") is None

    def test_the_governed_result_travels_intact_beside_the_envelope(self, frames):
        out = ask("What changed in the portfolio this month?")
        payload = out["periodChange"]
        assert payload["workflow_id"] == m.WORKFLOW_ID
        assert payload["metric_changes"]
        assert payload["period_resolution"]["resolved_start_snapshot"][
            "reporting_date"] == "2026-05-31"
        assert payload["audit"]["business_semantics"]["registry_version"] == "0.3.0"
        # The pre-existing envelope keys are untouched.
        for key in ("ok", "question", "answer", "spec", "artifacts", "validation",
                    "warnings", "metadata", "sourceNotes"):
            assert key in out

    def test_the_answer_only_restates_calculated_values(self, frames):
        out = ask("What changed in the portfolio this month?")
        summary = out["periodChange"]["summary"]
        assert str(summary["metrics_analysed"]) in out["answer"]
        assert m.MATERIALITY_DISCLAIMER in out["answer"]

    def test_the_artifacts_carry_the_metrics_distributions_and_bridge(self, frames):
        out = ask("What changed in the portfolio this month?")
        titles = {a["title"] for a in out["artifacts"]}
        assert {"Metric movements", "Composition shifts", "Balance bridge"} <= titles
        assert "kpi" in {a["type"] for a in out["artifacts"]}

    def test_the_metric_table_states_units_and_interpretation(self, frames):
        out = ask("What changed in the portfolio this month?")
        table = next(a for a in out["artifacts"] if a["title"] == "Metric movements")
        row = next(r for r in table["rows"]
                   if r["canonical_field"] == "current_loan_to_value")
        assert row["start_value"] == "40.00%"
        assert row["movement"] == "+3.00 pp"
        assert row["interpretation"] in ("deterioration", "improvement",
                                         "not_assessed", "no_movement")

    def test_the_table_states_the_basis_of_every_figure(self, frames):
        """A reader must not have to know the registry to tell a share of loans
        from a share of balance, or a weighted average from a simple one."""
        out = ask("What changed in the portfolio this month?")
        table = next(a for a in out["artifacts"] if a["title"] == "Metric movements")
        assert any(c["key"] == "basis" for c in table["columns"])

        by_field = {r["canonical_field"]: r for r in table["rows"]}
        assert by_field["interest_in_arrears"]["basis"] == "share of loan count"
        assert by_field["current_loan_to_value"]["basis"] == (
            "weighted average by current_outstanding_balance")
        assert by_field["current_outstanding_balance"]["basis"] == "sum"

    def test_the_table_states_the_scope_each_rank_was_made_within(self, frames):
        out = ask("What changed in the portfolio this month?")
        table = next(a for a in out["artifacts"] if a["title"] == "Metric movements")
        assert any(c["key"] == "rank_scope" for c in table["columns"])
        by_field = {r["canonical_field"]: r for r in table["rows"]}
        assert by_field["current_outstanding_balance"]["rank_scope"] == "currency"
        assert by_field["interest_in_arrears"]["rank_scope"] == "percentage_point"

    def test_the_answer_states_which_unit_each_ranking_covers(self, frames):
        """Presenting one "largest movements" list would assert a cross-unit
        comparison the workflow deliberately did not make."""
        out = ask("What changed in the portfolio this month?")
        assert "measured in currency" in out["answer"]
        assert "measured in percentage points" in out["answer"]

    def test_the_balance_bridge_reconciles_in_the_rendered_table(self, frames):
        out = ask("What changed in the portfolio this month?")
        bridge = out["periodChange"]["balance_bridge"]
        assert bridge["status"] == m.BRIDGE_STATUS_AVAILABLE
        assert bridge["reconciles"] is True
        assert bridge["new_loan_count"] == 10

    def test_the_source_notes_name_the_registry_and_the_snapshots(self, frames):
        out = ask("What changed in the portfolio this month?")
        labels = {n["label"] for n in out["sourceNotes"]}
        assert labels == {"Business Semantics Registry", "Governed snapshots"}

    def test_a_requested_metric_question_analyses_only_that_metric(self, frames):
        out = ask("How did current LTV change month-on-month?")
        fields = [c["canonical_field"] for c in out["periodChange"]["metric_changes"]]
        assert fields == ["current_loan_to_value"]
        assert out["metadata"]["periodChangeMode"] == m.MODE_REQUESTED_METRIC

    def test_a_concept_question_analyses_that_concept(self, frames):
        out = ask("What changed in credit quality?")
        assert out["metadata"]["periodChangeMode"] == m.MODE_CONCEPT


# --------------------------------------------------------------------------- #
# Controlled failure
# --------------------------------------------------------------------------- #
class TestControlledFailure:
    def test_one_snapshot_produces_a_controlled_refusal_not_a_guess(
            self, single_frame):
        out = ask("What changed in the portfolio this month?")
        assert out["ok"] is False
        assert out["metadata"]["controlledUnsupported"] is True
        assert out["metadata"]["periodChangeFailure"]["reason"] == \
            m.FAIL_INSUFFICIENT_SNAPSHOTS
        assert out["artifacts"] == []
        assert "two governed portfolio snapshots" in out["answer"]

    def test_a_failure_maps_onto_the_existing_error_taxonomy(self, single_frame):
        from trakt_core.errors import ErrorCode

        out = ask("What changed in the portfolio this month?")
        assert out["metadata"]["errorCode"] == ErrorCode.NO_MATCHING_RECORDS

    def test_every_failure_reason_has_an_error_code(self):
        assert set(pcr.FAILURE_ERROR_CODES) == set(m.FAILURE_MESSAGES)


# --------------------------------------------------------------------------- #
# Direct invocation — the capability is callable without the chat path
# --------------------------------------------------------------------------- #
class TestDirectInvocation:
    def test_the_workflow_can_be_called_directly(self, frames):
        result = pcr.analyse_period_change(
            client_id="client", output_root="blob://x", tenant_id="tenant_a")
        assert result.workflow_id == m.WORKFLOW_ID
        assert result.metric_changes
        assert result.portfolio_scope.tenant_id == "tenant_a"

    def test_a_direct_caller_cannot_widen_beyond_its_authorised_scope(self, frames):
        from mi_agent.portfolio_lens import SOURCE_ID_FIELD, PortfolioLens

        lens = PortfolioLens(name="cohort", label="Other",
                             filters={SOURCE_ID_FIELD: ["not_authorised"]},
                             cohort_id="not_authorised")
        with pytest.raises(m.PeriodChangeFailure) as err:
            pcr.analyse_period_change(
                client_id="client", output_root="blob://x", scope=lens,
                authorised_portfolio_ids=("alp_origination",))
        assert err.value.reason == m.FAIL_CROSS_TENANT_ACCESS


# --------------------------------------------------------------------------- #
# Registry precedence — the incumbents keep every question they answer
# --------------------------------------------------------------------------- #
class TestRegistryPrecedence:
    def test_the_route_is_registered_between_the_composites_and_compare(self):
        from mi_agent_api.recogniser_registry import REGISTRY

        names = list(REGISTRY.names())
        assert names.index("portfolio_summary") < names.index("period_change_analysis")
        assert names.index("period_change_analysis") < names.index("temporal_compare")
        assert names.index("period_movement") < names.index("period_change_analysis")

    def test_the_route_declares_its_business_semantics_dependency(self):
        from mi_agent_api.recogniser_registry import REGISTRY

        recogniser = REGISTRY.get("period_change_analysis")
        assert recogniser.lens_aware is True
        assert recogniser.metadata["business_semantics_workflow_tag"] == "period_change"
        assert recogniser.description

    @pytest.mark.parametrize("question,expected", [
        ("Compare October and November funded balance.", "temporal_compare"),
        ("Compare October and November loan count.", "temporal_compare"),
        ("What has changed versus the prior month?", "period_movement"),
        ("how has the portfolio changed month on month", "period_movement"),
        ("Summarise the portfolio.", "portfolio_summary"),
        ("Show funded balance evolution by month.", "evolution"),
        ("What changed in the portfolio this month?", "period_change_analysis"),
        ("What improved and deteriorated quarter-on-quarter?",
         "period_change_analysis"),
    ])
    def test_the_first_matching_recogniser_is_the_expected_one(self, question,
                                                              expected):
        """Recognition only — no handler is run, so this asserts precedence
        without needing every route's data."""
        from mi_agent.llm_query_parser import _deterministic_parse
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.recogniser_registry import REGISTRY, RouteRequest

        semantics = load_mi_semantics(SEMANTICS_PATH)
        spec, meta = _deterministic_parse(question, semantics)
        request = RouteRequest(
            question=question, spec=spec, spec_dict=spec.to_dict(),
            semantics=semantics, view="funded", client_id="client", run_id=None,
            portfolio_id=None, parse_meta=meta)
        candidates = [r.name for r, _ in REGISTRY.candidates(request)]
        assert candidates and candidates[0] == expected, candidates


# --------------------------------------------------------------------------- #
# Governed query integration
# --------------------------------------------------------------------------- #
def test_the_route_is_run_scoped_because_it_compares_two_snapshots():
    from mi_agent_api import mi_service

    assert mi_service._route_requires_run("period_change_analysis") is True


def test_try_route_dispatches_to_the_workflow(frames, monkeypatch):
    """The whole path: question → single parse → registry → workflow → envelope."""
    from mi_agent.mi_query_validator import load_mi_semantics

    semantics = load_mi_semantics(SEMANTICS_PATH)
    out = chat_routing.try_route(
        "What changed in the portfolio this month?",
        portfolio_id="client/2026-06-30", view="funded",
        output_root="blob://x", pipeline_root=None, semantics=semantics)
    assert out is not None
    assert out["metadata"]["route"] == "period_change_analysis"
    assert out["periodChange"]["workflow_id"] == m.WORKFLOW_ID


# --------------------------------------------------------------------------- #
# Against real on-disk governed runs (no monkeypatching of the frame service)
# --------------------------------------------------------------------------- #
def _write_run(root, run_id: str, reporting_date: str, *, n: int, balance: float,
               ltv: float, arrears: int, first_loan: int) -> None:
    frame = _frame(n=n, balance=balance, ltv=ltv, arrears=arrears,
                   region_split={"South East": 0.6, "London": 0.4},
                   first_loan=first_loan)
    frame["reporting_date"] = reporting_date
    directory = root / "client_001" / run_id / "output" / "central"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "18_central_lender_tape.csv", index=False)


@pytest.fixture
def on_disk_runs(tmp_path):
    """Two governed monthly runs on disk, discovered by the real loader."""
    root = tmp_path / "onboarding_output"
    _write_run(root, "mi_2026_05", "2026-05-31", n=40, balance=800_000.0,
               ltv=0.38, arrears=2, first_loan=0)
    _write_run(root, "mi_2026_06", "2026-06-30", n=44, balance=920_000.0,
               ltv=0.41, arrears=6, first_loan=0)
    return root


def test_the_workflow_runs_against_real_governed_runs_on_disk(on_disk_runs):
    """End to end through snapshot discovery, the tape loader and the funded
    preparation the rest of the MI product uses."""
    result = pcr.analyse_period_change(
        client_id="client_001", output_root=str(on_disk_runs), tenant_id="tenant_a")

    resolution = result.period_resolution
    assert resolution.start_snapshot.snapshot_id == "mi_2026_05"
    assert resolution.end_snapshot.snapshot_id == "mi_2026_06"
    assert resolution.start_snapshot.dataset_label == "18_central_lender_tape.csv"

    balance = next(c for c in result.metric_changes
                   if c.field == "current_outstanding_balance")
    assert balance.start_value == pytest.approx(800_000.0, rel=1e-6)
    assert balance.end_value == pytest.approx(920_000.0, rel=1e-6)
    assert balance.movement_value == pytest.approx(120_000.0, rel=1e-6)

    arrears = next(c for c in result.metric_changes
                   if c.field == "interest_in_arrears")
    assert arrears.start_value == pytest.approx(5.0)      # 2 of 40
    assert arrears.end_value == pytest.approx(6 / 44 * 100)
    assert arrears.interpretation == m.INTERPRETATION_DETERIORATION

    assert result.balance_bridge.status == m.BRIDGE_STATUS_AVAILABLE
    assert result.balance_bridge.reconciles is True
    assert result.balance_bridge.new_loan_count == 4


def test_a_narrow_metric_question_does_not_reconcile_the_whole_book(frames):
    """The bridge answers "what drove the movement?", not "how did LTV change?"."""
    narrow = ask("How did current LTV change month-on-month?")
    assert narrow["periodChange"]["balance_bridge"] is None

    drivers = ask("What are the main drivers of the balance movement?")
    assert drivers["periodChange"]["balance_bridge"]["status"] == \
        m.BRIDGE_STATUS_AVAILABLE
