"""Negative controls for the broad scorer.

The scorer's whole claim is that it catches what a bank replay cannot: a wrong
population, a lost filter, a wrong aggregation, a breakdown that went missing,
an unexpected refusal, a paraphrase that disagrees, a narrowing that widens.
A scorer that made that claim and did not check it would be the same defect it
was written to fix, one layer up.

So each test here MUTATES a well-formed envelope in exactly one of those ways
and asserts the scorer moves from ok to FAIL. The unmutated envelope is asserted
to pass in the same test, which is what stops a scorer that fails everything
from looking like a scorer that works.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from due_diligence.evidence.mi_api_certification import broad


def _kpi(population: int = 36, balance: float = 5_382_462.92,
         filters: Dict[str, Any] | None = None) -> Dict[str, Any]:
    filters = filters or {}
    return {
        "ok": True,
        "answer": f"Balance: £5.4MM · {population} loans.",
        "spec": {"metric": "current_outstanding_balance", "aggregation": "sum",
                 "dimensions": [], "filters": filters, "measures": []},
        "executionSummary": {"measure": "Balance", "aggregation": "sum",
                             "population": population, "populationTotal": 36,
                             "groupCount": None},
        "reconciliation": {"total_records": 36, "records_included": population,
                           "balance_included": balance},
        "validation": {"ok": True},
        "filterInvariant": {"ok": True, "applied_filters": list(filters),
                            "rejected_filters": [], "unavailable_filters": [],
                            "dropped": []},
        "dimensionInvariant": {"ok": True, "applied": [], "rejected": [],
                               "dropped": []},
        "artifacts": [{"type": "kpi", "kpis": [
            {"field": "loan_count", "rawValue": float(population)},
            {"field": "current_outstanding_balance_sum", "rawValue": balance},
        ]}],
    }


def _grouped(rows: List[Dict[str, Any]] | None = None,
             dimensions: List[str] | None = None,
             population: int = 36) -> Dict[str, Any]:
    dimensions = dimensions or ["collateral_geography"]
    rows = rows if rows is not None else [
        {"collateral_geography": "London", "current_outstanding_balance_sum": 3_000_000.0,
         "loan_count": 20, "concentration_pct": 55.74},
        {"collateral_geography": "Scotland", "current_outstanding_balance_sum": 2_382_462.92,
         "loan_count": 16, "concentration_pct": 44.26},
    ]
    return {
        "ok": True, "answer": "Here is the bar for your query, covering 2 groups.",
        "spec": {"metric": "current_outstanding_balance", "aggregation": "sum",
                 "dimensions": dimensions, "filters": {}, "measures": []},
        "executionSummary": {"measure": "Balance", "aggregation": "sum",
                             "population": population, "populationTotal": 36,
                             "groupCount": len(rows)},
        "reconciliation": {"total_records": 36, "records_included": population,
                           "balance_included": 5_382_462.92},
        "validation": {"ok": True},
        "filterInvariant": {"ok": True, "applied_filters": [],
                            "rejected_filters": [], "unavailable_filters": [],
                            "dropped": []},
        "dimensionInvariant": {"ok": True, "applied": dimensions, "rejected": [],
                               "dropped": []},
        "artifacts": [{"type": "table", "rows": rows,
                       "valueKey": "current_outstanding_balance_sum",
                       "columns": [{"key": d, "format": "text"} for d in dimensions]}],
    }


def _refusal(text: str = "That term is not governed.") -> Dict[str, Any]:
    return {"ok": False, "answer": text, "error": text,
            "spec": {"filters": {}, "dimensions": [], "measures": []},
            "metadata": {"controlledRefusal": True}}


def _session(book: Dict[str, Dict[str, Any]]) -> broad.Session:
    return broad.Session(lambda q: copy.deepcopy(book[q]))


# --------------------------------------------------------------------------- #
class TestCoherenceConvictsAConfidentAnswer:
    def test_a_clean_answer_is_coherent(self) -> None:
        assert broad.coherence(broad.read_evidence("q", _kpi())) == []

    def test_a_population_larger_than_the_book_is_caught(self) -> None:
        envelope = _kpi(population=40)
        envelope["executionSummary"]["populationTotal"] = 36
        problems = broad.coherence(broad.read_evidence("q", envelope))
        assert any("exceeds book" in p for p in problems), problems

    def test_a_filter_dropped_while_answering_is_caught(self) -> None:
        envelope = _kpi(filters={"collateral_geography": "Scotland"})
        envelope["filterInvariant"]["dropped"] = ["collateral_geography"]
        problems = broad.coherence(broad.read_evidence("q", envelope))
        assert any("dropping filters" in p for p in problems), problems

    def test_a_dimension_dropped_while_answering_is_caught(self) -> None:
        envelope = _grouped()
        envelope["dimensionInvariant"]["rejected"] = ["ltv_bucket"]
        problems = broad.coherence(broad.read_evidence("q", envelope))
        assert any("dropping dimensions" in p for p in problems), problems

    def test_group_counts_that_do_not_add_to_the_population_are_caught(self) -> None:
        envelope = _grouped()
        envelope["artifacts"][0]["rows"][0]["loan_count"] = 5   # was 20
        problems = broad.coherence(broad.read_evidence("q", envelope))
        assert any("group loan counts total" in p for p in problems), problems

    def test_a_refusal_is_never_convicted(self) -> None:
        assert broad.coherence(broad.read_evidence("q", _refusal())) == []


class TestTheSafetyClassIsScored:
    def test_an_answered_must_refuse_is_a_silent_wrong_answer(self) -> None:
        session = _session({"What is the platinum balance?": _kpi()})
        result = broad.score_single(session, {
            "id": "J01", "cls": "J_adversarial",
            "q": "What is the platinum balance?", "expect": "MUST_REFUSE"})
        assert result.status == broad.FAIL and result.silent_wrong

    def test_a_refused_must_refuse_passes(self) -> None:
        session = _session({"What is the platinum balance?": _refusal()})
        result = broad.score_single(session, {
            "id": "J01", "cls": "J_adversarial",
            "q": "What is the platinum balance?", "expect": "MUST_REFUSE"})
        assert result.status == broad.OK

    def test_a_data_refusal_is_reported_not_failed(self) -> None:
        session = _session({"WA LTV": _refusal(
            "Current LTV is not available in this dataset.")})
        result = broad.score_single(session, {
            "id": "A09", "cls": "A_basic_measure", "q": "WA LTV",
            "expect": "MUST_ANSWER"})
        assert result.status == broad.DATA

    def test_a_semantic_refusal_of_a_must_answer_fails(self) -> None:
        session = _session({"What is the total balance?": _refusal()})
        result = broad.score_single(session, {
            "id": "A01", "cls": "A_basic_measure",
            "q": "What is the total balance?", "expect": "MUST_ANSWER"})
        assert result.status == broad.FAIL


class TestEquivalenceScoresTheRelation:
    QUESTIONS = ["What is the total balance in Scotland?",
                 "What is the Scottish balance?"]

    def _case(self) -> Dict[str, Any]:
        return {"id": "EQ03", "cls": "F_equivalence", "relation": "EQUAL_TO",
                "questions": list(self.QUESTIONS)}

    def test_agreeing_wordings_pass(self) -> None:
        session = _session({q: _kpi(population=2, balance=388_146.06)
                            for q in self.QUESTIONS})
        assert broad.score_equivalence(session, self._case()).status == broad.OK

    def test_a_different_population_is_caught(self) -> None:
        session = _session({
            self.QUESTIONS[0]: _kpi(population=2, balance=388_146.06),
            self.QUESTIONS[1]: _kpi(population=36, balance=5_382_462.92)})
        result = broad.score_equivalence(session, self._case())
        assert result.status == broad.FAIL and result.silent_wrong

    def test_the_same_population_with_a_different_figure_is_caught(self) -> None:
        session = _session({
            self.QUESTIONS[0]: _kpi(population=2, balance=388_146.06),
            self.QUESTIONS[1]: _kpi(population=2, balance=400_000.00)})
        result = broad.score_equivalence(session, self._case())
        assert result.status == broad.FAIL and result.silent_wrong

    def test_one_wording_refusing_and_the_other_answering_is_caught(self) -> None:
        session = _session({self.QUESTIONS[0]: _kpi(population=2),
                            self.QUESTIONS[1]: _refusal()})
        result = broad.score_equivalence(session, self._case())
        assert result.status == broad.FAIL and result.silent_wrong

    def test_both_refusing_is_a_data_report_not_a_failure(self) -> None:
        session = _session({q: _refusal() for q in self.QUESTIONS})
        assert broad.score_equivalence(session, self._case()).status == broad.DATA

    def test_agreeing_totals_that_disagree_per_group_are_caught(self) -> None:
        left = _grouped()
        right = _grouped(rows=[
            {"collateral_geography": "London",
             "current_outstanding_balance_sum": 2_382_462.92, "loan_count": 16,
             "concentration_pct": 44.26},
            {"collateral_geography": "Scotland",
             "current_outstanding_balance_sum": 3_000_000.0, "loan_count": 20,
             "concentration_pct": 55.74}])
        session = _session({"Total balance by region": left,
                            "Chart balance by region": right})
        result = broad.score_equivalence(session, {
            "id": "EQ05", "cls": "F_equivalence", "relation": "SAME_RESULT_AS",
            "questions": ["Total balance by region", "Chart balance by region"]})
        assert result.status == broad.FAIL and "not per group" in result.detail


class TestNarrowingIsScored:
    def test_a_narrowing_that_widens_is_caught(self) -> None:
        session = _session({
            "What is the total balance?": _kpi(population=36),
            "What is the total balance in Scotland?": _kpi(population=40)})
        result = broad.score_subset(session, {
            "id": "SB01", "cls": "G_subset",
            "whole": "What is the total balance?",
            "part": "What is the total balance in Scotland?"})
        assert result.status == broad.FAIL and result.silent_wrong

    def test_a_narrowed_figure_larger_than_the_whole_is_caught(self) -> None:
        session = _session({
            "What is the total balance?": _kpi(population=36, balance=5_382_462.92),
            "What is the total balance in Scotland?": _kpi(
                population=2, balance=9_000_000.0)})
        result = broad.score_subset(session, {
            "id": "SB01", "cls": "G_subset",
            "whole": "What is the total balance?",
            "part": "What is the total balance in Scotland?"})
        assert result.status == broad.FAIL and result.silent_wrong

    def test_an_honest_narrowing_passes(self) -> None:
        session = _session({
            "What is the total balance?": _kpi(population=36, balance=5_382_462.92),
            "What is the total balance in Scotland?": _kpi(
                population=2, balance=388_146.06)})
        result = broad.score_subset(session, {
            "id": "SB01", "cls": "G_subset",
            "whole": "What is the total balance?",
            "part": "What is the total balance in Scotland?"})
        assert result.status == broad.OK


class TestOutputLocalNarrowing:
    CASE = {"id": "H01", "cls": "H_output_local",
            "broad": "Total balance by region",
            "narrowed": "Total balance by region for lump sum loans"}

    def test_a_narrowing_that_eats_the_breakdown_is_caught(self) -> None:
        flattened = _kpi(population=10, filters={"erm_product_type": "Lump Sum"})
        session = _session({self.CASE["broad"]: _grouped(),
                            self.CASE["narrowed"]: flattened})
        result = broad.score_output_local(session, self.CASE)
        assert result.status == broad.FAIL and "lost the breakdown" in result.detail

    def test_a_narrowing_that_was_silently_dropped_is_caught(self) -> None:
        ignored = _grouped(population=36)          # same population, no filter
        session = _session({self.CASE["broad"]: _grouped(),
                            self.CASE["narrowed"]: ignored})
        result = broad.score_output_local(session, self.CASE)
        assert result.status == broad.FAIL and "narrowing was dropped" in result.detail

    def test_a_real_output_local_narrowing_passes(self) -> None:
        narrowed = _grouped(rows=[
            {"collateral_geography": "London",
             "current_outstanding_balance_sum": 1_000_000.0, "loan_count": 6,
             "concentration_pct": 60.0},
            {"collateral_geography": "Scotland",
             "current_outstanding_balance_sum": 600_000.0, "loan_count": 4,
             "concentration_pct": 40.0}], population=10)
        narrowed["spec"]["filters"] = {"erm_product_type": "Lump Sum"}
        narrowed["filterInvariant"]["applied_filters"] = ["erm_product_type"]
        session = _session({self.CASE["broad"]: _grouped(),
                            self.CASE["narrowed"]: narrowed})
        assert broad.score_output_local(session, self.CASE).status == broad.OK


class TestMultiOutputCompleteness:
    CASE = {"id": "G01", "cls": "G_multi_output",
            "q": "Show total balance and loan count by region",
            "expect_outputs": ["balance", "loan count"]}

    def test_a_lost_output_is_caught(self) -> None:
        only_balance = _grouped(rows=[
            {"collateral_geography": "London",
             "current_outstanding_balance_sum": 3_000_000.0},
            {"collateral_geography": "Scotland",
             "current_outstanding_balance_sum": 2_382_462.92}])
        session = _session({self.CASE["q"]: only_balance})
        result = broad.score_outputs(session, self.CASE)
        assert result.status == broad.FAIL and result.silent_wrong

    def test_both_outputs_present_passes(self) -> None:
        session = _session({self.CASE["q"]: _grouped()})
        assert broad.score_outputs(session, self.CASE).status == broad.OK


class TestNumericIdentities:
    CASE = {"id": "RC01", "cls": "I_numeric_identity",
            "total": "What is the total balance?",
            "grouped": "Total balance by region"}

    def test_groups_that_do_not_sum_to_the_whole_are_caught(self) -> None:
        short = _grouped(rows=[
            {"collateral_geography": "London",
             "current_outstanding_balance_sum": 3_000_000.0, "loan_count": 20,
             "concentration_pct": 55.74}], population=36)
        session = _session({self.CASE["total"]: _kpi(),
                            self.CASE["grouped"]: short})
        result = broad.score_reconciliation(session, self.CASE)
        assert result.status == broad.FAIL and result.silent_wrong

    def test_a_partition_that_reconciles_passes(self) -> None:
        session = _session({self.CASE["total"]: _kpi(),
                            self.CASE["grouped"]: _grouped()})
        assert broad.score_reconciliation(session, self.CASE).status == broad.OK

    def test_a_non_additive_aggregation_is_not_established(self) -> None:
        averaged = _grouped()
        averaged["executionSummary"]["aggregation"] = "avg"
        session = _session({self.CASE["total"]: _kpi(),
                            self.CASE["grouped"]: averaged})
        result = broad.score_reconciliation(session, self.CASE)
        assert result.status == broad.NOT_ESTABLISHED

    def test_shares_that_do_not_sum_to_a_hundred_are_caught(self) -> None:
        skewed = _grouped()
        skewed["artifacts"][0]["rows"][0]["concentration_pct"] = 20.0
        session = _session({"Total balance by region": skewed})
        result = broad.score_arithmetic(session, {
            "id": "AR03", "cls": "I_numeric_identity",
            "identity": "group shares sum to the whole",
            "grouped": "Total balance by region"})
        assert result.status == broad.FAIL


class TestDimensionOrderIsScoredWithoutAssertingAnAxis:
    CASE = {"id": "AL03", "cls": "H_filter_algebra", "relation": "SAME_CELLS_AS",
            "questions": ["Total balance by region and LTV band",
                          "Total balance by LTV band and region"]}

    def _two_d(self, swap: bool = False, corrupt: bool = False) -> Dict[str, Any]:
        rows = [{"collateral_geography": "London", "ltv_bucket": "40-50%",
                 "current_outstanding_balance_sum": 426_077.91},
                {"collateral_geography": "Scotland", "ltv_bucket": "30-40%",
                 "current_outstanding_balance_sum": 200_000.00}]
        if corrupt:
            rows[0]["current_outstanding_balance_sum"] = 1.0
        dimensions = ["ltv_bucket", "collateral_geography"] if swap else \
            ["collateral_geography", "ltv_bucket"]
        return _grouped(rows=rows, dimensions=dimensions)

    def test_the_same_cells_in_either_order_pass(self) -> None:
        session = _session({self.CASE["questions"][0]: self._two_d(),
                            self.CASE["questions"][1]: self._two_d(swap=True)})
        assert broad.score_algebra(session, self.CASE).status == broad.OK

    def test_reordering_that_changes_a_cell_is_caught(self) -> None:
        session = _session({
            self.CASE["questions"][0]: self._two_d(),
            self.CASE["questions"][1]: self._two_d(swap=True, corrupt=True)})
        result = broad.score_algebra(session, self.CASE)
        assert result.status == broad.FAIL and result.silent_wrong


class TestTheScoredBankIsNotACounter:
    """The §1 defect, stated as a test.

    The old `run_bank` returned answered / refused / transport-failed and `main`
    failed only on transport. These two cases differ ONLY in whether the answers
    are internally coherent — the old counter could not tell them apart.
    """

    def _bank(self, tmp_path: Any, questions: List[str]) -> str:
        import json as _json
        path = tmp_path / "bank.json"
        path.write_text(_json.dumps(questions), encoding="utf-8")
        return str(path)

    def test_a_bank_of_incoherent_answers_is_scored_as_incoherent(
            self, tmp_path: Any) -> None:
        from due_diligence.evidence.mi_api_certification import certify_mi_api

        broken = _kpi(population=40)                  # 40 loans in a 36-loan book
        path = self._bank(tmp_path, ["q1", "q2"])
        answered, refused, transport, incoherent, _ = certify_mi_api.run_bank(
            lambda q: copy.deepcopy(broken), path)
        assert (answered, refused, transport) == (2, 0, 0)
        assert incoherent == 2

    def test_a_bank_of_coherent_answers_is_clean(self, tmp_path: Any) -> None:
        from due_diligence.evidence.mi_api_certification import certify_mi_api

        path = self._bank(tmp_path, ["q1", "q2"])
        answered, refused, transport, incoherent, _ = certify_mi_api.run_bank(
            lambda q: _kpi(), path)
        assert (answered, refused, transport, incoherent) == (2, 0, 0, 0)


class TestTransportIsNeverScoredAsAModelVerdict:
    def test_a_transport_failure_fails_the_case_as_transport(self) -> None:
        session = _session({"What is the total balance?": {
            "ok": False, "answer": "HTTP 502", "__transport_error__": True,
            "__http_status__": 502}})
        result = broad.score_single(session, {
            "id": "A01", "cls": "A_basic_measure",
            "q": "What is the total balance?", "expect": "MUST_ANSWER"})
        assert result.status == broad.FAIL
        assert "[transport]" in result.detail and not result.silent_wrong
        assert session.transport_failures and session.statuses == [502]


class TestTheHarnessDefectsFoundInTheDryRun:
    """Two ways the FIRST cut of this scorer was wrong, kept as controls.

    Both were found by running the suite before it was trusted, and both are the
    same species of mistake in opposite directions: one reported NOT ESTABLISHED
    where the evidence was present, the other reported FAIL where the response
    was honest. A harness gets the same treatment as the product it scores.
    """

    def test_a_two_measure_breakdown_yields_comparable_cells(self) -> None:
        """The table artefact publishes no `valueKey`; reading one found none."""
        envelope = _grouped()
        assert "valueKey" in envelope["artifacts"][0]
        del envelope["artifacts"][0]["valueKey"]
        evidence = broad.read_evidence("Total balance by region", envelope)
        assert evidence.cells is not None and len(evidence.cells) == 2

    def test_cells_disagreeing_on_a_second_measure_are_caught(self) -> None:
        left = _grouped()
        right = _grouped()
        right["artifacts"][0]["rows"][0]["loan_count"] = 19   # balance still agrees
        right["artifacts"][0]["rows"][1]["loan_count"] = 17
        session = _session({"Loan count by region": left,
                            "Number of loans by region": right})
        result = broad.score_equivalence(session, {
            "id": "EQ16", "cls": "F_equivalence", "relation": "SAME_RESULT_AS",
            "questions": ["Loan count by region", "Number of loans by region"]})
        assert result.status == broad.FAIL and "not per group" in result.detail

    def test_a_top_n_breakdown_is_not_convicted_for_showing_part(self) -> None:
        top = _grouped(rows=[{"collateral_geography": "London",
                              "current_outstanding_balance_sum": 3_000_000.0,
                              "loan_count": 20, "concentration_pct": 55.74}])
        top["spec"]["top_n"] = 3
        assert broad.coherence(broad.read_evidence("q", top)) == []

    def test_an_untruncated_breakdown_that_does_not_add_up_is_still_convicted(
            self) -> None:
        short = _grouped(rows=[{"collateral_geography": "London",
                                "current_outstanding_balance_sum": 3_000_000.0,
                                "loan_count": 20, "concentration_pct": 55.74}])
        problems = broad.coherence(broad.read_evidence("q", short))
        assert any("group loan counts total" in p for p in problems), problems

    def test_a_collapsed_tail_is_not_convicted_by_the_share_identity(self) -> None:
        collapsed = _grouped()
        collapsed["artifacts"][0]["otherCategories"] = ["the rest"]
        collapsed["artifacts"][0]["rows"][0]["concentration_pct"] = 20.0
        session = _session({"Total balance by region": collapsed})
        result = broad.score_arithmetic(session, {
            "id": "AR03", "cls": "I_numeric_identity",
            "identity": "group shares sum to the whole",
            "grouped": "Total balance by region"})
        assert result.status == broad.NOT_ESTABLISHED


class TestDeployedBuildProvenance:
    """`version=1.0.0` is not provenance, and the harness must not accept it.

    The live certification reported `deployed commit : version=1.0.0` for every
    run, because the reader fell through to the application's hand-written
    version string when no commit was published. That string has not changed
    this year, so it could not distinguish the release being certified from an
    older build that was never replaced. These assert that the reader now takes
    the immutable build stamp or reports nothing at all.
    """

    def test_an_unstamped_build_reports_no_commit(self, monkeypatch) -> None:
        from mi_agent_api import build_info as module

        module.build_info.cache_clear()
        monkeypatch.delenv("TRAKT_BUILD_COMMIT", raising=False)
        monkeypatch.setattr(module, "_STAMP",
                            module.Path("/nonexistent/build_info.json"))
        info = module.build_info()
        module.build_info.cache_clear()
        assert info["commit"] is None and info["source"] == "unstamped"

    def test_a_stamped_build_reports_its_commit(self, monkeypatch) -> None:
        from mi_agent_api import build_info as module

        module.build_info.cache_clear()
        monkeypatch.setenv("TRAKT_BUILD_COMMIT", "0" * 40)
        info = module.build_info()
        module.build_info.cache_clear()
        assert info["commit"] == "0" * 40 and info["source"] == "environment"

    def test_a_version_string_is_never_read_as_a_commit(self) -> None:
        """The exact regression: a health payload carrying only `version`."""
        import json as _json
        from unittest import mock

        from due_diligence.evidence.mi_api_certification import certify_mi_api

        payload = _json.dumps({"ok": True, "version": "1.0.0",
                               "service": "mi_agent_api"}).encode()

        class _Response:
            def read(self):
                return payload

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        with mock.patch.object(certify_mi_api, "_live_asker",
                               return_value=lambda q: {"ok": True}), \
             mock.patch("urllib.request.urlopen", return_value=_Response()):
            _reached, _auth, commit = certify_mi_api.preflight(
                "https://example.invalid", "/mi/query", [], None)
        assert commit is None, (
            f"a version string was read as provenance: {commit!r}")

    def test_a_published_stamp_is_read_as_the_commit(self) -> None:
        import json as _json
        from unittest import mock

        from due_diligence.evidence.mi_api_certification import certify_mi_api

        payload = _json.dumps({"ok": True, "version": "1.0.0",
                               "build": {"commit": "a" * 40,
                                         "source": "artefact"}}).encode()

        class _Response:
            def read(self):
                return payload

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        with mock.patch.object(certify_mi_api, "_live_asker",
                               return_value=lambda q: {"ok": True}), \
             mock.patch("urllib.request.urlopen", return_value=_Response()):
            _reached, _auth, commit = certify_mi_api.preflight(
                "https://example.invalid", "/mi/query", [], None)
        assert commit == "a" * 40
