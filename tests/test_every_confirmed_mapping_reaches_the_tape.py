"""Fifty-one confirmed mappings; the tape carried twenty-nine of them.

ERE's August delivery was reprocessed after its supporting files learned to
join. Postcode, valuations and ages arrived. Twenty-two columns an operator had
confirmed did not — `Customer 1 DOB`, `Full Redemption Date`, `Product Type`,
`Type`, the cash-flow columns — and nothing said why. Two filters, each
reasonable alone, dropped them after the operator had spoken:

* WHERE A FILE-LESS APPROVAL APPLIES was read from the mapping candidates, "every
  file that carries the column". The candidates leave out any column whose name
  suggests a field outside the run's mode, so `Customer 1 DOB` had no file to
  come from.
* THE MODE'S FIELD SCOPE (and the lender-domain filter) removed fields an
  MI-only run does not ask about — `redemption_date` is regulatory,
  `redemptions_received_in_period` is "cashflow".

The rule now: an operator's confirmation reaches the tape. Where it applies is
read from the files' own columns, less any file the column was set aside in;
and scope decides what Trakt asks about, not whether a confirmed answer counts.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pandas as pd
import yaml

from engine.onboarding_agent import central_tape_builder as ctb

LOAN = "LoanExtract One - OMNI 2026_09_01.xlsx"
PROP = "PropertyExtract - Omni 2026_09_01.xlsx"
INVENTORY = {LOAN: {"file_path": "/x/a.xlsx", "sheet_name": ""},
             PROP: {"file_path": "/x/b.xlsx", "sheet_name": ""}}
COLUMNS = {LOAN: ["Loan Policy Number", "Customer 1 DOB", "Full Redemption Date",
                  "Post Code"],
           PROP: ["Ref", "Type", "Post Code"]}


def _override(column, field):
    return {"source_file": "", "source_column": column, "canonical_field": field,
            "method": "operator_approved", "confidence": 1.0}


def _sources(overrides, *, candidates=(), included=(), set_aside=None,
             columns=COLUMNS):
    return ctb._collect_field_sources(
        list(candidates), {"user_overrides": overrides}, INVENTORY,
        set(included), columns_by_file=columns, set_aside=set_aside)


class TestWhereAFileLessApprovalApplies:

    def test_a_column_the_candidates_left_out_still_has_its_file(self):
        src = _sources([_override("Customer 1 DOB", "borrower_1_DOB")])
        assert [s.file_name for s in src["borrower_1_DOB"]] == [LOAN]
        assert src["borrower_1_DOB"][0].method == "operator_approved"

    def test_a_column_set_aside_in_one_file_comes_from_the_other(self):
        src = _sources([_override("Post Code", "postcode")],
                       set_aside=[[LOAN, "Post Code"]])
        assert [s.file_name for s in src["postcode"]] == [PROP]

    def test_a_set_aside_from_another_month_s_file_still_applies(self):
        src = _sources([_override("Post Code", "postcode")],
                       set_aside=[["LoanExtract One - OMNI 2026_08_01.xlsx",
                                   "Post Code"]])
        assert [s.file_name for s in src["postcode"]] == [PROP]

    def test_a_column_set_aside_everywhere_comes_from_nowhere(self):
        src = _sources([_override("Post Code", "postcode")],
                       set_aside=[["*", "Post Code"]])
        assert "postcode" not in src

    def test_a_set_aside_recorded_on_a_candidate_is_still_honoured(self):
        """Runs made before the run summary recorded its set-asides."""
        cand = {"source_file": LOAN, "source_column": "Post Code",
                "candidate_canonical_field": "", "method": "set_aside_by_operator"}
        src = _sources([_override("Post Code", "postcode")], candidates=[cand])
        assert [s.file_name for s in src["postcode"]] == [PROP]

    def test_without_column_profiles_the_candidates_still_count(self):
        cand = {"source_file": LOAN, "source_column": "Customer 1 DOB",
                "candidate_canonical_field": "borrower_1_DOB", "confidence": 0.9}
        src = _sources([_override("Customer 1 DOB", "borrower_1_DOB")],
                       candidates=[cand], columns={})
        assert src["borrower_1_DOB"][0].file_name == LOAN

    def test_a_column_no_file_carries_yields_nothing(self):
        assert _sources([_override("Nobody Sent This", "borrower_1_DOB")]) == {}


class TestScopeDoesNotOverruleAnOperator:

    def test_a_confirmed_field_outside_the_mode_s_scope_is_kept(self):
        src = _sources([_override("Full Redemption Date", "redemption_date")],
                       included={"current_outstanding_balance"})
        assert [s.file_name for s in src["redemption_date"]] == [LOAN]

    def test_a_merely_proposed_field_outside_scope_is_still_left_out(self):
        cand = {"source_file": LOAN, "source_column": "Full Redemption Date",
                "candidate_canonical_field": "redemption_date", "confidence": 0.9}
        src = _sources([], candidates=[cand],
                       included={"current_outstanding_balance"})
        assert "redemption_date" not in src


def test_the_run_records_what_it_set_aside():
    from engine.onboarding_agent.onboarding_models import OnboardingProject
    import dataclasses
    names = {f.name for f in dataclasses.fields(OnboardingProject)}
    assert "set_aside_columns" in names


class TestOnTheBuiltTape:
    """End to end on an MI-only run, with the operator's approvals written as
    the Operations Control Centre writes them: no file named."""

    def _build(self):
        from engine.onboarding_agent import storage_paths
        from engine.onboarding_agent import workflow as wf
        warnings.simplefilter("ignore")
        tmp = Path(tempfile.mkdtemp())
        inp, proj = tmp / "in", tmp / "proj"
        inp.mkdir()
        n = 6
        ids = [f"7603{i:02d}01" for i in range(n)]
        pd.DataFrame({"Loan Policy Number": ids,
                      "Current Outstanding Balance": [1000.0 + i for i in range(n)],
                      "Current Interest Rate": [5.0] * n,
                      "Policy Completion Date": ["2019-01-01"] * n,
                      "Customer 1 DOB": ["1950-03-04"] * n,
                      "Full Redemption Date": ["2026-08-15"] * n,
                      "Redemptions Received in Period": [10.0] * n}).to_csv(
            inp / "LoanExtract One - OMNI 2026_09_01.csv", index=False)
        wf.run_operator_workflow(
            input_dir=str(inp), client_name="T", client_id="t", run_id="run",
            mode="mi_only", project_dir=str(proj), reporting_date="2026-08",
            confirmed_mappings=[("Loan Policy Number", "loan_identifier")])
        overrides = {"version": 1, "user_overrides": [
            _override("Loan Policy Number", "loan_identifier"),
            _override("Current Outstanding Balance", "current_outstanding_balance"),
            _override("Customer 1 DOB", "borrower_1_DOB"),
            _override("Full Redemption Date", "redemption_date"),
            _override("Redemptions Received in Period",
                      "redemptions_received_in_period")]}
        (proj / "12_approved_mapping_overrides.yaml").write_text(
            yaml.safe_dump(overrides), encoding="utf-8")
        rp = storage_paths.resolve_run_paths(
            project_dir=str(proj), input_dir=str(inp), output_root=None,
            client_id="t", run_id="run", storage_backend="local", input_uri="",
            output_uri="")
        res = ctb.build_central_tapes(str(proj), rp,
                                      "config/system/fields_registry.yaml",
                                      mode="mi_only")
        return pd.read_csv(res["central_lender_tape_path"]), n

    def test_every_confirmed_field_is_on_the_tape_and_filled(self):
        tape, n = self._build()
        for field in ("borrower_1_DOB", "redemption_date",
                      "redemptions_received_in_period"):
            assert field in tape.columns, field
            assert int(tape[field].notna().sum()) == n, field
