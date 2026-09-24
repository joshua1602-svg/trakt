"""The property extract joined none of the 568 loans it describes.

ERE's live tape was built from three files. The loan extract supplied the
loans; the property extract and the principal-and-interest file were meant to
add valuations, postcodes, dates of birth, protected equity and cash flows.
The published tape carried none of it. The builder had chosen the property
extract's key column by name, and the word "ID" won: `Originator ID`, which
holds ONE value for all 568 rows. Nothing matched, and every property field
arrived empty on a tape that otherwise looked complete.

A supporting file's key is now checked against the loans once they are known.
When it matches few of them, the column (and number format — ``76034101`` vs
``760341``) that matches the most is used instead, with the evidence recorded.
A near-constant column is never a key.
"""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pytest

from engine.onboarding_agent import central_tape_builder as ctb
from engine.onboarding_agent import entity_key_resolver as ekr

N = 30
FULL = [f"7603{i:02d}01" for i in range(N)]
SHORT = [int(f"7603{i:02d}") for i in range(N)]


def _jk(v, rule):
    return ekr.normalise_key(v, rule) if rule else ctb._norm_key(v)


class TestChoosingTheKeyThatJoins:

    def test_a_constant_column_is_never_a_key(self):
        df = pd.DataFrame({"Originator ID": ["ERE"] * N})
        assert ctb._best_joining_key(df, {"ERE"}, _jk) is None

    def test_the_short_form_of_a_policy_number_is_found(self):
        universe = {ekr.normalise_key(v, "strip_trailing_01") for v in FULL}
        df = pd.DataFrame({"Originator ID": ["ERE"] * N, "Ref": SHORT})
        col, _file_rule, _loan_rule, hits = ctb._best_joining_key(df, universe, _jk)
        assert col == "Ref" and hits == N

    def test_long_and_short_forms_meet(self):
        """Loans keyed 76030101, file holding 760301: every one matches,
        including those whose short form itself ends in 01."""
        universe = {ctb._norm_key(v) for v in FULL}
        df = pd.DataFrame({"Ref": SHORT})
        col, _f, _l, hits = ctb._best_joining_key(df, universe, _jk)
        assert col == "Ref" and hits == N

    def test_a_column_that_matches_few_loans_is_not_chosen(self):
        universe = {str(i) for i in range(1000, 1000 + N)}
        df = pd.DataFrame({"Ref": [str(i) for i in range(5000, 5000 + N)]})
        assert ctb._best_joining_key(df, universe, _jk) is None


def _build(tmp: Path, monkeypatch, *, force_bad_key: bool):
    from engine.onboarding_agent import storage_paths
    from engine.onboarding_agent import workflow as wf
    warnings.simplefilter("ignore")
    inp = tmp / "in"
    inp.mkdir()
    pd.DataFrame({"Loan Policy Number": FULL,
                  "Current Outstanding Balance": [1000.0 + i for i in range(N)],
                  "Current Interest Rate": [5.0] * N,
                  "Policy Completion Date": ["2019-01-01"] * N}).to_csv(
        inp / "LoanExtract One - OMNI 2026_09_01.csv", index=False)
    pd.DataFrame({"Originator ID": ["ERE"] * N, "Ref": SHORT,
                  "Latest Valuation": [300000.0 + i for i in range(N)]}).to_csv(
        inp / "PropertyExtract - Omni 2026_09_01.csv", index=False)
    proj = tmp / "proj"
    wf.run_operator_workflow(
        input_dir=str(inp), client_name="T", client_id="t", run_id="run",
        mode="mi_only", project_dir=str(proj), reporting_date="2026-08",
        confirmed_mappings=[("Loan Policy Number", "loan_identifier"),
                            ("Latest Valuation", "current_valuation_amount")])
    if force_bad_key:
        # What happened on ERE: the name-driven choice lands on Originator ID
        # and the cross-file resolution offers nothing better.
        real = ctb._resolve_key_column
        monkeypatch.setattr(ctb, "_resolve_key_column",
                            lambda df, *a, **k: ("Originator ID"
                                                 if "Originator ID" in df.columns
                                                 else real(df, *a, **k)))
        monkeypatch.setattr(ekr, "load_resolution", lambda _p: {})
    rp = storage_paths.resolve_run_paths(
        project_dir=str(proj), input_dir=str(inp), output_root=None,
        client_id="t", run_id="run", storage_backend="local", input_uri="",
        output_uri="")
    res = ctb.build_central_tapes(str(proj), rp,
                                  "config/system/fields_registry.yaml",
                                  mode="mi_only")
    tape = pd.read_csv(res["central_lender_tape_path"])
    debug = json.loads(next(proj.rglob("18f_central_universe_debug.json"))
                       .read_text())
    return tape, debug


class TestTheTapeGetsThePropertyFields:

    def test_a_wrong_key_is_replaced_and_the_valuations_arrive(self, monkeypatch):
        tape, debug = _build(Path(tempfile.mkdtemp()), monkeypatch,
                             force_bad_key=True)
        assert int(tape["current_valuation_amount"].notna().sum()) == N
        prop = next(s for s in debug["considered_sources"]
                    if s["source_file"].startswith("PropertyExtract"))
        assert prop["key_column"] == "Ref"


class TestTheProductChartReadsTheEquityReleaseField:
    """`Product Category` reached the tape as `erm_product_type` on all 568
    loans; the chart read only generic names and said "not supplied"."""

    def test_erm_product_type_is_read(self):
        from mi_agent_api import snapshots
        df = pd.DataFrame({"current_outstanding_balance": [1.0, 2.0],
                           "erm_product_type": ["Lifetime", "Drawdown"]})
        series = snapshots._strat_series(df, "product")
        assert list(series) == ["Lifetime", "Drawdown"]
        assert snapshots._strat_columns_present(df, "product")
