"""The loan tape is built from every file in the pack, not from one of them.

REPORTED FROM A LIVE CASE. ERE Funding ships its balances in a separate
principal-and-interest extract: ``C/F Principal Balance`` is in
``Principal And Interest - OMNI 2026_09_01.xlsx``, not in the loan extract. The
operator mapped it, confirmed it, and validation then refused the delivery:

    current_principal_balance: CORE001 affects 0 record(s) (0.0%)
        — materiality BLOCKING

Nought records affected, because CORE001 at nought records is
``"Missing required core_canonical field (column not present)"`` — a schema
absence. The column was never BUILT. The adapter assembled its tape from
``_primary_tape()`` alone, the first file with "loan" in its name, and every
other file's mappings were recorded, promoted into governed rules, and left out
of the delivery they were mapped for.

THE PLATFORM DOES NOT WORK THAT WAY AND NEVER DID.
``engine.onboarding_agent.central_tape_builder`` consolidates a loan-domain
field "even when its authoritative source is the cashflow extract, because
domain membership follows the canonical field, not the file" — and names
``current_principal_balance`` as its example. So the rehearsal was refusing a
delivery the platform would have accepted, which is the one thing it must never
do. That is the defect class
``test_the_agent_refuses_only_what_the_platform_refuses`` exists for, arriving
in the tape ASSEMBLY rather than in a check.

WHAT THESE TESTS PIN, and why each rule is the safe direction:

* The primary file is the SPINE. A secondary file fills a column; it never adds
  a row. A loan that appears only in the cashflow extract is a reconciliation
  question, not a loan.
* The primary WINS a contested field and a secondary fills only blanks, which
  is the loan-domain precedence ``_order_sources`` already applies. So
  consolidation can add facts and never overwrite one.
* A file with no mapped loan identifier contributes NOTHING and says so.
  Joining on row order would attach one borrower's balance to another's loan,
  silently, and a tape is not a spreadsheet where that is visible.
* A file with REPEATED KEYS — the ordinary shape of a per-period extract — is
  collapsed to the latest reporting period, or contributes nothing. Never an
  arbitrary row: the fan-out makes every downstream count and average wrong in
  a way that looks like data rather than like a bug.
"""

from __future__ import annotations

import pandas as pd
import pytest

from operations_control.occ_agent.execution import (
    LOAN_KEY,
    PERIOD_FIELD,
    consolidate_pack,
)

PRIMARY = "LoanExtract One - OMNI 2026_09_01.xlsx"
CASHFLOW = "Principal And Interest - OMNI 2026_09_01.xlsx"
PROPERTY = "PropertyExtract - Omni 2026_09_01.xlsx"


def _pack():
    """The shape of the reported case: identity in one file, balance in another."""
    frames = {
        PRIMARY: pd.DataFrame({
            "Loan Ref": ["L1", "L2", "L3"],
            "Rate": [4.5, 5.0, 5.5],
        }),
        CASHFLOW: pd.DataFrame({
            "Account": ["L1", "L2", "L3"],
            "C/F Principal Balance": [100.0, 200.0, 300.0],
        }),
    }
    resolved = {
        PRIMARY: {"Loan Ref": LOAN_KEY, "Rate": "current_interest_rate"},
        CASHFLOW: {"Account": LOAN_KEY,
                   "C/F Principal Balance": "current_principal_balance"},
    }
    return frames, resolved


class TestTheReportedCase:
    def test_the_balance_reaches_the_tape(self):
        """THE DEFECT. Mapped, confirmed, and absent from the delivery."""
        frames, resolved = _pack()
        tape, _ = consolidate_pack(frames, resolved, PRIMARY)
        assert "current_principal_balance" in tape.columns
        assert list(tape["current_principal_balance"]) == [100.0, 200.0, 300.0]

    def test_it_is_joined_on_the_loan_and_not_on_row_order(self):
        """The rows are deliberately in a different order in each file.

        Row-order alignment would pass the test above and put one borrower's
        balance against another's loan — the failure that looks like data.
        """
        frames, resolved = _pack()
        frames[CASHFLOW] = frames[CASHFLOW].iloc[::-1].reset_index(drop=True)
        tape, _ = consolidate_pack(frames, resolved, PRIMARY)
        assert dict(zip(tape[LOAN_KEY], tape["current_principal_balance"])) \
            == {"L1": 100.0, "L2": 200.0, "L3": 300.0}

    def test_the_tape_says_where_the_column_came_from(self):
        """"Where did this balance come from?" is the first question an
        approver asks about a tape built from four files."""
        frames, resolved = _pack()
        _, report = consolidate_pack(frames, resolved, PRIMARY)
        assert report["added"]["current_principal_balance"] == CASHFLOW
        joined = next(f for f in report["files"]
                      if f["source_file"] == CASHFLOW)
        assert joined["joined"] is True


class TestTheSpineDecidesWhichLoansExist:
    def test_a_secondary_file_never_adds_a_row(self):
        """A loan only the cashflow extract knows about is a reconciliation
        question, not a loan."""
        frames, resolved = _pack()
        frames[CASHFLOW] = pd.DataFrame({
            "Account": ["L1", "L2", "L3", "L9"],
            "C/F Principal Balance": [100.0, 200.0, 300.0, 999.0],
        })
        tape, _ = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape[LOAN_KEY]) == ["L1", "L2", "L3"]
        assert 999.0 not in list(tape["current_principal_balance"])

    def test_a_loan_the_secondary_file_lacks_is_left_blank(self):
        """Blank is an answer. Filling it would be inventing one."""
        frames, resolved = _pack()
        frames[CASHFLOW] = pd.DataFrame({
            "Account": ["L1", "L3"],
            "C/F Principal Balance": [100.0, 300.0],
        })
        tape, _ = consolidate_pack(frames, resolved, PRIMARY)
        values = dict(zip(tape[LOAN_KEY], tape["current_principal_balance"]))
        assert values["L1"] == 100.0
        assert pd.isna(values["L2"])


class TestThePrimaryWinsAContestedField:
    def test_it_is_not_overwritten(self):
        """Consolidation adds facts; it never replaces one. Two files carrying
        one field is ordinary and is already flagged on the mapping table."""
        frames, resolved = _pack()
        frames[PRIMARY]["Balance"] = [1.0, 2.0, 3.0]
        resolved[PRIMARY]["Balance"] = "current_principal_balance"
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape["current_principal_balance"]) == [1.0, 2.0, 3.0]
        assert "current_principal_balance" not in report["added"]

    def test_a_column_the_primary_left_entirely_blank_is_filled(self):
        """A column present but empty is the same absence as a column missing,
        and a delivery blocked on an empty column the pack can fill is the
        original defect wearing a different hat."""
        frames, resolved = _pack()
        frames[PRIMARY]["Balance"] = [None, None, None]
        resolved[PRIMARY]["Balance"] = "current_principal_balance"
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape["current_principal_balance"]) == [100.0, 200.0, 300.0]
        assert report["added"]["current_principal_balance"] == CASHFLOW


class TestAFileThatCannotBeJoined:
    def test_no_loan_identifier_contributes_nothing_and_says_so(self):
        frames, resolved = _pack()
        resolved[CASHFLOW].pop("Account")
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert "current_principal_balance" not in tape.columns
        entry = next(f for f in report["files"]
                     if f["source_file"] == CASHFLOW)
        assert entry["joined"] is False
        assert "loan identifier" in entry["note"]

    def test_a_primary_without_a_key_joins_nothing(self):
        """Reported rather than crashed: the pack is still worth showing."""
        frames, resolved = _pack()
        resolved[PRIMARY].pop("Loan Ref")
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape.columns) == ["current_interest_rate"]
        assert report["files"][0]["joined"] is False


class TestAPerPeriodExtract:
    """The ordinary shape of a principal-and-interest file: a row per loan per
    reporting period. Joined as-is it fans the tape out — every loan repeated
    once per period — and every downstream count and average is wrong."""

    def _periodic(self):
        frames, resolved = _pack()
        frames[CASHFLOW] = pd.DataFrame({
            "Account": ["L1", "L1", "L2", "L2", "L3", "L3"],
            "C/F Principal Balance": [90.0, 100.0, 180.0, 200.0, 270.0, 300.0],
            "Month Run": ["2026-08-31", "2026-09-30"] * 3,
        })
        resolved[CASHFLOW]["Month Run"] = PERIOD_FIELD
        return frames, resolved

    def test_the_tape_does_not_fan_out(self):
        frames, resolved = self._periodic()
        tape, _ = consolidate_pack(frames, resolved, PRIMARY)
        assert len(tape) == 3

    def test_the_latest_period_speaks_for_the_loan(self):
        frames, resolved = self._periodic()
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape["current_principal_balance"]) == [100.0, 200.0, 300.0]
        entry = next(f for f in report["files"]
                     if f["source_file"] == CASHFLOW)
        assert "latest" in entry["note"]

    def test_without_a_reporting_date_it_contributes_nothing(self):
        """Picking a row arbitrarily would be inventing an answer about which
        month the balance came from. Refusing to guess is the point."""
        frames, resolved = self._periodic()
        resolved[CASHFLOW].pop("Month Run")
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert "current_principal_balance" not in tape.columns
        entry = next(f for f in report["files"]
                     if f["source_file"] == CASHFLOW)
        assert entry["joined"] is False
        assert "more than one row per loan" in entry["note"]


class TestTheFilesDoNotSpellTheLoanIdTheSameWay:
    """``entity_key_resolver`` names these from real packs, and a bare string
    comparison joins NONE of them — silently, leaving a column of blanks and a
    tape that looks assembled."""

    def test_a_trailing_component_suffix_still_joins(self):
        """"Loan Policy Number 760341 links to Account Number 76034101 — a
        stable trailing 01 suffix." """
        frames, resolved = _pack()
        frames[PRIMARY]["Loan Ref"] = ["760341", "760342", "760343"]
        frames[CASHFLOW]["Account"] = ["76034101", "76034201", "76034301"]
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape["current_principal_balance"]) == [100.0, 200.0, 300.0]
        assert report["added"]["current_principal_balance"] == CASHFLOW

    def test_an_excel_decimal_suffix_still_joins(self):
        """``76034101`` read from a numeric column arrives as ``76034101.0``."""
        frames, resolved = _pack()
        frames[PRIMARY]["Loan Ref"] = ["76034101", "76034201", "76034301"]
        frames[CASHFLOW]["Account"] = [76034101.0, 76034201.0, 76034301.0]
        tape, _ = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape["current_principal_balance"]) == [100.0, 200.0, 300.0]

    def test_unrelated_identifiers_are_refused_rather_than_joined_to_nothing(self):
        """THE SILENT FAILURE THIS GUARDS. Two files whose keys have nothing in
        common must not produce a 'joined' tape full of blanks — that looks
        assembled and is not."""
        frames, resolved = _pack()
        frames[CASHFLOW]["Account"] = ["L7", "L8", "L9"]
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert "current_principal_balance" not in tape.columns
        entry = next(f for f in report["files"]
                     if f["source_file"] == CASHFLOW)
        assert entry["joined"] is False
        assert entry["overlap"] == 0.0
        assert "loan identifiers" in entry["note"]

    def test_an_alpha_prefix_is_not_part_of_the_key(self):
        """PINNED AS THE PLATFORM'S BEHAVIOUR, not as a preference.

        ``entity_key_resolver``'s numeric rule is ``re.sub(r"\\D", "", s)``, so
        ``L1`` and ``XX1`` are one key. That is how the platform joins a pack
        whose files prefix the same id differently, and re-deciding it here
        would be exactly the drift this consolidation exists to remove. It is
        asserted so a change to the rule shows up as a failure HERE, beside the
        join that depends on it, rather than as a wrong tape.
        """
        frames, resolved = _pack()
        frames[CASHFLOW]["Account"] = ["XX1", "XX2", "XX3"]
        tape, _ = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape["current_principal_balance"]) == [100.0, 200.0, 300.0]

    def test_the_rule_and_the_overlap_are_on_the_record(self):
        """Which rule joined two files is a fact about the delivery, and an
        approver asking "are these the same loans?" needs the number."""
        frames, resolved = _pack()
        frames[PRIMARY]["Loan Ref"] = ["760341", "760342", "760343"]
        frames[CASHFLOW]["Account"] = ["76034101", "76034201", "76034301"]
        _, report = consolidate_pack(frames, resolved, PRIMARY)
        entry = next(f for f in report["files"]
                     if f["source_file"] == CASHFLOW)
        assert entry["overlap"] == 1.0
        assert entry["key_rule"] == "strip_trailing_01"

    def test_the_source_value_is_never_mutated(self):
        """"Normalisation never mutates source data: it produces a comparison
        key only; original values are preserved for lineage." """
        frames, resolved = _pack()
        frames[PRIMARY]["Loan Ref"] = ["760341", "760342", "760343"]
        frames[CASHFLOW]["Account"] = ["76034101", "76034201", "76034301"]
        tape, _ = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape[LOAN_KEY]) == ["760341", "760342", "760343"]


class TestAPackOfOneFile:
    def test_it_reads_exactly_as_before(self):
        """The change must not alter a single-file delivery, which is most of
        them."""
        frames, resolved = _pack()
        del frames[CASHFLOW]
        del resolved[CASHFLOW]
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert set(tape.columns) == {LOAN_KEY, "current_interest_rate"}
        assert report["added"] == {}


class TestThreeFiles:
    def test_each_contributes_its_own_columns(self):
        frames, resolved = _pack()
        frames[PROPERTY] = pd.DataFrame({
            "Pool Ref": ["L1", "L2", "L3"],
            "Latest Property Value": [500.0, 600.0, 700.0],
        })
        resolved[PROPERTY] = {"Pool Ref": LOAN_KEY,
                              "Latest Property Value": "current_valuation_amount"}
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert set(tape.columns) == {
            LOAN_KEY, "current_interest_rate", "current_principal_balance",
            "current_valuation_amount"}
        assert report["added"] == {
            "current_principal_balance": CASHFLOW,
            "current_valuation_amount": PROPERTY}
