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


class TestBothFilesCarryTheReportingDate:
    """REPORTED FROM THE LIVE CASE, with the balance mapped and confirmed:

        "current_principal_balance: CORE001 affects 0 record(s) — BLOCKING"
        "I have already tagged [it] as being mapped to a field from the P&I
         file. Why isn't this mapping being picked up?"

    Both halves of the consolidation were right and they cancelled each other
    out. ``spare`` is what a secondary file would ADD, and a loan extract
    states the same reporting date the cashflow extract does — so the period
    column was excluded as redundant, and the narrowing to ``spare`` took away
    the one column ``_one_row_per_loan`` needs to say which of a loan's monthly
    rows speaks for it.

    The file was then dropped for having "no data cut off date", against an
    operator who had mapped exactly that. The advice in the message was to map
    the reporting date. They had. Following it could not have helped, and the
    balance stayed out of the tape while the screen said the mapping was
    confirmed at 100%.

    ``TestAPerPeriodExtract`` above did not catch it because its spine carries
    no reporting date of its own, which is the one arrangement where the column
    survives the narrowing.
    """

    def _both_dated(self):
        frames, resolved = _pack()
        frames[PRIMARY] = pd.DataFrame({
            "Loan Ref": ["L1", "L2", "L3"],
            "Rate": [4.5, 5.0, 5.5],
            "Reporting Date": ["2026-09-30"] * 3,
        })
        resolved[PRIMARY]["Reporting Date"] = PERIOD_FIELD
        frames[CASHFLOW] = pd.DataFrame({
            "Account": ["L1", "L1", "L2", "L2", "L3", "L3"],
            "C/F Principal Balance": [90.0, 100.0, 180.0, 200.0, 270.0, 300.0],
            "Month Run": ["2026-08-31", "2026-09-30"] * 3,
        })
        resolved[CASHFLOW]["Month Run"] = PERIOD_FIELD
        return frames, resolved

    def test_the_balance_still_reaches_the_tape(self):
        """THE DEFECT, in one line."""
        frames, resolved = self._both_dated()
        tape, _ = consolidate_pack(frames, resolved, PRIMARY)
        assert "current_principal_balance" in tape.columns

    def test_the_latest_period_still_speaks_for_the_loan(self):
        """Carrying the period column through the narrowing must not change
        WHICH row wins — only whether the question can be answered at all."""
        frames, resolved = self._both_dated()
        tape, _ = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape["current_principal_balance"]) == [100.0, 200.0, 300.0]

    def test_the_tape_does_not_fan_out(self):
        frames, resolved = self._both_dated()
        tape, _ = consolidate_pack(frames, resolved, PRIMARY)
        assert len(tape) == 3

    def test_the_spine_keeps_its_own_reporting_date(self):
        """The period column is carried for the COLLAPSE and attached to
        nothing. The primary wins a contested field, and the date the loan
        extract states is the date on the tape."""
        frames, resolved = self._both_dated()
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape[PERIOD_FIELD]) == ["2026-09-30"] * 3
        assert PERIOD_FIELD not in (report["added"] or {})

    def test_it_is_not_reported_as_missing_the_date_that_was_mapped(self):
        """The note told the operator to do the thing they had already done."""
        frames, resolved = self._both_dated()
        _, report = consolidate_pack(frames, resolved, PRIMARY)
        entry = next(f for f in report["files"]
                     if f["source_file"] == CASHFLOW)
        assert entry["joined"] is True
        assert "no data cut off date" not in entry["note"]

    def test_a_file_that_genuinely_lacks_the_date_is_still_refused(self):
        """The refusal is right when it is true. Carrying the column through
        must not turn "Trakt cannot tell which row speaks for the loan" into a
        silent guess."""
        frames, resolved = self._both_dated()
        resolved[CASHFLOW].pop("Month Run")
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert "current_principal_balance" not in tape.columns
        entry = next(f for f in report["files"]
                     if f["source_file"] == CASHFLOW)
        assert entry["joined"] is False


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


class TestOnePackWhereTheFilesDisagreeWithEachOther:
    """ERE's actual pack, described by the operator:

        "Loan Policy Number in the LoanExtract matches Account Number in the
         P&I excel but the only joiner in the PropertyExtract is LoanID and it
         drops the 01 suffix."

    So the loan extract and the cashflow extract share the LONG form, and the
    property extract carries the SHORT one. There is no single rule that joins
    the whole pack: against the cashflow file the spine's key is already right,
    and against the property file the same key must be stripped.

    This is why the rule is resolved PER PAIR rather than once for the pack.
    A single global rule would have to strip for everyone — which breaks the
    cashflow join — or strip for nobody, which breaks the property join. Either
    way one file silently contributes nothing.
    """

    def _ere_pack(self):
        frames = {
            PRIMARY: pd.DataFrame({
                "Loan Policy Number": ["76034101", "76034201", "76034301"],
                "Rate": [4.5, 5.0, 5.5],
            }),
            CASHFLOW: pd.DataFrame({
                "Account Number": ["76034101", "76034201", "76034301"],
                "C/F Principal Balance": [100.0, 200.0, 300.0],
            }),
            PROPERTY: pd.DataFrame({
                "LoanID": ["760341", "760342", "760343"],
                "Latest Property Value": [500.0, 600.0, 700.0],
            }),
        }
        resolved = {
            PRIMARY: {"Loan Policy Number": LOAN_KEY,
                      "Rate": "current_interest_rate"},
            CASHFLOW: {"Account Number": LOAN_KEY,
                       "C/F Principal Balance": "current_principal_balance"},
            PROPERTY: {"LoanID": LOAN_KEY,
                       "Latest Property Value": "current_valuation_amount"},
        }
        return frames, resolved

    def test_both_files_join_under_their_own_rule(self):
        frames, resolved = self._ere_pack()
        tape, report = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape["current_principal_balance"]) == [100.0, 200.0, 300.0]
        assert list(tape["current_valuation_amount"]) == [500.0, 600.0, 700.0]
        assert report["added"] == {
            "current_principal_balance": CASHFLOW,
            "current_valuation_amount": PROPERTY}

    def test_the_suffix_is_stripped_only_where_it_has_to_be(self):
        """THE EVIDENCE that the resolution is per pair and not per pack.

        The SPINE's own rule differs between the two joins: left alone against
        the cashflow extract, which already carries the long key, and stripped
        against the property extract, which carries the short one. A single
        rule for the pack could not do both.
        """
        frames, resolved = self._ere_pack()
        _, report = consolidate_pack(frames, resolved, PRIMARY)
        by_file = {f["source_file"]: f for f in report["files"]}
        assert by_file[CASHFLOW]["primary_key_rule"] == "numeric_string"
        assert by_file[PROPERTY]["primary_key_rule"] == "strip_trailing_01"
        assert by_file[CASHFLOW]["key_rule"] == "numeric_string"
        assert by_file[PROPERTY]["key_rule"] == "numeric_string"
        assert by_file[CASHFLOW]["overlap"] == 1.0
        assert by_file[PROPERTY]["overlap"] == 1.0

    def test_the_tape_keeps_the_spine_s_own_identifier(self):
        """The long form is what the loan extract calls the loan, and the tape
        is the loan extract's. A comparison key is not an identifier."""
        frames, resolved = self._ere_pack()
        tape, _ = consolidate_pack(frames, resolved, PRIMARY)
        assert list(tape[LOAN_KEY]) == ["76034101", "76034201", "76034301"]


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
