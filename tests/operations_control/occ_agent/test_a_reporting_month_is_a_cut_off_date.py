"""A reporting month is a cut-off date, not a blank.

Reported from the live ERE onboarding. The blocker read:

    data cut off date: mandatory_null affects 568 record(s) (100.0%)

Every row of a 568-row book missing its cut-off date — from a client whose
extract states the period on every single row. Their file identifies its month
in a ``Month Run`` column carrying ``August``; the cut-off is 31/08/2026.

Nothing was mis-mapped. ``month run`` is a GOVERNED ALIAS of
``data_cut_off_date`` in ``config/system/aliases_mandatory.yaml``, so the column
resolves exactly as it is meant to. What then happened is that canonical typing
read ``August`` as a date, failed, and wrote a blank — and the run stopped on a
field the client had supplied in full.

THE PLATFORM ALREADY DOES THIS STEP. ``central_tape_builder`` canonicalises
precisely these fields (``_PERIOD_CUTOFF_FIELDS``) at precisely this point,
turning ``August`` into ``2026-08-31`` against the run year and keeping the raw
label in lineage. The Agent built its tape without it. The same defect class as
the core-presence check and the applicability lookup before it: a stage body
here drifting from the platform's own build. So the fix calls the builder's own
function rather than restating it.

WHAT IT WILL NOT DO IS GUESS. A month name alone does not name a year. The
delivery's file names carry one; where they do not, the value is left exactly as
the client wrote it, for a human to look at, rather than dated to whatever year
the run happens to be executed in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from operations_control.occ_agent.execution import (
    _canonicalise_period_cutoffs,
    _run_year,
)

#: How ERE's monthly extracts are named.
DATED = Path("LoanExtract One - OMNI 2026_09_01.xlsx")
UNDATED = Path("Loan Extract.xlsx")


def tape(values, column="data_cut_off_date"):
    return pd.DataFrame({"loan_identifier": [f"L{i}" for i in range(len(values))],
                         column: list(values)})


class TestTheReportedBlocker:

    def test_a_month_name_becomes_the_month_end(self):
        """The reported case: ``August`` on every row of a 568-row book."""
        frame = tape(["August"] * 3)
        _canonicalise_period_cutoffs(frame, [DATED])
        assert frame["data_cut_off_date"].tolist() == ["2026-08-31"] * 3

    def test_the_column_is_no_longer_blank_after_typing(self):
        """The end-to-end shape of the defect: before this step, canonical
        typing turned every one of those values into a null and the presence
        check called the field 100% missing."""
        from engine.gate_2_transform.canonical_transform import (
            apply_types,
            load_registry,
            select_fields_for_portfolio,
        )
        meta = select_fields_for_portfolio(
            load_registry(Path("config/system/fields_registry.yaml")), "direct")
        frame = tape(["August", "August"])
        _canonicalise_period_cutoffs(frame, [DATED])
        apply_types(frame, meta)
        assert frame["data_cut_off_date"].notna().all(), \
            "the cut-off date is still blank after typing"

    def test_a_short_month_label_is_understood_too(self):
        frame = tape(["Aug"])
        _canonicalise_period_cutoffs(frame, [DATED])
        assert frame["data_cut_off_date"].tolist() == ["2026-08-31"]

    def test_a_month_end_is_a_month_end_not_the_first(self):
        """February, so an off-by-one on the month length would show."""
        frame = tape(["February"])
        _canonicalise_period_cutoffs(frame, [Path("Extract 2024_03_01.xlsx")])
        assert frame["data_cut_off_date"].tolist() == ["2024-02-29"]


class TestItDoesNotGuess:

    def test_a_month_with_no_year_anywhere_is_left_alone(self):
        """A month name does not name a year. Dating it to whenever the run
        happens to execute would invent the client's reporting period."""
        frame = tape(["August"])
        applied = _canonicalise_period_cutoffs(frame, [UNDATED])
        assert frame["data_cut_off_date"].tolist() == ["August"]
        assert applied == {}

    def test_a_value_that_means_nothing_is_left_alone(self):
        frame = tape(["n/a"])
        _canonicalise_period_cutoffs(frame, [DATED])
        assert frame["data_cut_off_date"].tolist() == ["n/a"]

    def test_a_blank_stays_blank(self):
        frame = tape(["", None])
        _canonicalise_period_cutoffs(frame, [DATED])
        assert frame["data_cut_off_date"].tolist()[0] == ""

    def test_a_date_the_client_wrote_as_a_date_is_kept(self):
        frame = tape(["31/08/2026"])
        _canonicalise_period_cutoffs(frame, [DATED])
        assert frame["data_cut_off_date"].tolist() == ["2026-08-31"]

    def test_only_the_cut_off_field_is_touched(self):
        """``origination_date`` is a fact about a loan, not a reporting period,
        and a month name there is a data problem to be seen, not converted."""
        frame = tape(["August"], column="origination_date")
        _canonicalise_period_cutoffs(frame, [DATED])
        assert frame["origination_date"].tolist() == ["August"]


class TestTheTransformIsVisible:

    def test_it_reports_what_it_read_the_label_as(self):
        """A value the platform changed on the client's behalf is a thing an
        operator has to be able to see and dispute."""
        frame = tape(["August", "September"])
        applied = _canonicalise_period_cutoffs(frame, [DATED])
        assert applied["data_cut_off_date"]["labels"] == {
            "August": "2026-08-31", "September": "2026-09-30"}
        assert applied["data_cut_off_date"]["run_year"] == 2026

    def test_re_normalising_a_date_is_not_reported_as_a_change(self):
        """Reading a date a client already wrote as a date is not news."""
        frame = tape(["31/08/2026"])
        assert _canonicalise_period_cutoffs(frame, [DATED]) == {}


class TestTheYearComesFromTheDelivery:

    def test_it_is_read_from_the_file_names(self):
        assert _run_year([DATED]) == 2026

    def test_a_delivery_that_states_no_year_yields_none(self):
        assert _run_year([UNDATED]) is None

    def test_the_first_file_that_states_one_answers(self):
        assert _run_year([UNDATED, DATED]) == 2026


class TestItUsesThePlatformsOwnConversion:

    def test_the_agent_does_not_carry_its_own_month_table(self):
        """A second copy of a platform conversion drifts from the original.
        This asserts the copy does not exist rather than that it agrees."""
        source = Path("operations_control/occ_agent/execution.py").read_text(
            encoding="utf-8")
        assert "_canonicalise_period_cutoff" in source, \
            "the Agent does not call the builder's own cut-off conversion"
        for month in ("January", "february", "MARCH"):
            assert month not in source, \
                "the Agent is spelling out month names of its own"

    def test_which_fields_are_converted_is_the_builders_list(self):
        source = Path("operations_control/occ_agent/execution.py").read_text(
            encoding="utf-8")
        assert "_PERIOD_CUTOFF_FIELDS" in source, \
            "the Agent decides for itself which fields carry a period"
