"""The backfill plan: which delivery a filename becomes, and which period.

This is the only part of ``tools/ere_backfill.py`` worth testing, and it is the
part that matters most. Everything else is transport — SFTP, decryption, two
HTTP calls — and a mistake there fails loudly. A mistake HERE puts a real
client's real data in the wrong reporting period, where it looks correct and is
not, and where finding it later means reconciling sixty-four files by hand.

The rules encode what this client's names mean, which is not guessable:

    LoanExtract One - OMNI 2026_05_01.xlsx      April 2026's funded book
    PropertyExtract - Omni 2026_05_01.xlsx      April 2026, same pack
    M2L KFI and Pipeline 2026_09_14_103458.xlsx the pipeline as at 14 Sep

A tape stamped the first of a month reports the month BEFORE it: the stamp is
when the extract was taken, not the period it describes. A pipeline tape is a
snapshot and its own date IS the period.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

# Loaded by path: `tools/` is an operator toolbox, not an importable package,
# and the script deliberately has no dependency on the platform's layout. It
# must be registered in `sys.modules` before it executes, because `dataclass`
# resolves a class's own module while building it.
_SPEC = importlib.util.spec_from_file_location(
    "ere_backfill",
    pathlib.Path(__file__).resolve().parents[1] / "tools" / "ere_backfill.py")
backfill = importlib.util.module_from_spec(_SPEC)
sys.modules["ere_backfill"] = backfill
_SPEC.loader.exec_module(backfill)          # type: ignore[union-attr]

FUNDED = "LoanExtract One - OMNI 2026_05_01.xlsx"
PROPERTY = "PropertyExtract - Omni 2026_05_01.xlsx"
PIPELINE = "M2L KFI and Pipeline 2026_09_14_103458.xlsx"


def only(filename: str):
    planned = backfill.plan([filename])
    assert len(planned) == 1
    return planned[0]


class TestAFundedTapeReportsThePriorMonth:
    def test_the_first_of_may_is_aprils_book(self):
        assert only(FUNDED).period == "2026-04"

    def test_january_rolls_back_a_year(self):
        assert only("LoanExtract One - OMNI 2026_01_01.xlsx").period == "2025-12"

    def test_the_property_tape_lands_in_the_same_period(self):
        """Same pack: same client, portfolio, period and book."""
        assert only(PROPERTY).period == only(FUNDED).period

    @pytest.mark.parametrize("filename", [FUNDED, PROPERTY])
    def test_it_is_the_funded_book_monthly(self, filename):
        d = only(filename)
        assert d.rule.dataset == "funded"
        assert d.rule.frequency == "monthly"


class TestAPipelineTapeIsASnapshot:
    def test_its_own_date_is_the_period(self):
        assert only(PIPELINE).period == "2026-09-14"

    def test_it_is_adhoc_not_weekly(self):
        """These arrive every few days. One batch per snapshot, no collisions."""
        d = only(PIPELINE)
        assert d.rule.dataset == "pipeline"
        assert d.rule.frequency == "adhoc"

    def test_the_time_stamp_after_the_date_is_ignored(self):
        assert only("M2L KFI and Pipeline 2026_09_14_103458.xlsx").period \
            == "2026-09-14"

    def test_it_is_never_read_as_a_funded_tape(self):
        """Ordered rules: "Pipeline" is tested before any funded rule."""
        assert only(PIPELINE).rule.dataset != "funded"

    def test_a_pipeline_delivery_is_never_routed_to_regime(self):
        """The engine refuses pipeline + mi_annex2; the plan must not ask."""
        assert only(PIPELINE).rule.workflow == "mi"


class TestItRefusesRatherThanGuesses:
    def test_an_unrecognised_name_stops_the_whole_run(self):
        with pytest.raises(backfill.Refused) as exc:
            backfill.plan([FUNDED, "Some Other Report 2026_05_01.xlsx"])
        assert "no rule recognises" in str(exc.value)

    def test_a_name_with_no_date_stops_the_whole_run(self):
        with pytest.raises(backfill.Refused) as exc:
            backfill.plan(["LoanExtract One - OMNI.xlsx"])
        assert "no date" in str(exc.value)

    def test_one_bad_name_loads_nothing(self):
        """A partial load is the worst outcome: the gaps are invisible after."""
        with pytest.raises(backfill.Refused):
            backfill.plan([FUNDED, PROPERTY, PIPELINE, "mystery.xlsx"])

    def test_non_data_files_are_skipped_not_refused(self):
        """A README beside the tapes is not a delivery and not a problem."""
        planned = backfill.plan([FUNDED, "README.txt", ".DS_Store"])
        assert [d.filename for d in planned] == [FUNDED]


class TestAWholeMonthsPack:
    def test_the_three_files_plan_as_two_periods(self):
        planned = backfill.plan([FUNDED, PROPERTY, PIPELINE])
        assert len(planned) == 3
        periods = {(d.rule.dataset, d.period) for d in planned}
        assert periods == {("funded", "2026-04"), ("pipeline", "2026-09-14")}

    def test_every_frequency_is_one_trakt_writes(self):
        """The canonical spelling, or the pack key splits in two."""
        from apps.blob_trigger_app.path_parser import canonical_frequency
        for d in backfill.plan([FUNDED, PROPERTY, PIPELINE]):
            assert canonical_frequency(d.rule.frequency) == d.rule.frequency
