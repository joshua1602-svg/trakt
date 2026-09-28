"""ERE's delivery ran against a folder with nothing in it.

    "No file in this delivery was opened. … The delivery appears to hold no
     files."

That sentence was right, and it arrived four steps too late. The pack was in
blob, the delivery recorded three files, and step 1 said "3 files received. The
pack was complete" — but the working folder the agents read had been wiped (the
staging root is scratch: ``/tmp`` on the App Service, emptied by every restart,
including the two deploys that shipped the previous fix).

``restage_missing_files`` exists for exactly that and did not fire, because it
iterates ``_current_files`` — the files still marked ``current`` and ``unique``.
When that set is empty there is nothing to restore, nothing to report, and its
report reads identically to "nothing was lost". The run then proceeded into an
empty directory and produced an empty file inventory, an empty loan tape, and a
blocker about a missing loan listing.

AN ABSENT INPUT IS NOT A MODELLING RESULT. The run is refused where that is
true, with a sentence that says what happened to the working copy and that the
files themselves are safe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from operations_control.language import GENERIC_PROBLEM, humanise_blocker


class _Batch(dict):
    pass


class _FakeIntake:
    def __init__(self, batch):
        self._batch = batch

    def load_batch(self, client_id, batch_id):      # noqa: ARG002
        return self._batch


class _Run:
    def __init__(self, input_path, batch_id="batch_1"):
        self.client_id = "ERE"
        self.workflow_id = "wf_1"
        self.batch_id = batch_id
        self.delivery = {"input_path": str(input_path)}


def _shortfall(run, batch):
    """Call the engine's check without standing up a whole engine."""
    from operations_control.engine import OpsEngine
    eng = OpsEngine.__new__(OpsEngine)
    # ``intake`` is a lazy property; seed the cache it reads rather than
    # replacing the property.
    eng._intake = _FakeIntake(batch)
    return OpsEngine._input_shortfall(eng, run)


@pytest.fixture
def batch_of_three():
    return _Batch(files=[{"original_filename": n} for n in (
        "LoanExtract One - OMNI 2026_09_01.xlsx",
        "Principal And Interest - OMNI 2026_09_01.xlsx",
        "PropertyExtract - Omni 2026_09_01.xlsx")])


class TestAFolderWithDataInItIsNotAShortfall:

    def test_one_data_file_is_enough_to_proceed(self, tmp_path, batch_of_three):
        (tmp_path / "LoanExtract One - OMNI 2026_09_01.xlsx").write_bytes(b"x")
        assert _shortfall(_Run(tmp_path), batch_of_three) == ""

    def test_a_csv_counts_too(self, tmp_path, batch_of_three):
        (tmp_path / "loans.csv").write_text("a,b\n1,2\n")
        assert _shortfall(_Run(tmp_path), batch_of_three) == ""

    def test_a_marker_or_a_note_is_not_a_data_file(self, tmp_path,
                                                   batch_of_three):
        (tmp_path / "_READY").write_text("")
        (tmp_path / "notes.txt").write_text("hello")
        assert _shortfall(_Run(tmp_path), batch_of_three) != ""


class TestAnEmptyFolderIsRefusedAndSaysWhatHappened:

    def test_it_names_how_many_files_the_delivery_recorded(self, tmp_path,
                                                           batch_of_three):
        said = _shortfall(_Run(tmp_path), batch_of_three)
        assert "3 file(s)" in said

    def test_it_says_the_files_themselves_are_safe(self, tmp_path,
                                                   batch_of_three):
        said = _shortfall(_Run(tmp_path), batch_of_three)
        assert "safe where they were received" in said
        assert "working copy" in said

    def test_it_says_what_to_do_next(self, tmp_path, batch_of_three):
        assert "Send the pack again" in _shortfall(_Run(tmp_path),
                                                   batch_of_three)

    def test_a_folder_that_does_not_exist_at_all_reads_the_same(
            self, tmp_path, batch_of_three):
        gone = tmp_path / "never_created"
        assert "Send the pack again" in _shortfall(_Run(gone), batch_of_three)

    def test_no_recorded_location_says_that_rather_than_a_count(
            self, batch_of_three):
        run = _Run("")
        said = _shortfall(run, batch_of_three)
        assert "no recorded location" in said
        assert "3 file(s)" not in said

    def test_a_delivery_with_no_batch_does_not_invent_a_count(self, tmp_path):
        said = _shortfall(_Run(tmp_path, batch_id=""), None)
        assert "file(s)" not in said
        assert "No files are available" in said

    def test_the_sentence_reaches_the_operator_unchanged(self, tmp_path,
                                                         batch_of_three):
        said = _shortfall(_Run(tmp_path), batch_of_three)
        assert humanise_blocker(said) == said
        assert humanise_blocker(said) != GENERIC_PROBLEM


class TestRestagingStopsCallingSilenceGoodNews:
    """The report now distinguishes "nothing was lost" from "there was nothing
    I could even look at"."""

    def test_it_counts_what_it_recorded_and_what_it_considered(self, tmp_path):
        from operations_control.intake import IntakeService
        svc = IntakeService.__new__(IntakeService)
        svc.batch_dir = lambda b: Path(tmp_path)          # type: ignore[method-assign]
        batch = {"client_id": "ERE", "batch_id": "b1", "files": [
            {"original_filename": "a.xlsx", "superseded_status": "superseded",
             "duplicate_status": "unique", "storage_reference": "", "sha256": ""},
            {"original_filename": "a.xlsx", "superseded_status": "current",
             "duplicate_status": "duplicate_ignored", "storage_reference": "",
             "sha256": ""}]}
        report = IntakeService.restage_missing_files(svc, batch)
        assert report["recorded"] == 2
        assert report["considered"] == 0
        # …and it still reports nothing missing, which is exactly why the
        # caller may not read that as "the folder has files in it".
        assert report["missing"] == [] and report["restored"] == []
