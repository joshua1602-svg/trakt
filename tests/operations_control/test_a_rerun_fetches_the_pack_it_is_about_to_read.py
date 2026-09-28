"""A rerun replayed a folder Azure had already thrown away.

The repository already said so, in ``backfill``'s own docstring:

    Unlike ``ops rerun`` — which replays a run record's ``input_dir`` and
    therefore depends on an ephemeral run scratch that Azure has usually
    reclaimed — backfill re-fetches the pack's source files from the container
    every time, so it can always localise the source.

So the one tool that always works is the one that goes and gets the files, and
"Run again" was the one that trusted scratch. ``restage_missing_files`` was
written for the same problem but restores BY RECORD: it iterates the files the
pack still lists as current and unique, and when that set is empty it restores
nothing, reports nothing, and reads exactly like a delivery that lost nothing.

The pack is not only a record. It is a folder in the container, still holding
every file the delivery arrived with, and the blob trigger records where that
is on the first arrival:

    batch["source_prefix"] = f"{container}/{inner.rsplit('/', 1)[0]}"

A rerun now refetches from there before it gives up, into the folder the run
will actually read — which is separately recorded from the batch's own, and a
delivery is not helped by files restored beside the folder it reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from operations_control.intake import IntakeService

PREFIX = "raw-v2/ERE/direct/funded/monthly/direct_001/2026-08"
PACK = ("LoanExtract One - OMNI 2026_09_01.xlsx",
        "Principal And Interest - OMNI 2026_09_01.xlsx",
        "PropertyExtract - Omni 2026_09_01.xlsx")


class _Storage:
    """Just enough of the storage port: list a prefix, download a blob."""

    def __init__(self, names, *, fail=()):
        self._names = list(names)
        self._fail = set(fail)
        self.listed = []

    def list(self, prefix_uri):
        self.listed.append(prefix_uri)
        return [f"{prefix_uri.rstrip('/')}/{n}" for n in self._names]

    def download_file(self, uri, local_path):
        name = uri.rsplit("/", 1)[-1]
        if name in self._fail:
            raise OSError("boom")
        Path(local_path).write_bytes(b"x")
        return Path(local_path)


class _Store:
    def __init__(self, storage):
        self.storage = storage
        self.audits = []

    def append_audit(self, client_id, event, **kw):
        self.audits.append({"client_id": client_id, "event": event, **kw})


def _svc(storage):
    svc = IntakeService.__new__(IntakeService)
    svc.store = _Store(storage)
    return svc


def _batch(prefix=PREFIX):
    return {"client_id": "ERE", "batch_id": "batch_1", "source_prefix": prefix}


class TestItFetchesThePackFromWhereItArrived:

    def test_every_data_file_in_the_folder_comes_back(self, tmp_path):
        # A directory of its own: the suite's own fixtures also write under
        # tmp_path, and this asserts on exactly what was restored.
        dest = tmp_path / "run_input"
        svc = _svc(_Storage(PACK))
        rep = svc.restage_from_source_prefix(_batch(), dest=dest)
        assert sorted(rep["restored"]) == sorted(PACK)
        assert sorted(p.name for p in dest.iterdir()) == sorted(PACK)

    def test_it_reads_the_prefix_the_delivery_arrived_under(self, tmp_path):
        storage = _Storage(PACK)
        _svc(storage).restage_from_source_prefix(_batch(), dest=tmp_path)
        assert storage.listed == [f"blob://{PREFIX}/"]

    def test_it_restores_where_the_run_reads_not_where_the_batch_lives(
            self, tmp_path):
        """The run's input path and the batch's folder are separately
        recorded. Restoring to the wrong one helps nobody."""
        svc = _svc(_Storage(PACK))
        svc.batch_dir = lambda b: tmp_path / "the_batch_folder"
        run_reads = tmp_path / "what_the_run_reads"
        svc.restage_from_source_prefix(_batch(), dest=run_reads)
        assert sorted(p.name for p in run_reads.iterdir()) == sorted(PACK)
        assert not (tmp_path / "the_batch_folder").exists()

    def test_a_marker_or_a_note_is_not_refetched(self, tmp_path):
        svc = _svc(_Storage(list(PACK) + ["_READY.json", "notes.txt"]))
        rep = svc.restage_from_source_prefix(_batch(), dest=tmp_path)
        assert sorted(rep["restored"]) == sorted(PACK)

    def test_it_audits_what_it_rebuilt(self, tmp_path):
        svc = _svc(_Storage(PACK))
        svc.restage_from_source_prefix(_batch(), dest=tmp_path)
        events = [a["event"] for a in svc.store.audits]
        assert "delivery_refetched_from_source" in events


class TestItReportsRatherThanPretends:

    def test_no_recorded_location_says_so_and_fetches_nothing(self, tmp_path):
        svc = _svc(_Storage(PACK))
        rep = svc.restage_from_source_prefix(_batch(prefix=""), dest=tmp_path)
        assert rep["restored"] == []
        assert "no governed source location" in rep["reason"]
        assert svc.store.storage.listed == []

    def test_an_empty_folder_says_so(self, tmp_path):
        svc = _svc(_Storage([]))
        rep = svc.restage_from_source_prefix(_batch(), dest=tmp_path)
        assert rep["restored"] == []
        assert "no data files" in rep["reason"]

    def test_one_file_that_will_not_download_does_not_lose_the_others(
            self, tmp_path):
        svc = _svc(_Storage(PACK, fail={PACK[1]}))
        rep = svc.restage_from_source_prefix(_batch(), dest=tmp_path)
        assert sorted(rep["restored"]) == sorted([PACK[0], PACK[2]])
        assert [f["filename"] for f in rep["failed"]] == [PACK[1]]

    def test_storage_that_will_not_even_list_is_reported_not_raised(
            self, tmp_path):
        class _Broken:
            def list(self, _):
                raise RuntimeError("no network")
        svc = _svc(_Broken())
        rep = svc.restage_from_source_prefix(_batch(), dest=tmp_path)
        assert rep["restored"] == []
        assert "RuntimeError" in rep["reason"]


class TestTheRunTriesThisBeforeItGivesUp:
    """The engine's order: restore by record, look, refetch by location, look
    again, and only then stop."""

    def test_a_refetch_turns_a_shortfall_into_a_run(self, tmp_path):
        from operations_control.engine import OpsEngine

        eng = OpsEngine.__new__(OpsEngine)
        input_path = tmp_path / "wf" / "files"

        class _Intake:
            def load_batch(self, *_a, **_k):
                return _batch()

            def restage_from_source_prefix(self, batch, *, dest):
                Path(dest).mkdir(parents=True, exist_ok=True)
                (Path(dest) / PACK[0]).write_bytes(b"x")
                return {"restored": [PACK[0]]}

        eng._intake = _Intake()

        class _Run:
            client_id, workflow_id, batch_id = "ERE", "wf_1", "batch_1"
            delivery = {"input_path": str(input_path)}

        run = _Run()
        # Before: the folder the run reads holds nothing.
        assert OpsEngine._input_shortfall(eng, run) != ""
        eng._intake.restage_from_source_prefix(
            _batch(), dest=Path(run.delivery["input_path"]))
        # After: there is something to read, so the run proceeds.
        assert OpsEngine._input_shortfall(eng, run) == ""
