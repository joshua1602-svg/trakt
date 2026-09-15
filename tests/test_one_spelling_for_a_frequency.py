"""A delivery frequency has one spelling, and it reaches the batch.

TWO DEFECTS THAT COMPOUND

1. ``adhoc`` AND ``ad_hoc`` WERE BOTH ACCEPTED, AND THEY ARE NOT THE SAME KEY.

   ``VALID_FREQUENCIES`` took either. But the frequency is a path segment, part
   of the pack key, and part of a snapshot's ``logical_slot`` — and
   ``snapshot.keys.normalise_key_part`` leaves the underscore alone, so
   ``adhoc`` and ``ad_hoc`` normalise to two different key parts. One stream
   spelled two ways is two folder trees and two key spaces.

   Worse, only ``adhoc`` is in ``snapshot.model.VALID_CADENCES``, so the other
   spelling raises an ``invalid_cadence`` WARNING — non-fatal, so it does not
   stop anything, it just settles into the data as an unrecognised value.

   The file already solves this shape of problem for periods: real uploads use
   both separators and ``_normalise_period`` canonicalises them. Frequency now
   does the same. Input stays tolerant; what is WRITTEN is canonical.

2. THE FREQUENCY COULD NOT BE SET THROUGH THE API AT ALL.

   ``manual_intake`` derives the destination from the batch's client, portfolio,
   book, dataset, frequency and period. ``intake.create_batch`` stores
   ``frequency or BATCH_FREQUENCY_DEFAULT`` — and the default is ``monthly``.
   ``engine.create_batch`` accepts a frequency and passes it through.

   But the ``CreateBatch`` request model had no frequency field, and the Manual
   delivery screen never sent one. So every manually created delivery landed
   under ``/monthly/``, whatever it actually was — a weekly pipeline file
   included. The plumbing ran the whole way down and the HTTP door dropped it.
"""

from __future__ import annotations

import pytest

from apps.blob_trigger_app.path_parser import (
    VALID_FREQUENCIES,
    canonical_frequency,
)
from snapshot.keys import normalise_key_part
from snapshot.model import VALID_CADENCES


class TestOneSpellingSurvives:
    @pytest.mark.parametrize("given", ["adhoc", "ad_hoc", "ad hoc", "AD_HOC",
                                       "Ad Hoc", " adhoc "])
    def test_every_way_of_writing_it_lands_on_one(self, given):
        assert canonical_frequency(given) == "adhoc"

    def test_the_canonical_form_is_the_one_the_snapshot_layer_knows(self):
        """`ad_hoc` would have warned as an unrecognised cadence forever."""
        assert canonical_frequency("ad_hoc") in VALID_CADENCES
        assert "ad_hoc" not in VALID_CADENCES

    def test_the_two_spellings_really_were_different_keys(self):
        """The reason this matters: nothing downstream reconciled them."""
        assert normalise_key_part("adhoc") != normalise_key_part("ad_hoc")
        assert (normalise_key_part(canonical_frequency("adhoc"))
                == normalise_key_part(canonical_frequency("ad_hoc")))

    @pytest.mark.parametrize("given", ["monthly", "weekly", "daily"])
    def test_the_others_are_unchanged(self, given):
        assert canonical_frequency(given) == given

    def test_an_unknown_frequency_is_refused_rather_than_guessed(self):
        from apps.blob_trigger_app.path_parser import PathParseError
        with pytest.raises(PathParseError):
            canonical_frequency("fortnightly")

    def test_blank_stays_blank_for_the_caller_to_default(self):
        """Canonicalising must not invent a frequency nobody chose."""
        assert canonical_frequency("") == ""
        assert canonical_frequency(None) == ""

    def test_both_spellings_are_still_accepted_as_input(self):
        """Tolerant in, canonical out — existing callers keep working."""
        assert "adhoc" in VALID_FREQUENCIES
        assert "ad_hoc" in VALID_FREQUENCIES


class TestTheFrequencyReachesTheBatch:
    """Through the API door, which used to drop it."""

    def test_the_request_model_carries_it(self):
        from operations_control.api.app import CreateBatch
        assert "frequency" in CreateBatch.model_fields

    def test_a_pipeline_batch_can_be_weekly(self, tmp_path, monkeypatch):
        eng = _engine(tmp_path, monkeypatch)
        batch = eng.create_batch(
            client_id="ERE", portfolio_id="direct_001",
            reporting_date="2026-09-14", workflow_type="mi",
            created_by="op", dataset="pipeline", frequency="weekly")
        assert batch["frequency"] == "weekly"

    def test_an_unspelled_frequency_is_canonicalised_on_the_batch(
            self, tmp_path, monkeypatch):
        eng = _engine(tmp_path, monkeypatch)
        batch = eng.create_batch(
            client_id="ERE", portfolio_id="direct_001",
            reporting_date="2026-09-14", workflow_type="mi",
            created_by="op", dataset="pipeline", frequency="ad_hoc")
        assert batch["frequency"] == "adhoc"

    def test_omitting_it_still_means_monthly(self, tmp_path, monkeypatch):
        """Unchanged for every caller that never passed one."""
        eng = _engine(tmp_path, monkeypatch)
        batch = eng.create_batch(
            client_id="ERE", portfolio_id="direct_001",
            reporting_date="2026-04", workflow_type="mi",
            created_by="op", dataset="funded")
        assert batch["frequency"] == "monthly"

    def test_a_frequency_trakt_does_not_know_is_refused(self, tmp_path,
                                                        monkeypatch):
        from operations_control.engine import OpsError
        eng = _engine(tmp_path, monkeypatch)
        with pytest.raises(OpsError):
            eng.create_batch(
                client_id="ERE", portfolio_id="direct_001",
                reporting_date="2026-09-14", workflow_type="mi",
                created_by="op", dataset="pipeline", frequency="fortnightly")


def _engine(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    from apps.blob_trigger_app.storage import Storage
    from operations_control.engine import OpsEngine
    from operations_control.stores import OpsStore
    return OpsEngine(OpsStore(Storage(tmp_path / "blob")))
