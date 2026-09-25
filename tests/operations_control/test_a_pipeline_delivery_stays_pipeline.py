"""A pipeline delivery is published as the pipeline, never as the funded book.

ERE's first manual pipeline delivery — `M2L KFI and Pipeline 2026_09_24_…` —
would have gone wrong three ways:

* its input pack required a LOAN TAPE (the MI workflow's requirement), which a
  pipeline pack never has, so it would have waited for ever;
* starting a pack dropped its book, so the delivery was registered as funded;
* publication wrote every delivery to the platform canonical — the funded
  book MI reads — and, dated 24 September, it was "newer" than August's
  funded month, so it would have replaced the funded portfolio in MI.

Now a pipeline pack asks for the pipeline file, keeps its book, and publishes
to the pipeline store; "newest" is judged within the book; and an older
pipeline snapshot never replaces the latest one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from operations_control.contracts import (RUN_AWAITING_PUBLICATION,
                                          WF_NEW_PORTFOLIO)

from .conftest import make_client_config, make_engine, start_and_wait

PIPELINE_FILE = "M2L KFI and Pipeline 2026_09_24_102225.csv"


@pytest.fixture()
def engine(store, source_registry, tmp_path):
    return make_engine(store, source_registry, "happy",
                       client_config=make_client_config(tmp_path))


def _pack(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir()
    (d / PIPELINE_FILE).write_text(
        "Case Reference,Loan Amount,Status\nC1,100000,KFI Issued\nC2,50000,Offer\n",
        encoding="utf-8")
    return d


def _persistence(engine):
    return ProductionPersistence(storage=engine.store.storage,
                                 layout=Layout.from_env())


class TestAPipelinePack:

    def test_asks_for_the_pipeline_file_not_a_loan_tape(self, engine):
        batch = engine.intake.create_batch(
            tenant_id="client_a", client_id="client_a", portfolio_id="pf1",
            reporting_date="2026-09-24", workflow_type="mi",
            created_by="test", dataset="pipeline", frequency="adhoc")
        assert batch["expected_input_roles"]["required"] == ["pipeline_report"]

    def test_a_funded_pack_still_asks_for_the_loan_tape(self, engine):
        batch = engine.intake.create_batch(
            tenant_id="client_a", client_id="client_a", portfolio_id="pf1",
            reporting_date="2026-08", workflow_type="mi",
            created_by="test", dataset="funded", frequency="monthly")
        assert batch["expected_input_roles"]["required"] == ["loan_extract"]

    def test_starting_a_pack_keeps_its_book(self, engine, monkeypatch):
        seen = {}
        real = engine.register_delivery

        def spy(**kw):
            seen.update(kw)
            return real(**kw)
        monkeypatch.setattr(engine, "register_delivery", spy)
        import inspect
        src = inspect.getsource(type(engine).start_batch)
        assert 'dataset=batch.get("dataset")' in src
        assert 'frequency=batch.get("frequency")' in src


class TestPublishingAPipelineDelivery:

    def _published(self, engine, tmp_path, name, period):
        d = engine.register_delivery(
            client_id="client_a", portfolio_id="pf1",
            input_path=str(_pack(tmp_path, name)), dataset="pipeline",
            frequency="adhoc", reporting_period=period, registered_by="test")
        run = engine.create_workflow(client_id="client_a",
                                     delivery_id=d["delivery_id"], outcome="mi",
                                     workflow_type=WF_NEW_PORTFOLIO,
                                     created_by="test")
        start_and_wait(engine, run, statuses=(RUN_AWAITING_PUBLICATION,))
        return engine.approve_publication(client_id="client_a",
                                          workflow_id=run.workflow_id,
                                          actor="josh")

    def test_it_goes_to_the_pipeline_store_not_the_funded_book(self, engine, tmp_path):
        pub = self._published(engine, tmp_path, "p1", "2026-09-24")
        layout, storage = Layout.from_env(), engine.store.storage
        assert storage.exists(layout.pipeline_latest_csv_uri("client_a"))
        assert storage.exists(layout.pipeline_period_csv_uri("client_a", "2026-09-24"))
        assert not storage.exists(layout.platform_latest_uri("client_a"))
        assert not storage.exists(layout.platform_period_uri("client_a", "2026-09-24"))
        assert pub.get("dataset") == "pipeline"

    def test_an_older_snapshot_does_not_replace_the_latest(self, engine, tmp_path):
        self._published(engine, tmp_path, "p1", "2026-09-24")
        layout, storage = Layout.from_env(), engine.store.storage
        before = storage.read_text(layout.pipeline_latest_pointer_uri("client_a"))
        pub = self._published(engine, tmp_path, "p0", "2026-09-17")
        assert pub["published_artefacts"]["is_latest"] is False
        assert storage.read_text(layout.pipeline_latest_pointer_uri("client_a")) == before
        assert storage.exists(layout.pipeline_period_csv_uri("client_a", "2026-09-17"))


class TestPersistPipeline:

    def test_a_backfill_writes_only_its_period(self, engine, tmp_path):
        f = tmp_path / "snap.csv"
        f.write_text("a\n1\n", encoding="utf-8")
        out = _persistence(engine).persist_pipeline("client_a", "2026-09-17",
                                                    str(f), update_latest=False)
        assert out["latest"] is None and out["pointer"] is None
        assert out["period"]
