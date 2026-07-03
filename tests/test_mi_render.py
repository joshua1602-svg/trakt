#!/usr/bin/env python3
"""tests/test_mi_render.py

Phase 4 — make the latest data actually render in React.

  * data_source cache invalidation: a re-published source is picked up on the next
    request (ETag/mtime signature) without an API restart — no stale @lru_cache.
  * weekly pipeline blob round-trip: a weekly (dataset=pipeline) run persists a
    pipeline snapshot to the durable store, and the MI API resolves it from the
    blob pointer (MI_AGENT_PIPELINE_URI).

Run: python -m unittest tests.test_mi_render
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import router as R
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.repin import repin_source
from apps.blob_trigger_app.schema_fingerprint import fingerprint_pack
from apps.blob_trigger_app.source_registry import SourceRegistry
from apps.blob_trigger_app.storage import Storage


class _EnvGuard:
    def __init__(self, **kw):
        self.kw = kw
        self.saved = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestDataSourceCacheInvalidation(unittest.TestCase):

    def test_reloads_when_source_changes(self):
        import mi_agent_api.data_source as ds
        with tempfile.TemporaryDirectory() as td:
            csv = Path(td) / "data.csv"
            csv.write_text("a,b\n1,2\n")
            calls = {"n": 0}

            def _fake_load():
                calls["n"] += 1
                # value reflects current file size so a change is observable.
                return (f"df@{csv.stat().st_size}", {"n": calls["n"]})

            orig = ds._load_active
            ds._load_active = _fake_load
            try:
                with _EnvGuard(MI_AGENT_DATA_CSV=str(csv), MI_AGENT_DATA_CACHE_TTL="0",
                               MI_AGENT_PLATFORM_URI=None, MI_AGENT_ANALYTICS_DATASET=None,
                               MI_AGENT_CENTRAL_TAPE=None, MI_AGENT_ONBOARDING_OUTPUT_ROOT=None):
                    ds.reset_cache()
                    v1 = ds._active()
                    v2 = ds._active()                 # unchanged → served from cache
                    self.assertEqual(v1, v2)
                    self.assertEqual(calls["n"], 1)   # loaded once

                    csv.write_text("a,b\n1,2\n3,4\n5,6\n")   # republish (size changes)
                    v3 = ds._active()                 # signature changed → reload
                    self.assertNotEqual(v1[0], v3[0])
                    self.assertEqual(calls["n"], 2)
            finally:
                ds._load_active = orig
                ds.reset_cache()


class TestWeeklyPipelineBlobRoundTrip(unittest.TestCase):

    def _route_weekly(self, td, registry, persistence, central_csv):
        class _Inv:
            def __call__(self, **kw):
                return {"run_id": "orun", "status": "done",
                        "central_canonical_path": str(central_csv), "blockers": []}

        marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W02/_READY.json"
        pack = [str(Path(td) / "wk" / "PipelineExtract.csv")]
        role_schemas = R.role_schemas_for_pack(registry, marker, "raw-v2")
        schema = fingerprint_pack(pack, role_schemas=role_schemas)
        return R.handle_blob_event(
            marker, registry=registry, out_dir=str(Path(td) / "out"),
            container="raw-v2", pack_marker="_READY.json", schema_info=schema,
            input_dir_override=str(Path(td) / "wk"),
            orchestrator_invoker=_Inv(), persistence=persistence, now="2026-01-01T00:00:00+00:00")

    def test_weekly_run_persists_and_api_resolves_from_blob(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = Storage(root)
            layout = Layout()
            persistence = ProductionPersistence(storage, layout)
            registry = SourceRegistry("blob://trakt-state/registry/source_registry.yaml",
                                      storage=storage)
            # A pinned weekly pipeline source. The published snapshot must be the FULL
            # raw extract (rich fields like interest_rate), NOT the thin 18a tape.
            wk = root / "wk"
            wk.mkdir()
            (wk / "PipelineExtract.csv").write_text(
                "deal_id,amount,stage,interest_rate\nD1,250000,offer,0.062\n")
            repin_source(registry, client_id="ERE", source_portfolio_id="direct_001",
                         dataset="pipeline", frequency="weekly",
                         data_files=[str(wk / "PipelineExtract.csv")],
                         source_book_type="direct", regime_required=False)
            # The run's central tape is the THIN 18a (no rate) — must NOT be published.
            central = root / "central.csv"
            central.write_text("application_id,pipeline_stage\nD1,offer\n")

            m = self._route_weekly(td, registry, persistence, central)
            self.assertEqual(m["status"], "processed")
            self.assertEqual(m.get("pipeline_snapshot_source"), "raw_extract")
            self.assertTrue(m.get("pipeline_snapshot_uri"))
            local = persistence.pipeline_latest_path("ERE")
            self.assertIsNotNone(local)
            text = Path(local).read_text()
            self.assertIn("interest_rate", text)          # rich raw field published
            self.assertIn("0.062", text)
            self.assertNotIn("application_id", text)       # NOT the thin 18a tape

            # The MI API resolves the latest pipeline snapshot from the blob pointer.
            import mi_agent_api.app as app
            app._PIPELINE_URI_CACHE.update(etag=None, path=None)
            with _EnvGuard(TRAKT_STORAGE_BACKEND="file", TRAKT_LOCAL_BLOB_ROOT=str(root),
                           MI_AGENT_PIPELINE_URI="blob://processed-v2/pipeline/ERE/latest/latest_pipeline_snapshot.json"):
                resolved = app._resolve_pipeline_uri_local()
            self.assertIsNotNone(resolved)
            self.assertIn("interest_rate", Path(resolved).read_text())


if __name__ == "__main__":
    unittest.main()
