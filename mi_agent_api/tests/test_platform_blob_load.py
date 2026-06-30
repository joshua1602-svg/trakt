#!/usr/bin/env python3
"""tests/test_platform_blob_load — MI API loads the persisted platform canonical.

Confirms the production wiring: with TRAKT_STORAGE_BACKEND set and
MI_AGENT_PLATFORM_URI=blob://…, the MI API resolves and loads the central
platform canonical from the (storage-abstraction) blob store and reports
dataSourceKind=platform_canonical on /health. Uses the filesystem-emulated blob
backend so no Azure account is needed.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from fastapi.testclient import TestClient

_REPO = Path(__file__).resolve().parents[2]
_SYNTH = _REPO / "synthetic_demo" / "output" / "SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv"


class TestPlatformBlobLoad(unittest.TestCase):

    @unittest.skipUnless(_SYNTH.exists(), "synthetic canonical fixture not present")
    def test_health_reports_platform_canonical_from_blob(self):
        with tempfile.TemporaryDirectory() as td:
            blobroot = Path(td) / "blobstore"
            uri = "blob://processed-v2/platform/ERE/latest/platform_canonical_typed.csv"
            dest = blobroot / "processed-v2" / "platform" / "ERE" / "latest" / "platform_canonical_typed.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(_SYNTH)
            if "source_portfolio_id" not in df.columns:
                df["source_portfolio_id"] = "direct_001"
            df.to_csv(dest, index=False)

            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("MI_AGENT_")
                   and k not in ("WEBSITE_INSTANCE_ID", "WEBSITE_SITE_NAME",
                                 "TRAKT_BLOB_CONNECTION")}
            env.update({
                "TRAKT_STORAGE_BACKEND": "file",
                "TRAKT_LOCAL_BLOB_ROOT": str(blobroot),
                "MI_AGENT_PLATFORM_URI": uri,
                "MI_AGENT_SCRATCH": str(Path(td) / "scratch"),
            })
            from mi_agent_api import data_source as DS
            with mock.patch.dict(os.environ, env, clear=True):
                DS.reset_cache()                       # re-resolve under this env
                try:
                    from mi_agent_api.app import app
                    client = TestClient(app)
                    h = client.get("/health").json()
                finally:
                    DS.reset_cache()                   # don't leak the temp path

            self.assertTrue(h["ok"])
            self.assertEqual(h["dataSourceKind"], "platform_canonical")
            self.assertTrue(h["dataAvailable"])
            self.assertEqual(h["dataSource"], "platform_canonical_typed.csv")


if __name__ == "__main__":
    unittest.main()
