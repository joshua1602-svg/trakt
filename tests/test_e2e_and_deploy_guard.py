#!/usr/bin/env python3
"""tests/test_e2e_and_deploy_guard.py

Phase 5 — end-to-end render path + the deploy import guard.

E2E (filesystem backend, no Azure): a funded monthly pack is driven through the
router; its platform canonical is persisted to the durable store; the MI API's
data-source resolution then picks it up via MI_AGENT_PLATFORM_URI, and a SECOND
period overwrites the stable ``latest`` pointer so the newest data renders — the
source signature changes so the cache reloads (no restart).

Deploy guard: the CI workflow packages every runtime package the entrypoint
imports and its sanity check actually imports function_app — so an incomplete
deploy (the historical `No module named 'apps'` failure) fails the build.

Run: python -m unittest tests.test_e2e_and_deploy_guard
"""

from __future__ import annotations

import importlib.util
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

_FUNDED = ["loan_id", "current_balance", "current_ltv", "interest_rate", "origination_date"]


class _EnvGuard:
    def __init__(self, **kw):
        self.kw, self.saved = kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.saved[k] = os.environ.get(k)
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        return self

    def __exit__(self, *a):
        for k, v in self.saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _stub_assembler(**kw):
    return {}


class TestFundedRenderE2E(unittest.TestCase):

    def test_monthly_run_publishes_latest_and_signature_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage = Storage(root)
            persistence = ProductionPersistence(storage, Layout())
            registry = SourceRegistry("blob://trakt-state/registry/source_registry.yaml",
                                      storage=storage)
            src = root / "src"
            src.mkdir()
            (src / "LoanExtract.csv").write_text(",".join(_FUNDED) + "\n" + "1,100,0.5,0.03,2020-01-01\n")
            repin_source(registry, client_id="ERE", source_portfolio_id="direct_001",
                         dataset="funded", frequency="monthly",
                         data_files=[str(src / "LoanExtract.csv")],
                         source_book_type="direct", regime_required=False)

            def _run(period, central_body):
                central = root / f"central_{period}.csv"
                central.write_text(central_body)

                class _Inv:
                    def __call__(self, **kw):
                        return {"run_id": f"orun_{period}", "status": "done",
                                "central_canonical_path": str(central), "blockers": []}
                marker = f"raw-v2/ERE/direct/funded/monthly/direct_001/{period}/_READY.json"
                schema = fingerprint_pack([str(src / "LoanExtract.csv")],
                                          role_schemas=R.role_schemas_for_pack(registry, marker, "raw-v2"))
                return R.handle_blob_event(
                    marker, registry=registry, out_dir=str(root / "out"), container="raw-v2",
                    pack_marker="_READY.json", schema_info=schema,
                    input_dir_override=str(src), orchestrator_invoker=_Inv(),
                    assembler_refresher=_stub_assembler, persistence=persistence,
                    marker_metadata={"force_reprocess": True}, now=f"2026-01-01T00:00:00+00:00")

            hdr = ",".join(_FUNDED)
            m1 = _run("2025-10-31", hdr + "\n1,100,0.5,0.03,2020-01-01\n")
            self.assertEqual(m1["status"], "processed")
            self.assertTrue((root / "processed-v2/platform/ERE/latest/platform_canonical_typed.csv").exists())

            import mi_agent_api.data_source as ds
            with _EnvGuard(TRAKT_STORAGE_BACKEND="file", TRAKT_LOCAL_BLOB_ROOT=str(root),
                           MI_AGENT_DATA_CACHE_TTL="0",
                           MI_AGENT_PLATFORM_URI="blob://processed-v2/platform/ERE/latest/platform_canonical_typed.csv",
                           MI_AGENT_ANALYTICS_DATASET=None, MI_AGENT_CENTRAL_TAPE=None,
                           MI_AGENT_ONBOARDING_OUTPUT_ROOT=None, MI_AGENT_DATA_CSV=None):
                path1, base1 = ds.resolve_data_source()
                self.assertEqual(base1, "platform_canonical")
                self.assertIn("100", Path(path1).read_text())
                sig1 = ds._source_signature()

                # A second period overwrites the stable latest pointer.
                m2 = _run("2025-11-30", hdr + "\n1,90,0.45,0.03,2020-01-01\n2,250,0.6,0.04,2021-06-01\n")
                self.assertEqual(m2["status"], "processed")
                path2, base2 = ds.resolve_data_source()
                self.assertIn("250", Path(path2).read_text())     # newest data served
                sig2 = ds._source_signature()
                self.assertNotEqual(sig1, sig2)                   # cache would reload


class TestDeployImportGuard(unittest.TestCase):

    def _workflow_text(self):
        return (_REPO / ".github/workflows/main_trakt.yml").read_text()

    def test_deploy_packages_the_apps_package(self):
        wf = self._workflow_text()
        # Every runtime package the entrypoint imports must be zipped.
        for pkg in ("apps/", "engine/", "mi_agent/", "mi_agent_api/",
                    "mi_agent_pptx/", "analytics_lib/", "config/"):
            self.assertIn(pkg, wf, f"deploy package must include {pkg}")

    def test_ci_sanity_check_imports_function_app(self):
        wf = self._workflow_text()
        # py_compile never runs imports; the guard must actually import.
        self.assertIn("import function_app", wf)
        self.assertNotIn("python -m py_compile function_app.py", wf)

    def test_function_app_imports_when_azure_available(self):
        if importlib.util.find_spec("azure") is None or \
           importlib.util.find_spec("azure.functions") is None:
            self.skipTest("azure-functions not installed in this environment")
        # The entrypoint's module-top imports (apps.blob_trigger_app.*) must resolve.
        spec = importlib.util.spec_from_file_location(
            "function_app_guard", str(_REPO / "function_app.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(hasattr(mod, "app"))


if __name__ == "__main__":
    unittest.main()
