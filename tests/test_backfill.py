#!/usr/bin/env python3
"""tests/test_backfill.py

Phase 3 — historical backfill + durable idempotency.

Proves: the backfill enumerates monthly funded + weekly pipeline folders, drives
the SAME router code path chronologically, processes each pack exactly once
(auto-approving pinned recurring ones), and — thanks to the DURABLE trakt-state
run ledger — a re-run is a no-op unless --force. Dry-run produces a plan and
processes nothing.

Run: python -m unittest tests.test_backfill
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import backfill as BF
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.repin import repin_source
from apps.blob_trigger_app.source_registry import SourceRegistry
from apps.blob_trigger_app.storage import Storage

_LOAN = ["loan_id", "balance", "rate", "origination_date", "maturity_date"]
_PIPE = ["deal_id", "amount", "stage", "expected_close"]


class _Invoker:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return {"run_id": f"orun_{len(self.calls)}", "status": "done",
                "central_canonical_path": None, "blockers": []}


def _stub_assembler(**kw):
    return {}


class TestBackfill(unittest.TestCase):

    def _build_tree(self, root: Path):
        """raw-v2 container tree: two monthly funded periods + one weekly pipeline."""
        specs = [
            ("ERE/direct/funded/monthly/direct_001/2025-10-31", "LoanExtract.csv", _LOAN),
            ("ERE/direct/funded/monthly/direct_001/2025-11-30", "LoanExtract.csv", _LOAN),
            ("ERE/direct/pipeline/weekly/direct_001/2026-W01", "PipelineExtract.csv", _PIPE),
        ]
        for folder, fname, cols in specs:
            d = root / "raw-v2" / folder
            d.mkdir(parents=True, exist_ok=True)
            (d / fname).write_text(",".join(cols) + "\n" + ",".join(["x"] * len(cols)) + "\n")
            (d / "_READY.json").write_text("{}")

    def _ctx(self, td):
        root = Path(td)
        self._build_tree(root)
        storage = Storage(root)
        layout = Layout()
        persistence = ProductionPersistence(storage, layout)
        registry = SourceRegistry("blob://trakt-state/registry/source_registry.yaml",
                                  storage=storage)
        # Pin both sources from a representative pack so recurring packs auto-process.
        repin_source(registry, client_id="ERE", source_portfolio_id="direct_001",
                     dataset="funded", frequency="monthly",
                     data_files=[str(root / "raw-v2/ERE/direct/funded/monthly/direct_001/2025-10-31/LoanExtract.csv")],
                     source_book_type="direct", regime_required=False)
        repin_source(registry, client_id="ERE", source_portfolio_id="direct_001",
                     dataset="pipeline", frequency="weekly",
                     data_files=[str(root / "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W01/PipelineExtract.csv")],
                     source_book_type="direct", regime_required=False)
        return storage, persistence, registry

    def test_enumerate_chronological(self):
        with tempfile.TemporaryDirectory() as td:
            storage, _, _ = self._ctx(td)
            packs = BF.enumerate_packs(storage, "raw-v2")
            self.assertEqual([p.reporting_period for p in packs],
                             ["2025-10-31", "2025-11-30", "2026-W01"])
            self.assertEqual([p.dataset for p in packs], ["funded", "funded", "pipeline"])

    def test_backfill_processes_each_once_then_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            storage, persistence, registry = self._ctx(td)
            inv = _Invoker()
            res = BF.run_backfill(storage, persistence, registry, container="raw-v2",
                                  orchestrator_invoker=inv, assembler_refresher=_stub_assembler,
                                  out_dir=str(Path(td) / "out"))
            self.assertEqual(len(res), 3)
            self.assertTrue(all(r["status"] == "processed" for r in res))
            self.assertEqual(len(inv.calls), 3)             # each pack processed once

            # Re-run WITHOUT force → durable ledger makes it a no-op (no new invokes).
            inv2 = _Invoker()
            res2 = BF.run_backfill(storage, persistence, registry, container="raw-v2",
                                   orchestrator_invoker=inv2, assembler_refresher=_stub_assembler,
                                   out_dir=str(Path(td) / "out2"))  # fresh scratch → forces durable check
            self.assertTrue(all(r["status"] == "already_processed" for r in res2))
            self.assertEqual(len(inv2.calls), 0)

            # --force reprocesses.
            inv3 = _Invoker()
            res3 = BF.run_backfill(storage, persistence, registry, container="raw-v2",
                                   force=True, orchestrator_invoker=inv3,
                                   assembler_refresher=_stub_assembler,
                                   out_dir=str(Path(td) / "out3"))
            self.assertTrue(all(r["status"] == "processed" for r in res3))
            self.assertEqual(len(inv3.calls), 3)

    def test_dry_run_plans_without_processing(self):
        with tempfile.TemporaryDirectory() as td:
            storage, persistence, registry = self._ctx(td)
            inv = _Invoker()
            res = BF.run_backfill(storage, persistence, registry, container="raw-v2",
                                  dry_run=True, orchestrator_invoker=inv,
                                  out_dir=str(Path(td) / "out"))
            self.assertEqual(len(res), 3)
            self.assertTrue(all(r["planned_route"] == "deterministic" for r in res))
            self.assertEqual(len(inv.calls), 0)             # nothing processed
            # And the durable ledger is untouched.
            self.assertEqual(persistence.list_halted_runs(), [])

    def test_unpinned_new_source_plans_pending_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._build_tree(root)
            storage = Storage(root)
            layout = Layout()
            persistence = ProductionPersistence(storage, layout)
            registry = SourceRegistry("blob://trakt-state/registry/source_registry.yaml",
                                      storage=storage)  # EMPTY registry — nothing pinned
            res = BF.run_backfill(storage, persistence, registry, container="raw-v2",
                                  dry_run=True, out_dir=str(root / "out"))
            self.assertTrue(all(r["planned_route"].startswith("new_source") for r in res))


if __name__ == "__main__":
    unittest.main()
