#!/usr/bin/env python3
"""tests/test_scoped_backfill_decisions.py

Scoped backfill as the rerun path for a single pack, and the approved
target-first decisions it must apply.

``ops rerun`` replays a run record's ``input_dir``, which points at an ephemeral
Azure run scratch that has usually been reclaimed — so it cannot localise the
source pack. Backfill re-fetches the pack's source files from the container on
every run, so scoping it to one pack is the rerun path that actually works. To be
equivalent to ``ops rerun`` it must also localise the pack's APPROVED decisions
from ``runs/{pack_key}/onboarding/34_target_first_decisions_approved.yaml`` and
hand them to the router, so onboarding applies the approved mappings BEFORE
readiness and the handoff package are computed.

Run: python -m pytest tests/test_scoped_backfill_decisions.py
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
from apps.blob_trigger_app import router as R
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.repin import repin_source
from apps.blob_trigger_app.source_registry import SourceRegistry
from apps.blob_trigger_app.storage import Storage
from tests.helpers import acquired_pack as AP

_LOAN = ["loan_id", "balance", "rate", "origination_date", "maturity_date"]
_APPROVED_YAML = (
    "decisions:\n"
    "  - target_field: protected_equity_percentage\n"
    "    source_column: Protected Equity\n"
    "    status: approved\n"
)


class _Invoker:
    """Records every orchestrator invocation the router makes.

    The pack scratch is a temporary directory that is gone by the time
    ``run_backfill`` returns, so the input directory's contents are captured HERE,
    while the run is in flight.
    """

    def __init__(self):
        self.calls = []
        self.input_files = []
        self.mapping_bodies = []

    def __call__(self, **kw):
        self.calls.append(kw)
        path = Path(kw.get("input_path", ""))
        self.input_files.append(
            sorted(p.name for p in path.iterdir()) if path.is_dir() else [])
        mapping = kw.get("mapping_config_path")
        self.mapping_bodies.append(
            Path(mapping).read_text(encoding="utf-8")
            if mapping and Path(mapping).is_file() else "")
        return {"run_id": f"orun_{len(self.calls)}", "status": "done",
                "central_canonical_path": None, "blockers": []}


def _stub_assembler(**kw):
    return {}


class _Ctx:
    def __init__(self, td: Path):
        self.root = td
        # The acquired ad-hoc pack, plus a monthly direct pack so scoping has
        # something it must NOT touch.
        AP.write_blob_tree(td)
        direct = (td / "raw-v2" / "ERE" / "direct" / "funded" / "monthly"
                  / "direct_001" / "2026-06-30")
        direct.mkdir(parents=True, exist_ok=True)
        (direct / "LoanExtract.csv").write_text(
            ",".join(_LOAN) + "\n" + ",".join(["x"] * len(_LOAN)) + "\n")
        (direct / "_READY.json").write_text("{}")

        self.storage = Storage(td)
        self.layout = Layout()
        self.persistence = ProductionPersistence(self.storage, self.layout)
        self.registry = SourceRegistry(
            "blob://trakt-state/registry/source_registry.yaml", storage=self.storage)
        repin_source(self.registry, client_id="ERE",
                     source_portfolio_id="direct_001", dataset="funded",
                     frequency="monthly",
                     data_files=[str(direct / "LoanExtract.csv")],
                     source_book_type="direct", regime_required=False)

    def approve(self, pack_key: str = AP.PACK_KEY, body: str = _APPROVED_YAML):
        """Persist an APPROVED decisions artefact for a pack, as ops would."""
        uri = self.layout.run_onboarding_uri(
            pack_key, "34_target_first_decisions_approved.yaml")
        self.storage.write_text(uri, body)
        return uri

    def run(self, **kw):
        kw.setdefault("orchestrator_invoker", _Invoker())
        kw.setdefault("assembler_refresher", _stub_assembler)
        kw.setdefault("out_dir", str(self.root / "out"))
        results = BF.run_backfill(
            self.storage, self.persistence, self.registry,
            container="raw-v2", **kw)
        return results, kw["orchestrator_invoker"]


class _Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.ctx = _Ctx(Path(self._td.name))


# --------------------------------------------------------------------------- #
# Scoping
# --------------------------------------------------------------------------- #
class TestScopedSelection(_Base):

    def test_pack_key_matches_the_router_key(self):
        # The scoped rerun must address the SAME run record the live event made.
        packs, _ = BF.scan_folders(
            self.ctx.storage, "raw-v2",
            selector=BF.PackSelector(source_portfolio_id="acquired_001"))
        self.assertEqual([p.pack_key for p in packs], [AP.PACK_KEY])
        self.assertEqual(
            R.pack_key_for(client_id="ERE", source_portfolio_id="acquired_001",
                           dataset="funded", frequency="ad_hoc",
                           reporting_period="2026-06-30"),
            AP.PACK_KEY)

    def test_ad_hoc_pack_is_out_of_the_default_historical_scope(self):
        # Control: the bulk sweep covers monthly funded + weekly pipeline only.
        packs, skipped = BF.scan_folders(self.ctx.storage, "raw-v2")
        self.assertNotIn(AP.PACK_KEY, [p.pack_key for p in packs])
        self.assertTrue(any("out_of_scope" in s["reason"] for s in skipped))

    def test_explicit_selection_lifts_the_scope_default(self):
        # An operator naming a pack means it, even at funded/ad_hoc.
        packs, _ = BF.scan_folders(
            self.ctx.storage, "raw-v2",
            selector=BF.PackSelector(pack_key=AP.PACK_KEY))
        self.assertEqual([p.pack_key for p in packs], [AP.PACK_KEY])

    def test_selection_excludes_every_other_pack(self):
        results, invoker = self.ctx.run(
            selector=BF.PackSelector(pack_key=AP.PACK_KEY), force=True)
        self.assertEqual([r["pack_key"] for r in results], [AP.PACK_KEY])
        self.assertEqual([c["source_portfolio_id"] for c in invoker.calls],
                         ["acquired_001"])

    def test_unmatched_selection_processes_nothing(self):
        results, invoker = self.ctx.run(
            selector=BF.PackSelector(pack_key="ERE_nope_funded_ad_hoc_2026-06-30"))
        self.assertEqual(results, [])
        self.assertEqual(invoker.calls, [])

    def test_selector_fields_compose(self):
        sel = BF.PackSelector(client_id="ERE", source_portfolio_id="acquired_001",
                              reporting_period="2026-06-30")
        packs, _ = BF.scan_folders(self.ctx.storage, "raw-v2", selector=sel)
        self.assertEqual([p.pack_key for p in packs], [AP.PACK_KEY])
        self.assertIn("acquired_001", sel.describe())

    def test_source_pack_is_localised_from_the_container(self):
        # The capability ops rerun lacks: the pack's files are re-fetched, so a
        # reclaimed run scratch is irrelevant.
        _results, invoker = self.ctx.run(
            selector=BF.PackSelector(pack_key=AP.PACK_KEY), force=True)
        self.assertIn(AP.FILE_NAME, invoker.input_files[0])


# --------------------------------------------------------------------------- #
# Approved decisions
# --------------------------------------------------------------------------- #
class TestApprovedDecisions(_Base):

    def _run_scoped(self, **kw):
        return self.ctx.run(selector=BF.PackSelector(pack_key=AP.PACK_KEY),
                            force=True, **kw)

    def test_approved_decisions_are_retrieved_and_applied(self):
        # 8 — from runs/{pack_key}/onboarding/34_target_first_decisions_approved.yaml
        self.ctx.approve()
        results, invoker = self._run_scoped()
        self.assertEqual(results[0]["approved_decisions"], "applied")
        applied = Path(results[0]["applied_decisions_path"])
        self.assertEqual(applied.name, BF.APPROVED_DECISIONS_NAME)
        # …reached the orchestrator as the mapping to apply, with the approved
        # content intact.
        self.assertEqual(invoker.calls[0]["mapping_config_path"], str(applied))
        self.assertIn("protected_equity_percentage", invoker.mapping_bodies[0])

    def test_approved_mappings_are_applied_before_readiness(self):
        # 9 — the run is routed DETERMINISTIC-apply, so onboarding consumes the
        # approved decisions in the same pass that computes readiness/handoff.
        self.ctx.approve()
        _results, invoker = self._run_scoped()
        self.assertEqual(invoker.calls[0]["processing_mode"], "deterministic")
        self.assertEqual(len(invoker.calls), 1)

    def test_decisions_artefact_is_kept_out_of_the_source_input_dir(self):
        # Onboarding inventories the input dir; a decisions file among the source
        # files would be treated as source.
        self.ctx.approve()
        results, invoker = self._run_scoped()
        input_dir = Path(invoker.calls[0]["input_path"])
        applied = Path(results[0]["applied_decisions_path"])
        self.assertNotEqual(applied.parent, input_dir)
        self.assertEqual(invoker.input_files[0], [AP.FILE_NAME])

    def test_missing_artefact_is_reported_and_never_substituted(self):
        # 10 — no approved decisions for THIS pack: say so, apply nothing.
        results, invoker = self._run_scoped()
        self.assertEqual(results[0]["approved_decisions"], "absent")
        self.assertEqual(results[0]["applied_decisions_path"], "")
        self.assertIsNone(invoker.calls[0].get("mapping_config_path"))

    def test_another_packs_approved_decisions_are_never_borrowed(self):
        # 10 — the artefact is looked up by THIS pack key only.
        self.ctx.approve(pack_key="ERE_direct_001_funded_monthly_2026-06-30")
        results, _invoker = self._run_scoped()
        self.assertEqual(results[0]["approved_decisions"], "absent")

    def test_require_flag_fails_loudly_when_the_artefact_is_absent(self):
        # 10 — an explicit "this must be an approved rerun" cannot pass silently.
        with self.assertRaises(BF.MissingApprovedDecisions) as ctx:
            self._run_scoped(require_approved_decisions=True)
        self.assertIn(AP.PACK_KEY, str(ctx.exception))
        self.assertIn("approve-recommendations", str(ctx.exception))

    def test_require_flag_passes_when_the_artefact_exists(self):
        self.ctx.approve()
        results, _ = self._run_scoped(require_approved_decisions=True)
        self.assertEqual(results[0]["approved_decisions"], "applied")

    def test_application_can_be_disabled_explicitly(self):
        self.ctx.approve()
        results, invoker = self._run_scoped(apply_approved_decisions=False)
        self.assertEqual(results[0]["approved_decisions"], "not_requested")
        self.assertIsNone(invoker.calls[0].get("mapping_config_path"))

    def test_dry_run_reports_the_plan_without_processing(self):
        self.ctx.approve()
        results, invoker = self._run_scoped(dry_run=True)
        self.assertEqual(results[0]["planned_route"], "approved_decisions→deterministic")
        self.assertEqual(invoker.calls, [])

    def test_localiser_returns_a_reason_rather_than_raising_by_default(self):
        path, status = BF.localise_approved_decisions(
            self.ctx.persistence, AP.PACK_KEY, str(self.ctx.root))
        self.assertIsNone(path)
        self.assertEqual(status, "absent")


class TestUnscopedBackfillUnaffected(_Base):
    """Backwards compatibility: the historical bulk sweep still behaves as before."""

    def test_default_sweep_processes_the_monthly_direct_pack(self):
        results, invoker = self.ctx.run()
        self.assertEqual([r["source_portfolio_id"] for r in results], ["direct_001"])
        self.assertEqual(len(invoker.calls), 1)

    def test_default_sweep_applies_approved_decisions_when_present(self):
        self.ctx.approve(pack_key="ERE_direct_001_funded_monthly_2026-06-30")
        results, _ = self.ctx.run()
        self.assertEqual(results[0]["approved_decisions"], "applied")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
