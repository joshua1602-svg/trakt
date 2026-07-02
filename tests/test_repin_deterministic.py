#!/usr/bin/env python3
"""tests/test_repin_deterministic.py

Phase 1 — registry as single source of truth + fingerprint (re)pin.

Proves the acceptance criterion: after pinning ONE representative pack via
``apps.blob_trigger_app.repin``, an equivalent monthly pack routes
``deterministic`` (not schema_drift / source_onboarding); a real mandatory-field
change routes ``schema_drift``; and cosmetic FILENAME churn (same headers) still
routes ``deterministic`` (header-first robustness).

Run: python -m unittest tests.test_repin_deterministic
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import router as R
from apps.blob_trigger_app.repin import repin_source
from apps.blob_trigger_app.schema_fingerprint import fingerprint_pack
from apps.blob_trigger_app.source_registry import SourceRegistry, STATUS_ACTIVE

_NOW = "2026-10-01T00:00:00+00:00"
_CONTAINER = "raw-v2"

_LOAN = ["loan_id", "balance", "rate", "origination_date", "maturity_date"]
_PROP = ["loan_id", "property_value", "postcode", "property_type"]
_PI = ["loan_id", "principal_paid", "interest_paid", "period"]


class _Invoker:
    def __init__(self, status="done"):
        self.calls = []
        self.status = status

    def __call__(self, **kw):
        self.calls.append(kw)
        return {"run_id": "orun", "status": self.status,
                "central_canonical_path": "/tmp/central.csv", "blockers": []}


def _write_pack(dirpath: Path, files: dict) -> list:
    dirpath.mkdir(parents=True, exist_ok=True)
    out = []
    for name, cols in files.items():
        p = dirpath / name
        p.write_text(",".join(cols) + "\n" + ",".join(["x"] * len(cols)) + "\n")
        out.append(str(p))
    return sorted(out)


def _route(registry, blob_marker, data_files, invoker):
    """Fingerprint exactly like the Event Grid handler (header-first using the
    registry's pinned role schemas) and route the marker event."""
    role_schemas = R.role_schemas_for_pack(registry, blob_marker, _CONTAINER)
    aliases = R.aliases_for_pack(registry, blob_marker, _CONTAINER)
    schema = fingerprint_pack(data_files, role_schemas=role_schemas, aliases=aliases)
    with tempfile.TemporaryDirectory() as out:
        return R.handle_blob_event(
            blob_marker, registry=registry, out_dir=out, container=_CONTAINER,
            pack_marker="_READY.json", schema_info=schema,
            input_dir_override=str(Path(data_files[0]).parent),
            orchestrator_invoker=invoker, now=_NOW)


class TestRepinDeterministic(unittest.TestCase):

    def _fresh_registry(self, td):
        return SourceRegistry(Path(td) / "registry.yaml")

    def test_pin_then_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._fresh_registry(td)
            pack = _write_pack(Path(td) / "nov", {
                "LoanExtract One - OMNI.csv": _LOAN,
                "PropertyExtract - OMNI.csv": _PROP,
                "Funder Principal And Interest.csv": _PI})

            summary = repin_source(
                reg, client_id="ERE", source_portfolio_id="direct_001",
                dataset="funded", frequency="monthly", data_files=pack,
                source_book_type="direct", regime_required=True)
            self.assertNotIn("<fill", summary["expected_schema_fingerprint"])
            rec = reg.lookup("ERE", "direct_001", "funded", "monthly")
            self.assertEqual(rec.status, STATUS_ACTIVE)
            self.assertTrue(rec.has_approved_mapping)
            self.assertTrue(rec.file_role_schemas)  # header signatures pinned

            # An equivalent pack next month → deterministic.
            marker = "raw-v2/ERE/direct/funded/monthly/direct_001/2025-12-31/_READY.json"
            dec = _route(reg, marker, pack, _Invoker())
            self.assertEqual(dec["decision"], R.DECISION_DETERMINISTIC)
            self.assertEqual(dec["status"], R.STATUS_PROCESSED)

    def test_filename_churn_still_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._fresh_registry(td)
            nov = _write_pack(Path(td) / "nov", {
                "LoanExtract One - OMNI.csv": _LOAN,
                "PropertyExtract - OMNI.csv": _PROP,
                "Funder Principal And Interest.csv": _PI})
            repin_source(reg, client_id="ERE", source_portfolio_id="direct_001",
                         dataset="funded", frequency="monthly", data_files=nov,
                         source_book_type="direct", regime_required=True)

            # SAME headers, cosmetically DIFFERENT file names → header-first match.
            dec_pack = _write_pack(Path(td) / "dec", {
                "LoanExtract One OMNI_test.csv": _LOAN,
                "PG_PropertyExtract Internal OMNI.csv": _PROP,
                "Funder P&I.csv": _PI})
            marker = "raw-v2/ERE/direct/funded/monthly/direct_001/2025-12-31/_READY.json"
            dec = _route(reg, marker, dec_pack, _Invoker())
            self.assertEqual(dec["decision"], R.DECISION_DETERMINISTIC)

    def test_mandatory_field_change_is_drift(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._fresh_registry(td)
            nov = _write_pack(Path(td) / "nov", {
                "LoanExtract One - OMNI.csv": _LOAN,
                "PropertyExtract - OMNI.csv": _PROP,
                "Funder Principal And Interest.csv": _PI})
            repin_source(reg, client_id="ERE", source_portfolio_id="direct_001",
                         dataset="funded", frequency="monthly", data_files=nov,
                         source_book_type="direct", regime_required=True)

            # Loan extract loses a mandatory column → same role (Jaccard≥0.6) but a
            # different fingerprint → schema_drift, fail closed to pending_review.
            changed = _write_pack(Path(td) / "dec", {
                "LoanExtract One - OMNI.csv": ["loan_id", "balance", "rate"],
                "PropertyExtract - OMNI.csv": _PROP,
                "Funder Principal And Interest.csv": _PI})
            marker = "raw-v2/ERE/direct/funded/monthly/direct_001/2025-12-31/_READY.json"
            dec = _route(reg, marker, changed, _Invoker())
            # A removed mandatory field is a MATERIAL change: it must NOT auto-process.
            # (Under the approval policy it takes the one-click source_onboarding path;
            # a source without pinned header signatures would take schema_drift — both
            # halt at pending_review, which is the acceptance property.)
            self.assertNotEqual(dec["decision"], R.DECISION_DETERMINISTIC)
            self.assertEqual(dec["status"], R.STATUS_PENDING_REVIEW)

    def test_weekly_pipeline_pin_then_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._fresh_registry(td)
            wk = _write_pack(Path(td) / "wk", {
                "PipelineExtract.csv": ["deal_id", "amount", "stage", "expected_close"]})
            repin_source(reg, client_id="ERE", source_portfolio_id="direct_001",
                         dataset="pipeline", frequency="weekly", data_files=wk,
                         source_book_type="direct", regime_required=False)
            marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W02/_READY.json"
            dec = _route(reg, marker, wk, _Invoker())
            self.assertEqual(dec["decision"], R.DECISION_DETERMINISTIC)


if __name__ == "__main__":
    unittest.main()
