#!/usr/bin/env python3
"""tests/test_managed_service_run_context.py

Managed-service run-context contract (blob-triggered / headless execution).

Locks in the guarantee that, in ``managed_service=True`` mode, ALL run context
is derived ONLY from the blob path / folder structure — never from a CLI-supplied
value, and never from an LLM or operator decision.

Reference pack (the production ERE funded-MI monthly pack):

    ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json

resolves to:

    client_id               = ERE
    source_book_type        = direct
    dataset (dataset_type)  = funded
    frequency               = monthly
    source_portfolio_id     = direct_001
    reporting_period        = 2025-11-30
    data_cut_off_date       = 2025-11-30
    data_cut_off_date_source = folder_period   (SRC_FOLDER_PERIOD)

and additionally asserts:
  * no CLI fallback is possible when ``managed_service=True``;
  * no LLM / operator decision artefact is consulted for run-context fields
    (resolution succeeds against a bare pack with no decision/LLM inputs);
  * ``path_parser`` remains the single source for path-derived context;
  * interactive CLI fallback (``managed_service=False``) is unchanged.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps.blob_trigger_app.path_parser import parse_blob_path, ParsedPath
from engine.onboarding_agent import run_context as rc
from engine.onboarding_agent import onboarding_handoff as oh

#: The exact reference blob path (7-segment production convention).
BLOB_PATH = "ERE/direct/funded/monthly/direct_001/2025-11-30/_READY.json"

#: The path-derived context the managed-service run must resolve, field-for-field.
EXPECTED_PATH_CONTEXT = {
    "client_id": "ERE",
    "source_book_type": "direct",
    "dataset": "funded",           # "dataset_type"
    "frequency": "monthly",
    "source_portfolio_id": "direct_001",
    "reporting_period": "2025-11-30",
}


def _bare_pack() -> Path:
    """A minimal onboarding project dir with NO decision/LLM artefacts and no
    date-bearing source column — so ONLY the folder period can resolve the date.
    This is what proves no operator/LLM input is required for run context."""
    root = Path(tempfile.mkdtemp(prefix="ms_ctx_"))
    with open(root / "01_file_inventory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_name", "file_path", "detected_reporting_date"])
        w.writeheader()
        w.writerow({"file_name": "LoanExtract.csv", "file_path": "", "detected_reporting_date": ""})
    central = root / "central.csv"
    with open(central, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["loan_identifier"])
    return root, central


class TestManagedServicePathContext(unittest.TestCase):
    """path_parser is the single source of path-derived run context."""

    def setUp(self):
        self.parsed = parse_blob_path(BLOB_PATH)

    def test_path_parser_derives_every_path_field(self):
        self.assertIsInstance(self.parsed, ParsedPath)
        self.assertEqual(self.parsed.client_id, "ERE")
        self.assertEqual(self.parsed.source_book_type, "direct")
        self.assertEqual(self.parsed.dataset, "funded")          # dataset_type
        self.assertEqual(self.parsed.frequency, "monthly")
        self.assertEqual(self.parsed.source_portfolio_id, "direct_001")
        self.assertEqual(self.parsed.reporting_period, "2025-11-30")
        self.assertEqual(self.parsed.filename, "_READY.json")
        # preferred (7-segment) structure — not the deprecated compat path
        self.assertFalse(self.parsed.is_legacy_path)

    def test_every_expected_field_matches(self):
        for field, expected in EXPECTED_PATH_CONTEXT.items():
            self.assertEqual(getattr(self.parsed, field), expected,
                             msg=f"{field} should be {expected!r}")

    def test_path_context_is_pure_string_derivation(self):
        # No filesystem, no registry, no decision/LLM artefact is touched to derive
        # path context — parsing the string alone yields the full identity. Parsing
        # is deterministic: the same path always yields the same context.
        again = parse_blob_path(BLOB_PATH)
        self.assertEqual(self.parsed, again)


class TestManagedServiceDateContext(unittest.TestCase):
    """data_cut_off_date is derived from the folder period, never cli_fallback."""

    def setUp(self):
        self.parsed = parse_blob_path(BLOB_PATH)
        self.root, self.central = _bare_pack()

    def _resolve(self, **over):
        kwargs = dict(folder_period=self.parsed.reporting_period, managed_service=True)
        kwargs.update(over)
        return rc.extract_data_cut_off_date(self.root, self.central, **kwargs)

    def test_data_cut_off_date_from_folder_period(self):
        r = self._resolve()
        self.assertEqual(r["value"], "2025-11-30")
        self.assertEqual(r["source"], rc.SRC_FOLDER_PERIOD)
        self.assertEqual(r["source"], "folder_period")
        self.assertFalse(r["missing"])
        self.assertFalse(r["conflict"])

    def test_no_cli_fallback_possible_in_managed_service(self):
        # Even with a CLI value present, managed mode never records cli_fallback:
        # the folder period is authoritative.
        r = self._resolve(cli_reporting_date="2025-06-30")
        self.assertEqual(r["value"], "2025-11-30")
        self.assertEqual(r["source"], rc.SRC_FOLDER_PERIOD)
        self.assertNotEqual(r["source"], rc.SRC_CLI_FALLBACK)

    def test_managed_service_without_any_period_surfaces_missing_not_cli(self):
        # With neither a folder period nor a source date, managed mode surfaces the
        # value as missing — it is NEVER smuggled in as a cli_fallback.
        r = rc.extract_data_cut_off_date(
            self.root, self.central, cli_reporting_date="2025-06-30", managed_service=True)
        self.assertNotEqual(r["source"], rc.SRC_CLI_FALLBACK)
        self.assertTrue(r["missing"])
        self.assertEqual(r["value"], "")

    def test_no_llm_or_operator_decision_required(self):
        # The bare pack carries NO 28c decision queue, NO 34 decisions, NO 36 LLM
        # recommendations, NO approval artefact — resolution still succeeds purely
        # from the blob folder period. Run context needs no human/LLM in the loop.
        for artefact in ("28c_human_decision_queue.json",
                         "34_target_first_decisions.yaml",
                         "36_target_first_llm_recommendations.json"):
            self.assertFalse((self.root / artefact).exists())
        r = self._resolve()
        self.assertEqual(r["value"], "2025-11-30")
        self.assertEqual(r["source"], rc.SRC_FOLDER_PERIOD)

    def test_handoff_manifest_source_label(self):
        # The onboarding handoff wrapper surfaces the resolver source under the
        # manifest key data_cut_off_date_source (== folder_period here).
        r = oh._resolve_run_context(
            self.root, self.root, str(self.central),
            reporting_period=self.parsed.reporting_period, managed_service=True)
        self.assertEqual(r["value"], "2025-11-30")
        # This is exactly the value onboarding_handoff writes as
        # manifest["data_cut_off_date_source"].
        self.assertEqual(r["source"], "folder_period")


class TestInteractiveCliUnchanged(unittest.TestCase):
    """Interactive CLI (managed_service=False) keeps the cli_fallback behaviour."""

    def test_cli_fallback_still_available_off_managed_service(self):
        root, central = _bare_pack()
        r = rc.extract_data_cut_off_date(
            root, central, cli_reporting_date="2025-06-30", managed_service=False)
        self.assertEqual(r["value"], "2025-06-30")
        self.assertEqual(r["source"], rc.SRC_CLI_FALLBACK)

    def test_cli_override_still_available_off_managed_service(self):
        root, central = _bare_pack()
        r = rc.extract_data_cut_off_date(
            root, central, cli_reporting_date="2025-06-30", override_reporting_date=True,
            managed_service=False)
        self.assertEqual(r["value"], "2025-06-30")
        self.assertEqual(r["source"], rc.SRC_CLI_OVERRIDE)


if __name__ == "__main__":
    unittest.main()
