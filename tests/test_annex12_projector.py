from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.gate_4_projection.annex12_projector import (
    apply_identity_linkages,
    resolve_lei,
    resolve_reporting_date,
)


class TestAnnex12ProjectorResolution(unittest.TestCase):
    def test_resolve_lei_prefers_annex_config_then_master_default(self):
        annex_cfg = {"deal": {"IVSS1": "LEI_FROM_ANNEX"}}
        master_cfg = {"defaults": {"originator_legal_entity_identifier": "LEI_FROM_MASTER"}}
        self.assertEqual(resolve_lei(annex_cfg, master_cfg), "LEI_FROM_ANNEX")

        annex_cfg_missing = {"deal": {"IVSS5": "Contact"}}
        self.assertEqual(resolve_lei(annex_cfg_missing, master_cfg), "LEI_FROM_MASTER")

    def test_resolve_reporting_date_uses_canonical_derivation_column(self):
        annex_cfg = {"period": {}}
        canonical = pd.DataFrame({"data_cut_off_date": ["2025-11-30", "2025-11-30"]})
        self.assertEqual(resolve_reporting_date(annex_cfg, canonical, None), "2025-11-30")

    def test_resolve_reporting_date_rejects_non_deterministic_canonical_dates(self):
        annex_cfg = {"period": {}}
        canonical = pd.DataFrame({"data_cut_off_date": ["2025-10-31", "2025-11-30"]})
        with self.assertRaises(ValueError):
            resolve_reporting_date(annex_cfg, canonical, None)

    def test_apply_identity_linkages_keeps_annex12_identity_consistent(self):
        row = {"IVSS1": "LEI123", "IVSR1": "OLD", "IVSF1": "OLD"}
        apply_identity_linkages(row)
        self.assertEqual(row["IVSR1"], "LEI123")
        self.assertEqual(row["IVSF1"], "LEI123")


if __name__ == "__main__":
    unittest.main()
