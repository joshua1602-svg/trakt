#!/usr/bin/env python3
"""tests/test_annex2_collateral_projection.py — RREC5 through the real projector.

The unit-level table checks live in
``tests/test_annex2_collateral_enum_mapping.py``. This file proves the property
that actually matters: running the PRODUCTION ``engine/gate_4_projection/
regime_projector.py`` with PRODUCTION ``config/system/enum_mapping.yaml``
projects a real-data collateral value to an XSD-valid ``CollTp`` — **without**
the demo-only overlay in ``demo_platform/artefacts.py::_demo_enum_mapping``.

Before the correction the projector emitted the raw string
``'Residential property'`` into RREC5, because its synonym resolver discards a
synonym whose TARGET is absent from the regime table. The 105/107 benchmark was
only ever reachable through the demo overlay.

Run: python -m pytest tests/test_annex2_collateral_projection.py
"""

from __future__ import annotations

import glob
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_PROJECTOR = _REPO / "engine" / "gate_4_projection" / "regime_projector.py"
_REGISTRY = _REPO / "config" / "system" / "fields_registry.yaml"
_ENUM_MAPPING = _REPO / "config" / "system" / "enum_mapping.yaml"
_CODE_ORDER = _REPO / "config" / "system" / "esma_code_order.yaml"
_CLIENT_CONFIG = _REPO / "demo_platform" / "config" / "config_client_ALDERBRIDGE_DEMO.yaml"
_XSD = _REPO / "config" / "system" / "DRAFT1auth.099.001.04_1.3.0.xsd"
#: A real-style canonical committed in the repository.
_SAMPLE = _REPO / "ERE_Portfolio_122025_ESMA_Annex2_canonical_ESMA_Annex2_typed.csv"


def _collateral_enumeration() -> set:
    text = _XSD.read_text(encoding="utf-8", errors="replace")
    block = re.search(
        r'<xs:simpleType name="CollateralType7Code">(.*?)</xs:simpleType>',
        text, re.S).group(1)
    return set(re.findall(r'value="([^"]+)"', block))


@unittest.skipUnless(_SAMPLE.exists(), "sample canonical not present")
class TestCollateralResolvesWithoutTheDemoOverlay(unittest.TestCase):
    """Production configuration alone must produce a valid CollTp.

    This drives the PRODUCTION function that decides the value —
    ``regime_projector.apply_enum_mappings`` — with the PRODUCTION registry and
    the PRODUCTION ``enum_mapping.yaml``, over the real-data value the
    committed ERE canonical actually carries.

    Driving the whole projector CLI is deliberately NOT done here: this sample
    is a real-style extract, not a complete Annex 2 tape (``RREL5`` is blank on
    every row, and several strict enum fields carry ND codes), so a full run
    fails for reasons that have nothing to do with collateral. The end-to-end
    proof lives in the 105/107 benchmark reproduction instead.
    """

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_REPO / "engine" / "gate_4_projection"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("_rrec5_projector", _PROJECTOR)
        cls.projector = importlib.util.module_from_spec(spec)
        sys.modules["_rrec5_projector"] = cls.projector
        spec.loader.exec_module(cls.projector)

        frame = pd.read_csv(_SAMPLE, low_memory=False, nrows=25)
        cls.input_values = sorted(set(frame["collateral_type"].dropna().astype(str)))
        cls.registry = yaml.safe_load(_REGISTRY.read_text(encoding="utf-8"))
        cls.mapping = yaml.safe_load(_ENUM_MAPPING.read_text(encoding="utf-8"))
        cls.frame = frame[["collateral_type"]].copy()

    def _project(self, mapping):
        out, _report = self.projector.apply_enum_mappings(
            self.frame.copy(), self.registry, mapping, "ESMA_Annex2",
            allow_unreviewed=True)
        return sorted(set(out["collateral_type"].dropna().astype(str)))

    def test_the_sample_carries_the_labelled_real_data_form(self):
        self.assertEqual(self.input_values, ["Residential Building (RBLD)"])

    def test_production_mapping_resolves_it_to_a_valid_code(self):
        self.assertEqual(self._project(self.mapping), ["RBLD"])
        self.assertTrue({"RBLD"} <= _collateral_enumeration())

    def test_the_raw_label_does_not_survive(self):
        """The pre-correction failure mode, pinned."""
        values = self._project(self.mapping)
        self.assertNotIn("Residential Building (RBLD)", values)
        self.assertNotIn("Residential property", values)

    def test_no_legacy_code_appears(self):
        values = set(self._project(self.mapping))
        for code in ("R1", "R2", "C1", "C2"):
            self.assertNotIn(code, values)

    def test_an_unsupported_value_is_not_silently_made_valid(self):
        """An unrecognised value stays unrecognised so the XSD can reject it."""
        frame = pd.DataFrame({"collateral_type": ["Orbital Platform (ZZZZ)"]})
        out, _ = self.projector.apply_enum_mappings(
            frame, self.registry, self.mapping, "ESMA_Annex2",
            allow_unreviewed=True)
        value = out["collateral_type"].iloc[0]
        self.assertEqual(value, "Orbital Platform (ZZZZ)")
        self.assertNotIn(value, _collateral_enumeration())


class TestProductionMappingCoversTheDemoOverlay(unittest.TestCase):
    """The demo overlay must now be a no-op for collateral_type.

    ``_demo_enum_mapping`` ``setdefault``s an identity entry for every CollTp
    code. If production already carries them, the overlay adds nothing — which
    is the evidence that production no longer depends on it.
    """

    def test_production_already_has_every_identity_the_overlay_would_add(self):
        table = yaml.safe_load(_ENUM_MAPPING.read_text(encoding="utf-8"))
        table = table["ESMA_Annex2"]["collateral_type"]
        missing = sorted(c for c in _collateral_enumeration()
                         if table.get(c) != c)
        self.assertEqual(missing, [],
                         f"the demo overlay would still be adding: {missing}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
