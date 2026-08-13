#!/usr/bin/env python3
"""tests/test_synthetic_demo_pack_isolation.py

``synthetic_demo/input`` is a SINGLE-PORTFOLIO, SINGLE-PERIOD onboarding pack,
and nine test modules hand that directory to the onboarding workflow as one
client delivery. Onboarding walks a pack recursively — correctly, because a real
delivery is several files — so anything written underneath that directory
becomes part of the pack.

This guard exists because that happened. The multi-book demonstration generator
wrote six extracts spanning two further month-ends into
``synthetic_demo/input/multibook``. The onboarding run then saw three different
reporting dates in one delivery, so ``run_context`` raised a cut-off-date
conflict — a blocking operator decision, exactly as designed — and
``ready_for_transformation_validation`` went false. Eight tests across three
modules turned red, and the orchestrator's real-agent run halted, none of which
was a product regression: the pack had stopped being the thing the tests
described.

The refusal is right and stays. What is asserted here is the fixture's contract:
this pack carries one book for one period, so a test that onboards it is
onboarding what it thinks it is. Generated datasets that model several books or
periods belong beside the pack, not inside it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PACK = _REPO_ROOT / "synthetic_demo" / "input"
MULTIBOOK = _REPO_ROOT / "synthetic_demo" / "multibook_input"


class TestSyntheticDemoPackIsolation(unittest.TestCase):

    def test_pack_is_flat(self):
        """No subdirectories: onboarding recurses, so a subdirectory is part of
        the delivery whether or not it was meant to be."""
        subdirs = [p.name for p in PACK.iterdir() if p.is_dir()]
        self.assertEqual(subdirs, [],
                         f"{PACK.relative_to(_REPO_ROOT)} must stay flat — these "
                         f"directories would be onboarded as part of the pack: "
                         f"{subdirs}")

    def test_pack_is_one_source_file(self):
        files = sorted(p.name for p in PACK.iterdir() if p.is_file())
        self.assertEqual(files, ["SYNTHETIC_ERE_Portfolio_012026.csv"],
                         "the single-book pack carries exactly one source extract")

    def test_multibook_extracts_live_outside_the_pack(self):
        self.assertTrue(MULTIBOOK.is_dir(),
                        "the multi-book demonstration extracts must exist")
        self.assertFalse(str(MULTIBOOK).startswith(str(PACK) + "/"),
                         "the multi-book extracts must not sit inside the "
                         "single-book onboarding pack")
        self.assertTrue(any(MULTIBOOK.glob("*.csv")))


if __name__ == "__main__":
    unittest.main()
