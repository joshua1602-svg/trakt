#!/usr/bin/env python3
"""Phase 1B — the origination funnel must be invariant to the historical model.

``pipeline_funnel_evolution`` now passes the historical completion model down to
``load_prepared_pipeline`` so that it SHARES the prepared weekly frames with
``pipeline_evolution`` instead of preparing every extract a second time under a
different cache key. That is only sound because the funnel does not consume any
column the model influences.

These tests pin exactly that, in two independent ways:

  1. column-level — the two columns the funnel reads are byte-identical with and
     without the model, while the probability columns are the only ones that
     differ;
  2. output-level — the assembled funnel series is equal either way.

If a future change makes the funnel read a probability column, test (2) fails
and the sharing must be withdrawn.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack" / "pipeline"

#: The only columns ``pipeline_funnel_evolution`` reads off the prepared frame.
FUNNEL_CONSUMED_COLUMNS = ("pipeline_stage", "current_outstanding_balance")

#: The columns the historical model is allowed to change.
MODEL_AFFECTED_COLUMNS = {"completion_probability", "completion_probability_source",
                          "weighted_expected_funded_amount"}


class TestPreparedFrameInvariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _FIXTURE_ROOT.exists():
            raise unittest.SkipTest("fixture pack not present")
        from mi_agent_api import pipeline_contract as pc
        cls.pc = pc
        cls.model = pc.build_pipeline_history(str(_FIXTURE_ROOT), "client_001")
        extracts = pc.weekly_extract_inventory(str(_FIXTURE_ROOT), "client_001")["extracts"]
        if not extracts:
            raise unittest.SkipTest("no weekly extracts in the fixture pack")
        cls.extract = extracts[-1]

    def test_funnel_consumed_columns_are_identical(self):
        without, _ = self.pc.load_prepared_pipeline(self.extract)
        with_model, _ = self.pc.load_prepared_pipeline(
            self.extract, historical_model=self.model)
        for col in FUNNEL_CONSUMED_COLUMNS:
            if col not in without.columns:
                continue
            self.assertTrue(
                without[col].equals(with_model[col]),
                f"{col!r} changed when the historical model was supplied — the "
                "funnel may no longer share prepared frames")

    def test_only_probability_columns_differ(self):
        without, _ = self.pc.load_prepared_pipeline(self.extract)
        with_model, _ = self.pc.load_prepared_pipeline(
            self.extract, historical_model=self.model)
        differing = {
            c for c in set(without.columns) | set(with_model.columns)
            if c not in without.columns or c not in with_model.columns
            or not without[c].equals(with_model[c])
        }
        self.assertTrue(
            differing <= MODEL_AFFECTED_COLUMNS,
            f"the model changed unexpected columns: {sorted(differing - MODEL_AFFECTED_COLUMNS)}")


class TestFunnelOutputInvariance(unittest.TestCase):
    """The assembled funnel series must be equal with and without the model."""

    def test_funnel_series_is_identical(self):
        if not _FIXTURE_ROOT.exists():
            self.skipTest("fixture pack not present")
        from mi_agent_api import evolution as ev
        from mi_agent_api import pipeline_contract as pc

        model = pc.build_pipeline_history(str(_FIXTURE_ROOT), "client_001")
        without = ev.pipeline_funnel_evolution(str(_FIXTURE_ROOT), "client_001")
        with_model = ev.pipeline_funnel_evolution(
            str(_FIXTURE_ROOT), "client_001", historical_model=model)
        self.assertEqual(
            without, with_model,
            "the funnel output changed when the historical model was supplied")


if __name__ == "__main__":
    unittest.main()
