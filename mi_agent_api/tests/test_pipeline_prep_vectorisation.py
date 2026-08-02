#!/usr/bin/env python3
"""Phase 1B — the vectorised pipeline preparation must be semantically identical.

Two per-row loops in ``pipeline_prep`` were replaced with mask-based pandas
operations:

  * ``_derive_expected_completion`` — filled ``expected_completion_date`` row by
    row with ``.at[]`` reads and writes;
  * ``_derive_probabilities_and_amounts`` — applied the governed probability
    hierarchy with a per-row ``if/elif`` chain and ``.loc[]`` scalar writes.

The governed hierarchy is a business rule, so these tests exercise every tier
and every precedence boundary directly, rather than trusting an aggregate.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from mi_agent_api.pipeline_prep import (
    _derive_expected_completion,
    _derive_probabilities_and_amounts,
)

STAGE_PROBS = {"KFI": 0.20, "APPLICATION": 0.50, "OFFER": 0.80, "COMPLETED": 1.0}
HISTORICAL = {"KFI": 0.31, "OFFER": 0.77}


class TestProbabilityHierarchy(unittest.TestCase):
    """Each tier of the governed hierarchy, and each precedence boundary."""

    def _run(self, stages, explicit=None, historical=HISTORICAL, configured=STAGE_PROBS):
        out = pd.DataFrame({"pipeline_stage": stages})
        if explicit is not None:
            out["completion_probability"] = explicit
        _derive_probabilities_and_amounts(out, configured, historical, [])
        return out["completion_probability"], out["completion_probability_source"]

    def test_row_level_wins_over_every_other_tier(self):
        prob, src = self._run(["KFI", "OFFER", "WITHDRAWN"], explicit=[0.9, 0.1, 0.4])
        self.assertEqual(list(src), ["row_level"] * 3)
        self.assertEqual(list(prob), [0.9, 0.1, 0.4])

    def test_withdrawn_beats_historical_and_configured(self):
        prob, src = self._run(["WITHDRAWN"])
        self.assertEqual(src.iloc[0], "excluded_withdrawn")
        self.assertTrue(pd.isna(prob.iloc[0]), "withdrawn must carry no probability")

    def test_historical_beats_configured(self):
        prob, src = self._run(["KFI"])
        self.assertEqual(src.iloc[0], "historical_stage_rate")
        self.assertEqual(prob.iloc[0], 0.31)

    def test_configured_used_when_no_historical_rate(self):
        prob, src = self._run(["APPLICATION"])
        self.assertEqual(src.iloc[0], "configured_stage_rate")
        self.assertEqual(prob.iloc[0], 0.50)

    def test_unknown_stage_is_missing_stage(self):
        for token in ("UNKNOWN", "", "nan", "None"):
            prob, src = self._run([token])
            self.assertEqual(src.iloc[0], "missing_stage", token)
            self.assertTrue(pd.isna(prob.iloc[0]), token)

    def test_unmapped_stage_is_unavailable(self):
        prob, src = self._run(["SOME_NEW_STAGE"])
        self.assertEqual(src.iloc[0], "unavailable")
        self.assertTrue(pd.isna(prob.iloc[0]))

    def test_partial_explicit_falls_through_per_row(self):
        """A row without an explicit value must still reach the lower tiers."""
        prob, src = self._run(["KFI", "KFI", "WITHDRAWN"],
                              explicit=[0.9, np.nan, np.nan])
        self.assertEqual(list(src),
                         ["row_level", "historical_stage_rate", "excluded_withdrawn"])
        self.assertEqual(prob.iloc[0], 0.9)
        self.assertEqual(prob.iloc[1], 0.31)
        self.assertTrue(pd.isna(prob.iloc[2]))

    def test_empty_rate_tables_degrade_to_missing_or_unavailable(self):
        prob, src = self._run(["KFI", "UNKNOWN"], historical={}, configured={})
        self.assertEqual(list(src), ["unavailable", "missing_stage"])
        self.assertTrue(prob.isna().all())

    def test_empty_frame_is_handled(self):
        out = pd.DataFrame({"pipeline_stage": pd.Series([], dtype=object)})
        _derive_probabilities_and_amounts(out, STAGE_PROBS, HISTORICAL, [])
        self.assertEqual(len(out), 0)
        self.assertIn("completion_probability_source", out.columns)

    def test_non_default_index_is_respected(self):
        """Masks must align on labels, not positions."""
        out = pd.DataFrame({"pipeline_stage": ["KFI", "WITHDRAWN"]}, index=[7, 3])
        _derive_probabilities_and_amounts(out, STAGE_PROBS, HISTORICAL, [])
        self.assertEqual(out.loc[7, "completion_probability_source"],
                         "historical_stage_rate")
        self.assertEqual(out.loc[3, "completion_probability_source"],
                         "excluded_withdrawn")


class TestExpectedCompletionDerivation(unittest.TestCase):
    DAYS = {"KFI": 90, "OFFER": 30}

    def test_fills_only_missing_dates(self):
        out = pd.DataFrame({
            "pipeline_stage": ["KFI", "OFFER"],
            "expected_completion_date": [pd.Timestamp("2026-05-05"), pd.NaT],
        })
        derived: list = []
        _derive_expected_completion(out, pd.Timestamp("2026-01-01"), self.DAYS, derived)
        self.assertEqual(out["expected_completion_date"].iloc[0],
                         pd.Timestamp("2026-05-05"), "an existing date must not move")
        self.assertEqual(out["expected_completion_date"].iloc[1],
                         pd.Timestamp("2026-01-31"))

    def test_stage_without_a_configured_offset_is_left_alone(self):
        out = pd.DataFrame({"pipeline_stage": ["APPLICATION"],
                            "expected_completion_date": [pd.NaT]})
        _derive_expected_completion(out, pd.Timestamp("2026-01-01"), self.DAYS, [])
        self.assertTrue(pd.isna(out["expected_completion_date"].iloc[0]))

    def test_per_row_base_series_is_used_when_no_reporting_timestamp(self):
        out = pd.DataFrame({
            "pipeline_stage": ["KFI", "KFI"],
            "pipeline_stage_date": [pd.Timestamp("2026-01-01"), pd.NaT],
            "expected_completion_date": [pd.NaT, pd.NaT],
        })
        _derive_expected_completion(out, None, self.DAYS, [])
        self.assertEqual(out["expected_completion_date"].iloc[0],
                         pd.Timestamp("2026-04-01"))
        self.assertTrue(pd.isna(out["expected_completion_date"].iloc[1]),
                        "a row with no base date must stay unfilled")

    def test_derived_is_recorded_only_when_something_was_filled(self):
        derived: list = []
        out = pd.DataFrame({"pipeline_stage": ["APPLICATION"],
                            "expected_completion_date": [pd.NaT]})
        _derive_expected_completion(out, pd.Timestamp("2026-01-01"), self.DAYS, derived)
        self.assertEqual(derived, [], "nothing filled -> nothing derived")

        derived2: list = []
        out2 = pd.DataFrame({"pipeline_stage": ["KFI"],
                             "expected_completion_date": [pd.NaT]})
        _derive_expected_completion(out2, pd.Timestamp("2026-01-01"), self.DAYS, derived2)
        self.assertEqual(derived2, ["expected_completion_date"])

    def test_non_default_index_is_respected(self):
        out = pd.DataFrame({"pipeline_stage": ["KFI", "OFFER"],
                            "expected_completion_date": [pd.NaT, pd.NaT]},
                           index=[11, 2])
        _derive_expected_completion(out, pd.Timestamp("2026-01-01"), self.DAYS, [])
        self.assertEqual(out.loc[11, "expected_completion_date"],
                         pd.Timestamp("2026-04-01"))
        self.assertEqual(out.loc[2, "expected_completion_date"],
                         pd.Timestamp("2026-01-31"))


if __name__ == "__main__":
    unittest.main()
