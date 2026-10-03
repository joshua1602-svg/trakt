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
#: Validity windows the history MEASURED for the weighted stages (D21: a stage
#: with no measured window is undetermined, so the rate tiers are reached only
#: through one). No run-off hazards, so the run-off tier is not reached.
MEASURED_WINDOWS = {"available": False, "stages": {
    "APPLICATION": {"windowDays": 28, "windowBasis": "measured"},
    "OFFER": {"windowDays": 126, "windowBasis": "measured"}}}


class TestProbabilityHierarchy(unittest.TestCase):
    """Each tier of the governed hierarchy, and each precedence boundary.

    The tier MECHANICS are tested with every open stage forecast; which stages
    the shipped config forecasts (KFI is top of funnel, not weighted) is
    pinned separately in ``TestForecastStages``."""

    def setUp(self):
        from unittest import mock
        from mi_agent_api import pipeline_prep
        patcher = mock.patch.object(pipeline_prep, "_forecast_stages", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, stages, explicit=None, historical=HISTORICAL, configured=STAGE_PROBS,
             runoff=MEASURED_WINDOWS):
        out = pd.DataFrame({"pipeline_stage": stages})
        if explicit is not None:
            out["completion_probability"] = explicit
        _derive_probabilities_and_amounts(out, configured, historical, [],
                                          runoff=runoff)
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

    def test_no_measured_rate_is_undetermined_not_configured(self):
        """D21: a weighted stage the history cannot rate carries no
        probability — the configured 0.50 never stands in."""
        prob, src = self._run(["APPLICATION"])
        self.assertEqual(src.iloc[0], "insufficient_history_application")
        self.assertTrue(pd.isna(prob.iloc[0]))

    def test_no_measured_window_is_undetermined(self):
        """D21: whether a case has lapsed needs its stage's measured window;
        without one it is neither weighted nor lapsed, whatever the rate."""
        prob, src = self._run(["OFFER"], runoff={"available": False, "stages": {}})
        self.assertEqual(src.iloc[0], "insufficient_history_offer_window")
        self.assertTrue(pd.isna(prob.iloc[0]))

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


class TestForecastStages(unittest.TestCase):
    """The shipped config forecasts Applications and Offers; a KFI is open
    pipeline with zero expected-funding weight (the run-off method treats the
    KFI stage as reference only)."""

    def test_a_kfi_is_not_forecast(self):
        out = pd.DataFrame({"pipeline_stage": ["KFI", "OFFER", "WITHDRAWN"],
                            "current_outstanding_balance": [100.0, 200.0, 300.0]})
        _derive_probabilities_and_amounts(out, STAGE_PROBS, HISTORICAL, [],
                                          runoff=MEASURED_WINDOWS)
        src = list(out["completion_probability_source"])
        self.assertEqual(src, ["not_forecast_kfi", "historical_stage_rate",
                               "excluded_withdrawn"])
        self.assertEqual(out["completion_probability"].iloc[0], 0.0)
        self.assertEqual(out["weighted_expected_funded_amount"].iloc[0], 0.0)


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
