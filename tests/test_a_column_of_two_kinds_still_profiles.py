"""A lender's column is not always one type, and profiling must survive that.

The delivery's halt, this time WITH its location, because the swallow above it
had just been taught to record one::

    [mapping_review] error: TypeError: '<=' not supported between instances of
                            'str' and 'float'
    [mapping_review] where: engine/onboarding_agent/column_evidence.py:255
                            in build_column_evidence

      File ".../column_evidence.py", line 255, in build_column_evidence
        "min_value": str(non_null.min()) if len(non_null) else "",
      File ".../pandas/core/series.py", line 6528, in min
      ...
      File ".../numpy/_core/_methods.py", line 45, in _amin
        return umr_minimum(a, axis, None, out, keepdims, initial, where)
    TypeError: '<=' not supported between instances of 'str' and 'float'

`Current Balance` carries amounts and the word "N/A" where an amount is
unknown. A valuation date carries dates and "n/k". `pandas` holds that as an
object column, `Series.min()` hands it to `numpy`, and `numpy` orders the
values pairwise — raising on the first comparison between the two kinds.

WHAT IT COST IS OUT OF ALL PROPORTION TO WHAT IT IS. This row is descriptive:
the smallest and largest value, for a profile a human reads. It is not a
decision, it feeds no mapping, and nothing downstream depends on its answer.
It stopped `build_column_evidence`, which stopped `run_llm_assisted_mapping`,
which is the stage that writes `28a_target_coverage_matrix`,
`28c_human_decision_queue` and `34_target_first_decisions` — three of the four
artefacts the run must produce. The run then finished, reported FAILED with
three required artefacts missing, and told the operator "The first step did not
finish cleanly."

A descriptive statistic must never be able to do that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.onboarding_agent.column_evidence import _extreme


def _raises_natively(values: pd.Series) -> bool:
    try:
        values.min()
        return False
    except TypeError:
        return True


class TestTheColumnThatStoppedTheRun:

    def test_an_amount_column_with_a_word_in_it(self):
        """`Current Balance` where an unknown amount is written "N/A"."""
        col = pd.Series([120000.0, "N/A", 98000.0, 5.0])
        assert _raises_natively(col), "the fixture must reproduce the fault"
        assert _extreme(col, largest=False) == "5.0"
        assert _extreme(col, largest=True) == "120000.0"

    def test_a_date_column_with_a_word_in_it(self):
        col = pd.Series([pd.Timestamp("2026-08-31"), "n/k",
                         pd.Timestamp("2026-01-01")])
        assert _raises_natively(col)
        assert _extreme(col, largest=False).startswith("2026-01-01")
        assert _extreme(col, largest=True).startswith("2026-08-31")

    def test_the_numbers_decide_when_the_column_carries_numbers(self):
        """A column of amounts with some words in it is an amount column."""
        col = pd.Series(["N/A", 3.0, "unknown", 1.0, 2.0])
        assert _extreme(col, largest=False) == "1.0"
        assert _extreme(col, largest=True) == "3.0"


class TestTheOrdinaryColumnIsUnchanged:
    """Whatever this does for a mixed column must not disturb a clean one."""

    @pytest.mark.parametrize("values,low,high", [
        ([3.0, 1.0, 2.0], "1.0", "3.0"),
        ([3, 1, 2], "1", "3"),
        (["beta", "alpha", "gamma"], "alpha", "gamma"),
    ])
    def test_a_single_typed_column_keeps_its_own_ordering(self, values, low, high):
        col = pd.Series(values)
        assert _extreme(col, largest=False) == low
        assert _extreme(col, largest=True) == high

    def test_a_date_column_orders_as_dates(self):
        col = pd.to_datetime(pd.Series(["2026-08-31", "2026-01-01", "2026-05-05"]))
        assert _extreme(col, largest=False).startswith("2026-01-01")
        assert _extreme(col, largest=True).startswith("2026-08-31")

    def test_an_empty_column_reports_nothing_rather_than_raising(self):
        assert _extreme(pd.Series([], dtype=object), largest=False) == ""
        assert _extreme(pd.Series([], dtype=float), largest=True) == ""


class TestProfilingNeverStopsTheRun:
    """The point is not these cases. It is that no column can do this again."""

    @pytest.mark.parametrize("values", [
        [1.0, "text"],
        ["text", 1.0],
        [pd.Timestamp("2026-01-01"), 1.0],
        [None, "a", 2.0],
        [np.nan, "N/A", 7.0],
        [{"a": 1}, 2.0],
        [[1, 2], "x"],
        [True, "yes", 1.0],
        ["", 0.0],
    ])
    def test_no_mixture_raises(self, values):
        col = pd.Series(values, dtype=object).dropna()
        for largest in (False, True):
            assert isinstance(_extreme(col, largest=largest), str)


class TestTheWholeEvidenceRowSurvivesTheLiveShape:
    """End to end on a frame shaped like ERE's loan extract."""

    def test_build_column_evidence_completes(self, tmp_path):
        from engine.onboarding_agent import column_evidence as ce
        df = pd.DataFrame({
            "Loan Policy Number": [f"ERE{i:04d}" for i in range(6)],
            "Current Balance": [120000.0, "N/A", 98000.0, 5.0, "N/A", 42.0],
            "Valuation Date": [pd.Timestamp("2026-08-31"), "n/k",
                               pd.Timestamp("2026-01-01"), "n/k",
                               pd.Timestamp("2026-05-05"), "n/k"],
            "Month Run": ["August"] * 6,
        })
        rows = ce.build_column_evidence(
            df, source_file="LoanExtract One.xlsx", sheet_name="LoanExtractOne")
        assert len(rows) == len(df.columns)
        by_col = {r["source_column"]: r for r in rows}
        assert by_col["Current Balance"]["min_value"] == "5.0"
        assert by_col["Current Balance"]["max_value"] == "120000.0"
        # the placeholder words are not hidden — they stay in the profile
        assert "N/A" in by_col["Current Balance"]["sample_values_distinct_redacted"]
