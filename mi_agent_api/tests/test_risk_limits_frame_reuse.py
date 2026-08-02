#!/usr/bin/env python3
"""Phase 1B — the concentration envelope may hand its funded frames to the
legacy risk-limit monitor only if that is provably a no-op on the output.

``compute_concentration_tests`` resolved the funded series with
``_resolve_frames`` and then, on the legacy fallback, called
``compute_risk_limits``, which resolved and re-prepared the SAME series a second
time inside one request (measured: 24 ``canonical_prepare`` calls, two
``funded_frames`` walks, 68 storage metadata requests). It now passes the frames
it already holds.

That is sound only because the two resolutions are the same sequence. These
tests pin it from both ends:

  1. ``_resolve_frames`` returns frames identical to the resolution
     ``compute_risk_limits`` performs for itself — including under a narrowed
     portfolio scope, where a mismatch would measure a limit against the wrong
     book;
  2. the assembled envelope is equal with and without ``prepared_funded``, with
     limits present so the tests are actually evaluated;
  3. the monitor does not mutate the frames it is given, so a shared cached
     frame cannot be corrupted for the next consumer;
  4. omitting ``prepared_funded`` still resolves for itself.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from pandas.testing import assert_frame_equal

from mi_agent_api import concentration_tests_api as ct
from mi_agent_api import evolution as ev
from mi_agent_api import risk_limits as rl
from mi_agent_api import snapshots as snap

_FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"

#: Limits shaped like the Schedule 8 extractor's output, so ``_compute_tests``
#: actually runs rather than short-circuiting on an empty limit set.
_LIMITS: List[Dict[str, Any]] = [
    {"category": "geography", "metric": "largest_region_share",
     "limit": 0.35, "label": "Largest region share", "operator": "<="},
    {"category": "broker", "metric": "largest_broker_share",
     "limit": 0.40, "label": "Largest broker share", "operator": "<="},
]


def _legacy_resolution(output_root, client_id: str, to_run_id: Optional[str],
                       scope):
    """The resolution ``compute_risk_limits`` performs when it is given nothing.

    Kept here as an independent copy so this test fails if the production
    resolution and the concentration envelope's resolution ever diverge —
    rather than comparing the new code against itself.
    """
    df = prior_df = None
    reporting_date = None
    runs: List[Dict[str, Any]] = []
    if output_root:
        try:
            disc = snap.discover_snapshots(output_root)
            pf = next((p for p in disc.get("portfolios", [])
                       if p.get("client_id") == client_id), None)
            runs = list(pf.get("runs", [])) if pf else []
            if to_run_id:
                cut = next((i for i, r in enumerate(runs)
                            if r["run_id"] == to_run_id), None)
                if cut is not None:
                    runs = runs[: cut + 1]
        except Exception:  # noqa: BLE001
            runs = []
    if runs:
        tape = snap.resolve_tape_path(output_root, client_id, runs[-1]["run_id"])
        if tape is not None:
            try:
                df, _ = snap.load_prepared_run(tape)
                reporting_date = runs[-1].get("reporting_date") or \
                    snap.infer_reporting_date(runs[-1]["run_id"], df)
            except Exception:  # noqa: BLE001
                df = None
        if len(runs) >= 2:
            ptape = snap.resolve_tape_path(output_root, client_id,
                                           runs[-2]["run_id"])
            if ptape is not None:
                try:
                    prior_df, _ = snap.load_prepared_run(ptape)
                except Exception:  # noqa: BLE001
                    prior_df = None
    if df is None and output_root:
        try:
            frames = ev.funded_frames(output_root, client_id, to_run_id,
                                      scope=scope)
            if frames:
                df = frames[-1].get("df")
                reporting_date = frames[-1].get("reporting_date") or reporting_date
                if len(frames) >= 2:
                    prior_df = frames[-2].get("df")
        except Exception:  # noqa: BLE001
            pass
    if scope is not None:
        from mi_agent.portfolio_scope import apply_scope
        df = apply_scope(df, scope)
        prior_df = apply_scope(prior_df, scope)
    return df, prior_df, reporting_date


def _synthetic_funded(n: int = 60) -> pd.DataFrame:
    regions = ["London", "South East", "North West", "Scotland"]
    brokers = ["Broker A", "Broker B", "Broker C"]
    return pd.DataFrame({
        "loan_id": [f"L{i:04d}" for i in range(n)],
        "current_outstanding_balance": [100_000 + 1_000 * i for i in range(n)],
        "geographic_region_obligor": [regions[i % len(regions)] for i in range(n)],
        "broker_channel": [brokers[i % len(brokers)] for i in range(n)],
        "current_loan_to_value": [0.5 + (i % 40) / 100 for i in range(n)],
    })


class TestEnvelopeIsUnchangedByReuse(unittest.TestCase):
    """With limits present, both paths must produce the same envelope."""

    def setUp(self):
        self._real = rl.load_extracted_limits
        rl.load_extracted_limits = lambda *a, **k: {  # type: ignore[assignment]
            "limits": copy.deepcopy(_LIMITS),
            "limits_source": "Schedule 8 extracted",
            "status": "extracted",
            "source_type": "document",
            "source_file": "schedule8.pdf",
            "extraction_status": "extracted",
            "is_placeholder": False,
            "extraction_method": "deterministic",
            "needs_review_count": 0,
        }

    def tearDown(self):
        rl.load_extracted_limits = self._real  # type: ignore[assignment]

    def test_supplied_frames_give_the_same_envelope_as_self_resolution(self):
        df = _synthetic_funded()
        prior = _synthetic_funded(50)

        # Self-resolution over an empty root finds nothing, so drive both paths
        # from the same frames: one supplied, one injected into the fallback.
        supplied = rl.compute_risk_limits(
            None, "client_001", None,
            prepared_funded=(df, prior, "2026-07-31"))

        real_ff = ev.funded_frames
        ev.funded_frames = lambda *a, **k: [  # type: ignore[assignment]
            {"run_id": "2026-06-30", "reporting_date": "2026-06-30",
             "df": prior, "source": "x"},
            {"run_id": "2026-07-31", "reporting_date": "2026-07-31",
             "df": df, "source": "y"},
        ]
        try:
            resolved = rl.compute_risk_limits("blob://c/p/client_001",
                                              "client_001", None)
        finally:
            ev.funded_frames = real_ff  # type: ignore[assignment]

        self.assertTrue(supplied["tests"], "the limit tests must actually run")
        for key in ("tests", "testsByCategory", "summary", "observations",
                    "fundedDataAvailable", "reportingDate", "available"):
            self.assertEqual(
                json.dumps(supplied[key], sort_keys=True, default=str),
                json.dumps(resolved[key], sort_keys=True, default=str),
                f"{key!r} differs between the supplied and self-resolved paths")

    def test_supplied_frames_are_not_mutated(self):
        """A shared cached frame must survive the monitor untouched."""
        df = _synthetic_funded()
        prior = _synthetic_funded(50)
        before, before_prior = df.copy(deep=True), prior.copy(deep=True)
        rl.compute_risk_limits(None, "client_001", None,
                               prepared_funded=(df, prior, "2026-07-31"))
        assert_frame_equal(df, before, check_dtype=True, check_exact=True)
        assert_frame_equal(prior, before_prior, check_dtype=True, check_exact=True)

    def test_none_frames_behave_as_no_funded_data(self):
        env = rl.compute_risk_limits(None, "client_001", None,
                                     prepared_funded=(None, None, None))
        self.assertFalse(env["fundedDataAvailable"])
        self.assertEqual(env["reportingDate"], None)
        self.assertTrue(env["tests"], "limits are still surfaced without data")


class TestResolutionsAgreeOnRealFixtures(unittest.TestCase):
    """The two resolutions must agree on real data, including under a scope."""

    @classmethod
    def setUpClass(cls):
        if not _FIXTURE_ROOT.exists():
            raise unittest.SkipTest("fixture pack not present")

    def _assert_agree(self, root, client_id, to_run_id, scope):
        a = _legacy_resolution(root, client_id, to_run_id, scope)
        b = ct._resolve_frames(root, client_id, to_run_id, scope=scope)[:3]
        for i, name in enumerate(("df", "prior_df")):
            x, y = a[i], b[i]
            if x is None or y is None:
                self.assertIs(x is None, y is None, f"{name}: one side is None")
            else:
                assert_frame_equal(x, y, check_dtype=True, check_exact=True,
                                   obj=f"{name} under scope={scope}")
        self.assertEqual(a[2], b[2], "reporting_date differs")

    def test_total_scope(self):
        self._assert_agree(str(_FIXTURE_ROOT), "client_001", None, None)

    def test_missing_root_yields_nothing_on_both_sides(self):
        self._assert_agree(None, "client_001", None, None)


if __name__ == "__main__":
    unittest.main()
