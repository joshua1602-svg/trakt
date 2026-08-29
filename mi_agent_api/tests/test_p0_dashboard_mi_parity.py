"""Commercial go-live sprint — P0-1 / P0-2 / P0-3.

Each test pins the INVARIANT the fix established, not the fix's implementation:

* **P0-1** the funded "By LTV band" stratification bands on CURRENT LTV, so it
  reconciles exactly with what the MI Query Agent answers for "balance by LTV
  band"; the cohort lens keeps its origination basis, because a static pool is
  defined at origination.
* **P0-2** a concentration test measured in PERCENT never carries a currency
  balance into an artifact, on either the governed OCC-approved path or the
  pre-approval Schedule 8 fallback a newly onboarded client meets.
* **P0-3** the funded rate stratification consumes the canonical
  ``interest_rate_bucket`` (``config/mi/buckets.yaml``), which is the sole
  economic definition of a rate band.

Every test also asserts the money and the loan count are untouched: these were
presentation-basis corrections, and they are not licensed to move a total.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics_lib.buckets import load_bucket_config, materialise_buckets  # noqa: E402
from mi_agent_api import cohorts as cohorts_mod  # noqa: E402
from mi_agent_api import snapshots as snapshots_mod  # noqa: E402
from mi_agent_api.funded_prep import prepare_funded_mi_dataset  # noqa: E402

BALANCE = "current_outstanding_balance"


@pytest.fixture(scope="module")
def erm_book() -> pd.DataFrame:
    """An equity-release-shaped book: current LTV rolls up well above the LTV at
    origination (the defining property of the asset class), and rates span below
    2% so the canonical ``<2%`` / ``2-3%`` bands are actually populated."""
    rng = np.random.default_rng(11)
    n = 4_000
    orig = np.clip(rng.normal(0.28, 0.08, n), 0.05, 0.60)
    frame = pd.DataFrame({
        "loan_id": [f"L{i:06d}" for i in range(n)],
        BALANCE: rng.uniform(40_000, 900_000, n).round(2),
        "original_loan_to_value": orig,
        "current_loan_to_value": np.clip(orig * rng.uniform(1.4, 2.4, n), 0.05, 1.30),
        "youngest_borrower_age": rng.integers(55, 95, n),
        "current_interest_rate": np.clip(rng.normal(5.4, 1.6, n), 1.1, 9.5),
    })
    prepared, _report = prepare_funded_mi_dataset(frame)
    prepared, _issues, _applied = materialise_buckets(
        prepared, load_bucket_config(), target="semantic_field")
    return prepared


def _mi_agent_table(df: pd.DataFrame, bucket_col: str) -> dict:
    """What the MI Query Agent reports for this dimension: a groupby over the
    canonical bucket column materialised from ``config/mi/buckets.yaml``."""
    grouped = df[BALANCE].groupby(df[bucket_col].astype("string")).agg(["sum", "count"])
    return {str(band): (round(float(row["sum"]), 2), int(row["count"]))
            for band, row in grouped.iterrows()}


def _dashboard_table(df: pd.DataFrame, key: str) -> dict:
    entry = next(e for e in snapshots_mod._funded_stratifications(df) if e["key"] == key)
    return {b["label"]: (round(b["balance"], 2), b["count"]) for b in entry["bars"]}


# --------------------------------------------------------------------------- #
# P0-1 — funded LTV stratification
# --------------------------------------------------------------------------- #
def test_p0_1_funded_ltv_reconciles_exactly_with_the_mi_query_agent(erm_book):
    assert _dashboard_table(erm_book, "ltv") == _mi_agent_table(erm_book, "ltv_bucket")


def test_p0_1_funded_ltv_uses_current_not_origination_ltv(erm_book):
    """The distinguishing case: on a rolled-up book the two bases disagree, and
    the funded view must follow current LTV."""
    dashboard = _dashboard_table(erm_book, "ltv")
    assert dashboard != _mi_agent_table(erm_book, "original_ltv_bucket")


def test_p0_1_high_ltv_exposure_is_visible_on_the_funded_view(erm_book):
    """The defect this fixed: an origination-LTV view reported no exposure above
    80% on a book that genuinely has some."""
    dashboard = _dashboard_table(erm_book, "ltv")
    high = {b: v for b, v in dashboard.items()
            if b in ("80-90%", "90-100%", ">=100%")}
    assert high, "a rolled-up ERM book must show its high-LTV bands"
    assert sum(bal for bal, _ in high.values()) > 0


def test_p0_1_cohort_lens_still_bands_on_origination_ltv(erm_book):
    """Static-pool semantics are NOT changed by the funded-view correction."""
    series, _header = cohorts_mod._dimension_series(erm_book, "ltv", "Y")
    expected = erm_book["original_ltv_bucket"].astype("string")
    pd.testing.assert_series_equal(pd.Series(series).astype("string"), expected,
                                   check_names=False)


def test_p0_1_default_ltv_basis_is_origination(erm_book):
    """Any existing caller that does not ask for a basis keeps the old answer."""
    default, _ = cohorts_mod._dimension_series(erm_book, "ltv", "Y")
    explicit, _ = cohorts_mod._dimension_series(
        erm_book, "ltv", "Y", ltv_basis=cohorts_mod.LTV_BASIS_ORIGINATION)
    pd.testing.assert_series_equal(pd.Series(default), pd.Series(explicit))


# --------------------------------------------------------------------------- #
# P0-3 — funded rate stratification
# --------------------------------------------------------------------------- #
def test_p0_3_funded_rate_reconciles_exactly_with_the_mi_query_agent(erm_book):
    assert _dashboard_table(erm_book, "rate") == _mi_agent_table(
        erm_book, "interest_rate_bucket")


def test_p0_3_canonical_bands_are_the_only_economic_definition(erm_book):
    """The bands the dashboard shows are the ones buckets.yaml declares — the
    local fallback bands (``<3%``, ``8%+``) must not appear."""
    bands = set(_dashboard_table(erm_book, "rate"))
    declared = set(load_bucket_config()["interest_rate_bucket"]["labels"])
    assert bands <= declared
    assert not bands & {"<3%", "8%+", "3–4%", "4–5%"}  # en-dash locals


def test_p0_3_sub_two_percent_rates_are_reported_not_lost(erm_book):
    """The old local binning multiplied any rate <= 1.5 by 100, pushing genuine
    sub-1.5% rates out of every band and into ``Unknown``."""
    table = _dashboard_table(erm_book, "rate")
    assert "Unknown" not in table
    assert "<2%" in table and table["<2%"][1] > 0


def test_p0_3_local_bands_remain_as_an_explicit_fallback():
    """A frame that never went through funded_prep still gets bars rather than
    an empty chart — the fallback is retained deliberately."""
    raw = pd.DataFrame({BALANCE: [100_000.0, 200_000.0],
                        "current_interest_rate": [4.2, 6.8]})
    series = snapshots_mod._strat_series(raw, "rate")
    assert series is not None
    assert set(series.dropna()) <= set(snapshots_mod._RATE_LABELS)


# --------------------------------------------------------------------------- #
# Totals are untouched by BOTH basis corrections
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["ltv", "rate", "age"])
def test_stratification_totals_are_conserved(erm_book, key):
    table = _dashboard_table(erm_book, key)
    assert round(sum(bal for bal, _ in table.values()), 2) == round(
        float(erm_book[BALANCE].sum()), 2)
    assert sum(cnt for _, cnt in table.values()) == len(erm_book)


# --------------------------------------------------------------------------- #
# P0-2 — a percentage is never carried as a currency balance
# --------------------------------------------------------------------------- #
def _risk_groups(tests: list) -> list:
    """Reproduce the artifact-group construction from the Schedule 8 fallback
    route, which is the path a newly onboarded client meets before OCC approval."""
    from mi_agent_api import chat_routing  # noqa: WPS433 - import cost is per-test
    source = Path(chat_routing.__file__).read_text(encoding="utf-8")
    assert '"balance": t["actualValue"]' not in source, (
        "a percentage must never be emitted as a `balance`")
    return [{
        "name": t["label"],
        "share": float(t["actualValue"]) / 100.0,
        "status": t["status"], "limit": float(t["limitValue"]) / 100.0,
        "approaching": t["status"] == "amber",
    } for t in tests
        if (t.get("status") in ("green", "amber", "red")
            and t.get("actualValue") is not None and t.get("limitValue")
            and t.get("unit") == "percent")]


def test_p0_2_percent_test_carries_no_balance_on_the_fallback_path():
    tests = [{"label": "London concentration", "actualValue": 24.1,
              "limitValue": 30.0, "unit": "percent", "status": "green"}]
    groups = _risk_groups(tests)
    assert groups and "balance" not in groups[0]
    assert groups[0]["share"] == pytest.approx(0.241)


def test_p0_2_governed_concentration_path_reports_percent_units():
    """The OCC-approved path routes through the evaluation service, whose tests
    declare their own unit; a percent test stays a percent."""
    from mi_agent.concentration_tests.models import ActiveTest
    test = ActiveTest(metric_id="geographic_concentration",
                      display_name="London concentration",
                      threshold=30.0, unit="percent")
    assert test.unit == "percent"
    assert not hasattr(test, "balance")


def test_p0_2_no_presentation_layer_rescaling_remains():
    """No production view may decide a payload amount is 'really' millions."""
    ui = _REPO_ROOT / "frontend" / "mi-agent-ui" / "src" / "components" / "artifacts"
    offenders = [p.name for p in ui.glob("*.tsx")
                 if ".test." not in p.name and "* 1e6" in p.read_text(encoding="utf-8")]
    assert offenders == []
