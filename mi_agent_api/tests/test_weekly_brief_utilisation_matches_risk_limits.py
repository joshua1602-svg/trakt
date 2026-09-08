"""The Weekly Brief's utilisation figures must agree with the Risk Limits tab.

Reported defect: the brief's top two items read "Regional exposure — Scotland
is at 4340% of its limit" and "Average principal balance is at 4039% of its
limit", both flagged CONCERN — impossible values that did not reconcile with
the Risk Limits table, which showed Scotland at 4.34%, comfortably PASS.

Root cause, in insight_generators.concentration(): `utilization` is already
value/threshold*100 (concentration_tests.evaluation._utilization) — the SAME
points-scale figure the Risk Limits table renders. The brief treated it as a
0-1 fraction and multiplied by 100 a second time (43.4 -> 4340), and compared
it against amber/red thresholds that were ALSO divided by 100 down to a
fraction (90.0 -> 0.9) — so a test at 43% utilisation, nowhere near its limit,
tripped both the "worth reporting" gate and the "breached" check, which fire
at >= 0.9 and >= 1.0 points respectively: almost any nonzero utilisation.

These tests reproduce the two reported cases exactly, against the real
`concentration()` generator — not a re-derived fixture — and assert what the
Risk Limits tab already agrees is true: both are comfortably within limit and
must not surface as a Weekly Brief concern.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import insight_generators as gen  # noqa: E402

CTX = {"tenant_id": "ERE", "portfolio_id": "ERE", "portfolio_context": "total",
       "prior_reporting_date": None}


def _snapshot(*, test_id, name, unit, current_value, threshold, headroom):
    """A snapshot shaped exactly as concentration_tests_api.compute_
    concentration_tests emits it — utilization is value/threshold*100, the
    same figure the Risk Limits table's own `hd`/percentage columns read."""
    utilization = round(current_value / threshold * 100.0, 2)
    return {
        "available": True, "reportingDate": "2026-06-30",
        "forecast": {}, "states": {"available": False},
        "tests": [{
            "testId": test_id, "displayName": name,
            "currentValue": current_value, "threshold": threshold,
            "utilization": utilization, "unit": unit, "headroom": headroom,
            "status": "pass", "expected": {}, "fullPipeline": {},
            "expectedBreach": False, "expectedBreachHorizon": {},
        }],
    }


def test_scotland_at_4_34_percent_of_a_10_percent_limit_is_not_a_concern():
    """THE EXACT REPORTED CASE. 4.34% exposure against a 10.00% limit is 43.4%
    utilised — comfortable — not the impossible 4340% the brief showed."""
    snap = _snapshot(test_id="geo_scotland", name="Regional exposure — Scotland",
                     unit="percent", current_value=4.34, threshold=10.0,
                     headroom=5.66)
    assert snap["tests"][0]["utilization"] == 43.4
    ins, omit = gen.concentration(CTX, snap)
    assert ins == [], f"Scotland surfaced as a concern: {ins[0].headline if ins else None}"
    assert omit[0].category == "immaterial"


def test_average_principal_balance_at_121169_of_a_300000_limit_is_not_a_concern():
    """THE OTHER EXACT REPORTED CASE — a currency-unit test, not a percent
    one, so its own headroom (£178,831) must never be run through pct()."""
    snap = _snapshot(test_id="bal_avg", name="Average principal balance",
                     unit="currency", current_value=121_169.0, threshold=300_000.0,
                     headroom=178_831.0)
    assert snap["tests"][0]["utilization"] == 40.39
    ins, omit = gen.concentration(CTX, snap)
    assert ins == [], f"Average principal balance surfaced: {ins[0].headline if ins else None}"
    assert omit[0].category == "immaterial"


def test_a_test_genuinely_at_92_percent_of_limit_still_surfaces():
    """The gate must still catch a REAL near-limit test — the fix must not
    have swung the other way into silence."""
    snap = _snapshot(test_id="geo_london", name="Regional exposure — London",
                     unit="percent", current_value=46.0, threshold=50.0,
                     headroom=4.0)
    assert snap["tests"][0]["utilization"] == 92.0
    ins, _omit = gen.concentration(CTX, snap)
    assert len(ins) == 1
    assert "92%" in ins[0].headline, ins[0].headline
    assert "4340" not in ins[0].headline and "9200" not in ins[0].headline


def test_a_genuine_breach_reads_over_100_percent_not_over_10000():
    snap = _snapshot(test_id="geo_se", name="Regional exposure — South East",
                     unit="percent", current_value=55.0, threshold=50.0,
                     headroom=-5.0)
    assert snap["tests"][0]["utilization"] == 110.0
    ins, _omit = gen.concentration(CTX, snap)
    assert len(ins) == 1
    assert "110%" in ins[0].headline, ins[0].headline
