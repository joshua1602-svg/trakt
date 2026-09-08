"""mi_agent.borrowing_base.analysis — the MI analysis helper, proven by hand.

Same rule as the rest of this suite: no expected number is produced by calling
the code under test. Every reason row, every period verdict and every bridge
driver below is worked out from a frame or an envelope small enough to check by
eye.

Three things are proven here that the route tests then merely consume:

1. THE REASON TABLE RECONCILES, and it is a PRIMARY-reason table. A loan that
   fails two approved rules carries the first configured one and is counted
   once. UNDETERMINED never enters it. A blank reason on an INELIGIBLE loan
   fails closed. And an independent oracle written against the two governed
   columns alone agrees with it row for row.

2. PERIOD VALIDITY IS GOVERNED, NOT INFERRED. Applicability comes from the
   register's ``governance.approved_at`` and from nothing else — an
   ``effective_date`` alone demonstrates nothing. A drawing is valid for a
   snapshot only when its ``current_drawn_amount_as_of`` IS that snapshot's
   date; being the latest frame is not evidence.

3. THE BRIDGE IS EXACT. Opening + three drivers = closing, on a hand-worked
   pair of envelopes, including a rate change and a binding cap.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.borrowing_base import analysis  # noqa: E402
from mi_agent.borrowing_base.eligibility import derive_eligibility  # noqa: E402
from mi_agent.borrowing_base.models import (  # noqa: E402
    FIELD_ELIGIBILITY_REASON,
    FIELD_ELIGIBILITY_STATUS,
    INELIGIBLE,
    NOT_CALCULABLE,
    EligibilityRule,
    FacilityConfiguration,
)

BALANCE = "current_outstanding_balance"


# --------------------------------------------------------------------------- #
# Fixtures — a six-loan book, two approved rules, one loan failing both
# --------------------------------------------------------------------------- #
def two_rule_facility(**over) -> FacilityConfiguration:
    spec = dict(
        client_id="c", facility_id="F1", environment="production",
        commitment=95_000_000.0, advance_rate=0.9,
        eligibility_rule_version="r-1",
        eligibility_rules=[
            EligibilityRule(rule_id="max_current_ltv", field="current_loan_to_value",
                            operator="max", value=50, reason_code="ltv_above_limit",
                            reason="Current LTV above the facility's 50% ceiling"),
            EligibilityRule(rule_id="min_youngest_age", field="youngest_borrower_age",
                            operator="min", value=60, reason_code="borrower_too_young",
                            reason="Youngest borrower under the 60 minimum"),
        ])
    spec.update(over)
    return FacilityConfiguration(**spec)


def six_loan_frame() -> pd.DataFrame:
    """A ELIGIBLE · B fails LTV · C fails age · D fails BOTH · E, F UNDETERMINED."""
    return pd.DataFrame({
        "loan_id": list("ABCDEF"),
        BALANCE: [1_000_000.0, 2_000_000.0, 3_000_000.0, 4_000_000.0,
                  5_000_000.0, 6_000_000.0],
        "current_loan_to_value": [40, 60, 40, 60, None, 40],
        "youngest_borrower_age": [70, 70, 50, 50, 70, None],
    })


def derived() -> Tuple[pd.DataFrame, FacilityConfiguration]:
    fac = two_rule_facility()
    df = six_loan_frame()
    derive_eligibility(df, fac)
    return df, fac


def oracle(df: pd.DataFrame) -> Dict[str, Tuple[int, float]]:
    """An INDEPENDENT reason oracle: the two governed columns and the balance.

    Nothing from the analysis module, nothing from the calculator: a mask on the
    status column, a groupby on the reason column, a sum of the balance column.
    """
    ineligible = df[df[FIELD_ELIGIBILITY_STATUS] == INELIGIBLE]
    grouped = ineligible.groupby(FIELD_ELIGIBILITY_REASON)[BALANCE].agg(["count", "sum"])
    return {str(k): (int(v["count"]), round(float(v["sum"]), 2))
            for k, v in grouped.iterrows()}


# --------------------------------------------------------------------------- #
# 1. Reasons
# --------------------------------------------------------------------------- #
class TestTheReasonTable:
    def test_the_derivation_stamps_one_primary_reason_on_a_loan_failing_two_rules(self):
        df, _ = derived()
        assert df.loc[3, FIELD_ELIGIBILITY_STATUS] == INELIGIBLE
        # D fails LTV (rule 1) AND age (rule 2): the FIRST configured rule wins.
        assert df.loc[3, FIELD_ELIGIBILITY_REASON] == "ltv_above_limit"

    def test_rows_reconcile_and_carry_the_approved_wording(self):
        df, fac = derived()
        out = analysis.summarise_ineligibility_reasons(df, fac)
        assert out.available and out.reconciles
        assert out.ineligible_loan_count == 3
        assert out.ineligible_balance == 9_000_000.0
        assert out.financing_portfolio_loan_count == 6
        assert out.financing_portfolio_balance == 21_000_000.0
        assert out.ineligible_loan_share_pct == 50.0          # 3 of 6
        assert out.ineligible_balance_share_pct == 42.8571    # 9m of 21m
        assert [(r.reason_code, r.loan_count, r.balance) for r in out.rows] == [
            ("ltv_above_limit", 2, 6_000_000.0),      # B and D
            ("borrower_too_young", 1, 3_000_000.0),   # C
        ]
        assert out.rows[0].reason_description == \
            "Current LTV above the facility's 50% ceiling"
        assert out.rows[0].rule_id == "max_current_ltv"
        assert out.rows[0].share_of_ineligible_loans_pct == 66.6667
        assert out.rows[0].share_of_ineligible_balance_pct == 66.6667
        assert out.rows[1].share_of_ineligible_loans_pct == 33.3333
        assert sum(r.loan_count for r in out.rows) == out.ineligible_loan_count
        assert sum(r.balance for r in out.rows) == out.ineligible_balance
        assert "primary_reason" in out.basis

    def test_the_independent_oracle_agrees_row_for_row(self):
        df, fac = derived()
        out = analysis.summarise_ineligibility_reasons(df, fac)
        truth = oracle(df)
        assert {r.reason_code: (r.loan_count, r.balance) for r in out.rows} == truth
        assert sum(c for c, _ in truth.values()) == out.ineligible_loan_count
        assert abs(sum(b for _, b in truth.values()) - out.ineligible_balance) <= 0.02

    def test_undetermined_loans_are_never_mixed_into_ineligible(self):
        df, fac = derived()
        out = analysis.summarise_ineligibility_reasons(df, fac)
        codes = {r.reason_code for r in out.rows}
        assert not any(c.startswith("eligibility_rule_input_missing") for c in codes)
        # E (£5m) and F (£6m) are undetermined: the ineligible balance is £9m,
        # not £20m.
        assert out.ineligible_balance == 9_000_000.0

    def test_a_blank_reason_on_an_ineligible_loan_fails_closed(self):
        df, fac = derived()
        df.loc[1, FIELD_ELIGIBILITY_REASON] = None
        out = analysis.summarise_ineligibility_reasons(df, fac)
        assert out.available is False
        failed = {i["invariant"] for i in out.invariants if not i["holds"]}
        assert "every_ineligible_loan_carries_one_reason" in failed
        assert "reason_missing" in {r.reason_code for r in out.rows}

    def test_a_code_no_approved_rule_declares_is_shown_verbatim_and_noted(self):
        df, fac = derived()
        df.loc[2, FIELD_ELIGIBILITY_REASON] = "retired_rule_code"
        out = analysis.summarise_ineligibility_reasons(df, fac)
        assert out.available  # the population still reconciles
        row = next(r for r in out.rows if r.reason_code == "retired_rule_code")
        assert row.reason_description == "retired_rule_code" and row.rule_id is None
        assert any("retired_rule_code" in n for n in out.notes)

    def test_no_derivation_or_no_frame_is_unavailable_not_zero(self):
        fac = two_rule_facility()
        assert analysis.summarise_ineligibility_reasons(None, fac).available is False
        assert analysis.summarise_ineligibility_reasons(pd.DataFrame(), fac).available is False
        raw = six_loan_frame()   # no governed columns
        assert analysis.summarise_ineligibility_reasons(raw, fac).available is False


# --------------------------------------------------------------------------- #
# 2. Period validity
# --------------------------------------------------------------------------- #
class TestConfigurationApplicability:
    def test_only_a_governed_approval_date_demonstrates_history(self):
        approved = two_rule_facility(governance={"approved_at": "2026-04-30"})
        assert analysis.configuration_applicable_from(approved) == "2026-04-30"
        assert analysis.configuration_applies_at(approved, "2026-05-31", is_current=False)
        assert not analysis.configuration_applies_at(approved, "2026-03-31", is_current=False)

    def test_effective_date_alone_demonstrates_nothing(self):
        metadata_only = two_rule_facility(effective_date="2025-11-30")
        assert analysis.configuration_applicable_from(metadata_only) is None
        assert not analysis.configuration_applies_at(metadata_only, "2026-05-31",
                                                     is_current=False)

    def test_the_current_position_is_always_applicable(self):
        assert analysis.configuration_applies_at(two_rule_facility(), "2026-06-30",
                                                 is_current=True)
        assert analysis.configuration_applies_at(None, None, is_current=True)


class TestDrawnValidity:
    def test_the_drawing_is_valid_only_as_at_its_own_stated_date(self):
        fac = two_rule_facility(current_drawn_amount=6_000_000.0,
                                current_drawn_amount_as_of="2026-06-30")
        assert analysis.drawn_valid_at(fac, "2026-06-30") == analysis.DRAWN_BASIS_VALID
        assert analysis.drawn_valid_at(fac, "2026-05-31") == \
            analysis.DRAWN_BASIS_AS_OF_MISMATCH
        assert analysis.drawn_valid_at(two_rule_facility(), "2026-06-30") == \
            analysis.DRAWN_BASIS_NOT_SUPPLIED
        undated = two_rule_facility(current_drawn_amount=6_000_000.0)
        assert analysis.drawn_valid_at(undated, "2026-06-30") == \
            analysis.DRAWN_BASIS_AS_OF_MISMATCH

    def test_facility_for_period_withholds_the_drawing_unless_valid(self):
        fac = two_rule_facility(current_drawn_amount=6_000_000.0,
                                current_drawn_amount_as_of="2026-06-30")
        assert analysis.facility_for_period(fac, "2026-06-30").current_drawn_amount \
            == 6_000_000.0
        withheld = analysis.facility_for_period(fac, "2026-05-31")
        assert withheld.current_drawn_amount is None
        assert withheld.current_drawn_amount_as_of == ""
        assert withheld.advance_rate == fac.advance_rate  # terms untouched

    def test_a_position_refuses_a_drawn_dependent_number_it_cannot_stand_behind(self):
        fac = two_rule_facility(current_drawn_amount=6_000_000.0,
                                current_drawn_amount_as_of="2026-06-30",
                                governance={"approved_at": "2026-04-30"})
        envelope = {"available": True, "reconciles": True,
                    "measures": {"borrowing_base": 90.0,
                                 "borrowing_base_headroom": 84.0}}
        stale = analysis.position_for("r5", "2026-05-31", envelope,
                                      facility=fac, is_current=False)
        assert stale.measure("borrowing_base") == 90.0
        # The envelope carries 84.0, but no drawing is valid for May: refused.
        assert stale.measure("borrowing_base_headroom") == NOT_CALCULABLE
        assert "period_valid_drawn_amount" in stale.why_not_calculable(
            "borrowing_base_headroom")
        current = analysis.position_for("r6", "2026-06-30", envelope,
                                        facility=fac, is_current=True)
        assert current.measure("borrowing_base_headroom") == 84.0

    def test_a_period_before_the_approval_date_is_not_calculable_at_all(self):
        fac = two_rule_facility(governance={"approved_at": "2026-04-30"})
        envelope = {"available": True, "measures": {"borrowing_base": 90.0}}
        early = analysis.position_for("r3", "2026-03-31", envelope,
                                      facility=fac, is_current=False)
        assert early.measure("borrowing_base") == NOT_CALCULABLE
        assert "governed_configuration_applicability" in \
            early.why_not_calculable("borrowing_base")


class TestChange:
    def _positions(self):
        fac = two_rule_facility(governance={"approved_at": "2026-04-30"})
        o = analysis.position_for("r5", "2026-05-31", {
            "available": True, "measures": {"borrowing_base": 10_000_000.0,
                                            "ineligible_loan_count": 11,
                                            "ineligible_loan_share": 17.19}},
            facility=fac, is_current=False)
        c = analysis.position_for("r6", "2026-06-30", {
            "available": True, "measures": {"borrowing_base": 11_200_000.0,
                                            "ineligible_loan_count": 13,
                                            "ineligible_loan_share": 19.12}},
            facility=fac, is_current=True)
        return o, c

    def test_currency_count_and_percent_changes_in_their_own_units(self):
        o, c = self._positions()
        money = analysis.change(o, c, "borrowing_base")
        assert (money.change, money.change_pct) == (1_200_000.0, 12.0)
        count = analysis.change(o, c, "ineligible_loan_count", unit="count")
        assert count.change == 2 and isinstance(count.change, int)
        assert count.change_pct == 18.1818
        pct = analysis.change(o, c, "ineligible_loan_share", unit="percent")
        assert pct.change == 1.93 and pct.change_pct == NOT_CALCULABLE

    def test_a_missing_side_is_named_not_imputed(self):
        o, c = self._positions()
        out = analysis.change(o, c, "borrowing_base_headroom")
        assert not out.calculable and out.change == NOT_CALCULABLE
        assert out.reason.startswith("opening 2026-05")


# --------------------------------------------------------------------------- #
# 3. The bridge — hand-worked
# --------------------------------------------------------------------------- #
def opening_envelope(**over) -> dict:
    env = {"available": True, "reconciles": True,
           "eligibleCurrentBalance": 100_000_000.0, "advanceRate": 0.90,
           "grossBorrowingBase": 90_000_000.0, "availableBorrowingBase": 90_000_000.0,
           "currentDrawnAmount": 50_000_000.0, "borrowingBaseHeadroom": 40_000_000.0}
    env.update(over)
    return env


def closing_envelope(**over) -> dict:
    # Collateral up £20m, rate down 5 points, and the £95m cap now binds.
    env = {"available": True, "reconciles": True,
           "eligibleCurrentBalance": 120_000_000.0, "advanceRate": 0.85,
           "grossBorrowingBase": 102_000_000.0, "availableBorrowingBase": 95_000_000.0,
           "currentDrawnAmount": 60_000_000.0, "borrowingBaseHeadroom": 35_000_000.0}
    env.update(over)
    return env


class TestTheBridge:
    def test_opening_plus_three_drivers_equals_closing_exactly(self):
        out = analysis.bridge(opening_envelope(), closing_envelope(),
                              opening_label="2026-05", closing_label="2026-06",
                              opening_drawn_valid=True, closing_drawn_valid=True)
        assert out.available
        drivers = {d.key: d.value for d in out.drivers}
        # (120m − 100m) × 0.90 = 18m, priced at the OPENING rate.
        assert drivers["eligible_collateral_effect"] == 18_000_000.0
        # 120m × (0.85 − 0.90) = −6m, on the CLOSING collateral.
        assert drivers["advance_rate_effect"] == -6_000_000.0
        # (95m − 102m) − (90m − 90m) = −7m: the cap now binds.
        assert drivers["facility_cap_effect"] == -7_000_000.0
        assert [d.key for d in out.drivers] == list(analysis.DRIVER_ORDER)
        assert out.net_change == 5_000_000.0
        assert out.reconciliation["difference"] == 0.0 and out.reconciliation["holds"]
        assert 90_000_000.0 + 18_000_000.0 - 6_000_000.0 - 7_000_000.0 == 95_000_000.0

    def test_the_waterfall_rows_are_the_renderer_contract(self):
        out = analysis.bridge(opening_envelope(), closing_envelope(),
                              opening_label="2026-05", closing_label="2026-06")
        rows = out.waterfall_rows()
        assert [r["type"] for r in rows] == ["total", "delta", "delta", "delta", "total"]
        assert rows[0]["value"] == 90_000_000.0 and rows[-1]["value"] == 95_000_000.0
        assert rows[1]["label"] == "Eligible collateral effect"
        assert all(set(r) == {"label", "value", "type"} for r in rows)

    def test_headroom_bridge_holds_when_both_drawings_are_period_valid(self):
        out = analysis.bridge(opening_envelope(), closing_envelope(),
                              opening_label="2026-05", closing_label="2026-06",
                              opening_drawn_valid=True, closing_drawn_valid=True)
        h = out.headroom
        assert h["available"]
        # 40m + 5m − 10m = 35m
        assert (h["openingHeadroom"], h["changeInBorrowingBase"],
                h["changeInDrawn"], h["closingHeadroom"]) == (
            40_000_000.0, 5_000_000.0, 10_000_000.0, 35_000_000.0)
        assert h["reconciliation"]["difference"] == 0.0

    def test_headroom_bridge_is_not_calculable_without_period_valid_drawings(self):
        out = analysis.bridge(opening_envelope(), closing_envelope(),
                              opening_label="2026-05", closing_label="2026-06",
                              opening_drawn_valid=False, closing_drawn_valid=True)
        assert out.available                       # the base bridge still stands
        assert out.headroom["available"] is False
        assert out.headroom["status"] == NOT_CALCULABLE
        assert any("2026-05" in m and analysis.MISSING_PERIOD_DRAWN in m
                   for m in out.headroom["missingInputs"])

    def test_a_missing_or_not_calculable_input_refuses_by_name(self):
        out = analysis.bridge(opening_envelope(advanceRate=NOT_CALCULABLE),
                              closing_envelope(),
                              opening_label="2026-05", closing_label="2026-06")
        assert out.available is False and out.drivers == []
        assert any("advance rate (advanceRate)" in m for m in out.missing_inputs)
        unreconciled = analysis.bridge(opening_envelope(), closing_envelope(reconciles=False),
                                       opening_label="o", closing_label="c")
        assert unreconciled.available is False
        assert any("does not reconcile" in m for m in unreconciled.missing_inputs)

    def test_penny_rounded_envelopes_reconcile_within_the_disclosed_tolerance(self):
        o = opening_envelope(eligibleCurrentBalance=12_345_678.91, advanceRate=0.9,
                             grossBorrowingBase=11_111_111.02,
                             availableBorrowingBase=11_111_111.02)
        c = closing_envelope(eligibleCurrentBalance=13_579_246.13, advanceRate=0.9,
                             grossBorrowingBase=12_221_321.52,
                             availableBorrowingBase=12_221_321.52)
        out = analysis.bridge(o, c, opening_label="o", closing_label="c")
        assert out.available
        assert {d.key: d.value for d in out.drivers}["eligible_collateral_effect"] \
            == 1_110_210.50   # 1,233,567.22 × 0.9 = 1,110,210.498
        assert abs(out.reconciliation["difference"]) <= analysis.RECONCILIATION_TOLERANCE

    def test_a_bridge_that_does_not_reconcile_is_refused(self):
        # A closing gross base inconsistent with its own eligible × rate
        # (120m × 0.85 = 102m, not 101m) — a corrupted envelope. The three
        # drivers then sum to +6m against a net change of +5m.
        out = analysis.bridge(opening_envelope(),
                              closing_envelope(grossBorrowingBase=101_000_000.0),
                              opening_label="o", closing_label="c")
        assert out.available is False
        assert "does not reconcile" in out.reason
        assert out.reconciliation["holds"] is False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
