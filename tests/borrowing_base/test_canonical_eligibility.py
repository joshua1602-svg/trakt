"""Eligibility is produced by the CANONICAL pipeline, not by a dashboard.

The four governed columns must arrive on the prepared funded frame, because
that frame is what the Eligibility & Concentrations workspace, MI Query, the
Teams bot, the PPTX pack and the forecast all read. If eligibility were added
in the presentation layer, each of those would either miss it or invent its
own — which is the second source of truth this sprint exists to avoid.

The other half of the contract is additivity: a book with no configured
facility must come through preparation EXACTLY as it did before, and no
existing canonical economic field may change for any book.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent_api.funded_prep import (
    augment_platform_canonical_dimensions,
    prepare_funded_mi_dataset,
)

from mi_agent.borrowing_base.models import (
    ELIGIBLE,
    FIELD_ELIGIBILITY_REASON,
    FIELD_ELIGIBILITY_STATUS,
    FIELD_ELIGIBLE,
    FIELD_FACILITY_ID,
    UNDETERMINED,
)

GOVERNED_COLUMNS = (FIELD_ELIGIBLE, FIELD_ELIGIBILITY_STATUS,
                    FIELD_ELIGIBILITY_REASON, FIELD_FACILITY_ID)

#: The prototype client, which HAS a configured facility.
FACILITY_CLIENT = "ere_funding_uk"


def tape(client_id: str, rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(rows)],
        "client_id": [client_id] * rows,
        "current_outstanding_balance": [1_000_000.0 * (i + 1) for i in range(rows)],
        "original_principal_balance": [250_000.0] * rows,
        "original_valuation_amount": [500_000.0] * rows,
        "current_valuation_amount": [520_000.0] * rows,
        "current_interest_rate": [6.5] * rows,
        "origination_date": ["2021-06-30"] * rows,
        "collateral_geography": ["Scotland"] * rows,
    })


class TestTheCanonicalLayerProducesEligibility:
    def test_all_four_governed_columns_land_on_the_prepared_frame(self):
        out, _report = prepare_funded_mi_dataset(tape(FACILITY_CLIENT))
        for column in GOVERNED_COLUMNS:
            assert column in out.columns

    def test_the_determination_is_the_facilitys_own(self):
        out, _report = prepare_funded_mi_dataset(tape(FACILITY_CLIENT))
        assert set(out[FIELD_ELIGIBILITY_STATUS]) == {ELIGIBLE}
        assert set(out[FIELD_FACILITY_ID]) == {"ERE_WAREHOUSE_01"}

    def test_the_preparation_report_carries_the_derivation_provenance(self):
        _out, report = prepare_funded_mi_dataset(tape(FACILITY_CLIENT))
        provenance = report["borrowing_base_eligibility"]
        assert provenance["applied"] is True
        assert provenance["facility_id"] == "ERE_WAREHOUSE_01"
        assert provenance["derivation"].endswith("derive_eligibility")
        assert provenance["config_hash"]
        assert provenance["prototype_assumptions_used"]

    def test_the_derived_columns_are_declared_as_derived_not_sourced(self):
        _out, report = prepare_funded_mi_dataset(tape(FACILITY_CLIENT))
        assert set(GOVERNED_COLUMNS) <= set(report["derived_fields"])

    def test_the_platform_canonical_path_derives_them_too(self):
        # The read-time path used by the combined platform canonical, so a book
        # gets the determination without an onboarding re-run.
        out, derived = augment_platform_canonical_dimensions(tape(FACILITY_CLIENT))
        assert set(GOVERNED_COLUMNS) <= set(out.columns)
        assert set(GOVERNED_COLUMNS) <= set(derived)


class TestAdditivity:
    def test_a_book_with_no_facility_gets_no_eligibility_columns(self):
        out, report = prepare_funded_mi_dataset(tape("a_client_with_no_warehouse"))
        assert not [c for c in out.columns if c.startswith("borrowing_base_")]
        assert report["borrowing_base_eligibility"] == {
            "applied": False, "reason": "no_facility_configured"}

    def test_an_unidentifiable_client_is_a_no_op_not_a_guess(self):
        raw = tape(FACILITY_CLIENT).drop(columns=["client_id"])
        out, report = prepare_funded_mi_dataset(raw)
        assert report["borrowing_base_eligibility"]["applied"] is False
        assert not [c for c in out.columns if c.startswith("borrowing_base_")]

    def test_a_frame_carrying_TWO_clients_is_not_attributed_to_either(self):
        raw = tape(FACILITY_CLIENT)
        raw.loc[0, "client_id"] = "someone_else"
        _out, report = prepare_funded_mi_dataset(raw)
        assert report["borrowing_base_eligibility"]["applied"] is False

    @pytest.mark.parametrize("client_id", [FACILITY_CLIENT, "no_warehouse_here"])
    def test_no_existing_canonical_ECONOMIC_field_changes(self, client_id):
        """The additive guarantee, measured rather than asserted.

        Prepare the same tape twice — once as it is, once with the facility
        configuration made unreachable — and require every column the first run
        did NOT add to be identical. An eligibility derivation that touched a
        balance, a valuation, an LTV or a date would fail here.
        """
        import mi_agent.borrowing_base.config as bb_config

        raw = tape(client_id)
        with_facility, _ = prepare_funded_mi_dataset(raw.copy())

        real_loader = bb_config.load_facility
        try:
            bb_config.load_facility = lambda _client_id: None
            without_facility, _ = prepare_funded_mi_dataset(raw.copy())
        finally:
            bb_config.load_facility = real_loader

        shared = [c for c in without_facility.columns
                  if c in with_facility.columns]
        pd.testing.assert_frame_equal(with_facility[shared],
                                      without_facility[shared])
        added = set(with_facility.columns) - set(without_facility.columns)
        assert added <= set(GOVERNED_COLUMNS)

    def test_a_derivation_failure_never_breaks_preparation(self, monkeypatch):
        import mi_agent.borrowing_base.eligibility as elig

        def explode(*_a, **_k):
            raise RuntimeError("the eligibility engine fell over")

        monkeypatch.setattr(elig, "derive_eligibility", explode)
        out, report = prepare_funded_mi_dataset(tape(FACILITY_CLIENT))
        assert len(out) == 3                    # the frame still arrives
        assert report["borrowing_base_eligibility"]["reason"] == "derivation_error"


class TestTheStatusIsUsableDownstream:
    def test_the_frame_can_be_narrowed_to_the_eligible_population(self):
        from mi_agent.borrowing_base.config import load_facility
        from mi_agent.borrowing_base.eligibility import (
            eligibility_available,
            eligible_mask,
        )
        out, _report = prepare_funded_mi_dataset(tape(FACILITY_CLIENT))
        assert eligibility_available(out)
        assert eligible_mask(out, load_facility(FACILITY_CLIENT)).sum() == 3

    def test_a_production_book_with_no_rules_narrows_to_NOTHING(self, tmp_path,
                                                               monkeypatch):
        from mi_agent.borrowing_base import config as bb_config
        from mi_agent.borrowing_base.eligibility import eligible_mask

        (tmp_path / "config_client_prod_client.yaml").write_text(
            "client:\n  client_id: prod_client\n"
            "funding_facility:\n  facility_id: PROD_F\n"
            "  commitment: 100,000,000\n  advance_rate: 100\n",
            encoding="utf-8")
        monkeypatch.setenv(bb_config.CLIENT_CONFIG_DIR_ENV, str(tmp_path))
        out, report = prepare_funded_mi_dataset(tape("prod_client"))
        assert set(out[FIELD_ELIGIBILITY_STATUS]) == {UNDETERMINED}
        assert eligible_mask(out).sum() == 0
        assert report["borrowing_base_eligibility"]["eligibility_governed"] is False
