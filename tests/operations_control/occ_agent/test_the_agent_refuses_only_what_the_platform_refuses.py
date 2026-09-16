"""The Agent must not refuse a delivery the platform would accept.

Reported from a real case. A live equity release onboarding was blocked on
eight required fields, among them ``maturity_date`` — and a lifetime mortgage
HAS no contractual maturity date. The same client's files, loaded through the
platform's own ingestion route, were not blocked at all.

The configuration had answered this all along, in two places:

    maturity_date:
      applicability:
        equity_release:
          allowed_missing: true
          severity_if_missing: warning
          nd_default: ND2
          reason: "Lifetime mortgage / equity release products do not have a
                   fixed contractual maturity."

and again, per product, in ``config/asset/product_profiles.yaml``, where
``maturity_date`` is ``base_mi: not_applicable``.

The Agent consulted neither. It re-implemented the core-presence check inline
and hard-coded ``severity: "error"`` on every finding, so no applicability
block was ever read; and it passed the PORTFOLIO type ("direct" / "acquired")
where applicability is keyed on the ASSET CLASS ("equity_release"), so even a
restored lookup would have found nothing.

That is the defect class behind most of this: a second copy of a platform
check, which drifts from the original and refuses what the original permits.
The fix is not a better copy. It is to call the platform's own function and
give it the argument it asks for.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.gate_3_validation.validate_canonical import (
    get_core_required_fields,
    load_registry,
    select_fields_for_portfolio,
    validate_core_presence,
)

REGISTRY = Path("config/system/fields_registry.yaml")

#: Absent from the real ERE tape, and each one answered by configuration.
NOT_EXPECTED_FOR_EQUITY_RELEASE = ("maturity_date",
                                   "originator_legal_entity_identifier")


@pytest.fixture(scope="module")
def profile():
    from engine.onboarding_agent.product_profile import load_product_profiles
    return load_product_profiles()["equity_release_lifetime_mortgage"]


@pytest.fixture(scope="module")
def registry_fields():
    return select_fields_for_portfolio(load_registry(REGISTRY), "direct")


def tape_without(fields, registry_fields):
    """A one-row tape carrying every core field except those named."""
    core = get_core_required_fields(registry_fields)
    return pd.DataFrame({c: ["x"] for c in core if c not in set(fields)})


def severities(frame, registry_fields, asset_class):
    core = get_core_required_fields(registry_fields)
    return {v.field: v.severity
            for v in validate_core_presence(frame, core, registry_fields,
                                            asset_class)
            if v.rule_id == "CORE001"}


class TestApplicabilityIsHonoured:

    def test_a_lifetime_mortgage_without_a_maturity_date_is_not_an_error(
            self, registry_fields):
        """The reported blocker. An equity release loan has no contractual
        maturity, and the registry says so in the field's own declaration."""
        frame = tape_without(NOT_EXPECTED_FOR_EQUITY_RELEASE, registry_fields)
        found = severities(frame, registry_fields, "equity_release")
        assert found["maturity_date"] == "warn"

    def test_the_originators_identifier_is_not_an_error_either(
            self, registry_fields):
        """Supplied by the governed client configuration at projection, and
        enforced for the regulatory delivery by the Annex 2 preflight."""
        frame = tape_without(NOT_EXPECTED_FOR_EQUITY_RELEASE, registry_fields)
        found = severities(frame, registry_fields, "equity_release")
        assert found["originator_legal_entity_identifier"] == "warn"

    def test_a_field_with_no_applicability_block_still_errors(
            self, registry_fields):
        """Nothing here makes the check more forgiving in general. A core
        field the configuration has NOT excused is still an error."""
        frame = tape_without(["amortisation_type"], registry_fields)
        found = severities(frame, registry_fields, "equity_release")
        assert found["amortisation_type"] == "error"

    def test_the_asset_class_is_what_unlocks_it_not_the_portfolio_type(
            self, registry_fields):
        """Applicability is keyed on "equity_release". Passing "direct" — how
        the book was ACQUIRED — finds nothing and falls through to the strict
        default, which was the second half of the defect."""
        frame = tape_without(NOT_EXPECTED_FOR_EQUITY_RELEASE, registry_fields)
        by_portfolio = severities(frame, registry_fields, "direct")
        by_asset = severities(frame, registry_fields, "equity_release")

        assert by_portfolio["maturity_date"] == "error"
        assert by_asset["maturity_date"] == "warn"


class TestTheAgentUsesThePlatformsCheck:

    def test_the_agent_no_longer_carries_its_own_copy(self):
        """A second copy of a platform check drifts from the original and
        refuses what the original permits. This asserts the copy is gone
        rather than that it now behaves — a copy that agrees today is still a
        copy that can disagree tomorrow."""
        source = Path("operations_control/occ_agent/execution.py").read_text(
            encoding="utf-8")
        assert "validate_core_presence" in source, \
            "the Agent does not call the platform's core-presence check"
        assert '"rule_id": "CORE001"' not in source, \
            "the Agent is still constructing CORE001 findings itself"
        assert '"rule_id": "CORE002"' not in source, \
            "the Agent is still constructing CORE002 findings itself"

    def test_it_passes_the_asset_class(self):
        source = Path("operations_control/occ_agent/execution.py").read_text(
            encoding="utf-8")
        assert "self.asset_type or spec.source_portfolio_type" in source, \
            "the asset class is not reaching the applicability lookup"


class TestWhatTheProductProfileSays:
    """The governed per-product answer, which the Agent's blocking decision
    still has to be taught to read. Asserted here so the intended behaviour is
    recorded against the configuration rather than against an opinion."""

    @pytest.mark.parametrize("field", [
        "maturity_date", "amortisation_type", "interest_rate_type",
        "originator_legal_entity_identifier", "originator_name",
        # The denomination is the client's approved reporting currency, held in
        # the governed client configuration and defaulted by the asset pack. It
        # is not restated per loan on a monthly extract.
        "exposure_currency_denomination",
    ])
    def test_these_do_not_block_base_mi(self, profile, field):
        assert profile.is_non_blocking_for_base_mi(field), field

    @pytest.mark.parametrize("field", [
        "current_principal_balance", "data_cut_off_date",
    ])
    def test_these_do_block_base_mi(self, profile, field):
        """Management information without a balance or a cut-off date is not
        a report. The configuration says so; it is not a judgement made here."""
        assert not profile.is_non_blocking_for_base_mi(field), field
