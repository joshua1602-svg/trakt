"""The governed facility configuration loader.

Two things are being protected here. The first is that facility terms live in
CONFIGURATION and not in Python: the shipped prototype file must actually carry
the operator's figures, and a change to them must be a change to that file. The
second is precedence: a client's own approved configuration wins, and a client
with no facility gets nothing rather than somebody else's.
"""

from __future__ import annotations

import textwrap

import pytest

from mi_agent.borrowing_base import config as cfg
from mi_agent.borrowing_base.models import ENV_PRODUCTION, ENV_PROTOTYPE


@pytest.fixture(scope="module")
def ere():
    facility = cfg.load_facility("ere_funding_uk")
    assert facility is not None, "the prototype facility must be configured"
    return facility


def write_client_config(directory, client_id: str, block: str) -> None:
    """A client configuration document carrying a ``funding_facility:`` block.

    ``block`` is written as YAML lines indented two spaces under the key, which
    is what OCC's generator produces.
    """
    body = "\n".join("  " + line if line.strip() else line
                      for line in textwrap.dedent(block).strip().splitlines())
    (directory / f"config_client_{client_id}.yaml").write_text(
        f"client:\n  client_id: {client_id}\nfunding_facility:\n{body}\n",
        encoding="utf-8")


class TestTheShippedPrototypeFacility:
    """The operator's supplied terms, read off the governed file."""

    def test_the_facility_identity(self, ere):
        assert ere.facility_id == "ERE_WAREHOUSE_01"
        assert ere.facility_type == "warehouse"
        assert ere.currency == "GBP"

    def test_the_commitment_is_250m(self, ere):
        assert ere.commitment == 250_000_000.0

    def test_the_advance_rate_is_103_percent_held_as_a_ratio(self, ere):
        assert ere.advance_rate == 1.03
        assert ere.advance_rate_pct == 103.0

    def test_the_denominator_floor_is_33m(self, ere):
        assert ere.concentration_denominator_floor == 33_000_000.0

    def test_no_drawn_amount_is_fabricated(self, ere):
        assert ere.current_drawn_amount is None
        assert ere.drawn_available is False

    def test_it_is_marked_a_prototype_with_the_assumption_active(self, ere):
        assert ere.environment == ENV_PROTOTYPE
        assert ere.prototype_assumption_active is True
        assert ere.eligibility_governed is False

    def test_schedule_8_numerators_are_scoped_to_eligible_loans(self, ere):
        assert ere.concentration_population == "eligible_mortgage_loans"

    def test_breaches_are_monitored_and_nothing_is_deducted(self, ere):
        assert ere.borrowing_base_treatment == "monitor_only"

    def test_the_facility_documents_are_named_as_provenance(self, ere):
        docs = ere.governance["source_documents"]
        assert any(d["reference"].startswith("Warehouse Facility Schedule 8")
                   and d["received"] for d in docs)
        assert any("Facility agreement" in d["reference"] and not d["received"]
                   for d in docs)

    def test_it_validates(self, ere):
        assert ere.validate() == []

    def test_it_carries_a_version_and_a_content_hash_for_the_receipt(self, ere):
        assert ere.config_version
        assert len(ere.content_hash()) == 16


class TestNoFacility:
    def test_a_client_with_no_facility_gets_NOTHING_not_a_default(self):
        assert cfg.load_facility("a_client_with_no_warehouse") is None

    def test_a_blank_client_id_resolves_nothing(self):
        assert cfg.load_facility("") is None


class TestPrecedence:
    """The client's own approved configuration wins over the platform register."""

    def test_a_client_config_block_beats_the_platform_register(self, tmp_path,
                                                               monkeypatch):
        write_client_config(tmp_path, "ere_funding_uk", """
            facility_id: FROM_CLIENT_CONFIG
            commitment: 400,000,000
            advance_rate: 95
            environment: production
        """)
        monkeypatch.setenv(cfg.CLIENT_CONFIG_DIR_ENV, str(tmp_path))
        facility = cfg.load_facility("ere_funding_uk")
        assert facility.facility_id == "FROM_CLIENT_CONFIG"
        assert facility.commitment == 400_000_000.0
        assert facility.config_source.startswith("client_config:")

    def test_a_percentage_advance_rate_is_read_as_a_ratio(self, tmp_path,
                                                          monkeypatch):
        # OCC writes the operator's "103" straight through; 103 can only mean
        # 103%, and the loader normalises it rather than advancing 10,300%.
        write_client_config(tmp_path, "c1", "facility_id: F\nadvance_rate: 103\n")
        monkeypatch.setenv(cfg.CLIENT_CONFIG_DIR_ENV, str(tmp_path))
        assert cfg.load_facility("c1").advance_rate == 1.03

    def test_a_ratio_advance_rate_is_taken_verbatim(self, tmp_path, monkeypatch):
        write_client_config(tmp_path, "c2", "facility_id: F\nadvance_rate: 0.85\n")
        monkeypatch.setenv(cfg.CLIENT_CONFIG_DIR_ENV, str(tmp_path))
        assert cfg.load_facility("c2").advance_rate == 0.85

    def test_one_clients_block_is_never_served_to_another(self, tmp_path,
                                                          monkeypatch):
        write_client_config(tmp_path, "client_a", "facility_id: A_ONLY\n")
        monkeypatch.setenv(cfg.CLIENT_CONFIG_DIR_ENV, str(tmp_path))
        assert cfg.load_facility("client_b") is None

    def test_an_OCC_written_block_defaults_to_PRODUCTION_and_fails_closed(
            self, tmp_path, monkeypatch):
        # Nothing OCC writes says "prototype", so a facility onboarded through
        # the wizard can never inherit the demonstration assumption.
        write_client_config(tmp_path, "c3", """
            facility_id: F
            commitment: 100,000,000
            advance_rate: 100
            eligibility:
              prototype_assume_financing_portfolio_eligible: true
        """)
        monkeypatch.setenv(cfg.CLIENT_CONFIG_DIR_ENV, str(tmp_path))
        facility = cfg.load_facility("c3")
        assert facility.environment == ENV_PRODUCTION
        assert facility.prototype_assumption_active is False
        assert any("will NOT be honoured" in p for p in facility.validate())


class TestRobustness:
    def test_a_missing_amount_is_None_and_never_zero(self, tmp_path, monkeypatch):
        # A zero commitment and an absent commitment are different facts.
        (tmp_path / "config_client_c4.yaml").write_text(
            "client:\n  client_id: c4\nfunding_facility:\n  facility_id: F\n"
            "  commitment:\n", encoding="utf-8")
        monkeypatch.setenv(cfg.CLIENT_CONFIG_DIR_ENV, str(tmp_path))
        assert cfg.load_facility("c4").commitment is None

    def test_an_unreadable_register_behaves_as_no_facility_not_a_crash(
            self, tmp_path, monkeypatch):
        broken = tmp_path / "broken.yaml"
        broken.write_text("facilities: [ this is not: valid: yaml", encoding="utf-8")
        monkeypatch.setenv(cfg.FACILITIES_PATH_ENV, str(broken))
        monkeypatch.setenv(cfg.CLIENT_CONFIG_DIR_ENV, str(tmp_path))
        assert cfg.load_facility("ere_funding_uk") is None

    def test_load_facility_checked_surfaces_the_problems(self):
        facility, problems = cfg.load_facility_checked("ere_funding_uk")
        assert facility is not None
        assert problems == []


class TestValidation:
    def test_a_negative_commitment_is_refused(self):
        from mi_agent.borrowing_base.models import FacilityConfiguration
        problems = FacilityConfiguration(facility_id="F",
                                         commitment=-1.0).validate()
        assert any("commitment cannot be negative" in p for p in problems)

    def test_an_unknown_environment_is_refused(self):
        from mi_agent.borrowing_base.models import FacilityConfiguration
        problems = FacilityConfiguration(facility_id="F",
                                         environment="staging").validate()
        assert any("unknown environment" in p for p in problems)

    def test_an_unknown_breach_treatment_is_refused(self):
        from mi_agent.borrowing_base.models import FacilityConfiguration
        problems = FacilityConfiguration(
            facility_id="F", borrowing_base_treatment="deduct_it_all").validate()
        assert any("unknown borrowing_base_treatment" in p for p in problems)
