"""From the question a client is asked to the value a regulator receives.

The onboarding side of this platform is only worth anything if what a client
answers is what actually appears in the submission. This module walks the whole
distance for the three originator fields, without putting a value anywhere by
hand after onboarding:

    field catalogue
      -> onboarding answers (an operator's entity list)
      -> approved case
      -> OnboardingService.activate()
      -> versioned client artefact in the operations store
      -> EffectiveConfigResolver.client_config_for()
      -> the pinned client layer a run reads
      -> Gate 2 apply_config_defaults()      (canonical columns)
      -> Gate 4 regime_projector             (RREL82 / RREL83 / RREL84)

The client is deliberately not ERE, and its identifiers are synthetic and valid,
so nothing here depends on a legacy document or on any repository default.

Nothing in this module changes projector behaviour: it runs the real projector,
unmodified, with the configuration onboarding produced.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import operations_control.onboarding.catalogue as cat_mod
from operations_control.configuration.resolver import EffectiveConfigResolver
from operations_control.onboarding.service import OnboardingService
from operations_control.rules import RuleStore

REPO = Path(__file__).resolve().parents[2]
PROJECTOR = REPO / "engine" / "gate_4_projection" / "regime_projector.py"

#: The synthetic client. Distinct from ERE, and never written to the repository.
CLIENT_ID = "MERIDIAN_UK"
ORIGINATOR_NAME = "Meridian Origination Limited"
ORIGINATOR_LEI = "549300ABCDE123456702"
ORIGINATOR_COUNTRY = "GB"

#: A second entity, so the test proves the ORIGINATOR's identifier is chosen
#: rather than simply the only one available.
SPONSOR_NAME = "Meridian Capital Partners plc"
SPONSOR_LEI = "213800ABCDE123456701"

#: Canonical column -> Annex 2 code, as config/system/fields_registry.yaml maps
#: them. Asserted rather than assumed, so a registry change cannot silently
#: move a value to a different regulatory field.
CODE_FOR = {"originator_name": "RREL82",
            "originator_legal_entity_identifier": "RREL83",
            "originator_establishment_country": "RREL84"}


@pytest.fixture()
def onboarded(store, source_registry):
    """Take the synthetic client through onboarding and activate it.

    Every value below is supplied the way an operator supplies it — as an
    entity with a role. Nothing is written into a configuration file.
    """
    cat_mod.reset_cache()
    service = OnboardingService(store, registry_factory=lambda: source_registry)
    case = service.start_new_client(by="Operator")
    cid = case.case_id
    service.save_step(case_id=cid, step="client", by="Operator", payload={
        "client_id": CLIENT_ID, "client_name": "Meridian Lending",
        "jurisdiction": "GB", "reporting_currency": "GBP",
        "time_zone": "Europe/London"})
    case = service.save_step(case_id=cid, step="entities", by="Operator",
                             payload={"entities": [
                                 {"legal_name": ORIGINATOR_NAME,
                                  "roles": ["originator", "servicer"],
                                  "lei": ORIGINATOR_LEI,
                                  "country_of_establishment": ORIGINATOR_COUNTRY},
                                 {"legal_name": SPONSOR_NAME,
                                  "roles": ["sponsor"], "lei": SPONSOR_LEI,
                                  "country_of_establishment": "GB"}]})
    service.save_step(case_id=cid, step="contacts", by="Operator", payload={
        "reporting_contact_name": "Rae Reporter",
        "reporting_contact_email": "reporting@meridian.example",
        "operational_contact_name": "Ola Ops",
        "operational_contact_email": "ops@meridian.example"})
    owning = case.items("entities")[0]["entity_id"]
    service.save_step(case_id=cid, step="portfolios", by="Operator", payload={
        "portfolios": [{"portfolio_id": "direct_001",
                        "display_name": "Meridian Book",
                        "asset_class": "equity_release",
                        "portfolio_type": "direct",
                        "portfolio_structure": "whole_loan",
                        "originates": True, "owning_entity": owning,
                        "datasets": ["funded"], "cadence": "monthly"}]})
    service.save_step(case_id=cid, step="sources", by="Operator", payload={
        "sources": [{"source_key": "direct_001/funded",
                     "portfolio_id": "direct_001", "dataset": "funded",
                     "cadence": "monthly", "source_party": "Meridian core",
                     "delivery_channel": "sftp", "file_format": "csv"}]})
    service.save_step(case_id=cid, step="reporting", by="Operator",
                      payload={"products": ["esma_annex2"]})
    service.save_step(case_id=cid, step="regime", by="Operator", payload={
        "regime": {"esma_annex2": {
            "originator_name": ORIGINATOR_NAME,
            "originator_legal_entity_identifier": ORIGINATOR_LEI,
            "originator_establishment_country": ORIGINATOR_COUNTRY}}})
    case = service.load_case(cid)
    assert service.readiness(case)["ready"] is True, \
        service.readiness(case)["problems"]
    service.approve(case_id=cid, by="Administrator", reason="New client")
    result = service.activate(case_id=cid, by="Administrator")
    return service, result


@pytest.fixture()
def pinned_config(onboarded, store) -> Path:
    """The client layer a run would read — resolved, not hand-written."""
    resolver = EffectiveConfigResolver(store, RuleStore(store))
    return resolver.client_config_for(CLIENT_ID)


class TestOccToAnnex2Continuity:
    def test_the_registry_maps_these_canonical_fields_to_these_codes(self):
        registry = yaml.safe_load(
            (REPO / "config/system/fields_registry.yaml").read_text(
                encoding="utf-8"))
        for canonical, code in CODE_FOR.items():
            spec = registry["fields"][canonical]
            mapped = spec["regime_mapping"]["ESMA_Annex2"]["code"]
            assert mapped == code, (canonical, mapped, code)

    def test_activation_produced_a_versioned_artefact(self, onboarded):
        _, result = onboarded
        assert result["client_id"] == CLIENT_ID
        assert result["version"] == 1

    def test_the_resolved_client_layer_carries_the_answered_values(
            self, pinned_config):
        doc = yaml.safe_load(pinned_config.read_text(encoding="utf-8"))
        defaults = doc["defaults"]
        assert defaults["originator_name"] == ORIGINATOR_NAME
        assert defaults["originator_legal_entity_identifier"] == ORIGINATOR_LEI
        assert defaults["originator_establishment_country"] == ORIGINATOR_COUNTRY
        # The sponsor is present as itself, and did not become the originator.
        assert doc["reporting_parties"]["sponsor"][0]["lei"] == SPONSOR_LEI
        assert defaults["originator_legal_entity_identifier"] != SPONSOR_LEI

    def test_the_regulatory_preflight_accepts_the_resolved_configuration(
            self, pinned_config):
        """The deterministic Annex 2 gate reads the same pinned file, and the
        values it validates are the ones onboarding produced."""
        from operations_control.annex2 import preflight
        result = preflight.run_preflight(client_config_path=pinned_config,
                                         reporting_period="2026-06-30")
        assert result["blocked"] == [], result["blocked"]
        found = {c["key"]: c.get("found_value") for c in result["checks"]}
        assert found["originator_legal_entity_identifier"] == ORIGINATOR_LEI
        assert found["originator_establishment_country"] == ORIGINATOR_COUNTRY

    def test_gate_2_injects_the_answered_values_as_canonical_columns(
            self, pinned_config):
        """apply_config_defaults is where a client's standing values become
        row values. It is given the pinned configuration, not a literal."""
        import pandas as pd
        from engine.gate_2_transform.canonical_transform import (
            apply_config_defaults,
        )
        config = yaml.safe_load(pinned_config.read_text(encoding="utf-8"))
        frame = pd.DataFrame({"unique_identifier": ["L1", "L2"]})
        apply_config_defaults(frame, config)
        for canonical, expected in (
                ("originator_name", ORIGINATOR_NAME),
                ("originator_legal_entity_identifier", ORIGINATOR_LEI),
                ("originator_establishment_country", ORIGINATOR_COUNTRY)):
            assert canonical in frame.columns, canonical
            assert set(frame[canonical]) == {expected}, canonical

    def test_the_projector_emits_them_under_their_annex_2_codes(
            self, pinned_config, tmp_path):
        """The real projector, unmodified, over the pinned configuration.

        Gate 2 fills the columns from the client layer; Gate 4 renames them to
        the ESMA codes. The assertion is on the projected output, so it fails
        if either half of that path stops carrying the client's answer.
        """
        import pandas as pd
        from engine.gate_2_transform.canonical_transform import (
            apply_config_defaults,
        )
        config = yaml.safe_load(pinned_config.read_text(encoding="utf-8"))
        # The identifier and cut-off columns the Annex 2 guards require of any
        # record — supplied as loan data, which is where they come from. The
        # originator columns are deliberately NOT here: they arrive from the
        # client layer below, which is the whole point of the test.
        frame = pd.DataFrame({
            "unique_identifier": ["MERIDIAN-0001", "MERIDIAN-0002"],
            "data_cut_off_date": ["2026-06-30", "2026-06-30"],
            "new_underlying_exposure_identifier": ["MERIDIAN-0001",
                                                   "MERIDIAN-0002"],
            "underlying_exposure_identifier": ["MERIDIAN-0001",
                                               "MERIDIAN-0002"],
            "new_obligor_identifier": ["OBL-0001", "OBL-0002"],
        })
        apply_config_defaults(frame, config)
        canonical_csv = tmp_path / "canonical.csv"
        frame.to_csv(canonical_csv, index=False)

        out = tmp_path / "projected"
        out.mkdir()
        proc = subprocess.run(
            [sys.executable, str(PROJECTOR), str(canonical_csv),
             "--regime", "ESMA_Annex2",
             "--registry", str(REPO / "config/system/fields_registry.yaml"),
             "--enum-mapping", str(REPO / "config/system/enum_mapping.yaml"),
             "--config", str(pinned_config),
             "--template-order", str(REPO / "config/system/esma_code_order.yaml"),
             "--portfolio-type", "equity_release",
             "--output-dir", str(out)],
            capture_output=True, text=True, cwd=str(REPO))
        assert proc.returncode == 0, (proc.stderr or "")[-3000:]

        projected = sorted(out.glob("*_ESMA_Annex2_projected.csv"))
        assert projected, sorted(p.name for p in out.iterdir())
        result = pd.read_csv(projected[0], dtype=str)

        assert set(result["RREL82"]) == {ORIGINATOR_NAME}
        assert set(result["RREL83"]) == {ORIGINATOR_LEI}
        assert set(result["RREL84"]) == {ORIGINATOR_COUNTRY}
        # And the sponsor's identifier is nowhere in the regulatory output.
        assert SPONSOR_LEI not in projected[0].read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# The entity work must not move a single Annex 2 value
# --------------------------------------------------------------------------- #

def _project(config_path: Path, out: Path) -> "object":
    """Run the real projector over a fixed two-row frame."""
    import pandas as pd
    from engine.gate_2_transform.canonical_transform import apply_config_defaults
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    frame = pd.DataFrame({
        "unique_identifier": ["MERIDIAN-0001", "MERIDIAN-0002"],
        "data_cut_off_date": ["2026-06-30", "2026-06-30"],
        "new_underlying_exposure_identifier": ["MERIDIAN-0001", "MERIDIAN-0002"],
        "underlying_exposure_identifier": ["MERIDIAN-0001", "MERIDIAN-0002"],
        "new_obligor_identifier": ["OBL-0001", "OBL-0002"],
    })
    apply_config_defaults(frame, config)
    canonical = out / "canonical.csv"
    frame.to_csv(canonical, index=False)
    proc = subprocess.run(
        [sys.executable, str(PROJECTOR), str(canonical),
         "--regime", "ESMA_Annex2",
         "--registry", str(REPO / "config/system/fields_registry.yaml"),
         "--enum-mapping", str(REPO / "config/system/enum_mapping.yaml"),
         "--config", str(config_path),
         "--template-order", str(REPO / "config/system/esma_code_order.yaml"),
         "--portfolio-type", "equity_release", "--output-dir", str(out)],
        capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, (proc.stderr or "")[-3000:]
    return sorted(out.glob("*_ESMA_Annex2_projected.csv"))[0]


class TestAnnex2IsUnmovedByTheEntityWork:
    def test_naming_a_reporting_entity_changes_no_annex_2_value(
            self, pinned_config, tmp_path):
        """``reporting_parties`` records who the parties are. Annex 2 reports
        the exposures, and takes its originator fields from ``defaults`` — so
        adding a reporting entity to the client's structure must leave every
        projected value exactly where it was.
        """
        base = yaml.safe_load(pinned_config.read_text(encoding="utf-8"))
        assert "reporting_parties" in base

        without = dict(base)
        without["reporting_parties"] = {
            k: v for k, v in base["reporting_parties"].items()
            if k != "reporting_entity"}
        with_entity = dict(base)
        with_entity["reporting_parties"] = {
            **without["reporting_parties"],
            "reporting_entity": [{"legal_name": "Reporting Services Limited",
                                  "lei": "984500ABCDE123456704"}]}

        a = tmp_path / "a.yaml"
        a.write_text(yaml.safe_dump(without, sort_keys=False), encoding="utf-8")
        b = tmp_path / "b.yaml"
        b.write_text(yaml.safe_dump(with_entity, sort_keys=False),
                     encoding="utf-8")

        import pandas as pd
        left = pd.read_csv(_project(a, tmp_path / "pa"), dtype=str).fillna("")
        right = pd.read_csv(_project(b, tmp_path / "pb"), dtype=str).fillna("")

        assert list(left.columns) == list(right.columns)      # field set
        assert left.equals(right)                             # every value
        # And the reporting entity's identifier reaches no regulatory field.
        assert "984500ABCDE123456704" not in right.to_csv(index=False)

    def test_the_preflight_still_blocks_on_the_originator_identifier(
            self, tmp_path):
        """Wording now names the originator rather than the reporting entity.
        The key, the validator and the blocking decision are unchanged."""
        from operations_control.annex2 import preflight
        cfg = tmp_path / "c.yaml"

        cfg.write_text(yaml.safe_dump({
            "portfolio": {"country": "GB", "base_currency": "GBP"},
            "defaults": {"originator_establishment_country": "GB"}}),
            encoding="utf-8")
        missing = preflight.run_preflight(client_config_path=cfg,
                                          reporting_period="2026-06-30")
        assert [c["key"] for c in missing["blocked"]] \
            == ["originator_legal_entity_identifier"]

        cfg.write_text(yaml.safe_dump({
            "portfolio": {"country": "GB", "base_currency": "GBP"},
            "defaults": {"originator_legal_entity_identifier": "NOT-A-LEI",
                         "originator_establishment_country": "GB"}}),
            encoding="utf-8")
        invalid = preflight.run_preflight(client_config_path=cfg,
                                          reporting_period="2026-06-30")
        assert [c["key"] for c in invalid["blocked"]] \
            == ["originator_legal_entity_identifier"]
        assert "20-character LEI" in invalid["blocked"][0]["problem"]
