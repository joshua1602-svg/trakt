"""Tests for the ESMA Annex 2 provider.

These exercise the *real* route — the frozen Gate 4b normaliser, the frozen
Gate 5 builder, the authoritative 1.3.1 mapping workbook and the official XSD.
Nothing is mocked, because the claim under test is precisely that the agent
wraps the proven components rather than reimplementing them.

The full route takes roughly ten seconds, dominated by reading the 9.6 MB
mapping workbook, so one prepared run is shared across the assertions in
:class:`TestAnnex2Route` via a module-scoped fixture. Tests that must not build
(preflight refusals) run independently and cost nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from trakt_core.context import ExecutionContext
from trakt_core.errors import ErrorCode, TraktError

from engine.annex_delivery_agent.agent import AnnexDeliveryAgent
from engine.annex_delivery_agent.contracts import (
    DeliveryRequest, DeliveryResult, DeliveryStatus,
)
from engine.annex_delivery_agent.persistence import DeliveryRunStore
from engine.annex_delivery_agent.providers.annex2 import (
    ANNEX2_DEFINITION, ANNEX_ID, Annex2Paths, Annex2Provider,
)
from engine.annex_delivery_agent.publication import FilesystemPublicationSink
from engine.annex_delivery_agent.registry import default_registry

_REPO_ROOT = Path(__file__).resolve().parents[1]
CI_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "annex2_projected_ci.csv"

ANNEX2_NAMESPACE = "urn:esma:xsd:DRAFT1auth.099.001.04"
RECORD_TAG = f"{{{ANNEX2_NAMESPACE}}}UndrlygXpsrRcrd"

pytestmark = pytest.mark.skipif(
    not CI_FIXTURE.is_file(), reason="Annex 2 CI fixture is not present")


def _ctx(tenant: str = "ERE") -> ExecutionContext:
    return ExecutionContext.for_internal(tenant, actor_id=f"{tenant}-operator")


def _request(projected: Path = CI_FIXTURE, **overrides) -> DeliveryRequest:
    base = dict(annex_id=ANNEX_ID, reporting_date="2026-01-31",
                projected_input=str(projected), output_root="unused")
    base.update(overrides)
    return DeliveryRequest(**base)


# --------------------------------------------------------------------------- #
# Component resolution — the claim that this wraps the proven route
# --------------------------------------------------------------------------- #


class TestAuthoritativeComponents:
    def test_definition_names_the_exact_repository_components(self):
        d = ANNEX2_DEFINITION
        assert d.authoritative_projector == "engine/gate_4_projection/regime_projector.py"
        assert d.normaliser.startswith("engine/gate_4b_delivery/annex2_delivery_normalizer.py")
        assert d.builder.startswith("engine/gate_5_delivery/xml_builder_annex2.py")
        assert d.schema_reference == "config/system/DRAFT1auth.099.001.04_1.3.0.xsd"
        assert d.validator == "lxml.etree.XMLSchema"

    @pytest.mark.parametrize("attribute", [
        "delivery_rules", "field_universe", "code_order", "enum_mapping",
        "mapping_workbook", "xsd",
    ])
    def test_default_paths_exist_in_this_repository(self, attribute):
        assert Path(getattr(Annex2Paths(), attribute)).is_file(), (
            f"{attribute} must point at a real repository file, not an invented copy")

    def test_default_paths_match_the_proven_orchestrator_route(self):
        paths = Annex2Paths()
        assert paths.xsd.endswith("config/system/DRAFT1auth.099.001.04_1.3.0.xsd")
        assert paths.mapping_sheet == "DRAFT1auth.099.001.04"
        assert paths.mapping_workbook.endswith("_Version_1.3.1.xlsx")

    def test_component_versions_fingerprint_the_wrapped_modules(self):
        versions = Annex2Provider().component_versions()
        assert set(versions) == {"regime_projector", "annex2_delivery_normalizer",
                                 "xml_builder_annex2", "annex2_provider"}
        assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in versions.values())

    def test_annex2_is_registered_in_production(self):
        assert default_registry().supports(ANNEX_ID)
        assert default_registry().definition(ANNEX_ID).annex_id == ANNEX_ID


# --------------------------------------------------------------------------- #
# Preflight — must block before any expensive build
# --------------------------------------------------------------------------- #


class TestPreflight:
    @pytest.fixture
    def agent(self, tmp_path):
        return AnnexDeliveryAgent(store=DeliveryRunStore(tmp_path / "store"))

    def test_missing_projected_input_blocks(self, agent, tmp_path):
        result = DeliveryResult.from_dict(
            agent.prepare(_ctx(), _request(tmp_path / "absent.csv")).result)
        assert result.status == DeliveryStatus.PREFLIGHT_BLOCKED
        assert "projected Annex 2 data" in result.summary
        assert result.artefact is None

    def test_missing_reporting_date_blocks(self, agent):
        result = DeliveryResult.from_dict(
            agent.prepare(_ctx(), _request(reporting_date="")).result)
        assert result.status == DeliveryStatus.PREFLIGHT_BLOCKED
        assert "reporting date" in result.summary

    def test_empty_projected_input_blocks(self, agent, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text(CI_FIXTURE.read_text().splitlines()[0] + "\n", encoding="utf-8")
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(empty)).result)
        assert result.status == DeliveryStatus.PREFLIGHT_BLOCKED
        assert "no exposure records" in result.summary

    def test_missing_client_lei_blocks_with_the_operator_sentence(self, agent, tmp_path):
        """The example failure from the specification, end to end."""
        import csv

        rows = list(csv.reader(CI_FIXTURE.open(newline="")))
        header = rows[0]
        for column in ("RREL1", "RREL83"):
            if column in header:
                idx = header.index(column)
                for row in rows[1:]:
                    row[idx] = ""
        stripped = tmp_path / "no_lei.csv"
        with stripped.open("w", newline="") as fh:
            csv.writer(fh).writerows(rows)

        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(stripped)).result)
        assert result.status == DeliveryStatus.PREFLIGHT_BLOCKED
        assert "reporting entity LEI has not been configured for this client" in result.summary
        assert result.artefact is None, "no file may be built when preflight blocks"

    def test_non_annex2_input_blocks(self, agent, tmp_path):
        stranger = tmp_path / "not_annex2.csv"
        stranger.write_text("alpha,beta\n1,2\n", encoding="utf-8")
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(stranger)).result)
        assert result.status == DeliveryStatus.PREFLIGHT_BLOCKED
        assert "does not look like projected Annex 2 data" in result.summary

    def test_missing_xsd_blocks_with_a_configuration_message(self, agent, tmp_path):
        result = DeliveryResult.from_dict(
            agent.prepare(_ctx(), _request(options={"xsd": str(tmp_path / "absent.xsd")})).result)
        assert result.status == DeliveryStatus.PREFLIGHT_BLOCKED
        assert "official Annex 2 XML schema (XSD)" in result.summary

    def test_client_without_annex2_enabled_is_refused(self, agent, tmp_path):
        client_config = tmp_path / "client.yaml"
        client_config.write_text("supported_regimes:\n  - ESMA_Annex12\n", encoding="utf-8")
        result = DeliveryResult.from_dict(
            agent.prepare(_ctx(), _request(client_config_reference=str(client_config))).result)
        assert result.status == DeliveryStatus.PREFLIGHT_BLOCKED
        assert "not enabled for this client" in result.summary

    def test_preflight_failures_never_expose_a_traceback(self, agent, tmp_path):
        result = DeliveryResult.from_dict(
            agent.prepare(_ctx(), _request(tmp_path / "absent.csv")).result)
        assert "Traceback" not in result.summary
        assert "Error:" not in result.summary


# --------------------------------------------------------------------------- #
# The real route
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """One real Annex 2 run, shared by the assertions below."""
    root = tmp_path_factory.mktemp("annex2")
    agent = AnnexDeliveryAgent(store=DeliveryRunStore(root / "store"),
                               publication_sink=FilesystemPublicationSink(root / "published"))
    governed = agent.prepare(_ctx(), _request())
    return agent, governed, DeliveryResult.from_dict(governed.result)


class TestAnnex2Route:
    def test_produces_a_schema_valid_artefact(self, prepared):
        _, _, result = prepared
        assert result.status in (DeliveryStatus.PREPARED,
                                 DeliveryStatus.PREPARED_WITH_WARNINGS)
        assert result.schema_validation.executed is True
        assert result.schema_validation.valid is True
        assert result.schema_validation.error_count == 0

    def test_uses_the_authoritative_xsd(self, prepared):
        _, _, result = prepared
        assert result.schema_validation.schema_reference.endswith(
            "DRAFT1auth.099.001.04_1.3.0.xsd")
        assert result.schema_validation.validator == "lxml.etree.XMLSchema"
        assert re.fullmatch(r"[0-9a-f]{64}", result.schema_validation.schema_sha256)

    def test_artefact_exists_and_is_hashed(self, prepared):
        _, _, result = prepared
        path = Path(result.artefact.path)
        assert path.is_file()
        assert path.suffix == ".xml"
        assert result.artefact.size_bytes == path.stat().st_size > 0
        assert re.fullmatch(r"[0-9a-f]{64}", result.artefact.sha256)

    def test_normaliser_output_is_persisted_as_an_intermediate(self, prepared):
        _, _, result = prepared
        kinds = {a.kind for a in result.intermediate_artefacts}
        assert "delivery_ready_csv" in kinds
        assert "delivery_issues_csv" in kinds
        assert "delivery_report_json" in kinds
        ready = next(a for a in result.intermediate_artefacts
                     if a.kind == "delivery_ready_csv")
        assert Path(ready.path).is_file()
        assert Path(ready.path).name.endswith("_delivery_ready.csv"), (
            "the Gate 4b output contract is preserved")

    def test_record_count_matches_the_input(self, prepared):
        _, _, result = prepared
        expected = len(CI_FIXTURE.read_text().strip().splitlines()) - 1
        assert result.record_count == expected

    def test_one_xml_record_per_input_row(self, prepared):
        from lxml import etree

        _, _, result = prepared
        tree = etree.parse(result.artefact.path)
        assert len(tree.getroot().findall(f".//{RECORD_TAG}")) == result.record_count

    def test_namespace_and_schema_location_are_correct(self, prepared):
        from lxml import etree

        _, _, result = prepared
        root = etree.parse(result.artefact.path).getroot()
        assert etree.QName(root).namespace == ANNEX2_NAMESPACE
        assert etree.QName(root).localname == "Document"
        location = root.get("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation")
        assert location.startswith(ANNEX2_NAMESPACE)
        assert "DRAFT1auth.099.001.04_1.3.0.xsd" in location

    def test_field_counts_are_reported(self, prepared):
        _, _, result = prepared
        header = CI_FIXTURE.read_text().splitlines()[0].split(",")
        assert result.projected_field_count == len(header)
        assert 0 < result.delivered_field_count <= result.projected_field_count

    def test_element_order_follows_the_schema(self, prepared):
        """The artefact validating against the XSD *is* the ordering proof.

        Annex 2's schema is sequence-based, so a mis-ordered child is a
        validation error. Asserting the ordering separately would re-implement
        the XSD; asserting that a validator with sequence constraints accepted
        the file is the stronger check.
        """
        from lxml import etree

        _, _, result = prepared
        schema = etree.XMLSchema(etree.parse(str(Annex2Paths().xsd)))
        assert schema.validate(etree.parse(result.artefact.path))

    def test_configuration_versions_cover_every_annex2_input(self, prepared):
        _, _, result = prepared
        assert set(result.configuration_versions) >= {
            "annex2_delivery_rules", "annex2_field_universe", "esma_code_order",
            "enum_mapping", "annex2_mapping_workbook", "annex2_xsd"}
        assert all(v not in ("missing", "unset")
                   for v in result.configuration_versions.values())

    def test_timings_cover_each_stage(self, prepared):
        _, _, result = prepared
        stages = {t.stage: t.seconds for t in result.timings}
        assert set(stages) == {"preflight", "normalise", "build", "validate"}
        assert result.total_seconds >= sum(stages.values()) * 0.9


class TestAnnex2Evidence:
    def test_no_data_insertion_is_counted_and_reconciled(self, prepared):
        _, _, result = prepared
        evidence = result.transformation_evidence
        recon = evidence["nd_reconciliation"]
        assert recon["no_data_values_in_report"] >= recon["no_data_values_in_prepared_data"]
        assert (recon["difference_added_by_builder"]
                == recon["no_data_values_in_report"]
                - recon["no_data_values_in_prepared_data"])
        assert recon["unattributed"] == 0, (
            "every no-data value must be attributed to the data or to the builder")
        assert recon["attributed_as_automatic"] == recon["difference_added_by_builder"]

    def test_builder_inserted_nd5_is_disclosed_with_its_rule(self, prepared):
        """Whatever the builder inserts must be disclosed with its rule.

        Since Phase 2 the honest answer for this route is *nothing*: the
        reconciliation below proves every no-data node in the report came from
        the prepared data. The disclosure contract is unchanged and still
        applies to anything that does appear.
        """
        _, _, result = prepared
        fills = result.transformation_evidence["automatic_fills"]
        for fill in fills:
            assert fill["generated_value"] == "ND5"
            assert fill["records_affected"] > 0
            assert "builder rule:" in fill["reason"]
            assert fill["review_required"] is True
            assert fill["detection"] in ("structural", "intercepted", "reconciled")

    def test_every_no_data_node_is_attributed_to_the_prepared_data(self, prepared):
        """Phase 2: the builder adds no no-data node of its own.

        This is the assertion that replaced "ScndryOblgrIncm must appear in the
        automatic fills". RREL20/RREL21 are now declared in
        ``config/regime/annex2_delivery_rules.yaml`` and arrive as data, so the
        builder inserts nothing and the reconciliation closes at zero.
        """
        _, _, result = prepared
        recon = result.transformation_evidence["nd_reconciliation"]
        assert recon["no_data_values_in_report"] == recon["no_data_values_in_prepared_data"]
        assert recon["difference_added_by_builder"] == 0
        assert recon["attributed_as_automatic"] == 0
        assert recon["unattributed"] == 0
        assert recon["per_field_in_prepared_data"]["RREL20"] > 0
        assert recon["per_field_in_prepared_data"]["RREL21"] > 0

        paths = {f["xml_path"] for f in result.transformation_evidence["automatic_fills"]}
        assert not any("ScndryOblgrIncm" in p for p in paths), (
            "the secondary-obligor income branch is sourced from the prepared "
            "data now; the builder must not be credited with filling it")

    def test_omitted_fields_are_disclosed(self, prepared):
        _, _, result = prepared
        omissions = result.transformation_evidence["omissions"]
        assert omissions
        for omission in omissions:
            assert omission["records_affected"] > 0
            assert omission["reason"]

    def test_structural_assumptions_are_disclosed(self, prepared):
        _, _, result = prepared
        names = {a["name"] for a in
                 result.transformation_evidence["structural_assumptions"]}
        assert "one_record_per_row" in names
        assert "single_collateral_per_exposure" in names

    def test_unsupported_fields_are_listed(self, prepared):
        _, _, result = prepared
        unsupported = result.transformation_evidence["unsupported_fields"]
        assert unsupported
        assert all(u["field_code"] and u["reason"] for u in unsupported)

    def test_operator_warnings_are_plain_english(self, prepared):
        _, _, result = prepared
        lines = result.transformation_evidence["operator_warnings"]
        assert lines
        assert all(not line.startswith("_") for line in lines)
        # No internal identifier may leak into operator-facing text. Before
        # Phase 2 this route always produced a "filled automatically" line; it
        # now produces none, because nothing is filled automatically. What must
        # hold is that every line reads as prose.
        assert all(re.search(r"[a-z] [a-z]", line) for line in lines)
        assert any("prepared data" in line for line in lines)

    def test_evidence_is_persisted(self, prepared):
        import json

        _, _, result = prepared
        stored = json.loads(
            (Path(result.run_directory) / "04_transformation_evidence.json").read_text())
        assert stored["automatic_fill_count"] == \
            result.transformation_evidence["automatic_fill_count"]

    def test_status_reflects_that_review_is_required(self, prepared):
        _, _, result = prepared
        assert result.status == DeliveryStatus.PREPARED_WITH_WARNINGS
        assert result.requires_operator_review is True
        assert "not submitted" in result.why_it_matters
        assert "does not confirm" in result.why_it_matters


class TestAnnex2Coercion:
    def test_interception_is_transparent_and_records_the_rrel12_rewrite(self):
        """The observer must return exactly what the builder would have returned.

        This is the load-bearing property of the instrumentation: if the wrapper
        altered a value, the agent would be changing the builder's behaviour
        rather than reporting it.
        """
        from engine.gate_5_delivery import xml_builder_annex2 as builder
        from engine.annex_delivery_agent.evidence import CoercionCollector
        from engine.annex_delivery_agent.instrumentation import observe_coercions

        original = builder._coerce_record_value_for_branch
        cases = [("RREL12", "NOT-A-YEAR"), ("RREL12", "2019"), ("RREL1", "NOT-A-YEAR"),
                 ("RREL12", ""), ("RREC5", "GBZZZ")]
        expected = [original(code, value) for code, value in cases]

        collector = CoercionCollector()
        with observe_coercions(builder, "_coerce_record_value_for_branch", collector):
            observed = [builder._coerce_record_value_for_branch(c, v) for c, v in cases]

        assert observed == expected, "the observer must not change any value"
        assert builder._coerce_record_value_for_branch is original, "the patch is removed"

        evidence = {c.field_code: c for c in
                    collector.as_evidence(reason_for={"RREL12": "year branch"})}
        assert set(evidence) == {"RREL12"}, "only the changed value is recorded"
        assert evidence["RREL12"].records_affected == 1
        # Phase 2: the value is routed to no-data, not rewritten to "2026".
        assert evidence["RREL12"].generated_value == ""
        assert "NOT-A-YEAR" in evidence["RREL12"].original_value_sample

    def test_observer_is_removed_even_when_the_build_raises(self):
        from engine.gate_5_delivery import xml_builder_annex2 as builder
        from engine.annex_delivery_agent.evidence import CoercionCollector
        from engine.annex_delivery_agent.instrumentation import observe_coercions

        original = builder._coerce_record_value_for_branch
        with pytest.raises(RuntimeError):
            with observe_coercions(builder, "_coerce_record_value_for_branch",
                                   CoercionCollector()):
                raise RuntimeError("build failed")
        assert builder._coerce_record_value_for_branch is original

    def test_rrel12_non_year_values_are_refused_not_rewritten(self, tmp_path):
        """A non-year RREL12 is rejected at Gate 4b; nothing is rewritten.

        This previously asserted the builder produced ``"2026"`` and disclosed
        it as a coercion. Both halves of that are gone: the builder no longer
        substitutes a year, and the delivery rule no longer maps region names
        onto one, so the value now fails the declared regex guard and the run
        stops. The test used to pass only because it skipped when normalisation
        failed — the skip is now the expected outcome, so it is asserted.
        """
        import csv

        rows = list(csv.reader(CI_FIXTURE.open(newline="")))
        header = rows[0]
        if "RREL12" not in header:
            pytest.skip("fixture does not carry RREL12")
        idx = header.index("RREL12")
        for row in rows[1:]:
            row[idx] = "NOT-A-YEAR"
        mutated = tmp_path / "coerce.csv"
        with mutated.open("w", newline="") as fh:
            csv.writer(fh).writerows(rows)

        agent = AnnexDeliveryAgent(store=DeliveryRunStore(tmp_path / "store"))
        result = DeliveryResult.from_dict(agent.prepare(_ctx(), _request(mutated)).result)

        # The governed outcome: refused at Gate 4b, before any XML is built.
        assert result.status == DeliveryStatus.NORMALISATION_FAILED, (
            f"a non-year RREL12 must fail delivery normalisation, got "
            f"{result.status}")

        # And it must never have been turned into a year anywhere.
        coercions = {c["field_code"]: c
                     for c in (result.transformation_evidence or {}).get("coercions", ())}
        assert coercions.get("RREL12", {}).get("generated_value") != "2026", (
            "the RREL12 -> '2026' fabrication is gone from both the builder "
            "and config/regime/annex2_delivery_rules.yaml")

    def test_instrumentation_leaves_the_builder_unpatched(self, prepared):
        """The observers must be removed once a build finishes."""
        from engine.gate_5_delivery import xml_builder_annex2 as builder

        assert builder._coerce_record_value_for_branch.__module__ == builder.__name__
        assert builder.select_specs_for_value.__module__ == builder.__name__


class TestAnnex2Publication:
    def test_a_valid_report_is_prepared_not_published(self, prepared):
        _, _, result = prepared
        assert result.status != DeliveryStatus.PUBLISHED
        assert result.approval.state == "not_requested"
        assert result.publication is None
        assert result.publication_eligible is True

    def test_publication_requires_approval_then_succeeds(self, prepared):
        agent, _, result = prepared
        gate = agent.gate_for(_ctx(), _request(), run_id=result.run_id)

        with pytest.raises(TraktError) as exc:
            gate.publish(_ctx())
        assert exc.value.code == ErrorCode.ARTEFACT_NOT_APPROVED

        gate.approve(_ctx(), note="ND5 fills reviewed")
        published = gate.publish(_ctx())

        assert published.status == DeliveryStatus.PUBLISHED
        package = published.publication["package"]
        assert package["schema"].endswith("DRAFT1auth.099.001.04_1.3.0.xsd")
        assert package["validation_report"]["valid"] is True
        # The evidence block travels with the package whether or not anything
        # was filled. Since Phase 2 nothing is, and the package says so rather
        # than omitting the key.
        assert package["automatic_value_evidence"]["automatic_fill_count"] == 0
        assert package["automatic_value_evidence"]["nd_reconciliation"][
            "difference_added_by_builder"] == 0
        assert package["operator_approval"]["decided_by"] == "ERE-operator"
        assert Path(published.publication["receipt"]["artefact"]).is_file()

    def test_another_tenant_cannot_reach_the_run(self, prepared):
        agent, _, result = prepared
        with pytest.raises(TraktError):
            agent.gate_for(_ctx("OTHER"), _request(), run_id=result.run_id)
