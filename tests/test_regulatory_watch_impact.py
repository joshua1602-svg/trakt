"""tests/test_regulatory_watch_impact.py — delta -> Trakt component impact.

Asserts that a regulatory delta is mapped onto the components that actually
carry Annex 2 behaviour in this repository, using the live configuration as it
stands. The assertions are written against facts the repo really holds (e.g.
RREL1 has a delivery rule with a regex validator and no permitted ND value), so
a config change that invalidates an assertion is a signal, not noise.

It also pins the Stage 1 boundary: the impact pass reads configuration and
writes nothing.

Run: python -m pytest tests/test_regulatory_watch_impact.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch.contracts import (                    # noqa: E402
    COMPONENT_CODE_ORDER,
    COMPONENT_ENUM_MAPPING,
    COMPONENT_FIELD_REGISTRY,
    COMPONENT_ND_BEHAVIOUR,
    COMPONENT_TESTS,
    COMPONENT_VALIDATION,
    COMPONENT_XML_MAPPING,
    CONFIG_CHANGE_REQUIRED,
    DETERMINISTIC,
    ENUM_CHANGED,
    FIELD_ADDED,
    FIELD_DESCRIPTION_CHANGED,
    FIELD_REMOVED,
    FORMAT_CHANGED,
    IMPACT_STATUSES,
    MANDATORY_STATUS_CHANGED,
    MANUAL_REVIEW_REQUIRED,
    ND_PERMISSION_CHANGED,
    NO_IMPLEMENTATION_CHANGE,
    ORDER_CHANGED,
    REVIEW_REQUIRED,
    SEV_HIGH,
    SpecDelta,
    TEST_CHANGE_REQUIRED,
    VALIDATION_CHANGE_REQUIRED,
    XML_CHANGE_REQUIRED,
    XML_PATH_CHANGED,
)
from regulatory_watch.impact import (                       # noqa: E402
    P_CODE_ORDER, P_DELIVERY_RULES, P_ENUM_MAPPING, P_REGISTRY,
    P_XSD_PATH_MAP, TraktImplementationIndex, assess,
)


@pytest.fixture(scope="module")
def index():
    return TraktImplementationIndex.load(_REPO)


def _delta(code, change_type, old=None, new=None,
           confidence=DETERMINISTIC) -> SpecDelta:
    return SpecDelta(code=code, change_type=change_type, old_value=old,
                     new_value=new, severity=SEV_HIGH, confidence=confidence,
                     source_version_a="A", source_version_b="B")


def _statuses(findings, component):
    return {f.status for f in findings if f.component == component}


# --------------------------------------------------------------------------- #
# The index reflects the live repository
# --------------------------------------------------------------------------- #

def test_index_loads_the_live_annex2_configuration(index):
    assert index.registry_by_code, "fields_registry regime mapping is empty"
    assert index.code_order, "esma_code_order.yaml Record list is empty"
    assert index.delivery_rules, "annex2_delivery_rules field_rules is empty"
    assert index.universe, "annex2_field_universe fields is empty"
    assert index.xsd_path_map, "annex2_field_xsd_path_map fields is empty"
    assert index.registry_by_code["RREL69"]["canonical_field"] == \
        "account_status"


def test_index_resolves_enum_mappings_through_the_canonical_field(index):
    maps = index.enum_map_for("RREL42")
    assert "delivery_rule_enum_map" in maps
    assert index.canonical_field("RREL42") == "interest_rate_type"
    assert "FXRL" in index.configured_enum_targets("RREL42")


def test_index_finds_annex2_test_references(index):
    assert index.test_references, "no Annex 2 code found in the test surface"
    assert all(paths for paths in index.test_references.values())


# --------------------------------------------------------------------------- #
# Per-change-type mapping
# --------------------------------------------------------------------------- #

def test_field_added_requires_config_order_xml_and_tests(index):
    findings = assess([_delta("RREL999", FIELD_ADDED,
                              new={"label": "New Field", "xml_path": "/x",
                                   "mandatory": "mandatory",
                                   "nd_allowed": ["ND5"]})], index=index)
    assert _statuses(findings, COMPONENT_FIELD_REGISTRY) == \
        {CONFIG_CHANGE_REQUIRED}
    assert _statuses(findings, COMPONENT_CODE_ORDER) == {CONFIG_CHANGE_REQUIRED}
    assert _statuses(findings, COMPONENT_XML_MAPPING) == {XML_CHANGE_REQUIRED}
    assert _statuses(findings, COMPONENT_ND_BEHAVIOUR) == \
        {CONFIG_CHANGE_REQUIRED}
    assert _statuses(findings, COMPONENT_TESTS) == {TEST_CHANGE_REQUIRED}
    assert all(f.locations for f in findings)


def test_field_removed_flags_every_component_that_still_holds_the_code(index):
    findings = assess([_delta("RREL1", FIELD_REMOVED,
                              old={"label": "Unique Identifier",
                                   "xml_path": "/x", "mandatory": "mandatory",
                                   "nd_allowed": []})], index=index)
    assert _statuses(findings, COMPONENT_FIELD_REGISTRY) == \
        {CONFIG_CHANGE_REQUIRED}
    assert _statuses(findings, COMPONENT_ND_BEHAVIOUR) == \
        {CONFIG_CHANGE_REQUIRED}
    assert P_REGISTRY in [loc for f in findings for loc in f.locations]


def test_field_removed_for_an_unmapped_code_is_no_change(index):
    findings = assess([_delta("RREL997", FIELD_REMOVED,
                              old={"label": "x", "xml_path": None,
                                   "mandatory": "optional",
                                   "nd_allowed": []})], index=index)
    assert {f.status for f in findings} == {NO_IMPLEMENTATION_CHANGE}


def test_nd_withdrawal_that_breaks_a_configured_value_requires_change(index):
    """RREC9 is configured with nd_allowed [ND5]; withdrawing ND5 breaks it."""
    assert (index.delivery_rules["RREC9"].get("nd_allowed")) == ["ND5"]
    findings = assess([_delta("RREC9", ND_PERMISSION_CHANGED,
                              old={"nd_allowed": ["ND5"]},
                              new={"nd_allowed": []})], index=index)
    assert _statuses(findings, COMPONENT_ND_BEHAVIOUR) == \
        {CONFIG_CHANGE_REQUIRED}
    assert _statuses(findings, COMPONENT_VALIDATION) == \
        {VALIDATION_CHANGE_REQUIRED}
    nd = next(f for f in findings if f.component == COMPONENT_ND_BEHAVIOUR)
    assert nd.current_implementation["no_longer_permitted"] == ["ND5"]


def test_nd_widening_that_keeps_the_configured_envelope_is_review_only(index):
    findings = assess([_delta("RREC9", ND_PERMISSION_CHANGED,
                              old={"nd_allowed": ["ND5"]},
                              new={"nd_allowed": ["ND1", "ND5"]})], index=index)
    assert _statuses(findings, COMPONENT_ND_BEHAVIOUR) == \
        {MANUAL_REVIEW_REQUIRED}
    assert COMPONENT_VALIDATION not in {f.component for f in findings}


def test_enum_withdrawal_of_a_configured_target_requires_config_change(index):
    """RREL42 maps canonical 'Fixed' -> FXRL; withdrawing FXRL breaks it."""
    assert "FXRL" in index.configured_enum_targets("RREL42")
    findings = assess([_delta("RREL42", ENUM_CHANGED,
                              old={"enum_values": ["FXRL", "FLIF", "OTHR"]},
                              new={"enum_values": ["FLIF", "OTHR"]})],
                      index=index)
    assert _statuses(findings, COMPONENT_ENUM_MAPPING) == \
        {CONFIG_CHANGE_REQUIRED}
    assert _statuses(findings, COMPONENT_VALIDATION) == \
        {VALIDATION_CHANGE_REQUIRED}
    enum = next(f for f in findings if f.component == COMPONENT_ENUM_MAPPING)
    assert "FXRL" in enum.rationale
    assert P_ENUM_MAPPING in enum.locations


def test_enum_addition_is_manual_review_where_a_mapping_exists(index):
    findings = assess([_delta("RREL42", ENUM_CHANGED,
                              old={"enum_values": ["FXRL"]},
                              new={"enum_values": ["FXRL", "NEWC"]})],
                      index=index)
    assert _statuses(findings, COMPONENT_ENUM_MAPPING) == \
        {MANUAL_REVIEW_REQUIRED}


def test_format_change_hits_the_configured_regex_validator(index):
    """RREL1 carries a regex validator pinned to the previous format."""
    assert "regex" in (index.delivery_rules["RREL1"].get("validators") or {})
    findings = assess([_delta("RREL1", FORMAT_CHANGED,
                              old={"format_pattern": "a"},
                              new={"format_pattern": "b"})], index=index)
    assert _statuses(findings, COMPONENT_VALIDATION) == \
        {VALIDATION_CHANGE_REQUIRED}
    assert _statuses(findings, COMPONENT_FIELD_REGISTRY) == \
        {CONFIG_CHANGE_REQUIRED}
    validation = next(f for f in findings
                      if f.component == COMPONENT_VALIDATION)
    assert P_DELIVERY_RULES in validation.locations


def test_mandatory_change_hits_validation_and_registry(index):
    findings = assess([_delta("RREL69", MANDATORY_STATUS_CHANGED,
                              old={"mandatory": "optional"},
                              new={"mandatory": "mandatory"})], index=index)
    assert _statuses(findings, COMPONENT_VALIDATION) == \
        {VALIDATION_CHANGE_REQUIRED}
    assert _statuses(findings, COMPONENT_FIELD_REGISTRY) == \
        {CONFIG_CHANGE_REQUIRED}


def test_xml_path_change_hits_the_path_map_and_the_workbook_semantic(index):
    findings = assess([_delta("RREL1", XML_PATH_CHANGED,
                              old={"xml_path": "/a"},
                              new={"xml_path": "/b"})], index=index)
    assert _statuses(findings, COMPONENT_XML_MAPPING) == {XML_CHANGE_REQUIRED}
    assert _statuses(findings, COMPONENT_ND_BEHAVIOUR) == \
        {CONFIG_CHANGE_REQUIRED}
    xml = next(f for f in findings if f.component == COMPONENT_XML_MAPPING)
    assert P_XSD_PATH_MAP in xml.locations


def test_order_change_hits_the_configured_code_ordering(index):
    assert "RREL42" in index.code_order
    findings = assess([_delta("RREL42", ORDER_CHANGED,
                              old={"position": 10}, new={"position": 3})],
                      index=index)
    assert _statuses(findings, COMPONENT_CODE_ORDER) == {CONFIG_CHANGE_REQUIRED}
    order = next(f for f in findings if f.component == COMPONENT_CODE_ORDER)
    assert P_CODE_ORDER in order.locations
    assert order.current_implementation["configured_position"] is not None


def test_description_only_change_needs_no_implementation_change(index):
    findings = assess([_delta("RREL1", FIELD_DESCRIPTION_CHANGED,
                              old={"description": "a"},
                              new={"description": "b"})], index=index)
    assert {f.status for f in findings} == {NO_IMPLEMENTATION_CHANGE}


def test_a_review_required_delta_always_adds_a_manual_review_finding(index):
    findings = assess([_delta("RREL1", FORMAT_CHANGED,
                              old={"format_pattern": "a"},
                              new={"format_pattern": "b"},
                              confidence=REVIEW_REQUIRED)], index=index)
    assert MANUAL_REVIEW_REQUIRED in {f.status for f in findings}


def test_one_delta_may_affect_several_components(index):
    findings = assess([_delta("RREC9", ND_PERMISSION_CHANGED,
                              old={"nd_allowed": ["ND5"]},
                              new={"nd_allowed": []})], index=index)
    assert len({f.component for f in findings}) > 1


def test_every_finding_uses_the_published_status_vocabulary(index):
    deltas = [
        _delta("RREL1", FORMAT_CHANGED, {"format_pattern": "a"},
               {"format_pattern": "b"}),
        _delta("RREC9", ND_PERMISSION_CHANGED, {"nd_allowed": ["ND5"]},
               {"nd_allowed": []}),
        _delta("RREL42", ENUM_CHANGED, {"enum_values": ["FXRL"]},
               {"enum_values": []}),
        _delta("RREL996", FIELD_ADDED, None,
               {"label": "x", "xml_path": "/x", "mandatory": "mandatory",
                "nd_allowed": []}),
    ]
    findings = assess(deltas, index=index)
    assert {f.status for f in findings} <= set(IMPACT_STATUSES)
    assert all(f.rationale for f in findings)
    assert all(f.delta_id for f in findings)


def test_assessment_output_is_stable_across_runs(index):
    deltas = [_delta("RREC9", ND_PERMISSION_CHANGED, {"nd_allowed": ["ND5"]},
                     {"nd_allowed": []})]
    first = [f.to_dict() for f in assess(deltas, index=index)]
    second = [f.to_dict() for f in assess(deltas, index=index)]
    assert first == second


# --------------------------------------------------------------------------- #
# Stage 1 boundary
# --------------------------------------------------------------------------- #

CONFIG_FILES = (P_REGISTRY, P_CODE_ORDER, P_DELIVERY_RULES, P_ENUM_MAPPING,
                P_XSD_PATH_MAP, "config/regime/annex2_field_universe.yaml",
                "config/system/esma_model_structure.yaml")


def test_the_impact_pass_does_not_modify_any_active_config(index):
    """Observational only: the assessed configs are byte-identical after."""
    before = {p: hashlib.sha256((_REPO / p).read_bytes()).hexdigest()
              for p in CONFIG_FILES}
    assess([_delta("RREL1", FORMAT_CHANGED, {"format_pattern": "a"},
                   {"format_pattern": "b"}),
            _delta("RREL999", FIELD_ADDED, None,
                   {"label": "x", "xml_path": "/x", "mandatory": "mandatory",
                    "nd_allowed": []})],
           index=TraktImplementationIndex.load(_REPO))
    after = {p: hashlib.sha256((_REPO / p).read_bytes()).hexdigest()
             for p in CONFIG_FILES}
    assert before == after
