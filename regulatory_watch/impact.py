"""regulatory_watch.impact — regulatory delta -> likely Trakt impact.

Answers one question per delta:

    "If this regulatory change became effective, which existing Trakt Annex 2
     components are likely affected?"

It does **not** fix anything. Every Trakt config is opened read-only; nothing
here writes to ``config/``, creates a regime version or touches the
canonical -> regime -> validation -> XML pathway. A single regulatory delta may
produce several findings, one per affected component.

The components assessed are the ones that actually carry Annex 2 behaviour
today:

======================================  =====================================
component                               backing artefact(s)
======================================  =====================================
canonical field registry regime map     config/system/fields_registry.yaml
ESMA code ordering                      config/system/esma_code_order.yaml,
                                        config/system/esma_model_structure.yaml
ND permissions / defaults               the derived Annex 2 contract,
                                        config/regime/annex2_field_universe.yaml
enum mapping                            config/system/enum_mapping.yaml,
                                        the contract's enum vocabulary
validation rules                        the contract's XSD-derived validators
XML/XSD mapping and building            config/delivery/annex2_field_xsd_path_map.yaml
Annex 2 fixtures and tests              tests/fixtures/annex2_*, tests/test_annex2_*
======================================  =====================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Set, Tuple,
)

import yaml

from .contracts import (
    COMPONENT_CODE_ORDER,
    COMPONENT_ENUM_MAPPING,
    COMPONENT_FIELD_REGISTRY,
    COMPONENT_ND_BEHAVIOUR,
    COMPONENT_TESTS,
    COMPONENT_VALIDATION,
    COMPONENT_XML_MAPPING,
    CONFIG_CHANGE_REQUIRED,
    ENUM_CHANGED,
    FIELD_ADDED,
    FIELD_DESCRIPTION_CHANGED,
    FIELD_REMOVED,
    FORMAT_CHANGED,
    ImpactFinding,
    MANDATORY_STATUS_CHANGED,
    MANUAL_REVIEW_REQUIRED,
    MULTIPLICITY_CHANGED,
    ND_PERMISSION_CHANGED,
    NO_IMPLEMENTATION_CHANGE,
    ORDER_CHANGED,
    REVIEW_REQUIRED,
    SpecDelta,
    TEST_CHANGE_REQUIRED,
    VALIDATION_CHANGE_REQUIRED,
    VALIDATION_RULE_CHANGED,
    XML_CHANGE_REQUIRED,
    XML_PATH_CHANGED,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

REGIME = "ESMA_Annex2"

# Relative paths of the live Trakt Annex 2 configuration. Read-only, always.
P_REGISTRY = "config/system/fields_registry.yaml"
P_CODE_ORDER = "config/system/esma_code_order.yaml"
P_MODEL_STRUCTURE = "config/system/esma_model_structure.yaml"
P_DELIVERY_RULES = "engine.regime_contract.annex2_contract (derived)"
P_FIELD_UNIVERSE = "config/regime/annex2_field_universe.yaml"
P_ENUM_MAPPING = "config/system/enum_mapping.yaml"
P_XSD_PATH_MAP = "config/delivery/annex2_field_xsd_path_map.yaml"

#: Annex 2 test/fixture surface scanned for code references. Fixed and
#: explicit so the scan is deterministic rather than a repo-wide grep.
TEST_GLOBS = (
    "tests/fixtures/annex2_*.csv",
    "tests/test_annex2_*.py",
    "tests/test_annex_delivery_agent_annex2.py",
    "tests/test_onboarding_annex2_workflow.py",
    "tests/test_regime_projector_annex2_guards.py",
    "tests/test_xml_builder_annex2_shape_fixes.py",
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}


@dataclass
class TraktImplementationIndex:
    """A read-only index of what the live Trakt Annex 2 implementation holds."""

    repo_root: Path
    registry_by_code: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    code_order: List[str] = field(default_factory=list)
    model_structure_order: List[str] = field(default_factory=list)
    delivery_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    deferred_fields: List[str] = field(default_factory=list)
    representation_codes: List[str] = field(default_factory=list)
    universe: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    enum_mapping: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    xsd_path_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    test_references: Dict[str, List[str]] = field(default_factory=dict)

    # -- loading ------------------------------------------------------------ #
    @classmethod
    def load(cls, repo_root: Optional[Path] = None) -> "TraktImplementationIndex":
        root = Path(repo_root or REPO_ROOT)
        index = cls(repo_root=root)

        registry = _load_yaml(root / P_REGISTRY)
        for canonical, spec in (registry.get("fields") or {}).items():
            mapping = ((spec or {}).get("regime_mapping") or {}).get(REGIME)
            if not mapping or not mapping.get("code"):
                continue
            index.registry_by_code[str(mapping["code"])] = {
                "canonical_field": canonical,
                "priority": mapping.get("priority"),
                "format": (spec or {}).get("format"),
                "allowed_values": (spec or {}).get("allowed_values"),
                "category": (spec or {}).get("category"),
            }

        order = _load_yaml(root / P_CODE_ORDER)
        index.code_order = [str(c) for c in (order.get("Record") or [])]
        structure = _load_yaml(root / P_MODEL_STRUCTURE)
        index.model_structure_order = [str(c) for c
                                       in (structure.get("Record") or [])]

        # The Annex 2 contract is derived from the field universe, the fields
        # registry, the mapping workbook and the XSD; there is no rules file to
        # read. Impact analysis compares against the same contract the pipeline
        # uses, so a regulatory change is assessed against what actually runs.
        from engine.regime_contract.annex2_contract import as_delivery_rules
        rules = as_delivery_rules()
        index.delivery_rules = {str(k): (v or {}) for k, v
                                in (rules.get("field_rules") or {}).items()}
        scope = rules.get("reconciliation_scope") or {}
        index.deferred_fields = [str(c) for c in (scope.get("deferred_fields")
                                                  or [])]
        index.representation_codes = [str(c) for c
                                      in (scope.get("representation") or {})]

        index.universe = {str(k): (v or {}) for k, v in
                          ((_load_yaml(root / P_FIELD_UNIVERSE).get("fields"))
                           or {}).items()}

        enums = _load_yaml(root / P_ENUM_MAPPING)
        index.enum_mapping = {str(k): (v or {}) for k, v
                              in ((enums.get(REGIME)) or {}).items()}

        path_map = ((_load_yaml(root / P_XSD_PATH_MAP)
                     .get("field_xsd_path_map")) or {}).get("fields") or []
        for entry in path_map:
            code = str((entry or {}).get("esma_code") or "")
            if code:
                index.xsd_path_map[code] = entry

        index.test_references = _scan_test_references(root)
        return index

    # -- lookups ------------------------------------------------------------ #
    def canonical_field(self, code: str) -> Optional[str]:
        entry = self.registry_by_code.get(code)
        return entry["canonical_field"] if entry else None

    def enum_map_for(self, code: str) -> Dict[str, Any]:
        """Configured enum mapping for a code: delivery rule + canonical map."""
        out: Dict[str, Any] = {}
        rule = self.delivery_rules.get(code) or {}
        transform = (rule.get("transform") or {}).get("enum_map") or {}
        if transform:
            out["delivery_rule_enum_map"] = dict(transform)
        canonical = self.canonical_field(code)
        if canonical and canonical in self.enum_mapping:
            out["enum_mapping_yaml"] = dict(self.enum_mapping[canonical])
        return out

    def configured_enum_targets(self, code: str) -> Set[str]:
        targets: Set[str] = set()
        for mapping in self.enum_map_for(code).values():
            targets.update(str(v) for v in mapping.values())
        return targets


def _scan_test_references(root: Path) -> Dict[str, List[str]]:
    """Map each Annex 2 code found in the test surface to the files citing it."""
    token = re.compile(r"\b(RREL|RREC)\d+\b")
    out: Dict[str, List[str]] = {}
    for pattern in TEST_GLOBS:
        for path in sorted(root.glob(pattern)):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel = str(path.relative_to(root))
            for match in {m.group(0) for m in token.finditer(text)}:
                out.setdefault(match, [])
                if rel not in out[match]:
                    out[match].append(rel)
    return {k: sorted(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# The decision table
# --------------------------------------------------------------------------- #
#
# Impact assessment is a CONTROLLED DECISION TABLE, not procedural code. Each
# entry below is one auditable row:
#
#     regulatory delta -> Trakt component -> impact status -> rationale
#
# Two boundaries are structural rather than conventional:
#
# 1. **Factual comparison stays separate from implementation judgement.** The
#    comparator establishes WHAT the authority changed; nothing in this module
#    can create, suppress or reword a delta. This module only answers "given
#    that fact, which Trakt component is likely affected".
#
# 2. **No status authorises an automatic change.** Every status is advisory and
#    names a location for a human to inspect. ``CONFIG_CHANGE_REQUIRED`` means
#    "a person must change this config", never "the watch may".
#
# Conservatism rule applied throughout: where the evidence does not decide the
# answer, the row resolves to MANUAL_REVIEW_REQUIRED. NO_IMPLEMENTATION_CHANGE
# is reserved for cases where Trakt demonstrably holds nothing that depends on
# the changed attribute.


@dataclass(frozen=True)
class Context:
    """Everything a decision row is allowed to look at, computed once."""

    delta: SpecDelta
    code: str
    registry: Optional[Dict[str, Any]]          # fields_registry regime entry
    rule: Optional[Dict[str, Any]]              # effective contract entry
    path_entry: Optional[Dict[str, Any]]        # annex2_field_xsd_path_map
    in_order: bool                              # esma_code_order / structure
    order_position: Optional[int]
    in_universe: bool                           # annex2_field_universe
    tests: List[str]                            # concrete referencing paths
    enum_maps: Dict[str, Any]
    enum_targets: Set[str]

    @property
    def validators(self) -> Dict[str, Any]:
        return (self.rule or {}).get("validators") or {}

    @property
    def configured_nd(self) -> Set[str]:
        return set((self.rule or {}).get("nd_allowed") or [])

    def old(self, key: str, default: Any = None) -> Any:
        return (self.delta.old_value or {}).get(key, default)

    def new(self, key: str, default: Any = None) -> Any:
        return (self.delta.new_value or {}).get(key, default)


#: A decision returns (status, rationale, current_implementation), or None to
#: emit no finding for that component at all.
Decision = Optional[Tuple[str, str, Any]]


@dataclass(frozen=True)
class Row:
    """One row of the decision table."""

    component: str
    locations: Tuple[str, ...]
    decide: Callable[[Context], Decision]


def _no_change(rationale: str, current: Any = None) -> Decision:
    return NO_IMPLEMENTATION_CHANGE, rationale, current


# --------------------------------------------------------------------------- #
# FIELD_ADDED — the authority publishes a code Trakt has never seen
# --------------------------------------------------------------------------- #
# Note the conservatism: pre-existing configuration for a NEWLY PUBLISHED code
# is not reassurance. That config was written before the authority published
# the definition, so it cannot have been written against it. Those rows resolve
# to MANUAL_REVIEW_REQUIRED, never NO_IMPLEMENTATION_CHANGE.

def _added_registry(c: Context) -> Decision:
    if c.registry:
        return (MANUAL_REVIEW_REQUIRED,
                "the registry already maps this code, but the mapping predates "
                "the authority publishing it — confirm the canonical field "
                "still matches the published definition", c.registry)
    return (CONFIG_CHANGE_REQUIRED,
            "a newly published Annex 2 code needs a canonical field mapping "
            "before it can be projected", None)


def _added_order(c: Context) -> Decision:
    if c.in_order:
        return (MANUAL_REVIEW_REQUIRED,
                "the code ordering already lists this code; confirm its "
                "position matches the newly published sequence",
                {"configured_position": c.order_position})
    return (CONFIG_CHANGE_REQUIRED,
            "a new code absent from the ESMA code ordering would be emitted "
            "out of sequence or not at all", {"in_code_order": False})


def _added_xml(c: Context) -> Decision:
    if c.path_entry:
        return (MANUAL_REVIEW_REQUIRED,
                "an XSD path is already mapped for this code; confirm it "
                "matches the newly published path",
                c.path_entry.get("xml_path"))
    return (XML_CHANGE_REQUIRED,
            "a new code needs an XSD path before the XML builder can place it",
            None)


def _added_nd(c: Context) -> Decision:
    current = {"delivery_rule": bool(c.rule),
               "in_field_universe": c.in_universe,
               "regulatory_nd_allowed": c.new("nd_allowed")}
    if c.rule:
        return (MANUAL_REVIEW_REQUIRED,
                "a delivery rule already exists for this newly published code; "
                "its ND envelope was not written against the published "
                "definition and must be reconciled with it", current)
    return (CONFIG_CHANGE_REQUIRED,
            "the ND envelope for a new code is undefined until a delivery rule "
            "records it", current)


def _added_tests(c: Context) -> Decision:
    return (TEST_CHANGE_REQUIRED,
            "a newly published code has no Annex 2 coverage"
            if not c.tests else
            "Annex 2 tests already reference this newly published code; "
            "confirm they assert the published definition",
            {"existing_references": c.tests})


# --------------------------------------------------------------------------- #
# FIELD_REMOVED — the authority withdraws a code Trakt may still emit
# --------------------------------------------------------------------------- #

def _removed_registry(c: Context) -> Decision:
    if c.registry:
        return (CONFIG_CHANGE_REQUIRED,
                "the registry maps a code the authority no longer publishes",
                c.registry)
    return _no_change("the registry does not map this code")


def _removed_order(c: Context) -> Decision:
    if c.in_order:
        return (CONFIG_CHANGE_REQUIRED,
                "the code ordering still emits a withdrawn code",
                {"configured_position": c.order_position})
    return _no_change("the code ordering does not list this code")


def _removed_nd(c: Context) -> Decision:
    current = {"delivery_rule": bool(c.rule), "in_field_universe": c.in_universe}
    if c.rule or c.in_universe:
        return (CONFIG_CHANGE_REQUIRED,
                "delivery/ND configuration exists for a withdrawn code", current)
    return _no_change("no ND configuration for this code", current)


def _removed_xml(c: Context) -> Decision:
    if c.path_entry:
        return (XML_CHANGE_REQUIRED,
                "the XSD path map still targets a withdrawn code",
                c.path_entry.get("xml_path"))
    return _no_change("no XSD path mapped for this code")


def _removed_tests(c: Context) -> Decision:
    if not c.tests:
        return None
    return (TEST_CHANGE_REQUIRED,
            "Annex 2 tests/fixtures assert a withdrawn code",
            {"existing_references": c.tests})


# --------------------------------------------------------------------------- #
# ND_PERMISSION_CHANGED — the permitted ND set moved
# --------------------------------------------------------------------------- #

def _nd_lost(c: Context) -> List[str]:
    """Configured ND values the authority no longer permits."""
    return sorted(c.configured_nd - set(c.new("nd_allowed") or []))


def _nd_behaviour(c: Context) -> Decision:
    lost = _nd_lost(c)
    current = {"configured_nd_allowed": sorted(c.configured_nd),
               "in_field_universe": c.in_universe, "no_longer_permitted": lost}
    if lost:
        return (CONFIG_CHANGE_REQUIRED,
                f"configured ND value(s) {lost} are no longer permitted by the "
                f"authority", current)
    if c.rule or c.in_universe:
        return (MANUAL_REVIEW_REQUIRED,
                "the permitted ND set moved but the configured envelope stays "
                "within it; the derived field universe still needs "
                "regenerating and the widened options may be usable", current)
    return _no_change("no ND configuration exists for this code", current)


def _nd_validation(c: Context) -> Decision:
    if not _nd_lost(c):
        return None
    return (VALIDATION_CHANGE_REQUIRED,
            "regime validation would accept an ND value the authority no "
            "longer permits", (c.rule or {}).get("nd_allowed"))


def _nd_tests(c: Context) -> Decision:
    if not _nd_lost(c) or not c.tests:
        return None
    return (TEST_CHANGE_REQUIRED,
            "Annex 2 fixtures/tests pin the previous ND envelope",
            {"existing_references": c.tests})


# --------------------------------------------------------------------------- #
# ENUM_CHANGED — the authoritative code list moved
# --------------------------------------------------------------------------- #

def _enum_broken(c: Context) -> List[str]:
    """Configured enum TARGETS the authority has withdrawn."""
    withdrawn = set(c.old("enum_values") or []) - set(c.new("enum_values") or [])
    return sorted(c.enum_targets & withdrawn)


def _enum_mapping(c: Context) -> Decision:
    broken = _enum_broken(c)
    if broken:
        return (CONFIG_CHANGE_REQUIRED,
                f"configured enum target(s) {broken} were withdrawn from the "
                f"authoritative code list", c.enum_maps or None)
    if c.enum_maps:
        return (MANUAL_REVIEW_REQUIRED,
                "the authoritative code list moved; existing targets remain "
                "valid but the mapping may need extending", c.enum_maps)
    return _no_change("no enum mapping is configured for this code")


def _enum_validation(c: Context) -> Decision:
    if not _enum_broken(c):
        return None
    return (VALIDATION_CHANGE_REQUIRED,
            "projection would emit a code the schema no longer allows",
            c.enum_maps or None)


def _enum_tests(c: Context) -> Decision:
    if not _enum_broken(c) or not c.tests:
        return None
    return (TEST_CHANGE_REQUIRED,
            "Annex 2 tests/fixtures assert a withdrawn enum value",
            {"existing_references": c.tests})


# --------------------------------------------------------------------------- #
# FORMAT_CHANGED — data type / pattern / value-leaf shape moved
# --------------------------------------------------------------------------- #

def _format_validation(c: Context) -> Decision:
    if c.validators:
        return (VALIDATION_CHANGE_REQUIRED,
                "a configured format validator is pinned to the previous "
                "format", c.validators)
    if c.registry or c.rule:
        return (MANUAL_REVIEW_REQUIRED,
                "no format validator is configured; confirm the projected "
                "value still satisfies the new format", None)
    return _no_change("no delivery rule or registry mapping for this code")


def _format_registry(c: Context) -> Decision:
    if not c.registry:
        return None
    return (CONFIG_CHANGE_REQUIRED,
            "the registry declares a format for the mapped canonical field",
            c.registry)


def _format_tests(c: Context) -> Decision:
    if not c.tests:
        return None
    return (TEST_CHANGE_REQUIRED,
            "Annex 2 fixtures carry values in the previous format",
            {"existing_references": c.tests})


# --------------------------------------------------------------------------- #
# MANDATORY_STATUS_CHANGED — presence obligation moved
# --------------------------------------------------------------------------- #

def _mandatory_validation(c: Context) -> Decision:
    current = {"mandatory": (c.rule or {}).get("mandatory"),
               "enforce_presence": (c.rule or {}).get("enforce_presence")}
    if c.rule:
        return (VALIDATION_CHANGE_REQUIRED,
                "the delivery rule pins presence enforcement for this code",
                current)
    return (MANUAL_REVIEW_REQUIRED,
            "no delivery rule exists; presence enforcement for this code is "
            "undefined and must be decided", current)


def _mandatory_registry(c: Context) -> Decision:
    if not c.registry:
        return None
    return (CONFIG_CHANGE_REQUIRED,
            "the registry records a regime priority for this code", c.registry)


def _mandatory_tests(c: Context) -> Decision:
    if not c.tests:
        return None
    return (TEST_CHANGE_REQUIRED,
            "Annex 2 tests assert the previous mandatory status",
            {"existing_references": c.tests})


# --------------------------------------------------------------------------- #
# XML_PATH_CHANGED — the element moved in the message tree
# --------------------------------------------------------------------------- #

def _path_xml(c: Context) -> Decision:
    if c.path_entry:
        return (XML_CHANGE_REQUIRED,
                "the mapped XSD path no longer matches the authoritative path",
                c.path_entry.get("xml_path"))
    return (MANUAL_REVIEW_REQUIRED,
            "no XSD path is mapped for this code; confirm the builder is "
            "unaffected", None)


def _path_delivery_rule(c: Context) -> Decision:
    semantic = (c.rule or {}).get("workbook_semantic")
    if not semantic:
        return None
    return (CONFIG_CHANGE_REQUIRED,
            "the delivery rule pins a workbook leaf token derived from the "
            "previous path", semantic)


def _path_tests(c: Context) -> Decision:
    if not c.tests:
        return None
    return (TEST_CHANGE_REQUIRED,
            "Annex 2 XML tests assert the previous element path",
            {"existing_references": c.tests})


# --------------------------------------------------------------------------- #
# MULTIPLICITY_CHANGED — cardinality moved
# --------------------------------------------------------------------------- #

def _multiplicity_xml(c: Context) -> Decision:
    if c.path_entry:
        return (XML_CHANGE_REQUIRED,
                "the path map records a cardinality for this code; a repeating "
                "or newly optional element changes how the builder emits it",
                c.path_entry.get("cardinality"))
    return (MANUAL_REVIEW_REQUIRED,
            "cardinality moved for a code with no mapped path; confirm the "
            "builder is unaffected", None)


def _multiplicity_tests(c: Context) -> Decision:
    if not c.tests:
        return None
    return (TEST_CHANGE_REQUIRED,
            "Annex 2 fixtures encode one occurrence per record and would not "
            "exercise the new cardinality", {"existing_references": c.tests})


# --------------------------------------------------------------------------- #
# ORDER_CHANGED — the authoritative field sequence moved
# --------------------------------------------------------------------------- #

def _order_config(c: Context) -> Decision:
    current = {"in_code_order": c.in_order,
               "configured_position": c.order_position}
    if c.in_order:
        return (CONFIG_CHANGE_REQUIRED,
                "the configured ESMA code ordering no longer matches the "
                "authoritative sequence; XSD sequence validation would reject "
                "it", current)
    # A code Trakt does not emit cannot mis-order Trakt's output: the relative
    # order of the codes that ARE configured is unaffected by it moving.
    return _no_change("this code is not in the configured ordering, so its "
                      "position cannot affect the emitted sequence", current)


def _order_xml(c: Context) -> Decision:
    if not c.path_entry:
        return None
    return (XML_CHANGE_REQUIRED,
            "element sequence is driven by the code ordering at build time",
            c.path_entry.get("sequence_order"))


# --------------------------------------------------------------------------- #
# VALIDATION_RULE_CHANGED — the published rule text moved
# --------------------------------------------------------------------------- #

def _validation_rule(c: Context) -> Decision:
    if c.validators:
        return (VALIDATION_CHANGE_REQUIRED,
                "a configured validator implements the previous rule text",
                c.validators)
    if c.rule or c.registry:
        return (MANUAL_REVIEW_REQUIRED,
                "no validator is configured for this code; confirm whether the "
                "changed rule needs implementing", None)
    return _no_change("no delivery rule or registry mapping for this code")


# --------------------------------------------------------------------------- #
# FIELD_DESCRIPTION_CHANGED — wording only
# --------------------------------------------------------------------------- #

def _description_only(c: Context) -> Decision:
    return _no_change(
        "wording only: no projected value, ND behaviour, enum, path, "
        "cardinality or validator depends on the description text (the "
        "derived field universe carries it for reference)", c.registry)


# --------------------------------------------------------------------------- #
# The table itself
# --------------------------------------------------------------------------- #

IMPACT_RULES: Dict[str, Tuple[Row, ...]] = {
    FIELD_ADDED: (
        Row(COMPONENT_FIELD_REGISTRY, (P_REGISTRY,), _added_registry),
        Row(COMPONENT_CODE_ORDER, (P_CODE_ORDER, P_MODEL_STRUCTURE),
            _added_order),
        Row(COMPONENT_XML_MAPPING, (P_XSD_PATH_MAP,), _added_xml),
        Row(COMPONENT_ND_BEHAVIOUR, (P_DELIVERY_RULES, P_FIELD_UNIVERSE),
            _added_nd),
        Row(COMPONENT_TESTS, (), _added_tests),
    ),
    FIELD_REMOVED: (
        Row(COMPONENT_FIELD_REGISTRY, (P_REGISTRY,), _removed_registry),
        Row(COMPONENT_CODE_ORDER, (P_CODE_ORDER, P_MODEL_STRUCTURE),
            _removed_order),
        Row(COMPONENT_ND_BEHAVIOUR, (P_DELIVERY_RULES, P_FIELD_UNIVERSE),
            _removed_nd),
        Row(COMPONENT_XML_MAPPING, (P_XSD_PATH_MAP,), _removed_xml),
        Row(COMPONENT_TESTS, (), _removed_tests),
    ),
    ND_PERMISSION_CHANGED: (
        Row(COMPONENT_ND_BEHAVIOUR, (P_DELIVERY_RULES, P_FIELD_UNIVERSE),
            _nd_behaviour),
        Row(COMPONENT_VALIDATION, (P_DELIVERY_RULES,), _nd_validation),
        Row(COMPONENT_TESTS, (), _nd_tests),
    ),
    ENUM_CHANGED: (
        Row(COMPONENT_ENUM_MAPPING, (P_ENUM_MAPPING, P_DELIVERY_RULES),
            _enum_mapping),
        Row(COMPONENT_VALIDATION, (P_DELIVERY_RULES,), _enum_validation),
        Row(COMPONENT_TESTS, (), _enum_tests),
    ),
    FORMAT_CHANGED: (
        Row(COMPONENT_VALIDATION, (P_DELIVERY_RULES,), _format_validation),
        Row(COMPONENT_FIELD_REGISTRY, (P_REGISTRY,), _format_registry),
        Row(COMPONENT_TESTS, (), _format_tests),
    ),
    MANDATORY_STATUS_CHANGED: (
        Row(COMPONENT_VALIDATION, (P_DELIVERY_RULES,), _mandatory_validation),
        Row(COMPONENT_FIELD_REGISTRY, (P_REGISTRY,), _mandatory_registry),
        Row(COMPONENT_TESTS, (), _mandatory_tests),
    ),
    XML_PATH_CHANGED: (
        Row(COMPONENT_XML_MAPPING, (P_XSD_PATH_MAP,), _path_xml),
        Row(COMPONENT_ND_BEHAVIOUR, (P_DELIVERY_RULES,), _path_delivery_rule),
        Row(COMPONENT_TESTS, (), _path_tests),
    ),
    MULTIPLICITY_CHANGED: (
        Row(COMPONENT_XML_MAPPING, (P_XSD_PATH_MAP,), _multiplicity_xml),
        Row(COMPONENT_TESTS, (), _multiplicity_tests),
    ),
    ORDER_CHANGED: (
        Row(COMPONENT_CODE_ORDER, (P_CODE_ORDER, P_MODEL_STRUCTURE),
            _order_config),
        Row(COMPONENT_XML_MAPPING, (P_XSD_PATH_MAP,), _order_xml),
    ),
    VALIDATION_RULE_CHANGED: (
        Row(COMPONENT_VALIDATION, (P_DELIVERY_RULES,), _validation_rule),
    ),
    FIELD_DESCRIPTION_CHANGED: (
        Row(COMPONENT_FIELD_REGISTRY, (P_FIELD_UNIVERSE,), _description_only),
    ),
}


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def _context(delta: SpecDelta, index: TraktImplementationIndex) -> Context:
    code = delta.code
    return Context(
        delta=delta,
        code=code,
        registry=index.registry_by_code.get(code),
        rule=index.delivery_rules.get(code),
        path_entry=index.xsd_path_map.get(code),
        in_order=(code in index.code_order
                  or code in index.model_structure_order),
        order_position=(index.code_order.index(code)
                        if code in index.code_order else None),
        in_universe=code in index.universe,
        tests=index.test_references.get(code, []),
        enum_maps=index.enum_map_for(code),
        enum_targets=index.configured_enum_targets(code),
    )


def _assess_one(delta: SpecDelta, index: TraktImplementationIndex
                ) -> List[ImpactFinding]:
    rows = IMPACT_RULES.get(delta.change_type)
    if rows is None:
        # An unmapped change type is never silently ignored.
        return [ImpactFinding(
            delta_id=delta.delta_id, code=delta.code,
            change_type=delta.change_type, component=COMPONENT_FIELD_REGISTRY,
            status=MANUAL_REVIEW_REQUIRED, locations=[],
            current_implementation=None,
            rationale=(f"no impact rule is defined for change type "
                       f"'{delta.change_type}'"))]

    context = _context(delta, index)
    out: List[ImpactFinding] = []
    for row in rows:
        decision = row.decide(context)
        if decision is None:
            continue
        status, rationale, current = decision
        locations = list(row.locations)
        if row.component == COMPONENT_TESTS:
            # Concrete referencing paths only; never glob patterns.
            locations = list(context.tests)
        out.append(ImpactFinding(
            delta_id=delta.delta_id, code=delta.code,
            change_type=delta.change_type, component=row.component,
            status=status, locations=locations,
            current_implementation=current, rationale=rationale))
    return out


def assess(deltas: Sequence[SpecDelta],
           index: Optional[TraktImplementationIndex] = None,
           repo_root: Optional[Path] = None) -> List[ImpactFinding]:
    """Map every delta onto the Trakt components it likely affects.

    Read-only and advisory. No status this returns authorises the watch to
    change anything; each names a location for a human to inspect.
    """
    idx = index or TraktImplementationIndex.load(repo_root)
    out: List[ImpactFinding] = []
    for delta in deltas:
        findings = _assess_one(delta, idx)
        if delta.confidence == REVIEW_REQUIRED:
            findings.append(ImpactFinding(
                delta_id=delta.delta_id, code=delta.code,
                change_type=delta.change_type,
                component=COMPONENT_FIELD_REGISTRY,
                status=MANUAL_REVIEW_REQUIRED, locations=[],
                current_implementation=None,
                rationale="the regulatory delta itself is review-required: the "
                          "parser could not determine the changed attribute "
                          "from the authoritative artefacts"))
        out.extend(findings)
    out.sort(key=lambda f: (f.code, f.change_type, f.component, f.status))
    return out


def decision_table() -> List[Dict[str, Any]]:
    """The table as data, for documentation and for the report's evidence."""
    return [
        {"change_type": change_type,
         "component": row.component,
         "locations": list(row.locations),
         "decision": row.decide.__name__}
        for change_type in sorted(IMPACT_RULES)
        for row in IMPACT_RULES[change_type]
    ]
