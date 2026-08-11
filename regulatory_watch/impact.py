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
ND permissions / defaults               config/regime/annex2_delivery_rules.yaml,
                                        config/regime/annex2_field_universe.yaml
enum mapping                            config/system/enum_mapping.yaml,
                                        annex2_delivery_rules.yaml transforms
validation rules                        annex2_delivery_rules.yaml validators
XML/XSD mapping and building            config/delivery/annex2_field_xsd_path_map.yaml
Annex 2 fixtures and tests              tests/fixtures/annex2_*, tests/test_annex2_*
======================================  =====================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

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
P_DELIVERY_RULES = "config/regime/annex2_delivery_rules.yaml"
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

        rules = _load_yaml(root / P_DELIVERY_RULES)
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
# Per-change-type assessment
# --------------------------------------------------------------------------- #

def _finding(delta: SpecDelta, component: str, status: str,
             locations: Sequence[str], current: Any, rationale: str
             ) -> ImpactFinding:
    return ImpactFinding(
        delta_id=delta.delta_id, code=delta.code,
        change_type=delta.change_type, component=component, status=status,
        locations=list(locations), current_implementation=current,
        rationale=rationale)


def _test_locations(index: TraktImplementationIndex, code: str) -> List[str]:
    return index.test_references.get(code, [])


def _assess_one(delta: SpecDelta, index: TraktImplementationIndex
                ) -> List[ImpactFinding]:
    code = delta.code
    out: List[ImpactFinding] = []

    registry = index.registry_by_code.get(code)
    rule = index.delivery_rules.get(code)
    path_entry = index.xsd_path_map.get(code)
    in_order = code in index.code_order or code in index.model_structure_order
    in_universe = code in index.universe
    tests = _test_locations(index, code)

    ct = delta.change_type

    # ------------------------------------------------------------------ #
    if ct == FIELD_ADDED:
        out.append(_finding(
            delta, COMPONENT_FIELD_REGISTRY,
            NO_IMPLEMENTATION_CHANGE if registry else CONFIG_CHANGE_REQUIRED,
            [P_REGISTRY], registry,
            "a newly published Annex 2 code needs a canonical field mapping "
            "before it can be projected"
            if not registry else
            "the registry already maps this code to a canonical field"))
        out.append(_finding(
            delta, COMPONENT_CODE_ORDER,
            NO_IMPLEMENTATION_CHANGE if in_order else CONFIG_CHANGE_REQUIRED,
            [P_CODE_ORDER, P_MODEL_STRUCTURE],
            {"in_code_order": in_order},
            "a new code absent from the ESMA code ordering would be emitted "
            "out of sequence or not at all"))
        out.append(_finding(
            delta, COMPONENT_XML_MAPPING,
            NO_IMPLEMENTATION_CHANGE if path_entry else XML_CHANGE_REQUIRED,
            [P_XSD_PATH_MAP], (path_entry or {}).get("xml_path"),
            "a new code needs an XSD path before the XML builder can place it"))
        new_nd = (delta.new_value or {}).get("nd_allowed")
        out.append(_finding(
            delta, COMPONENT_ND_BEHAVIOUR,
            NO_IMPLEMENTATION_CHANGE if rule else CONFIG_CHANGE_REQUIRED,
            [P_DELIVERY_RULES, P_FIELD_UNIVERSE],
            {"delivery_rule": bool(rule), "in_field_universe": in_universe,
             "regulatory_nd_allowed": new_nd},
            "the ND envelope for a new code is undefined until a delivery rule "
            "records it"))
        out.append(_finding(
            delta, COMPONENT_TESTS, TEST_CHANGE_REQUIRED, tests or list(TEST_GLOBS),
            {"existing_references": tests},
            "a newly published code has no Annex 2 coverage"))
        return out

    # ------------------------------------------------------------------ #
    if ct == FIELD_REMOVED:
        out.append(_finding(
            delta, COMPONENT_FIELD_REGISTRY,
            CONFIG_CHANGE_REQUIRED if registry else NO_IMPLEMENTATION_CHANGE,
            [P_REGISTRY], registry,
            "the registry maps a code the authority no longer publishes"
            if registry else "the registry does not map this code"))
        out.append(_finding(
            delta, COMPONENT_CODE_ORDER,
            CONFIG_CHANGE_REQUIRED if in_order else NO_IMPLEMENTATION_CHANGE,
            [P_CODE_ORDER, P_MODEL_STRUCTURE], {"in_code_order": in_order},
            "the code ordering still emits a withdrawn code"
            if in_order else "the code ordering does not list this code"))
        out.append(_finding(
            delta, COMPONENT_ND_BEHAVIOUR,
            CONFIG_CHANGE_REQUIRED if (rule or in_universe)
            else NO_IMPLEMENTATION_CHANGE,
            [P_DELIVERY_RULES, P_FIELD_UNIVERSE],
            {"delivery_rule": bool(rule), "in_field_universe": in_universe},
            "delivery/ND configuration exists for a withdrawn code"
            if (rule or in_universe) else "no ND configuration for this code"))
        out.append(_finding(
            delta, COMPONENT_XML_MAPPING,
            XML_CHANGE_REQUIRED if path_entry else NO_IMPLEMENTATION_CHANGE,
            [P_XSD_PATH_MAP], (path_entry or {}).get("xml_path"),
            "the XSD path map still targets a withdrawn code"
            if path_entry else "no XSD path mapped for this code"))
        if tests:
            out.append(_finding(
                delta, COMPONENT_TESTS, TEST_CHANGE_REQUIRED, tests,
                {"existing_references": tests},
                "Annex 2 tests/fixtures assert a withdrawn code"))
        return out

    # ------------------------------------------------------------------ #
    if ct == ND_PERMISSION_CHANGED:
        new_nd = set((delta.new_value or {}).get("nd_allowed") or [])
        configured = set((rule or {}).get("nd_allowed") or [])
        now_impermissible = sorted(configured - new_nd)
        status = (CONFIG_CHANGE_REQUIRED if now_impermissible
                  else (MANUAL_REVIEW_REQUIRED if (rule or in_universe)
                        else NO_IMPLEMENTATION_CHANGE))
        out.append(_finding(
            delta, COMPONENT_ND_BEHAVIOUR, status,
            [P_DELIVERY_RULES, P_FIELD_UNIVERSE],
            {"configured_nd_allowed": sorted(configured),
             "in_field_universe": in_universe,
             "no_longer_permitted": now_impermissible},
            (f"configured ND value(s) {now_impermissible} are no longer "
             f"permitted by the authority") if now_impermissible else
            ("the permitted ND set moved but the configured envelope stays "
             "within it; the derived field universe still needs regenerating")))
        if now_impermissible:
            out.append(_finding(
                delta, COMPONENT_VALIDATION, VALIDATION_CHANGE_REQUIRED,
                [P_DELIVERY_RULES], (rule or {}).get("nd_allowed"),
                "regime validation would accept an ND value the authority "
                "no longer permits"))
            if tests:
                out.append(_finding(
                    delta, COMPONENT_TESTS, TEST_CHANGE_REQUIRED, tests,
                    {"existing_references": tests},
                    "Annex 2 fixtures/tests pin the previous ND envelope"))
        return out

    # ------------------------------------------------------------------ #
    if ct == ENUM_CHANGED:
        old_values = set((delta.old_value or {}).get("enum_values") or [])
        new_values = set((delta.new_value or {}).get("enum_values") or [])
        removed = old_values - new_values
        configured_targets = index.configured_enum_targets(code)
        broken = sorted(configured_targets & removed)
        maps = index.enum_map_for(code)
        if broken:
            status = CONFIG_CHANGE_REQUIRED
            rationale = (f"configured enum target(s) {broken} were withdrawn "
                         f"from the authoritative code list")
        elif maps:
            status = MANUAL_REVIEW_REQUIRED
            rationale = ("the authoritative code list moved; existing targets "
                         "remain valid but the mapping may need extending")
        else:
            status = NO_IMPLEMENTATION_CHANGE
            rationale = "no enum mapping is configured for this code"
        out.append(_finding(delta, COMPONENT_ENUM_MAPPING, status,
                            [P_ENUM_MAPPING, P_DELIVERY_RULES], maps or None,
                            rationale))
        if broken:
            out.append(_finding(
                delta, COMPONENT_VALIDATION, VALIDATION_CHANGE_REQUIRED,
                [P_DELIVERY_RULES], maps or None,
                "projection would emit a code the schema no longer allows"))
            if tests:
                out.append(_finding(
                    delta, COMPONENT_TESTS, TEST_CHANGE_REQUIRED, tests,
                    {"existing_references": tests},
                    "Annex 2 tests/fixtures assert a withdrawn enum value"))
        return out

    # ------------------------------------------------------------------ #
    if ct == FORMAT_CHANGED:
        validators = (rule or {}).get("validators") or {}
        out.append(_finding(
            delta, COMPONENT_VALIDATION,
            VALIDATION_CHANGE_REQUIRED if validators else
            (MANUAL_REVIEW_REQUIRED if registry else NO_IMPLEMENTATION_CHANGE),
            [P_DELIVERY_RULES], validators or None,
            "a configured format validator is pinned to the previous format"
            if validators else
            "no format validator is configured; confirm the projected value "
            "still satisfies the new format"))
        if registry:
            out.append(_finding(
                delta, COMPONENT_FIELD_REGISTRY, CONFIG_CHANGE_REQUIRED,
                [P_REGISTRY], registry,
                "the registry declares a format for the mapped canonical field"))
        if tests:
            out.append(_finding(
                delta, COMPONENT_TESTS, TEST_CHANGE_REQUIRED, tests,
                {"existing_references": tests},
                "Annex 2 fixtures carry values in the previous format"))
        return out

    # ------------------------------------------------------------------ #
    if ct == MANDATORY_STATUS_CHANGED:
        out.append(_finding(
            delta, COMPONENT_VALIDATION,
            VALIDATION_CHANGE_REQUIRED if rule else MANUAL_REVIEW_REQUIRED,
            [P_DELIVERY_RULES],
            {"mandatory": (rule or {}).get("mandatory"),
             "enforce_presence": (rule or {}).get("enforce_presence")},
            "the delivery rule pins presence enforcement for this code"
            if rule else
            "no delivery rule exists; presence enforcement is undefined"))
        if registry:
            out.append(_finding(
                delta, COMPONENT_FIELD_REGISTRY, CONFIG_CHANGE_REQUIRED,
                [P_REGISTRY], registry,
                "the registry records a regime priority for this code"))
        if tests:
            out.append(_finding(
                delta, COMPONENT_TESTS, TEST_CHANGE_REQUIRED, tests,
                {"existing_references": tests},
                "Annex 2 tests assert the previous mandatory status"))
        return out

    # ------------------------------------------------------------------ #
    if ct == XML_PATH_CHANGED:
        out.append(_finding(
            delta, COMPONENT_XML_MAPPING,
            XML_CHANGE_REQUIRED if path_entry else MANUAL_REVIEW_REQUIRED,
            [P_XSD_PATH_MAP], (path_entry or {}).get("xml_path"),
            "the mapped XSD path no longer matches the authoritative path"
            if path_entry else
            "no XSD path is mapped for this code; confirm the builder is "
            "unaffected"))
        if (rule or {}).get("workbook_semantic"):
            out.append(_finding(
                delta, COMPONENT_ND_BEHAVIOUR, CONFIG_CHANGE_REQUIRED,
                [P_DELIVERY_RULES], rule.get("workbook_semantic"),
                "the delivery rule pins a workbook leaf token derived from the "
                "previous path"))
        if tests:
            out.append(_finding(
                delta, COMPONENT_TESTS, TEST_CHANGE_REQUIRED, tests,
                {"existing_references": tests},
                "Annex 2 XML tests assert the previous element path"))
        return out

    # ------------------------------------------------------------------ #
    if ct == MULTIPLICITY_CHANGED:
        out.append(_finding(
            delta, COMPONENT_XML_MAPPING,
            XML_CHANGE_REQUIRED if path_entry else MANUAL_REVIEW_REQUIRED,
            [P_XSD_PATH_MAP], (path_entry or {}).get("cardinality"),
            "the path map records a cardinality for this code"
            if path_entry else
            "cardinality moved for a code with no mapped path; confirm the "
            "builder is unaffected"))
        return out

    # ------------------------------------------------------------------ #
    if ct == ORDER_CHANGED:
        out.append(_finding(
            delta, COMPONENT_CODE_ORDER,
            CONFIG_CHANGE_REQUIRED if in_order else NO_IMPLEMENTATION_CHANGE,
            [P_CODE_ORDER, P_MODEL_STRUCTURE],
            {"in_code_order": in_order,
             "configured_position": (index.code_order.index(code)
                                     if code in index.code_order else None)},
            "the configured ESMA code ordering no longer matches the "
            "authoritative sequence; XSD sequence validation would reject it"
            if in_order else "this code is not in the configured ordering"))
        if path_entry:
            out.append(_finding(
                delta, COMPONENT_XML_MAPPING, XML_CHANGE_REQUIRED,
                [P_XSD_PATH_MAP], (path_entry or {}).get("sequence_order"),
                "element sequence is driven by the code ordering at build time"))
        return out

    # ------------------------------------------------------------------ #
    if ct == VALIDATION_RULE_CHANGED:
        validators = (rule or {}).get("validators") or {}
        out.append(_finding(
            delta, COMPONENT_VALIDATION,
            VALIDATION_CHANGE_REQUIRED if validators else
            (MANUAL_REVIEW_REQUIRED if (rule or registry)
             else NO_IMPLEMENTATION_CHANGE),
            [P_DELIVERY_RULES], validators or None,
            "a configured validator implements the previous rule text"
            if validators else
            "no validator is configured for this code; confirm whether the "
            "changed rule needs implementing"))
        return out

    # ------------------------------------------------------------------ #
    if ct == FIELD_DESCRIPTION_CHANGED:
        out.append(_finding(
            delta, COMPONENT_FIELD_REGISTRY, NO_IMPLEMENTATION_CHANGE,
            [P_FIELD_UNIVERSE], registry,
            "wording only: no projected value, ND behaviour, enum, path or "
            "validator depends on the description text (the derived field "
            "universe carries it for reference)"))
        return out

    # Unknown change type: never silently ignored.
    out.append(_finding(
        delta, COMPONENT_FIELD_REGISTRY, MANUAL_REVIEW_REQUIRED, [],
        None, f"no impact rule is defined for change type '{ct}'"))
    return out


def assess(deltas: Sequence[SpecDelta],
           index: Optional[TraktImplementationIndex] = None,
           repo_root: Optional[Path] = None) -> List[ImpactFinding]:
    """Map every delta onto the Trakt components it likely affects."""
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
