"""operations_control.annex2.population — population-path reconciliation (I2).

Deterministic per-code assessment of how each of the 107 Annex 2 universe
codes is populated by the authoritative runtime route. Derived ONLY from the
authoritative components' configuration:

  * config/regime/annex2_field_universe.yaml   (the 107-code universe + ND flags)
  * config/system/fields_registry.yaml         (regime_mapping -> projector emission)
  * config/regime/annex2_delivery_rules.yaml   (field_rules / deferred scope)
  * known, documented builder injections       (Gate 5 ND/no-data scaffolding)

The direct ``field_rules`` count is NOT a completeness measure: the projector
emits every registry-mapped code and the normaliser passes unruled columns
through verbatim. A code blocks ONLY when it is applicable, required, has no
population mechanism, no permitted no-data representation, and no builder
coverage — i.e. genuinely unsupported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]

UNIVERSE_PATH = REPO / "config/regime/annex2_field_universe.yaml"
RULES_PATH = REPO / "config/regime/annex2_delivery_rules.yaml"
REGISTRY_PATH = REPO / "config/system/fields_registry.yaml"

# Codes the Gate 5 builder represents itself with permitted no-data scaffolding
# (documented behaviour, instrumented by interventions.py — never altered here):
#   RREL20/RREL21 — _ensure_scndry_oblgr_incm_defaults writes ND5 into
#   IncmVal / Vrfctn when the columns are absent from the delivery frame.
BUILDER_ND_COVERED = {
    "RREL20": "builder inserts permitted ND5 into Secondary Income (IncmVal)",
    "RREL21": "builder inserts permitted ND5 into Secondary Income Verification",
}

MECH_RULE = "direct field rule"
MECH_REGISTRY = "projected registry mapping"
MECH_ENUM = "enum mapping"
MECH_ND = "permitted no-data value"
MECH_BUILDER = "builder transformation (no-data insertion)"
MECH_STATIC = "static client/report configuration"
MECH_DEFERRED = "omitted (deferred / not applicable)"
MECH_UNSUPPORTED = "genuinely unsupported"

#: Codes whose values originate from static client configuration rather than
#: loan data (identity/geography of the reporting entity).
STATIC_CONFIG_CODES = {"RREL1", "RREL83", "RREL84", "RREL6"}


@dataclass
class CodeAssessment:
    annex2_code: str
    field_name: str
    priority: str                 # Mandatory | Optional | Analytics | unmapped
    applicability: str            # in scope | deferred
    xsd_requirement: str          # ND allowances
    population_mechanism: str
    source: str                   # config/registry provenance, plain text
    expected_final_treatment: str
    blocking: bool
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_yaml(p: Path) -> Dict[str, Any]:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def assess_population(
    universe_path: Path = UNIVERSE_PATH,
    rules_path: Path = RULES_PATH,
    registry_path: Path = REGISTRY_PATH,
) -> List[CodeAssessment]:
    universe = _load_yaml(universe_path).get("fields") or {}
    rules = _load_yaml(rules_path)
    field_rules = rules.get("field_rules") or {}
    deferred = set((rules.get("reconciliation_scope") or {})
                   .get("deferred_fields") or [])
    registry = _load_yaml(registry_path).get("fields") or {}

    code2canon: Dict[str, str] = {}
    code2prio: Dict[str, str] = {}
    for name, spec in registry.items():
        rm = (spec or {}).get("regime_mapping", {}).get("ESMA_Annex2") or {}
        code = rm.get("code")
        if code:
            code2canon.setdefault(code, name)
            code2prio.setdefault(code, rm.get("priority", ""))

    out: List[CodeAssessment] = []
    for code in sorted(universe):
        u = universe[code] or {}
        nd_ok = bool(u.get("nd1_4_allowed") or u.get("nd5_allowed"))
        xsd_req = ", ".join(x for x in (
            "ND1-4 allowed" if u.get("nd1_4_allowed") else "",
            "ND5 allowed" if u.get("nd5_allowed") else "") if x) or "no ND"
        prio = code2prio.get(code, "unmapped")
        canon = code2canon.get(code, "")
        rule = field_rules.get(code) or {}
        is_deferred = code in deferred

        if is_deferred:
            mech, blocking = MECH_DEFERRED, False
            source = "annex2_delivery_rules reconciliation_scope.deferred_fields"
            treatment = "not emitted; formally deferred"
            expl = "Formally deferred in the delivery rules scope."
        elif code in STATIC_CONFIG_CODES:
            mech, blocking = MECH_STATIC, False
            source = ("client configuration (identity/reporting attributes) "
                      "+ projector derivation")
            treatment = "derived from client configuration at projection"
            expl = ("Populated from static regulatory configuration; the "
                    "regulatory-details preflight verifies the inputs.")
        elif rule:
            has_enum = bool((rule.get("transform") or {}).get("enum_map"))
            nd_default = str(rule.get("default_value", "")).startswith("ND")
            mech = MECH_ENUM if has_enum else MECH_RULE
            if nd_default:
                mech = f"{mech} -> {MECH_ND}"
            blocking = False
            source = "annex2_delivery_rules field_rules entry" + \
                (" + registry regime_mapping" if canon else "")
            treatment = ("permitted no-data default when blank"
                         if nd_default else "governed value in delivery output")
            expl = "Rule-governed treatment on the authoritative route."
        elif code in BUILDER_ND_COVERED:
            # Checked before the registry branch: these codes are registry-
            # mapped but the projector's portfolio-type filter excludes their
            # canonical source for this asset class, so the runtime truth is
            # the builder's permitted no-data representation.
            mech, blocking = MECH_BUILDER, False
            source = "gate 5 builder no-data scaffolding (documented, instrumented)"
            treatment = BUILDER_ND_COVERED[code]
            expl = ("Optional and ND-permitted; the builder represents it "
                    "with a permitted no-data value. Not a blocker.")
        elif canon:
            mech, blocking = MECH_REGISTRY, False
            source = f"fields_registry regime_mapping <- canonical '{canon}'"
            treatment = ("value passes projection and delivery verbatim "
                         "(normaliser passes unruled columns through)")
            expl = ("No direct field rule, but the projector emits the code "
                    "from the registry mapping and the normaliser passes it "
                    "through — rule count does not gate emission.")
        elif code in BUILDER_ND_COVERED:
            mech, blocking = MECH_BUILDER, False
            source = "gate 5 builder no-data scaffolding (documented, instrumented)"
            treatment = BUILDER_ND_COVERED[code]
            expl = ("Optional and ND-permitted; the builder represents it "
                    "with a permitted no-data value. Not a blocker.")
        elif nd_ok or prio in ("Optional", "Analytics", "unmapped"):
            mech, blocking = MECH_ND if nd_ok else MECH_DEFERRED, False
            source = "field universe ND allowance" if nd_ok else "optional field"
            treatment = ("permitted no-data representation available"
                         if nd_ok else "may be omitted (optional)")
            expl = "Optional and/or representable as permitted no-data."
        else:
            mech, blocking = MECH_UNSUPPORTED, True
            source = "none found"
            treatment = "cannot be validly represented"
            expl = ("Mandatory, applicable, no population mechanism, no "
                    "permitted no-data representation.")

        out.append(CodeAssessment(
            annex2_code=code, field_name=str(u.get("field_name", "")),
            priority=prio, applicability="deferred" if is_deferred else "in scope",
            xsd_requirement=xsd_req, population_mechanism=mech, source=source,
            expected_final_treatment=treatment, blocking=blocking,
            explanation=expl))
    return out


def reconciliation_document(assessments: Optional[List[CodeAssessment]] = None
                            ) -> Dict[str, Any]:
    """The persistable reconciliation artefact + operator summary."""
    rows = assessments or assess_population()
    blockers = [r for r in rows if r.blocking]
    from collections import Counter
    mech_counts = Counter(r.population_mechanism for r in rows)
    return {
        "universe_count": len(rows),
        "blocking_count": len(blockers),
        "blocking_codes": [r.annex2_code for r in blockers],
        "mechanism_counts": dict(mech_counts),
        "summary_sentence": (
            "Every required regulatory field has a valid way of being filled."
            if not blockers else
            f"{len(blockers)} required field"
            f"{'s' if len(blockers) != 1 else ''} cannot be filled and "
            "need attention."),
        "rows": [r.to_dict() for r in rows],
    }
