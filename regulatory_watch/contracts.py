"""regulatory_watch.contracts — the Stage 1 data contract.

Pure data + controlled vocabularies. No I/O, no repo paths, no pandas, no
network. Everything here is JSON-round-trippable so the Stage 2 OCC Regulatory
Config view can consume it without importing any of the parsing machinery.

Two rules are enforced structurally rather than by convention:

1. **A failed source check is never "no change".** ``SOURCE_CHECK_FAILED`` is a
   first-class source status and :func:`overall_status` refuses to report a
   clean run when it is present.
2. **An unparsed attribute is never a guess.** A field records the attributes it
   could not derive in :attr:`SpecField.unresolved`; the comparator downgrades
   any delta touching one of them to ``REVIEW_REQUIRED``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #

# -- source-level status (did the authoritative bytes change?) -------------- #
SOURCE_UNCHANGED = "SOURCE_UNCHANGED"
SOURCE_CHANGED = "SOURCE_CHANGED"
SOURCE_CHECK_FAILED = "SOURCE_CHECK_FAILED"
SOURCE_MISSING = "SOURCE_MISSING"

SOURCE_STATUSES = (SOURCE_UNCHANGED, SOURCE_CHANGED, SOURCE_CHECK_FAILED,
                   SOURCE_MISSING)

# -- specification-level status (did the *requirement* change?) ------------- #
SPEC_UNCHANGED = "SPEC_UNCHANGED"
SPEC_CHANGED = "SPEC_CHANGED"
SPEC_UNDETERMINED = "SPEC_UNDETERMINED"

SPEC_STATUSES = (SPEC_UNCHANGED, SPEC_CHANGED, SPEC_UNDETERMINED)

# -- the three outcomes Stage 1 must be able to tell apart ------------------ #
#: bytes differ, parsed Annex 2 requirements identical
SOURCE_CHANGED_SPEC_UNCHANGED = "SOURCE_CHANGED_SPEC_UNCHANGED"
#: parsed Annex 2 requirements differ
REGULATORY_SPEC_CHANGED = "REGULATORY_SPEC_CHANGED"
#: the parser could not establish whether a material change occurred
SPEC_CHANGE_UNDETERMINED = "SPEC_CHANGE_UNDETERMINED"
#: nothing changed anywhere
NO_CHANGE_DETECTED = "NO_CHANGE_DETECTED"
#: at least one source could not be checked — explicitly not "no change"
WATCH_INCONCLUSIVE = "WATCH_INCONCLUSIVE"

# -- delta change types ----------------------------------------------------- #
FIELD_ADDED = "FIELD_ADDED"
FIELD_REMOVED = "FIELD_REMOVED"
FIELD_DESCRIPTION_CHANGED = "FIELD_DESCRIPTION_CHANGED"
FORMAT_CHANGED = "FORMAT_CHANGED"
MANDATORY_STATUS_CHANGED = "MANDATORY_STATUS_CHANGED"
ND_PERMISSION_CHANGED = "ND_PERMISSION_CHANGED"
ENUM_CHANGED = "ENUM_CHANGED"
XML_PATH_CHANGED = "XML_PATH_CHANGED"
MULTIPLICITY_CHANGED = "MULTIPLICITY_CHANGED"
ORDER_CHANGED = "ORDER_CHANGED"
VALIDATION_RULE_CHANGED = "VALIDATION_RULE_CHANGED"

CHANGE_TYPES = (
    FIELD_ADDED, FIELD_REMOVED, FIELD_DESCRIPTION_CHANGED, FORMAT_CHANGED,
    MANDATORY_STATUS_CHANGED, ND_PERMISSION_CHANGED, ENUM_CHANGED,
    XML_PATH_CHANGED, MULTIPLICITY_CHANGED, ORDER_CHANGED,
    VALIDATION_RULE_CHANGED,
)

# -- confidence ------------------------------------------------------------- #
DETERMINISTIC = "deterministic"
REVIEW_REQUIRED = "review-required"

CONFIDENCES = (DETERMINISTIC, REVIEW_REQUIRED)

# -- severity (derived deterministically from the change type) -------------- #
SEV_INFORMATIONAL = "INFORMATIONAL"
SEV_LOW = "LOW"
SEV_MEDIUM = "MEDIUM"
SEV_HIGH = "HIGH"

#: Severity is a property of the change type, not of a model's opinion. A
#: change that can make a previously valid submission rejected (or a required
#: value unreportable) is HIGH; one that changes how a value must be produced is
#: MEDIUM; wording alone is INFORMATIONAL.
SEVERITY_BY_CHANGE_TYPE: Dict[str, str] = {
    FIELD_ADDED: SEV_HIGH,
    FIELD_REMOVED: SEV_HIGH,
    MANDATORY_STATUS_CHANGED: SEV_HIGH,
    XML_PATH_CHANGED: SEV_HIGH,
    ORDER_CHANGED: SEV_HIGH,          # XSD sequence order — wrong order rejects
    ND_PERMISSION_CHANGED: SEV_MEDIUM,
    ENUM_CHANGED: SEV_MEDIUM,
    FORMAT_CHANGED: SEV_MEDIUM,
    MULTIPLICITY_CHANGED: SEV_MEDIUM,
    VALIDATION_RULE_CHANGED: SEV_MEDIUM,
    FIELD_DESCRIPTION_CHANGED: SEV_INFORMATIONAL,
}

# -- Trakt implementation components ---------------------------------------- #
COMPONENT_FIELD_REGISTRY = "canonical_field_registry_regime_mapping"
COMPONENT_CODE_ORDER = "esma_code_order"
COMPONENT_ND_BEHAVIOUR = "nd_permissions_and_defaults"
COMPONENT_ENUM_MAPPING = "enum_mapping"
COMPONENT_VALIDATION = "validation_rules"
COMPONENT_XML_MAPPING = "xml_xsd_mapping_and_building"
COMPONENT_TESTS = "annex2_fixtures_and_tests"

COMPONENTS = (
    COMPONENT_FIELD_REGISTRY, COMPONENT_CODE_ORDER, COMPONENT_ND_BEHAVIOUR,
    COMPONENT_ENUM_MAPPING, COMPONENT_VALIDATION, COMPONENT_XML_MAPPING,
    COMPONENT_TESTS,
)

# -- impact statuses -------------------------------------------------------- #
NO_IMPLEMENTATION_CHANGE = "NO_IMPLEMENTATION_CHANGE"
CONFIG_CHANGE_REQUIRED = "CONFIG_CHANGE_REQUIRED"
VALIDATION_CHANGE_REQUIRED = "VALIDATION_CHANGE_REQUIRED"
XML_CHANGE_REQUIRED = "XML_CHANGE_REQUIRED"
TEST_CHANGE_REQUIRED = "TEST_CHANGE_REQUIRED"
MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"

IMPACT_STATUSES = (
    NO_IMPLEMENTATION_CHANGE, CONFIG_CHANGE_REQUIRED,
    VALIDATION_CHANGE_REQUIRED, XML_CHANGE_REQUIRED, TEST_CHANGE_REQUIRED,
    MANUAL_REVIEW_REQUIRED,
)

# -- artefact types (Stage 1 allowlist vocabulary) -------------------------- #
ARTEFACT_DISCLOSURE_WORKBOOK = "disclosure_template_workbook"
ARTEFACT_XML_SCHEMA = "xml_schema"
ARTEFACT_VALIDATION_RULES = "validation_rules"
ARTEFACT_REPORTING_INSTRUCTIONS = "reporting_instructions"
ARTEFACT_SAMPLE_MESSAGE = "sample_message"

ARTEFACT_TYPES = (
    ARTEFACT_DISCLOSURE_WORKBOOK, ARTEFACT_XML_SCHEMA,
    ARTEFACT_VALIDATION_RULES, ARTEFACT_REPORTING_INSTRUCTIONS,
    ARTEFACT_SAMPLE_MESSAGE,
)

#: Sentinel used wherever an attribute could not be derived from the
#: authoritative artefacts. Never replaced by a guess.
UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _clean(value: Any) -> Any:
    """Recursively convert dataclasses/tuples to JSON-safe primitives."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _clean(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Source manifest / snapshot model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SourceArtefact:
    """One entry of the explicit authoritative-source allowlist.

    This is the *declaration* of an artefact Trakt relies upon. It carries no
    retrieved content; that lives in :class:`SourceSnapshot`.
    """

    artefact_id: str
    authority: str                       # e.g. "ESMA"
    regime: str                          # e.g. "ESMA_Annex2"
    artefact_type: str                   # one of ARTEFACT_TYPES
    title: str = ""
    external_version: str = UNKNOWN      # ESMA's own version label, if published
    source_url: str = ""                 # primary ESMA URL (allowlist entry)
    publication_date: str = UNKNOWN      # ISO date where deterministically known
    effective_date: str = UNKNOWN        # ISO date where deterministically known
    source_status: str = UNKNOWN         # e.g. "draft", "final", "superseded"
    local_path: str = ""                 # repo-relative vendored copy, if any
    parser: str = ""                     # which normalizer consumes it
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)


@dataclass(frozen=True)
class SourceSnapshot:
    """Immutable evidence that a specific artefact had specific bytes.

    ``sha256`` is over the raw retrieved bytes. ``snapshot_path`` points at the
    stored immutable copy. ``retrieved_at`` is supplied by the caller so that
    snapshot creation stays deterministic under test.
    """

    artefact_id: str
    sha256: str
    byte_size: int
    retrieved_at: str
    snapshot_path: str
    parser_version: str
    source_url: str = ""
    external_version: str = UNKNOWN
    publication_date: str = UNKNOWN
    effective_date: str = UNKNOWN
    source_status: str = UNKNOWN
    retrieval_status: str = "OK"         # "OK" | SOURCE_CHECK_FAILED
    retrieval_detail: str = ""

    @property
    def ok(self) -> bool:
        return self.retrieval_status == "OK"

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SourceSnapshot":
        known = {f.name for f in dataclasses.fields(SourceSnapshot)}
        return SourceSnapshot(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------- #
# Normalized Annex 2 specification
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Provenance:
    """Where a parsed value came from, precisely enough to re-check by hand."""

    artefact_id: str
    locator: str            # "sheet=<name>;row=<1-based row>" or "xsd:<type>"
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)


@dataclass
class SpecField:
    """One Annex 2 field/code as normalized from the authoritative artefacts.

    Every attribute is either derived deterministically from an authoritative
    artefact or set to :data:`UNKNOWN`/``None`` and listed in
    :attr:`unresolved`. There is no inference layer and no LLM anywhere in the
    derivation path.
    """

    code: str
    record_group: str = UNKNOWN          # RREL (exposure) / RREC (collateral)
    label: str = UNKNOWN                 # ESMA "RTS Field name"
    description: str = UNKNOWN           # ESMA "MESSAGE ITEM DEFINITION"
    data_type: str = UNKNOWN             # ESMA "DATA TYPE / CODE" of value leaf
    format_pattern: str = UNKNOWN        # ESMA "Pattern" of value leaf
    mandatory: str = UNKNOWN             # "mandatory" | "optional"
    multiplicity: str = UNKNOWN          # e.g. "[1..1]"
    nd_allowed: Optional[List[str]] = None      # e.g. ["ND1","ND2","ND3","ND5"]
    nd_justification_type: str = UNKNOWN        # XSD code-list type name
    enum_type: str = UNKNOWN                    # XSD code-list type name
    enum_values: Optional[List[str]] = None     # XSD enumeration values
    xml_tag: str = UNKNOWN
    xml_path: str = UNKNOWN                     # element-node path
    value_paths: List[str] = field(default_factory=list)   # value leaves
    nd_path: str = UNKNOWN                      # NoDataOptn branch path
    validation_rules: List[str] = field(default_factory=list)
    order_index: int = -1                       # position in the source sequence
    provenance: List[Provenance] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SpecField":
        data = dict(data)
        prov = [Provenance(**p) if isinstance(p, dict) else p
                for p in data.pop("provenance", [])]
        known = {f.name for f in dataclasses.fields(SpecField)}
        out = SpecField(**{k: v for k, v in data.items() if k in known})
        out.provenance = prov
        return out


@dataclass
class NormalizedSpec:
    """A whole normalized Annex 2 specification for one source version."""

    regime: str
    spec_version: str                        # label for this parsed version
    parser_version: str
    fields: Dict[str, SpecField] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)   # source sequence of codes
    snapshots: List[SourceSnapshot] = field(default_factory=list)
    scope: Dict[str, Any] = field(default_factory=dict)  # branch/template scope
    parse_warnings: List[Dict[str, Any]] = field(default_factory=list)

    # -- convenience -------------------------------------------------------- #
    @property
    def codes(self) -> List[str]:
        return sorted(self.fields)

    @property
    def source_digest(self) -> str:
        """Combined digest of the underlying source snapshots (ordered)."""
        parts = [f"{s.artefact_id}:{s.sha256}"
                 for s in sorted(self.snapshots, key=lambda s: s.artefact_id)]
        return "|".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "spec_version": self.spec_version,
            "parser_version": self.parser_version,
            "scope": _clean(self.scope),
            "order": list(self.order),
            "fields": {c: self.fields[c].to_dict() for c in sorted(self.fields)},
            "snapshots": [s.to_dict() for s in self.snapshots],
            "parse_warnings": _clean(self.parse_warnings),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NormalizedSpec":
        return NormalizedSpec(
            regime=data["regime"],
            spec_version=data["spec_version"],
            parser_version=data["parser_version"],
            fields={k: SpecField.from_dict(v)
                    for k, v in (data.get("fields") or {}).items()},
            order=list(data.get("order") or []),
            snapshots=[SourceSnapshot.from_dict(s)
                       for s in (data.get("snapshots") or [])],
            scope=dict(data.get("scope") or {}),
            parse_warnings=list(data.get("parse_warnings") or []),
        )


# --------------------------------------------------------------------------- #
# Deltas and impact
# --------------------------------------------------------------------------- #

@dataclass
class SpecDelta:
    """One detected regulatory difference between two normalized specs."""

    code: str
    change_type: str
    old_value: Any
    new_value: Any
    severity: str
    confidence: str
    source_version_a: str
    source_version_b: str
    evidence: List[Provenance] = field(default_factory=list)
    detail: str = ""

    @property
    def delta_id(self) -> str:
        return f"{self.code}:{self.change_type}"

    def to_dict(self) -> Dict[str, Any]:
        out = _clean(self)
        out["delta_id"] = self.delta_id
        return out


@dataclass
class ImpactFinding:
    """The likely impact of one delta on one Trakt Annex 2 component."""

    delta_id: str
    code: str
    change_type: str
    component: str
    status: str
    locations: List[str] = field(default_factory=list)
    current_implementation: Any = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)


@dataclass
class WatchReport:
    """The Stage 1 deliverable. UI-agnostic; consumed as JSON by Stage 2."""

    regime: str
    contract_version: str
    generated_at: str
    baseline: Dict[str, Any] = field(default_factory=dict)
    candidate: Dict[str, Any] = field(default_factory=dict)
    source_status: str = SOURCE_UNCHANGED
    spec_status: str = SPEC_UNCHANGED
    outcome: str = NO_CHANGE_DETECTED
    regulatory_delta_count: int = 0
    implementation_impact_count: int = 0
    deltas: List[Dict[str, Any]] = field(default_factory=list)
    impacts: List[Dict[str, Any]] = field(default_factory=list)
    unresolved: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)


# --------------------------------------------------------------------------- #
# Status resolution
# --------------------------------------------------------------------------- #

def overall_status(source_status: str, spec_status: str) -> str:
    """Collapse (source, spec) into the single outcome Stage 1 reports.

    A confirmed regulatory change is reported even when another source could
    not be checked — an unchecked source can only add findings, never retract
    one. Otherwise a failed source check dominates: the run can never resolve
    to "no regulatory change" or to "source changed but spec unchanged" while
    any authoritative source is unverified.
    """
    if spec_status == SPEC_CHANGED:
        return REGULATORY_SPEC_CHANGED
    if source_status in (SOURCE_CHECK_FAILED, SOURCE_MISSING):
        return WATCH_INCONCLUSIVE
    if spec_status == SPEC_UNDETERMINED:
        return SPEC_CHANGE_UNDETERMINED
    if source_status == SOURCE_CHANGED:
        return SOURCE_CHANGED_SPEC_UNCHANGED
    return NO_CHANGE_DETECTED


def severity_for(change_type: str) -> str:
    """Deterministic severity for a change type (never model-derived)."""
    return SEVERITY_BY_CHANGE_TYPE.get(change_type, SEV_MEDIUM)


def source_status_for(baseline: Sequence[SourceSnapshot],
                      candidate: Sequence[SourceSnapshot]) -> str:
    """Compare two snapshot sets by artefact id + sha256.

    Any failed retrieval on either side yields ``SOURCE_CHECK_FAILED``; a
    missing counterpart yields ``SOURCE_MISSING``. Only when every artefact is
    present and readable on both sides does this report CHANGED/UNCHANGED.
    """
    if any(not s.ok for s in list(baseline) + list(candidate)):
        return SOURCE_CHECK_FAILED
    a = {s.artefact_id: s.sha256 for s in baseline}
    b = {s.artefact_id: s.sha256 for s in candidate}
    if not a or not b:
        return SOURCE_MISSING
    if set(a) != set(b):
        return SOURCE_MISSING
    return SOURCE_UNCHANGED if a == b else SOURCE_CHANGED
