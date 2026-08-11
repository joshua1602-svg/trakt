"""regulatory_watch.discovery.contracts — the Stage 1B data contract.

Pure data and controlled vocabularies. No I/O, no network, no model, no import
of Stage 1A's comparison engine. Everything is JSON-round-trippable so a Stage 2
OCC Regulatory Config view can consume it unchanged.

Three boundaries are enforced structurally rather than by convention:

1. **A development is not a delta.** :class:`RegulatoryDevelopment` can only
   *reference* a Stage 1A result (:class:`TechnicalSpecLinkage` holds ids,
   a report path and counts). There is no field on it that could express a
   field-level change, so narrative text cannot become one.
2. **A discovery failure is never "nothing new".** ``DISCOVERY_FAILED`` is a
   first-class status and :func:`overall_discovery_status` refuses to report a
   clean sweep when any source failed.
3. **Unknown stays unknown.** Dates and legal status default to ``None`` and
   are only populated when the source states them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Source vocabulary
# --------------------------------------------------------------------------- #

SOURCE_RSS_FEED = "rss_feed"
SOURCE_PUBLICATION_PAGE = "publication_page"
SOURCE_LANDING_PAGE = "landing_page"

SOURCE_TYPES = (SOURCE_RSS_FEED, SOURCE_PUBLICATION_PAGE, SOURCE_LANDING_PAGE)

PARSER_RSS = "rss"
PARSER_ATOM = "atom"
PARSER_HTML_LISTING = "html_listing"

PARSER_TYPES = (PARSER_RSS, PARSER_ATOM, PARSER_HTML_LISTING)

#: Same two-tier control model as Stage 1A. A gating discovery source is one
#: whose silence would leave a real blind spot; a corroborating one adds
#: coverage. Both failures are reported; only a gating failure withholds the
#: "no new relevant publication" claim.
GATING = "gating"
CORROBORATING = "corroborating"

CRITICALITIES = (GATING, CORROBORATING)

# --------------------------------------------------------------------------- #
# Discovery status
# --------------------------------------------------------------------------- #

DISCOVERY_OK = "DISCOVERY_OK"
DISCOVERY_PARTIAL = "DISCOVERY_PARTIAL"       # a corroborating source failed
DISCOVERY_FAILED = "DISCOVERY_FAILED"         # a gating source failed

DISCOVERY_STATUSES = (DISCOVERY_OK, DISCOVERY_PARTIAL, DISCOVERY_FAILED)

# -- per-source outcome ----------------------------------------------------- #
SOURCE_OK = "OK"
SOURCE_FETCH_FAILED = "FETCH_FAILED"
SOURCE_PARSE_FAILED = "PARSE_FAILED"
SOURCE_UNEXPECTED_SCHEMA = "UNEXPECTED_SCHEMA"
SOURCE_NOT_ALLOWLISTED = "NOT_ALLOWLISTED"
SOURCE_DISABLED = "DISABLED"

SOURCE_OUTCOMES = (SOURCE_OK, SOURCE_FETCH_FAILED, SOURCE_PARSE_FAILED,
                   SOURCE_UNEXPECTED_SCHEMA, SOURCE_NOT_ALLOWLISTED,
                   SOURCE_DISABLED)

# --------------------------------------------------------------------------- #
# Publication classification vocabulary
# --------------------------------------------------------------------------- #

CONSULTATION = "CONSULTATION"
FINAL_REPORT = "FINAL_REPORT"
RTS_ITS = "RTS_ITS"
Q_AND_A = "Q_AND_A"
TECHNICAL_INSTRUCTION = "TECHNICAL_INSTRUCTION"
TEMPLATE_SCHEMA_UPDATE = "TEMPLATE_SCHEMA_UPDATE"
VALIDATION_RULE_UPDATE = "VALIDATION_RULE_UPDATE"
OTHER_REGULATORY_PUBLICATION = "OTHER_REGULATORY_PUBLICATION"

PUBLICATION_TYPES = (CONSULTATION, FINAL_REPORT, RTS_ITS, Q_AND_A,
                     TECHNICAL_INSTRUCTION, TEMPLATE_SCHEMA_UPDATE,
                     VALIDATION_RULE_UPDATE, OTHER_REGULATORY_PUBLICATION)

# -- development status ----------------------------------------------------- #
INFORMATION_ONLY = "INFORMATION_ONLY"
POTENTIAL_FUTURE_CHANGE = "POTENTIAL_FUTURE_CHANGE"
FUTURE_CHANGE_EXPECTED = "FUTURE_CHANGE_EXPECTED"
INTERPRETATION_REVIEW_REQUIRED = "INTERPRETATION_REVIEW_REQUIRED"
TECHNICAL_SPEC_CHANGE = "TECHNICAL_SPEC_CHANGE"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
NOT_RELEVANT = "NOT_RELEVANT"

DEVELOPMENT_STATUSES = (INFORMATION_ONLY, POTENTIAL_FUTURE_CHANGE,
                        FUTURE_CHANGE_EXPECTED,
                        INTERPRETATION_REVIEW_REQUIRED, TECHNICAL_SPEC_CHANGE,
                        REVIEW_REQUIRED, NOT_RELEVANT)

#: Statuses that always require a human to look. TECHNICAL_SPEC_CHANGE is here
#: because Stage 1B must never be the thing that decides what a new technical
#: package means — Stage 1A computes the delta, a person acts on it.
ALWAYS_REVIEW_STATUSES = (INTERPRETATION_REVIEW_REQUIRED, TECHNICAL_SPEC_CHANGE,
                          REVIEW_REQUIRED, FUTURE_CHANGE_EXPECTED)

# -- relevance -------------------------------------------------------------- #
RELEVANCE_RELEVANT = "RELEVANT"
RELEVANCE_POTENTIAL = "POTENTIAL"
RELEVANCE_NOT_RELEVANT = "NOT_RELEVANT"
RELEVANCE_UNKNOWN = "UNKNOWN"

RELEVANCE_VALUES = (RELEVANCE_RELEVANT, RELEVANCE_POTENTIAL,
                    RELEVANCE_NOT_RELEVANT, RELEVANCE_UNKNOWN)

# -- confidence ------------------------------------------------------------- #
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

CONFIDENCES = (HIGH, MEDIUM, LOW)

# -- classifier provenance -------------------------------------------------- #
CLASSIFIER_DETERMINISTIC = "deterministic"
CLASSIFIER_SEMANTIC = "semantic"
CLASSIFIER_SEMANTIC_FAILED = "semantic_failed"

CLASSIFIER_KINDS = (CLASSIFIER_DETERMINISTIC, CLASSIFIER_SEMANTIC,
                    CLASSIFIER_SEMANTIC_FAILED)

# --------------------------------------------------------------------------- #
# Technical-artefact linkage
# --------------------------------------------------------------------------- #

NO_TECHNICAL_ARTEFACT = "NO_TECHNICAL_ARTEFACT"
MATCHES_STAGE1A_SOURCE = "MATCHES_STAGE1A_SOURCE"
NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED = "NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED"

LINKAGE_STATUSES = (NO_TECHNICAL_ARTEFACT, MATCHES_STAGE1A_SOURCE,
                    NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED)

# -- deduplication ---------------------------------------------------------- #
SEEN_NEW = "NEW"
SEEN_UNCHANGED = "UNCHANGED"
SEEN_REVISED = "REVISED"

SEEN_STATES = (SEEN_NEW, SEEN_UNCHANGED, SEEN_REVISED)


def _clean(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _clean(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PublicationSource:
    """One entry of the upstream ESMA publication allowlist."""

    source_id: str
    source_type: str                      # one of SOURCE_TYPES
    url: str                              # official ESMA URL
    parser: str                           # one of PARSER_TYPES
    criticality: str = GATING
    enabled: bool = True
    title: str = ""
    retrieval_frequency: str = "weekly"
    last_known_identifier: Optional[str] = None
    verified_reachable: bool = False      # has anyone actually fetched it?
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)


@dataclass
class SourceOutcome:
    """What happened when one source was checked. Never silently dropped."""

    source_id: str
    criticality: str
    outcome: str                          # one of SOURCE_OUTCOMES
    item_count: int = 0
    detail: str = ""
    retrieved_at: str = ""
    content_sha256: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome == SOURCE_OK

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)


# --------------------------------------------------------------------------- #
# Publications
# --------------------------------------------------------------------------- #

@dataclass
class RawPublication:
    """One publication as the source published it. Verbatim; nothing inferred.

    ``official_identifier`` is used for identity when the source supplies one
    (an RSS ``guid``, an ESMA document reference). Otherwise identity is a
    deterministic fingerprint — see :mod:`regulatory_watch.discovery.state`.
    """

    source_id: str
    title: str
    url: str
    official_identifier: Optional[str] = None
    publication_date: Optional[str] = None       # ISO date, only if published
    summary: str = ""
    categories: List[str] = field(default_factory=list)
    linked_urls: List[str] = field(default_factory=list)
    raw_locator: str = ""                        # e.g. "item[3]" in the feed

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RawPublication":
        known = {f.name for f in dataclasses.fields(RawPublication)}
        return RawPublication(**{k: v for k, v in data.items() if k in known})


@dataclass
class PublicationIdentity:
    """Deterministic identity plus what the state store knew about it."""

    development_id: str
    fingerprint: str                # content fingerprint (revision detection)
    identity_basis: str             # "official_identifier" | "derived"
    seen_state: str = SEEN_NEW      # one of SEEN_STATES
    first_seen: Optional[str] = None
    previous_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)


# --------------------------------------------------------------------------- #
# Triage and classification
# --------------------------------------------------------------------------- #

@dataclass
class TriageResult:
    """Stage A: the deterministic keyword/metadata filter.

    Deliberately broad — a false negative here is invisible, so the filter errs
    towards passing things through. A rejection always records WHY.
    """

    passed: bool
    matched_terms: List[str] = field(default_factory=list)
    matched_groups: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)


@dataclass
class Classification:
    """Stage B output. Strict schema; ambiguity becomes REVIEW_REQUIRED."""

    publication_type: str
    development_status: str
    annex2_relevance: str
    relevance_reason: str
    confidence: str
    classifier: str = CLASSIFIER_DETERMINISTIC   # one of CLASSIFIER_KINDS
    review_required: bool = True
    matched_signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)


# --------------------------------------------------------------------------- #
# Linkage to Stage 1A
# --------------------------------------------------------------------------- #

@dataclass
class TechnicalArtefactLink:
    """A technical artefact URL a publication points at."""

    url: str
    status: str                     # MATCHES_STAGE1A_SOURCE | NEW_..._REQUIRED
    stage1a_artefact_id: Optional[str] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)


@dataclass
class TechnicalSpecLinkage:
    """A REFERENCE to a Stage 1A result. Never a computation of one.

    Stage 1B can say "Stage 1A should run" and, when handed a Stage 1A report,
    can quote its identifiers and counts. It cannot produce them: there is no
    field here that expresses a field-level change.
    """

    status: str                                   # one of LINKAGE_STATUSES
    artefact_links: List[TechnicalArtefactLink] = field(default_factory=list)
    stage1a_verification_required: bool = False
    #: Populated only from a Stage 1A report supplied by the caller.
    stage1a_report_path: Optional[str] = None
    stage1a_outcome: Optional[str] = None
    stage1a_baseline_snapshot_ids: List[str] = field(default_factory=list)
    stage1a_candidate_snapshot_ids: List[str] = field(default_factory=list)
    regulatory_delta_count: Optional[int] = None
    implementation_impact_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)


# --------------------------------------------------------------------------- #
# The regulatory-development record
# --------------------------------------------------------------------------- #

@dataclass
class RegulatoryDevelopment:
    """The Stage 1B deliverable: one reviewable regulatory-development record."""

    development_id: str
    authority: str
    regime: str
    publication: Dict[str, Any]
    publication_type: str
    development_status: str
    annex2_relevance: str
    relevance_reason: str
    confidence: str
    review_required: bool
    seen_state: str = SEEN_NEW
    first_seen: Optional[str] = None
    discovered_at: str = ""
    discovery_version: str = ""
    classifier: str = CLASSIFIER_DETERMINISTIC
    # Dates: populated only where the source states them. Never guessed.
    effective_date: Optional[str] = None
    implementation_date: Optional[str] = None
    consultation_deadline: Optional[str] = None
    supersedes: Optional[str] = None
    final_status: Optional[str] = None            # "final" | "consultation"
    technical_spec_linkage: Optional[Dict[str, Any]] = None
    triage: Dict[str, Any] = field(default_factory=dict)
    matched_signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RegulatoryDevelopment":
        known = {f.name for f in dataclasses.fields(RegulatoryDevelopment)}
        return RegulatoryDevelopment(
            **{k: v for k, v in data.items() if k in known})


@dataclass
class DiscoveryReport:
    """The weekly discovery report. UI-agnostic."""

    authority: str
    regime: str
    contract_version: str
    discovery_version: str
    discovery_date: str
    discovery_status: str
    sources_checked: int = 0
    sources_failed: int = 0
    source_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    publications_seen: int = 0
    new_publications: int = 0
    revised_publications: int = 0
    previously_seen_publications: int = 0
    potentially_relevant: int = 0
    not_relevant: int = 0
    review_required_count: int = 0
    technical_spec_change_candidates: int = 0
    developments: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    stage1a_linkages: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self)


# --------------------------------------------------------------------------- #
# Status resolution
# --------------------------------------------------------------------------- #

def overall_discovery_status(outcomes: Sequence[SourceOutcome]) -> str:
    """Collapse per-source outcomes into the run's discovery status.

    A **gating** source that could not be checked means the sweep is not a
    sweep: ``DISCOVERY_FAILED``. A corroborating failure narrows coverage
    without blinding the run: ``DISCOVERY_PARTIAL``. Neither is ever reported
    as a clean pass, and no caller may read "no new publications" from either.
    """
    considered = [o for o in outcomes if o.outcome != SOURCE_DISABLED]
    if not considered:
        return DISCOVERY_FAILED
    if any(not o.ok and o.criticality == GATING for o in considered):
        return DISCOVERY_FAILED
    if any(not o.ok for o in considered):
        return DISCOVERY_PARTIAL
    return DISCOVERY_OK


def requires_review(development_status: str, confidence: str,
                    relevance: str) -> bool:
    """Whether a human must look. Errs towards yes."""
    if development_status in ALWAYS_REVIEW_STATUSES:
        return True
    if development_status == NOT_RELEVANT:
        return False
    # Judged not relevant to Annex 2: the reason is recorded, but asking a
    # human to confirm every ESMA market statistic would make the review queue
    # worthless, which is itself a control failure.
    if relevance == RELEVANCE_NOT_RELEVANT:
        return False
    if confidence != HIGH:
        return True
    if relevance in (RELEVANCE_POTENTIAL, RELEVANCE_UNKNOWN):
        return True
    return development_status != INFORMATION_ONLY
