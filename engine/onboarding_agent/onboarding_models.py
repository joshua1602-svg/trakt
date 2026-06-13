"""
onboarding_models.py
====================

Plain dataclasses describing the state of an onboarding run. These mirror the
style of ``agents/onboarding_schemas.py`` (no pandas / heavy deps; safe
defaults; JSON-serialisable) so the whole project is auditable as files.

The central object is :class:`OnboardingProject`, which is threaded through
every stage of the orchestrator and accumulates artefacts as it goes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Stage-level records
# ---------------------------------------------------------------------------


@dataclass
class FileInventoryItem:
    """One classified source file (PART 3)."""

    file_path: str = ""
    file_name: str = ""
    file_type: str = ""              # csv | xlsx | xls | pdf | docx | txt | md | unknown
    classification: str = "unknown"
    confidence: float = 0.0
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    sheet_name: str = ""
    detected_reporting_date: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ColumnProfile:
    """Deterministic profile of one column in one structured file (PART 4)."""

    file_path: str = ""
    file_name: str = ""
    sheet_name: str = ""
    source_column: str = ""
    normalized_column_name: str = ""
    inferred_type: str = ""          # integer | decimal | date | boolean | identifier | string
    non_null_count: int = 0
    null_rate: float = 0.0
    unique_count: int = 0
    sample_values_redacted: List[str] = field(default_factory=list)
    min_value: str = ""
    max_value: str = ""
    date_min: str = ""
    date_max: str = ""
    likely_identifier: bool = False
    likely_reporting_date: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateKey:
    """A column that looks like a join / business key (PART 5)."""

    candidate_key: str = ""          # canonical key concept (loan_identifier, ...)
    file_path: str = ""
    file_name: str = ""
    source_column: str = ""
    unique_count: int = 0
    null_rate: float = 0.0
    uniqueness_ratio: float = 0.0
    confidence: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OverlapFinding:
    """Two source columns that appear to be the same business field (PART 5)."""

    canonical_candidate: str = ""
    source_file_a: str = ""
    source_column_a: str = ""
    source_file_b: str = ""
    source_column_b: str = ""
    similarity_score: float = 0.0
    sample_match_rate: float = 0.0
    recommended_primary_source: str = ""
    recommended_secondary_source: str = ""
    review_required: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MappingCandidate:
    """A proposed canonical mapping for one source column (PART 6)."""

    source_file: str = ""
    source_file_classification: str = ""
    source_column: str = ""
    candidate_canonical_field: str = ""
    confidence: float = 0.0
    method: str = "unmapped"
    sample_values_redacted: List[str] = field(default_factory=list)
    requires_review: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfigSuggestion:
    """One inferred config value (PART 7)."""

    field: str = ""
    suggested_value: str = ""
    confidence: float = 0.0
    source_file: str = ""
    source_column_or_document_reference: str = ""
    evidence: str = ""
    review_status: str = "requires_review"   # suggested | requires_review | missing

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentExtraction:
    """One config-relevant fact extracted from an unstructured document.

    Extraction minimisation (PART 6): we retain only the field, value, a short
    capped evidence excerpt and a reference — never full document text, pages,
    clauses, signatures, addresses or bank details.
    """

    field: str = ""
    value: str = ""
    source_document: str = ""
    source_reference: str = ""
    confidence: float = 0.0
    retained_evidence: str = ""      # capped by policy (allowed_retained_evidence_chars)
    status: str = "requires_review"  # suggested | requires_review | missing

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GapQuestion:
    """A user-facing question raised by the gap analyzer (PART 8)."""

    question_id: str = ""
    category: str = ""               # date | source_of_truth | enum | config | regime | geography
    severity: str = "info"           # blocking | high | medium | info
    question: str = ""
    reason: str = ""
    candidate_answers: List[str] = field(default_factory=list)
    default_recommendation: str = ""
    blocking_for: List[str] = field(default_factory=list)
    source_evidence: str = ""
    # Structured targets so answer ingestion is deterministic (no text parsing):
    #   subject        - the thing the answer configures (canonical field, config
    #                     key, 'reporting_date', 'uk_geography_mode', enum field…)
    #   subject_value  - a secondary qualifier (e.g. the raw enum value 'manual')
    subject: str = ""
    subject_value: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# OnboardingProject — the central state object
# ---------------------------------------------------------------------------


@dataclass
class OnboardingProject:
    """
    Central state for one onboarding run. Threaded through every stage.

    The orchestrator populates the list/dict fields as each stage runs and then
    serialises the artefacts to ``output_dir``.
    """

    project_id: str = ""
    client_name: str = ""
    input_dir: str = ""
    output_dir: str = ""

    # Onboarding mode (PART 1): mi_mna | regulatory_mi | warehouse_securitisation
    onboarding_mode: str = "regulatory_mi"

    # Inputs / discovered files
    source_files: List[str] = field(default_factory=list)

    # Stage outputs
    file_inventory: List[FileInventoryItem] = field(default_factory=list)
    field_profiles: List[ColumnProfile] = field(default_factory=list)
    candidate_keys: List[CandidateKey] = field(default_factory=list)
    overlap_analysis: List[OverlapFinding] = field(default_factory=list)
    mapping_candidates: List[MappingCandidate] = field(default_factory=list)
    config_suggestions: List[ConfigSuggestion] = field(default_factory=list)
    document_extractions: List[DocumentExtraction] = field(default_factory=list)
    gap_questions: List[GapQuestion] = field(default_factory=list)

    # Run-level status
    review_status: str = "draft"     # draft | review_required | blocked
    generated_artifacts: List[str] = field(default_factory=list)

    # Reference paths used by the mapping engine
    registry_path: str = ""
    aliases_dir: str = ""

    # ------------------------------------------------------------------

    def to_summary_dict(self) -> Dict[str, Any]:
        """A compact, JSON-friendly summary (used for 09_onboarding_run_summary.json)."""
        return {
            "project_id": self.project_id,
            "client_name": self.client_name,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "onboarding_mode": self.onboarding_mode,
            "review_status": self.review_status,
            "counts": {
                "source_files": len(self.source_files),
                "classified_files": len(self.file_inventory),
                "column_profiles": len(self.field_profiles),
                "candidate_keys": len(self.candidate_keys),
                "overlap_findings": len(self.overlap_analysis),
                "mapping_candidates": len(self.mapping_candidates),
                "config_suggestions": len(self.config_suggestions),
                "gap_questions": len(self.gap_questions),
                "blocking_questions": sum(
                    1 for q in self.gap_questions if q.severity == "blocking"
                ),
            },
            "generated_artifacts": list(self.generated_artifacts),
        }
