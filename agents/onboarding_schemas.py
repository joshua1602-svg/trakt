"""
agents/onboarding_schemas.py

Typed result objects for the Onboarding Agent v1.

These schemas define what each stage of the onboarding pipeline emits.
They are used by:
  - config_bootstrap_agent.py  → ConfigBootstrapResult
  - onboarding_agent.py        → OnboardingResult (wraps the above)
  - downstream Validation Agent reads proceed_to_validation from OnboardingResult

Design principles:
  - Serialise cleanly to/from JSON for file-based auditability.
  - No pandas/heavy deps — plain dataclasses only.
  - All fields have safe defaults so partial results are always valid.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ConfigBootstrapResult
# ---------------------------------------------------------------------------

@dataclass
class ConfigBootstrapResult:
    """
    Emitted by ConfigBootstrapAgent after profiling the tape and
    merging client/asset/regime config templates.
    """

    run_id: str = ""

    # Overall bootstrap status
    status: str = "review_required"          # approved | review_required | blocked

    # Asset class detection
    detected_asset_class: str = ""
    detected_asset_confidence: float = 0.0

    # Regime selection
    selected_regime: str = ""
    selected_regime_confidence: float = 0.0

    # Config paths resolved
    client_config_path: str = ""
    asset_config_path: str = ""
    regime_config_path: str = ""
    draft_config_path: str = ""              # always written even if not approved
    approved_config_path: str = ""          # only set when status == approved

    # Config decisions
    default_values_applied: List[Dict[str, Any]] = field(default_factory=list)
    # [{field, value, source: "template|detected|default", confidence}]

    missing_critical_config: List[Dict[str, Any]] = field(default_factory=list)
    # [{field, category, why_needed, blocking}]

    config_questions: List[Dict[str, Any]] = field(default_factory=list)
    # Structured question objects — see question schema in config_bootstrap_agent.py

    user_answers_path: str = ""             # path to answered questions JSON if provided

    approval_required: bool = True
    proceed: bool = False

    # Tape profile (non-sensitive stats; no raw values)
    tape_row_count: int = 0
    tape_field_count: int = 0
    tape_file_name: str = ""

    # Errors / warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ConfigBootstrapResult":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # Filter to only known fields to be forward-compatible
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


# ---------------------------------------------------------------------------
# MappingReviewItem
# ---------------------------------------------------------------------------

@dataclass
class MappingReviewItem:
    """
    One field-level mapping decision, after deterministic + LLM passes.
    Used in the mapping_review.json output and by the approval UI.
    """

    raw_field: str = ""
    suggested_canonical_field: Optional[str] = None

    # How this mapping was arrived at
    mapping_source: str = "unmapped"
    # exact | alias | normalized | token_set | fuzz_token_set | fuzz_ratio_norm
    # | llm | manual | unmapped

    confidence: float = 0.0

    required_for_regime: bool = False       # True if Mandatory for selected regime
    requires_review: bool = False           # True if confidence < review_threshold or source=llm
    blocker: bool = False                   # True if mandatory + unmapped/low-confidence

    reason: str = ""                        # Human-readable reason for flagging
    sample_values: List[str] = field(default_factory=list)   # Redacted sample values

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# EnumReviewItem
# ---------------------------------------------------------------------------

@dataclass
class EnumReviewItem:
    """
    One enum value that could not be deterministically resolved to a
    canonical allowed value and requires human review.
    """

    field_name: str = ""
    raw_value: str = ""
    suggested_value: Optional[str] = None

    mapping_source: str = "unmapped"        # exact | synonym | llm | unmapped

    confidence: float = 0.0

    requires_review: bool = True
    blocker: bool = False                   # True if field is mandatory and value unresolved

    reason: str = ""
    sample_count: int = 0                   # How many rows have this raw value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# OnboardingResult — master output of the Onboarding Agent
# ---------------------------------------------------------------------------

@dataclass
class OnboardingResult:
    """
    Master output of run_onboarding_agent().

    The field proceed_to_validation is the machine-readable gate signal.
    The field narrative_summary is the human-readable summary.

    Downstream Validation Agent should read proceed_to_validation from the
    onboarding_result.json file before starting.
    """

    run_id: str = ""

    # Overall status
    status: str = "failed"
    # ready_for_validation | review_required | blocked | failed

    # ---- Field mapping statistics ----
    total_input_fields: int = 0
    mapped_fields_count: int = 0            # all methods combined
    deterministic_mapped_count: int = 0     # tiers 1-6 exact/alias/fuzzy
    llm_suggested_count: int = 0            # LLM suggestions pending/confirmed
    review_fields_count: int = 0            # flagged for human review
    unmapped_fields_count: int = 0          # no mapping found
    unmapped_mandatory_count: int = 0       # unmapped AND required for regime

    # ---- Enum mapping statistics ----
    enum_fields_total: int = 0
    enum_mapped_count: int = 0
    enum_review_count: int = 0
    enum_success_rate: float = 0.0          # enum_mapped_count / enum_fields_total

    # ---- Nested results ----
    config_bootstrap: Optional[ConfigBootstrapResult] = None
    mapping_review_items: List[MappingReviewItem] = field(default_factory=list)
    enum_review_items: List[EnumReviewItem] = field(default_factory=list)

    # ---- Questions for the user ----
    blocker_questions: List[Dict[str, Any]] = field(default_factory=list)
    user_questions: List[Dict[str, Any]] = field(default_factory=list)

    # ---- Output artefact paths ----
    mapping_report_path: str = ""           # mapping_report.csv from semantic_alignment
    enum_report_path: str = ""              # enum_review.json
    onboarding_result_path: str = ""        # this file's own path
    approved_config_path: str = ""          # approved_config.yaml
    governance_artifact_path: str = ""      # LLM governance artifact if used
    canonical_draft_path: str = ""          # _canonical_full.csv from alignment

    # ---- Gate signal ----
    proceed_to_validation: bool = False

    # ---- Re-run metadata (used by workbench to re-invoke the agent) ----
    raw_tape_path: str = ""
    schema_registry_path: str = ""
    aliases_dir: str = ""
    enum_mapping_path: str = ""

    # ---- Narrative ----
    narrative_summary: str = ""

    # ---- Errors ----
    errors: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "OnboardingResult":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # Reconstruct nested objects
        if isinstance(raw.get("config_bootstrap"), dict):
            cb_data = raw["config_bootstrap"]
            known_cb = {f.name for f in ConfigBootstrapResult.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            raw["config_bootstrap"] = ConfigBootstrapResult(
                **{k: v for k, v in cb_data.items() if k in known_cb}
            )
        if isinstance(raw.get("mapping_review_items"), list):
            raw["mapping_review_items"] = [
                MappingReviewItem(**{k: v for k, v in item.items()
                                    if k in MappingReviewItem.__dataclass_fields__})  # type: ignore[attr-defined]
                for item in raw["mapping_review_items"]
            ]
        if isinstance(raw.get("enum_review_items"), list):
            raw["enum_review_items"] = [
                EnumReviewItem(**{k: v for k, v in item.items()
                                  if k in EnumReviewItem.__dataclass_fields__})  # type: ignore[attr-defined]
                for item in raw["enum_review_items"]
            ]
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})
