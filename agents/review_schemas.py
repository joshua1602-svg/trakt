"""
agents/review_schemas.py

Typed models for human review decisions made in the Onboarding Workbench.

These are produced by:
  - ui/onboarding_review.py  (Streamlit)
  - cli/onboarding_review_cli.py  (CLI fallback)

and consumed by:
  - agents/learning_persistence.py  (persist to alias/enum/config files)
  - agents/onboarding_agent.py      (re-run with questionnaire_answers_path)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------

@dataclass
class QuestionAnswer:
    """User answer to one config bootstrap question."""
    question_id: str = ""
    answer: str = ""
    approved: bool = False
    comments: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MappingDecision:
    """User decision on one field mapping review item."""
    raw_field: str = ""
    approved: bool = False                       # accepted as-is or with override
    selected_canonical_field: Optional[str] = None   # None = ignored / skipped
    comments: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnumDecision:
    """User decision on one enum value review item."""
    field_name: str = ""
    raw_value: str = ""
    approved: bool = False
    selected_value: Optional[str] = None
    comments: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewSubmission:
    """
    Complete set of human review decisions for one onboarding run.
    Written to {run_dir}/{run_id}_review_submission.json after the user submits.
    """
    run_id: str = ""
    question_answers: List[QuestionAnswer] = field(default_factory=list)
    mapping_decisions: List[MappingDecision] = field(default_factory=list)
    enum_decisions: List[EnumDecision] = field(default_factory=list)
    submitted_at: str = ""

    def __post_init__(self) -> None:
        if not self.submitted_at:
            self.submitted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---- counts ----

    @property
    def approved_question_count(self) -> int:
        return sum(1 for q in self.question_answers if q.approved)

    @property
    def approved_mapping_count(self) -> int:
        return sum(1 for m in self.mapping_decisions if m.approved and m.selected_canonical_field)

    @property
    def approved_enum_count(self) -> int:
        return sum(1 for e in self.enum_decisions if e.approved and e.selected_value)

    # ---- serialisation ----

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ReviewSubmission":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw.get("question_answers"), list):
            raw["question_answers"] = [
                QuestionAnswer(**{k: v for k, v in q.items()
                                  if k in QuestionAnswer.__dataclass_fields__})
                for q in raw["question_answers"]
            ]
        if isinstance(raw.get("mapping_decisions"), list):
            raw["mapping_decisions"] = [
                MappingDecision(**{k: v for k, v in m.items()
                                   if k in MappingDecision.__dataclass_fields__})
                for m in raw["mapping_decisions"]
            ]
        if isinstance(raw.get("enum_decisions"), list):
            raw["enum_decisions"] = [
                EnumDecision(**{k: v for k, v in e.items()
                                if k in EnumDecision.__dataclass_fields__})
                for e in raw["enum_decisions"]
            ]
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in known})


# ---------------------------------------------------------------------------
# Convenience builders used by both UI and CLI
# ---------------------------------------------------------------------------

def build_questionnaire_answers_json(answers: List[QuestionAnswer]) -> List[Dict[str, Any]]:
    """
    Convert QuestionAnswers to the format expected by run_onboarding_agent
    (questionnaire_answers_path).  Format: [{question_id, answer}, ...]
    """
    return [{"question_id": a.question_id, "answer": a.answer} for a in answers if a.approved]


def build_mapping_overrides_json(decisions: List[MappingDecision]) -> List[Dict[str, Any]]:
    """
    Approved mapping decisions for persist + re-run.
    [{raw_field, canonical_field, action: confirmed|remapped}]
    """
    return [
        {
            "raw_field": d.raw_field,
            "canonical_field": d.selected_canonical_field,
            "action": "confirmed",
            "comments": d.comments,
        }
        for d in decisions
        if d.approved and d.selected_canonical_field
    ]


def build_enum_overrides_json(decisions: List[EnumDecision]) -> List[Dict[str, Any]]:
    """
    Approved enum decisions for persist + re-run.
    [{field_name, raw_value, canonical_value}]
    """
    return [
        {
            "field_name": e.field_name,
            "raw_value": e.raw_value,
            "canonical_value": e.selected_value,
            "comments": e.comments,
        }
        for e in decisions
        if e.approved and e.selected_value
    ]
