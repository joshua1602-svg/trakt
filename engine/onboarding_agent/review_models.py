"""
review_models.py
===============

PART 4 — declarative spec objects for the LLM-assisted review workbench.

Mirrors the MI Agent's MIQuerySpec philosophy: a small, serialisable,
declarative description that NEVER executes anything and is always subject to
deterministic validation + human confirmation before any approved artefact is
written.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class InterpretedAnswer:
    """One question's answer as interpreted from natural language."""

    answer: str = ""
    confidence: float = 0.0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewAnswerSpec:
    """Structured interpretation of a natural-language review answer.

    Produced by the interpreter (deterministic or LLM). It is declarative only:
    it does not write approved artefacts. ``requires_confirmation`` is always
    True for v1 — a human must confirm before ingestion writes anything.
    """

    answers: Dict[str, InterpretedAnswer] = field(default_factory=dict)
    requires_confirmation: bool = True
    interpreter: str = "deterministic"     # deterministic | llm
    validation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answers": {qid: a.to_dict() for qid, a in self.answers.items()},
            "requires_confirmation": self.requires_confirmation,
            "interpreter": self.interpreter,
            "validation": self.validation,
        }

    def to_ingestible_dict(self) -> Dict[str, Any]:
        """Shape accepted by answer_ingestion (top-level ``answers``)."""
        return {
            "requires_confirmation": self.requires_confirmation,
            "interpreter": self.interpreter,
            "validation": self.validation,
            "answers": {
                qid: {
                    "answer": a.answer,
                    "approved_by": "",          # filled by the human on confirmation
                    "confidence": a.confidence,
                    "rationale": a.rationale,
                    "note": a.rationale,
                }
                for qid, a in self.answers.items()
            },
        }
