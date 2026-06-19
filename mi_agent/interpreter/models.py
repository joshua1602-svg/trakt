"""mi_agent.interpreter.models — interpretation result + context models (Phase 8A).

Data structures for translating a business question into a governed
MIQuerySpec v2. The interpreter NEVER computes analytics — it only proposes a
spec, which is normalised and validated deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_spec_validation import SpecValidationResult

# interpretation_method values.
DETERMINISTIC = "deterministic"
LLM_STUB = "llm_stub"
FIXTURE = "fixture"


@dataclass
class InterpreterContext:
    """Deterministic context the interpreter resolves relative dates / client
    against. Defaults make the harness fully deterministic (no 'now')."""

    snapshot_client_id: str = "clientA"
    route_id: str = "mi"
    as_of: str = "2024-03-31"           # 'current' / 'latest' anchor
    prev_period: str = "2024-02-29"     # 'last month'
    range_start: str = "2024-01-01"     # start of 'last three months'
    portfolio_config_available: bool = True


@dataclass
class InterpretationResult:
    raw_question: str
    candidate_spec: Optional[Dict[str, Any]] = None      # raw dict before normalise
    normalized_spec: Optional[MIQuerySpec] = None
    validation_result: Optional[SpecValidationResult] = None
    confidence: Optional[float] = None
    issues: List[Dict[str, Any]] = field(default_factory=list)
    clarification_required: bool = False
    clarification_question: Optional[str] = None
    interpretation_method: str = DETERMINISTIC

    @property
    def ok(self) -> bool:
        """A successful, validated, non-clarifying interpretation."""
        return (not self.clarification_required
                and self.validation_result is not None
                and self.validation_result.ok)

    def issue_codes(self) -> List[str]:
        codes = [i["code"] for i in self.issues]
        if self.validation_result is not None:
            codes += self.validation_result.codes()
        return codes
