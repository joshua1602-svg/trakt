"""mi_agent.interpreter.evaluator — grade an interpretation against expectations.

Phase 8A. Compares an :class:`InterpretationResult` to a golden expectation
(expected spec fields, validity, issue codes, clarification). Reused later to
grade an LLM interpreter — it never executes analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import InterpretationResult

# Spec fields the evaluator can compare on (the discriminating ones).
COMPARABLE_FIELDS = (
    "route_id", "execution_mode", "state", "temporal_mode", "dimension",
    "metric", "aggregation", "risk_monitor", "risk_monitor_mode",
    "bucket_strategy", "baseline_date", "current_date", "start_date",
    "end_date", "chart_type", "output_type",
)


@dataclass
class EvalReport:
    passed: bool = True
    mismatches: List[str] = field(default_factory=list)
    checked: Dict[str, Any] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.mismatches.append(msg)


def evaluate_interpretation(
    result: InterpretationResult, *,
    expected_spec: Optional[Dict[str, Any]] = None,
    expected_valid: Optional[bool] = None,
    expected_issue_codes: Optional[List[str]] = None,
    expected_clarification_required: bool = False,
) -> EvalReport:
    """Grade *result* against golden expectations. Returns an :class:`EvalReport`."""
    report = EvalReport()

    # Clarification expectation takes precedence.
    if expected_clarification_required:
        if not result.clarification_required:
            report.fail("expected clarification_required=True")
        return report
    if result.clarification_required:
        report.fail("unexpected clarification_required=True "
                    f"({result.clarification_question!r})")
        return report

    # Spec field comparison.
    if expected_spec:
        spec = result.normalized_spec
        if spec is None:
            report.fail("no normalized_spec produced")
        else:
            for key, want in expected_spec.items():
                if key not in COMPARABLE_FIELDS:
                    continue
                got = getattr(spec, key, None)
                report.checked[key] = got
                if got != want:
                    report.fail(f"{key}: expected {want!r}, got {got!r}")

    # Validity.
    if expected_valid is not None:
        got_valid = bool(result.validation_result and result.validation_result.ok)
        if got_valid != expected_valid:
            report.fail(f"expected_valid={expected_valid}, got {got_valid} "
                        f"(codes={result.issue_codes()})")

    # Issue codes (expected ⊆ produced).
    for code in (expected_issue_codes or []):
        if code not in result.issue_codes():
            report.fail(f"missing expected issue code {code!r} "
                        f"(got {result.issue_codes()})")

    return report
