"""mi_agent.interpreter.runtime_bridge — NLQ → interpreter → runtime (Phase 8C).

End-to-end smoke bridge that takes a natural-language MI question, interprets it
into a governed MIQuerySpec v2, and — only if that interpretation is valid and
unambiguous — executes it through the existing deterministic runtime
(``run_mi_query``).

The separation of concerns is strict and is the whole point of this phase:

* the interpreter (deterministic baseline or the Anthropic-first adapter) ONLY
  proposes a spec; it never computes analytics;
* ``run_mi_query`` remains the single execution engine;
* an invalid, ambiguous, or clarification-requiring interpretation is NEVER
  executed.

No external LLM calls live here — the interpreter is supplied by the caller
(fake clients in tests). No Azure, Streamlit, M&A, Annex 2, or chart-type work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from mi_agent.mi_runtime import RuntimeResult, run_mi_query
from mi_agent.mi_query_spec import MIQuerySpec

from .anthropic import AnthropicMIInterpreterClient, interpret_with_anthropic
from .models import InterpretationResult, InterpreterContext

# A caller may pass either an Anthropic-style client (has complete_mi_spec_json),
# or a plain interpreter callable ``f(question, context) -> InterpretationResult``.
InterpreterLike = Union[AnthropicMIInterpreterClient,
                        Callable[[str, Optional[InterpreterContext]],
                                 InterpretationResult]]

NOT_EXECUTED_CLARIFICATION = "not_executed_clarification_required"
NOT_EXECUTED_INVALID_SPEC = "not_executed_invalid_spec"


@dataclass
class BridgeResult:
    """Combined NLQ→runtime outcome.

    ``executed`` is True only when a valid, unambiguous spec was actually run
    through ``run_mi_query``. Otherwise ``runtime_result`` is None and the
    ``issues`` explain why nothing was executed.
    """

    raw_question: str
    interpretation: InterpretationResult
    normalized_spec: Optional[MIQuerySpec] = None
    runtime_result: Optional[RuntimeResult] = None
    issues: List[Dict[str, Any]] = field(default_factory=list)
    executed: bool = False

    @property
    def ok(self) -> bool:
        """The full pipeline succeeded: interpreted, executed, runtime ok."""
        return (self.executed and self.runtime_result is not None
                and self.runtime_result.ok)

    @property
    def clarification_required(self) -> bool:
        return self.interpretation.clarification_required

    @property
    def data(self):
        return self.runtime_result.data if self.runtime_result is not None else None

    @property
    def chart_instruction(self) -> Optional[Dict[str, Any]]:
        return (self.runtime_result.chart_instruction
                if self.runtime_result is not None else None)

    def issue_codes(self) -> List[str]:
        return [i["code"] for i in self.issues]


def _interpret(question: str, context: Optional[InterpreterContext],
               interpreter: InterpreterLike, *,
               semantics: Optional[dict]) -> InterpretationResult:
    """Run the supplied interpreter, supporting both an Anthropic-style client
    and a plain interpreter callable."""
    if hasattr(interpreter, "complete_mi_spec_json"):
        return interpret_with_anthropic(question, context, interpreter,
                                        semantics=semantics)
    if callable(interpreter):
        return interpreter(question, context)
    raise TypeError("interpreter must be an Anthropic-style client "
                    "(complete_mi_spec_json) or a callable interpreter")


def interpret_and_run_mi_query(question: str,
                               context: Optional[InterpreterContext],
                               llm_client_or_interpreter: InterpreterLike,
                               store=None,
                               *,
                               data=None,
                               semantics=None,
                               risk_config: Optional[dict] = None,
                               build_chart: bool = False,
                               routes_dir=None,
                               store_root: Optional[str] = None,
                               allow_mna_risk: bool = False,
                               stage_probabilities: Optional[Dict[str, float]] = None,
                               ) -> BridgeResult:
    """Interpret *question* into a governed spec and, only if valid, execute it.

    The interpreter proposes a spec; ``run_mi_query`` executes it. An ambiguous
    or invalid interpretation is reported but never run.
    """
    ctx = context or InterpreterContext()
    interp = _interpret(question, ctx, llm_client_or_interpreter,
                        semantics=semantics)

    # Gate 1 — a clarification is not an answer; never execute.
    if interp.clarification_required:
        return BridgeResult(
            raw_question=question, interpretation=interp,
            normalized_spec=interp.normalized_spec, runtime_result=None,
            issues=[{"code": NOT_EXECUTED_CLARIFICATION, "severity": "error",
                     "field": None,
                     "message": "clarification required; not executed: "
                                f"{interp.clarification_question!r}"},
                    *interp.issues],
            executed=False)

    # Gate 2 — the spec must have validated. ``interp.ok`` means non-clarifying,
    # validated, and (for the LLM adapter) free of adapter-level errors.
    if not interp.ok or interp.normalized_spec is None:
        return BridgeResult(
            raw_question=question, interpretation=interp,
            normalized_spec=interp.normalized_spec, runtime_result=None,
            issues=[{"code": NOT_EXECUTED_INVALID_SPEC, "severity": "error",
                     "field": None,
                     "message": "interpretation did not produce a valid spec; "
                                "not executed"},
                    *interp.issues],
            executed=False)

    # Execute the validated, normalised spec through the single runtime engine.
    runtime = run_mi_query(
        interp.normalized_spec, semantics=semantics, data=data, store=store,
        store_root=store_root, routes_dir=routes_dir, risk_config=risk_config,
        allow_mna_risk=allow_mna_risk, stage_probabilities=stage_probabilities,
        build_chart=build_chart)

    return BridgeResult(
        raw_question=question, interpretation=interp,
        normalized_spec=interp.normalized_spec, runtime_result=runtime,
        issues=[*interp.issues, *runtime.issues], executed=True)
