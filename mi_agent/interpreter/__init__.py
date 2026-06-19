"""mi_agent.interpreter — MI question interpretation harness (Phase 8A).

A deterministic, rule-based baseline that maps a controlled set of business
questions onto governed MIQuerySpec v2 (normalised + validated), plus an
evaluator and golden examples that form the grading harness for a future LLM
interpreter.

Constraints: no external LLM calls, no analytics computed by the interpreter,
no Azure/Streamlit/onboarding/M&A/Annex 2 work. Generated specs never bypass
``validate_query_spec``.
"""

from __future__ import annotations

from .anthropic import (
    AnthropicClient,
    AnthropicMIInterpreterClient,
    interpret_from_llm_output,
    interpret_with_anthropic,
    parse_spec_json,
)
from .deterministic import interpret
from .evaluator import EvalReport, evaluate_interpretation
from .examples import load_golden
from .models import (
    DETERMINISTIC,
    FIXTURE,
    LLM_STUB,
    InterpretationResult,
    InterpreterContext,
)
from .prompt import build_mi_spec_prompt

__all__ = [
    "interpret",
    "InterpreterContext",
    "InterpretationResult",
    "evaluate_interpretation",
    "EvalReport",
    "load_golden",
    "DETERMINISTIC",
    "LLM_STUB",
    "FIXTURE",
    "build_mi_spec_prompt",
    "interpret_with_anthropic",
    "interpret_from_llm_output",
    "parse_spec_json",
    "AnthropicClient",
    "AnthropicMIInterpreterClient",
]
