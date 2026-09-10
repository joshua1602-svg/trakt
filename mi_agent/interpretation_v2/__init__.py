"""mi_agent.interpretation_v2 — the natural-language control plane.

    question
        -> OpusInterpreter          (the model owns language)
        -> CandidateIntent          (meaning, never a binding)
        -> DeterministicCompiler    (the compiler owns binding)
        -> GovernedQueryPlan | Refuse | Clarify

SHADOW ONLY. Nothing here is wired into ``/mi/query``, the API, the UI or any
deterministic engine, and nothing here executes a plan. The sprint ends at the
plan.

The ownership split is the whole design:

    Opus owns      what the user means — capability, operation, measures,
                   statistic, population, filters, dimensions, geography
                   request, time request, comparison, output structure
    the compiler   whether those concepts exist, what they bind to, which
                   compositions are allowed, and whether to refuse

The model cannot write an executable contract. There is no schema slot for a
canonical field, a snapshot, SQL, pandas or code, and the parser fails closed on
anything shaped like one.
"""

from __future__ import annotations

from .compiler import (
    COMPILER_VERSION,
    CompilerContext,
    DeterministicCompiler,
    compile_intent,
)
from .equivalence import (
    EquivalenceReport,
    SCORED_DIMENSIONS,
    compare_results,
    observed_dimensions,
    outcome_fingerprint,
    plan_fingerprint,
    score_intent,
)
from .intent import (
    INTENT_SCHEMA_VERSION,
    Ambiguity,
    CandidateIntent,
    IntentParseError,
    IntentProvenance,
    RequestedOutput,
    SemanticComparison,
    SemanticFilter,
    SemanticGeography,
    SemanticMeasure,
    SemanticPopulation,
    SemanticTime,
    SourceSpan,
    candidate_intent_json_schema,
    parse_candidate_intent,
)
from .opus_interpreter import (
    CONFIGURED_MODEL,
    INTERPRETER_VERSION,
    AnthropicInterpreterClient,
    InterpretationOutcome,
    ModelResponse,
    OpusInterpreter,
    ReplayClient,
    UnavailableClient,
    build_system_blocks,
    build_tool_schema,
    build_user_prompt,
    interpret_and_compile,
)
from .outcomes import (
    OUTCOME_CLARIFY,
    OUTCOME_PLAN,
    OUTCOME_REFUSE,
    REASON_CODES,
    CompileReason,
    CompileResult,
    refuse,
)
from .plan import (
    PLAN_SCHEMA_VERSION,
    DimensionBinding,
    FilterBinding,
    GeographyBinding,
    GovernedQueryPlan,
    MeasureBinding,
    OutputPlan,
    PeriodBinding,
    PlanProvenance,
    PopulationBinding,
)
from .vocabulary import (
    CAPABILITIES,
    CAPABILITY_OPERATIONS,
    OPERATIONS,
    STATISTICS,
    VOCABULARY_VERSION,
    GovernedVocabulary,
    SemanticConcept,
    canonical_field_names,
    load_governed_vocabulary,
)

__all__ = [
    # intent
    "CandidateIntent", "INTENT_SCHEMA_VERSION", "parse_candidate_intent",
    "candidate_intent_json_schema", "IntentParseError", "IntentProvenance",
    "SemanticMeasure", "SemanticFilter", "SemanticGeography", "SemanticTime",
    "SemanticPopulation", "SemanticComparison", "RequestedOutput", "Ambiguity",
    "SourceSpan",
    # vocabulary
    "GovernedVocabulary", "SemanticConcept", "load_governed_vocabulary",
    "canonical_field_names", "CAPABILITIES", "OPERATIONS", "STATISTICS",
    "CAPABILITY_OPERATIONS", "VOCABULARY_VERSION",
    # interpreter
    "OpusInterpreter", "AnthropicInterpreterClient", "ReplayClient",
    "UnavailableClient", "InterpretationOutcome", "ModelResponse",
    "build_user_prompt", "build_system_blocks", "build_tool_schema",
    "interpret_and_compile",
    "CONFIGURED_MODEL", "INTERPRETER_VERSION",
    # compiler
    "DeterministicCompiler", "CompilerContext", "compile_intent",
    "COMPILER_VERSION",
    # outcomes
    "CompileResult", "CompileReason", "OUTCOME_PLAN", "OUTCOME_REFUSE",
    "OUTCOME_CLARIFY", "REASON_CODES", "refuse",
    # plan
    "GovernedQueryPlan", "OutputPlan", "MeasureBinding", "FilterBinding",
    "DimensionBinding", "GeographyBinding", "PeriodBinding", "PopulationBinding",
    "PlanProvenance", "PLAN_SCHEMA_VERSION",
    # equivalence
    "compare_results", "EquivalenceReport", "plan_fingerprint",
    "outcome_fingerprint", "score_intent", "observed_dimensions",
    "SCORED_DIMENSIONS",
]
