"""mi_agent.interpreter.anthropic — Anthropic/Claude-first LLM interpreter adapter (Phase 8B).

Translates a natural-language MI question into a governed MIQuerySpec v2 using
Claude. The LLM is used ONLY to interpret the question into spec JSON — it never
computes analytics, numbers, or results. Every candidate spec is normalised via
``MIQuerySpec.normalized()`` and validated by ``validate_query_spec`` before it is
ever considered runnable; any adapter-level error forces a clarification rather
than executing an unsafe spec.

The Anthropic-specific API surface is isolated behind a thin, mockable client
boundary (:class:`AnthropicMIInterpreterClient`). Tests use fake clients only —
this module performs no network calls and imports no SDK at module load. A real
client (:class:`AnthropicClient`) lazily imports the ``anthropic`` SDK inside its
method so the dependency is optional and never touched in tests.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from mi_agent.mi_query_spec import CHART_TYPES, MIQuerySpec
from mi_agent.mi_spec_validation import validate_query_spec
from mi_agent.quantile_buckets import QUANTILE_DIMENSIONS
from mi_agent.states.models import ERROR, WARNING, make_issue

from .models import LLM_STUB, InterpretationResult, InterpreterContext
from .prompt import ALLOWED_SPEC_FIELDS, CLARIFICATION_FIELDS, build_mi_spec_prompt

# Adapter-level issue codes (distinct from spec-validation codes).
LLM_MALFORMED_JSON = "llm_malformed_json"
LLM_OUTPUT_NOT_OBJECT = "llm_output_not_object"
LLM_OUTPUT_CONTAINS_CODE = "llm_output_contains_code"
LLM_UNKNOWN_FIELD = "llm_unknown_field"
LLM_UNSUPPORTED_CHART_TYPE = "llm_unsupported_chart_type"
LLM_HALLUCINATED_FIELD = "llm_hallucinated_field"
LLM_CLIENT_ERROR = "llm_client_error"
LLM_EMPTY_OUTPUT = "llm_empty_output"

# Markers that indicate the model returned code instead of a spec.
_CODE_MARKERS = (
    "import ", "def ", "lambda ", "pd.", "df[", "df.", "```python", "```sql",
    "select ", "print(", "return ", "np.",
)

# Spec fields that may carry a chart type.
_CHART_FIELDS = ("chart_type", "chart_preference")
# Spec fields that may carry a field/dimension reference subject to hallucination
# checks (only enforced when a semantics registry is supplied).
_DIMENSION_FIELDS = ("dimension", "concentration_dimension", "migration_dimension",
                     "risk_dimension", "trajectory_dimension")


class AnthropicMIInterpreterClient(Protocol):
    """Thin, mockable boundary to the Anthropic API.

    The only contract the adapter depends on: given a fully-built prompt, return
    the model's raw completion as a JSON string (or an already-parsed dict).
    Implementations own all transport, auth, model selection, and retries.
    """

    def complete_mi_spec_json(self, prompt: str) -> Union[str, dict]:
        ...


class AnthropicClient:
    """Real Anthropic-backed client. Never imported or constructed in tests.

    The ``anthropic`` SDK is imported lazily inside :meth:`complete_mi_spec_json`
    so the dependency stays optional and the module never touches the network at
    import time.
    """

    def __init__(self, *, model: str = "claude-opus-4-8", api_key: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> None:
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete_mi_spec_json(self, prompt: str) -> str:  # pragma: no cover - networked
        import importlib

        sdk = importlib.import_module("anthropic")  # lazy, optional, not in tests
        client = sdk.Anthropic(api_key=self._api_key) if self._api_key \
            else sdk.Anthropic()
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [getattr(block, "text", "") for block in message.content]
        return "".join(parts)


# --------------------------------------------------------------------------- #
# Safe parsing of the model's raw output.
# --------------------------------------------------------------------------- #


def _strip_fences(text: str) -> str:
    """Remove ```json / ``` markdown fences while keeping the inner payload."""
    fence = re.match(r"\s*```[a-zA-Z0-9]*\s*(.*?)\s*```\s*$", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _first_balanced_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` block in *text* (ignoring braces in
    strings), or ``None`` if there isn't a complete one."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_spec_json(raw: Union[str, dict, None]
                    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse a model completion into a spec dict.

    Returns ``(obj_or_None, issues)``. Never raises. A dict passes through; a list
    or scalar is rejected; markdown fences and surrounding prose are stripped;
    output that looks like code is reported as such.
    """
    issues: List[Dict[str, Any]] = []

    if raw is None:
        return None, [make_issue(LLM_EMPTY_OUTPUT, ERROR,
                                 "model returned no output")]
    if isinstance(raw, dict):
        return raw, issues
    if isinstance(raw, (list, tuple)):
        return None, [make_issue(LLM_OUTPUT_NOT_OBJECT, ERROR,
                                 "model returned a list, expected a single JSON object")]
    if not isinstance(raw, str):
        return None, [make_issue(LLM_OUTPUT_NOT_OBJECT, ERROR,
                                 f"model returned {type(raw).__name__}, expected JSON")]

    text = raw.strip()
    if not text:
        return None, [make_issue(LLM_EMPTY_OUTPUT, ERROR,
                                 "model returned empty output")]

    candidate = _strip_fences(text)

    # Direct parse first.
    parsed = _try_load(candidate)
    if parsed is _SENTINEL:
        # Try to extract a balanced object embedded in prose.
        block = _first_balanced_object(candidate)
        if block is not None:
            parsed = _try_load(block)

    if parsed is _SENTINEL:
        if _looks_like_code(candidate):
            return None, [make_issue(
                LLM_OUTPUT_CONTAINS_CODE, ERROR,
                "model returned code/markup instead of a JSON spec")]
        return None, [make_issue(LLM_MALFORMED_JSON, ERROR,
                                 "model output was not valid JSON")]

    if isinstance(parsed, dict):
        return parsed, issues
    if isinstance(parsed, list):
        return None, [make_issue(LLM_OUTPUT_NOT_OBJECT, ERROR,
                                 "model returned a JSON list, expected a single object")]
    return None, [make_issue(LLM_OUTPUT_NOT_OBJECT, ERROR,
                             "model returned a JSON scalar, expected an object")]


_SENTINEL = object()


def _try_load(text: str):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return _SENTINEL


def _looks_like_code(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _CODE_MARKERS)


# --------------------------------------------------------------------------- #
# Post-parse governance checks (before normalise + validate).
# --------------------------------------------------------------------------- #


def _is_clarification(obj: Dict[str, Any]) -> bool:
    return bool(obj.get("clarification_required"))


def _check_unknown_fields(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    allowed = set(ALLOWED_SPEC_FIELDS) | set(CLARIFICATION_FIELDS)
    issues: List[Dict[str, Any]] = []
    for key in obj:
        if key not in allowed:
            # from_dict drops unknown keys, so this is a warning, not fatal.
            issues.append(make_issue(
                LLM_UNKNOWN_FIELD, WARNING,
                f"model emitted unknown field {key!r}; ignored", field=key))
    return issues


def _check_chart_type(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for fld in _CHART_FIELDS:
        val = obj.get(fld)
        if val is not None and val not in CHART_TYPES:
            issues.append(make_issue(
                LLM_UNSUPPORTED_CHART_TYPE, ERROR,
                f"{fld}={val!r} is not an allowed chart type "
                f"{sorted(CHART_TYPES)}", field=fld))
    return issues


def _check_hallucinated_fields(obj: Dict[str, Any],
                               semantics: Optional[dict]) -> List[Dict[str, Any]]:
    """Flag dimension references that name no known semantic field.

    Only enforced when a semantics registry is supplied; the known universe is
    the semantic field keys plus the quantile bucket dimensions. Bare ambiguous
    terms are left to ``validate_query_spec`` (ambiguous_dimension).
    """
    if not semantics:
        return []
    known = set((semantics.get("fields") or {}).keys()) | set(QUANTILE_DIMENSIONS)
    issues: List[Dict[str, Any]] = []
    for fld in _DIMENSION_FIELDS:
        val = obj.get(fld)
        if isinstance(val, str) and val and val not in known:
            issues.append(make_issue(
                LLM_HALLUCINATED_FIELD, ERROR,
                f"{fld}={val!r} is not a known semantic field", field=fld))
    return issues


# --------------------------------------------------------------------------- #
# Public adapter entry points.
# --------------------------------------------------------------------------- #


def interpret_from_llm_output(question: str, raw: Union[str, dict, None],
                              context: Optional[InterpreterContext] = None,
                              *, semantics: Optional[dict] = None,
                              confidence: float = 0.7) -> InterpretationResult:
    """Build an :class:`InterpretationResult` from an already-obtained model
    completion *raw*. Directly testable with canned strings/dicts — no client,
    no network. Safe by construction: any adapter ERROR forces a clarification.
    """
    ctx = context or InterpreterContext()
    obj, parse_issues = parse_spec_json(raw)

    # Hard parse failure → clarify, never execute.
    if obj is None:
        return _clarify_from_issues(
            question, parse_issues,
            "I couldn't read a valid query specification from the model. "
            "Could you rephrase the question?")

    # Explicit clarification object from the model.
    if _is_clarification(obj):
        ask = obj.get("clarification_question") or \
            "Could you clarify what you'd like to see?"
        return InterpretationResult(
            raw_question=question, candidate_spec=obj, normalized_spec=None,
            validation_result=None, confidence=None, issues=list(parse_issues),
            clarification_required=True, clarification_question=ask,
            interpretation_method=LLM_STUB)

    # Governance checks before touching the runtime contract.
    adapter_issues = list(parse_issues)
    adapter_issues += _check_unknown_fields(obj)
    adapter_issues += _check_chart_type(obj)
    adapter_issues += _check_hallucinated_fields(obj, semantics)

    # Fill in context anchors the model is not allowed to invent but the spec
    # needs to be runnable (client id / route default).
    spec_dict = dict(obj)
    spec_dict.setdefault("snapshot_client_id", ctx.snapshot_client_id)
    spec_dict.setdefault("route_id", ctx.route_id)

    spec = MIQuerySpec.from_dict(spec_dict).normalized()
    vr = validate_query_spec(spec, semantics=semantics)

    combined = adapter_issues + list(vr.issues)
    has_adapter_error = any(i.get("severity") == ERROR for i in adapter_issues)

    if has_adapter_error or not vr.ok:
        # Never execute an unsafe/invalid spec: ask for clarification instead.
        ask = _clarification_text(adapter_issues, vr)
        return InterpretationResult(
            raw_question=question, candidate_spec=obj, normalized_spec=spec,
            validation_result=vr, confidence=None, issues=combined,
            clarification_required=True, clarification_question=ask,
            interpretation_method=LLM_STUB)

    return InterpretationResult(
        raw_question=question, candidate_spec=obj, normalized_spec=spec,
        validation_result=vr, confidence=confidence, issues=combined,
        clarification_required=False, interpretation_method=LLM_STUB)


def interpret_with_anthropic(question: str,
                             context: Optional[InterpreterContext],
                             client: AnthropicMIInterpreterClient,
                             *, semantics: Optional[dict] = None,
                             confidence: float = 0.7) -> InterpretationResult:
    """Interpret *question* into a governed MIQuerySpec v2 using *client*.

    Builds the constrained prompt, calls the supplied (mockable) client, then
    parses + normalises + validates the result. The LLM only proposes a spec;
    all analytics remain deterministic downstream. Client failures are captured
    as a structured clarification, never raised.
    """
    ctx = context or InterpreterContext()
    prompt = build_mi_spec_prompt(question, ctx, semantics=semantics)
    try:
        raw = client.complete_mi_spec_json(prompt)
    except Exception as exc:  # noqa: BLE001 - boundary: never leak transport errors
        issue = make_issue(LLM_CLIENT_ERROR, ERROR,
                           f"Anthropic client error: {exc}")
        return _clarify_from_issues(
            question, [issue],
            "I wasn't able to reach the interpreter. Please try again.")
    return interpret_from_llm_output(question, raw, ctx, semantics=semantics,
                                     confidence=confidence)


# --------------------------------------------------------------------------- #
# Clarification helpers.
# --------------------------------------------------------------------------- #


def _clarify_from_issues(question: str, issues: List[Dict[str, Any]],
                         ask: str) -> InterpretationResult:
    return InterpretationResult(
        raw_question=question, candidate_spec=None, normalized_spec=None,
        validation_result=None, confidence=None, issues=list(issues),
        clarification_required=True, clarification_question=ask,
        interpretation_method=LLM_STUB)


def _clarification_text(adapter_issues: List[Dict[str, Any]], vr) -> str:
    msgs = [i["message"] for i in adapter_issues if i.get("severity") == ERROR]
    if vr is not None:
        msgs += [i["message"] for i in vr.issues if i.get("severity") == ERROR]
    if msgs:
        return ("I couldn't build a valid query: " + "; ".join(msgs)
                + ". Could you clarify?")
    return "Could you clarify the question?"
