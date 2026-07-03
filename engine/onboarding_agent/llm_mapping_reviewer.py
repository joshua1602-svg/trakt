"""
llm_mapping_reviewer.py
=======================

PART 4 / 5 / 6 / 7 — targeted, bounded, low-cost LLM interpreter for *mapping
uncertainties only*.

Design principles (the LLM helps interpret uncertainty, it is NOT the onboarding
engine):

  * Deterministic-first. Only unresolved / ambiguous, in-scope items are ever
    considered. Items already resolved above the deterministic-confidence
    threshold, with a high value-match rate, or out of scope, are skipped with
    zero token spend.
  * Compact prompts. The model is shown ONLY a small, redacted, per-item
    catalogue (column name, inferred type, <=3 redacted sample values and a
    short list of candidate canonical fields with category / core flag). It is
    NEVER shown full rows, full documents, the full registry, large source
    files, unrelated columns, unredacted PII or raw document text.
  * Hard budgets. Caps on calls, items per call, total prompt chars and output
    tokens per run. Excess uncertainty is converted to user gap questions
    instead of unbounded token spend.
  * Suggestion-only. Output is parsed as data and must pass deterministic
    validation; no final mapping/config is ever written by the LLM.
  * Mockable seam. A live Claude call is optional; ``llm_callable`` lets tests
    drive deterministic mock behaviour with no network / API key.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .llm_policy import LLMPolicy
from .onboarding_models import GapQuestion, MappingAmbiguity, MappingCandidate

# Reuse the MI Agent's token/cost estimator (provider/config pattern reuse only).
try:  # pragma: no cover - import guard
    from mi_agent.llm_query_parser import estimate_cost as _mi_estimate_cost
except Exception:  # pragma: no cover
    _mi_estimate_cost = None


_SYSTEM_INSTRUCTIONS = (
    "You help interpret AMBIGUOUS or UNMAPPED data-onboarding source columns.\n"
    "For each item you are given the source column name, an inferred type, a few\n"
    "REDACTED sample values and a short list of CANDIDATE canonical fields (with\n"
    "category and core flag). Choose the most likely canonical field FROM THE\n"
    "CANDIDATES ONLY.\n"
    "RULES:\n"
    "1. recommended_canonical_field MUST be one of the provided candidate fields.\n"
    "2. Return a confidence in [0,1] and a one-sentence rationale.\n"
    "3. You are a SUGGESTION engine only; a human reviews every suggestion.\n"
    "4. Output STRICT JSON only — no prose, no markdown fences.\n"
)


# --------------------------------------------------------------------------- #
# Item construction (compact, redacted) — what the LLM is allowed to see.
# --------------------------------------------------------------------------- #


def _profile_index(column_profiles) -> Dict[Tuple[str, str], Any]:
    idx: Dict[Tuple[str, str], Any] = {}
    for p in column_profiles or []:
        idx[(getattr(p, "file_name", ""), getattr(p, "source_column", ""))] = p
    return idx


def _field_meta(registry_fields: dict, field_name: str) -> dict:
    return (registry_fields or {}).get(field_name, {}) or {}


def _short_desc(meta: dict) -> str:
    desc = str(meta.get("business_description") or meta.get("display_name") or "")
    return desc[:80]


def _candidate_fields_for(
    cand: MappingCandidate, field_scope, registry_fields: dict, mode: str,
    regulatory_reporting_enabled: bool,
) -> List[Dict[str, Any]]:
    """Build the mode-filtered candidate field list shown to the LLM (PART 7)."""
    seen: List[str] = []
    raw: List[Tuple[str, float]] = []
    if cand.candidate_canonical_field:
        raw.append((cand.candidate_canonical_field, cand.confidence))
    for alt in cand.alternative_candidates or []:
        f = alt.get("field")
        if f:
            raw.append((f, float(alt.get("confidence", 0.0))))

    out: List[Dict[str, Any]] = []
    for f, conf in raw:
        if f in seen or not f:
            continue
        category = field_scope.category_of(f) if field_scope else ""
        core = bool(field_scope and f in getattr(field_scope, "core_canonical_fields", set()))
        # Mode-specific exclusion of what the LLM may even consider.
        if field_scope is not None and field_scope.is_excluded(f):
            continue
        if mode == "mi_only" and category == "regulatory" and not core:
            # mi_only never spends tokens on regulatory non-core fields.
            continue
        if (mode == "warehouse_securitisation" and category == "regulatory"
                and not core and not regulatory_reporting_enabled):
            continue
        seen.append(f)
        out.append({
            "field": f,
            "category": category,
            "core_canonical": core,
            "confidence": round(float(conf), 4),
            "description": _short_desc(_field_meta(registry_fields, f)),
        })
    return out


def build_item(
    cand: MappingCandidate, field_scope, registry_fields: dict, mode: str,
    policy: LLMPolicy, profile_idx: Dict[Tuple[str, str], Any],
    regulatory_reporting_enabled: bool,
) -> Optional[Dict[str, Any]]:
    """Build a compact, redacted LLM item for one mapping candidate, or None if
    it should not be sent (no in-scope candidate fields)."""
    candidate_fields = _candidate_fields_for(
        cand, field_scope, registry_fields, mode, regulatory_reporting_enabled
    )
    if not candidate_fields:
        return None

    prof = profile_idx.get((cand.source_file, cand.source_column))
    inferred_type = getattr(prof, "inferred_type", "") if prof else ""
    normalized = getattr(prof, "normalized_column_name", "") if prof else \
        str(cand.source_column).lower().replace(" ", "_")
    samples: List[str] = []
    if policy.include_sample_values != "none":
        samples = list(cand.sample_values_redacted or [])[: policy.max_sample_values_per_field]

    return {
        "source_file": cand.source_file,
        "source_file_type": _file_type(cand.source_file),
        "source_column": cand.source_column,
        "normalized_column_name": normalized,
        "inferred_type": inferred_type,
        "redacted_sample_values": samples,
        "candidate_canonical_fields": candidate_fields,
        "mode": mode,
        "field_scope_status": "in_scope",
    }


def _file_type(file_name: str) -> str:
    suffix = Path(str(file_name)).suffix.lower().lstrip(".")
    return suffix or "unknown"


# --------------------------------------------------------------------------- #
# Selection / prioritisation (deterministic, zero token spend)
# --------------------------------------------------------------------------- #


def _priority(item: Dict[str, Any], mode: str) -> int:
    """Lower number = higher priority. Mode-specific (PART 7)."""
    cats = {c["category"] for c in item["candidate_canonical_fields"]}
    core = any(c["core_canonical"] for c in item["candidate_canonical_fields"])
    warehouse_like = any(
        any(k in c["field"] for k in ("warehouse", "facility", "advance", "funding",
                                      "pipeline", "cashflow", "drawdown"))
        for c in item["candidate_canonical_fields"]
    )
    if core:
        return 0
    if mode == "regulatory_mi" and "regulatory" in cats:
        return 1
    if mode == "warehouse_securitisation" and warehouse_like:
        return 1
    if "analytics" in cats:
        return 2
    return 3


def select_items(
    items: List[Dict[str, Any]], policy: LLMPolicy, mode: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Prioritise and cap items for the LLM. Returns
    ``(selected, overflow, meta)`` where ``overflow`` items become user gap
    questions instead of token spend (PART 6)."""
    ordered = sorted(items, key=lambda it: _priority(it, mode))
    if policy.prioritise_core_blocking_only:
        ordered = [it for it in ordered if _priority(it, mode) <= 1] or ordered

    capacity = max(0, int(policy.max_llm_calls_per_run) * int(policy.max_items_per_call))
    over_budget = len(items) > policy.unresolved_items_budget

    selected = ordered[:capacity]
    overflow = ordered[capacity:]

    meta = {
        "total_unresolved_in_scope": len(items),
        "uncertainty_budget": policy.unresolved_items_budget,
        "over_uncertainty_budget": over_budget,
        "llm_capacity": capacity,
        "selected_for_llm": len(selected),
        "overflow_to_gap_questions": len(overflow),
    }
    return selected, overflow, meta


# --------------------------------------------------------------------------- #
# Prompt building (compact, data-free except redacted samples)
# --------------------------------------------------------------------------- #


def build_prompt(batch: List[Dict[str, Any]], mode: str) -> Dict[str, str]:
    payload = {"mode": mode, "items": batch}
    user = (
        "Interpret the following ambiguous/unmapped onboarding columns.\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nReturn STRICT JSON: {\"llm_mapping_suggestions\": [ {"
        "\"source_file\":..., \"source_column\":..., "
        "\"recommended_canonical_field\":..., \"confidence\":0-1, "
        "\"rationale\":..., \"alternatives\":[...], \"requires_review\":true } ] }"
    )
    return {"system": _SYSTEM_INSTRUCTIONS, "user": user}


# --------------------------------------------------------------------------- #
# Deterministic validation of LLM suggestions (PART 4)
# --------------------------------------------------------------------------- #


def validate_suggestion(
    suggestion: Dict[str, Any], item: Dict[str, Any], field_scope, mode: str,
    registry_fields: dict, min_confidence: float = 0.0,
) -> Tuple[bool, str]:
    """Validate one LLM suggestion deterministically. Returns (ok, reason)."""
    field_name = suggestion.get("recommended_canonical_field", "")
    if not field_name:
        return False, "no recommended_canonical_field"
    # Must be one of the candidate fields offered for this item.
    candidate_names = {c["field"] for c in item["candidate_canonical_fields"]}
    if field_name not in candidate_names:
        return False, f"recommended field '{field_name}' was not an offered candidate"
    # Must exist in the registry.
    if registry_fields is not None and field_name not in registry_fields:
        return False, f"recommended field '{field_name}' not in registry"
    # Must be in scope for the mode.
    if field_scope is not None and field_scope.is_excluded(field_name):
        return False, f"recommended field '{field_name}' is out of scope for mode '{mode}'"
    # Regulatory non-core must never be accepted in mi_only.
    if mode == "mi_only":
        category = field_scope.category_of(field_name) if field_scope else ""
        core = bool(field_scope and field_name in getattr(field_scope, "core_canonical_fields", set()))
        if category == "regulatory" and not core:
            return False, "regulatory non-core field rejected in mi_only"
    # Confidence threshold respected.
    try:
        conf = float(suggestion.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False, "confidence not numeric"
    if conf < min_confidence:
        return False, f"confidence {conf} below minimum {min_confidence}"
    return True, "valid"


def _parse_suggestions(text: Any) -> List[Dict[str, Any]]:
    if isinstance(text, dict):
        data = text
    else:
        from engine.onboarding_agent.llm_json import extract_json

        data, _, _ = extract_json(text)

    if not isinstance(data, dict):
        return []

    suggestions = data.get("llm_mapping_suggestions", []) or []

    if not isinstance(suggestions, list):
        return []

    return [
        s for s in suggestions
        if isinstance(s, dict)
    ]


# --------------------------------------------------------------------------- #
# Provider seam (live + mock)
# --------------------------------------------------------------------------- #


def _provider_available(policy: LLMPolicy, llm_callable) -> bool:
    if llm_callable is not None:
        return True
    if policy.provider == "anthropic":
        try:
            import anthropic  # noqa: F401
        except Exception:
            return False
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return False


def _call_live(prompt: Dict[str, str], model: str) -> Tuple[str, dict]:  # pragma: no cover
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    message = client.messages.create(
        model=model, max_tokens=1024, temperature=0.0,
        system=prompt["system"],
        messages=[{"role": "user", "content": prompt["user"]}],
    )
    text = message.content[0].text
    u = getattr(message, "usage", None)
    usage = {}
    if u is not None:
        for k in ("input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens"):
            usage[k] = getattr(u, k, 0) or 0
    return text, usage


def _invoke(prompt: Dict[str, str], model: str, llm_callable) -> Tuple[Any, dict]:
    if llm_callable is not None:
        res = llm_callable(prompt)
        if isinstance(res, tuple):
            return res[0], (res[1] if len(res) > 1 else {}) or {}
        if isinstance(res, dict) and ("text" in res or "content" in res):
            return res.get("text") or res.get("content"), res.get("usage") or {}
        return res, {}
    return _call_live(prompt, model)


# --------------------------------------------------------------------------- #
# Gap-question fallback (PART 6)
# --------------------------------------------------------------------------- #


def _gap_question_for_item(item: Dict[str, Any], idx: int, mode: str) -> GapQuestion:
    candidate_names = [c["field"] for c in item["candidate_canonical_fields"]]
    priority = _priority(item, mode)
    severity = {0: "high", 1: "high", 2: "medium"}.get(priority, "info")
    return GapQuestion(
        question_id=f"LLMQ{idx}",
        category="mapping",
        severity=severity,
        question=(
            f"How should source column '{item['source_column']}' in "
            f"'{item['source_file']}' be mapped?"
        ),
        reason=(
            "Unresolved/ambiguous mapping converted to a user review question "
            "instead of additional LLM token spend (uncertainty budget)."
        ),
        candidate_answers=candidate_names,
        default_recommendation=candidate_names[0] if candidate_names else "",
        blocking_for=[],
        source_evidence=f"{item['source_file']}::{item['source_column']}",
        subject="mapping",
        subject_value=item["source_column"],
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def _empty_usage(policy: LLMPolicy, enabled: bool) -> Dict[str, Any]:
    return {
        "llm_enabled": enabled,
        "provider": policy.provider if enabled else None,
        "model": policy.model if enabled else None,
        "budget_profile": policy.profile or "default",
        "calls_attempted": 0,
        "calls_completed": 0,
        "items_sent": 0,
        "prompt_chars_estimated": 0,
        "output_tokens_estimated_or_reported": 0,
        "skipped_due_to_zero_cost_first": 0,
        "skipped_due_to_budget": 0,
        "unresolved_items_converted_to_gap_questions": 0,
        "valid_suggestions": 0,
        "rejected_suggestions": 0,
        "estimated_cost": 0,
    }


def _unresolved_candidates(
    mapping_candidates: List[MappingCandidate], policy: LLMPolicy,
) -> Tuple[List[MappingCandidate], int]:
    """Select candidates that are genuinely unresolved/ambiguous. Returns
    ``(unresolved, zero_cost_skips)`` where zero_cost_skips counts items skipped
    because the deterministic confidence already exceeds the threshold."""
    unresolved: List[MappingCandidate] = []
    zero_cost_skips = 0
    for m in mapping_candidates:
        ambiguous = bool(m.ambiguity_rule_applied)
        unmapped = not m.candidate_canonical_field
        low_conf = m.confidence <= policy.deterministic_confidence_above
        if not (ambiguous or unmapped or m.requires_review):
            continue
        if policy.zero_cost_first and not ambiguous and not unmapped and not low_conf:
            zero_cost_skips += 1
            continue
        unresolved.append(m)
    return unresolved, zero_cost_skips


def run_llm_mapping_review(
    *,
    mapping_candidates: List[MappingCandidate],
    mapping_ambiguities: List[MappingAmbiguity],
    field_scope,
    registry_fields: dict,
    mode: str,
    policy: LLMPolicy,
    regulatory_reporting_enabled: bool = False,
    column_profiles=None,
    llm_callable: Optional[Callable] = None,
    gap_question_start_index: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[GapQuestion]]:
    """Run the bounded LLM mapping review.

    Returns ``(suggestions, usage_summary, gap_questions)``. Suggestions are
    advisory only — the orchestrator never promotes them to final mappings. When
    the LLM is disabled (default) this returns ``([], {llm_enabled: False...},
    [])`` immediately with zero spend.
    """
    if not policy.enabled:
        return [], _empty_usage(policy, enabled=False), []

    usage = _empty_usage(policy, enabled=True)
    profile_idx = _profile_index(column_profiles)

    # Provider availability (PART 8): never fail the run if unavailable.
    if not _provider_available(policy, llm_callable):
        usage["llm_enabled"] = False
        usage["status"] = "llm_enabled_requested_but_provider_unavailable"
        # Convert in-scope unresolved items to gap questions so nothing is lost.
        unresolved, zc = _unresolved_candidates(mapping_candidates, policy)
        items = [
            it for it in (
                build_item(m, field_scope, registry_fields, mode, policy,
                           profile_idx, regulatory_reporting_enabled)
                for m in unresolved
            ) if it is not None
        ]
        gaps = [
            _gap_question_for_item(it, gap_question_start_index + i, mode)
            for i, it in enumerate(items)
        ]
        usage["skipped_due_to_zero_cost_first"] = zc
        usage["unresolved_items_converted_to_gap_questions"] = len(gaps)
        return [], usage, gaps

    # Deterministic-first selection.
    unresolved, zero_cost_skips = _unresolved_candidates(mapping_candidates, policy)
    usage["skipped_due_to_zero_cost_first"] = zero_cost_skips

    items = [
        it for it in (
            build_item(m, field_scope, registry_fields, mode, policy,
                       profile_idx, regulatory_reporting_enabled)
            for m in unresolved
        ) if it is not None
    ]

    selected, overflow, sel_meta = select_items(items, policy, mode)
    usage.update({
        "total_unresolved_in_scope": sel_meta["total_unresolved_in_scope"],
        "over_uncertainty_budget": sel_meta["over_uncertainty_budget"],
        "llm_capacity": sel_meta["llm_capacity"],
    })

    gap_questions: List[GapQuestion] = []
    gap_idx = gap_question_start_index
    for it in overflow:
        gap_questions.append(_gap_question_for_item(it, gap_idx, mode))
        gap_idx += 1

    # Batch + call within hard budgets.
    suggestions: List[Dict[str, Any]] = []
    item_to_suggestion: Dict[Tuple[str, str], Dict[str, Any]] = {}
    prompt_chars = 0
    output_tokens = 0
    calls_completed = 0
    items_sent = 0
    deferred_to_budget: List[Dict[str, Any]] = []

    batches = [
        selected[i:i + policy.max_items_per_call]
        for i in range(0, len(selected), max(1, policy.max_items_per_call))
    ]

    for batch in batches:
        if calls_completed >= policy.max_llm_calls_per_run:
            deferred_to_budget.extend(batch)
            continue
        prompt = build_prompt(batch, mode)
        this_chars = len(prompt["system"]) + len(prompt["user"])
        if (prompt_chars + this_chars) > policy.max_total_prompt_chars_per_run:
            deferred_to_budget.extend(batch)
            continue
        if output_tokens >= policy.max_total_output_tokens_per_run:
            deferred_to_budget.extend(batch)
            continue

        usage["calls_attempted"] += 1
        text, call_usage = _invoke(prompt, policy.model, llm_callable)
        calls_completed += 1
        prompt_chars += this_chars
        items_sent += len(batch)

        out_tok = int((call_usage or {}).get("output_tokens", 0) or 0)
        if not out_tok and _mi_estimate_cost is not None:
            est = _mi_estimate_cost(policy.model, call_usage)
            out_tok = est.get("output_tokens", 0)
        output_tokens += out_tok

        raw_suggestions = _parse_suggestions(text)
        by_col = {(it["source_file"], it["source_column"]): it for it in batch}
        for sugg in raw_suggestions:
            key = (sugg.get("source_file", ""), sugg.get("source_column", ""))
            item = by_col.get(key)
            if item is None:
                continue
            ok, reason = validate_suggestion(
                sugg, item, field_scope, mode, registry_fields,
            )
            record = {
                "source_file": sugg.get("source_file", ""),
                "source_column": sugg.get("source_column", ""),
                "recommended_canonical_field": sugg.get("recommended_canonical_field", ""),
                "confidence": sugg.get("confidence", 0.0),
                "rationale": str(sugg.get("rationale", ""))[:200],
                "alternatives": list(sugg.get("alternatives", []) or [])[:4],
                "requires_review": True,
                "validation_ok": ok,
                "validation_reason": reason,
            }
            suggestions.append(record)
            item_to_suggestion[key] = record
            if ok:
                usage["valid_suggestions"] += 1
            else:
                usage["rejected_suggestions"] += 1

    # Items the LLM could not reach (budget) become gap questions too.
    for it in deferred_to_budget:
        gap_questions.append(_gap_question_for_item(it, gap_idx, mode))
        gap_idx += 1
    usage["skipped_due_to_budget"] = len(deferred_to_budget)

    # Cost estimate (best-effort, reuse MI estimator).
    estimated_cost = 0
    if _mi_estimate_cost is not None and output_tokens:
        est = _mi_estimate_cost(policy.model, {"output_tokens": output_tokens})
        estimated_cost = est.get("estimated_total_cost", 0)

    usage.update({
        "calls_completed": calls_completed,
        "items_sent": items_sent,
        "prompt_chars_estimated": prompt_chars,
        "output_tokens_estimated_or_reported": output_tokens,
        "unresolved_items_converted_to_gap_questions": len(gap_questions),
        "estimated_cost": estimated_cost,
    })
    return suggestions, usage, gap_questions
