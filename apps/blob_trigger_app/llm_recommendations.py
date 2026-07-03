"""apps.blob_trigger_app.llm_recommendations — advisory LLM, deterministic truth.

LLM recommendations are **advisory only**. Deterministic mapping/registry stays
the production execution source of truth; the LLM never auto-applies anything and
never blocks a run. If the LLM is disabled, unavailable, rate-limited, or errors,
the run falls back to the deterministic diagnostics + operator workflow — never
to failure.

Controls (env):
    TRAKT_LLM_ENABLED = true|false        (default: false — off in Azure unless set)
    TRAKT_LLM_MODE    = advisory|off      (default: advisory when enabled)
    TRAKT_LLM_FALLBACK= deterministic     (informational; always deterministic)
    ANTHROPIC_API_KEY = <key>             (or an existing configured provider key)

Recommendations are persisted to ``trakt-state/runs/{pack_key}/llm/
recommendations.json`` with an explicit ``approval_required`` + ``status`` so an
operator must approve/promote before anything becomes production-active.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

# When to consider generating recommendations, by routing decision.
_DECISION_NEW_SOURCE = "source_onboarding"
_DECISION_SCHEMA_DRIFT = "schema_drift"
_DECISION_DETERMINISTIC = "deterministic"

# A generator takes a context dict and returns a list of recommendation dicts.
LLMGenerator = Callable[[Dict[str, Any]], List[Dict[str, Any]]]


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def resolve_llm_policy(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Resolve the LLM policy from env. Never raises.

    ``enabled`` — TRAKT_LLM_ENABLED true AND TRAKT_LLM_MODE != off.
    ``available`` — enabled AND a provider key is present.
    """
    env = env if env is not None else os.environ
    enabled_flag = _truthy(env.get("TRAKT_LLM_ENABLED"))
    mode = (env.get("TRAKT_LLM_MODE") or ("advisory" if enabled_flag else "off")).strip().lower()
    fallback = (env.get("TRAKT_LLM_FALLBACK") or "deterministic").strip().lower()
    enabled = enabled_flag and mode != "off"
    # ``resolving`` additionally wires the agentic mapping RESOLVER into the
    # automated path (a new/changed source gets a pre-filled mapping instead of an
    # empty review queue). ``advisory`` keeps the LLM advisory-only. Both still
    # require an operator one-click for a new source / material change.
    resolve_mapping = bool(enabled and mode == "resolving")
    key = env.get("ANTHROPIC_API_KEY") or env.get("TRAKT_LLM_API_KEY")
    available = bool(enabled and key)
    if not enabled_flag:
        reason = "TRAKT_LLM_ENABLED not set/false — deterministic only"
    elif mode == "off":
        reason = "TRAKT_LLM_MODE=off — deterministic only"
    elif not key:
        reason = "LLM enabled but no ANTHROPIC_API_KEY — deterministic fallback"
    elif resolve_mapping:
        reason = "LLM enabled (resolving) with provider key present"
    else:
        reason = "LLM enabled (advisory) with provider key present"
    return {
        "enabled": enabled,
        "mode": mode,
        "resolve_mapping": resolve_mapping,
        "available": available,
        "fallback": fallback,
        "reason": reason,
        "provider": "anthropic",
        "model": env.get("TRAKT_LLM_MODEL") or "claude-haiku-4-5-20251001",
        "has_key": bool(key),
    }


def should_generate(decision: Optional[str], *, gate_failed: bool,
                    policy: Dict[str, Any]) -> bool:
    """Advisory recommendations are considered for a NEW source, SCHEMA DRIFT, or a
    known source whose gate is blocked (incomplete handoff/transform). A CLEAN
    recurring known source never triggers an LLM call."""
    if not policy.get("enabled"):
        return False
    if decision in (_DECISION_NEW_SOURCE, _DECISION_SCHEMA_DRIFT):
        return True
    if decision == _DECISION_DETERMINISTIC and gate_failed:
        return True
    return False


def _candidate_fields(gates: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Deterministic candidate context for the recommender: the unresolved /
    missing / affected fields the operator would otherwise chase by hand."""
    src: List[str] = []
    tgt: List[str] = []
    for g in gates or []:
        pl = g.get("payload") or {}
        tgt += pl.get("missing_target_fields") or []
        tgt += pl.get("unresolved_fields") or []
        tgt += g.get("affected_fields") or []
    return {"source_fields": sorted(set(src)), "target_fields": sorted(set(tgt))}


def generate_recommendations(
    *,
    pack_key: str,
    decision: Optional[str],
    gates: List[Dict[str, Any]],
    gate_failed: bool,
    generator: Optional[LLMGenerator] = None,
    policy: Optional[Dict[str, Any]] = None,
    now: str = "",
) -> "tuple[List[Dict[str, Any]], Dict[str, Any]]":
    """Return (recommendations, meta). Advisory-only; never raises.

    ``meta`` carries the run-record fields: llm_enabled / llm_invoked /
    llm_available / llm_reason / llm_error / recommendations_present /
    deterministic_fallback_used.
    """
    policy = policy or resolve_llm_policy()
    meta: Dict[str, Any] = {
        "llm_enabled": bool(policy.get("enabled")),
        "llm_available": bool(policy.get("available")),
        "llm_invoked": False,
        "llm_reason": policy.get("reason"),
        "llm_error": None,
        "recommendations_present": False,
        "deterministic_fallback_used": True,   # deterministic is always the base
        "model": policy.get("model"),
        "provider": policy.get("provider"),
    }
    if not should_generate(decision, gate_failed=gate_failed, policy=policy):
        meta["llm_reason"] = (
            "clean recurring known source — deterministic only"
            if decision == _DECISION_DETERMINISTIC and not gate_failed
            else meta["llm_reason"])
        return [], meta
    if not policy.get("available"):
        # Enabled but key missing / off → deterministic fallback, not failure.
        return [], meta

    ctx = {"pack_key": pack_key, "decision": decision, **_candidate_fields(gates)}
    try:
        meta["llm_invoked"] = True
        raw = (generator or _default_generator)(ctx)
        recs = _normalise_recommendations(raw, model=policy.get("model"),
                                          provider=policy.get("provider"), now=now)
        meta["recommendations_present"] = bool(recs)
        # LLM ran, but deterministic remains the execution source of truth.
        meta["deterministic_fallback_used"] = not recs
        return recs, meta
    except Exception as exc:  # noqa: BLE001 — advisory must never fail the run
        meta["llm_error"] = f"{type(exc).__name__}: {exc}"
        meta["llm_reason"] = "LLM error — deterministic fallback"
        meta["deterministic_fallback_used"] = True
        return [], meta


def _default_generator(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Real provider call. Constructed lazily; if the SDK/key is unavailable it
    raises and the caller falls back deterministically. (No network in tests —
    tests inject a generator.)"""
    from engine.onboarding_agent.cli import _build_mapping_llm_callable  # type: ignore

    callable_ = _build_mapping_llm_callable("low")
    if callable_ is None:
        raise RuntimeError("no LLM callable available")

    prompt = (
        "Propose advisory canonical mappings/defaults for these unresolved "
        f"target fields: {ctx.get('target_fields')}. Return JSON list of "
        "{target_field, recommended_mapping, rationale, confidence}."
    )

    text = callable_(prompt)

    if isinstance(text, str):
        from engine.onboarding_agent.llm_json import extract_json_list

        data, _, _ = extract_json_list(text)
    else:
        data = text

    if isinstance(data, dict):
        data = (
            data.get("recommendations")
            or data.get("llm_mapping_suggestions")
            or data.get("proposals")
            or data.get("mappings")
            or data.get("rows")
            or []
        )

    if not isinstance(data, list):
        return []

    return [
        row for row in data
        if isinstance(row, dict)
    ]


def _normalise_recommendations(raw: Any, *, model: str, provider: str,
                               now: str) -> List[Dict[str, Any]]:
    rows = raw if isinstance(raw, list) else (raw or {}).get("rows", [])
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        out.append({
            "target_field": r.get("target_field") or r.get("field") or "",
            "source_field": r.get("source_field") or r.get("source_column") or "",
            "recommended_mapping": (r.get("recommended_mapping") or r.get("mapping")
                                    or r.get("recommended_default") or r.get("derivation") or ""),
            "rationale": r.get("rationale") or r.get("reason") or "",
            "confidence": r.get("confidence"),
            "approval_required": True,          # advisory — never auto-applied
            "status": "pending",                # pending | approved | rejected | superseded
        })
    return out


def build_recommendations_doc(recommendations: List[Dict[str, Any]],
                              meta: Dict[str, Any], *,
                              pack_key: str, now: str) -> Dict[str, Any]:
    """The persisted ``recommendations.json`` document."""
    return {
        "pack_key": pack_key,
        "generated_at": now,
        "provider": meta.get("provider"),
        "model": meta.get("model"),
        "llm_invoked": meta.get("llm_invoked"),
        "llm_error": meta.get("llm_error"),
        "advisory_only": True,
        "deterministic_is_source_of_truth": True,
        "recommendations": recommendations,
    }
