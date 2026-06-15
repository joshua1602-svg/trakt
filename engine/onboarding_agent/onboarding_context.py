"""
onboarding_context.py
=====================

v1 asset / regime / use-case detector.

Infers the onboarding *context* (asset class, jurisdiction, product type,
reporting regime, use cases, required domains) from deterministic evidence —
file names, sheet names, column names, sample values and the selected mode — so
the mapping resolver can target an asset/regime-specific REQUIRED data contract
rather than a generic field universe.

Artefacts:
    27_onboarding_context.json
    27_onboarding_context_summary.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Known named target contracts (selected from the FINAL context, not mode alone).
KNOWN_TARGET_CONTRACTS = {
    "uk_equity_release_mi_v1": {"asset_class": "equity_release_mortgage",
                                "jurisdiction": "UK"},
    "uk_equity_release_esma_annex12_v1": {"asset_class": "equity_release_mortgage",
                                          "regime": "esma_annex_12"},
}
CONFIDENCE_THRESHOLD = 0.6

# Token signals -> asset class.
_ASSET_SIGNALS = {
    "equity_release_mortgage": ("equity release", "ere", "lifetime mortgage", "lifetime",
                                "drawdown", "lump sum", "rio", "erm", "roll-up", "rollup"),
    "residential_mortgage": ("residential mortgage", "rmbs", "btl", "buy to let",
                             "owner occupier", "repayment mortgage"),
    "consumer_loan": ("personal loan", "consumer", "unsecured"),
    "sme_loan": ("sme", "commercial loan", "business loan"),
}
_PRODUCT_SIGNALS = {
    "lifetime_mortgage": ("lifetime mortgage", "lifetime"),
    "drawdown": ("drawdown",),
    "lump_sum": ("lump sum",),
}
_REGIME_BY_MODE = {
    "mi_only": "mi_only",
    "mna_dd": "mna_due_diligence",
    "regulatory_mi": "esma_annex_12",
    "warehouse_securitisation": "warehouse_reporting",
}
# Domain signals from file/column tokens.
_DOMAIN_SIGNALS = {
    "pipeline": ("kfi", "pipeline", "application", "offer", "broker", "dpr"),
    "funded_loan": ("loan", "account", "policy number", "balance", "interest rate",
                    "original principal", "maturity", "origination"),
    "collateral_property": ("property", "valuation", "post code", "postcode", "ltv",
                            "region", "collateral"),
    "cashflow_ledger": ("principal and interest", "b/f", "c/f", "payment_allocation",
                        "payment allocation", "redemption", "cash paid", "ledger",
                        "balance movement", "arrears"),
    "concentration_limits": ("concentration", "schedule 8", "limit", "covenant"),
}
_UK_SIGNALS = ("uk", "gbp", "£", "post code", "postcode", "sonia", "fca", "esma",
               "england", "scotland", "wales")


def _tokens(strings: List[str]) -> str:
    return " \n ".join(str(s).lower() for s in strings if s)


def detect_context(
    inventory: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    mode: str = "mi_only",
    client_name: str = "",
    document_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Infer the onboarding context from deterministic evidence."""
    file_names = [i.get("file_name", "") for i in inventory]
    columns = [e.get("source_column", "") for e in evidence_rows]
    samples = []
    for e in evidence_rows[:300]:
        samples.append(str(e.get("sample_values_distinct_redacted", "")))
    blob = _tokens(file_names + columns + samples + [client_name] + (document_terms or []))

    def score(signals):
        return {k: sum(blob.count(t) for t in toks) for k, toks in signals.items()}

    asset_scores = score(_ASSET_SIGNALS)
    asset_class = max(asset_scores, key=asset_scores.get) if any(asset_scores.values()) \
        else "equity_release_mortgage"
    prod_scores = score(_PRODUCT_SIGNALS)
    product_type = max(prod_scores, key=prod_scores.get) if any(prod_scores.values()) else "lifetime_mortgage"

    jurisdiction = "UK" if any(t in blob for t in _UK_SIGNALS) else "unknown"
    reporting_regime = _REGIME_BY_MODE.get(mode, mode)

    dom_scores = score(_DOMAIN_SIGNALS)
    required_domains = sorted([d for d, s in dom_scores.items() if s > 0])
    if not required_domains:
        required_domains = ["funded_loan", "collateral_property", "cashflow_ledger", "pipeline"]

    use_cases = ["portfolio_mi"]
    if "cashflow_ledger" in required_domains:
        use_cases.append("cashflow_monitoring")
    if "funded_loan" in required_domains or "collateral_property" in required_domains:
        use_cases.append("static_pools")
    if mode in ("warehouse_securitisation", "regulatory_mi") or "pipeline" in required_domains:
        use_cases.append("securitisation_readiness")

    # Confidence: how strong + consistent the signals are.
    asset_strength = max(asset_scores.values()) if asset_scores else 0
    confidence = round(min(1.0, 0.4 + 0.1 * asset_strength + 0.1 * len(required_domains)
                           + (0.1 if jurisdiction == "UK" else 0)), 2)
    rationale = (
        f"asset_class={asset_class} (signal strength {asset_strength}); "
        f"jurisdiction={jurisdiction}; product={product_type}; "
        f"regime from mode '{mode}'; domains from file/column tokens: "
        f"{', '.join(required_domains)}.")

    return {
        "asset_class": asset_class,
        "jurisdiction": jurisdiction,
        "product_type": product_type,
        "reporting_regime": reporting_regime,
        "use_cases": sorted(set(use_cases)),
        "required_domains": required_domains,
        "confidence": confidence,
        "needs_user_confirmation": confidence < 0.6,
        "rationale": rationale,
        "mode": mode,
        "client_name": client_name,
    }


# ---------------------------------------------------------------------------
# LLM context resolver (semantic) + deterministic backstop
# ---------------------------------------------------------------------------

_CONTEXT_PROMPT = """\
You infer the ONBOARDING CONTEXT for a lender data pack. Use ONLY the provided
compact evidence (file names, columns, sample values, deterministic guess). Return
STRUCTURED JSON with keys: asset_class, jurisdiction, product_type,
reporting_regime, use_cases, required_domains, suggested_target_contract,
confidence (0-1), rationale, supporting_evidence (list), open_questions (list).
Do not invent terms not supported by the evidence.
"""


def build_context_evidence(
    inventory: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    deterministic_guess: Dict[str, Any],
    document_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compact, redacted evidence package for the LLM context resolver."""
    return {
        "file_names": [i.get("file_name", "") for i in inventory][:40],
        "column_names": [e.get("source_column", "") for e in evidence_rows][:120],
        "sample_values": [str(e.get("sample_values_distinct_redacted", ""))
                          for e in evidence_rows[:40]],
        "document_terms": (document_terms or [])[:20],
        "deterministic_context_guess": {
            k: deterministic_guess.get(k) for k in
            ("asset_class", "jurisdiction", "product_type", "reporting_regime",
             "required_domains", "confidence")},
        "known_target_contracts": list(KNOWN_TARGET_CONTRACTS.keys()),
    }


def resolve_context_with_llm(
    package: Dict[str, Any], llm_callable: Callable[[str], str]
) -> Dict[str, Any]:
    """Call the LLM context resolver. Returns {"context": {...}, "usage": {...}}."""
    import uuid
    batch_id = "ctx_" + uuid.uuid4().hex[:10]
    prompt = _CONTEXT_PROMPT + "\nEVIDENCE = " + json.dumps(package, default=str)
    raw = llm_callable(prompt)
    usage = {"llm_enabled": True, "calls_completed": 1, "estimated_cost_gbp": 0.01,
             "llm_batch_id": batch_id, "prompt_chars": len(prompt)}
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    parsed["llm_batch_id"] = batch_id
    return {"context": parsed, "usage": usage}


def _token_blob(inventory, evidence_rows, document_terms) -> str:
    files = [i.get("file_name", "") for i in inventory]
    cols = [e.get("source_column", "") for e in evidence_rows]
    samples = [str(e.get("sample_values_distinct_redacted", "")) for e in evidence_rows[:300]]
    return _tokens(files + cols + samples + (document_terms or []))


def backstop_context(
    deterministic_guess: Dict[str, Any],
    llm_context: Optional[Dict[str, Any]],
    inventory: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    mode: str,
    document_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate the (LLM or deterministic) context against deterministic evidence.

    Never blindly accepts LLM context. Produces the FINAL context plus
    context_backstop_decision / reason / final_context_source.
    """
    blob = _token_blob(inventory, evidence_rows, document_terms)
    proposed = dict(llm_context) if llm_context else None
    base = proposed or deterministic_guess
    asset = base.get("asset_class", deterministic_guess.get("asset_class"))
    regime = base.get("reporting_regime", deterministic_guess.get("reporting_regime"))
    domains = base.get("required_domains") or deterministic_guess.get("required_domains", [])
    confidence = float(base.get("confidence", deterministic_guess.get("confidence", 0.0)) or 0.0)
    contract_name = base.get("suggested_target_contract", "")

    # Deterministic checks.
    asset_terms = _ASSET_SIGNALS.get(asset, ())
    asset_supported = any(t in blob for t in asset_terms) or asset == deterministic_guess.get("asset_class")
    juris = base.get("jurisdiction", deterministic_guess.get("jurisdiction"))
    juris_supported = juris != "UK" or any(t in blob for t in _UK_SIGNALS)
    domains_supported = sum(1 for d in domains
                            if any(t in blob for t in _DOMAIN_SIGNALS.get(d, ()))) >= max(1, len(domains) // 2)
    regime_ok = _REGIME_BY_MODE.get(mode, mode) == regime or not proposed
    contract_exists = (not contract_name) or contract_name in KNOWN_TARGET_CONTRACTS
    conflict = bool(proposed and proposed.get("asset_class")
                    and deterministic_guess.get("asset_class")
                    and proposed["asset_class"] != deterministic_guess["asset_class"]
                    and not asset_supported)
    threshold_met = confidence >= CONFIDENCE_THRESHOLD

    checks = {
        "asset_class_supported_by_columns": asset_supported,
        "jurisdiction_supported_by_terms": juris_supported,
        "required_domains_supported_by_files": domains_supported,
        "contract_exists": contract_exists,
        "regime_compatible_with_mode": regime_ok,
        "confidence_threshold_met": threshold_met,
        "conflict_detected": conflict,
    }

    # Decide.
    if proposed is None:
        decision, source, reason = ("deterministic_only", "deterministic",
                                    "LLM context resolver not run")
        final = dict(deterministic_guess)
    elif conflict or not asset_supported or not domains_supported:
        decision, source = "downgraded_to_deterministic", "deterministic"
        reason = ("LLM asset/domains not supported by evidence; using deterministic guess "
                  "and asking the user")
        final = dict(deterministic_guess)
    elif not threshold_met:
        decision, source = "needs_user_confirmation", "llm"
        reason = f"LLM confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}"
        final = dict(proposed)
    else:
        decision, source = "accepted_llm", "llm"
        reason = "LLM context validated against deterministic evidence"
        final = dict(proposed)

    # Normalise final context shape.
    final.setdefault("asset_class", deterministic_guess["asset_class"])
    final.setdefault("jurisdiction", deterministic_guess["jurisdiction"])
    final.setdefault("product_type", deterministic_guess["product_type"])
    final.setdefault("reporting_regime", deterministic_guess["reporting_regime"])
    final.setdefault("use_cases", deterministic_guess["use_cases"])
    final.setdefault("required_domains", deterministic_guess["required_domains"])
    final.setdefault("confidence", confidence)
    final.setdefault("rationale", deterministic_guess.get("rationale", ""))
    final["mode"] = mode
    final["context_backstop_decision"] = decision
    final["context_backstop_reason"] = reason
    final["context_backstop_checks"] = checks
    final["final_context_source"] = source
    final["needs_user_confirmation"] = decision in (
        "needs_user_confirmation", "downgraded_to_deterministic") or confidence < CONFIDENCE_THRESHOLD
    final["open_questions"] = (proposed.get("open_questions", []) if proposed else []) or (
        ["Confirm asset class / reporting regime / required outputs."]
        if final["needs_user_confirmation"] else [])
    final["supporting_evidence"] = (proposed.get("supporting_evidence", []) if proposed else [])
    final["selected_target_contract"] = select_target_contract(final)
    return final


def select_target_contract(context: Dict[str, Any]) -> str:
    """Select a named target contract from the FINAL context (not mode alone)."""
    suggested = context.get("suggested_target_contract", "")
    if suggested in KNOWN_TARGET_CONTRACTS:
        return suggested
    asset = context.get("asset_class", "")
    regime = context.get("reporting_regime", "")
    if asset == "equity_release_mortgage" and regime == "esma_annex_12":
        return "uk_equity_release_esma_annex12_v1"
    if asset == "equity_release_mortgage":
        return "uk_equity_release_mi_v1"
    return "uk_equity_release_mi_v1"


def resolve_onboarding_context(
    inventory: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    mode: str = "mi_only",
    client_name: str = "",
    document_terms: Optional[List[str]] = None,
    llm_callable: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Full context resolution: deterministic guess -> optional LLM -> backstop.

    Returns {"deterministic": {...}, "llm": {...|None}, "final": {...},
             "usage": {...}}.
    """
    guess = detect_context(inventory, evidence_rows, mode=mode, client_name=client_name,
                           document_terms=document_terms)
    llm_ctx, usage = None, {"llm_enabled": False, "calls_completed": 0,
                            "estimated_cost_gbp": 0.0}
    if llm_callable is not None:
        pkg = build_context_evidence(inventory, evidence_rows, guess, document_terms)
        out = resolve_context_with_llm(pkg, llm_callable)
        llm_ctx, usage = out["context"], out["usage"]
    final = backstop_context(guess, llm_ctx, inventory, evidence_rows, mode, document_terms)
    return {"deterministic": guess, "llm": llm_ctx, "final": final, "usage": usage}


def write_context_artifacts(context: Dict[str, Any], output_dir: str | Path,
                            deterministic: Optional[Dict[str, Any]] = None,
                            llm: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if deterministic is not None:
        (out_dir / "27a_deterministic_context_guess.json").write_text(
            json.dumps(deterministic, indent=2, default=str), encoding="utf-8")
    (out_dir / "27b_llm_context_resolution.json").write_text(
        json.dumps(llm if llm is not None else {"llm_enabled": False}, indent=2, default=str),
        encoding="utf-8")
    json_path = out_dir / "27_onboarding_context.json"
    json_path.write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
    md = ["# Onboarding context (final, backstopped)", ""]
    md.append(f"- **Asset class:** {context['asset_class']}")
    md.append(f"- **Jurisdiction:** {context['jurisdiction']}")
    md.append(f"- **Product type:** {context['product_type']}")
    md.append(f"- **Reporting regime:** {context['reporting_regime']}")
    md.append(f"- **Use cases:** {', '.join(context['use_cases'])}")
    md.append(f"- **Required domains:** {', '.join(context['required_domains'])}")
    md.append(f"- **Selected target contract:** {context.get('selected_target_contract','')}")
    md.append(f"- **Confidence:** {context['confidence']}")
    md.append(f"- **Context source:** {context.get('final_context_source','deterministic')} "
              f"({context.get('context_backstop_decision','')})")
    if context.get("needs_user_confirmation"):
        md.append("- ⚠️ **Confirm asset class / regime / required outputs in the workbench.**")
    md.append("")
    md.append(f"_Backstop: {context.get('context_backstop_reason','')}_")
    md.append(f"_Rationale: {context['rationale']}_")
    if context.get("open_questions"):
        md.append("")
        md.append("## Open questions")
        for q in context["open_questions"]:
            md.append(f"- {q}")
    md_path = out_dir / "27_onboarding_context_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"json": str(json_path), "summary_md": str(md_path)}
