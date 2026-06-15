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
from typing import Any, Dict, List, Optional

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


def write_context_artifacts(context: Dict[str, Any], output_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "27_onboarding_context.json"
    json_path.write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
    md = ["# Onboarding context (detected)", ""]
    md.append(f"- **Asset class:** {context['asset_class']}")
    md.append(f"- **Jurisdiction:** {context['jurisdiction']}")
    md.append(f"- **Product type:** {context['product_type']}")
    md.append(f"- **Reporting regime:** {context['reporting_regime']}")
    md.append(f"- **Use cases:** {', '.join(context['use_cases'])}")
    md.append(f"- **Required domains:** {', '.join(context['required_domains'])}")
    md.append(f"- **Confidence:** {context['confidence']}"
              + ("  ⚠️ low — confirm asset class / regime in the workbench"
                 if context["needs_user_confirmation"] else ""))
    md.append("")
    md.append(f"_Rationale: {context['rationale']}_")
    md_path = out_dir / "27_onboarding_context_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"json": str(json_path), "summary_md": str(md_path)}
