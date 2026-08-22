"""demo_platform.onboarding — Stage 1: establish a governed contract per portfolio.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

This is the "configure once" stage. For each of the two portfolios it runs the
REAL production components:

  * ``engine.onboarding_agent.file_profiler.profile_file`` — source profiling
    (per-column inferred type, null rate, cardinality, redacted samples,
    identifier and reporting-date detection);
  * ``engine.gate_1_alignment.semantic_alignment.HeaderMapper`` with the client's
    approved alias overlay — the actual field-mapping decision engine, tier by
    tier, with the tier and confidence each header resolved at;

and records the result as an **approved onboarding contract** that the recurring
orchestration then applies unchanged, every period.

What is NOT done here: no LLM call. The demo runs fully offline and
deterministically, so the Tier 7 LLM mapper (which requires human confirmation
before any mapping is applied — see ``engine/gate_1_alignment/agent_orchestrator.py``)
is deliberately out of scope. Headers that the deterministic tiers cannot resolve
are recorded as review items and closed by the client's alias contract, which is
exactly the review-first path the production agent follows once a human confirms.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

import pandas as pd

from engine.gate_1_alignment.semantic_alignment import (
    HeaderMapper,
    apply_alias_overlay,
    load_aliases_from_dir,
    load_field_registry,
    select_registry_fields,
)
from engine.onboarding_agent.file_profiler import profile_file

from . import config as cfg
from . import schemas

_REGISTRY = cfg.REPO_ROOT / "config" / "system" / "fields_registry.yaml"
_GLOBAL_ALIASES = cfg.REPO_ROOT / "config" / "system"

#: Tiers the production mapper reports, in resolution order, with the plain-English
#: label the demonstration uses on screen.
TIER_LABELS: Dict[str, str] = {
    "exact": "Exact canonical name",
    "normalized": "Normalised name match",
    "alias": "Alias / client contract",
    "token_set": "Token-set similarity",
    "fuzz_token_set": "Fuzzy token match",
    "fuzz_ratio_norm": "Fuzzy normalised match",
    "unmapped": "Referred for review",
}

#: Below this confidence a deterministic match is reported as low confidence and
#: referred for review, matching semantic_alignment's own LOW_CONFIDENCE_THRESHOLD
#: reporting behaviour.
LOW_CONFIDENCE = 0.90


class OnboardingError(RuntimeError):
    """Raised when a portfolio's onboarding contract cannot be established."""


def _mapper_for(portfolio: cfg.PortfolioSpec) -> tuple[HeaderMapper, List[Dict[str, str]]]:
    """The production header mapper, with this portfolio's contract overlay."""
    registry = load_field_registry(_REGISTRY)
    canonical = select_registry_fields(registry, "equity_release")
    if not canonical:  # pragma: no cover - registry misconfiguration
        raise OnboardingError("No canonical fields selected for equity_release.")
    alias_map = load_aliases_from_dir(_GLOBAL_ALIASES)
    overlay_dir = cfg.demo_aliases_dir() / portfolio.source_portfolio_id
    alias_map, applied = apply_alias_overlay(alias_map, overlay_dir)
    return HeaderMapper(canonical, alias_map), applied


def profile_source(portfolio: cfg.PortfolioSpec,
                   prd: Optional[cfg.PeriodSpec] = None) -> List[Dict[str, Any]]:
    """Profile one portfolio's source extract with the production profiler."""
    prd = prd or cfg.PERIODS[0]
    path = cfg.source_file(portfolio, prd)
    if not path.exists():
        raise OnboardingError(
            f"{portfolio.display_id}: source extract {path} not found — run the "
            f"generator first (python -m demo_platform.run_demo --generate)."
        )
    return [asdict(p) for p in profile_file(path)]


def map_headers(portfolio: cfg.PortfolioSpec,
                prd: Optional[cfg.PeriodSpec] = None) -> Dict[str, Any]:
    """Run the real Gate 1 mapper over the source headers.

    Returns the per-header decision (canonical field, tier, confidence), the
    review items, and the declared-vs-achieved mapping check. A declared mapping
    that the engine does not actually produce fails the onboarding step — the demo
    never claims a mapping the pipeline would not make.
    """
    prd = prd or cfg.PERIODS[0]
    path = cfg.source_file(portfolio, prd)
    if not path.exists():
        raise OnboardingError(f"{portfolio.display_id}: source extract {path} not found.")
    headers = list(pd.read_csv(path, nrows=0).columns)
    schema = schemas.schema_for(portfolio.schema_name)
    declared = {c.header: (c.canonical_field or None) for c in schema.columns}
    notes = {c.header: c.note for c in schema.columns}
    resolution = {c.header: c.resolution for c in schema.columns}

    mapper, overlay_applied = _mapper_for(portfolio)
    decisions: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    for header in headers:
        canonical, tier, confidence = mapper.map_one(header)
        want = declared.get(header)
        low = canonical is not None and float(confidence) < LOW_CONFIDENCE
        decisions.append({
            "source_header": header,
            "canonical_field": canonical,
            "tier": tier,
            "tier_label": TIER_LABELS.get(tier, tier),
            "confidence": round(float(confidence), 4),
            "low_confidence": bool(low),
            "resolved_by_client_contract": resolution.get(header) == "contract",
            "declared_canonical_field": want,
            "note": notes.get(header, ""),
        })
        if canonical != want:
            mismatches.append({
                "source_header": header, "declared": want, "achieved": canonical,
            })

    if mismatches:
        raise OnboardingError(
            f"{portfolio.display_id}: the onboarding contract declares mappings the "
            f"Gate 1 engine does not produce: {mismatches}. Fix "
            f"demo_platform/schemas.py or the client alias contract — do not "
            f"present a mapping the pipeline would not make."
        )

    mapped = [d for d in decisions if d["canonical_field"]]
    unmapped = [d for d in decisions if not d["canonical_field"]]
    return {
        "decisions": decisions,
        "mapped_count": len(mapped),
        "unmapped_count": len(unmapped),
        "unmapped_headers": [d["source_header"] for d in unmapped],
        "low_confidence_count": sum(1 for d in decisions if d["low_confidence"]),
        "client_contract_count": sum(
            1 for d in decisions if d["resolved_by_client_contract"]),
        "alias_overlay_applied": overlay_applied,
        "tier_breakdown": {
            label: sum(1 for d in decisions if d["tier_label"] == label)
            for label in TIER_LABELS.values()
            if any(d["tier_label"] == label for d in decisions)
        },
    }


def _transformation_rules(portfolio: cfg.PortfolioSpec) -> List[Dict[str, str]]:
    """The client- and asset-specific transformations the contract establishes.

    Each one names the production component that performs it, so the film only
    claims transformations that genuinely happen.
    """
    common = [
        {
            "rule": "Balance coherence (ERM)",
            "detail": "current_outstanding_balance = current_principal_balance + "
                      "accrued_interest. This book capitalises interest monthly, so "
                      "accrued interest is nil and the two balances are equal.",
            "implemented_by": "engine/gate_2_transform/canonical_transform.py :: derive_fields",
        },
        {
            "rule": "Current LTV derivation",
            "detail": "current_loan_to_value = current_principal_balance / "
                      "current_valuation_amount, in percentage points.",
            "implemented_by": "engine/gate_2_transform/canonical_transform.py :: derive_fields",
        },
        {
            "rule": "Original LTV derivation",
            "detail": "original_loan_to_value = original_principal_balance / "
                      "original_valuation_amount.",
            "implemented_by": "engine/gate_2_transform/canonical_transform.py :: derive_fields",
        },
        {
            "rule": "UK geography enrichment",
            "detail": "Property postcode → ITL3 code (uk_itl_master_lookup_v2.csv); "
                      "readable region label routed to collateral_geography; "
                      "regulatory NUTS fields carry the ITL3 code.",
            "implemented_by": "engine/gate_2_transform/canonical_transform.py :: normalize_geography",
        },
        {
            "rule": "Source-portfolio provenance stamp",
            "detail": f"Every row stamped source_portfolio_id={portfolio.source_portfolio_id}, "
                      f"source_portfolio_type={portfolio.source_portfolio_type}, "
                      f"portfolio_cohort={portfolio.source_portfolio_id}.",
            "implemented_by": "engine/provenance.py :: stamp_dataframe",
        },
        {
            "rule": "Reporting date resolution",
            "detail": "The extract's own cut-off column is the reporting date; no "
                      "static override is configured for this client.",
            "implemented_by": "engine/gate_2_transform/canonical_transform.py :: derive_reporting_date",
        },
    ]
    if portfolio.source_portfolio_id == cfg.PORTFOLIO_B.source_portfolio_id:
        common.append({
            "rule": "Servicer status normalisation",
            "detail": "The servicer's account state vocabulary (Open / Redeemed) is "
                      "normalised to the canonical account_status enumeration.",
            "implemented_by": "engine/gate_2_transform/canonical_transform.py (enum layer)",
        })
        common.append({
            "rule": "Indexed valuation basis",
            "detail": "The servicer's indexed valuation is accepted as the current "
                      "valuation basis for current LTV, per the approved contract.",
            "implemented_by": "demo_platform/aliases/alp_acquired/aliases_client_contract.yaml",
        })
    else:
        common.append({
            "rule": "Advance vocabulary",
            "detail": "'Original Advance' is the originated principal; 'Current "
                      "Balance' is capital plus capitalised roll-up interest.",
            "implemented_by": "demo_platform/aliases/alp_origination/aliases_client_contract.yaml",
        })
    return common


def _validation_rules() -> List[Dict[str, str]]:
    """The validation configuration the contract establishes for every period."""
    return [
        {
            "gate": "Gate 2 — canonical validation",
            "detail": "Type, format and registry conformance for every mapped "
                      "canonical field.",
            "implemented_by": "engine/gate_3_validation/validate_canonical.py",
        },
        {
            "gate": "Gate 2.5 — lineage",
            "detail": "Field-level and value-level lineage for every canonical field.",
            "implemented_by": "engine/gate_2_transform/lineage_tracker.py",
        },
        {
            "gate": "Gate 3 — business rules",
            "detail": "Cross-field business rules: LTV reconciles to balance and "
                      "valuation, borrower age within product bounds, redeemed "
                      "accounts carry nil balance, mandatory provenance present.",
            "implemented_by": "engine/gate_3_validation/validate_business_rules.py",
        },
        {
            "gate": "Gate 3b — aggregation",
            "detail": "Field summary and exception dashboard per run.",
            "implemented_by": "engine/gate_3_validation/aggregate_validation_results.py",
        },
    ]


def build_contract(portfolio: cfg.PortfolioSpec) -> Dict[str, Any]:
    """Establish and return the approved onboarding contract for one portfolio."""
    prd = cfg.PERIODS[0]
    profile = profile_source(portfolio, prd)
    mapping = map_headers(portfolio, prd)
    schema = schemas.schema_for(portfolio.schema_name)
    source_path = cfg.source_file(portfolio, prd)
    sample = pd.read_csv(source_path, nrows=5000, low_memory=False)

    reporting_date_cols = [p["source_column"] for p in profile if p.get("likely_reporting_date")]
    identifier_cols = [p["source_column"] for p in profile if p.get("likely_identifier")]

    return {
        "synthetic_notice": cfg.SYNTHETIC_BANNER,
        "contract_version": "1.0",
        "approved_at_utc": "2026-07-01T00:00:00Z",   # fixed: the contract is an artefact, not a run
        "approved_by": "Trakt managed service (demonstration)",
        "client_id": cfg.CLIENT_ID,
        "client_display_name": cfg.CLIENT_DISPLAY_NAME,
        "portfolio": {
            "key": portfolio.key,
            "source_portfolio_id": portfolio.source_portfolio_id,
            "source_portfolio_type": portfolio.source_portfolio_type,
            "display_id": portfolio.display_id,
            "label": portfolio.label,
            "description": portfolio.description,
            "acquisition_date": portfolio.acquisition_date,
            "seller_name": portfolio.seller_name,
        },
        "source": {
            "schema_name": schema.name,
            "system_of_record": schema.system_of_record,
            "description": schema.description,
            "example_file": source_path.name,
            "column_count": len(schema.columns),
            "row_count_profiled": int(len(sample)),
            "detected_reporting_date_columns": reporting_date_cols,
            "detected_identifier_columns": identifier_cols,
        },
        "profile": profile,
        "mapping": mapping,
        "transformations": _transformation_rules(portfolio),
        "validation": _validation_rules(),
        "recurring_execution": {
            "cadence": "monthly, on receipt of the month-end extract",
            "orchestrator": "engine/orchestrator/trakt_run.py --mode mi",
            "assembler": "engine/platform_assembler.py",
            "note": "The contract is applied unchanged every period. No mapping "
                    "decision is repeated manually.",
        },
    }


def run(*, verbose: bool = True) -> Dict[str, Any]:
    """Establish both contracts and write them to the onboarding directory."""
    out_root = cfg.onboarding_dir()
    out_root.mkdir(parents=True, exist_ok=True)
    contracts: Dict[str, Any] = {}
    # Every portfolio the sponsor is on the hook for, including the deal they sold.
    for portfolio in cfg.ALL_PORTFOLIOS:
        contract = build_contract(portfolio)
        pdir = out_root / portfolio.source_portfolio_id
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "onboarding_contract.json").write_text(
            json.dumps(contract, indent=2, default=str), encoding="utf-8")
        # A flat mapping report, the same shape the pipeline's own Gate 1 report
        # uses, so an operator can diff the two.
        pd.DataFrame(contract["mapping"]["decisions"]).to_csv(
            pdir / "mapping_report.csv", index=False)
        contracts[portfolio.source_portfolio_id] = contract
        if verbose:
            m = contract["mapping"]
            print(f"  [onboarding] {portfolio.display_id}: "
                  f"{m['mapped_count']} mapped, {m['unmapped_count']} unmapped, "
                  f"{m['client_contract_count']} resolved by client contract, "
                  f"{m['low_confidence_count']} low confidence")

    # Publish the governed per-portfolio metadata this stage established. The
    # asset class is decided ONCE, here, and every downstream MI view reads it
    # from the registry rather than re-deriving it — which is what makes
    # "Equity Release MI = common core + ER semantics" a property of the book
    # rather than of whichever component happens to be asking.
    registry = _publish_portfolio_metadata()
    if verbose:
        declared = {e["source_portfolio_id"]: e.get("asset_class")
                    for e in registry["portfolios"]}
        print(f"  [onboarding] governed portfolio metadata: {declared}")
    return contracts


def _publish_portfolio_metadata() -> Dict[str, Any]:
    """Write each portfolio's governed asset class to the portfolio registry.

    The asset class comes from the client's onboarding configuration — the same
    ``portfolio.asset_class`` the pipeline already reads — and is normalised to
    the Business Semantics Registry vocabulary on the way in.
    """
    from engine.onboarding_agent import portfolio_registry_writer as registry_writer

    client_cfg = yaml.safe_load(
        cfg.demo_client_config().read_text(encoding="utf-8")) or {}
    declared = (client_cfg.get("portfolio") or {}).get("asset_class")
    return registry_writer.publish(
        cfg.portfolio_registry_path(),
        [{"source_portfolio_id": p.source_portfolio_id}
         for p in cfg.ALL_PORTFOLIOS],
        asset_class=declared)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse
    ap = argparse.ArgumentParser(description="Establish the demo onboarding contracts.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    contracts = run(verbose=not args.quiet)
    print(json.dumps({
        pid: {
            "mapped": c["mapping"]["mapped_count"],
            "unmapped": c["mapping"]["unmapped_headers"],
            "tiers": c["mapping"]["tier_breakdown"],
        } for pid, c in contracts.items()
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
