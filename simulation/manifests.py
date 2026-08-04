"""simulation.manifests — the case catalogue and its validation.

Eighteen economic cases (six per asset family) plus two deliberate guard cases
that must fail at a governed boundary. The catalogue is deliberately narrow:
every case has a stated hardening purpose and at least one independently
verifiable expected outcome. Nothing is here to inflate a test count.

A manifest is the whole reproducibility contract — case id, simulator version,
seed, asset class and subtype, tenancy, funding structure, reporting window,
dialects, declared schema evolution, intended risk state, regime applicability
and the expected outcome. The runner writes it verbatim into every run
directory, so a failing generated case becomes a permanent regression fixture by
copying one JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from . import SIMULATION_VERSION
from .knobs import INTENT_MODIFIERS
from .models import (
    ARTEFACT_AWAITING_CONFIG,
    ARTEFACT_NOT_APPLICABLE,
    ARTEFACT_SUPPORTED,
    ASSET_BRIDGE,
    ASSET_CLASSES,
    ASSET_EQUITY_RELEASE,
    ASSET_FINANCE,
    ASSET_SUBTYPES,
    DIALECTS,
    DIALECT_CLEAN_CSV,
    DIALECT_LENDER_EXCEL,
    DIALECT_LOCALE_CSV,
    EXPECT_FAILURE,
    EXPECT_SUCCESS,
    REGIME_APPLICABLE,
    REGIME_NOT_APPLICABLE,
    RISK_AMBER,
    RISK_GREEN,
    RISK_RED,
    SOURCE_ACQUIRED,
    SOURCE_DIRECT,
    SOURCE_PORTFOLIO_TYPES,
    CaseManifest,
    FailureClass,
    FundedSource,
    RegimeIntent,
    RiskIntent,
    SchemaEvolution,
)

#: Every economic case renders through all three dialects.
ALL_DIALECTS = (DIALECT_CLEAN_CSV, DIALECT_LENDER_EXCEL, DIALECT_LOCALE_CSV)

#: Profiles a case can belong to.
PROFILE_SMOKE = "smoke"
PROFILE_STANDARD = "standard"
PROFILE_PERFORMANCE = "performance"
PROFILE_GUARD = "guard"
PROFILES = (PROFILE_SMOKE, PROFILE_STANDARD, PROFILE_PERFORMANCE, PROFILE_GUARD)

_START = "2025-01-31"
_MONTHS = 6


def _annex2() -> RegimeIntent:
    return RegimeIntent(
        applicability=REGIME_APPLICABLE, regime_id="ESMA_Annex2",
        reason="UK residential-secured exposures reported at exposure level.",
        artefact_status=ARTEFACT_SUPPORTED)


def _annex3() -> RegimeIntent:
    return RegimeIntent(
        applicability=REGIME_APPLICABLE, regime_id="ESMA_Annex3",
        reason=("Commercial-real-estate-secured exposures. The projection is "
                "supported; no XML delivery template is committed for Annex 3, "
                "so the artefact awaits deployment configuration."),
        artefact_status=ARTEFACT_AWAITING_CONFIG)


def _annex8() -> RegimeIntent:
    return RegimeIntent(
        applicability=REGIME_APPLICABLE, regime_id="ESMA_Annex8",
        reason=("Leasing exposures. The projection is supported; no XML "
                "delivery template is committed for Annex 8, so the artefact "
                "awaits deployment configuration."),
        artefact_status=ARTEFACT_AWAITING_CONFIG)


def _no_regime(reason: str) -> RegimeIntent:
    return RegimeIntent(applicability=REGIME_NOT_APPLICABLE, regime_id=None,
                        reason=reason, artefact_status=ARTEFACT_NOT_APPLICABLE)


def _case(**kw) -> CaseManifest:
    kw.setdefault("simulation_version", SIMULATION_VERSION)
    kw.setdefault("start_reporting_date", _START)
    kw.setdefault("months", _MONTHS)
    kw.setdefault("dialects", ALL_DIALECTS)
    kw.setdefault("currency", "GBP")
    kw.setdefault("jurisdiction", "GB")
    kw.setdefault("source_portfolio_type", "direct")
    kw.setdefault("profiles", (PROFILE_STANDARD,))
    return CaseManifest(**kw)


# --------------------------------------------------------------------------- #
# Equity release
# --------------------------------------------------------------------------- #
_EQUITY_RELEASE: List[CaseManifest] = [
    _case(
        case_id="equity_release_clean_origination_v1", seed=41_101,
        purpose=("Baseline: a clean, newly originated roll-up book. Proves the "
                 "whole pathway end to end and gives every other equity-release "
                 "case a green risk baseline to depart from."),
        asset_class=ASSET_EQUITY_RELEASE, asset_subtype="lump_sum",
        client_id="sim_equity_release", portfolio_id="direct_001",
        spv_id="SPV_ER_1", opening_loan_count=120,
        economic_intent=("new_origination", "clean_performing"),
        risk_intent=RiskIntent(RISK_GREEN), regime=_annex2(),
        profiles=(PROFILE_SMOKE, PROFILE_STANDARD),
    ),
    _case(
        case_id="equity_release_seasoned_rollup_v1", seed=41_102,
        purpose=("A mature book where almost all of the balance movement is "
                 "capitalised interest. Proves rolled-up interest reaches MI as "
                 "a movement rather than as unexplained growth, and that a "
                 "mid-history column reorder changes nothing canonical."),
        asset_class=ASSET_EQUITY_RELEASE, asset_subtype="mixed",
        client_id="sim_equity_release", portfolio_id="direct_001",
        spv_id="SPV_ER_1", opening_loan_count=140,
        economic_intent=("seasoned_book", "rolled_up_interest"),
        risk_intent=RiskIntent(RISK_GREEN), regime=_annex2(),
        schema_evolution=(
            SchemaEvolution(from_period=2, kind="reorder_columns",
                            purpose="A lender re-exports with a different column "
                                    "order; canonical truth must be unchanged."),
        ),
    ),
    _case(
        case_id="equity_release_mixed_cohorts_v1", seed=41_103,
        purpose=("Direct originations and an acquired back book in one "
                 "portfolio. Proves cohort attribution survives ingestion and "
                 "that an optional column appearing mid-history is additive."),
        asset_class=ASSET_EQUITY_RELEASE, asset_subtype="drawdown",
        client_id="sim_equity_release", portfolio_id="acquired_001",
        source_portfolio_type="acquired",
        acquisition_date="2024-12-31",
        seller_name="Simulated Portfolio Vendor Ltd",
        spv_id="SPV_ER_2", opening_loan_count=150,
        economic_intent=("mixed_cohorts", "seasoned_book"),
        risk_intent=RiskIntent(RISK_GREEN), regime=_annex2(),
        schema_evolution=(
            SchemaEvolution(from_period=3, kind="add_optional_column",
                            purpose="An optional column the lender starts "
                                    "supplying part-way through the history.",
                            detail={"column": "number_of_bedrooms"}),
        ),
    ),
    _case(
        case_id="equity_release_high_ltv_adverse_v1", seed=41_104,
        purpose=("High initial LTV meeting a falling property market. Proves "
                 "current LTV rises from BOTH sides (roll-up and revaluation) "
                 "and that NNEG-relevant high-LTV exposure is visible in MI."),
        asset_class=ASSET_EQUITY_RELEASE, asset_subtype="lump_sum",
        client_id="sim_equity_release", portfolio_id="direct_001",
        spv_id="SPV_ER_1", opening_loan_count=130,
        economic_intent=("seasoned_book", "high_ltv_adverse_valuation"),
        risk_intent=RiskIntent(RISK_RED, ("ltv_limit",)), regime=_annex2(),
    ),
    _case(
        case_id="equity_release_redemption_heavy_v1", seed=41_105,
        purpose=("A shrinking book: redemptions dominate. Proves the funded "
                 "movement is negative, redemptions reconcile, and a controlled "
                 "restatement of a prior period follows production policy."),
        asset_class=ASSET_EQUITY_RELEASE, asset_subtype="mixed",
        client_id="sim_equity_release", portfolio_id="direct_001",
        spv_id="SPV_ER_1", opening_loan_count=160,
        economic_intent=("seasoned_book", "redemption_heavy",
                         "controlled_restatement"),
        risk_intent=RiskIntent(RISK_GREEN), regime=_annex2(),
        schema_evolution=(
            SchemaEvolution(from_period=3, kind="restate_prior",
                            purpose="A controlled correction to previously "
                                    "reported balances."),
        ),
    ),
    _case(
        case_id="equity_release_concentration_stress_v1", seed=41_106,
        purpose=("Engineered geographic and single-borrower concentration. "
                 "Proves the governed limit library moves to warning/breach on "
                 "the named limits and stays green on the others."),
        asset_class=ASSET_EQUITY_RELEASE, asset_subtype="lump_sum",
        client_id="sim_equity_release", portfolio_id="direct_001",
        spv_id="SPV_ER_1", opening_loan_count=150,
        economic_intent=("seasoned_book", "geographic_concentration_stress",
                         "borrower_concentration_stress"),
        risk_intent=RiskIntent(RISK_RED, ("geographic_concentration",
                                          "borrower_concentration")),
        regime=_no_regime("MI-only book: no securitisation deal exists for this "
                          "portfolio, so no exposure-level regime applies."),
    ),
]

# --------------------------------------------------------------------------- #
# Bridge
# --------------------------------------------------------------------------- #
_BRIDGE: List[CaseManifest] = [
    _case(
        case_id="bridge_clean_short_duration_v1", seed=41_201,
        purpose=("Baseline: a clean, performing short-duration first-charge "
                 "book. Proves bridge runs the full production pathway under "
                 "--portfolio-type cre with no registry change."),
        asset_class=ASSET_BRIDGE, asset_subtype="residential_bridge",
        client_id="sim_bridge", portfolio_id="direct_001",
        spv_id="SPV_BR_1", opening_loan_count=110,
        economic_intent=("clean_performing",),
        risk_intent=RiskIntent(RISK_GREEN), regime=_annex3(),
        profiles=(PROFILE_SMOKE, PROFILE_STANDARD),
    ),
    _case(
        case_id="bridge_mixed_charge_rank_v1", seed=41_202,
        purpose=("First and second charges over mixed collateral. Proves "
                 "charge_type and lien survive ingestion and drive a "
                 "charge-rank concentration measure."),
        asset_class=ASSET_BRIDGE, asset_subtype="mixed_bridge",
        client_id="sim_bridge", portfolio_id="direct_001",
        spv_id="SPV_BR_1", opening_loan_count=120,
        economic_intent=("mixed_charge_rank",),
        risk_intent=RiskIntent(RISK_AMBER, ("charge_rank_concentration",)),
        regime=_annex3(),
    ),
    _case(
        case_id="bridge_extension_heavy_v1", seed=41_203,
        purpose=("Extensions dominate the maturity profile. Proves an "
                 "extension is visible as a moved maturity date rather than a "
                 "new loan, and that dropping an optional column mid-history "
                 "degrades disclosure without corrupting canonical truth."),
        asset_class=ASSET_BRIDGE, asset_subtype="residential_bridge",
        client_id="sim_bridge", portfolio_id="direct_001",
        spv_id="SPV_BR_1", opening_loan_count=130,
        economic_intent=("extension_heavy",),
        risk_intent=RiskIntent(RISK_RED, ("extension_concentration",)),
        regime=_annex3(),
        schema_evolution=(
            SchemaEvolution(from_period=4, kind="drop_optional_column",
                            purpose="The lender stops supplying an optional "
                                    "column part-way through the history.",
                            detail={"column": "prepayment_terms_description"}),
        ),
    ),
    _case(
        case_id="bridge_maturity_wall_v1", seed=41_204,
        purpose=("A concentrated maturity wall inside 90 days. Proves the "
                 "days-to-maturity measure is computed from canonical dates and "
                 "that an alias change between periods still resolves."),
        asset_class=ASSET_BRIDGE, asset_subtype="commercial_bridge",
        client_id="sim_bridge", portfolio_id="direct_001",
        spv_id="SPV_BR_1", opening_loan_count=125,
        economic_intent=("maturity_wall",),
        risk_intent=RiskIntent(RISK_RED, ("maturity_concentration",)),
        regime=_annex3(),
        schema_evolution=(
            SchemaEvolution(from_period=3, kind="change_alias",
                            purpose="The lender renames a header mid-history; "
                                    "the approved alias contract must still "
                                    "resolve it.",
                            detail={"canonical_field": "current_outstanding_balance",
                                    "new_header": "Balance O/S"}),
        ),
    ),
    _case(
        case_id="bridge_arrears_default_enforcement_v1", seed=41_205,
        purpose=("The full impairment ladder: performing → arrears → default → "
                 "enforcement → security realisation. Proves status migration "
                 "and the recovery/write-off split reconcile."),
        asset_class=ASSET_BRIDGE, asset_subtype="mixed_bridge",
        client_id="sim_bridge", portfolio_id="direct_001",
        spv_id="SPV_BR_1", opening_loan_count=140,
        economic_intent=("arrears_transitions",),
        risk_intent=RiskIntent(RISK_RED, ("arrears_concentration",)),
        regime=_annex3(),
    ),
    _case(
        case_id="bridge_geographic_borrower_stress_v1", seed=41_206,
        purpose=("Geographic, single-borrower and loan-size concentration in "
                 "one book. Proves three independent limits move together and "
                 "that the largest-exposure figures reconcile to loan level."),
        asset_class=ASSET_BRIDGE, asset_subtype="commercial_bridge",
        client_id="sim_bridge", portfolio_id="direct_001",
        spv_id="SPV_BR_1", opening_loan_count=135,
        economic_intent=("geographic_concentration_stress",
                         "borrower_concentration_stress",
                         "loan_size_concentration_stress"),
        risk_intent=RiskIntent(RISK_RED, ("geographic_concentration",
                                          "single_borrower_concentration",
                                          "loan_size_concentration")),
        regime=_no_regime("Warehoused bridge book with no securitisation deal; "
                          "no exposure-level regime applies."),
    ),
]

# --------------------------------------------------------------------------- #
# Asset finance (equipment finance is a SUBTYPE, not a separate family)
# --------------------------------------------------------------------------- #
_ASSET_FINANCE: List[CaseManifest] = [
    _case(
        case_id="asset_finance_clean_sme_equipment_v1", seed=41_301,
        purpose=("Baseline: a clean SME equipment-finance book. Proves "
                 "equipment finance runs as an asset-finance SUBTYPE through "
                 "--portfolio-type equipment with no new architecture."),
        asset_class=ASSET_FINANCE, asset_subtype="equipment_finance",
        client_id="sim_asset_finance", portfolio_id="direct_001",
        spv_id="SPV_AF_1", opening_loan_count=115,
        economic_intent=("clean_performing",),
        risk_intent=RiskIntent(RISK_GREEN), regime=_annex8(),
        profiles=(PROFILE_SMOKE, PROFILE_STANDARD),
    ),
    _case(
        case_id="asset_finance_hp_lease_mix_v1", seed=41_302,
        purpose=("Hire purchase, finance lease and asset loans side by side. "
                 "Proves contract type is a configured attribute and that "
                 "changing the source enum synonyms mid-history still resolves."),
        asset_class=ASSET_FINANCE, asset_subtype="equipment_finance",
        client_id="sim_asset_finance", portfolio_id="direct_001",
        spv_id="SPV_AF_1", opening_loan_count=125,
        economic_intent=("hire_purchase_and_lease_mix",),
        risk_intent=RiskIntent(RISK_GREEN), regime=_annex8(),
        schema_evolution=(
            SchemaEvolution(from_period=2, kind="change_enum_synonyms",
                            purpose="The lender switches to a different status "
                                    "vocabulary part-way through the history."),
        ),
    ),
    _case(
        case_id="asset_finance_mixed_collateral_v1", seed=41_303,
        purpose=("Several equipment categories and manufacturers in one book. "
                 "Proves collateral-type and manufacturer dimensions reach the "
                 "risk engine, and that a CSV→XLSX format switch mid-history is "
                 "canonically neutral."),
        asset_class=ASSET_FINANCE, asset_subtype="mixed_asset_finance",
        client_id="sim_asset_finance", portfolio_id="direct_001",
        spv_id="SPV_AF_1", opening_loan_count=140,
        economic_intent=("mixed_collateral_categories",),
        risk_intent=RiskIntent(RISK_GREEN), regime=_annex8(),
        schema_evolution=(
            SchemaEvolution(from_period=4, kind="switch_format",
                            purpose="The lender migrates from CSV to XLSX "
                                    "part-way through the history, keeping the "
                                    "same columns in the same order.",
                            detail={"to": "xlsx"}),
        ),
    ),
    _case(
        case_id="asset_finance_balloon_residual_v1", seed=41_304,
        purpose=("Balloon and residual-value exposure. Proves balloon_amount "
                 "and the residual-value fields survive ingestion and drive a "
                 "residual-value concentration limit."),
        asset_class=ASSET_FINANCE, asset_subtype="equipment_finance",
        client_id="sim_asset_finance", portfolio_id="direct_001",
        spv_id="SPV_AF_1", opening_loan_count=130,
        economic_intent=("balloon_and_residual",),
        risk_intent=RiskIntent(RISK_RED, ("balloon_residual_concentration",)),
        regime=_annex8(),
    ),
    _case(
        case_id="asset_finance_delinquency_repossession_v1", seed=41_305,
        purpose=("Delinquency through to repossession and disposal. Proves the "
                 "disposal proceeds and the shortfall are reported separately "
                 "and reconcile to the closing balance."),
        asset_class=ASSET_FINANCE, asset_subtype="mixed_asset_finance",
        client_id="sim_asset_finance", portfolio_id="direct_001",
        spv_id="SPV_AF_1", opening_loan_count=135,
        economic_intent=("delinquency_repossession",),
        risk_intent=RiskIntent(RISK_RED, ("delinquency_concentration",)),
        regime=_annex8(),
    ),
    _case(
        case_id="asset_finance_vendor_sector_stress_v1", seed=41_306,
        purpose=("Sector, vendor and manufacturer concentration in one book. "
                 "Proves the three composition dimensions added to the governed "
                 "limit library evaluate independently."),
        asset_class=ASSET_FINANCE, asset_subtype="vehicle_finance",
        client_id="sim_asset_finance", portfolio_id="direct_001",
        spv_id="SPV_AF_1", opening_loan_count=120,
        economic_intent=("vendor_concentration_stress",),
        risk_intent=RiskIntent(RISK_RED, ("industry_concentration",
                                          "vendor_concentration",
                                          "manufacturer_concentration",
                                          "collateral_type_concentration")),
        regime=_no_regime("Vendor-programme book held on balance sheet; no "
                          "securitisation deal, so no regime applies."),
    ),
]

# --------------------------------------------------------------------------- #
# Guard cases — these MUST fail, at the right boundary, with the right reason
# --------------------------------------------------------------------------- #
_GUARDS: List[CaseManifest] = [
    _case(
        case_id="bridge_unsupported_schema_guard_v1", seed=41_901,
        purpose=("An ambiguous source schema whose headers cannot be resolved "
                 "to the canonical model. Proves ingestion refuses rather than "
                 "guessing, and that the refusal is classified."),
        asset_class=ASSET_BRIDGE, asset_subtype="residential_bridge",
        client_id="sim_bridge", portfolio_id="direct_001",
        spv_id="SPV_BR_1", opening_loan_count=60, months=6,
        dialects=(DIALECT_CLEAN_CSV,),
        economic_intent=("clean_performing",),
        risk_intent=RiskIntent(RISK_GREEN),
        regime=_no_regime("Guard case: the run never reaches a regime."),
        expectation=EXPECT_FAILURE,
        expected_failure_class=FailureClass.UNSUPPORTED_SOURCE_SCHEMA,
        schema_evolution=(
            SchemaEvolution(from_period=0, kind="ambiguous_schema",
                            purpose="Every header is replaced with an opaque "
                                    "positional label; nothing may resolve."),
        ),
        profiles=(PROFILE_GUARD, PROFILE_STANDARD),
    ),
    _case(
        case_id="equity_release_missing_mandatory_guard_v1", seed=41_902,
        purpose=("A source that omits a mandatory canonical input. Proves the "
                 "canonical validator blocks the run at Gate 2 rather than "
                 "producing a book with a silently absent core field."),
        asset_class=ASSET_EQUITY_RELEASE, asset_subtype="lump_sum",
        client_id="sim_equity_release", portfolio_id="direct_001",
        spv_id="SPV_ER_1", opening_loan_count=60, months=6,
        dialects=(DIALECT_CLEAN_CSV,),
        economic_intent=("clean_performing",),
        risk_intent=RiskIntent(RISK_GREEN),
        regime=_no_regime("Guard case: the run never reaches a regime."),
        expectation=EXPECT_FAILURE,
        expected_failure_class=FailureClass.MISSING_MANDATORY_DATA,
        schema_evolution=(
            SchemaEvolution(from_period=0, kind="drop_mandatory_column",
                            purpose="The mandatory current-balance column is "
                                    "withheld.",
                            detail={"column": "current_outstanding_balance"}),
        ),
        profiles=(PROFILE_GUARD, PROFILE_STANDARD),
    ),
]

#: The whole catalogue, in a stable order.
# --------------------------------------------------------------------------- #
# Multi-source funded books — platform assembly
#
# One client, ONE SPV, two separately delivered funded populations. This is the
# only case shape that exercises ``engine/platform_assembler.py``, because
# assembly is a no-op when a client delivers a single portfolio.
#
# Direct/acquired here is PROVENANCE INSIDE the SPV, not a second SPV: both
# sources share ``spv_id``, report on the same month-ends, and are consolidated
# into one platform canonical that every downstream stage then reads.
# --------------------------------------------------------------------------- #
_MULTI_SOURCE: List[CaseManifest] = [
    _case(
        case_id="equity_release_spv_multi_source_v1", seed=41_701,
        purpose=("One SPV fed by two separately delivered funded populations — "
                 "directly originated lending and an acquired back book — on "
                 "aligned monthly reporting dates. Proves the production "
                 "platform assembler consolidates them into one SPV-level view "
                 "with no duplication or omission, that direct/acquired "
                 "provenance survives assembly, and that governed MI answers "
                 "total-SPV, direct-only and acquired-only questions from the "
                 "same assembled book without leaking between them."),
        asset_class=ASSET_EQUITY_RELEASE, asset_subtype="mixed",
        client_id="sim_multi_source", portfolio_id="direct_001",
        spv_id="SPV_MS_1", opening_loan_count=120,
        economic_intent=("mixed_cohorts", "seasoned_book"),
        risk_intent=RiskIntent(RISK_GREEN), regime=_annex2(),
        sources=(
            FundedSource(portfolio_id="direct_001",
                         source_portfolio_type=SOURCE_DIRECT,
                         share=0.6, seed_offset=0),
            FundedSource(portfolio_id="acquired_001",
                         source_portfolio_type=SOURCE_ACQUIRED,
                         share=0.4, seed_offset=7_919,
                         acquisition_date="2024-12-31",
                         seller_name="Simulated Portfolio Vendor Ltd"),
        ),
    ),
]

CATALOGUE: List[CaseManifest] = (_EQUITY_RELEASE + _BRIDGE + _ASSET_FINANCE
                                 + _MULTI_SOURCE + _GUARDS)

#: One standard-scale history per asset family, plus one large case overall.
STANDARD_SCALE_CASES = ("equity_release_seasoned_rollup_v1",
                        "bridge_clean_short_duration_v1",
                        "asset_finance_clean_sme_equipment_v1")
LARGE_SCALE_CASE = "equity_release_seasoned_rollup_v1"

#: The performance schedule, in order. Each entry is an explicit run spec, not
#: an id whose scale is inferred elsewhere.
#:
#: The large run is deliberately one of the standard cases run AGAIN at a bigger
#: row count: holding the economics constant and changing only the scale is what
#: makes the two timings comparable.
#:
#: It is also deliberately NARROW. Measured at 100 000 loans, regime projection
#: and XML delivery took 3 108s of a 5 045s run (62%) — and the framework runs
#: Gate 4/5 twice there, once for the artefact and once to prove the artefact
#: reproduces, producing two 2.6 GB submissions. The governed MI Agent added a
#: further 235s. None of that measures the funded ingestion path, so the large
#: run covers one asset class, one reporting period, one dialect, and stops at
#: risk. Regime and Agent throughput at scale is a separate diagnostic, asked
#: for explicitly with ``--stages``.
PERFORMANCE_RUNS = tuple(
    [{"case_id": cid, "scale": "standard"} for cid in STANDARD_SCALE_CASES]
    + [{"case_id": LARGE_SCALE_CASE, "scale": "large", "months": 1,
        "stages": ("generate", "render", "ingest", "canonical",
                   "history_integration", "mi", "risk")}])


# --------------------------------------------------------------------------- #
# Access + validation
# --------------------------------------------------------------------------- #
def all_cases() -> List[CaseManifest]:
    return list(CATALOGUE)


def get_case(case_id: str) -> CaseManifest:
    for case in CATALOGUE:
        if case.case_id == case_id:
            return case
    raise KeyError(f"unknown simulation case {case_id!r}; run "
                   f"`python -m simulation.runner list` for the catalogue")


def cases_for_profile(profile: str) -> List[CaseManifest]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {PROFILES}")
    if profile == PROFILE_PERFORMANCE:
        return [get_case(spec["case_id"]) for spec in PERFORMANCE_RUNS]
    return [c for c in CATALOGUE if profile in c.profiles]


class ManifestError(ValueError):
    """A manifest that could not be trusted to run."""


def _supported_regimes() -> Sequence[str]:
    """The regimes the repository genuinely supports, read from production."""
    from engine.orchestrator.trakt_run import VALID_REGULATORY_REGIMES
    return VALID_REGULATORY_REGIMES


def _validate_sources(case: CaseManifest) -> List[str]:
    """Rules a multi-source case must satisfy to mean what it claims."""
    problems: List[str] = []
    if not case.sources:
        return problems
    if len(case.sources) < 2:
        problems.append("declare two or more sources or none at all; a single "
                        "entry is the default single-source case with extra "
                        "ceremony")
    ids = [s.portfolio_id for s in case.sources]
    if len(set(ids)) != len(ids):
        problems.append(f"source portfolio ids must be distinct: {ids}")
    # Distinct ids give distinct loan identifiers for free, because the
    # generator prefixes every loan id with its portfolio id.
    offsets = [s.seed_offset for s in case.sources]
    if len(set(offsets)) != len(offsets):
        problems.append(f"source seed offsets must be distinct, or two sources "
                        f"are the same book under two labels: {offsets}")
    total = round(sum(s.share for s in case.sources), 9)
    if total != 1.0:
        problems.append(f"source shares must sum to exactly 1.0, got {total}")
    types = {s.source_portfolio_type for s in case.sources}
    if types != set(SOURCE_PORTFOLIO_TYPES):
        problems.append(
            f"a multi-source case exists to prove direct/acquired provenance "
            f"survives assembly, so it must carry both; got {sorted(types)}")
    if case.portfolio_id not in ids:
        problems.append(f"the case's portfolio_id {case.portfolio_id!r} must be "
                        f"one of its sources {ids}")
    return problems


def validate_manifest(case: CaseManifest, *,
                      check_regimes: bool = True) -> List[str]:
    """Return the problems with *case*; empty means it is safe to run."""
    problems: List[str] = []
    if not case.case_id or " " in case.case_id:
        problems.append("case_id must be a non-empty identifier with no spaces")
    if not case.purpose.strip():
        problems.append("purpose is mandatory: every case must state what it hardens")
    if case.asset_class not in ASSET_CLASSES:
        problems.append(f"unknown asset_class {case.asset_class!r}")
    elif case.asset_subtype not in ASSET_SUBTYPES[case.asset_class]:
        problems.append(f"asset_subtype {case.asset_subtype!r} is not a subtype "
                        f"of {case.asset_class!r}")
    if case.months < 6:
        problems.append("a case must carry at least six monthly snapshots")
    if case.opening_loan_count < 1:
        problems.append("opening_loan_count must be positive")
    if not case.dialects:
        problems.append("at least one source dialect must be declared")
    for dialect in case.dialects:
        if dialect not in DIALECTS:
            problems.append(f"unknown dialect {dialect!r}")
    for intent in case.economic_intent:
        if intent not in INTENT_MODIFIERS:
            problems.append(f"economic intent {intent!r} has no declared modifier")
    if case.expectation not in (EXPECT_SUCCESS, EXPECT_FAILURE):
        problems.append(f"unknown expectation {case.expectation!r}")
    if case.expectation == EXPECT_FAILURE:
        if case.expected_failure_class not in FailureClass.DECLARABLE:
            problems.append(
                f"expected_failure_class {case.expected_failure_class!r} is not "
                f"declarable; a case may never expect a production or simulator "
                f"defect. Declarable: {FailureClass.DECLARABLE}")
    elif case.expected_failure_class:
        problems.append("expected_failure_class is set on a success case")
    if case.source_portfolio_type == "acquired" and not case.acquisition_date:
        problems.append("an acquired portfolio must state its acquisition_date; "
                        "engine.provenance fails closed without one")
    problems += _validate_sources(case)
    if case.regime.applicability == REGIME_APPLICABLE:
        if not case.regime.regime_id:
            problems.append("an applicable regime must name a regime_id")
        elif check_regimes and case.regime.regime_id not in _supported_regimes():
            problems.append(
                f"regime {case.regime.regime_id!r} is not in the repository's "
                f"supported set {list(_supported_regimes())}")
    elif not case.regime.reason.strip():
        problems.append("a non-applicable regime must state a reason")
    for evolution in case.schema_evolution:
        if not 0 <= evolution.from_period < case.months:
            problems.append(f"schema evolution from_period "
                            f"{evolution.from_period} is outside the history")
        if not evolution.purpose.strip():
            problems.append(f"schema evolution {evolution.kind!r} has no purpose")
    if case.risk_intent.intended_status != RISK_GREEN and not case.risk_intent.targeted_limits:
        problems.append("a non-green risk intent must name the targeted limits")
    return problems


def validate_catalogue(*, check_regimes: bool = True) -> Dict[str, List[str]]:
    """``{case_id: problems}`` for every case that has any."""
    seen: Dict[str, int] = {}
    out: Dict[str, List[str]] = {}
    for case in CATALOGUE:
        problems = validate_manifest(case, check_regimes=check_regimes)
        seen[case.case_id] = seen.get(case.case_id, 0) + 1
        if seen[case.case_id] > 1:
            problems.append("duplicate case_id in the catalogue")
        if problems:
            out[case.case_id] = problems
    return out


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def write_manifest(case: CaseManifest, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case.to_dict(), indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")
    return path


def read_manifest(path: Path) -> CaseManifest:
    return CaseManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "ALL_DIALECTS", "CATALOGUE", "LARGE_SCALE_CASE", "PERFORMANCE_RUNS",
    "ManifestError",
    "PROFILES", "PROFILE_GUARD", "PROFILE_PERFORMANCE", "PROFILE_SMOKE",
    "PROFILE_STANDARD", "STANDARD_SCALE_CASES", "all_cases", "cases_for_profile",
    "get_case", "read_manifest", "validate_catalogue", "validate_manifest",
    "write_manifest",
]
