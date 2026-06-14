"""
domain_coverage.py
==================

PART 5 / PART 6 — domain-aware coverage assessment.

The Onboarding Agent expects DATA DOMAINS, not a fixed number of files. A single
container (e.g. a combined ``master_loan_tape``) may carry several domains at
once; conversely a client may split loan / collateral / cashflow across files.
Either way the engine reasons about which domains are *covered by mapped /
in-scope fields*, never about whether a particular file is present.

Domains
-------
    loan, borrower, collateral, cashflow, pipeline,
    warehouse_terms, securitisation_terms, unknown

Coverage is mode-aware (PART 6). The same detected gap is a blocking domain in
one mode and merely a visible diligence gap in another.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from engine.gate_1_alignment.semantic_alignment import tokenize
from .onboarding_models import DomainCoverage

# ---------------------------------------------------------------------------
# Domain taxonomy
# ---------------------------------------------------------------------------

LOAN = "loan"
BORROWER = "borrower"
COLLATERAL = "collateral"
CASHFLOW = "cashflow"
PIPELINE = "pipeline"
WAREHOUSE = "warehouse_terms"
SECURITISATION = "securitisation_terms"
UNKNOWN = "unknown"

DOMAINS = [LOAN, BORROWER, COLLATERAL, CASHFLOW, PIPELINE, WAREHOUSE, SECURITISATION]

DOMAIN_LABELS = {
    LOAN: "Loan/borrower",
    BORROWER: "Borrower",
    COLLATERAL: "Collateral/property",
    CASHFLOW: "Cashflow/performance",
    PIPELINE: "Pipeline/origination",
    WAREHOUSE: "Warehouse/securitisation terms",
    SECURITISATION: "Securitisation terms",
}

# Status values
COVERED = "covered"
PARTIAL = "partially_covered"
MISSING = "missing"
OUT_OF_SCOPE = "out_of_scope"

# ---------------------------------------------------------------------------
# Field -> domain rules (driven by registry layer + canonical-field name)
# ---------------------------------------------------------------------------

# Registry `layer` -> domain hint.
_LAYER_DOMAIN = {
    "collateral": COLLATERAL,
    "performance": CASHFLOW,
}

# Token signals on canonical-field / column names. Order matters only in that a
# field/column may pick up several domains (that is expected and allowed).
_DOMAIN_TOKENS: Dict[str, List[str]] = {
    COLLATERAL: [
        "property", "valuation", "collateral", "postcode", "post_code", "ltv",
        "loan_to_value", "epc", "occupancy", "security",
    ],
    CASHFLOW: [
        "payment", "cashflow", "cash_flow", "redemption", "prepayment",
        "arrears", "instalment", "installment", "repayment", "recover",
        "scheduled_interest", "scheduled_principal", "servicer", "fee_amount",
        "default_amount", "total_cashflow",
    ],
    BORROWER: [
        "borrower", "obligor", "employment", "income", "customer", "guarantor",
        "date_of_birth", "occupation",
    ],
    PIPELINE: [
        "application", "pipeline", "broker", "offer", "completion",
        "expected_fund", "expected_completion", "requested_loan", "product_name",
        "decision", "case_id", "origination_channel",
    ],
    WAREHOUSE: [
        "warehouse", "advance_rate", "drawn", "undrawn", "facility", "margin",
        "borrowing_base", "eligibility",
    ],
    SECURITISATION: [
        "securit", "tranche", "note_class", "issuer", "spv", "rmbs", "abs",
    ],
    LOAN: [
        "loan", "balance", "principal", "interest_rate", "maturity",
        "origination", "account", "amortis", "amortiz", "drawdown",
    ],
}

# Representative canonical fields whose presence determines covered / partial.
# Only those that survive the active field scope count as "required".
_DOMAIN_REQUIRED_FIELDS: Dict[str, List[str]] = {
    LOAN: [
        "loan_identifier", "current_principal_balance", "original_principal_balance",
        "current_interest_rate", "origination_date", "maturity_date",
    ],
    BORROWER: ["borrower_identifier"],
    COLLATERAL: [
        "property_post_code", "current_valuation_amount", "current_loan_to_value",
    ],
    # cashflow / pipeline / warehouse / securitisation are presence-based
    # (their canonical universe rarely survives MI scope and pipeline/warehouse
    # live outside the loan registry), so required-field lists stay empty.
    CASHFLOW: [],
    PIPELINE: [],
    WAREHOUSE: [],
    SECURITISATION: [],
}

# File classification -> primary domain hint.
_CLASSIFICATION_DOMAIN = {
    "current_loan_report": LOAN,
    "historical_loan_report": LOAN,
    "collateral_report": COLLATERAL,
    "cashflow_report": CASHFLOW,
    "pipeline_report": PIPELINE,
    "warehouse_agreement": WAREHOUSE,
    "securitisation_document": SECURITISATION,
    "investor_report": CASHFLOW,
}

# ---------------------------------------------------------------------------
# Mode policy for domain requirements (PART 6)
# ---------------------------------------------------------------------------

# required  - domain expected; a gap is surfaced if absent
# blocking  - absence blocks handoff
# out_of_scope - domain not assessed for this mode
_MODE_DOMAIN_POLICY: Dict[str, Dict[str, Set[str]]] = {
    "mi_only": {
        "required": {LOAN, BORROWER},
        "optional": {COLLATERAL, CASHFLOW, PIPELINE},
        "out_of_scope": {WAREHOUSE, SECURITISATION},
        "blocking": {LOAN},
    },
    "mna_dd": {
        "required": {LOAN, BORROWER, COLLATERAL, CASHFLOW, PIPELINE},
        "optional": {WAREHOUSE, SECURITISATION},
        "out_of_scope": set(),
        "blocking": {LOAN},
    },
    "regulatory_mi": {
        "required": {LOAN, BORROWER, COLLATERAL},
        "optional": {CASHFLOW, PIPELINE},
        "out_of_scope": {WAREHOUSE, SECURITISATION},
        "blocking": {LOAN, COLLATERAL},
    },
    "warehouse_securitisation": {
        "required": {LOAN, COLLATERAL, CASHFLOW, WAREHOUSE},
        "optional": {BORROWER, PIPELINE, SECURITISATION},
        "out_of_scope": set(),
        "blocking": {LOAN, COLLATERAL, CASHFLOW, WAREHOUSE},
    },
}


def _mode_policy(mode: str) -> Dict[str, Set[str]]:
    return _MODE_DOMAIN_POLICY.get(mode, _MODE_DOMAIN_POLICY["regulatory_mi"])


# ---------------------------------------------------------------------------
# Field / column -> domain inference
# ---------------------------------------------------------------------------


def field_domains(field_name: str, registry_meta: Optional[Dict[str, Any]] = None) -> Set[str]:
    """Return the data domain(s) a canonical field belongs to."""
    domains: Set[str] = set()
    if not field_name:
        return domains
    meta = registry_meta or {}
    layer = (meta.get("layer") or "").lower()
    if layer in _LAYER_DOMAIN:
        domains.add(_LAYER_DOMAIN[layer])
    name = field_name.lower()
    for domain, tokens in _DOMAIN_TOKENS.items():
        if any(t in name for t in tokens):
            domains.add(domain)
    # A pure core loan-level field with no other signal is a loan field.
    if not domains:
        domains.add(LOAN)
    # Identifiers / balances / rates always count toward the loan domain too.
    if any(t in name for t in ("identifier", "balance", "principal", "interest_rate",
                               "maturity", "origination", "account")):
        domains.add(LOAN)
    return domains


def _column_domains(column: str) -> Set[str]:
    """Domains implied directly by a raw source column name."""
    domains: Set[str] = set()
    norm = str(column).lower().replace(" ", "_")
    for domain, tokens in _DOMAIN_TOKENS.items():
        if any(t in norm for t in tokens):
            domains.add(domain)
    return domains


def detect_file_domains(
    classification: str,
    columns: Iterable[str],
    mapped_fields: Iterable[str],
    registry_fields: Dict[str, Any],
) -> List[str]:
    """Detect the domain(s) one container (file) carries.

    Evidence (PART 5): file classification + raw column names + matched canonical
    fields + registry layer/category metadata. Never relies on file name alone.
    """
    domains: Set[str] = set()
    primary = _CLASSIFICATION_DOMAIN.get(classification)
    if primary:
        domains.add(primary)
    for col in columns:
        domains |= _column_domains(col)
    for f in mapped_fields:
        domains |= field_domains(f, registry_fields.get(f, {}))
    if not domains:
        domains.add(UNKNOWN)
    # Stable, taxonomy-ordered output.
    ordered = [d for d in DOMAINS if d in domains]
    if UNKNOWN in domains and not ordered:
        ordered = [UNKNOWN]
    return ordered


# ---------------------------------------------------------------------------
# Coverage assessment
# ---------------------------------------------------------------------------


def assess_domain_coverage(
    inventory: List[Any],
    mapping_candidates: List[Any],
    field_scope: Any,
    mode: str,
    registry_fields: Dict[str, Any],
    document_extractions: Optional[List[Any]] = None,
    column_index: Optional[Dict[str, List[str]]] = None,
) -> List[DomainCoverage]:
    """Assess coverage for every data domain for the selected ``mode``.

    ``inventory`` items must already carry ``domains_detected`` (see
    :func:`annotate_inventory_domains`). ``mapping_candidates`` provide the
    in-scope mapped canonical fields used to count coverage.
    """
    policy = _mode_policy(mode)
    included = getattr(field_scope, "included_fields", set()) or set()

    # canonical field -> domains, restricted to mapped + in-scope fields.
    mapped_by_domain: Dict[str, Set[str]] = {d: set() for d in DOMAINS}
    files_by_domain: Dict[str, Set[str]] = {d: set() for d in DOMAINS}

    for m in mapping_candidates:
        canon = getattr(m, "candidate_canonical_field", "") or ""
        if not canon:
            continue
        if included and canon not in included:
            continue
        for d in field_domains(canon, registry_fields.get(canon, {})):
            if d in mapped_by_domain:
                mapped_by_domain[d].add(canon)
                files_by_domain[d].add(getattr(m, "source_file", ""))

    # Domain evidence from file inventory (classification / column tokens).
    for item in inventory:
        for d in getattr(item, "domains_detected", []) or []:
            if d in files_by_domain:
                files_by_domain[d].add(item.file_name)

    # Warehouse / securitisation terms are evidenced by extracted document facts.
    doc_fields_present = {
        getattr(x, "field", "") for x in (document_extractions or [])
        if getattr(x, "value", "")
    }
    if any("warehouse" in f for f in doc_fields_present):
        # already added by classification, but ensure the domain has evidence
        files_by_domain[WAREHOUSE]  # noqa: B018 - dict access ensures key exists
    if any("securit" in f for f in doc_fields_present):
        pass

    rows: List[DomainCoverage] = []
    for domain in DOMAINS:
        required_fields_universe = _DOMAIN_REQUIRED_FIELDS.get(domain, [])
        required_in_scope = [
            f for f in required_fields_universe
            if (not included) or f in included
        ]
        mapped = sorted(mapped_by_domain[domain])
        source_files = sorted(f for f in files_by_domain[domain] if f)
        has_source = bool(source_files) or bool(mapped)

        is_out_of_scope = domain in policy["out_of_scope"]
        is_required = domain in policy["required"]
        is_blocking_domain = domain in policy["blocking"]

        missing_required = [f for f in required_in_scope if f not in mapped]

        # ---- status ----
        if is_out_of_scope:
            # Domain not assessed for this mode (e.g. regulatory non-core in
            # mi_only, or warehouse terms in regulatory_mi).
            status = OUT_OF_SCOPE
        elif required_in_scope:
            if not mapped and not has_source:
                status = MISSING if is_required else OUT_OF_SCOPE
            elif not missing_required:
                status = COVERED
            elif not mapped:
                # Domain files present but no required canonical field mapped yet.
                status = PARTIAL if has_source else (MISSING if is_required else OUT_OF_SCOPE)
            else:
                status = PARTIAL
        else:
            # Presence-based domain (no required canonical fields in scope).
            if has_source:
                status = COVERED
            elif is_required:
                status = MISSING
            else:
                status = OUT_OF_SCOPE

        blocking = bool(is_blocking_domain and status == MISSING)

        rows.append(
            DomainCoverage(
                domain=domain,
                status=status,
                source_files=source_files,
                mapped_fields_count=len(mapped),
                required_fields_count=len(required_in_scope),
                missing_required_fields=missing_required,
                blocking=blocking,
                notes=_coverage_note(domain, status, source_files, mapped, missing_required),
            )
        )
    return rows


def _coverage_note(
    domain: str, status: str, source_files: List[str], mapped: List[str],
    missing_required: List[str],
) -> str:
    """PART 11 — plain-English, domain-based wording (never 'file missing')."""
    label = DOMAIN_LABELS.get(domain, domain)
    files = ", ".join(source_files) if source_files else "the provided onboarding pack"
    fields = ", ".join(mapped[:6]) if mapped else ""
    if status == COVERED:
        if fields:
            return f"{label} domain covered by {files} using {fields}."
        return f"{label} domain covered by {files}."
    if status == PARTIAL:
        miss = ", ".join(missing_required) if missing_required else "some expected fields"
        if fields:
            return f"{label} domain partially covered by {files}. Missing {miss}."
        return f"{label} domain partially covered by {files}. Missing {miss}."
    if status == MISSING:
        return f"No {label.lower()} fields were found in the provided onboarding pack."
    return f"{label} domain is out of scope for this onboarding mode."


# ---------------------------------------------------------------------------
# Inventory annotation + artefact writers
# ---------------------------------------------------------------------------


def annotate_inventory_domains(
    inventory: List[Any],
    mapping_candidates: List[Any],
    registry_fields: Dict[str, Any],
    column_index: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Populate ``item.domains_detected`` for every inventory item in place."""
    mapped_by_file: Dict[str, List[str]] = {}
    for m in mapping_candidates:
        canon = getattr(m, "candidate_canonical_field", "") or ""
        if canon:
            mapped_by_file.setdefault(getattr(m, "source_file", ""), []).append(canon)
    column_index = column_index or {}
    for item in inventory:
        cols = column_index.get(item.file_name, [])
        item.domains_detected = detect_file_domains(
            item.classification, cols, mapped_by_file.get(item.file_name, []),
            registry_fields,
        )


def load_coverage(json_path: str | Path) -> List[DomainCoverage]:
    """Reconstruct ``List[DomainCoverage]`` from a 17_domain_coverage.json file."""
    p = Path(json_path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    rows: List[DomainCoverage] = []
    for d in data:
        rows.append(DomainCoverage(
            domain=d.get("domain", ""),
            status=d.get("status", MISSING),
            source_files=list(d.get("source_files", []) or []),
            mapped_fields_count=int(d.get("mapped_fields_count", 0) or 0),
            required_fields_count=int(d.get("required_fields_count", 0) or 0),
            missing_required_fields=list(d.get("missing_required_fields", []) or []),
            blocking=bool(d.get("blocking", False)),
            notes=d.get("notes", ""),
        ))
    return rows


def rebuild_coverage(
    project_dir: str | Path,
    registry_path: str | Path,
    mode: str,
    regulatory_reporting_enabled: bool = False,
) -> List[DomainCoverage]:
    """Rebuild domain coverage from the on-disk onboarding artefacts (01 + 05)."""
    from engine.gate_1_alignment.semantic_alignment import load_field_registry
    from .field_scope import resolve_field_scope
    from .mode_policy import load_mode_policy
    from .onboarding_models import FileInventoryItem, MappingCandidate

    project_dir = Path(project_dir)
    inv_raw = json.loads((project_dir / "01_file_inventory.json").read_text(encoding="utf-8")) \
        if (project_dir / "01_file_inventory.json").exists() else []
    map_raw = json.loads((project_dir / "05_mapping_candidates.json").read_text(encoding="utf-8")) \
        if (project_dir / "05_mapping_candidates.json").exists() else []

    inventory = [FileInventoryItem(**{k: v for k, v in i.items()
                                      if k in FileInventoryItem.__dataclass_fields__}) for i in inv_raw]
    mapping_candidates = [MappingCandidate(**{k: v for k, v in m.items()
                                              if k in MappingCandidate.__dataclass_fields__}) for m in map_raw]

    registry_fields = load_field_registry(Path(registry_path)).get("fields", {}) or {}
    policy = load_mode_policy(mode)
    field_scope = resolve_field_scope(
        str(registry_path), policy, regulatory_reporting_enabled=regulatory_reporting_enabled
    )
    annotate_inventory_domains(inventory, mapping_candidates, registry_fields)
    return assess_domain_coverage(
        inventory, mapping_candidates, field_scope, mode, registry_fields
    )


def write_domain_coverage_artifacts(
    coverage: List[DomainCoverage], out_dir: str | Path
) -> List[str]:
    """Write ``17_domain_coverage.csv`` and ``.json``; return the file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [f.name for f in dataclasses.fields(DomainCoverage)]

    csv_path = out_dir / "17_domain_coverage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in coverage:
            d = dataclasses.asdict(row)
            for k, v in d.items():
                if isinstance(v, list):
                    d[k] = "; ".join(str(x) for x in v)
            writer.writerow(d)

    json_path = out_dir / "17_domain_coverage.json"
    json_path.write_text(
        json.dumps([dataclasses.asdict(r) for r in coverage], indent=2, default=str),
        encoding="utf-8",
    )
    return [str(csv_path), str(json_path)]
