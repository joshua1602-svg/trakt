"""
mapping_candidate_finder.py
===========================

PART 4 — deterministic candidate shortlists, built BEFORE the LLM is called.

For each source column we shortlist plausible targets from (in priority order):
client memory, approved run overrides, aliases, the Gate 1 semantic alignment
adapter, the existing Pipeline MI contract, the formal registry, and
value-profile / domain compatibility. The LLM only ever sees these compact
shortlists plus the column evidence — never the raw file.

Artefacts:
    30_mapping_candidate_shortlist.csv / .json
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from engine.gate_1_alignment.semantic_alignment import normalise_name
from .pipeline_field_contract import _FIELD_META as _PIPE_META, RAW_TO_PIPELINE_FIELD

# Curated MI-useful header -> registry field map. These are regulatory-category
# fields that mi_only normally excludes, but which ARE useful for portfolio MI,
# static pools, valuation, geography and redemption analysis. The finder maps
# them and (for mi_only/mna_dd) keeps them in scope for the REVIEW (see
# MI_RELEVANT_FIELDS) without changing the regulatory funded-tape field scope.
MI_USEFUL_HEADER_TO_FIELD: Dict[str, str] = {
    "post code": "property_post_code",
    "postcode": "property_post_code",
    "property post code": "property_post_code",
    "latest property value": "current_valuation_amount",
    "current property value": "current_valuation_amount",
    "original property value": "original_valuation_amount",
    "full redemption date": "redemption_date",
    "redemption date": "redemption_date",
    "loan policy number": "original_underlying_exposure_identifier",
    "current ltv": "current_loan_to_value",
    "property region": "collateral_geography",
}

# Registry fields kept IN SCOPE for the mapping review under mi_only / mna_dd
# (they feed MI/static-pools, not the regulatory funded tape). Review-scoped
# only; resolve_field_scope and the central tape builder are unchanged.
MI_RELEVANT_FIELDS: Set[str] = {
    "property_post_code", "current_valuation_amount", "original_valuation_amount",
    "redemption_date", "original_underlying_exposure_identifier",
    "current_loan_to_value", "collateral_geography",
}

# Cashflow / servicing / ledger header patterns. Columns matching these have no
# canonical home in the funded registry; they are proposed as a cashflow-ledger
# extension domain rather than dumped into generic missing fields.
_CASHFLOW_LEDGER_PATTERNS = (
    "balance", "principal", "interest", "fees", "redemption", "cash paid",
    "cash allocated", "payment_allocation", "payment allocation", "b/f", "c/f",
    "brought forward", "carried forward", "cashflow", "arrears", "repayment",
)


def is_cashflow_ledger_header(header: str) -> bool:
    low = str(header).lower()
    return any(p in low for p in _CASHFLOW_LEDGER_PATTERNS)


def proposed_cashflow_field(header: str) -> str:
    """Snake-case a cashflow ledger header into a proposed extension field name."""
    s = re.sub(r"[^a-z0-9]+", "_", str(header).strip().lower()).strip("_")
    return f"cf_{s}" if s else ""


# Registry expected_type (registry "format") -> coarse evidence type bucket.
_TYPE_BUCKETS = {
    "date": {"date"},
    "decimal": {"amount", "numeric", "rate", "percentage"},
    "number": {"amount", "numeric", "rate", "percentage"},
    "numeric": {"amount", "numeric", "rate", "percentage"},
    "integer": {"numeric", "amount", "identifier"},
    "string": {"string", "free_text", "enum", "identifier", "postcode"},
    "text": {"string", "free_text", "enum"},
    "boolean": {"boolean"},
}


def _type_compatible(evidence_type: str, registry_format: str) -> bool:
    rf = (registry_format or "string").strip().lower()
    bucket = _TYPE_BUCKETS.get(rf, {rf})
    return evidence_type in bucket or rf in ("", "string", "text")


def _shortlist_for_column(
    ev: Dict[str, Any],
    registry_fields: Dict[str, Any],
    field_scope: Any,
    extra_in_scope: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    col = ev["source_column"]
    src_file = ev["source_file"]
    src_sheet = ev.get("source_sheet", "")
    etype = ev["data_type_guess"]
    extra_in_scope = extra_in_scope or set()
    out: List[Dict[str, Any]] = []

    def add(target: str, source: str, conf: float, reason: str):
        if not target:
            return
        in_scope = True
        scope_status = "in_scope"
        if field_scope is not None and getattr(field_scope, "is_excluded", None) \
                and target in registry_fields and field_scope.is_excluded(target):
            # MI-relevant fields stay in scope for the review (they feed MI/static
            # pools), surfaced as a useful-MI candidate rather than out of scope.
            if target in extra_in_scope:
                in_scope = True
                scope_status = "in_scope_mi_useful"
            else:
                in_scope = False
                scope_status = "out_of_scope"
        reg_fmt = (registry_fields.get(target, {}) or {}).get("format", "")
        is_pipeline = target in _PIPE_META
        type_ok = _type_compatible(etype, reg_fmt) if target in registry_fields else (
            _pipeline_type_ok(etype, target))
        # Material economic/regulatory targets always need approval.
        material = target in registry_fields and (
            (registry_fields.get(target, {}) or {}).get("category") == "regulatory"
            or (registry_fields.get(target, {}) or {}).get("core_canonical") is True)
        requires_approval = (not in_scope) or (source == "llm_suggested") or material \
            or (source in ("value_profile", "registry_description"))
        out.append({
            "source_file": src_file,
            "source_sheet": src_sheet,
            "source_column": col,
            "candidate_target_field": target,
            "candidate_source": source,
            "candidate_confidence": round(conf, 4),
            "candidate_reason": reason,
            "field_scope_status": scope_status,
            "type_compatible": bool(type_ok),
            "value_profile_compatible": bool(type_ok),
            "domain_compatible": True,
            "is_pipeline_field": is_pipeline,
            "requires_user_approval": bool(requires_approval),
        })

    # 1. client memory (highest priority deterministic source).
    if ev.get("known_client_memory_matches"):
        tgt = ev["known_client_memory_matches"].split(" ")[0]
        add(tgt, "client_memory", 0.99, "approved client mapping memory")
    # 2. curated MI-useful header -> field map (high confidence; header semantics).
    mi_tgt = MI_USEFUL_HEADER_TO_FIELD.get(normalise_name(col).replace("_", " ")) \
        or MI_USEFUL_HEADER_TO_FIELD.get(str(col).strip().lower())
    if mi_tgt and mi_tgt in registry_fields:
        add(mi_tgt, "mi_useful", 0.95, "curated MI-useful header mapping")
    # 3. alias.
    if ev.get("candidate_alias_matches"):
        tgt = ev["candidate_alias_matches"].split(" ")[0]
        add(tgt, "alias", 0.97, ev["candidate_alias_matches"])
    # 4. semantic alignment adapter.
    if ev.get("candidate_semantic_alignment_matches"):
        tgt = ev["candidate_semantic_alignment_matches"].split(" ")[0]
        add(tgt, "semantic_alignment", 0.85, ev["candidate_semantic_alignment_matches"])
    # 5. pipeline MI contract. An EXACT raw-header match to the working Pipeline
    #    MI rename map is a known, high-confidence contract mapping; a softer
    #    normalised match stays medium confidence (review).
    if ev.get("candidate_existing_pipeline_contract_fields"):
        exact = col in RAW_TO_PIPELINE_FIELD
        add(ev["candidate_existing_pipeline_contract_fields"], "pipeline_contract",
            0.96 if exact else 0.85,
            "exact match to existing Pipeline MI contract field" if exact
            else "matches existing Pipeline MI contract field")
    # 6. registry exact name.
    if ev.get("candidate_existing_registry_fields"):
        add(ev["candidate_existing_registry_fields"], "registry_description", 0.8,
            "registry field name match")
    # 7. cashflow / servicing / ledger header -> proposed extension field. These
    #    have no funded-registry home; propose them as a cashflow-ledger domain
    #    extension rather than dumping them in generic missing fields.
    if not out and is_cashflow_ledger_header(col):
        prop = proposed_cashflow_field(col)
        out.append({
            "source_file": src_file, "source_sheet": src_sheet, "source_column": col,
            "candidate_target_field": prop, "candidate_source": "cashflow_ledger",
            "candidate_confidence": 0.6,
            "candidate_reason": "cashflow/servicing/ledger field — propose extension",
            "field_scope_status": "proposed_extension", "type_compatible": True,
            "value_profile_compatible": True, "domain_compatible": True,
            "is_pipeline_field": False, "requires_user_approval": True,
        })
    # 8. value-profile compatible (a soft hint when nothing else matched).
    if not out and ev.get("candidate_value_profile_matches"):
        add("", "value_profile", 0.4, ev["candidate_value_profile_matches"])
    return out


def _pipeline_type_ok(etype: str, target: str) -> bool:
    meta = _PIPE_META.get(target, {})
    want = meta.get("type", "string")
    mapping = {
        "date": {"date"}, "numeric": {"numeric", "amount", "percentage", "rate"},
        "amount": {"amount", "numeric"}, "rate": {"rate", "numeric", "percentage"},
        "percentage": {"percentage", "numeric", "rate"},
        "identifier": {"identifier", "string"}, "enum": {"enum", "string"},
        "string": {"string", "free_text", "enum", "identifier"},
        "boolean": {"boolean"},
    }
    return etype in mapping.get(want, {want}) or want in ("string", "")


def build_candidate_shortlist(
    evidence_rows: List[Dict[str, Any]],
    registry_fields: Optional[Dict[str, Any]] = None,
    field_scope: Any = None,
    extra_in_scope: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    registry_fields = registry_fields or {}
    rows: List[Dict[str, Any]] = []
    for ev in evidence_rows:
        rows.extend(_shortlist_for_column(ev, registry_fields, field_scope, extra_in_scope))
    return rows


_SHORTLIST_COLUMNS = [
    "source_file", "source_sheet", "source_column", "candidate_target_field",
    "candidate_source", "candidate_confidence", "candidate_reason",
    "field_scope_status", "type_compatible", "value_profile_compatible",
    "domain_compatible", "is_pipeline_field", "requires_user_approval",
]


def column_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """Composite key (source_file, source_sheet, source_column) — unique across files."""
    return (row.get("source_file", ""), row.get("source_sheet", ""),
            row.get("source_column", ""))


def write_shortlist_artifacts(rows: List[Dict[str, Any]], output_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "30_mapping_candidate_shortlist.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_SHORTLIST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _SHORTLIST_COLUMNS})
    json_path = out_dir / "30_mapping_candidate_shortlist.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}


def shortlist_by_column(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["source_column"], []).append(r)
    return out


def shortlist_by_key(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """Group shortlist rows by composite key (file, sheet, column) — multi-file safe."""
    out: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(column_key(r), []).append(r)
    return out
