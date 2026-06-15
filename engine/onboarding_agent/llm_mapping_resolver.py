"""
llm_mapping_resolver.py
=======================

v1 asset/regime-aware mapping resolver.

For each source column it resolves a mapping against the REQUIRED target data
contract (not a generic field universe). The LLM is the semantic resolver when a
callable is supplied; otherwise a deterministic resolution is used (existing
shortlist best candidate + contract synonyms) so the architecture still produces
its artefacts and the deterministic behaviour is preserved.

Every resolved mapping is then validated by the deterministic backstop. The LLM
never finalises a mapping.

Artefacts:
    31_llm_mapping_resolver.csv / .json / _summary.md
    31_llm_usage_summary.json
"""

from __future__ import annotations

import csv
import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .required_target_contract import (
    contract_domain_map, contract_field_set, synonym_index,
)

# Allowed resolver decisions.
MAP_EXISTING = "map_existing_target"
PROPOSE_NEW = "propose_new_target_field"
IGNORE = "ignore_source_field"
NEEDS_CLARIFICATION = "needs_user_clarification"
REJECT = "reject_candidate"
HEADER_ISSUE = "header_or_parse_issue"

_SOURCE_RANK = {"client_memory": 0, "mi_useful": 1, "pipeline_contract": 2, "alias": 3,
                "semantic_alignment": 4, "registry_description": 5, "cashflow_ledger": 6,
                "value_profile": 7}

_CONTRACT_PROMPT = """\
You map a lender source column to a REQUIRED target data contract for the
detected asset class / regime. Map to a contract target_field when one fits;
otherwise return propose_new_target_field, ignore_source_field,
needs_user_clarification, reject_candidate or header_or_parse_issue. Use only the
provided evidence; do NOT invent broad generic fields. Return STRUCTURED JSON: a
list of objects with keys source_file, source_column, resolved_target_field,
resolved_domain, decision, confidence, rationale, alternative_targets,
new_field_candidate, out_of_scope_reason, required_user_question.
"""


def _best_candidate(cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    have = [c for c in cands if c.get("candidate_target_field")]
    if not have:
        return None
    return sorted(have, key=lambda c: (_SOURCE_RANK.get(c["candidate_source"], 9),
                                       -float(c.get("candidate_confidence", 0))))[0]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


_CONF_WORDS = {"high": 0.9, "medium": 0.6, "low": 0.3, "no_match": 0.0, "none": 0.0}


def _coerce_conf(value: Any, default: float = 0.6) -> float:
    """Coerce an LLM confidence (number or word) to a float; never raises."""
    if isinstance(value, (int, float)):
        return float(value)
    v = str(value or "").strip().lower()
    if v in _CONF_WORDS:
        return _CONF_WORDS[v]
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _resolve_deterministic(
    ev: Dict[str, Any], cands: List[Dict[str, Any]],
    contract_fields: set, domain_map: Dict[str, str], syn_idx: Dict[str, str],
) -> Dict[str, Any]:
    col = ev.get("source_column", "")
    if re.match(r"^Unnamed: ?\d+$", str(col)) or str(col).strip() == "":
        return _row(ev, "", "", HEADER_ISSUE, 0.0,
                    "blank/Unnamed header — header detection or parse issue")
    best = _best_candidate(cands)
    # 1. deterministic best candidate (preserves existing behaviour).
    if best is not None:
        tgt = best["candidate_target_field"]
        if best["candidate_source"] == "cashflow_ledger":
            return _row(ev, tgt, "cashflow_ledger", PROPOSE_NEW,
                        float(best.get("candidate_confidence", 0.6)),
                        "cashflow/ledger field — propose contract extension",
                        new_field=tgt)
        domain = domain_map.get(tgt, ev.get("file_domain_guess", "") or "unknown")
        return _row(ev, tgt, domain, MAP_EXISTING,
                    float(best.get("candidate_confidence", 0.0)),
                    f"deterministic {best['candidate_source']} match")
    # 2. contract synonym match.
    key = _norm(col)
    tgt = syn_idx.get(key) or syn_idx.get(key.replace("_", " "))
    if tgt:
        return _row(ev, tgt, domain_map.get(tgt, "unknown"), MAP_EXISTING, 0.7,
                    "matched required-contract synonym")
    # 3. cashflow-like proposed extension.
    from .mapping_candidate_finder import is_cashflow_ledger_header, proposed_cashflow_field
    if is_cashflow_ledger_header(col):
        return _row(ev, proposed_cashflow_field(col), "cashflow_ledger", PROPOSE_NEW, 0.5,
                    "cashflow/ledger header — propose extension",
                    new_field=proposed_cashflow_field(col))
    # 4. nothing — ignore or ask.
    if ev.get("null_rate", 0) >= 0.999 or ev.get("distinct_count", 1) == 0:
        return _row(ev, "", "", IGNORE, 0.0, "empty / 100% null column")
    return _row(ev, "", "", NEEDS_CLARIFICATION, 0.0,
                "no confident target in the required contract")


def _row(ev, target, domain, decision, conf, rationale, new_field="",
         alternatives=None, oos="", question="") -> Dict[str, Any]:
    return {
        "source_file": ev.get("source_file", ""),
        "source_sheet": ev.get("source_sheet", ""),
        "source_column": ev.get("source_column", ""),
        "resolved_target_field": target,
        "resolved_domain": domain,
        "decision": decision,
        "confidence": round(float(conf), 4),
        "rationale": rationale,
        "alternative_targets": alternatives or [],
        "new_field_candidate": new_field,
        "out_of_scope_reason": oos,
        "required_user_question": question,
        "llm_used": False,
        "llm_batch_id": "",
    }


def _build_package(ev, cands, context, contract) -> Dict[str, Any]:
    """Compact, redacted LLM package for one column (evidence + contract subset)."""
    domain = ev.get("file_domain_guess", "")
    subset = [r for r in contract if r["domain"] == domain] or contract
    return {
        "onboarding_context": {k: context.get(k) for k in
                               ("asset_class", "jurisdiction", "product_type",
                                "reporting_regime", "required_domains")},
        "required_target_contract": [{"target_field": r["target_field"], "domain": r["domain"],
                                      "required_level": r["required_level"],
                                      "expected_type": r["expected_type"]} for r in subset[:40]],
        "source_file": ev.get("source_file"), "source_sheet": ev.get("source_sheet"),
        "detected_source_domain": ev.get("file_domain_guess"),
        "source_column": ev.get("source_column"),
        "nearby_columns": ev.get("nearby_columns"),
        "sample_values": ev.get("sample_values_distinct_redacted"),
        "null_ratio": ev.get("null_rate"), "value_profile": ev.get("candidate_value_profile_matches"),
        "deterministic_candidates": [c.get("candidate_target_field") for c in cands],
        "current_deterministic_suggestion": (_best_candidate(cands) or {}).get("candidate_target_field", ""),
    }


def resolve_mappings(
    evidence_rows: List[Dict[str, Any]],
    shortlist_by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
    context: Dict[str, Any],
    contract: List[Dict[str, Any]],
    llm_callable: Optional[Callable[[str], str]] = None,
    only_unresolved: bool = False,
    max_items: int = 60,
    cost_per_call_gbp: float = 0.01,
    max_cost_gbp: float = 1.0,
) -> Dict[str, Any]:
    """Resolve every source column against the required contract."""
    cfields = contract_field_set(contract)
    dmap = contract_domain_map(contract)
    syn = synonym_index(contract)

    def key(ev):
        return (ev.get("source_file", ""), ev.get("source_sheet", ""), ev.get("source_column", ""))

    resolved: List[Dict[str, Any]] = []
    det_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for ev in evidence_rows:
        r = _resolve_deterministic(ev, shortlist_by_key.get(key(ev), []), cfields, dmap, syn)
        det_by_key[key(ev)] = r
        resolved.append(r)

    usage = {"llm_enabled": False, "field_llm_callable_present": bool(llm_callable),
             "calls_completed": 0, "items_sent": 0, "estimated_cost_gbp": 0.0,
             "llm_batch_id": "", "resolver": True, "parse_status": "", "parse_error": "",
             "rows_llm_reviewed": 0, "eligible_field_rows": 0,
             "eligible_reason_counts": {}, "field_rows_selected_for_llm": 0,
             "field_rows_skipped_due_to_cap": 0, "field_rows_skipped_reason_counts": {}}

    # --- Explicit field-level LLM eligibility (computed + persisted) ---
    def _eligible(r) -> bool:
        if r["decision"] in (IGNORE, HEADER_ISSUE):
            return False
        return (r["decision"] in (NEEDS_CLARIFICATION, PROPOSE_NEW, REJECT)
                or (r["decision"] == MAP_EXISTING and r["confidence"] < 0.85))

    def _skip_reason(r) -> str:
        if r["decision"] == IGNORE:
            return "ignored_null_empty"
        if r["decision"] == HEADER_ISSUE:
            return "header_failure"
        return "auto_approved_high_confidence"

    eligible = [ev for ev in evidence_rows if _eligible(det_by_key[key(ev)])]
    ineligible = [ev for ev in evidence_rows if not _eligible(det_by_key[key(ev)])]
    elig_reasons: Dict[str, int] = {}
    for ev in eligible:
        d = det_by_key[key(ev)]["decision"]
        elig_reasons[d] = elig_reasons.get(d, 0) + 1
    skip_reasons: Dict[str, int] = {}
    for ev in ineligible:
        rr = _skip_reason(det_by_key[key(ev)])
        skip_reasons[rr] = skip_reasons.get(rr, 0) + 1
    usage["eligible_field_rows"] = len(eligible)
    usage["eligible_reason_counts"] = elig_reasons
    usage["field_rows_skipped_reason_counts"] = skip_reasons

    # When only_unresolved is False, also send confident maps (still excludes
    # ignored/header junk) so the LLM can audit them.
    candidate_rows = eligible if only_unresolved else (eligible + [
        ev for ev in ineligible
        if det_by_key[key(ev)]["decision"] not in (IGNORE, HEADER_ISSUE)])

    # LLM resolution over the eligible rows (capped).
    if llm_callable is not None and candidate_rows and cost_per_call_gbp <= max_cost_gbp:
        from .llm_json import extract_json_list
        usage["llm_enabled"] = True
        targets = candidate_rows[:max_items]
        usage["field_rows_selected_for_llm"] = len(targets)
        usage["field_rows_skipped_due_to_cap"] = max(0, len(candidate_rows) - max_items)
        batch_id = "res_" + uuid.uuid4().hex[:10]
        packages = [_build_package(ev, shortlist_by_key.get(key(ev), []), context, contract)
                    for ev in targets]
        prompt = _CONTRACT_PROMPT + "\nPACKAGES = " + json.dumps(packages, default=str)
        raw = llm_callable(prompt)
        parsed, parse_status, parse_error = extract_json_list(raw)
        usage.update(calls_completed=1, items_sent=len(packages),
                     estimated_cost_gbp=round(cost_per_call_gbp, 6), llm_batch_id=batch_id,
                     parse_status=parse_status, parse_error=parse_error)
        # Match by (file,column) then by normalised column (models don't always
        # echo the file name or exact casing).
        def _nk(s):
            return _norm(str(s))
        by_fc = {(p.get("source_file", ""), p.get("source_column", "")): p for p in parsed}
        by_col = {_nk(p.get("source_column", "")): p for p in parsed}
        reviewed = 0
        for ev in targets:
            p = by_fc.get((ev.get("source_file", ""), ev.get("source_column", ""))) \
                or by_col.get(_nk(ev.get("source_column", "")))
            if not p:
                continue
            reviewed += 1
            tgt = str(p.get("resolved_target_field", "") or "").strip()
            decision = str(p.get("decision", "") or "").strip() or MAP_EXISTING
            # Hard rule: LLM may only map to a contract field, else propose_new.
            if tgt and tgt not in cfields and decision == MAP_EXISTING:
                decision = PROPOSE_NEW
            det_by_key[key(ev)].update({
                "resolved_target_field": tgt or det_by_key[key(ev)]["resolved_target_field"],
                "resolved_domain": p.get("resolved_domain") or dmap.get(tgt, ""),
                "decision": decision, "confidence": _coerce_conf(p.get("confidence")),
                "rationale": "LLM: " + str(p.get("rationale", "")), "llm_used": True,
                "llm_batch_id": batch_id,
                "new_field_candidate": p.get("new_field_candidate", ""),
                "required_user_question": p.get("required_user_question", ""),
                "out_of_scope_reason": p.get("out_of_scope_reason", ""),
            })
        usage["rows_llm_reviewed"] = reviewed

    return {"resolved": list(det_by_key.values()), "usage": usage}


_RESOLVER_COLUMNS = [
    "source_file", "source_sheet", "source_column", "resolved_target_field",
    "resolved_domain", "decision", "confidence", "rationale", "new_field_candidate",
    "out_of_scope_reason", "required_user_question", "llm_used", "llm_batch_id",
]


def write_resolver_artifacts(result: Dict[str, Any], output_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = result["resolved"]
    csv_path = out_dir / "31_llm_mapping_resolver.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_RESOLVER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _RESOLVER_COLUMNS})
    (out_dir / "31_llm_mapping_resolver.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8")
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    md = ["# LLM mapping resolver (against required contract)", "",
          f"{len(rows)} columns resolved. LLM used: {result['usage'].get('llm_enabled')}.", ""]
    for d, c in sorted(counts.items()):
        md.append(f"- {d}: {c}")
    (out_dir / "31_llm_mapping_resolver_summary.md").write_text("\n".join(md) + "\n",
                                                                encoding="utf-8")
    (out_dir / "31_llm_usage_summary.json").write_text(
        json.dumps(result["usage"], indent=2, default=str), encoding="utf-8")
    return {"csv": str(csv_path)}
