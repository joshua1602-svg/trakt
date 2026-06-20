"""
target_first_llm_advisor.py
===========================

Target-contract-first LLM ADVISOR.

The deterministic target-first flow (28a coverage matrix, 28b residual register,
28c decision queue, 34 operator decisions, 35 application log) is the source of
truth and is never mutated here. This module adds an OPTIONAL advisory layer that
operates on the SAME target-first state: it takes the 28c Gate 4 decisions (plus
28a/28b evidence + file/domain context) and asks the LLM to advise on each
decision — confirm the deterministic selected source, choose a better candidate
(from the supplied candidates only), confirm a config/default/ND, or mark
not-applicable.

It is advisory only. It does NOT:
  * remove rows from 28c, mutate 28a, change coverage/blocking/Gate 5 readiness,
  * apply approved decisions (only the operator-approved 34 file does that).

Artefacts (written only when the target advisor is enabled):
    36_target_first_llm_recommendations.csv / .json / _summary.md
    36_target_first_llm_raw_response.json
    36_target_first_llm_usage_summary.json
"""

from __future__ import annotations

import csv
import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Allowed advisory actions (superset of the operator actions + escalation).
ALLOWED_ACTIONS = {
    "confirm_selected", "choose_alternative", "merge_or_reconcile",
    "configure_static_value", "confirm_default_or_nd", "mark_not_applicable",
    "provide_source_mapping", "defer", "reject_recommendation",
    "requires_operator_review",
}

# Advice statuses.
ADVISED = "advised"
NO_ADVICE = "no_advice"
INVALID_RESPONSE = "invalid_response"
SKIPPED_BUDGET = "skipped_budget"
SKIPPED_NO_DECISIONS = "skipped_no_decisions"
PARSE_FAILED = "parse_failed"

# Deterministic default action per decision type (fallback / no-LLM display).
_DEFAULT_ACTION = {
    "source_priority_confirmation": "confirm_selected",
    "conflicting_source_candidates": "confirm_selected",
    "missing_required_target": "requires_operator_review",
    "config_value_required": "configure_static_value",
    "config_confirmation": "configure_static_value",
    "nd_default_confirmation": "confirm_default_or_nd",
    "default_or_nd_confirmation": "confirm_default_or_nd",
    "not_applicable_confirmation": "mark_not_applicable",
    "value_compatibility_conflict": "requires_operator_review",
    "reporting_extension_candidate": "requires_operator_review",
    "parse_or_header_blocker": "requires_operator_review",
}

_PROMPT = """\
You are a TARGET-DATA-CONTRACT advisor for a lender onboarding pack. You are given
GATE 4 target decisions (one per target field that needs an operator decision),
each with the deterministic recommendation, the selected source, the ONLY allowed
candidate sources, and supporting evidence.

For EACH decision:
 1. Decide whether the deterministic selected source should be confirmed.
 2. If not, choose the best alternative ONLY from candidate_source_columns.
 3. Explain using source type/classification, field naming, current/original/recency
    semantics, data domain and overlap evidence.
 4. State whether human confirmation is still required.
 5. NEVER invent a source file/column not present in candidate_source_columns.

Return STRICT JSON of this exact shape and nothing else:
{"recommendations": [
  {"decision_id": "...", "recommended_action": "confirm_selected|choose_alternative|
    merge_or_reconcile|configure_static_value|confirm_default_or_nd|mark_not_applicable|
    provide_source_mapping|defer|reject_recommendation|requires_operator_review",
   "recommended_source_file": "", "recommended_source_column": "",
   "recommended_configured_value": "", "default_confirmation": false,
   "not_applicable": false, "confidence": 0.0, "rationale": "",
   "alternative_assessment": "", "operator_note": "",
   "requires_human_confirmation": true}
]}
Echo each decision_id exactly.
"""

_RECOMMENDATION_COLUMNS = [
    "decision_id", "target_field", "decision_type", "llm_recommended_action",
    "llm_recommended_source_file", "llm_recommended_source_column",
    "llm_recommended_configured_value", "llm_recommended_default_confirmation",
    "llm_recommended_not_applicable", "llm_confidence", "llm_rationale",
    "llm_alternative_assessment", "llm_operator_note",
    "llm_requires_human_confirmation", "llm_advice_status",
]


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


def _coerce_conf(v: Any) -> float:
    if isinstance(v, (int, float)):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0
    m = {"high": 0.9, "medium": 0.6, "low": 0.3, "": 0.0, "none": 0.0}
    key = str(v or "").strip().lower()
    if key in m:
        return m[key]
    try:
        return max(0.0, min(1.0, float(key)))
    except (ValueError, TypeError):
        return 0.0


def _alt_columns(alt_str: str) -> List[str]:
    """Parse '<file>::<sheet>::<col> (conf); ...' into the column names."""
    cols = []
    for part in str(alt_str or "").split(";"):
        part = part.strip()
        if not part:
            continue
        body = part.split(" (")[0]
        toks = body.split("::")
        if toks:
            cols.append(toks[-1].strip())
    return [c for c in cols if c]


def build_decision_packets(
    decision_rows: List[Dict[str, Any]],
    coverage_rows: List[Dict[str, Any]],
    residual_rows: Optional[List[Dict[str, Any]]] = None,
    file_inventory: Optional[List[Dict[str, Any]]] = None,
    evidence_rows: Optional[List[Dict[str, Any]]] = None,
    overlap: Optional[List[Dict[str, Any]]] = None,
    domain_coverage: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, set]]:
    """Build one compact evidence packet per 28c decision (target-first only).

    Returns ``(packets, allowed_sources_by_decision_id)``. The allowed-sources map
    is used to reject any LLM-recommended source not present in the candidates.
    """
    cov_by_field = {r.get("target_field", ""): r for r in coverage_rows}
    classif = {}
    for f in (file_inventory or []):
        name = f.get("file_name", "") if isinstance(f, dict) else getattr(f, "file_name", "")
        cls = f.get("classification", "") if isinstance(f, dict) else getattr(f, "classification", "")
        if name:
            classif[name] = cls
    residual_by_col: Dict[str, str] = {}
    residual_for_target: Dict[str, List[str]] = {}
    for r in (residual_rows or []):
        residual_by_col[r.get("source_column", "")] = r.get("residual_class", "")
        dup = r.get("duplicate_of_target_field", "") or ""
        for tf in [t.strip() for t in dup.split(";") if t.strip()]:
            residual_for_target.setdefault(tf, []).append(r.get("source_column", ""))
    ev_by_col: Dict[str, Dict[str, Any]] = {}
    for e in (evidence_rows or []):
        ev_by_col.setdefault(e.get("source_column", ""), e)

    packets: List[Dict[str, Any]] = []
    allowed: Dict[str, set] = {}
    for d in decision_rows:
        did = d.get("decision_id", "")
        tfield = d.get("target_field", "")
        cov = cov_by_field.get(tfield, {})
        alt_cols = _alt_columns(cov.get("alternative_source_candidates", ""))
        dup_cols = residual_for_target.get(tfield, [])
        sel_col = cov.get("selected_source_column", "") or d.get("source_column", "")
        candidate_cols = [c for c in ([sel_col] + alt_cols + dup_cols) if c]
        # de-dup, preserve order
        seen, cand = set(), []
        for c in candidate_cols:
            if c not in seen:
                seen.add(c)
                cand.append(c)
        allowed[did] = set(cand)

        cand_detail = []
        for c in cand:
            ev = ev_by_col.get(c, {})
            cand_detail.append({
                "source_column": c,
                "residual_class": residual_by_col.get(c, ""),
                "data_type_guess": ev.get("data_type_guess", ""),
                "value_profile": ev.get("candidate_value_profile_matches", ""),
                "sample_values": ev.get("sample_values_distinct_redacted", ""),
            })
        sel_file = cov.get("selected_source_file", "") or d.get("source_file", "")
        packets.append({
            "decision_id": did,
            "decision_type": d.get("decision_type", ""),
            "target_field": tfield,
            "target_contract_id": d.get("target_contract_id", ""),
            "mode": d.get("mode", ""),
            "blocking": bool(d.get("blocking", False)),
            "operator_question": d.get("operator_question", ""),
            "deterministic_recommendation": d.get("recommendation", ""),
            "selected_source_file": sel_file,
            "selected_source_column": sel_col,
            "selected_source_classification": classif.get(sel_file, ""),
            "alternative_source_candidates": cov.get("alternative_source_candidates", ""),
            "evidence_summary": d.get("evidence_summary", ""),
            "coverage_status": cov.get("coverage_status", ""),
            "target_domain": cov.get("target_domain", ""),
            "required_status": cov.get("required_status", ""),
            "value_compatibility_status": cov.get("value_compatibility_status", ""),
            "candidate_source_columns": cand,
            "candidate_source_evidence": cand_detail,
        })
    return packets, allowed


def _base_row(d: Dict[str, Any], status: str, action: str = "", *,
              conf: float = 0.0, rationale: str = "", requires_human: bool = True,
              source_file: str = "", source_column: str = "", configured_value: str = "",
              default_conf: Any = "", not_applicable: Any = "",
              alt_assessment: str = "", operator_note: str = "") -> Dict[str, Any]:
    return {
        "decision_id": d.get("decision_id", ""),
        "target_field": d.get("target_field", ""),
        "decision_type": d.get("decision_type", ""),
        "llm_recommended_action": action,
        "llm_recommended_source_file": source_file,
        "llm_recommended_source_column": source_column,
        "llm_recommended_configured_value": configured_value,
        "llm_recommended_default_confirmation": default_conf,
        "llm_recommended_not_applicable": not_applicable,
        "llm_confidence": conf,
        "llm_rationale": rationale,
        "llm_alternative_assessment": alt_assessment,
        "llm_operator_note": operator_note,
        "llm_requires_human_confirmation": requires_human,
        "llm_advice_status": status,
    }


def run_target_advisor(
    decision_rows: List[Dict[str, Any]],
    coverage_rows: List[Dict[str, Any]],
    residual_rows: Optional[List[Dict[str, Any]]] = None,
    file_inventory: Optional[List[Dict[str, Any]]] = None,
    evidence_rows: Optional[List[Dict[str, Any]]] = None,
    overlap: Optional[List[Dict[str, Any]]] = None,
    domain_coverage: Optional[List[Dict[str, Any]]] = None,
    llm_callable: Optional[Callable[[str], str]] = None,
    max_items: int = 20,
    max_calls: int = 1,
    max_cost_gbp: float = 1.0,
    cost_per_call_gbp: float = 0.01,
    model: str = "",
) -> Dict[str, Any]:
    """Run the advisory pass over the 28c decisions. Never raises."""
    usage = {
        "llm_target_advisor_enabled": True,
        "decision_rows_available": len(decision_rows),
        "decision_rows_sent": 0,
        "decision_rows_reviewed": 0,
        "decision_rows_advised": 0,
        "decision_rows_parse_failed": 0,
        "calls_completed": 0,
        "estimated_cost_gbp": 0.0,
        "model": model,
        "budget_exhausted": False,
    }
    raw_response = {"llm_batch_id": "", "raw_response": ""}

    if not decision_rows:
        return {"recommendations": [], "usage": usage, "raw_response": raw_response,
                "advice_status": SKIPPED_NO_DECISIONS}

    packets, allowed = build_decision_packets(
        decision_rows, coverage_rows, residual_rows, file_inventory, evidence_rows,
        overlap, domain_coverage)

    # Budget guardrails (reuse the existing call/cost limits).
    if max_calls <= 0 or cost_per_call_gbp > max_cost_gbp:
        usage["budget_exhausted"] = True
        recs = [_base_row(d, SKIPPED_BUDGET,
                          action=_DEFAULT_ACTION.get(d.get("decision_type", ""),
                                                     "requires_operator_review"),
                          rationale="skipped — LLM budget/cost guardrail")
                for d in decision_rows]
        return {"recommendations": recs, "usage": usage, "raw_response": raw_response,
                "advice_status": SKIPPED_BUDGET}

    # No client available: emit no_advice rows echoing the deterministic default.
    if llm_callable is None:
        recs = [_base_row(d, NO_ADVICE,
                          action=_DEFAULT_ACTION.get(d.get("decision_type", ""),
                                                     "requires_operator_review"),
                          rationale="LLM target advisor enabled but no LLM client/API "
                                    "key available; showing deterministic default")
                for d in decision_rows]
        return {"recommendations": recs, "usage": usage, "raw_response": raw_response,
                "advice_status": NO_ADVICE}

    sent_packets = packets[:max_items]
    sent_ids = {p["decision_id"] for p in sent_packets}
    usage["decision_rows_sent"] = len(sent_packets)
    batch_id = "tadv_" + uuid.uuid4().hex[:10]
    prompt = _PROMPT + "\nDECISIONS = " + json.dumps(sent_packets, default=str)
    try:
        raw = llm_callable(prompt)
        raw_text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    except Exception as exc:  # never break the run on an LLM error
        raw_text = ""
        raw_response = {"llm_batch_id": batch_id, "raw_response": f"ERROR: {exc}"}
        recs = [_base_row(d, PARSE_FAILED if d["decision_id"] in sent_ids else NO_ADVICE,
                          action=_DEFAULT_ACTION.get(d.get("decision_type", ""),
                                                     "requires_operator_review"),
                          rationale="LLM call failed")
                for d in decision_rows]
        usage["calls_completed"] = 1
        usage["decision_rows_parse_failed"] = len(sent_packets)
        usage["estimated_cost_gbp"] = round(cost_per_call_gbp, 6)
        return {"recommendations": recs, "usage": usage, "raw_response": raw_response,
                "advice_status": PARSE_FAILED}

    usage["calls_completed"] = 1
    usage["estimated_cost_gbp"] = round(cost_per_call_gbp, 6)
    raw_response = {"llm_batch_id": batch_id, "raw_response": raw_text}

    # Parse the response.
    from .llm_json import extract_json
    obj, parse_status, _err = extract_json(raw_text)
    results: List[Dict[str, Any]] = []
    if isinstance(obj, dict) and isinstance(obj.get("recommendations"), list):
        results = [r for r in obj["recommendations"] if isinstance(r, dict)]
    elif isinstance(obj, list):
        results = [r for r in obj if isinstance(r, dict)]
    by_id = {str(r.get("decision_id", "")): r for r in results}

    parse_failed_all = not results
    recs: List[Dict[str, Any]] = []
    for d in decision_rows:
        did = d.get("decision_id", "")
        if did not in sent_ids:
            recs.append(_base_row(d, SKIPPED_BUDGET,
                                  action=_DEFAULT_ACTION.get(d.get("decision_type", ""),
                                                             "requires_operator_review"),
                                  rationale="not sent (per-call item cap)"))
            continue
        usage["decision_rows_reviewed"] += 1
        res = by_id.get(did)
        if parse_failed_all or res is None:
            usage["decision_rows_parse_failed"] += 1
            recs.append(_base_row(d, PARSE_FAILED,
                                  action="requires_operator_review",
                                  rationale="no parseable recommendation for this decision"))
            continue
        action = _norm(res.get("recommended_action")).replace(" ", "_")
        rec_file = str(res.get("recommended_source_file", "") or "")
        rec_col = str(res.get("recommended_source_column", "") or "")
        conf = _coerce_conf(res.get("confidence"))
        rationale = str(res.get("rationale", "") or "")
        alt = str(res.get("alternative_assessment", "") or "")
        note = str(res.get("operator_note", "") or "")
        req_human = bool(res.get("requires_human_confirmation", True))
        cfg_val = str(res.get("recommended_configured_value", "") or "")
        default_conf = bool(res.get("default_confirmation", False))
        not_app = bool(res.get("not_applicable", False))

        status = ADVISED
        # Unknown action -> escalate.
        if action not in ALLOWED_ACTIONS:
            action = "requires_operator_review"
            status = ADVISED
            note = (note + " | LLM returned an unrecognised action").strip(" |")
        # Source containment: a recommended source must be a supplied candidate.
        if action in ("choose_alternative", "provide_source_mapping"):
            if rec_col and rec_col not in allowed.get(did, set()):
                status = INVALID_RESPONSE
                action = "requires_operator_review"
                note = (note + " | LLM recommended a source not in the supplied "
                        "candidates; not accepted").strip(" |")
                req_human = True
        if status == ADVISED:
            usage["decision_rows_advised"] += 1
        recs.append(_base_row(
            d, status, action=action, conf=conf, rationale=rationale,
            requires_human=req_human, source_file=rec_file, source_column=rec_col,
            configured_value=cfg_val, default_conf=default_conf, not_applicable=not_app,
            alt_assessment=alt, operator_note=note))

    return {"recommendations": recs, "usage": usage, "raw_response": raw_response,
            "advice_status": ADVISED}


# ---------------------------------------------------------------------------
# Artefact writers
# ---------------------------------------------------------------------------

def advisor_summary(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    action_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    for r in recs:
        action_counts[r["llm_recommended_action"]] = action_counts.get(r["llm_recommended_action"], 0) + 1
        status_counts[r["llm_advice_status"]] = status_counts.get(r["llm_advice_status"], 0) + 1
    return {
        "recommendations_total": len(recs),
        "advised": status_counts.get(ADVISED, 0),
        "requires_operator_review": sum(1 for r in recs
                                        if r["llm_recommended_action"] == "requires_operator_review"),
        "action_counts": action_counts,
        "advice_status_counts": status_counts,
    }


def write_advisor_artifacts(result: Dict[str, Any], out_dir: str | Path) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = result["recommendations"]
    usage = result["usage"]
    summary = advisor_summary(recs)

    csv_path = out / "36_target_first_llm_recommendations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_RECOMMENDATION_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow({c: r.get(c, "") for c in _RECOMMENDATION_COLUMNS})
    (out / "36_target_first_llm_recommendations.json").write_text(
        json.dumps({"summary": summary, "rows": recs}, indent=2, default=str),
        encoding="utf-8")

    md = ["# Target-first LLM advisor recommendations", "",
          f"- **Recommendations:** {summary['recommendations_total']}",
          f"- **Advised:** {summary['advised']}",
          f"- **Require operator review:** {summary['requires_operator_review']}", "",
          "LLM recommendations are advisory — deterministic 28a/28c are unchanged. "
          "To apply them, run `accept-target-advice` (writes "
          "34_target_first_decisions_approved.yaml) or approve manually in "
          "34_target_first_decisions.yaml, then rerun with "
          "--target-first-decisions.", "",
          "## Recommended action counts", ""]
    for a, c in sorted(summary["action_counts"].items(), key=lambda kv: -kv[1]):
        md.append(f"- `{a}`: {c}")
    (out / "36_target_first_llm_recommendations_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")

    (out / "36_target_first_llm_raw_response.json").write_text(
        json.dumps(result["raw_response"], indent=2, default=str), encoding="utf-8")
    (out / "36_target_first_llm_usage_summary.json").write_text(
        json.dumps(usage, indent=2, default=str), encoding="utf-8")
    return {
        "csv": str(csv_path),
        "json": str(out / "36_target_first_llm_recommendations.json"),
        "summary_md": str(out / "36_target_first_llm_recommendations_summary.md"),
        "raw": str(out / "36_target_first_llm_raw_response.json"),
        "usage": str(out / "36_target_first_llm_usage_summary.json"),
    }
