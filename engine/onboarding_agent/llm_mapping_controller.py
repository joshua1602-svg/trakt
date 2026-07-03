"""
llm_mapping_controller.py
=========================

PART 5 — a controlled LLM mapping reviewer.

The LLM *proposes* mappings from compact evidence packs + deterministic
shortlists. It is never the final authority and is OFF by default. It is invoked
through an injectable ``llm_callable`` (so it is fully testable without any API
key and never reaches the network in tests/CI).

Hard rules enforced deterministically here (not merely requested in the prompt):
  * The LLM cannot approve its own mapping (``requires_user_approval`` is forced
    on for anything not auto-approvable downstream; the controller never sets a
    mapping active).
  * The LLM cannot bypass field scope (out-of-scope targets are flagged).
  * The LLM cannot map to a non-existent field unless it is explicitly labelled
    ``proposed_new_field``.
  * The LLM never writes registry/alias files or client memory (this module has
    no write side-effects to those stores).
  * The LLM never sees the full raw file — only redacted evidence packs.

Artefacts:
    31_llm_mapping_review.json / .csv / _summary.md
    31_llm_usage_summary.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

CONFIDENCE_LEVELS = ("high", "medium", "low", "no_match")

LLM_SYSTEM_INSTRUCTIONS = """\
You map lender source columns to Trakt canonical/pipeline fields.
You are PROPOSING mappings, not finalising them. Rules:
- Use ONLY the provided evidence packs and candidate shortlists. You are NOT
  given the raw file.
- Do NOT invent a target field. If no suitable existing field exists, return
  proposed_target_source="registry_target_missing" or
  "pipeline_contract_target_missing" (and you MAY suggest a proposed_new_field).
- If ambiguous, set requires_user_approval=true and confidence in
  {high,medium,low,no_match}.
- Do NOT map pipeline/workflow columns into regulatory/funded-loan fields unless
  the evidence is very strong AND field scope allows it.
- Return STRUCTURED JSON ONLY: a list of objects with keys: source_column,
  proposed_business_meaning, proposed_target_field, proposed_target_source,
  confidence, reasoning_summary, evidence_used, alternative_targets,
  ambiguity_flags, validation_risks, requires_user_approval,
  registry_action_recommended, alias_action_recommended,
  pipeline_contract_action_recommended, question_for_user.
"""

# Evidence keys exposed to the LLM (compact + redacted; NO raw file).
_PACK_KEYS = [
    "source_column", "normalized_column", "data_type_guess", "null_rate",
    "distinct_count", "uniqueness_ratio", "sample_values_distinct_redacted",
    "rate_like_score", "percentage_like_score", "amount_like_score",
    "identifier_like_score", "stage_like_score", "postcode_like_score",
    "gender_like_score", "chronology_relationships", "amount_relationships",
    "candidate_value_profile_matches",
]


def build_evidence_pack(
    ev: Dict[str, Any], shortlist: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compact, redacted evidence pack for one column (the only LLM input)."""
    pack = {k: ev.get(k) for k in _PACK_KEYS}
    pack["candidate_shortlist"] = [
        {"target": s["candidate_target_field"], "source": s["candidate_source"],
         "confidence": s["candidate_confidence"], "field_scope_status": s["field_scope_status"],
         "type_compatible": s["type_compatible"]}
        for s in shortlist
    ]
    return pack


def build_prompt(packs: List[Dict[str, Any]]) -> str:
    return (LLM_SYSTEM_INSTRUCTIONS + "\n\nEVIDENCE_PACKS = "
            + json.dumps(packs, default=str) + "\n\nReturn the JSON list now.")


def _coerce_conf(value: Any) -> str:
    v = str(value or "").strip().lower()
    return v if v in CONFIDENCE_LEVELS else "low"


def _sanitize_proposal(
    p: Dict[str, Any],
    shortlist_targets: set,
    registry_fields: set,
    field_scope: Any,
) -> Dict[str, Any]:
    """Enforce the hard rules on one raw LLM proposal."""
    col = str(p.get("source_column", ""))
    target = str(p.get("proposed_target_field", "") or "").strip()
    source = str(p.get("proposed_target_source", "") or "").strip() or "llm_suggested"
    conf = _coerce_conf(p.get("confidence"))

    target_exists = target in registry_fields
    proposed_new = source == "proposed_new_field" or bool(p.get("proposed_new_field"))

    # Rule: cannot map to a non-existent field unless labelled proposed_new_field.
    if target and not target_exists and not proposed_new and target not in shortlist_targets:
        # Treat as a missing-target signal rather than a silent invention.
        source = "registry_target_missing"
        target = ""

    # Rule: cannot bypass field scope — flag out-of-scope targets.
    out_of_scope = bool(
        target and field_scope is not None and getattr(field_scope, "is_excluded", None)
        and target in registry_fields and field_scope.is_excluded(target))

    # Rule: the LLM cannot approve its own mapping — always require approval unless
    # it is a clean, high-confidence shortlist target (downstream backstop decides
    # auto-approval; the LLM proposal itself never becomes active here).
    requires_approval = True
    if (conf == "high" and target and not out_of_scope
            and (target in shortlist_targets or target_exists)
            and not p.get("ambiguity_flags")):
        # Still requires the deterministic backstop to auto-approve; we mark the
        # LLM's own recommendation but never set active.
        requires_approval = bool(p.get("requires_user_approval", True))

    return {
        "source_column": col,
        "proposed_business_meaning": str(p.get("proposed_business_meaning", "")),
        "proposed_target_field": target,
        "proposed_target_source": source,
        "confidence": conf if target else "no_match",
        "reasoning_summary": str(p.get("reasoning_summary", "")),
        "evidence_used": p.get("evidence_used", []),
        "alternative_targets": p.get("alternative_targets", []),
        "ambiguity_flags": p.get("ambiguity_flags", []),
        "validation_risks": p.get("validation_risks", []),
        "requires_user_approval": bool(requires_approval) or out_of_scope,
        "field_scope_status": "out_of_scope" if out_of_scope else "in_scope",
        "registry_action_recommended": str(p.get("registry_action_recommended", "")),
        "alias_action_recommended": str(p.get("alias_action_recommended", "")),
        "pipeline_contract_action_recommended": str(p.get("pipeline_contract_action_recommended", "")),
        "question_for_user": str(p.get("question_for_user", "")),
        "llm_used": True,
    }


class LLMMappingController:
    """Runs the controlled LLM reviewer over evidence packs (off by default)."""

    def __init__(
        self,
        llm_callable: Optional[Callable[[str], str]] = None,
        registry_fields: Optional[Dict[str, Any]] = None,
        field_scope: Any = None,
        max_items: int = 60,
        cost_per_call_gbp: float = 0.01,
        max_cost_gbp: float = 1.0,
    ):
        self.llm_callable = llm_callable
        self.registry_fields = set((registry_fields or {}).keys())
        self.field_scope = field_scope
        self.max_items = max_items
        self.cost_per_call_gbp = cost_per_call_gbp
        self.max_cost_gbp = max_cost_gbp

    def review(
        self,
        evidence_rows: List[Dict[str, Any]],
        shortlist_by_col: Dict[str, List[Dict[str, Any]]],
        only_unresolved: Optional[set] = None,
    ) -> Dict[str, Any]:
        usage = {
            "llm_enabled": bool(self.llm_callable),
            "calls_completed": 0,
            "items_sent": 0,
            "estimated_cost_gbp": 0.0,
            "prompt_chars": 0,
            "stopped_for_budget": False,
        }

        if not self.llm_callable:
            return {"proposals": [], "usage": usage}

        rows = evidence_rows
        if only_unresolved is not None:
            rows = [e for e in rows if e["source_column"] in only_unresolved]

        rows = rows[: self.max_items]

        packs = [
            build_evidence_pack(e, shortlist_by_col.get(e["source_column"], []))
            for e in rows
        ]

        if not packs:
            return {"proposals": [], "usage": usage}

        if usage["estimated_cost_gbp"] + self.cost_per_call_gbp > self.max_cost_gbp:
            usage["stopped_for_budget"] = True
            return {"proposals": [], "usage": usage}

        import uuid

        batch_id = "llm_" + uuid.uuid4().hex[:10]
        usage["llm_batch_id"] = batch_id

        prompt = build_prompt(packs)
        usage["prompt_chars"] = len(prompt)

        raw = self.llm_callable(prompt)

        usage["calls_completed"] = 1
        usage["items_sent"] = len(packs)
        usage["estimated_cost_gbp"] = round(self.cost_per_call_gbp, 6)

        try:
            if isinstance(raw, str):
                from engine.onboarding_agent.llm_json import extract_json

                parsed, _, _ = extract_json(raw)
            else:
                parsed = raw

            if isinstance(parsed, dict):
                parsed = (
                    parsed.get("proposals")
                    or parsed.get("mappings")
                    or parsed.get("llm_mapping_suggestions")
                    or []
                )

            if not isinstance(parsed, list):
                parsed = []

        except Exception:
            parsed = []

        shortlist_targets = {
            s["candidate_target_field"]
            for sl in shortlist_by_col.values()
            for s in sl
            if isinstance(s, dict) and "candidate_target_field" in s
        }

        proposals = []

        for p in parsed:
            if not isinstance(p, dict):
                continue

            sp = _sanitize_proposal(
                p,
                shortlist_targets,
                self.registry_fields,
                self.field_scope,
            )
            sp["llm_batch_id"] = batch_id
            proposals.append(sp)

        return {"proposals": proposals, "usage": usage}


_REVIEW_COLUMNS = [
    "source_column", "proposed_business_meaning", "proposed_target_field",
    "proposed_target_source", "confidence", "field_scope_status",
    "requires_user_approval", "reasoning_summary", "registry_action_recommended",
    "alias_action_recommended", "pipeline_contract_action_recommended",
    "question_for_user",
]


def write_llm_review_artifacts(
    result: Dict[str, Any], output_dir: str | Path
) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proposals = result.get("proposals", [])
    json_path = out_dir / "31_llm_mapping_review.json"
    json_path.write_text(json.dumps(proposals, indent=2, default=str), encoding="utf-8")
    csv_path = out_dir / "31_llm_mapping_review.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_REVIEW_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for p in proposals:
            w.writerow({c: p.get(c, "") for c in _REVIEW_COLUMNS})
    md_path = out_dir / "31_llm_mapping_review_summary.md"
    md_lines = ["# LLM mapping review (proposals only — not final)", ""]
    md_lines.append(f"{len(proposals)} proposals. The LLM never finalises a mapping; "
                    "every proposal passes through the deterministic backstop validator.")
    md_lines.append("")
    for p in proposals:
        md_lines.append(f"- **{p['source_column']}** → `{p['proposed_target_field'] or '—'}` "
                        f"({p['proposed_target_source']}, {p['confidence']}) — "
                        f"{p['reasoning_summary']}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    usage_path = out_dir / "31_llm_usage_summary.json"
    usage_path.write_text(json.dumps(result.get("usage", {}), indent=2, default=str),
                          encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path),
            "summary_md": str(md_path), "usage": str(usage_path)}
