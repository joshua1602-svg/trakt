"""
workflow.py
===========

Thin managed-service OPERATOR WORKFLOW wrapper for the Onboarding Agent.

This is a wrapper only — it orchestrates the existing, working pieces and never
reimplements onboarding logic:

    deterministic target-first onboarding  (run_onboarding)
      -> 28a coverage / 28b residual / 28c decision queue / 34 template
    optional target-first LLM advisor       (advisory only -> 36_*)
    optional approved-decision application   (--target-first-decisions -> 35_*)

It then reads the resulting artefacts and emits an operator-readable workflow
summary plus a conservative, non-mutating legacy-file audit:

    40_operator_workflow_summary.json / .md
    41_onboarding_legacy_file_audit.json / .md

Run it with::

    python -m engine.onboarding_agent.workflow --input-dir ... --client-name ...

The existing lower-level CLI (``engine.onboarding_agent.cli``) is unchanged and
remains available for development / debugging.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent.mode_policy import VALID_MODES  # noqa: E402

# Workflow status vocabulary (target-first state only — never legacy gaps).
READY = "READY"
NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
BLOCKED = "BLOCKED"
FAILED = "FAILED"

FIRST_PASS = "first_pass_review_generation"
SECOND_PASS = "second_pass_decision_application"


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_yaml(path: Path) -> Optional[dict]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Status derivation (pure — unit-testable)
# ---------------------------------------------------------------------------

def derive_status(decisions_after: int, blocking_after: int,
                  required_artifacts_missing: bool) -> str:
    """Workflow status from the resulting target-first 28c state."""
    if required_artifacts_missing:
        return FAILED
    if blocking_after > 0:
        return BLOCKED
    if decisions_after > 0:
        return NEEDS_CONFIRMATION
    return READY


def _next_action(stage: str, status: str, advisor_enabled: bool) -> str:
    if status == FAILED:
        return ("Workflow failed to produce the required target-first artefacts — "
                "re-run with --enable-mapping-review and check the logs.")
    if status == BLOCKED:
        return ("Resolve the remaining BLOCKING Gate 4 decisions (provide a source, "
                "configure a value, or mark not applicable) in a copy of "
                "34_target_first_decisions.yaml and rerun with --target-first-decisions.")
    if status == READY:
        return ("All Gate 4 decisions are resolved — the pack is ready for MI handoff / "
                "promotion.")
    # NEEDS_CONFIRMATION
    advisor_hint = (" Use the LLM advisor column as optional guidance." if advisor_enabled
                    else "")
    if stage == FIRST_PASS:
        return ("Review Gate 4 in the HTML review pack." + advisor_hint +
                " Then approve decisions in a copy of 34_target_first_decisions.yaml and "
                "rerun with --target-first-decisions.")
    return ("Confirm the remaining non-blocking Gate 4 decisions (approve them in a copy "
            "of 34_target_first_decisions.yaml) and rerun with --target-first-decisions.")


# ---------------------------------------------------------------------------
# Summary assembly (reads existing artefacts — never mutates them)
# ---------------------------------------------------------------------------

_CORE_ARTIFACTS = [
    "08_onboarding_review_pack.html",
    "28a_target_coverage_matrix.csv",
    "28c_human_decision_queue.csv",
    "34_target_first_decisions.yaml",
]
_DECISION_ARTIFACTS = [
    "35_target_first_decision_application_log.json",
    "35_target_first_decision_application_log.csv",
]
_ADVISOR_ARTIFACTS = [
    "36_target_first_llm_recommendations.csv",
    "36_target_first_llm_usage_summary.json",
]


def build_workflow_summary(
    project_dir: Path,
    output_root: Path,
    *,
    client_id: str,
    client_name: str,
    run_id: str,
    mode: str,
    decisions_supplied_file: str,
    advisor_enabled: bool,
    input_source_files_count: int,
    run_error: str = "",
) -> Dict[str, Any]:
    """Read the run's artefacts and build the 40 workflow summary dict."""
    stage = SECOND_PASS if decisions_supplied_file else FIRST_PASS

    cov = _read_json(project_dir / "28a_target_coverage_matrix.json") or {}
    cov_sum = cov.get("summary", {}) or {}
    dec = _read_json(project_dir / "28c_human_decision_queue.json") or {}
    dec_sum = dec.get("summary", {}) or {}
    app = _read_json(project_dir / "35_target_first_decision_application_log.json")
    app_sum = (app or {}).get("summary", {}) or {}
    advu = _read_json(project_dir / "36_target_first_llm_usage_summary.json")
    advr = _read_json(project_dir / "36_target_first_llm_recommendations.json")
    adv_sum = (advr or {}).get("summary", {}) or {}

    decisions_after = int(dec_sum.get("human_decision_rows_total", len(dec.get("rows", []) or [])))
    blocking_after = int(dec_sum.get("blocking_decisions", 0))
    non_blocking_after = max(0, decisions_after - blocking_after)
    applied = int(app_sum.get("applied", 0))
    # Deterministic reconstruction: applied decisions were removed from 28c.
    decisions_before = decisions_after + applied if stage == SECOND_PASS else decisions_after

    # Required-artefact validation (warnings; FAILED only when CORE is missing).
    warnings: List[str] = []
    core_missing = [n for n in _CORE_ARTIFACTS if not (project_dir / n).exists()]
    for n in core_missing:
        warnings.append(f"required artefact missing: {n}")
    if decisions_supplied_file:
        for n in _DECISION_ARTIFACTS:
            if not (project_dir / n).exists():
                warnings.append(f"expected decision-application artefact missing: {n}")
    if advisor_enabled:
        for n in _ADVISOR_ARTIFACTS:
            if not (project_dir / n).exists():
                warnings.append(f"expected LLM advisor artefact missing: {n}")

    failed = bool(run_error) or bool(core_missing)
    status = FAILED if failed else derive_status(decisions_after, blocking_after, False)

    def _p(name: str) -> str:
        p = project_dir / name
        return str(p) if p.exists() else ""

    summary = {
        "workflow_run_id": run_id,
        "client_id": client_id,
        "client_name": client_name,
        "mode": mode,
        "project_dir": str(project_dir),
        "output_root": str(output_root),
        "workflow_stage": stage,
        "status": status,
        "generated_at": _now(),
        "input_source_files_count": input_source_files_count,
        "target_fields_count": int(cov_sum.get("target_fields_total", len(cov.get("rows", []) or []))),
        "target_coverage_file": _p("28a_target_coverage_matrix.csv"),
        "human_decision_queue_file": _p("28c_human_decision_queue.csv"),
        "human_decision_queue_count_before": decisions_before,
        "human_decision_queue_count_after": decisions_after,
        "blocking_decisions_count": blocking_after,
        "non_blocking_confirmations_count": non_blocking_after,
        "target_first_decisions_template_file": _p("34_target_first_decisions.yaml"),
        "target_first_decisions_supplied_file": decisions_supplied_file or "",
        "approved_decisions_supplied_count": int(app_sum.get("decisions_supplied", 0)),
        "applied_decisions_count": applied,
        "deferred_decisions_count": int(app_sum.get("deferred", 0)),
        "invalid_decisions_count": int(app_sum.get("invalid", 0)),
        "requires_operator_review_count": int(app_sum.get("requires_operator_review", 0)),
        "application_log_json": _p("35_target_first_decision_application_log.json"),
        "application_log_csv": _p("35_target_first_decision_application_log.csv"),
        "llm_target_advisor_enabled": bool(advisor_enabled),
        "llm_target_advisor_file": _p("36_target_first_llm_recommendations.csv"),
        "llm_target_advisor_rows": int(adv_sum.get("recommendations_total",
                                                   (advu or {}).get("decision_rows_available", 0))),
        "llm_target_advisor_advised_count": int((advu or {}).get("decision_rows_advised",
                                                                 adv_sum.get("advised", 0))),
        "llm_target_advisor_parse_failed_count": int((advu or {}).get("decision_rows_parse_failed", 0)),
        "llm_target_advisor_estimated_cost_gbp": float((advu or {}).get("estimated_cost_gbp", 0.0)),
        "review_pack_html": _p("08_onboarding_review_pack.html"),
        "warnings": warnings,
        "error": run_error,
    }
    summary["next_operator_action"] = _next_action(stage, status, advisor_enabled)
    return summary


def _write_summary_md(path: Path, s: Dict[str, Any]) -> None:
    supplied = (Path(s["target_first_decisions_supplied_file"]).name
                if s["target_first_decisions_supplied_file"] else "none")
    md = [
        "# Operator workflow summary", "",
        f"Client: {s['client_name']}  ",
        f"Mode: {s['mode']}  ",
        f"Run: {s['workflow_run_id']}  ",
        f"Status: {s['status']}", "",
        "## Target-first status", "",
        f"- Target fields: {s['target_fields_count']}",
        f"- Gate 4 decisions remaining: {s['human_decision_queue_count_after']}",
        f"- Blocking decisions: {s['blocking_decisions_count']}",
        f"- Non-blocking confirmations: {s['non_blocking_confirmations_count']}", "",
        "## Decisions", "",
        f"- Decision template: {Path(s['target_first_decisions_template_file']).name or '—'}",
        f"- Approved decisions supplied: {supplied}",
        f"- Applied decisions: {s['applied_decisions_count']}",
        f"- Deferred: {s['deferred_decisions_count']} · "
        f"invalid: {s['invalid_decisions_count']} · "
        f"requires review: {s['requires_operator_review_count']}", "",
        "## LLM target advisor", "",
        f"- Enabled: {'yes' if s['llm_target_advisor_enabled'] else 'no'}",
        f"- Decisions reviewed: {s['llm_target_advisor_rows']}",
        f"- Decisions advised: {s['llm_target_advisor_advised_count']}",
        f"- Estimated cost: £{s['llm_target_advisor_estimated_cost_gbp']}", "",
        "## Next operator action", "",
        s["next_operator_action"], "",
    ]
    if s.get("warnings"):
        md += ["## Warnings", ""] + [f"- {w}" for w in s["warnings"]] + [""]
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Legacy file audit (41) — conservative, NON-MUTATING
# ---------------------------------------------------------------------------

# (basename, kind, classification, reason). Existence is checked at write time.
_AUDIT_CATALOG = [
    # --- target-first artefacts that MUST be kept ---
    ("28a_target_coverage_matrix.csv", "artefact", "keep_core", "Target-first coverage matrix (primary gate)."),
    ("28b_source_residual_register.csv", "artefact", "keep_core", "Target-first residual register (suppressed source columns)."),
    ("28c_human_decision_queue.csv", "artefact", "keep_core", "Target-first compact Gate 4 decision queue (primary operator workflow)."),
    ("34_target_first_decisions.yaml", "artefact", "keep_core", "Target-first operator decision template / persistence."),
    ("35_target_first_decision_application_log.json", "artefact", "keep_core", "Target-first decision application log."),
    ("36_target_first_llm_recommendations.csv", "artefact", "keep_core", "Target-first LLM advisory recommendations (advisory only)."),
    ("36_target_first_llm_usage_summary.json", "artefact", "keep_core", "Target-first LLM advisor usage summary."),
    ("08_onboarding_review_pack.html", "artefact", "keep_core", "Operator-first review pack (primary deliverable)."),
    # --- workflow artefacts that MUST be kept ---
    ("40_operator_workflow_summary.json", "artefact", "keep_core", "Operator workflow summary (this command)."),
    ("41_onboarding_legacy_file_audit.json", "artefact", "keep_core", "Legacy-file audit (this command)."),
    # --- source-column legacy decision artefacts RETAINED FOR AUDIT ---
    ("33_mapping_review_queue.csv", "artefact", "keep_legacy_audit", "Source-column review queue; retained as audit detail, no longer the primary gate."),
    ("34_mapping_review_decisions.yaml", "artefact", "keep_legacy_audit", "Source-column decision template; superseded by 34_target_first_decisions.yaml; kept for audit."),
    ("35_mapping_review_action_log.json", "artefact", "keep_legacy_audit", "Source-column action log; superseded by 35_target_first_*; kept for audit."),
    # --- old LLM source-column resolver artefacts ---
    ("31_llm_mapping_resolver.csv", "artefact", "keep_legacy_audit", "Source-column LLM resolver output; separate layer from the target-first advisor; kept for audit."),
    ("31_llm_mapping_resolver_summary.md", "artefact", "keep_legacy_audit", "Source-column LLM resolver summary; kept for audit."),
    ("31_llm_resolver_usage_summary.json", "artefact", "keep_legacy_audit", "Source-column LLM resolver usage; kept for audit."),
    ("31_llm_field_raw_response.json", "artefact", "keep_legacy_audit", "Source-column LLM raw response; kept for audit."),
    # --- old gap-question readiness artefacts ---
    ("07_gap_questions.csv", "artefact", "keep_legacy_audit", "Legacy gap questions; superseded by target-first Gate 4; do NOT drive workflow status."),
    ("07_gap_questions.yaml", "artefact", "candidate_for_deprecation", "Legacy gap-question readiness; superseded by 28c/40 status; deprecate after migration."),
    # --- supporting deterministic artefacts (kept) ---
    ("28_required_target_contract.csv", "artefact", "keep_core", "Required target contract that 28a is built against."),
    ("32_mapping_backstop_validation.csv", "artefact", "keep_legacy_audit", "Source-column backstop validation; audit detail."),
    # --- modules ---
    ("engine/onboarding_agent/target_coverage.py", "module", "keep_core", "Deterministic target-first coverage."),
    ("engine/onboarding_agent/target_first_decisions.py", "module", "keep_core", "Target-first decision capture / application."),
    ("engine/onboarding_agent/target_first_llm_advisor.py", "module", "keep_core", "Target-first LLM advisor (advisory only)."),
    ("engine/onboarding_agent/workflow.py", "module", "keep_core", "Operator workflow wrapper (this command)."),
    ("engine/onboarding_agent/mapping_review_queue.py", "module", "keep_legacy_audit", "Source-column 33 queue generation; retained for audit detail."),
    ("engine/onboarding_agent/llm_mapping_resolver.py", "module", "keep_legacy_audit", "Source-column LLM resolver; separate layer; retained."),
]


def build_legacy_audit(project_dir: Path) -> Dict[str, Any]:
    entries = []
    counts: Dict[str, int] = {}
    for name, kind, classification, reason in _AUDIT_CATALOG:
        present = ((project_dir / name).exists() if kind == "artefact"
                   else (_REPO_ROOT / name).exists())
        entries.append({
            "name": name, "kind": kind, "classification": classification,
            "reason": reason, "present_in_run": bool(present),
            "retained": classification != "candidate_for_removal_after_migration",
        })
        counts[classification] = counts.get(classification, 0) + 1
    return {
        "generated_at": _now(),
        "scope": "onboarding-related files only",
        "non_mutating": True,
        "note": ("Audit only — nothing is deleted or modified. Conservative: when "
                 "unsure, classify as unknown_review_required."),
        "classification_counts": counts,
        "entries": entries,
    }


def _write_audit_md(path: Path, audit: Dict[str, Any]) -> None:
    md = ["# Onboarding legacy file audit", "",
          "Non-mutating audit — nothing is deleted in this PR.", "",
          "## Classification counts", ""]
    for cls, c in sorted(audit["classification_counts"].items()):
        md.append(f"- `{cls}`: {c}")
    md += ["", "## Entries", "",
           "| Name | Kind | Classification | Present | Reason |",
           "|---|---|---|---|---|"]
    for e in audit["entries"]:
        md.append(f"| `{e['name']}` | {e['kind']} | {e['classification']} | "
                  f"{'yes' if e['present_in_run'] else 'no'} | {e['reason']} |")
    md.append("")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_operator_workflow(
    *,
    input_dir: str,
    client_name: str,
    client_id: str = "",
    run_id: str = "",
    project_dir: str = "",
    output_root: str = "",
    mode: str = "mi_only",
    registry: str = "config/system/fields_registry.yaml",
    aliases_dir: str = "config/system",
    enable_mapping_review: bool = True,
    enable_llm_target_advisor: bool = False,
    target_first_decisions: str = "",
    llm_max_cost_gbp: float = 1.0,
    llm_max_calls: Optional[int] = None,
    llm_max_items_per_call: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the managed-service operator workflow; returns the 40 summary dict."""
    client_id = client_id or client_name.lower().replace(" ", "_")
    run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    pdir = Path(project_dir) if project_dir else (
        _REPO_ROOT / "onboarding_output" / client_id / run_id)
    oroot = Path(output_root) if output_root else (pdir / "output")
    pdir.mkdir(parents=True, exist_ok=True)

    # Build a shared LLM callable only when the advisor is enabled (None when no
    # ANTHROPIC_API_KEY — the advisor then records a deterministic no_advice).
    advisor_callable = None
    advisor_model = ""
    if enable_llm_target_advisor:
        from engine.onboarding_agent.cli import _build_mapping_llm_callable
        advisor_callable = _build_mapping_llm_callable("low")
        advisor_model = "claude-haiku-4-5-20251001"
        if advisor_callable is None:
            print("[workflow] No ANTHROPIC_API_KEY — LLM target advisor will record a "
                  "deterministic no_advice fallback.")

    run_error = ""
    input_files = 0
    try:
        from engine.onboarding_agent.onboarding_orchestrator import run_onboarding
        project = run_onboarding(
            input_dir=input_dir, client_name=client_name, output_dir=str(pdir),
            registry_path=registry, aliases_dir=aliases_dir, mode=mode,
            client_id=client_id, run_id=run_id, output_uri="",
            enable_mapping_review=enable_mapping_review,
            target_first_decisions_path=(target_first_decisions or ""),
            enable_llm_target_advisor=enable_llm_target_advisor,
            llm_target_advisor_callable=advisor_callable,
            llm_target_advisor_model=advisor_model,
            llm_max_cost_gbp=llm_max_cost_gbp,
            llm_max_calls=llm_max_calls,
            llm_max_items_per_call=llm_max_items_per_call,
        )
        input_files = len(project.file_inventory)
    except Exception as exc:  # produce a FAILED summary instead of crashing
        run_error = f"{type(exc).__name__}: {exc}"

    summary = build_workflow_summary(
        pdir, oroot, client_id=client_id, client_name=client_name, run_id=run_id,
        mode=mode, decisions_supplied_file=(target_first_decisions or ""),
        advisor_enabled=enable_llm_target_advisor,
        input_source_files_count=input_files, run_error=run_error)

    (pdir / "40_operator_workflow_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_summary_md(pdir / "40_operator_workflow_summary.md", summary)

    audit = build_legacy_audit(pdir)
    (pdir / "41_onboarding_legacy_file_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8")
    _write_audit_md(pdir / "41_onboarding_legacy_file_audit.md", audit)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m engine.onboarding_agent.workflow",
        description="Managed-service operator workflow for the Onboarding Agent "
        "(thin wrapper around the existing target-first pipeline).")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--client-name", required=True)
    p.add_argument("--client-id", default="")
    p.add_argument("--run-id", default="")
    p.add_argument("--project-dir", default="")
    p.add_argument("--output-root", default="")
    p.add_argument("--mode", choices=list(VALID_MODES) + ["mi_mna"], default="mi_only")
    p.add_argument("--registry", default="config/system/fields_registry.yaml")
    p.add_argument("--aliases-dir", default="config/system")
    # Target-first artefacts are required by this workflow, so mapping review is
    # ON by default; --no-mapping-review disables it for debugging.
    p.add_argument("--enable-mapping-review", action="store_true", default=True,
                   help="(default on) generate the deterministic target-first artefacts.")
    p.add_argument("--no-mapping-review", dest="enable_mapping_review",
                   action="store_false", help="disable target-first artefact generation.")
    p.add_argument("--enable-llm-target-advisor", action="store_true",
                   help="run the target-first LLM advisor (advisory only; writes 36_*).")
    p.add_argument("--no-llm-target-advisor", dest="enable_llm_target_advisor",
                   action="store_false", help="explicitly disable the LLM target advisor.")
    p.set_defaults(enable_llm_target_advisor=False)
    p.add_argument("--target-first-decisions", default="",
                   help="approved 34_target_first_decisions.yaml to apply (second pass).")
    p.add_argument("--llm-max-cost-gbp", type=float, default=1.0)
    p.add_argument("--llm-max-calls", type=int, default=None)
    p.add_argument("--llm-max-items-per-call", type=int, default=None)
    return p


def _print_console(s: Dict[str, Any]) -> None:
    line = "=" * 64
    print(line)
    print(f"Operator workflow completed for: {s['client_name']}")
    print(f"Status: {s['status']}")
    print(f"Run ID: {s['workflow_run_id']}")
    print("")
    print(f"Target fields: {s['target_fields_count']}")
    print(f"Gate 4 decisions remaining: {s['human_decision_queue_count_after']}")
    print(f"Blocking decisions: {s['blocking_decisions_count']}")
    print(f"Non-blocking confirmations: {s['non_blocking_confirmations_count']}")
    if s["workflow_stage"] == SECOND_PASS:
        print("")
        print(f"Approved decisions supplied: {s['approved_decisions_supplied_count']}")
        print(f"Applied decisions: {s['applied_decisions_count']}")
        print(f"Remaining Gate 4 decisions: {s['human_decision_queue_count_after']}")
    print("")
    print("Decision template:")
    print(f"  {s['target_first_decisions_template_file'] or '—'}")
    print("")
    print("LLM target advisor:")
    print(f"  enabled: {str(s['llm_target_advisor_enabled']).lower()}")
    print(f"  advised: {s['llm_target_advisor_advised_count']}")
    print(f"  cost: £{s['llm_target_advisor_estimated_cost_gbp']}")
    print("")
    print("Review pack:")
    print(f"  {s['review_pack_html'] or '—'}")
    print("")
    print("Next action:")
    print(f"  {s['next_operator_action']}")
    print(line)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_operator_workflow(
        input_dir=args.input_dir, client_name=args.client_name,
        client_id=args.client_id, run_id=args.run_id, project_dir=args.project_dir,
        output_root=args.output_root, mode=args.mode, registry=args.registry,
        aliases_dir=args.aliases_dir, enable_mapping_review=args.enable_mapping_review,
        enable_llm_target_advisor=args.enable_llm_target_advisor,
        target_first_decisions=args.target_first_decisions,
        llm_max_cost_gbp=args.llm_max_cost_gbp, llm_max_calls=args.llm_max_calls,
        llm_max_items_per_call=args.llm_max_items_per_call)
    _print_console(summary)
    return 0 if summary["status"] != FAILED else 2


if __name__ == "__main__":
    raise SystemExit(main())
