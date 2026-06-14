"""
demo_onboarding_v1.py
=====================

PART 12 — end-to-end v1 Onboarding Agent demo on the synthetic domain pack.

Runs a complete, review-first onboarding story for a synthetic lender and prints
a plain-English summary. It deliberately reuses the existing engine functions
(no new onboarding process):

    1. clean a demo project dir
    2. run onboarding on synthetic_onboarding_pack_domain_based/scenario_a_combined
    3. generate the review pack + gap questions
    4. apply a demo answers file that closes the key gaps
       (reporting date, balance source-of-truth, employment_status=manual,
        missing core fields, geography policy, warehouse terms)
    5. save selected decisions into client mapping memory
    6. re-run mapping WITH memory applied to demonstrate fewer unresolved gaps
    7. promote dry-run (central tapes, lineage, gaps, manifests, trigger)
    8. print a simple final summary

Nothing here runs Gates 1–5, uploads to Azure, or writes production config.

CLI::

    python -m engine.onboarding_agent.demo_onboarding_v1 \\
      --output-dir onboarding_output/demo_onboarding_v1 \\
      --client-id demo_client \\
      --run-id demo_run_001
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import mapping_memory as mm
from engine.onboarding_agent import streamlit_onboarding_workbench as wb
from engine.onboarding_agent.answer_ingestion import ingest_answers
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

PACK = _REPO_ROOT / "synthetic_onboarding_pack_domain_based" / "scenario_a_combined"
REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ALIASES = str(_REPO_ROOT / "config" / "system")

# Gap categories that count as "unresolved mapping work" for the before/after
# memory comparison.
_MAPPING_GAP_CATEGORIES = {"source_of_truth", "enum", "core_field", "memory_conflict"}


def _count_blocking(project) -> int:
    return sum(1 for q in project.gap_questions if q.severity == "blocking")


def _count_mapping_gaps(project) -> int:
    return sum(1 for q in project.gap_questions if q.category in _MAPPING_GAP_CATEGORIES)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _build_demo_answers(project_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Build a demo answers dict that closes the key gaps for scenario A.

    Driven by the actual generated gap questions so it stays valid as the pack
    evolves: each question is answered with a sensible, demo-appropriate choice.
    """
    questions = yaml.safe_load((project_dir / "07_gap_questions.yaml").read_text(encoding="utf-8")) or []
    answers: Dict[str, Dict[str, Any]] = {}
    for q in questions:
        qid = q.get("question_id")
        cat = q.get("category")
        cands = q.get("candidate_answers", []) or []
        default = q.get("default_recommendation", "")
        subject = q.get("subject", "")
        subject_value = q.get("subject_value", "")

        if cat == "date":
            ans = "2026-01-31"  # choose the loan-tape reporting date as authoritative
        elif cat == "source_of_truth":
            # Prefer the master loan/collateral tape as the balance source of truth.
            ans = next((c for c in cands if "master" in c), default or (cands[0] if cands else ""))
        elif cat == "enum":
            if subject_value == "manual":
                ans = "treat_as_missing"   # placeholder/process token -> missing
            else:
                ans = "map_to_OTHR"         # e.g. PART_TIME -> OTHR
        elif cat == "geography":
            ans = "GBZZZ"
        elif cat == "core_field":
            ans = "mark_unavailable"        # accept missing valuation/rate/etc. fields
        elif cat == "warehouse":
            ans = default if default and default != "requires_review" else (
                cands[0] if cands else "mark_unavailable")
        else:
            ans = default if (default in cands or not cands) else cands[0]

        if ans:
            answers[qid] = {"answer": ans, "approved_by": "demo_analyst",
                            "note": q.get("question", "")}
    return answers


def _demo_memory_decisions(client_id: str, mode: str, project_dir: Path) -> List[Dict[str, Any]]:
    """Selected decisions to persist as client memory (PART 9).

    These are the decisions a reviewer would not want to re-make next month:
      * the balance source-of-truth precedence
      * the employment_status enum decisions
      * a remembered mapping for loan_amount -> original_principal_balance
    """
    decisions: List[Dict[str, Any]] = [
        {
            "decision_type": mm.DECISION_SOURCE_PRECEDENCE,
            "canonical_field": "current_principal_balance",
            "mode": mode, "domain": "loan",
            "evidence": {"primary_source_file": "master_loan_collateral_tape.csv",
                         "reviewed_in_run_id": project_dir.name},
        },
        {
            "decision_type": mm.DECISION_ENUM_MAPPING,
            "canonical_field": "employment_status", "source_value": "manual",
            "mode": mode, "domain": "borrower",
            "evidence": {"decision": "treat_as_missing"},
        },
        {
            "decision_type": mm.DECISION_ENUM_MAPPING,
            "canonical_field": "employment_status", "source_value": "PART_TIME",
            "mode": mode, "domain": "borrower",
            "evidence": {"decision": "map_to_OTHR"},
        },
        {
            "decision_type": mm.DECISION_MAPPING_OVERRIDE,
            "source_file_pattern": "master_loan_collateral_tape*",
            "source_column": "loan_amount",
            "canonical_field": "original_principal_balance",
            "mode": mode, "domain": "loan",
            "evidence": {"value_match_rate": 1.0, "reviewed_in_run_id": project_dir.name},
        },
    ]
    return decisions


def run_demo(output_dir: str, client_id: str, run_id: str,
             mode: str = "regulatory_mi") -> Dict[str, Any]:
    """Run the full demo and return a structured result summary."""
    out_root = Path(output_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    project_dir = out_root
    memory_dir = out_root.parent / client_id / "client_memory"
    if memory_dir.exists():
        shutil.rmtree(memory_dir)

    # --- 1/2/3: first onboarding run (no memory) ---
    project1 = run_onboarding(
        input_dir=str(PACK), client_name="DEMO LENDER",
        output_dir=str(project_dir), registry_path=REGISTRY, aliases_dir=ALIASES,
        mode=mode, client_id=client_id, run_id=run_id,
    )
    blocking_before = _count_blocking(project1)
    mapping_gaps_before = _count_mapping_gaps(project1)
    input_files = len(project1.file_inventory)
    domains_detected = sorted({
        d.domain for d in project1.domain_coverage
        if d.status in ("covered", "partially_covered")
    })

    # --- 4: apply demo answers (generate 25 + ingest -> approved 10–15) ---
    demo_answers = _build_demo_answers(project_dir)
    wb.generate_answers_yaml(project_dir, demo_answers, project_id=client_id)
    wb.append_action_log(project_dir, client_id, run_id, "demo_generate_answers",
                         outputs_written=[wb.ANSWERS_FILE])
    ingest_report = ingest_answers(str(project_dir), str(project_dir / wb.ANSWERS_FILE),
                                   confirm=True)
    blocking_after = ingest_report["blocking_total"] - ingest_report["blocking_answered"]
    wb.append_action_log(project_dir, client_id, run_id, "demo_ingest_answers",
                         status=ingest_report["approval_status"],
                         outputs_written=ingest_report["artefacts_written"])

    # --- 5: save selected decisions to client mapping memory ---
    mem_decisions = _demo_memory_decisions(client_id, mode, project_dir)
    mem_res = wb.save_decisions_to_memory(
        mem_decisions, client_id, memory_dir=str(memory_dir), run_id=run_id,
        approved_by="demo_analyst")
    wb.append_action_log(project_dir, client_id, run_id, "demo_save_client_memory",
                         outputs_written=[mem_res["memory_dir"]])

    # --- 6: re-run mapping WITH memory applied (separate dir, for comparison) ---
    rerun_dir = out_root / "rerun_with_memory"
    project2 = run_onboarding(
        input_dir=str(PACK), client_name="DEMO LENDER",
        output_dir=str(rerun_dir), registry_path=REGISTRY, aliases_dir=ALIASES,
        mode=mode, client_id=client_id, run_id=run_id + "_rerun",
        client_memory_dir=str(memory_dir), apply_client_memory=True,
    )
    mapping_gaps_after = _count_mapping_gaps(project2)

    # --- 7: promote dry-run on the answered run (central tapes + manifests) ---
    promote = wb.promote_dry_run(project_dir, REGISTRY, client_id, run_id, mode)
    wb.append_action_log(project_dir, client_id, run_id, "demo_promote_dry_run",
                         status=promote["plan"]["readiness_status"])
    tape = promote["tape_result"]
    readiness = promote["plan"]["readiness"]

    # --- 8: final review pack refresh ---
    wb.refresh_review_pack(project_dir)

    lender_rows = len(_read_csv(Path(tape["central_lender_tape_path"])))
    pipeline_rows = (len(_read_csv(Path(tape["central_pipeline_tape_path"])))
                     if tape["central_pipeline_tape_created"] else 0)

    return {
        "project_dir": str(project_dir),
        "input_files": input_files,
        "domains_detected": domains_detected,
        "lender_tape_rows": lender_rows,
        "pipeline_rows": pipeline_rows,
        "blocking_before": blocking_before,
        "blocking_after": max(0, blocking_after),
        "mapping_gaps_before": mapping_gaps_before,
        "mapping_gaps_after": mapping_gaps_after,
        "client_memory_entries_saved": mem_res["saved"],
        "memory_applied_summary": project2.client_memory_summary,
        "pipeline_trigger_path": promote["plan"]["pipeline_trigger_path"],
        "readiness_status": promote["plan"]["readiness_status"],
        "ready_for_mi": bool(readiness.get("ready_for_mi_agent")),
        "ready_for_regulatory_projection": bool(readiness.get("ready_for_regulatory_projection")),
    }


def _print_summary(r: Dict[str, Any]) -> None:
    print("=" * 64)
    print("Demo completed.")
    print(f"Input files: {r['input_files']}")
    print(f"Domains detected: {', '.join(r['domains_detected'])}")
    print(f"Central lender tape rows: {r['lender_tape_rows']}")
    print(f"Central pipeline rows: {r['pipeline_rows']}")
    print(f"Blocking gaps before answers: {r['blocking_before']}")
    print(f"Blocking gaps after answers: {r['blocking_after']}")
    print(f"Unresolved mapping gaps before memory: {r['mapping_gaps_before']}")
    print(f"Unresolved mapping gaps after memory: {r['mapping_gaps_after']}")
    print(f"Client memory entries saved: {r['client_memory_entries_saved']}")
    print(f"Pipeline trigger written: {r['pipeline_trigger_path']}")
    print(f"Readiness: {r['readiness_status']}")
    print(f"Ready for MI: {'yes' if r['ready_for_mi'] else 'no'}")
    print(f"Ready for regulatory projection: "
          f"{'yes' if r['ready_for_regulatory_projection'] else 'no'}")
    print("=" * 64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trakt v1 Onboarding Agent end-to-end demo.")
    p.add_argument("--output-dir", default="onboarding_output/demo_onboarding_v1")
    p.add_argument("--client-id", default="demo_client")
    p.add_argument("--run-id", default="demo_run_001")
    p.add_argument("--mode", default="regulatory_mi")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run_demo(args.output_dir, args.client_id, args.run_id, args.mode)
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
