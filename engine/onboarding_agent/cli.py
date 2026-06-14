"""
cli.py
======

Command-line entry point for the Trakt Onboarding Agent v2.

Example
-------
    python -m engine.onboarding_agent.cli \\
      --input-dir synthetic_onboarding_pack \\
      --client-name SYNTHETIC_ONBOARDING_TEST \\
      --output-dir onboarding_output/synthetic_onboarding_test \\
      --registry config/system/fields_registry.yaml \\
      --aliases-dir config/system

Produces the numbered onboarding pack artefacts (01..09) and a static HTML
review pack under ``--output-dir``. It does not run Gates 1–5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python -m engine.onboarding_agent.cli` or directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent.answer_ingestion import ingest_answers
from engine.onboarding_agent.mode_policy import VALID_MODES
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding
from engine.onboarding_agent.review_interpreter import interpret_answers_to_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trakt Onboarding Agent v2 — produce a reviewable onboarding pack."
    )
    p.add_argument("--input-dir", required=True, help="Folder of lender onboarding artefacts.")
    p.add_argument("--client-name", required=True, help="Client / lender name.")
    p.add_argument(
        "--output-dir",
        default="",
        help="Output folder for the onboarding pack (defaults to --project-dir).",
    )
    # --- PART 3/4: Azure-ready run-folder contract ---
    p.add_argument("--project-dir", default="", help="Run project dir (holds numbered artefacts).")
    p.add_argument("--output-root", default="", help="Consolidated output root (default <project>/output).")
    p.add_argument("--client-id", default="", help="Stable client id for the run folder / manifests.")
    p.add_argument("--run-id", default="", help="Run id (e.g. run_001 or an ISO timestamp).")
    p.add_argument(
        "--storage-backend",
        choices=["local", "azure_blob_compatible"],
        default="local",
        help="local (default) | azure_blob_compatible. The latter only adds "
        "Azure-style URIs to manifests; it never uploads.",
    )
    p.add_argument("--input-uri", default="", help="Optional azure:// URI for the uploaded input pack.")
    p.add_argument("--output-uri", default="", help="Optional azure:// URI for the output root.")
    p.add_argument(
        "--mode",
        choices=list(VALID_MODES) + ["mi_mna"],
        default="",
        help="Onboarding mode (default: regulatory_mi). mi_only | mna_dd | "
        "regulatory_mi | warehouse_securitisation. ('mi_mna' is a deprecated "
        "alias for mna_dd.)",
    )
    p.add_argument(
        "--registry",
        default="config/system/fields_registry.yaml",
        help="Canonical field registry YAML.",
    )
    p.add_argument(
        "--aliases-dir",
        default="config/system",
        help="Directory containing aliases_*.yaml.",
    )
    p.add_argument("--project-id", default="", help="Optional explicit project id.")
    p.add_argument(
        "--no-handoff",
        action="store_true",
        help="Skip writing the draft pipeline handoff artefacts (09_*).",
    )
    p.add_argument(
        "--answers",
        default="",
        help="Optional answers YAML — after generating the pack, ingest these "
        "answers and write the approved artefacts (10..15).",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Required to actually write approved artefacts when ingesting answers.",
    )
    p.add_argument(
        "--enable-regulatory-reporting",
        action="store_true",
        help="For warehouse_securitisation: activate regulatory fields in scope.",
    )
    # --- PART 9/10: client mapping memory ---
    p.add_argument(
        "--client-memory-dir",
        default="",
        help="Client-scoped mapping-memory dir (default "
        "<output-parent>/<client_id>/client_memory).",
    )
    p.add_argument(
        "--apply-client-memory",
        dest="apply_client_memory",
        action="store_true",
        default=None,
        help="Force-apply client mapping memory (default: apply if present and "
        "--client-id is provided).",
    )
    p.add_argument(
        "--no-apply-client-memory",
        dest="apply_client_memory",
        action="store_false",
        help="Never apply client mapping memory for this run.",
    )
    # --- Low-cost LLM mapping review (PART 8). Off / deterministic by default. ---
    p.add_argument(
        "--enable-llm-review",
        action="store_true",
        help="Opt in to the targeted, bounded LLM mapping reviewer (off by default).",
    )
    p.add_argument(
        "--llm-max-calls",
        type=int,
        default=None,
        help="Cap the number of LLM calls per onboarding run (default from config).",
    )
    p.add_argument(
        "--llm-max-items-per-call",
        type=int,
        default=None,
        help="Cap the number of items sent per LLM call (default from config).",
    )
    p.add_argument(
        "--llm-budget-profile",
        choices=["off", "low", "standard"],
        default="",
        help="LLM budget profile: off (default) | low | standard.",
    )
    return p


def build_ingest_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest answered gap questions and write approved artefacts."
    )
    p.add_argument("--project-dir", required=True, help="Existing onboarding output folder.")
    p.add_argument("--answers", required=True, help="Answers YAML file.")
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Required to write approved artefacts (human confirmation gate).",
    )
    return p


def build_promote_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dry-run Azure-ready promotion: build central tapes, lineage, "
        "gaps, domain coverage and handoff manifests. Never runs Gates 1–5."
    )
    p.add_argument("--project-dir", required=True, help="Existing onboarding output folder.")
    p.add_argument("--output-root", default="", help="Consolidated output root (default <project>/output).")
    p.add_argument("--client-id", default="", help="Client id (defaults to run summary / project name).")
    p.add_argument("--run-id", default="", help="Run id (defaults to run summary).")
    p.add_argument("--input-dir", default="", help="Override input dir (default from run summary).")
    p.add_argument(
        "--registry", default="config/system/fields_registry.yaml",
        help="Canonical field registry YAML.",
    )
    p.add_argument("--mode", default="", help="Override onboarding mode (default from run summary).")
    p.add_argument("--approved-only", action="store_true",
                   help="Require approved artefacts (10–15) to be present.")
    p.add_argument("--dry-run", action="store_true",
                   help="Dry-run (default behaviour; no Azure upload, no Gates).")
    p.add_argument(
        "--storage-backend", choices=["local", "azure_blob_compatible"], default="local",
        help="local | azure_blob_compatible (adds Azure URIs to manifests only).",
    )
    p.add_argument("--input-uri", default="", help="Optional azure:// URI for the input pack.")
    p.add_argument("--output-uri", default="", help="Optional azure:// URI for the output root.")
    p.add_argument("--enable-regulatory-reporting", action="store_true",
                   help="Activate regulatory fields in scope (warehouse mode).")
    return p


def run_promote(args) -> int:
    import json as _json

    from engine.onboarding_agent import (
        central_tape_builder,
        domain_coverage as _dc,
        promotion_planner,
        storage_paths,
    )

    project_dir = Path(args.project_dir)
    run_summary = {}
    rs_path = project_dir / "09_onboarding_run_summary.json"
    if rs_path.exists():
        run_summary = _json.loads(rs_path.read_text(encoding="utf-8"))

    mode = args.mode or run_summary.get("onboarding_mode", "regulatory_mi")
    client_id = args.client_id or run_summary.get("client_id", "") or project_dir.name
    run_id = args.run_id or run_summary.get("run_id", "") or "run"
    client_name = run_summary.get("client_name", client_id)
    input_dir = args.input_dir or run_summary.get("input_dir", "")

    if args.approved_only:
        missing = [n for n in ("10_approved_onboarding_project.yaml", "11_approved_config.yaml")
                   if not (project_dir / n).exists()]
        if missing:
            print(f"[promote] --approved-only set but approved artefacts missing: {missing}")
            print("[promote] Run `ingest-answers --confirm` first.")
            return 2

    run_paths = storage_paths.resolve_run_paths(
        project_dir=str(project_dir),
        input_dir=input_dir or None,
        output_root=args.output_root or None,
        client_id=client_id,
        run_id=run_id,
        storage_backend=args.storage_backend,
        input_uri=args.input_uri,
        output_uri=args.output_uri,
    )

    # Domain coverage — build if not already present.
    coverage = _dc.load_coverage(project_dir / "17_domain_coverage.json")
    if not coverage:
        coverage = _dc.rebuild_coverage(
            project_dir, args.registry, mode,
            regulatory_reporting_enabled=args.enable_regulatory_reporting,
        )
        _dc.write_domain_coverage_artifacts(coverage, project_dir)

    tape_result = central_tape_builder.build_central_tapes(
        project_dir, run_paths, args.registry, mode=mode,
        regulatory_reporting_enabled=args.enable_regulatory_reporting,
    )
    plan = promotion_planner.build_promotion_plan(
        project_dir, run_paths, tape_result, coverage, mode,
        args.enable_regulatory_reporting, client_name=client_name, project_id=client_id,
    )

    # Reflect the promotion results back into the static review pack.
    try:
        from engine.onboarding_agent.review_pack_builder import refresh_review_pack_promotion
        refresh_review_pack_promotion(project_dir, Path(run_paths.output_root))
    except Exception:
        pass

    print("=" * 64)
    print("Onboarding promotion (DRY-RUN — no Gates, no Azure upload)")
    print(f"Mode: {mode} · storage_backend: {run_paths.storage_backend}")
    print(f"Central lender tape: {tape_result['central_lender_tape_created']} "
          f"({tape_result['loan_count']} loans, {tape_result['mapped_field_count']} fields)")
    print(f"Central pipeline tape: {tape_result['central_pipeline_tape_created']} "
          f"({tape_result['pipeline_count']} applications)")
    print(f"Conflicts: {tape_result['conflict_count']} · gaps: {tape_result['gap_count']}")
    print(f"Readiness: {plan['readiness_status']}")
    print("Manifests:")
    for k in ("promotion_plan_path", "handoff_manifest_path", "readiness_path", "pipeline_trigger_path"):
        print(f"  - {plan[k]}")
    print("=" * 64)
    return 0


def build_interpret_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interpret a natural-language answer into a structured spec (dry-run)."
    )
    p.add_argument("--project-dir", required=True, help="Existing onboarding output folder.")
    p.add_argument("--text", required=True, help="Natural-language answer to interpret.")
    return p


def build_compare_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Semantic-alignment parity audit: run the same headers through the "
        "Gate 1 semantic alignment engine and the new Onboarding Agent mapping path "
        "and emit a column-by-column delta (27_* artefacts). Read-only; no LLM."
    )
    p.add_argument("--input-file", required=True, help="Source CSV/XLSX to audit.")
    p.add_argument("--registry", default="config/system/fields_registry.yaml")
    p.add_argument("--aliases-dir", default="config/system")
    p.add_argument(
        "--mode", choices=list(VALID_MODES) + ["mi_mna"], default="regulatory_mi",
        help="Onboarding mode (default regulatory_mi).",
    )
    p.add_argument("--output-dir", required=True, help="Where to write 27_* artefacts.")
    p.add_argument("--enable-regulatory-reporting", action="store_true",
                   help="Activate regulatory fields in scope (warehouse mode).")
    return p


def run_compare(args) -> int:
    from engine.onboarding_agent.compare_semantic_alignment import run_and_write

    res = run_and_write(
        input_file=args.input_file, registry=args.registry, aliases_dir=args.aliases_dir,
        output_dir=args.output_dir, mode=args.mode,
        regulatory_reporting_enabled=args.enable_regulatory_reporting,
    )
    s = res["summary"]
    print("=" * 64)
    print("Semantic alignment parity audit")
    print(f"Input: {s['source_file']} · mode: {s['mode']} · columns: {s['columns_total']}")
    print(f"  mapped the same by both paths:        {s['mapped_both']}")
    print(f"  old mapped / new UNMAPPED (regression):{s['old_mapped_new_unmapped']}")
    print(f"  mapped differently:                   {s['old_mapped_new_different']}")
    print(f"  new mapped / old unmapped (stronger): {s['new_mapped_old_unmapped']}")
    print(f"  unmapped by both:                     {s['unmapped_both']}")
    print(f"  diverted out-of-scope (mode-safe):    {s['field_scope_excluded']}")
    print(f"  LLM used:                             {s['llm_used']}")
    print("Artefacts:")
    for k in ("csv", "json", "summary_md"):
        print(f"  - {res['paths'][k]}")
    print("=" * 64)
    return 0


def _print_ingestion_report(report) -> None:
    print("=" * 64)
    print("Answer ingestion complete")
    print(f"Approval status: {report['approval_status']}")
    print(f"  questions_total:   {report['questions_total']}")
    print(f"  blocking_total:    {report['blocking_total']}")
    print(f"  blocking_answered: {report['blocking_answered']}")
    print(f"  answers_invalid:   {report['answers_invalid']}")
    for inv in report.get("invalid_detail", []):
        print(f"    ! {inv['question_id']}: {inv['reason']}")
    print("Approved artefacts:")
    for a in report["artefacts_written"]:
        print(f"  - {a}")
    print("=" * 64)


def main(argv=None) -> int:
    import sys as _sys

    argv = list(_sys.argv[1:] if argv is None else argv)

    # Subcommand: interpret-answers (dry-run NL -> structured spec; never writes approved artefacts)
    if argv and argv[0] == "interpret-answers":
        args = build_interpret_parser().parse_args(argv[1:])
        out = interpret_answers_to_file(args.project_dir, args.text)
        print("=" * 64)
        print("Interpreted natural-language answer (dry-run, no approval written)")
        print(f"Spec written to: {out}")
        print("Run `ingest-answers --answers <spec> --confirm` to apply.")
        print("=" * 64)
        return 0

    # Subcommand: ingest-answers
    if argv and argv[0] == "ingest-answers":
        args = build_ingest_parser().parse_args(argv[1:])
        report = ingest_answers(args.project_dir, args.answers, confirm=args.confirm)
        _print_ingestion_report(report)
        return 0

    # Subcommand: promote (Azure-ready dry-run handoff)
    if argv and argv[0] == "promote":
        args = build_promote_parser().parse_args(argv[1:])
        return run_promote(args)

    # Subcommand: compare-semantic-alignment (parity audit, read-only)
    if argv and argv[0] == "compare-semantic-alignment":
        args = build_compare_parser().parse_args(argv[1:])
        return run_compare(args)

    args = build_parser().parse_args(argv)

    # Resolve the Azure-ready run-folder contract (PART 3/4). The numbered
    # review artefacts continue to live in the project/output dir.
    from engine.onboarding_agent import storage_paths
    output_dir = args.output_dir or args.project_dir
    if not output_dir:
        build_parser().error("one of --output-dir or --project-dir is required")
    project_dir = args.project_dir or output_dir
    run_paths = storage_paths.resolve_run_paths(
        project_dir=project_dir,
        input_dir=args.input_dir,
        output_root=args.output_root or None,
        client_id=args.client_id,
        run_id=args.run_id,
        storage_backend=args.storage_backend,
        input_uri=args.input_uri,
        output_uri=args.output_uri,
    )
    args.output_dir = output_dir

    # Backward-compatibility: warn on deprecated mode aliases.
    if args.mode:
        from engine.onboarding_agent.mode_policy import resolve_mode_alias
        _canonical, _dep = resolve_mode_alias(args.mode)
        if _dep:
            print(f"[deprecation] {_dep}")

    project = run_onboarding(
        input_dir=args.input_dir,
        client_name=args.client_name,
        output_dir=args.output_dir,
        registry_path=args.registry,
        aliases_dir=args.aliases_dir,
        project_id=args.project_id,
        enable_handoff=not args.no_handoff,
        mode=args.mode,
        regulatory_reporting_enabled=args.enable_regulatory_reporting,
        enable_llm_review=args.enable_llm_review,
        llm_budget_profile=args.llm_budget_profile,
        llm_max_calls=args.llm_max_calls,
        llm_max_items_per_call=args.llm_max_items_per_call,
        client_id=args.client_id,
        run_id=args.run_id,
        storage_backend=args.storage_backend,
        input_uri=args.input_uri,
        output_uri=args.output_uri,
        client_memory_dir=args.client_memory_dir,
        apply_client_memory=args.apply_client_memory,
    )

    print("=" * 64)
    print(f"Onboarding pack generated for: {project.client_name}")
    print(f"Status: {project.review_status}")
    summary = project.to_summary_dict()["counts"]
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Output dir: {project.output_dir}")
    print("Artifacts:")
    for a in project.generated_artifacts:
        print(f"  - {Path(a).name}")
    print("=" * 64)

    # Optional inline answer ingestion (requires --confirm to write approved artefacts).
    if args.answers:
        report = ingest_answers(args.output_dir, args.answers, confirm=args.confirm)
        _print_ingestion_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
