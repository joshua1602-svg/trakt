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
    p.add_argument("--output-dir", required=True, help="Output folder for the onboarding pack.")
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


def build_interpret_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interpret a natural-language answer into a structured spec (dry-run)."
    )
    p.add_argument("--project-dir", required=True, help="Existing onboarding output folder.")
    p.add_argument("--text", required=True, help="Natural-language answer to interpret.")
    return p


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

    args = build_parser().parse_args(argv)

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
