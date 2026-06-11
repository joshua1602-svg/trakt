"""
cli/onboarding_review_cli.py

CLI fallback for the Onboarding Review & Approval Workbench.

Usage:
    python cli/onboarding_review_cli.py --run-dir out/run_20240101_120000_abc123

Interface:
    For each review item the user is prompted:
        A = Approve suggestion
        E = Edit / override
        S = Skip / ignore

After all sections are completed the CLI:
    1. Writes decision JSON files (questionnaire_answers, mapping_overrides, enum_overrides)
    2. Persists approved decisions to alias/enum/config learning files
    3. Re-runs the Onboarding Agent with the approved answers
    4. Displays the final onboarding status
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Local imports — resilient to running from different CWD
# ---------------------------------------------------------------------------

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from agents.onboarding_schemas import OnboardingResult
from agents.review_schemas import (
    EnumDecision,
    MappingDecision,
    QuestionAnswer,
    ReviewSubmission,
    build_enum_overrides_json,
    build_mapping_overrides_json,
    build_questionnaire_answers_json,
)
from agents.learning_persistence import (
    persist_config_answers,
    persist_enum_decisions,
    persist_mapping_decisions,
)


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_DIM = "\033[2m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + _RESET


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def _wrap(text: str, indent: int = 4) -> str:
    return textwrap.fill(text, width=72, initial_indent=" " * indent, subsequent_indent=" " * indent)


def _prompt(msg: str, valid: str = "AES", default: str = "") -> str:
    """Prompt until a valid single-char response is received."""
    valid_upper = valid.upper()
    hint = "/".join(
        (c.upper() if c.upper() == default.upper() else c.lower()) for c in valid_upper
    )
    while True:
        raw = input(f"{msg} [{hint}]: ").strip().upper()
        if not raw and default:
            return default.upper()
        if raw and raw[0] in valid_upper:
            return raw[0]
        print(f"  Please enter one of: {', '.join(valid_upper)}")


def _input_line(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw if raw else default


# ---------------------------------------------------------------------------
# Section 1: Config bootstrap questions
# ---------------------------------------------------------------------------

def _review_questions(
    questions: List[Dict[str, Any]],
) -> List[QuestionAnswer]:
    if not questions:
        return []

    print()
    print(_c(_hr("═"), _BOLD))
    print(_c("  SECTION 1 — CONFIG BOOTSTRAP QUESTIONS", _BOLD))
    print(_c(_hr("═"), _BOLD))

    answers: List[QuestionAnswer] = []
    blockers = [q for q in questions if q.get("blocking")]
    advisories = [q for q in questions if not q.get("blocking")]

    def _ask_question(q: Dict[str, Any]) -> QuestionAnswer:
        qid = q.get("question_id", "")
        text = q.get("question_text") or q.get("question", "")
        suggestion = q.get("suggested_answer") or q.get("default", "")
        options = q.get("options") or []
        blocking = q.get("blocking", False)

        flag = _c(" [BLOCKER]", _RED) if blocking else ""
        print()
        print(_c(f"  Q: {text}{flag}", _BOLD))
        if suggestion:
            print(_c(f"     Suggested: {suggestion}", _DIM))
        if options:
            print(_c("     Options: " + ", ".join(str(o) for o in options), _DIM))

        choice = _prompt("  → (A)pprove suggestion  (E)dit  (S)kip", default="A" if suggestion else "E")

        if choice == "S":
            return QuestionAnswer(question_id=qid, answer="", approved=False)

        if choice == "A" and suggestion:
            return QuestionAnswer(question_id=qid, answer=str(suggestion), approved=True)

        # Edit
        final_answer = _input_line("  → Enter answer", default=str(suggestion) if suggestion else "")
        approved = bool(final_answer.strip())
        return QuestionAnswer(question_id=qid, answer=final_answer, approved=approved)

    if blockers:
        print()
        print(_c(f"  Blocker questions ({len(blockers)}) — must be resolved:", _RED))
        for q in blockers:
            answers.append(_ask_question(q))

    if advisories:
        print()
        print(_c(f"  Advisory questions ({len(advisories)}):", _CYAN))
        for q in advisories:
            answers.append(_ask_question(q))

    return answers


# ---------------------------------------------------------------------------
# Section 2: Field mapping review
# ---------------------------------------------------------------------------

def _review_mappings(
    mapping_items: List[Any],
    canonical_fields: Optional[List[str]] = None,
) -> List[MappingDecision]:
    flagged = [m for m in mapping_items if getattr(m, "requires_review", False) or getattr(m, "blocker", False)]
    if not flagged:
        return []

    print()
    print(_c(_hr("═"), _BOLD))
    print(_c("  SECTION 2 — FIELD MAPPING REVIEW", _BOLD))
    print(_c(_hr("═"), _BOLD))
    print(_c(f"  {len(flagged)} field mapping(s) require review\n", _DIM))

    decisions: List[MappingDecision] = []

    for item in flagged:
        raw = item.raw_field
        suggested = item.suggested_canonical_field or ""
        conf = item.confidence
        source = item.mapping_source
        blocker = item.blocker

        flag = _c(" [BLOCKER]", _RED) if blocker else ""
        conf_color = _GREEN if conf >= 0.92 else (_YELLOW if conf >= 0.75 else _RED)
        print()
        print(_c(f"  Raw field:   {raw}{flag}", _BOLD))
        print(f"  Suggested:   {suggested or '(none)'}")
        print(f"  Confidence:  {_c(f'{conf:.0%}', conf_color)}  Source: {source}")
        if item.sample_values:
            print(_c("  Samples:     " + ", ".join(str(v) for v in item.sample_values[:3]), _DIM))

        choice = _prompt("  → (A)pprove  (E)dit/Override  (S)kip", default="A" if suggested else "E")

        if choice == "S":
            decisions.append(MappingDecision(raw_field=raw, approved=False))
            continue

        if choice == "A" and suggested:
            decisions.append(MappingDecision(raw_field=raw, approved=True, selected_canonical_field=suggested))
            continue

        # Edit / Override
        print("  Enter canonical field name (leave blank to skip):")
        if canonical_fields:
            print(_c("  Known fields (sample): " + ", ".join(canonical_fields[:10]), _DIM))
        override = _input_line("  → Canonical field", default=suggested)
        if override.strip():
            decisions.append(MappingDecision(raw_field=raw, approved=True, selected_canonical_field=override.strip()))
        else:
            decisions.append(MappingDecision(raw_field=raw, approved=False))

    return decisions


# ---------------------------------------------------------------------------
# Section 3: Enum value review
# ---------------------------------------------------------------------------

def _review_enums(
    enum_items: List[Any],
) -> List[EnumDecision]:
    flagged = [e for e in enum_items if e.requires_review]
    if not flagged:
        return []

    print()
    print(_c(_hr("═"), _BOLD))
    print(_c("  SECTION 3 — ENUM VALUE REVIEW", _BOLD))
    print(_c(_hr("═"), _BOLD))
    print(_c(f"  {len(flagged)} enum value(s) require review\n", _DIM))

    decisions: List[EnumDecision] = []

    # Group by field name for display
    by_field: Dict[str, List[Any]] = {}
    for e in flagged:
        by_field.setdefault(e.field_name, []).append(e)

    for field_name, items in by_field.items():
        print()
        print(_c(f"  Field: {field_name}", _BOLD))
        print(_c(_hr("─", 60), _DIM))

        for item in items:
            raw = item.raw_value
            suggested = item.suggested_value or ""
            conf = item.confidence
            count = getattr(item, "sample_count", 0)
            blocker = item.blocker

            flag = _c(" [BLOCKER]", _RED) if blocker else ""
            conf_color = _GREEN if conf >= 0.92 else (_YELLOW if conf >= 0.75 else _RED)
            print()
            print(f"    Raw value:  {_c(raw, _BOLD)}{flag}  ({count} rows)")
            print(f"    Suggested:  {suggested or '(none)'}  {_c(f'{conf:.0%}', conf_color)}")

            choice = _prompt("    → (A)pprove  (E)dit  (S)kip", default="A" if suggested else "E")

            if choice == "S":
                decisions.append(EnumDecision(field_name=field_name, raw_value=raw, approved=False))
                continue

            if choice == "A" and suggested:
                decisions.append(EnumDecision(field_name=field_name, raw_value=raw, approved=True, selected_value=suggested))
                continue

            override = _input_line("    → Enter canonical value", default=suggested)
            if override.strip():
                decisions.append(EnumDecision(field_name=field_name, raw_value=raw, approved=True, selected_value=override.strip()))
            else:
                decisions.append(EnumDecision(field_name=field_name, raw_value=raw, approved=False))

    return decisions


# ---------------------------------------------------------------------------
# Write decision files
# ---------------------------------------------------------------------------

def _write_decision_files(
    run_output_dir: Path,
    run_id: str,
    answers: List[QuestionAnswer],
    mapping_decisions: List[MappingDecision],
    enum_decisions: List[EnumDecision],
) -> ReviewSubmission:
    submission = ReviewSubmission(
        run_id=run_id,
        question_answers=answers,
        mapping_decisions=mapping_decisions,
        enum_decisions=enum_decisions,
    )

    # questionnaire_answers.json — format consumed by run_onboarding_agent
    qa_path = run_output_dir / "questionnaire_answers.json"
    qa_path.write_text(
        json.dumps(build_questionnaire_answers_json(answers), indent=2),
        encoding="utf-8",
    )

    # mapping_overrides.json
    mo_path = run_output_dir / "mapping_overrides.json"
    mo_path.write_text(
        json.dumps(build_mapping_overrides_json(mapping_decisions), indent=2),
        encoding="utf-8",
    )

    # enum_overrides.json
    eo_path = run_output_dir / "enum_overrides.json"
    eo_path.write_text(
        json.dumps(build_enum_overrides_json(enum_decisions), indent=2),
        encoding="utf-8",
    )

    # full review submission
    submission.to_json(run_output_dir / f"{run_id}_review_submission.json")

    return submission


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _run_persistence(
    submission: ReviewSubmission,
    result: OnboardingResult,
    run_output_dir: Path,
    session_id: str,
) -> None:
    aliases_dir = result.aliases_dir
    enum_path = result.enum_mapping_path
    draft_path = result.config_bootstrap.draft_config_path if result.config_bootstrap else ""

    if aliases_dir and submission.mapping_decisions:
        try:
            n = persist_mapping_decisions(
                submission.mapping_decisions,
                Path(aliases_dir),
                session_id=session_id,
            )
            if n:
                print(_c(f"  ✓ {n} field alias(es) persisted to learning store", _GREEN))
        except Exception as exc:
            print(_c(f"  Warning: could not persist field aliases: {exc}", _YELLOW))

    if enum_path and submission.enum_decisions:
        try:
            n = persist_enum_decisions(
                submission.enum_decisions,
                Path(enum_path).parent / "enum_synonyms_confirmed.yaml",
                session_id=session_id,
            )
            if n:
                print(_c(f"  ✓ {n} enum synonym(s) persisted to learning store", _GREEN))
        except Exception as exc:
            print(_c(f"  Warning: could not persist enum synonyms: {exc}", _YELLOW))

    if draft_path and submission.question_answers:
        try:
            approved_out = run_output_dir / "approved_config_post_review.yaml"
            n = persist_config_answers(
                submission.question_answers,
                Path(draft_path),
                approved_out,
                session_id=session_id,
            )
            if n:
                print(_c(f"  ✓ {n} config value(s) applied to approved config", _GREEN))
        except Exception as exc:
            print(_c(f"  Warning: could not persist config answers: {exc}", _YELLOW))


# ---------------------------------------------------------------------------
# Re-run agent
# ---------------------------------------------------------------------------

def _rerun_agent(result: OnboardingResult, run_output_dir: Path) -> Optional[OnboardingResult]:
    try:
        from agents.onboarding_agent import run_onboarding_agent
    except ImportError as exc:
        print(_c(f"  Cannot import onboarding agent: {exc}", _RED))
        return None

    tape_path = result.raw_tape_path
    if not tape_path or not Path(tape_path).exists():
        print(_c("  Cannot re-run: original tape path not found in result.", _YELLOW))
        return None

    qa_path = run_output_dir / "questionnaire_answers.json"
    approved_config = run_output_dir / "approved_config_post_review.yaml"

    kwargs: Dict[str, Any] = {
        "tape_path": tape_path,
        "output_dir": str(run_output_dir.parent),
        "questionnaire_answers_path": str(qa_path) if qa_path.exists() else None,
        "approved_config_path": str(approved_config) if approved_config.exists() else None,
    }
    if result.schema_registry_path:
        kwargs["schema_registry_path"] = result.schema_registry_path
    if result.aliases_dir:
        kwargs["aliases_dir"] = result.aliases_dir
    if result.enum_mapping_path:
        kwargs["enum_mapping_path"] = result.enum_mapping_path

    print()
    print(_c("  Re-running Onboarding Agent…", _CYAN))
    try:
        new_result = run_onboarding_agent(**kwargs)
        return new_result
    except Exception as exc:
        print(_c(f"  Re-run failed: {exc}", _RED))
        return None


# ---------------------------------------------------------------------------
# Display final status
# ---------------------------------------------------------------------------

def _display_status(result: OnboardingResult) -> None:
    print()
    print(_c(_hr("═"), _BOLD))
    print(_c("  FINAL ONBOARDING STATUS", _BOLD))
    print(_c(_hr("═"), _BOLD))

    status_color = {
        "ready_for_validation": _GREEN,
        "review_required": _YELLOW,
        "blocked": _RED,
        "failed": _RED,
    }.get(result.status, _RESET)

    print()
    print(f"  Run ID:   {result.run_id}")
    print(f"  Status:   {_c(result.status.upper(), _BOLD, status_color)}")
    print()
    print(f"  Fields mapped:      {result.mapped_fields_count}/{result.total_input_fields}")
    print(f"  Fields for review:  {result.review_fields_count}")
    print(f"  Unmapped mandatory: {result.unmapped_mandatory_count}")
    print(f"  Enum success rate:  {result.enum_success_rate:.0%}")

    if result.narrative_summary:
        print()
        print(_c("  Summary:", _BOLD))
        for line in result.narrative_summary.splitlines():
            print(_wrap(line.strip(), indent=4))

    if result.errors:
        print()
        print(_c("  Errors:", _RED, _BOLD))
        for e in result.errors:
            print(_wrap(e, indent=4))

    if result.proceed_to_validation:
        print()
        print(_c("  ✓ Ready for Validation Agent", _GREEN, _BOLD))
        if result.onboarding_result_path:
            print(_c(f"    Result: {result.onboarding_result_path}", _DIM))
    else:
        print()
        print(_c("  ✗ Not ready for validation — resolve blockers and re-run.", _RED, _BOLD))

    print()
    print(_c(_hr("═"), _BOLD))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Onboarding Review CLI — review and approve onboarding agent decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to onboarding run output directory (e.g. out/run_20240101_120000_abc123)",
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Skip automatic re-run of the Onboarding Agent after submission",
    )
    parser.add_argument(
        "--result-file",
        default="",
        help="Override name of the onboarding result JSON file (default: auto-detected)",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(_c(f"Error: run directory not found: {run_dir}", _RED))
        return 1

    # Load onboarding result
    result_path: Optional[Path] = None
    if args.result_file:
        result_path = run_dir / args.result_file
    else:
        candidates = list(run_dir.glob("*onboarding_result*.json"))
        if candidates:
            result_path = sorted(candidates)[-1]

    if not result_path or not result_path.exists():
        print(_c(f"Error: no onboarding_result JSON found in {run_dir}", _RED))
        print("  Run the Onboarding Agent first, then pass its output directory here.")
        return 1

    try:
        result = OnboardingResult.from_json(result_path)
    except Exception as exc:
        print(_c(f"Error loading result file: {exc}", _RED))
        return 1

    run_id = result.run_id or result_path.stem

    # Banner
    print()
    print(_c(_hr("═"), _BOLD))
    print(_c("  ONBOARDING REVIEW WORKBENCH — CLI", _BOLD))
    print(_c(_hr("═"), _BOLD))
    print(f"  Run:    {run_id}")
    print(f"  Status: {result.status}")
    print(f"  Dir:    {run_dir}")
    print()

    # Show summary metrics
    print(f"  {result.mapped_fields_count}/{result.total_input_fields} fields mapped  |  "
          f"{result.review_fields_count} for review  |  "
          f"{result.unmapped_mandatory_count} unmapped mandatory  |  "
          f"enum {result.enum_success_rate:.0%}")

    if result.narrative_summary:
        print()
        for line in result.narrative_summary.splitlines():
            print(_wrap(line.strip(), indent=2))

    if result.proceed_to_validation and not result.blocker_questions and not result.mapping_review_items and not result.enum_review_items:
        print()
        print(_c("  No review items found. Agent has already approved this run.", _GREEN))
        _display_status(result)
        return 0

    # --- Section 1: Config questions ---
    all_questions = list(result.blocker_questions or []) + list(result.user_questions or [])
    answers = _review_questions(all_questions)

    # --- Section 2: Field mapping ---
    mapping_decisions = _review_mappings(result.mapping_review_items or [])

    # --- Section 3: Enum review ---
    enum_decisions = _review_enums(result.enum_review_items or [])

    # --- Confirm submission ---
    print()
    print(_c(_hr("─"), _DIM))
    approved_q = sum(1 for a in answers if a.approved)
    approved_m = sum(1 for m in mapping_decisions if m.approved and m.selected_canonical_field)
    approved_e = sum(1 for e in enum_decisions if e.approved and e.selected_value)
    print(f"  Ready to submit:  {approved_q} question(s)  |  {approved_m} mapping(s)  |  {approved_e} enum(s)")

    confirm = _prompt("  Submit and persist decisions?", "YN", default="Y")
    if confirm != "Y":
        print("  Aborted — no changes written.")
        return 0

    # --- Write decision files ---
    print()
    submission = _write_decision_files(run_dir, run_id, answers, mapping_decisions, enum_decisions)
    print(_c(f"  ✓ Decision files written to {run_dir}", _GREEN))

    # --- Persistence ---
    _run_persistence(submission, result, run_dir, session_id=run_id)

    # --- Re-run ---
    if not args.no_rerun:
        new_result = _rerun_agent(result, run_dir)
        if new_result:
            _display_status(new_result)
            return 0 if new_result.proceed_to_validation else 2

    # Display current status if no re-run
    _display_status(result)
    return 0 if result.proceed_to_validation else 2


if __name__ == "__main__":
    sys.exit(main())
