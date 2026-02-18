#!/usr/bin/env python3
"""
agent_orchestrator.py

End-to-end pipeline that chains:
  1. Deterministic Tiers 1-6 (semantic_alignment.py — FROZEN)
  2. LLM Tier 7 for unmapped / low-confidence headers
  3. Human review (CLI or Streamlit)
  4. Alias learning (persist confirmed mappings)
  5. Re-run deterministic pass with augmented aliases
  6. Governance artifact

Usage:
  python agent_orchestrator.py \\
      --input data/ERE_Portfolio_112025.csv \\
      --portfolio-type equity_release \\
      --registry config/system/fields_registry.yaml \\
      --aliases-dir engine/gate_1_alignment/aliases \\
      --mode cli \\
      --output-dir out
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal imports (relative to this file's location)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from llm_mapper_agent import (  # noqa: E402
    LLMFieldMapper,
    HumanReviewSession,
    AliasLearner,
    GovernanceLogger,
    LLMSuggestion,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _run_deterministic_pass(
    input_path: Path,
    portfolio_type: str,
    registry_path: Path,
    aliases_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Invoke semantic_alignment.py as a subprocess.
    Returns a dict with paths to the generated outputs.
    """
    aligner = HERE / "semantic_alignment.py"
    if not aligner.exists():
        raise FileNotFoundError(f"semantic_alignment.py not found at {aligner}")

    cmd = [
        sys.executable, str(aligner),
        "--input", str(input_path),
        "--portfolio-type", portfolio_type,
        "--registry", str(registry_path),
        "--aliases-dir", str(aliases_dir),
        "--output-dir", str(output_dir),
        "--output-schema", "active",
    ]

    logger.info("Running deterministic pass: %s", " ".join(cmd))
    if dry_run:
        logger.info("[DRY RUN] Would execute: %s", " ".join(cmd))
        stem = input_path.stem
        return {
            "canonical_csv": output_dir / f"{stem}_canonical_full.csv",
            "mapping_report_csv": output_dir / f"{stem}_mapping_report.csv",
            "unmapped_csv": output_dir / f"{stem}_unmapped_headers.csv",
            "mapping_report_json": output_dir / f"{stem}_header_mapping_report.json",
        }

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Deterministic pass failed:\n%s", result.stderr)
        raise RuntimeError("semantic_alignment.py exited with non-zero status.")
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.info(result.stderr)

    stem = input_path.stem
    return {
        "canonical_csv": output_dir / f"{stem}_canonical_full.csv",
        "mapping_report_csv": output_dir / f"{stem}_mapping_report.csv",
        "unmapped_csv": output_dir / f"{stem}_unmapped_headers.csv",
        "mapping_report_json": output_dir / f"{stem}_header_mapping_report.json",
    }


def _load_mapping_report(report_csv: Path) -> List[dict]:
    if not report_csv.exists():
        return []
    df = pd.read_csv(report_csv, dtype=str).fillna("")
    return df.to_dict(orient="records")


def _collect_llm_targets(
    report: List[dict],
    review_threshold: float,
) -> List[str]:
    """
    Return raw headers that should go to the LLM:
      - method == 'unmapped'  OR
      - method not in {exact, normalized, alias} AND confidence < review_threshold
    """
    targets: List[str] = []
    deterministic_wins = {"exact", "normalized", "alias"}
    for row in report:
        method = row.get("mapping_method", "unmapped")
        try:
            conf = float(row.get("confidence", 0.0))
        except (ValueError, TypeError):
            conf = 0.0

        if method == "unmapped":
            targets.append(row["raw_header"])
        elif method not in deterministic_wins and conf < review_threshold:
            targets.append(row["raw_header"])

    return targets


def _load_agent_config(config_path: Optional[Path]) -> dict:
    defaults = {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.0,
        "max_tokens": 4096,
        "review_threshold": 0.92,
        "auto_approve_threshold": None,
        "max_batch_size": 10,
        "max_api_calls_per_session": 10,
        "alias_output_file": "aliases_llm_confirmed.yaml",
        "governance_dir": "governance/agent_sessions",
    }
    if config_path and config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        agent_cfg = raw.get("agent", {}) or {}
        defaults.update(agent_cfg)
    return defaults


def _load_canonical_fields(registry_path: Path, portfolio_type: str) -> List[str]:
    """Load field names from registry filtered by portfolio_type."""
    import sys as _sys
    # Re-use the same select_registry_fields logic from semantic_alignment
    _sys.path.insert(0, str(HERE))
    from semantic_alignment import load_field_registry, select_registry_fields  # type: ignore
    registry = load_field_registry(registry_path)
    return select_registry_fields(registry, portfolio_type)


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    session_id = f"agent_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    logger.info("=== Agent Session: %s ===", session_id)

    input_path = Path(args.input)
    registry_path = Path(args.registry)
    aliases_dir = Path(args.aliases_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load agent config
    config_path = Path(args.config) if args.config else None
    cfg = _load_agent_config(config_path)
    review_threshold = float(args.review_threshold or cfg["review_threshold"])
    auto_approve_threshold = (
        float(args.auto_approve_above)
        if args.enable_auto_approve and args.auto_approve_above
        else None
    )
    governance_dir = output_dir / "governance" / "agent_sessions"

    # -----------------------------------------------------------------------
    # STEP 1: Deterministic pass (Tiers 1-6)
    # -----------------------------------------------------------------------
    logger.info("STEP 1: Deterministic alignment (Tiers 1-6)")
    if args.dry_run:
        logger.info("[DRY RUN] Skipping actual deterministic pass.")
        first_pass = _run_deterministic_pass(
            input_path, args.portfolio_type, registry_path, aliases_dir, output_dir, dry_run=True
        )
        det_report: List[dict] = []
    else:
        first_pass = _run_deterministic_pass(
            input_path, args.portfolio_type, registry_path, aliases_dir, output_dir
        )
        det_report = _load_mapping_report(first_pass["mapping_report_csv"])

    total_headers = len(det_report)
    mapped_headers = sum(1 for r in det_report if r.get("mapping_method", "unmapped") != "unmapped")
    unmapped_count = total_headers - mapped_headers
    low_conf_count = sum(
        1 for r in det_report
        if r.get("mapping_method", "unmapped") not in ("unmapped", "exact", "normalized", "alias")
        and float(r.get("confidence", 0) or 0) < review_threshold
    )

    det_stats = {
        "total_headers": total_headers,
        "mapped": mapped_headers,
        "unmapped": unmapped_count,
        "low_confidence": low_conf_count,
    }
    logger.info(
        "Deterministic pass complete — mapped=%d  unmapped=%d  low_conf=%d",
        mapped_headers, unmapped_count, low_conf_count,
    )

    # -----------------------------------------------------------------------
    # STEP 2: Skip LLM?
    # -----------------------------------------------------------------------
    if args.skip_llm:
        logger.info("--skip-llm: Producing unmapped report only (no LLM calls).")
        unmapped_headers = _collect_llm_targets(det_report, review_threshold)
        if unmapped_headers:
            unmapped_report_path = output_dir / f"{input_path.stem}_llm_candidates.csv"
            import pandas as _pd
            _pd.DataFrame({"raw_header": unmapped_headers}).to_csv(unmapped_report_path, index=False)
            logger.info("LLM candidates written to: %s", unmapped_report_path)
        return

    # -----------------------------------------------------------------------
    # STEP 3: Identify headers for LLM
    # -----------------------------------------------------------------------
    llm_targets = _collect_llm_targets(det_report, review_threshold)
    max_calls = int(args.max_llm_calls or cfg["max_api_calls_per_session"])
    batch_size = int(cfg["max_batch_size"])
    max_headers = max_calls * batch_size
    if len(llm_targets) > max_headers:
        logger.warning(
            "LLM target count (%d) exceeds budget cap (%d headers). Truncating.",
            len(llm_targets), max_headers,
        )
        llm_targets = llm_targets[:max_headers]

    if not llm_targets:
        logger.info("All headers resolved deterministically — no LLM calls needed.")
        _write_governance_skip(
            session_id, str(input_path), args.portfolio_type,
            det_stats, governance_dir,
        )
        return

    logger.info("STEP 3: Sending %d header(s) to LLM Tier 7", len(llm_targets))

    if args.dry_run:
        logger.info("[DRY RUN] LLM targets:\n  %s", "\n  ".join(llm_targets))
        return

    # -----------------------------------------------------------------------
    # STEP 4: LLM suggestions
    # -----------------------------------------------------------------------
    df_raw = _read_input(input_path)

    llm_mapper = LLMFieldMapper(
        registry_path=registry_path,
        portfolio_type=args.portfolio_type,
        aliases_dir=aliases_dir,
        api_key=os.environ.get("ANTHROPIC_API_KEY", "") if args.api_key is None else args.api_key,
    )

    suggestions = llm_mapper.suggest_mappings(llm_targets, df_raw, det_report)
    logger.info("LLM returned %d suggestion(s).", len(suggestions))

    # -----------------------------------------------------------------------
    # STEP 5: Auto-approve high-confidence suggestions
    # -----------------------------------------------------------------------
    if auto_approve_threshold is not None:
        for sugg in suggestions:
            if (
                sugg.status == "pending"
                and sugg.suggested_field
                and sugg.confidence >= auto_approve_threshold
            ):
                sugg.status = "confirmed"
                sugg.confirmed_field = sugg.suggested_field
                sugg.reviewer_note = f"auto-approved (conf={sugg.confidence:.2f} >= {auto_approve_threshold})"
                logger.info(
                    "Auto-approved: '%s' → '%s' (conf=%.2f)",
                    sugg.raw_header, sugg.suggested_field, sugg.confidence,
                )

    # -----------------------------------------------------------------------
    # STEP 6: Human review
    # -----------------------------------------------------------------------
    canonical_fields = _load_canonical_fields(registry_path, args.portfolio_type)
    reviewer = HumanReviewSession(canonical_fields=canonical_fields)

    mode = args.mode or "cli"
    pending = [s for s in suggestions if s.status == "pending"]
    if pending:
        logger.info("STEP 6: Human review of %d suggestion(s) (mode=%s)", len(pending), mode)
        if mode == "streamlit":
            suggestions = reviewer.review_streamlit(suggestions)
        else:
            suggestions = reviewer.review_cli(suggestions)
    else:
        logger.info("STEP 6: All suggestions already resolved (auto-approve) — skipping review.")

    # -----------------------------------------------------------------------
    # STEP 7: Persist confirmed aliases
    # -----------------------------------------------------------------------
    confirmed = [s for s in suggestions if s.status in ("confirmed", "remapped")]
    learner = AliasLearner()
    aliases_added = learner.persist_confirmed(confirmed, aliases_dir, session_id)
    logger.info("STEP 7: %d alias(es) persisted.", aliases_added)

    # -----------------------------------------------------------------------
    # STEP 8: Re-run deterministic pass with augmented aliases
    # -----------------------------------------------------------------------
    if aliases_added > 0:
        logger.info("STEP 8: Re-running deterministic pass with new aliases.")
        second_pass = _run_deterministic_pass(
            input_path, args.portfolio_type, registry_path, aliases_dir, output_dir
        )
        logger.info("Second pass canonical output: %s", second_pass["canonical_csv"])
    else:
        logger.info("STEP 8: No new aliases — skipping second pass.")

    # -----------------------------------------------------------------------
    # STEP 9: Governance artifact
    # -----------------------------------------------------------------------
    gov_logger = GovernanceLogger(governance_dir)
    artifact_path = gov_logger.write_session(
        session_id=session_id,
        input_file=str(input_path),
        portfolio_type=args.portfolio_type,
        deterministic_stats=det_stats,
        suggestions=suggestions,
        aliases_persisted=aliases_added,
    )
    logger.info("STEP 9: Governance artifact: %s", artifact_path)
    logger.info("=== Session %s complete ===", session_id)


def _write_governance_skip(
    session_id: str,
    input_file: str,
    portfolio_type: str,
    det_stats: dict,
    governance_dir: Path,
) -> None:
    gov_logger = GovernanceLogger(governance_dir)
    gov_logger.write_session(
        session_id=session_id,
        input_file=input_file,
        portfolio_type=portfolio_type,
        deterministic_stats=det_stats,
        suggestions=[],
        aliases_persisted=0,
        extra={"note": "All headers mapped deterministically — LLM not invoked."},
    )


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Agentic Field Mapper — LLM Tier 7 orchestrator"
    )
    p.add_argument("--input", required=True, help="Input CSV/XLSX loan tape")
    p.add_argument(
        "--portfolio-type", default="equity_release",
        help="Portfolio type (equity_release | sme | cre | rre)"
    )
    p.add_argument(
        "--registry",
        default="config/system/fields_registry.yaml",
        help="Path to canonical field registry YAML",
    )
    p.add_argument(
        "--aliases-dir",
        default="engine/gate_1_alignment/aliases",
        help="Directory containing aliases_*.yaml files",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to config_agent.yaml (optional)",
    )
    p.add_argument(
        "--mode",
        choices=["cli", "streamlit", "batch"],
        default="cli",
        help="Human review mode",
    )
    p.add_argument(
        "--auto-approve-above",
        type=float,
        default=None,
        help="Auto-approve LLM suggestions above this confidence (requires --enable-auto-approve)",
    )
    p.add_argument(
        "--enable-auto-approve",
        action="store_true",
        default=False,
        help="Enable auto-approve for high-confidence suggestions (off by default for regulatory safety)",
    )
    p.add_argument(
        "--review-threshold",
        type=float,
        default=None,
        help="Confidence below which headers are sent to LLM (default: 0.92)",
    )
    p.add_argument(
        "--output-dir",
        default="out",
        help="Output directory for canonical CSV and governance artifacts",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would happen without calling LLM or persisting anything",
    )
    p.add_argument(
        "--max-llm-calls",
        type=int,
        default=None,
        help="Maximum LLM API calls per session (budget cap)",
    )
    p.add_argument(
        "--skip-llm",
        action="store_true",
        default=False,
        help="Run deterministic-only pass; emit unmapped report for manual review",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (falls back to ANTHROPIC_API_KEY env var)",
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)
