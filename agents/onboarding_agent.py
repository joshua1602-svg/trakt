"""
agents/onboarding_agent.py

Onboarding Agent v1 — ESMA securitisation workflow Gate 1.

Orchestrates:
  1. Config bootstrapping (client/asset/regime config merge)
  2. Deterministic semantic mapping (Tiers 1-6 via semantic_alignment.py)
  3. LLM-assisted mapping for unresolved/low-confidence fields (Tier 7)
  4. Enum mapping (deterministic + optional LLM)
  5. MappingReviewItem / EnumReviewItem generation
  6. Blocker question generation
  7. Status determination and narrative summary
  8. File-based audit output

Design constraints:
  - LLM never mutates financial data; only assists with semantic field name
    suggestions, enum suggestions, and narrative wording.
  - Full loan tape is NEVER sent to the LLM.
  - Deterministic code applies all actual mappings.
  - Result is useful even with llm_enabled=False.
  - All outputs are file-based for auditability.

Usage:
  from agents.onboarding_agent import run_onboarding_agent
  result = run_onboarding_agent(
      raw_tape_path="data/lender_tape.csv",
      client_id="ERE",
      schema_registry_path="config/system/fields_registry.yaml",
      aliases_dir="config/system",
      output_dir="out/onboarding/ERE_2025_10",
  )

CLI:
  python -m agents.onboarding_agent \\
    --raw-tape data/lender_tape.csv \\
    --client-id ERE \\
    --output-dir out/onboarding/ERE_2025_10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agents.config_bootstrap_agent import ConfigBootstrapAgent
from agents.onboarding_schemas import (
    ConfigBootstrapResult,
    EnumReviewItem,
    MappingReviewItem,
    OnboardingResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Project path resolution
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENGINE_ROOT = _PROJECT_ROOT / "engine"
_SEMANTIC_ALIGNMENT = _ENGINE_ROOT / "gate_1_alignment" / "semantic_alignment.py"
_AGENT_ORCHESTRATOR = _ENGINE_ROOT / "gate_1_alignment" / "agent_orchestrator.py"
_DEFAULT_REGISTRY = _PROJECT_ROOT / "config" / "system" / "fields_registry.yaml"
_DEFAULT_ALIASES_DIR = _PROJECT_ROOT / "engine" / "gate_1_alignment" / "aliases"
_DEFAULT_ENUM_MAPPING = _PROJECT_ROOT / "config" / "system" / "enum_mapping.yaml"
_DEFAULT_AGENT_CONFIG = _PROJECT_ROOT / "config" / "system" / "config_agent.yaml"
_ONBOARDING_CONFIG = _PROJECT_ROOT / "config" / "system" / "onboarding_agent.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_onboarding_config() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "confidence_threshold_auto": 0.92,
        "confidence_threshold_review": 0.75,
        "llm": {
            "provider": "anthropic",
            "questionnaire_model": "claude-haiku-4-5-20251001",
            "mapping_model": "claude-haiku-4-5-20251001",
            "escalation_model": "claude-sonnet-4-6",
            "allow_frontier_escalation": False,
        },
        "max_sample_values_per_field": 5,
        "max_fields_sent_to_llm": 20,
        "never_send_full_tape": True,
    }
    if _ONBOARDING_CONFIG.exists():
        try:
            raw = yaml.safe_load(_ONBOARDING_CONFIG.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict):
                # shallow merge top-level; deep merge llm section
                for k, v in raw.items():
                    if k == "llm" and isinstance(v, dict) and isinstance(defaults.get("llm"), dict):
                        defaults["llm"].update(v)
                    else:
                        defaults[k] = v
        except Exception as exc:
            logger.warning("Could not load onboarding_agent.yaml: %s", exc)
    return defaults


# ---------------------------------------------------------------------------
# Tape reading
# ---------------------------------------------------------------------------

def _read_tape_headers(tape_path: Path) -> List[str]:
    """Read only the header row — no financial data."""
    if tape_path.suffix.lower() in (".xlsx", ".xls"):
        try:
            import pandas as pd
            df = pd.read_excel(tape_path, nrows=0)
            return list(df.columns)
        except Exception as exc:
            logger.warning("Could not read xlsx headers from %s: %s", tape_path, exc)
            return []

    # CSV: stdlib is sufficient and avoids pandas dependency
    try:
        import csv as _csv
        with tape_path.open(newline="", encoding="utf-8") as f:
            reader = _csv.reader(f)
            headers = next(reader, [])
        return headers
    except Exception as exc:
        logger.warning("Could not read tape headers from %s: %s", tape_path, exc)
        return []


# ---------------------------------------------------------------------------
# Semantic alignment (deterministic pass, Tiers 1-6)
# ---------------------------------------------------------------------------

def _run_semantic_alignment(
    tape_path: Path,
    portfolio_type: str,
    registry_path: Path,
    aliases_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Invoke semantic_alignment.py as a subprocess (matches existing pipeline pattern).
    Returns paths to outputs.
    """
    if not _SEMANTIC_ALIGNMENT.exists():
        raise FileNotFoundError(f"semantic_alignment.py not found at {_SEMANTIC_ALIGNMENT}")

    cmd = [
        sys.executable, str(_SEMANTIC_ALIGNMENT),
        "--input", str(tape_path.resolve()),
        "--portfolio-type", portfolio_type,
        "--registry", str(registry_path.resolve()),
        "--aliases-dir", str(aliases_dir.resolve()),
        "--output-dir", str(output_dir.resolve()),
        "--output-schema", "active",
    ]
    logger.info("[SemanticAlignment] Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            cwd=str(_PROJECT_ROOT))
    if result.returncode != 0:
        logger.error("semantic_alignment.py stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"semantic_alignment.py exited with code {result.returncode}. "
            f"stderr: {result.stderr[:500]}"
        )

    stem = tape_path.stem
    return {
        "canonical_csv": output_dir / f"{stem}_canonical_full.csv",
        "mapping_report_csv": output_dir / f"{stem}_mapping_report.csv",
        "unmapped_csv": output_dir / f"{stem}_unmapped_headers.csv",
        "mapping_report_json": output_dir / f"{stem}_header_mapping_report.json",
    }


def _load_mapping_report(mapping_report_csv: Path) -> List[Dict[str, Any]]:
    """Load the mapping report CSV produced by semantic_alignment.py."""
    if not mapping_report_csv.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_csv(mapping_report_csv, dtype=str).fillna("")
        return df.to_dict(orient="records")
    except Exception:
        pass   # fall through to stdlib fallback

    # stdlib fallback (no pandas or broken pandas)
    try:
        import csv as _csv
        rows = []
        with mapping_report_csv.open(newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                rows.append({k: (v or "") for k, v in row.items()})
        return rows
    except Exception as exc:
        logger.warning("Could not load mapping report (stdlib): %s", exc)
        return []


# ---------------------------------------------------------------------------
# Mandatory field detection
# ---------------------------------------------------------------------------

def _get_mandatory_fields_for_regime(
    registry_path: Path,
    regime: str,
    portfolio_type: str,
) -> set:
    """
    Return the set of canonical field names that are Mandatory for the given regime.
    """
    mandatory: set = set()
    try:
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        for field_name, meta in (data.get("fields") or {}).items():
            if not isinstance(meta, dict):
                continue
            rm = meta.get("regime_mapping") or {}
            regime_info = rm.get(regime) or {}
            priority = (regime_info.get("priority") or "").strip()
            if priority == "Mandatory":
                # Check portfolio_type applicability
                fpt = str(meta.get("portfolio_type", "common")).strip().lower()
                if fpt in ("common", portfolio_type.lower(), ""):
                    mandatory.add(field_name)
    except Exception as exc:
        logger.warning("Could not load mandatory fields from registry: %s", exc)
    return mandatory


# ---------------------------------------------------------------------------
# Mapping review item conversion
# ---------------------------------------------------------------------------

_DETERMINISTIC_METHODS = {"exact", "normalized", "alias"}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def _build_mapping_review_items(
    mapping_report: List[Dict[str, Any]],
    mandatory_fields: set,
    confidence_threshold_review: float,
) -> List[MappingReviewItem]:
    """
    Convert the raw mapping report rows into typed MappingReviewItem objects.
    """
    items: List[MappingReviewItem] = []
    for row in mapping_report:
        raw = str(row.get("raw_header", "")).strip()
        canonical = row.get("canonical_field") or row.get("mapped_field") or None
        if canonical == "":
            canonical = None
        method = str(row.get("mapping_method", "unmapped")).strip()
        conf = _safe_float(row.get("confidence", 0.0))

        required = bool(canonical in mandatory_fields) if canonical else False
        requires_review = (
            method == "unmapped"
            or (method not in _DETERMINISTIC_METHODS and conf < confidence_threshold_review)
        )
        blocker = (method == "unmapped" and raw in mandatory_fields) or (
            requires_review and required
        )

        # Sample values — already in report if semantic_alignment emits them
        samples = []
        sv = row.get("sample_values") or row.get("samples") or ""
        if sv and sv != "":
            try:
                samples = json.loads(sv) if sv.startswith("[") else [str(v) for v in sv.split("|") if v]
            except Exception:
                samples = [str(sv)[:60]]

        reason = ""
        if method == "unmapped":
            reason = "No deterministic or alias mapping found."
        elif requires_review:
            reason = f"Low confidence ({conf:.2f}) mapping via {method}; requires review."
        elif method in _DETERMINISTIC_METHODS:
            reason = ""   # clean — no review needed

        items.append(MappingReviewItem(
            raw_field=raw,
            suggested_canonical_field=canonical,
            mapping_source=method,
            confidence=conf,
            required_for_regime=required,
            requires_review=requires_review,
            blocker=blocker,
            reason=reason,
            sample_values=samples,
        ))
    return items


# ---------------------------------------------------------------------------
# LLM mapping for unresolved fields (Tier 7 wrapper)
# ---------------------------------------------------------------------------

def _run_llm_mapping(
    tape_path: Path,
    portfolio_type: str,
    registry_path: Path,
    aliases_dir: Path,
    output_dir: Path,
    review_items: List[MappingReviewItem],
    confidence_threshold_review: float,
    llm_model: str,
    api_key: str,
) -> List[MappingReviewItem]:
    """
    Call LLM Tier 7 ONLY for unresolved/low-confidence fields.
    Updates MappingReviewItem objects in-place where LLM returns suggestions.
    Uses batch mode — no interactive review at this stage.
    """
    # Identify headers that need LLM
    llm_targets = [
        item.raw_field for item in review_items
        if item.mapping_source == "unmapped"
        or (item.mapping_source not in _DETERMINISTIC_METHODS
            and item.confidence < confidence_threshold_review)
    ]
    if not llm_targets:
        logger.info("[LLM] All fields resolved deterministically — skipping LLM.")
        return review_items

    logger.info("[LLM] Sending %d field(s) to LLM Tier 7.", len(llm_targets))

    # Use agent_orchestrator in batch mode (writes JSON suggestions file; no interactive review)
    cmd = [
        sys.executable, str(_AGENT_ORCHESTRATOR),
        "--input", str(tape_path),
        "--portfolio-type", portfolio_type,
        "--registry", str(registry_path),
        "--aliases-dir", str(aliases_dir),
        "--output-dir", str(output_dir),
        "--mode", "batch",
        "--review-threshold", str(confidence_threshold_review),
        "--api-key", api_key,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=120)
        if result.returncode != 0:
            logger.warning("[LLM] agent_orchestrator batch mode failed: %s", result.stderr[:300])
            return review_items
    except Exception as exc:
        logger.warning("[LLM] agent_orchestrator call failed: %s", exc)
        return review_items

    # Load LLM suggestions JSON
    stem = tape_path.stem
    suggestions_json = output_dir / f"{stem}_llm_suggestions.json"
    if not suggestions_json.exists():
        logger.info("[LLM] No suggestions file found (may mean all resolved deterministically).")
        return review_items

    try:
        suggestions = json.loads(suggestions_json.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[LLM] Could not load suggestions JSON: %s", exc)
        return review_items

    # Merge LLM suggestions back into review items
    sugg_by_header = {s.get("raw_header", ""): s for s in suggestions if isinstance(s, dict)}
    for item in review_items:
        sugg = sugg_by_header.get(item.raw_field)
        if not sugg:
            continue
        suggested = sugg.get("suggested_field")
        conf = _safe_float(sugg.get("confidence", 0.0))
        reasoning = str(sugg.get("reasoning", ""))
        if suggested and conf > 0.0:
            item.suggested_canonical_field = suggested
            item.confidence = conf
            item.mapping_source = "llm"
            item.requires_review = True   # LLM suggestions always require human review
            item.reason = reasoning or f"LLM suggestion (conf={conf:.2f}); requires review."
            item.sample_values = sugg.get("sample_values") or item.sample_values

    return review_items


# ---------------------------------------------------------------------------
# Enum mapping (thin wrapper around existing enum agent)
# ---------------------------------------------------------------------------

def _run_enum_mapping(
    canonical_csv: Path,
    registry_path: Path,
    enum_mapping_path: Path,
    regime: str,
    output_dir: Path,
) -> List[EnumReviewItem]:
    """
    Load enum fields from registry, call resolve_enums_for_field for each,
    and return EnumReviewItem objects for values requiring review.
    No interactive review — outputs enum_review.json for the control room.
    """
    enum_review_items: List[EnumReviewItem] = []

    try:
        import pandas as pd
        df = pd.read_csv(canonical_csv, low_memory=False)
    except Exception as exc:
        logger.warning("[EnumMapping] Could not load canonical CSV %s: %s", canonical_csv, exc)
        return enum_review_items

    # Load registry to find enum fields
    try:
        registry_data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("[EnumMapping] Could not load registry: %s", exc)
        return enum_review_items

    # Load enum mapping for allowed values
    enum_mapping: Dict[str, Any] = {}
    if enum_mapping_path.exists():
        try:
            enum_mapping = yaml.safe_load(enum_mapping_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("[EnumMapping] Could not load enum_mapping.yaml: %s", exc)

    # Build list of (canonical_field_name, allowed_values) for fields present in df
    regime_key = regime  # e.g. "ESMA_Annex2"
    fields_to_check: Dict[str, List[str]] = {}
    for field_name, meta in (registry_data.get("fields") or {}).items():
        if not isinstance(meta, dict):
            continue
        if field_name not in df.columns:
            continue
        allowed_key = meta.get("allowed_values")
        if not allowed_key:
            continue
        # Look up allowed values from enum_mapping
        regime_enums = enum_mapping.get(regime_key) or {}
        enum_block = regime_enums.get(allowed_key) or {}
        if not enum_block:
            # Try top-level
            enum_block = enum_mapping.get(allowed_key) or {}
        if isinstance(enum_block, dict):
            allowed = list(enum_block.keys())
        elif isinstance(enum_block, list):
            allowed = enum_block
        else:
            continue
        if allowed:
            fields_to_check[field_name] = allowed

    # Use the existing enum resolution engine (deterministic only; no LLM in v1)
    try:
        sys.path.insert(0, str(_ENGINE_ROOT / "enum_agent"))
        from enum_mapping_agent import resolve_enums_for_field, EnumSuggestion  # type: ignore
    except ImportError as exc:
        logger.warning("[EnumMapping] Could not import enum_mapping_agent: %s — skipping.", exc)
        return enum_review_items

    for field_name, allowed_values in fields_to_check.items():
        try:
            _mapped, report, _candidates, _meta = resolve_enums_for_field(
                field_name=field_name,
                series=df[field_name],
                allowed_values=allowed_values,
                namespace="global",
                regime=regime,
            )
        except Exception as exc:
            logger.warning("[EnumMapping] resolve_enums_for_field failed for %s: %s", field_name, exc)
            continue

        for sugg in report:
            if not isinstance(sugg, EnumSuggestion):
                continue
            if sugg.status in ("exact", "synonym"):
                continue   # resolved — no review needed
            is_mandatory = False
            regime_info = (registry_data["fields"].get(field_name) or {}).get("regime_mapping", {})
            if isinstance(regime_info, dict):
                ri = regime_info.get(regime) or {}
                is_mandatory = ri.get("priority", "").strip() == "Mandatory"

            enum_review_items.append(EnumReviewItem(
                field_name=field_name,
                raw_value=sugg.raw_value,
                suggested_value=sugg.suggested_value,
                mapping_source=sugg.deterministic_method or "unmapped",
                confidence=sugg.confidence,
                requires_review=True,
                blocker=is_mandatory and sugg.suggested_value is None,
                reason=sugg.reasoning or "Enum value not resolved deterministically.",
                sample_count=sugg.count,
            ))

    return enum_review_items


# ---------------------------------------------------------------------------
# Blocker question generation
# ---------------------------------------------------------------------------

def _build_blocker_questions(
    mapping_items: List[MappingReviewItem],
    enum_items: List[EnumReviewItem],
    config_bootstrap: ConfigBootstrapResult,
) -> List[Dict[str, Any]]:
    """Generate structured blocker questions from mapping/enum/config gaps."""
    questions: List[Dict[str, Any]] = []

    # Config blockers
    for mc in config_bootstrap.missing_critical_config:
        questions.append({
            "question_id": f"cfg_{mc['field'].replace('.', '_')}",
            "category": "config",
            "field": mc["field"],
            "question": mc.get("why_needed", f"Provide value for {mc['field']}"),
            "blocking": True,
            "source": "config_bootstrap",
        })

    # Unmapped mandatory field blockers
    for item in mapping_items:
        if item.blocker and item.mapping_source == "unmapped":
            questions.append({
                "question_id": f"map_{item.raw_field.replace(' ', '_')}",
                "category": "field_mapping",
                "field": item.raw_field,
                "question": (
                    f"Field '{item.raw_field}' could not be mapped to any canonical field. "
                    "Please confirm the correct canonical field name."
                ),
                "blocking": True,
                "source": "mapping",
                "sample_values": item.sample_values,
            })

    # Low-confidence mandatory LLM suggestions
    for item in mapping_items:
        if item.blocker and item.mapping_source == "llm":
            questions.append({
                "question_id": f"llm_{item.raw_field.replace(' ', '_')}",
                "category": "field_mapping",
                "field": item.raw_field,
                "question": (
                    f"Field '{item.raw_field}' was tentatively mapped to "
                    f"'{item.suggested_canonical_field}' by the LLM (confidence {item.confidence:.0%}). "
                    "Please confirm or correct this mapping."
                ),
                "blocking": True,
                "source": "llm_mapping",
                "suggested_canonical_field": item.suggested_canonical_field,
                "confidence": item.confidence,
            })

    # Enum blockers
    for item in enum_items:
        if item.blocker:
            questions.append({
                "question_id": f"enum_{item.field_name}_{item.raw_value[:20].replace(' ', '_')}",
                "category": "enum_mapping",
                "field": item.field_name,
                "raw_value": item.raw_value,
                "question": (
                    f"Enum value '{item.raw_value}' in field '{item.field_name}' "
                    "could not be mapped to a canonical allowed value. "
                    "Please provide the correct canonical value."
                ),
                "blocking": True,
                "source": "enum_mapping",
                "sample_count": item.sample_count,
            })

    return questions


# ---------------------------------------------------------------------------
# Narrative summary
# ---------------------------------------------------------------------------

_NARRATIVE_TEMPLATE = (
    "Onboarding {status_phrase}. "
    "The agent detected a {asset_class} portfolio targeting {regime} "
    "with {asset_conf:.0%} confidence. "
    "{mapped_count} of {total_count} fields were mapped automatically "
    "({det_count} deterministic, {llm_count} LLM-assisted). "
    "{review_count} fields require review. "
    "{unmapped_mandatory} mandatory fields are unmapped. "
    "Enum mapping: {enum_success:.0%} successful ({enum_review} values flagged for review). "
    "{proceed_phrase}"
)

_STATUS_PHRASE = {
    "ready_for_validation": "completed successfully",
    "review_required": "completed with review required",
    "blocked": "blocked",
    "failed": "failed",
}

_PROCEED_PHRASE = {
    "ready_for_validation": "The run is ready to proceed to the Validation Agent.",
    "review_required": (
        "The run cannot proceed to validation until flagged items are reviewed and approved."
    ),
    "blocked": (
        "The run is blocked. Mandatory fields are unmapped or critical config is missing."
    ),
    "failed": "The run failed. See errors for details.",
}


def _build_narrative(result: OnboardingResult) -> str:
    """Generate a deterministic narrative from result fields."""
    cb = result.config_bootstrap
    asset_class = (cb.detected_asset_class if cb else "") or "unknown"
    regime = (cb.selected_regime if cb else "") or "unknown"
    asset_conf = (cb.detected_asset_confidence if cb else 0.0)

    return _NARRATIVE_TEMPLATE.format(
        status_phrase=_STATUS_PHRASE.get(result.status, result.status),
        asset_class=asset_class,
        regime=regime,
        asset_conf=asset_conf,
        mapped_count=result.mapped_fields_count,
        total_count=result.total_input_fields,
        det_count=result.deterministic_mapped_count,
        llm_count=result.llm_suggested_count,
        review_count=result.review_fields_count,
        unmapped_mandatory=result.unmapped_mandatory_count,
        enum_success=result.enum_success_rate,
        enum_review=result.enum_review_count,
        proceed_phrase=_PROCEED_PHRASE.get(result.status, ""),
    )


def _improve_narrative_llm(
    narrative: str,
    result: OnboardingResult,
    llm_model: str,
    api_key: str,
) -> str:
    """Optionally improve the narrative via LLM. Falls back to deterministic on error."""
    try:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=api_key)
        system = (
            "You are a data operations assistant. Rewrite the following onboarding summary "
            "to be clear, professional, and concise for a regulated financial data analyst. "
            "Do not change any numbers, field names, or decisions — only improve the wording. "
            "Return only the improved narrative text, no JSON or markdown."
        )
        msg = client.messages.create(
            model=llm_model,
            max_tokens=300,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": narrative}],
        )
        improved = msg.content[0].text.strip()
        return improved if improved else narrative
    except Exception as exc:
        logger.debug("LLM narrative improvement skipped: %s", exc)
        return narrative


# ---------------------------------------------------------------------------
# Status determination
# ---------------------------------------------------------------------------

def _determine_status(
    config_bootstrap: ConfigBootstrapResult,
    mapping_items: List[MappingReviewItem],
    enum_items: List[EnumReviewItem],
) -> str:
    """
    Determine overall onboarding status from config, mapping and enum results.
    """
    if config_bootstrap.status == "blocked":
        return "blocked"
    if any(item.blocker for item in mapping_items):
        return "blocked"
    if any(item.blocker for item in enum_items):
        return "blocked"
    if (
        config_bootstrap.status == "review_required"
        or any(item.requires_review for item in mapping_items)
        or any(item.requires_review for item in enum_items)
    ):
        return "review_required"
    return "ready_for_validation"


# ---------------------------------------------------------------------------
# Primary public function
# ---------------------------------------------------------------------------

def run_onboarding_agent(
    raw_tape_path: str,
    run_id: Optional[str] = None,
    client_id: Optional[str] = None,
    client_config_path: Optional[str] = None,
    asset_config_path: Optional[str] = None,  # reserved; used by bootstrap
    regime_config_path: Optional[str] = None,  # reserved; not yet used
    schema_registry_path: Optional[str] = None,
    aliases_dir: Optional[str] = None,
    enum_mapping_path: Optional[str] = None,
    output_dir: str = "out/onboarding",
    questionnaire_answers_path: Optional[str] = None,
    llm_enabled: bool = True,
    cheap_llm_model: Optional[str] = None,
    frontier_llm_model: Optional[str] = None,
    confidence_threshold_auto: float = 0.92,
    confidence_threshold_review: float = 0.75,
    allow_frontier_escalation: bool = False,
) -> OnboardingResult:
    """
    Run the full Onboarding Agent pipeline and return an OnboardingResult.

    The proceed_to_validation field on the result is the downstream gate signal.
    All output files are written to output_dir / run_id.
    """

    # --- Setup ---
    run_id = run_id or f"ob_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    tape_path = Path(raw_tape_path)
    run_output_dir = Path(output_dir) / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=== Onboarding Agent run: %s ===", run_id)
    logger.info("Tape: %s", tape_path)

    cfg = _load_onboarding_config()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    llm_cfg = cfg.get("llm", {})
    mapping_model = cheap_llm_model or llm_cfg.get("mapping_model", "claude-haiku-4-5-20251001")
    narrative_model = cheap_llm_model or llm_cfg.get("questionnaire_model", "claude-haiku-4-5-20251001")

    registry_path = Path(schema_registry_path) if schema_registry_path else _DEFAULT_REGISTRY
    aliases_path = Path(aliases_dir) if aliases_dir else _DEFAULT_ALIASES_DIR
    enum_path = Path(enum_mapping_path) if enum_mapping_path else _DEFAULT_ENUM_MAPPING

    result = OnboardingResult(run_id=run_id)
    result.raw_tape_path = str(tape_path.resolve())
    result.schema_registry_path = str(registry_path)
    result.aliases_dir = str(aliases_path)
    result.enum_mapping_path = str(enum_path)
    errors: List[str] = []

    # Write run_info.json so the workbench can re-run without extra args
    run_info = {
        "run_id": run_id,
        "raw_tape_path": str(tape_path.resolve()),
        "client_config_path": client_config_path or "",
        "schema_registry_path": str(registry_path),
        "aliases_dir": str(aliases_path),
        "enum_mapping_path": str(enum_path),
        "output_dir": str(Path(output_dir)),
        "llm_enabled": llm_enabled,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )

    # -----------------------------------------------------------------------
    # STEP 1: Config Bootstrap
    # -----------------------------------------------------------------------
    logger.info("[Step 1] Config Bootstrap")
    try:
        bootstrap = ConfigBootstrapAgent(
            raw_tape_path=tape_path,
            run_id=run_id,
            output_dir=run_output_dir,
            client_id=client_id,
            existing_client_config_path=client_config_path,
            questionnaire_answers_path=questionnaire_answers_path,
            llm_model_questionnaire=narrative_model,
            llm_enabled=llm_enabled,
            api_key=api_key,
        ).run()
    except Exception as exc:
        logger.error("[Step 1] Config bootstrap failed: %s", exc)
        result.status = "failed"
        result.errors.append(f"Config bootstrap failed: {exc}")
        result.narrative_summary = f"Onboarding failed at config bootstrap: {exc}"
        return _finalise(result, run_output_dir)

    result.config_bootstrap = bootstrap

    if bootstrap.status == "blocked":
        logger.warning("[Step 1] Config blocked — cannot proceed to mapping.")
        result.status = "blocked"
        result.blocker_questions = bootstrap.config_questions
        result.approved_config_path = bootstrap.draft_config_path
        result.narrative_summary = _build_narrative(result)
        return _finalise(result, run_output_dir)

    portfolio_type = bootstrap.detected_asset_class or "equity_release"
    regime = bootstrap.selected_regime or "ESMA_Annex2"
    approved_config_path = bootstrap.approved_config_path or bootstrap.draft_config_path
    result.approved_config_path = approved_config_path

    # -----------------------------------------------------------------------
    # STEP 2: Semantic alignment (deterministic Tiers 1-6)
    # -----------------------------------------------------------------------
    logger.info("[Step 2] Semantic alignment — portfolio_type=%s", portfolio_type)
    try:
        alignment_paths = _run_semantic_alignment(
            tape_path=tape_path,
            portfolio_type=portfolio_type,
            registry_path=registry_path,
            aliases_dir=aliases_path,
            output_dir=run_output_dir,
        )
    except Exception as exc:
        logger.error("[Step 2] Semantic alignment failed: %s", exc)
        result.status = "failed"
        result.errors.append(f"Semantic alignment failed: {exc}")
        result.narrative_summary = _build_narrative(result)
        return _finalise(result, run_output_dir)

    mapping_report = _load_mapping_report(alignment_paths["mapping_report_csv"])
    result.mapping_report_path = str(alignment_paths["mapping_report_csv"])
    result.canonical_draft_path = str(alignment_paths["canonical_csv"])

    headers = _read_tape_headers(tape_path)
    result.total_input_fields = len(headers) if headers else len(mapping_report)

    # -----------------------------------------------------------------------
    # STEP 3: Get mandatory fields for regime
    # -----------------------------------------------------------------------
    mandatory_fields = _get_mandatory_fields_for_regime(registry_path, regime, portfolio_type)

    # -----------------------------------------------------------------------
    # STEP 4: Build initial mapping review items
    # -----------------------------------------------------------------------
    mapping_items = _build_mapping_review_items(
        mapping_report, mandatory_fields, confidence_threshold_review
    )

    # -----------------------------------------------------------------------
    # STEP 5: LLM mapping for unresolved/low-confidence fields
    # -----------------------------------------------------------------------
    gov_artifact_path = ""
    if llm_enabled and api_key:
        logger.info("[Step 5] LLM-assisted mapping")
        try:
            mapping_items = _run_llm_mapping(
                tape_path=tape_path,
                portfolio_type=portfolio_type,
                registry_path=registry_path,
                aliases_dir=aliases_path,
                output_dir=run_output_dir,
                review_items=mapping_items,
                confidence_threshold_review=confidence_threshold_review,
                llm_model=mapping_model,
                api_key=api_key,
            )
            # Look for governance artifact
            gov_dir = run_output_dir / "governance" / "agent_sessions"
            if gov_dir.exists():
                gov_files = sorted(gov_dir.glob("agent_*.json"))
                if gov_files:
                    gov_artifact_path = str(gov_files[-1])
        except Exception as exc:
            logger.warning("[Step 5] LLM mapping failed (non-blocking): %s", exc)
            errors.append(f"LLM mapping failed: {exc}")
    else:
        logger.info("[Step 5] LLM disabled or no API key — skipping LLM mapping.")

    result.governance_artifact_path = gov_artifact_path
    result.mapping_review_items = mapping_items

    # -----------------------------------------------------------------------
    # STEP 6: Mapping statistics
    # -----------------------------------------------------------------------
    result.deterministic_mapped_count = sum(
        1 for i in mapping_items
        if i.mapping_source in _DETERMINISTIC_METHODS
    )
    result.llm_suggested_count = sum(
        1 for i in mapping_items if i.mapping_source == "llm"
    )
    result.mapped_fields_count = sum(
        1 for i in mapping_items if i.suggested_canonical_field is not None
    )
    result.review_fields_count = sum(1 for i in mapping_items if i.requires_review)
    result.unmapped_fields_count = sum(1 for i in mapping_items if i.mapping_source == "unmapped")
    result.unmapped_mandatory_count = sum(
        1 for i in mapping_items
        if i.mapping_source == "unmapped" and i.required_for_regime
    )

    # -----------------------------------------------------------------------
    # STEP 7: Enum mapping
    # -----------------------------------------------------------------------
    enum_items: List[EnumReviewItem] = []
    canonical_csv = alignment_paths["canonical_csv"]
    if canonical_csv.exists():
        logger.info("[Step 7] Enum mapping")
        try:
            enum_items = _run_enum_mapping(
                canonical_csv=canonical_csv,
                registry_path=registry_path,
                enum_mapping_path=enum_path,
                regime=regime,
                output_dir=run_output_dir,
            )
        except Exception as exc:
            logger.warning("[Step 7] Enum mapping failed (non-blocking): %s", exc)
            errors.append(f"Enum mapping failed: {exc}")
    else:
        logger.info("[Step 7] No canonical CSV yet — skipping enum mapping.")

    result.enum_review_items = enum_items
    result.enum_fields_total = len(enum_items)          # flagged values (exact/synonym already resolved)
    # Items with a suggestion have a resolution candidate; truly unsolved have suggested_value=None
    _has_suggestion = sum(1 for e in enum_items if e.suggested_value)
    result.enum_mapped_count = _has_suggestion
    result.enum_review_count = len(enum_items) - _has_suggestion  # unsolvable
    result.enum_success_rate = (
        _has_suggestion / len(enum_items) if enum_items else 1.0
    )

    # -----------------------------------------------------------------------
    # STEP 8: Blocker questions
    # -----------------------------------------------------------------------
    all_blockers = _build_blocker_questions(mapping_items, enum_items, bootstrap)
    non_blockers = [q for q in bootstrap.config_questions if not q.get("blocking")]
    result.blocker_questions = all_blockers
    result.user_questions = non_blockers

    # -----------------------------------------------------------------------
    # STEP 9: Status determination
    # -----------------------------------------------------------------------
    result.status = _determine_status(bootstrap, mapping_items, enum_items)
    result.proceed_to_validation = (result.status == "ready_for_validation")

    # -----------------------------------------------------------------------
    # STEP 10: Narrative summary
    # -----------------------------------------------------------------------
    narrative = _build_narrative(result)
    if llm_enabled and api_key:
        narrative = _improve_narrative_llm(narrative, result, narrative_model, api_key)
    result.narrative_summary = narrative

    result.errors = errors

    # -----------------------------------------------------------------------
    # STEP 11: Write enum review JSON
    # -----------------------------------------------------------------------
    enum_report_path = run_output_dir / f"{run_id}_enum_review.json"
    enum_report_path.write_text(
        json.dumps([i.to_dict() for i in enum_items], indent=2),
        encoding="utf-8",
    )
    result.enum_report_path = str(enum_report_path)

    # -----------------------------------------------------------------------
    # STEP 12: Write mapping review JSON
    # -----------------------------------------------------------------------
    mapping_review_path = run_output_dir / f"{run_id}_mapping_review.json"
    mapping_review_path.write_text(
        json.dumps([i.to_dict() for i in mapping_items], indent=2),
        encoding="utf-8",
    )

    logger.info(
        "=== Onboarding Agent complete: %s | proceed=%s | "
        "mapped=%d/%d | review=%d | blockers=%d ===",
        result.status,
        result.proceed_to_validation,
        result.mapped_fields_count,
        result.total_input_fields,
        result.review_fields_count,
        len(all_blockers),
    )

    return _finalise(result, run_output_dir)


def _finalise(result: OnboardingResult, run_output_dir: Path) -> OnboardingResult:
    """Write onboarding_result.json and set its path on the result."""
    result_path = run_output_dir / "onboarding_result.json"
    result.onboarding_result_path = str(result_path)
    result.to_json(result_path)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Onboarding Agent v1 — ESMA securitisation Gate 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--raw-tape", required=True, metavar="PATH", help="Input raw loan tape CSV/XLSX")
    p.add_argument("--run-id", default=None, help="Run ID (auto-generated if omitted)")
    p.add_argument("--client-id", default=None, help="Client ID for config lookup")
    p.add_argument("--client-config", default=None, metavar="PATH", help="Existing client config YAML")
    p.add_argument(
        "--schema-registry",
        default=str(_DEFAULT_REGISTRY),
        metavar="PATH",
        help="Canonical field registry YAML",
    )
    p.add_argument(
        "--aliases-dir",
        default=str(_DEFAULT_ALIASES_DIR),
        metavar="PATH",
        help="Directory containing aliases_*.yaml files",
    )
    p.add_argument(
        "--enum-mapping",
        default=str(_DEFAULT_ENUM_MAPPING),
        metavar="PATH",
        help="Enum mapping YAML",
    )
    p.add_argument(
        "--output-dir",
        default="out/onboarding",
        metavar="PATH",
        help="Root output directory (run_id subdirectory is created automatically)",
    )
    p.add_argument(
        "--questionnaire-answers",
        default=None,
        metavar="PATH",
        help="JSON file with questionnaire answers from previous review",
    )
    p.add_argument(
        "--llm-enabled",
        default=True,
        type=lambda v: v.lower() not in ("false", "0", "no"),
        help="Enable LLM-assisted mapping (default: true)",
    )
    p.add_argument(
        "--cheap-llm-model",
        default=None,
        help="Claude model for questionnaire wording and mapping (default: haiku)",
    )
    p.add_argument(
        "--confidence-threshold-auto",
        type=float,
        default=0.92,
        help="Confidence above which mappings are auto-accepted without review",
    )
    p.add_argument(
        "--confidence-threshold-review",
        type=float,
        default=0.75,
        help="Confidence below which fields are sent to LLM / flagged for review",
    )
    return p


def _print_summary(result: OnboardingResult) -> None:
    print("\n" + "=" * 60)
    print("ONBOARDING AGENT RESULT")
    print("=" * 60)
    print(f"Status              : {result.status}")
    print(f"Run ID              : {result.run_id}")
    print(f"Proceed to validation: {result.proceed_to_validation}")
    print(f"Total input fields  : {result.total_input_fields}")
    print(f"Mapped fields       : {result.mapped_fields_count}")
    print(f"  Deterministic     : {result.deterministic_mapped_count}")
    print(f"  LLM-suggested     : {result.llm_suggested_count}")
    print(f"Review fields       : {result.review_fields_count}")
    print(f"Unmapped fields     : {result.unmapped_fields_count}")
    print(f"Unmapped mandatory  : {result.unmapped_mandatory_count}")
    print(f"Enum success rate   : {result.enum_success_rate:.1%}")
    print(f"Enum review items   : {result.enum_review_count}")
    print(f"Blocker questions   : {len(result.blocker_questions)}")
    print(f"Onboarding result   : {result.onboarding_result_path}")
    print("-" * 60)
    print("Narrative:")
    print(result.narrative_summary)
    print("=" * 60)


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    res = run_onboarding_agent(
        raw_tape_path=args.raw_tape,
        run_id=args.run_id,
        client_id=args.client_id,
        client_config_path=args.client_config,
        schema_registry_path=args.schema_registry,
        aliases_dir=args.aliases_dir,
        enum_mapping_path=args.enum_mapping,
        output_dir=args.output_dir,
        questionnaire_answers_path=args.questionnaire_answers,
        llm_enabled=args.llm_enabled,
        cheap_llm_model=args.cheap_llm_model,
        confidence_threshold_auto=args.confidence_threshold_auto,
        confidence_threshold_review=args.confidence_threshold_review,
    )

    _print_summary(res)
    sys.exit(0 if res.proceed_to_validation or res.status == "review_required" else 1)
