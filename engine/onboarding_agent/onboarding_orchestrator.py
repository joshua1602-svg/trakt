"""
onboarding_orchestrator.py
=========================

Drives the onboarding workflow end to end and writes the reviewable pack.

Stages (each delegated to its own module):
  classify -> profile -> candidate keys -> mappings -> overlap -> config ->
  gaps -> review pack -> optional handoff

All outputs land under ``output_dir`` with numbered filenames. Nothing here
mutates production config or canonical data; the handoff artefacts are drafts.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from . import config_suggester, file_classifier, file_profiler, gap_analyzer, source_consolidator
from .mapping_proposer import propose_mappings
from .onboarding_models import OnboardingProject
from .review_pack_builder import build_review_pack

# Confidence at/above which a mapping counts as "high confidence" for handoff.
HANDOFF_CONFIDENCE = 0.92


def _write_csv(path: Path, items: List, fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for it in items:
            row = dataclasses.asdict(it) if dataclasses.is_dataclass(it) else dict(it)
            out = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, (list, dict)):
                    v = "; ".join(str(x) for x in v) if isinstance(v, list) else json.dumps(v)
                out[k] = v
            writer.writerow(out)


def _write_json(path: Path, items: List) -> None:
    data = [dataclasses.asdict(it) if dataclasses.is_dataclass(it) else it for it in items]
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _load_structured_dataframes(inventory) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for item in inventory:
        if item.file_type not in ("csv", "xlsx", "xls"):
            continue
        try:
            if item.file_type in ("xlsx", "xls"):
                frames[item.file_path] = pd.read_excel(item.file_path)
            else:
                frames[item.file_path] = pd.read_csv(item.file_path, low_memory=False)
        except Exception:
            continue
    return frames


def run_onboarding(
    input_dir: str,
    client_name: str,
    output_dir: str,
    registry_path: str,
    aliases_dir: str,
    project_id: str = "",
    enable_handoff: bool = True,
) -> OnboardingProject:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project = OnboardingProject(
        project_id=project_id or client_name.lower().replace(" ", "_"),
        client_name=client_name,
        input_dir=str(in_dir),
        output_dir=str(out_dir),
        registry_path=str(registry_path),
        aliases_dir=str(aliases_dir),
    )

    # --- PART 3: classify ---
    inventory = file_classifier.classify_directory(in_dir)
    project.file_inventory = inventory
    project.source_files = [i.file_path for i in inventory]

    dataframes = _load_structured_dataframes(inventory)

    # --- PART 4: profile ---
    profiles = []
    for item in inventory:
        if item.file_type in ("csv", "xlsx", "xls"):
            file_profiles = file_profiler.profile_file(Path(item.file_path))
            profiles.extend(file_profiles)
            # Surface a single reporting date onto the inventory item.
            rep_dates = sorted(
                {p.date_max for p in file_profiles if p.likely_reporting_date and p.date_max}
            )
            if rep_dates:
                item.detected_reporting_date = rep_dates[-1]
    project.field_profiles = profiles

    # --- PART 5a: candidate keys ---
    project.candidate_keys = source_consolidator.detect_candidate_keys(profiles)

    # --- PART 6: mapping candidates ---
    project.mapping_candidates = propose_mappings(
        inventory, dataframes, Path(registry_path), Path(aliases_dir)
    )

    # --- PART 5b: overlap analysis (uses mappings + keys) ---
    project.overlap_analysis = source_consolidator.analyze_overlap(
        inventory, project.mapping_candidates, project.candidate_keys, dataframes
    )

    # --- PART 7: config suggestions ---
    project.config_suggestions = config_suggester.suggest_config(
        client_name, in_dir, inventory, profiles
    )

    # --- PART 8: gap questions ---
    project.gap_questions = gap_analyzer.analyze_gaps(
        inventory, profiles, project.overlap_analysis, project.config_suggestions, dataframes
    )

    # --- review status ---
    blocking = [q for q in project.gap_questions if q.severity == "blocking"]
    project.review_status = "blocked" if blocking else "review_required"

    # --- write artefacts ---
    _write_artifacts(project)

    # --- PART 9: review pack ---
    pack_path = out_dir / "08_onboarding_review_pack.html"
    build_review_pack(project, pack_path)
    project.generated_artifacts.append(str(pack_path))

    # --- PART 10: optional handoff ---
    if enable_handoff:
        _write_handoff(project)

    # run summary
    summary_path = out_dir / "09_onboarding_run_summary.json"
    summary_path.write_text(
        json.dumps(project.to_summary_dict(), indent=2, default=str), encoding="utf-8"
    )
    project.generated_artifacts.append(str(summary_path))

    return project


def _write_artifacts(project: OnboardingProject) -> None:
    out = Path(project.output_dir)

    # 01 file inventory
    inv_fields = [f.name for f in dataclasses.fields(project.file_inventory[0])] if project.file_inventory else []
    if project.file_inventory:
        _write_csv(out / "01_file_inventory.csv", project.file_inventory, inv_fields)
    _write_json(out / "01_file_inventory.json", project.file_inventory)

    # 02 column profiles
    if project.field_profiles:
        prof_fields = [f.name for f in dataclasses.fields(project.field_profiles[0])]
        _write_csv(out / "02_column_profiles.csv", project.field_profiles, prof_fields)
    _write_json(out / "02_column_profiles.json", project.field_profiles)

    # 03 candidate keys
    if project.candidate_keys:
        key_fields = [f.name for f in dataclasses.fields(project.candidate_keys[0])]
        _write_csv(out / "03_candidate_keys.csv", project.candidate_keys, key_fields)
    else:
        _write_csv(out / "03_candidate_keys.csv", [], ["candidate_key", "file_name", "source_column"])

    # 04 overlap analysis
    overlap_fields = [
        "canonical_candidate", "source_file_a", "source_column_a", "source_file_b",
        "source_column_b", "similarity_score", "sample_match_rate",
        "recommended_primary_source", "recommended_secondary_source", "review_required", "reason",
    ]
    _write_csv(out / "04_source_overlap_analysis.csv", project.overlap_analysis, overlap_fields)

    # 05 mapping candidates
    map_fields = [
        "source_file", "source_file_classification", "source_column",
        "candidate_canonical_field", "confidence", "method",
        "sample_values_redacted", "requires_review", "reason",
    ]
    _write_csv(out / "05_mapping_candidates.csv", project.mapping_candidates, map_fields)
    _write_json(out / "05_mapping_candidates.json", project.mapping_candidates)

    # 06 config suggestions
    cfg_fields = [
        "field", "suggested_value", "confidence", "source_file",
        "source_column_or_document_reference", "evidence", "review_status",
    ]
    _write_csv(out / "06_config_suggestions.csv", project.config_suggestions, cfg_fields)
    cfg_yaml = {
        "client_name": project.client_name,
        "generated_by": "trakt_onboarding_agent_v2",
        "review_status": project.review_status,
        "suggestions": [dataclasses.asdict(s) for s in project.config_suggestions],
    }
    (out / "06_config_suggestions.yaml").write_text(
        yaml.safe_dump(cfg_yaml, sort_keys=False), encoding="utf-8"
    )

    # 07 gap questions
    gap_fields = [
        "question_id", "category", "severity", "question", "reason",
        "candidate_answers", "default_recommendation", "blocking_for", "source_evidence",
        "subject", "subject_value",
    ]
    _write_csv(out / "07_gap_questions.csv", project.gap_questions, gap_fields)
    (out / "07_gap_questions.yaml").write_text(
        yaml.safe_dump([dataclasses.asdict(q) for q in project.gap_questions], sort_keys=False),
        encoding="utf-8",
    )

    # example_answers.yaml — a pre-filled answer template the user can edit and
    # feed back via `--answers` (answer ingestion, PART 4).
    _write_example_answers(project, out / "example_answers.yaml")
    project.generated_artifacts.append(str(out / "example_answers.yaml"))

    for name in [
        "01_file_inventory.csv", "01_file_inventory.json",
        "02_column_profiles.csv", "02_column_profiles.json",
        "03_candidate_keys.csv", "04_source_overlap_analysis.csv",
        "05_mapping_candidates.csv", "05_mapping_candidates.json",
        "06_config_suggestions.csv", "06_config_suggestions.yaml",
        "07_gap_questions.csv", "07_gap_questions.yaml",
    ]:
        project.generated_artifacts.append(str(out / name))


def _example_answer_for(q) -> str:
    """Pick a sensible pre-filled answer for the example template."""
    if q.default_recommendation and q.default_recommendation in q.candidate_answers:
        return q.default_recommendation
    if q.candidate_answers:
        return q.candidate_answers[0]
    return q.default_recommendation or ""


def _write_example_answers(project: OnboardingProject, path: Path) -> None:
    answers = {}
    for q in project.gap_questions:
        answers[q.question_id] = {
            "answer": _example_answer_for(q),
            "approved_by": "user",
            "note": q.question,
        }
    payload = {
        "_doc": (
            "Answer the gap questions below, then run answer ingestion with "
            "`--answers <this file>`. 'answer' must be one of the candidate answers "
            "where candidates exist; dates must be ISO; source answers must be a "
            "source file name."
        ),
        "project_id": project.project_id,
        "answers": answers,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_handoff(project: OnboardingProject) -> None:
    """PART 10 — draft, review-only handoff artefacts (never production config)."""
    out = Path(project.output_dir)

    high_conf = [
        m for m in project.mapping_candidates
        if m.candidate_canonical_field and m.confidence >= HANDOFF_CONFIDENCE
    ]

    # 09 draft client config
    cfg = {f.field: f.suggested_value for f in project.config_suggestions}
    draft_config = {
        "_warning": "DRAFT — review-only. Do not deploy to production without approval.",
        "client_name": project.client_name,
        "portfolio": {
            "asset_class": cfg.get("asset_class", ""),
            "base_currency": cfg.get("currency", ""),
            "country": cfg.get("jurisdiction", ""),
        },
        "default_regime": cfg.get("regime", ""),
        "reporting_date": cfg.get("reporting_date", ""),
        "warehouse": {
            "present": cfg.get("warehouse_facility_present", "unknown"),
            "lender_name": cfg.get("warehouse_lender_name", ""),
            "advance_rate": cfg.get("advance_rate", ""),
            "margin": cfg.get("margin", ""),
            "interest_index": cfg.get("interest_index", ""),
        },
        "geography_policy": cfg.get("geography_policy", ""),
    }
    (out / "09_draft_client_config.yaml").write_text(
        yaml.safe_dump(draft_config, sort_keys=False), encoding="utf-8"
    )

    # 09 draft mapping overrides (high-confidence only)
    overrides = {
        "_warning": "DRAFT — review-only. High-confidence mappings for the existing pipeline.",
        "mappings": [
            {
                "source_file": m.source_file,
                "source_column": m.source_column,
                "canonical_field": m.candidate_canonical_field,
                "confidence": m.confidence,
                "method": m.method,
            }
            for m in high_conf
        ],
    }
    (out / "09_draft_mapping_overrides.yaml").write_text(
        yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8"
    )

    for name in ["09_draft_client_config.yaml", "09_draft_mapping_overrides.yaml"]:
        project.generated_artifacts.append(str(out / name))
