"""
streamlit_onboarding_workbench.py
=================================

PART 2–11 — a lightweight, file/artefact-based review workbench for the Trakt
Onboarding Agent.

A junior analyst can open an existing onboarding run, review unresolved
mappings / gaps / conflicts / source precedence / enum issues, approve or correct
decisions, save client-specific mapping memory, rerun consolidation and watch
readiness improve — all without a database.

Run::

    python -m streamlit run engine/onboarding_agent/streamlit_onboarding_workbench.py

The module is deliberately split so that **all data loading, decision
serialisation, answer generation, action logging and memory persistence live in
plain importable functions** with no Streamlit dependency. Streamlit is imported
lazily inside :func:`main`, so the module imports cleanly (and is unit-testable)
in an environment without Streamlit installed.

Workbench artefacts written into the project dir:
    24_workbench_pending_decisions.yaml   staged (not-yet-approved) decisions
    25_workbench_answers.yaml             answers YAML (ingestion-compatible)
    26_workbench_action_log.json          append-only audit trail of actions
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Allow running as a script (streamlit run ...) or as a module.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import mapping_memory as mm  # noqa: E402

# Artefact filenames.
PENDING_DECISIONS_FILE = "24_workbench_pending_decisions.yaml"
ANSWERS_FILE = "25_workbench_answers.yaml"
ACTION_LOG_FILE = "26_workbench_action_log.json"

DEFAULT_REGISTRY = "config/system/fields_registry.yaml"


# ===========================================================================
# Loading
# ===========================================================================


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class WorkbenchContext:
    """All artefacts a workbench session needs, loaded from a project dir.

    File/artefact based: no database. Missing artefacts degrade gracefully to
    empty collections so the workbench still opens on a partial run.
    """

    def __init__(
        self,
        project_dir: str | Path,
        client_id: str = "",
        run_id: str = "",
        mode: str = "",
        regulatory_reporting_enabled: bool = False,
        registry_path: str = DEFAULT_REGISTRY,
    ):
        self.project_dir = Path(project_dir)
        self.registry_path = registry_path

        self.run_summary = _load_json(self.project_dir / "09_onboarding_run_summary.json") or {}
        self.inventory = _load_json(self.project_dir / "01_file_inventory.json") or []
        self.mapping_candidates = _load_json(self.project_dir / "05_mapping_candidates.json") or []
        self.mapping_ambiguities = _load_json(self.project_dir / "05b_mapping_ambiguities.json") or []
        self.mapping_trace = _load_csv(self.project_dir / "05c_mapping_trace.csv")
        self.out_of_scope = _load_csv(self.project_dir / "05a_out_of_scope_fields.csv")
        self.overlap = _load_csv(self.project_dir / "04_source_overlap_analysis.csv")
        self.gap_questions = _load_yaml(self.project_dir / "07_gap_questions.yaml") or []
        self.domain_coverage = _load_json(self.project_dir / "17_domain_coverage.json") or []
        self.llm_usage = _load_json(self.project_dir / "22_llm_usage_summary.json") or {}
        # Output-side artefacts (present after a promote dry-run).
        self.readiness = _load_json(self._output("manifests", "21_pipeline_handoff_readiness.json")) or {}
        self.central_gaps = _load_csv(self._output("gaps", "18c_central_tape_gaps.csv"))
        self.central_lineage = _load_csv(self._output("lineage", "18b_central_tape_lineage.csv"))

        # Sidebar-supplied run context (falls back to run summary).
        self.client_id = client_id or self.run_summary.get("client_id", "") or self.project_dir.name
        self.run_id = run_id or self.run_summary.get("run_id", "") or "run"
        self.mode = mode or self.run_summary.get("onboarding_mode", "regulatory_mi")
        self.regulatory_reporting_enabled = regulatory_reporting_enabled

    def _output(self, *parts: str) -> Path:
        return self.project_dir / "output" / Path(*parts)

    # -- derived views -------------------------------------------------
    @property
    def source_file_names(self) -> List[str]:
        return [i.get("file_name", "") for i in self.inventory]

    def question_by_id(self) -> Dict[str, Dict[str, Any]]:
        return {q.get("question_id"): q for q in self.gap_questions}


def load_project(
    project_dir: str | Path,
    client_id: str = "",
    run_id: str = "",
    mode: str = "",
    regulatory_reporting_enabled: bool = False,
    registry_path: str = DEFAULT_REGISTRY,
) -> WorkbenchContext:
    """Load an existing onboarding run directory into a :class:`WorkbenchContext`."""
    return WorkbenchContext(
        project_dir, client_id=client_id, run_id=run_id, mode=mode,
        regulatory_reporting_enabled=regulatory_reporting_enabled,
        registry_path=registry_path,
    )


# ===========================================================================
# Field scope (mode-aware canonical targets for the mapping dropdown)
# ===========================================================================


def in_scope_canonical_fields(
    registry_path: str | Path,
    mode: str,
    regulatory_reporting_enabled: bool = False,
) -> List[str]:
    """Sorted list of canonical fields that are IN SCOPE for the mode.

    Used to populate the "change mapping" dropdown. A mapping to a field outside
    this set is not offered (PART 5): MI-only must not list regulatory non-core
    fields as normal mapping targets.
    """
    try:
        from engine.onboarding_agent.field_scope import resolve_field_scope
        from engine.onboarding_agent.mode_policy import load_mode_policy
        policy = load_mode_policy(mode)
        scope = resolve_field_scope(
            str(registry_path), policy,
            regulatory_reporting_enabled=regulatory_reporting_enabled,
        )
        return sorted(scope.included_fields)
    except Exception:
        # Fall back to the raw registry field list.
        data = _load_yaml(Path(registry_path)) or {}
        return sorted((data.get("fields", {}) or {}).keys())


# ===========================================================================
# Run overview (PART 3)
# ===========================================================================


def _readiness_indicator(review_status: str, blocking_gaps: int) -> str:
    if review_status == "blocked" or blocking_gaps > 0:
        return "Blocked"
    if review_status in ("ready_for_handoff", "ready_for_regulatory_handoff",
                          "ready_for_mi", "ready_for_pipeline"):
        return "Ready"
    return "Needs review"


def run_overview(ctx: WorkbenchContext) -> Dict[str, Any]:
    """Compact run overview dict (PART 3)."""
    rs = ctx.run_summary
    counts = rs.get("counts", {}) or {}
    blocking = sum(1 for q in ctx.gap_questions if q.get("severity") == "blocking")
    domains = sorted({
        d.get("domain") for d in ctx.domain_coverage
        if d.get("status") in ("covered", "partially_covered")
    })
    llm = ctx.llm_usage or {}
    readiness = ctx.readiness or {}
    review_status = rs.get("review_status", "draft")
    warnings = len(readiness.get("warnings", []) or []) or sum(
        1 for q in ctx.gap_questions if q.get("severity") in ("high", "medium")
    )
    central_lender = bool(readiness.get("central_lender_tape_created")) or (
        ctx._output("central", "18_central_lender_tape.csv").exists()
    )
    central_pipeline = bool(readiness.get("central_pipeline_tape_created")) or (
        ctx._output("central", "18a_central_pipeline_tape.csv").exists()
    )
    return {
        "client_id": ctx.client_id,
        "run_id": ctx.run_id,
        "mode": ctx.mode,
        "storage_backend": rs.get("storage_backend", "local"),
        "input_uri": rs.get("input_uri", ""),
        "output_uri": rs.get("output_uri", ""),
        "input_files": counts.get("source_files", len(ctx.inventory)),
        "domains_detected": domains,
        "readiness_status": _readiness_indicator(review_status, blocking),
        "review_status_raw": review_status,
        "central_lender_tape_created": central_lender,
        "central_pipeline_tape_created": central_pipeline,
        "blocking_gaps": blocking,
        "warnings": warnings,
        "llm_used": bool(llm.get("llm_enabled")),
        "estimated_llm_cost": llm.get("estimated_cost", 0),
        "client_mapping_memory_loaded": bool(
            (rs.get("client_memory_summary", {}) or {}).get("client_mapping_memory_loaded")
        ),
        "memory_entries_applied": (rs.get("client_memory_summary", {}) or {}).get(
            "memory_entries_applied", 0),
    }


# ===========================================================================
# Section views (PART 4–8)
# ===========================================================================


def domain_rows(ctx: WorkbenchContext) -> List[Dict[str, Any]]:
    """Domain coverage rows (PART 4)."""
    rows = []
    for d in ctx.domain_coverage:
        rows.append({
            "domain": d.get("domain", ""),
            "status": d.get("status", ""),
            "source_files": d.get("source_files", []),
            "mapped_fields_count": d.get("mapped_fields_count", 0),
            "missing_required_fields": d.get("missing_required_fields", []),
            "blocking": d.get("blocking", False),
            "notes": d.get("notes", ""),
        })
    return rows


def mapping_rows(ctx: WorkbenchContext) -> List[Dict[str, Any]]:
    """Mapping review rows (PART 5)."""
    trace_by_key = {
        (t.get("source_file"), t.get("source_column")): t for t in ctx.mapping_trace
    }
    oos_keys = {(o.get("source_file"), o.get("source_column")) for o in ctx.out_of_scope}
    rows = []
    for m in ctx.mapping_candidates:
        key = (m.get("source_file"), m.get("source_column"))
        trace = trace_by_key.get(key, {})
        method = m.get("method", "")
        rows.append({
            "source_file": m.get("source_file", ""),
            "source_column": m.get("source_column", ""),
            "selected_candidate": m.get("candidate_canonical_field", ""),
            "confidence": m.get("confidence", 0),
            "selection_reason": m.get("reason", "") or trace.get("decision_reason", ""),
            "alias_hit": method == "alias",
            "ambiguity_rule_applied": bool(m.get("ambiguity_rule_applied")),
            "review_required": bool(m.get("requires_review")),
            "field_scope_status": "out_of_scope" if key in oos_keys else "in_scope",
            "unmapped_reason": "" if m.get("candidate_canonical_field") else m.get("reason", ""),
        })
    # Append the explicitly out-of-scope columns (diverted before mapping).
    seen = {(r["source_file"], r["source_column"]) for r in rows}
    for o in ctx.out_of_scope:
        key = (o.get("source_file"), o.get("source_column"))
        if key in seen:
            continue
        rows.append({
            "source_file": o.get("source_file", ""),
            "source_column": o.get("source_column", ""),
            "selected_candidate": o.get("candidate_field", ""),
            "confidence": 0,
            "selection_reason": o.get("reason", ""),
            "alias_hit": False,
            "ambiguity_rule_applied": False,
            "review_required": False,
            "field_scope_status": "out_of_scope",
            "unmapped_reason": o.get("reason", ""),
        })
    return rows


def unresolved_mapping_rows(ctx: WorkbenchContext) -> List[Dict[str, Any]]:
    """Mapping rows needing attention (review-required or unmapped, in scope)."""
    return [
        r for r in mapping_rows(ctx)
        if r["field_scope_status"] == "in_scope"
        and (r["review_required"] or not r["selected_candidate"])
    ]


_GAP_SEVERITIES = ["blocking", "high", "medium", "low", "info"]
_GAP_ISSUE_BY_CATEGORY = {
    "date": "value_conflict",
    "source_of_truth": "value_conflict",
    "enum": "enum_issue",
    "config": "config_missing",
    "geography": "config_missing",
    "core_field": "missing_required_field",
    "warehouse": "config_missing",
    "scope": "domain_missing",
    "memory_conflict": "mapping_unresolved",
}


def gap_rows(ctx: WorkbenchContext) -> List[Dict[str, Any]]:
    """Gap rows with a derived issue_type (PART 6)."""
    rows = []
    for q in ctx.gap_questions:
        cat = q.get("category", "")
        rows.append({
            "question_id": q.get("question_id", ""),
            "severity": q.get("severity", "info"),
            "category": cat,
            "issue_type": _GAP_ISSUE_BY_CATEGORY.get(cat, "mapping_unresolved"),
            "question": q.get("question", ""),
            "reason": q.get("reason", ""),
            "candidate_answers": q.get("candidate_answers", []) or [],
            "default_recommendation": q.get("default_recommendation", ""),
            "subject": q.get("subject", ""),
            "subject_value": q.get("subject_value", ""),
            "source_evidence": q.get("source_evidence", ""),
        })
    return rows


def conflict_rows(ctx: WorkbenchContext) -> List[Dict[str, Any]]:
    """Source conflicts from overlap analysis + central tape gaps (PART 7)."""
    rows = []
    for o in ctx.overlap:
        try:
            match_rate = float(o.get("sample_match_rate", 0) or 0)
        except (ValueError, TypeError):
            match_rate = 0.0
        rows.append({
            "canonical_field": o.get("canonical_candidate", ""),
            "source_a": f"{o.get('source_file_a','')}:{o.get('source_column_a','')}",
            "source_b": f"{o.get('source_file_b','')}:{o.get('source_column_b','')}",
            "source_file_a": o.get("source_file_a", ""),
            "source_file_b": o.get("source_file_b", ""),
            "match_rate": match_rate,
            "recommended_primary_source": o.get("recommended_primary_source", ""),
            "review_required": str(o.get("review_required", "")).lower() == "true",
        })
    # Augment with value_conflict gaps from the central tape (per-loan detail).
    for g in ctx.central_gaps:
        if g.get("issue_type") == "value_conflict":
            rows.append({
                "canonical_field": g.get("canonical_field", ""),
                "source_a": f"{g.get('source_file','')}:{g.get('source_column','')}",
                "source_b": "(see description)",
                "source_file_a": g.get("source_file", ""),
                "source_file_b": "",
                "match_rate": None,
                "recommended_primary_source": "",
                "review_required": True,
                "description": g.get("description", ""),
            })
    return rows


def precedence_rows(ctx: WorkbenchContext) -> List[Dict[str, Any]]:
    """Source-precedence review rows: the source_of_truth gaps (PART 7)."""
    return [r for r in gap_rows(ctx) if r["category"] == "source_of_truth"]


def enum_rows(ctx: WorkbenchContext) -> List[Dict[str, Any]]:
    """Enum decision rows: the enum gaps (PART 8)."""
    return [r for r in gap_rows(ctx) if r["category"] == "enum"]


# ===========================================================================
# Decision serialisation (PART 5/6/11)
# ===========================================================================


def write_pending_decisions(
    project_dir: str | Path, pending: Dict[str, Any]
) -> Path:
    """Write staged (not-yet-approved) decisions to 24_workbench_pending_decisions.yaml."""
    project_dir = Path(project_dir)
    payload = {
        "_doc": "Workbench staged decisions. Not applied until ingested / promoted.",
        "generated_at": _now_iso(),
        **pending,
    }
    path = project_dir / PENDING_DECISIONS_FILE
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def generate_answers_yaml(
    project_dir: str | Path,
    gap_answers: Dict[str, Dict[str, Any]],
    project_id: str = "",
) -> Path:
    """Write 25_workbench_answers.yaml — compatible with answer ingestion.

    ``gap_answers`` is ``{question_id: {answer, approved_by, note}}`` (exactly the
    schema :func:`answer_ingestion.ingest_answers` consumes).
    """
    project_dir = Path(project_dir)
    payload = {
        "_doc": "Workbench-generated answers. Compatible with `ingest-answers`.",
        "project_id": project_id,
        "answers": gap_answers,
    }
    path = project_dir / ANSWERS_FILE
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def answers_from_decisions(
    ctx: WorkbenchContext, decisions: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Roll workbench view decisions up into a unified gap-answer dict.

    ``decisions['gap_answers']`` is taken verbatim; any not-explicitly-answered
    blocking/high gap falls back to its default recommendation so the generated
    answers file is as complete as possible.
    """
    answers: Dict[str, Dict[str, Any]] = {}
    explicit = decisions.get("gap_answers", {}) or {}
    for q in ctx.gap_questions:
        qid = q.get("question_id")
        if qid in explicit:
            answers[qid] = explicit[qid]
            continue
        # Fall back to a sensible default for the answer template.
        default = q.get("default_recommendation", "")
        cands = q.get("candidate_answers", []) or []
        if default and default in cands:
            ans = default
        elif cands:
            ans = cands[0]
        else:
            ans = default
        if ans:
            answers[qid] = {"answer": ans, "approved_by": "workbench",
                            "note": q.get("question", "")}
    return answers


# ===========================================================================
# Action log (PART 11)
# ===========================================================================


def append_action_log(
    project_dir: str | Path,
    client_id: str,
    run_id: str,
    action: str,
    inputs: Optional[Dict[str, Any]] = None,
    outputs_written: Optional[List[str]] = None,
    status: str = "ok",
) -> Path:
    """Append one workbench action to 26_workbench_action_log.json (append-only)."""
    project_dir = Path(project_dir)
    path = project_dir / ACTION_LOG_FILE
    log = _load_json(path) or []
    if not isinstance(log, list):
        log = []
    log.append({
        "timestamp": _now_iso(),
        "client_id": client_id,
        "run_id": run_id,
        "action": action,
        "inputs": inputs or {},
        "outputs_written": outputs_written or [],
        "status": status,
    })
    path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    return path


# ===========================================================================
# Client memory persistence (PART 9/11)
# ===========================================================================


def save_decisions_to_memory(
    decisions: List[Dict[str, Any]],
    client_id: str,
    memory_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    approved_by: str = "workbench",
    run_id: str = "",
) -> Dict[str, Any]:
    """Persist selected workbench decisions into client-scoped mapping memory.

    Each decision dict may carry: decision_type, source_file_pattern,
    source_column, canonical_field, mode, domain, source_value, evidence, notes.
    Returns ``{"saved": N, "memory_dir": ...}``.
    """
    mem_dir = mm.resolve_memory_dir(
        memory_dir=memory_dir, output_dir=output_dir, client_id=client_id
    )
    store = mm.MappingMemoryStore(mem_dir, client_id=client_id)
    saved = 0
    for d in decisions:
        evidence = dict(d.get("evidence", {}) or {})
        if run_id and "reviewed_in_run_id" not in evidence:
            evidence["reviewed_in_run_id"] = run_id
        entry = mm.MemoryEntry(
            client_id=client_id,
            decision_type=d.get("decision_type", mm.DECISION_MAPPING_OVERRIDE),
            source_file_pattern=d.get("source_file_pattern", "*"),
            source_column=d.get("source_column", ""),
            canonical_field=d.get("canonical_field", ""),
            mode=d.get("mode", ""),
            domain=d.get("domain", ""),
            source_value=d.get("source_value", ""),
            confidence=float(d.get("confidence", 1.0) or 1.0),
            approved_by=approved_by,
            evidence=evidence,
            notes=d.get("notes", ""),
        )
        store.save_entry(entry)
        saved += 1
    return {"saved": saved, "memory_dir": str(mem_dir)}


# ===========================================================================
# Action wrappers (PART 11) — call existing Python functions directly (safer
# than shelling out).
# ===========================================================================


def ingest_workbench_answers(project_dir: str | Path, confirm: bool = True) -> Dict[str, Any]:
    """Ingest 25_workbench_answers.yaml via the existing answer-ingestion logic."""
    from engine.onboarding_agent.answer_ingestion import ingest_answers
    project_dir = Path(project_dir)
    return ingest_answers(str(project_dir), str(project_dir / ANSWERS_FILE), confirm=confirm)


def promote_dry_run(
    project_dir: str | Path,
    registry_path: str = DEFAULT_REGISTRY,
    client_id: str = "",
    run_id: str = "",
    mode: str = "",
    regulatory_reporting_enabled: bool = False,
) -> Dict[str, Any]:
    """Build central tapes + dry-run promotion plan (no Gates, no Azure upload)."""
    from engine.onboarding_agent import (
        central_tape_builder, domain_coverage as dc, promotion_planner, storage_paths,
    )
    project_dir = Path(project_dir)
    rs = _load_json(project_dir / "09_onboarding_run_summary.json") or {}
    mode = mode or rs.get("onboarding_mode", "regulatory_mi")
    client_id = client_id or rs.get("client_id", "") or project_dir.name
    run_id = run_id or rs.get("run_id", "") or "run"
    run_paths = storage_paths.resolve_run_paths(
        project_dir=str(project_dir),
        input_dir=rs.get("input_dir", "") or None,
        output_root=str(project_dir / "output"),
        client_id=client_id, run_id=run_id,
    )
    coverage = dc.load_coverage(project_dir / "17_domain_coverage.json")
    if not coverage:
        coverage = dc.rebuild_coverage(project_dir, registry_path, mode,
                                       regulatory_reporting_enabled=regulatory_reporting_enabled)
    tape_result = central_tape_builder.build_central_tapes(
        project_dir, run_paths, registry_path, mode=mode,
        regulatory_reporting_enabled=regulatory_reporting_enabled,
    )
    plan = promotion_planner.build_promotion_plan(
        project_dir, run_paths, tape_result, coverage, mode,
        regulatory_reporting_enabled, client_name=rs.get("client_name", client_id),
        project_id=client_id,
    )
    return {"tape_result": tape_result, "plan": plan}


def apply_memory_and_rerun(
    project_dir: str | Path,
    registry_path: str = DEFAULT_REGISTRY,
    aliases_dir: str = "config/system",
    client_id: str = "",
    run_id: str = "",
    mode: str = "",
    regulatory_reporting_enabled: bool = False,
    memory_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-run onboarding mapping with client memory applied (PART 10/11)."""
    from engine.onboarding_agent.onboarding_orchestrator import run_onboarding
    project_dir = Path(project_dir)
    rs = _load_json(project_dir / "09_onboarding_run_summary.json") or {}
    client_id = client_id or rs.get("client_id", "")
    project = run_onboarding(
        input_dir=rs.get("input_dir", ""),
        client_name=rs.get("client_name", client_id),
        output_dir=str(project_dir),
        registry_path=registry_path,
        aliases_dir=aliases_dir,
        mode=mode or rs.get("onboarding_mode", "regulatory_mi"),
        regulatory_reporting_enabled=regulatory_reporting_enabled,
        client_id=client_id,
        run_id=run_id or rs.get("run_id", ""),
        client_memory_dir=memory_dir or "",
        apply_client_memory=True,
    )
    return {
        "review_status": project.review_status,
        "gap_questions": len(project.gap_questions),
        "client_memory_summary": project.client_memory_summary,
    }


def refresh_review_pack(project_dir: str | Path) -> List[str]:
    """Refresh the static HTML review pack with the latest approval/promotion state."""
    project_dir = Path(project_dir)
    refreshed: List[str] = []
    try:
        from engine.onboarding_agent.review_pack_builder import (
            refresh_review_pack_approval, refresh_review_pack_promotion,
        )
        refresh_review_pack_approval(project_dir)
        out_root = project_dir / "output"
        if out_root.exists():
            refresh_review_pack_promotion(project_dir, out_root)
        refreshed.append(str(project_dir / "08_onboarding_review_pack.html"))
    except Exception:
        pass
    return refreshed


# ===========================================================================
# Streamlit UI (lazy import — keeps the module importable without Streamlit)
# ===========================================================================


def main() -> None:  # pragma: no cover - exercised only under `streamlit run`
    import streamlit as st

    st.set_page_config(page_title="Trakt · Onboarding Workbench",
                       page_icon="🧭", layout="wide")
    st.title("🧭 Trakt · Onboarding Review Workbench")
    st.caption("Open an onboarding run, review unresolved items, approve/correct, "
               "save client memory, rerun consolidation, and watch readiness improve.")

    with st.sidebar:
        st.header("Load run")
        project_dir = st.text_input("project_dir", value=st.session_state.get("wb_pd", ""),
                                    placeholder="onboarding_output/demo_onboarding_v1")
        client_id = st.text_input("client_id", value="")
        run_id = st.text_input("run_id", value="")
        mode = st.selectbox(
            "mode", ["", "regulatory_mi", "mi_only", "mna_dd", "warehouse_securitisation"],
            index=0)
        reg = st.checkbox("regulatory_reporting_enabled", value=False)
        load = st.button("Load", use_container_width=True)
        if load:
            st.session_state["wb_pd"] = project_dir

    project_dir = st.session_state.get("wb_pd", project_dir)
    if not project_dir or not Path(project_dir).exists():
        st.info("Enter an existing onboarding project directory in the sidebar and click **Load**.")
        return

    ctx = load_project(project_dir, client_id=client_id, run_id=run_id, mode=mode,
                       regulatory_reporting_enabled=reg)
    decisions: Dict[str, Any] = st.session_state.setdefault("wb_decisions",
                                                            {"gap_answers": {}, "memory": []})

    tabs = st.tabs([
        "1. Overview", "2. Domains", "3. Mappings", "4. Gaps", "5. Conflicts",
        "6. Precedence", "7. Enums", "8. Client memory", "9. Actions", "10. Readiness",
    ])

    _ui_overview(st, ctx, tabs[0])
    _ui_domains(st, ctx, tabs[1])
    _ui_mappings(st, ctx, tabs[2], decisions)
    _ui_gaps(st, ctx, tabs[3], decisions)
    _ui_conflicts(st, ctx, tabs[4], decisions)
    _ui_precedence(st, ctx, tabs[5], decisions)
    _ui_enums(st, ctx, tabs[6], decisions)
    _ui_memory(st, ctx, tabs[7], decisions)
    _ui_actions(st, ctx, tabs[8], decisions)
    _ui_readiness(st, ctx, tabs[9])


# -- per-tab renderers (pragma: no cover — UI only) -------------------------


def _ui_overview(st, ctx, tab):  # pragma: no cover
    with tab:
        ov = run_overview(ctx)
        badge = {"Ready": "🟢", "Needs review": "🟡", "Blocked": "🔴"}.get(
            ov["readiness_status"], "⚪")
        st.subheader(f"{badge} {ov['readiness_status']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input files", ov["input_files"])
        c2.metric("Blocking gaps", ov["blocking_gaps"])
        c3.metric("Warnings", ov["warnings"])
        c4.metric("Est. LLM cost", ov["estimated_llm_cost"])
        st.write(f"**client_id:** {ov['client_id']} · **run_id:** {ov['run_id']} · "
                 f"**mode:** {ov['mode']} · **storage:** {ov['storage_backend']}")
        st.write(f"**Domains detected:** {', '.join(ov['domains_detected']) or '—'}")
        st.write(f"**Central lender tape:** {'yes' if ov['central_lender_tape_created'] else 'no'} · "
                 f"**Central pipeline tape:** {'yes' if ov['central_pipeline_tape_created'] else 'no'}")
        st.write(f"**LLM used:** {'yes' if ov['llm_used'] else 'no'} · "
                 f"**Client memory loaded:** {'yes' if ov['client_mapping_memory_loaded'] else 'no'} "
                 f"({ov['memory_entries_applied']} applied)")


def _ui_domains(st, ctx, tab):  # pragma: no cover
    with tab:
        st.subheader("Domain coverage")
        st.dataframe(domain_rows(ctx), use_container_width=True)


def _ui_mappings(st, ctx, tab, decisions):  # pragma: no cover
    with tab:
        st.subheader("Mapping review")
        fields = in_scope_canonical_fields(ctx.registry_path, ctx.mode,
                                           ctx.regulatory_reporting_enabled)
        st.dataframe(mapping_rows(ctx), use_container_width=True)
        st.markdown("**Unresolved / ambiguous columns**")
        actions = ["approve", "change_mapping", "ignore", "validation_only",
                   "out_of_scope", "ask_later"]
        for r in unresolved_mapping_rows(ctx):
            key = f"{r['source_file']}|{r['source_column']}"
            with st.expander(f"{r['source_file']} · {r['source_column']} → "
                             f"{r['selected_candidate'] or '(unmapped)'}"):
                act = st.selectbox("Action", actions, key=f"map_act_{key}")
                target = r["selected_candidate"]
                if act == "change_mapping":
                    target = st.selectbox("Canonical field (in scope)", [""] + fields,
                                          key=f"map_tgt_{key}")
                if st.button("Stage decision", key=f"map_btn_{key}"):
                    decisions.setdefault("mapping", []).append(
                        {"source_file": r["source_file"], "source_column": r["source_column"],
                         "action": act, "canonical_field": target})
                    st.success("Staged.")


def _ui_gaps(st, ctx, tab, decisions):  # pragma: no cover
    with tab:
        st.subheader("Gap review")
        rows = gap_rows(ctx)
        for sev in _GAP_SEVERITIES:
            group = [r for r in rows if r["severity"] == sev]
            if not group:
                continue
            st.markdown(f"### {sev} ({len(group)})")
            for r in group:
                with st.expander(f"{r['question_id']} · {r['issue_type']} · {r['question']}"):
                    st.caption(r["reason"])
                    cands = r["candidate_answers"] or [r["default_recommendation"]]
                    ans = st.selectbox("Answer", cands, key=f"gap_{r['question_id']}")
                    if st.button("Stage answer", key=f"gapbtn_{r['question_id']}"):
                        decisions["gap_answers"][r["question_id"]] = {
                            "answer": ans, "approved_by": "workbench", "note": r["question"]}
                        st.success("Staged.")


def _ui_conflicts(st, ctx, tab, decisions):  # pragma: no cover
    with tab:
        st.subheader("Conflict review")
        st.dataframe(conflict_rows(ctx), use_container_width=True)


def _ui_precedence(st, ctx, tab, decisions):  # pragma: no cover
    with tab:
        st.subheader("Source precedence review")
        for r in precedence_rows(ctx):
            with st.expander(f"{r['question_id']} · {r['subject']}"):
                st.caption(r["reason"])
                ans = st.selectbox("Primary source", r["candidate_answers"],
                                   key=f"prec_{r['question_id']}")
                save_mem = st.checkbox("Save as client memory default", key=f"precmem_{r['question_id']}")
                if st.button("Stage", key=f"precbtn_{r['question_id']}"):
                    decisions["gap_answers"][r["question_id"]] = {
                        "answer": ans, "approved_by": "workbench", "note": r["question"]}
                    if save_mem:
                        decisions["memory"].append({
                            "decision_type": mm.DECISION_SOURCE_PRECEDENCE,
                            "canonical_field": r["subject"], "mode": ctx.mode,
                            "evidence": {"primary_source_file": ans}})
                    st.success("Staged.")


def _ui_enums(st, ctx, tab, decisions):  # pragma: no cover
    with tab:
        st.subheader("Enum decision review")
        for r in enum_rows(ctx):
            with st.expander(f"{r['question_id']} · {r['subject']} = {r['subject_value']}"):
                st.caption(r["reason"])
                ans = st.selectbox("Decision", r["candidate_answers"], key=f"enum_{r['question_id']}")
                save_mem = st.checkbox("Save to client memory", key=f"enummem_{r['question_id']}")
                if st.button("Stage", key=f"enumbtn_{r['question_id']}"):
                    decisions["gap_answers"][r["question_id"]] = {
                        "answer": ans, "approved_by": "workbench", "note": r["question"]}
                    if save_mem:
                        decisions["memory"].append({
                            "decision_type": mm.DECISION_ENUM_MAPPING,
                            "canonical_field": r["subject"], "source_value": r["subject_value"],
                            "mode": ctx.mode, "evidence": {"decision": ans}})
                    st.success("Staged.")


def _ui_memory(st, ctx, tab, decisions):  # pragma: no cover
    with tab:
        st.subheader("Client mapping memory")
        st.write(f"Decisions staged for memory: {len(decisions.get('memory', []))}")
        st.json(decisions.get("memory", []))


def _ui_actions(st, ctx, tab, decisions):  # pragma: no cover
    with tab:
        st.subheader("Save / apply / rerun actions")
        st.warning("These write project-scoped artefacts. Review staged decisions first.")
        if st.button("Save pending decisions (24)"):
            p = write_pending_decisions(ctx.project_dir, decisions)
            append_action_log(ctx.project_dir, ctx.client_id, ctx.run_id,
                              "save_pending_decisions", outputs_written=[str(p)])
            st.success(f"Wrote {p}")
        if st.button("Generate answers YAML (25)"):
            ans = answers_from_decisions(ctx, decisions)
            p = generate_answers_yaml(ctx.project_dir, ans, project_id=ctx.client_id)
            append_action_log(ctx.project_dir, ctx.client_id, ctx.run_id,
                              "generate_answers_yaml", outputs_written=[str(p)])
            st.success(f"Wrote {p}")
        if st.button("Ingest answers (10–15)"):
            rep = ingest_workbench_answers(ctx.project_dir, confirm=True)
            append_action_log(ctx.project_dir, ctx.client_id, ctx.run_id,
                              "ingest_answers", status=rep.get("approval_status", ""),
                              outputs_written=rep.get("artefacts_written", []))
            st.json(rep)
        if st.button("Save selected decisions to client memory"):
            res = save_decisions_to_memory(decisions.get("memory", []), ctx.client_id,
                                           output_dir=str(ctx.project_dir.parent),
                                           run_id=ctx.run_id)
            append_action_log(ctx.project_dir, ctx.client_id, ctx.run_id,
                              "save_to_client_memory", outputs_written=[res["memory_dir"]])
            st.success(f"Saved {res['saved']} entries to {res['memory_dir']}")
        if st.button("Promote dry-run (central tapes + manifests)"):
            res = promote_dry_run(ctx.project_dir, ctx.registry_path, ctx.client_id,
                                  ctx.run_id, ctx.mode, ctx.regulatory_reporting_enabled)
            append_action_log(ctx.project_dir, ctx.client_id, ctx.run_id, "promote_dry_run",
                              status=res["plan"]["readiness_status"])
            st.json({"readiness": res["plan"]["readiness_status"]})
        if st.button("Apply client memory and rerun mapping"):
            res = apply_memory_and_rerun(ctx.project_dir, ctx.registry_path,
                                         client_id=ctx.client_id, run_id=ctx.run_id,
                                         mode=ctx.mode,
                                         regulatory_reporting_enabled=ctx.regulatory_reporting_enabled,
                                         memory_dir=str(ctx.project_dir.parent / ctx.client_id / "client_memory"))
            append_action_log(ctx.project_dir, ctx.client_id, ctx.run_id, "apply_memory_and_rerun")
            st.json(res)
        if st.button("Refresh review pack"):
            refreshed = refresh_review_pack(ctx.project_dir)
            append_action_log(ctx.project_dir, ctx.client_id, ctx.run_id,
                              "refresh_review_pack", outputs_written=refreshed)
            st.success("Refreshed.")


def _ui_readiness(st, ctx, tab):  # pragma: no cover
    with tab:
        st.subheader("Readiness and artefacts")
        st.json(ctx.readiness or {"note": "Run a promote dry-run to compute readiness."})
        st.markdown("**Artefacts in project dir**")
        st.write(sorted(p.name for p in ctx.project_dir.glob("*")
                        if p.is_file())[:60])


if __name__ == "__main__":  # pragma: no cover
    main()
