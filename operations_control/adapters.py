"""operations_control.adapters — governed wrappers around the agent seam.

``GovernedAdapters`` wraps any :class:`engine.orchestrator_agent.adapters.
AgentAdapters` (production: :class:`RealAgentAdapters`) and observes every
``StepResult`` through a recorder callback — no agent code is modified, the
inner adapter does all the work.

The translators in this module turn orchestrator state + onboarding decision
artefacts into Governed Agent Results and operator decisions. All operator-facing
text is produced via :mod:`operations_control.language`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

from engine.orchestrator_agent.adapters import AgentAdapters, PortfolioSpec, StepResult
from engine.orchestrator_agent.state import (
    RunState,
    STEP_DONE,
    STEP_FAILED,
    STEP_HALTED,
    STEP_PENDING,
    STEP_RUNNING,
)

from . import language
from .contracts import (
    DecisionRequired,
    GovernedAgentResult,
    KIND_FIELD_MAPPING,
    KIND_PUBLICATION,
    KIND_TRANSFORMATION,
    KIND_VALIDATION_EXCEPTION,
    OUTCOME_MI_ANNEX2,
    ST_BLOCKED,
    ST_COMPLETED,
    ST_NEEDS_REVIEW,
    ST_READY,
    ST_RUNNING,
    ST_WAITING,
    STAGE_ASSEMBLY,
    STAGE_MAPPING,
    STAGE_PROJECTION,
    STAGE_PUBLICATION,
    STAGE_RECEIVED,
    STAGE_UNDERSTANDING,
    STAGE_VALIDATION,
    WorkflowRun,
    new_id,
    now_iso,
)

#: Orchestrator per-portfolio step -> operator stage.
STEP_TO_STAGE = {
    "onboard": STAGE_UNDERSTANDING,     # halts on decisions surface as mapping
    "transform": STAGE_VALIDATION,
    "validate": STAGE_VALIDATION,
    "stamp": STAGE_ASSEMBLY,
    "assemble": STAGE_ASSEMBLY,
    "route": STAGE_ASSEMBLY,
    "project": STAGE_PROJECTION,
}

#: The onboarding agent's action for "this required field lives in THAT source
#: column". It is the one action that needs a value from the operator beyond the
#: choice itself — the column name — so OCC carries it explicitly.
#: (engine.onboarding_agent.target_first_decisions.SUPPORTED_ACTIONS)
ACTION_PROVIDE_SOURCE_MAPPING = "provide_source_mapping"

DECISIONS_FILE = "34_target_first_decisions.yaml"
OVERRIDES_FILE = "12_approved_mapping_overrides.yaml"
TAPE_GAPS_FILE = "18c_central_tape_gaps.csv"
APPROVED_DECISIONS_FILE = "34_target_first_decisions_approved.yaml"
REVIEW_QUEUE_FILE = "33_mapping_review_queue.json"
LLM_RECS_FILE = "36_target_first_llm_recommendations.json"

#: Review-queue groups that require an operator decision (deterministic
#: backstop statuses that PASSED compatibility checks but need a human), vs the
#: groups that are unsafe/blocked (decision offered, flagged non-recommended).
_QUEUE_DECISION_GROUPS = (
    "needs_user_decision_risky_conflicting",
    "missing_target_propose_schema_extension",
)


class GovernedAdapters(AgentAdapters):
    """Delegating wrapper: every stage call passes through to ``inner`` and the
    observed ``StepResult`` is reported to ``recorder(step_name, result)``."""

    def __init__(self, inner: AgentAdapters,
                 recorder: Callable[[str, StepResult], None]):
        self.inner = inner
        self.recorder = recorder

    def _observe(self, step: str, fn: Callable[[], StepResult]) -> StepResult:
        try:
            r = fn()
        except Exception:
            self.recorder(step, StepResult(ok=False, blocking=False,
                                           blockers=["internal error"],
                                           message="internal error"))
            raise
        self.recorder(step, r)
        return r

    def onboard(self, spec: PortfolioSpec, work_dir: Path) -> StepResult:
        return self._observe("onboard", lambda: self.inner.onboard(spec, work_dir))

    def transform(self, spec: PortfolioSpec, handoff_manifest: str,
                  work_dir: Path) -> StepResult:
        return self._observe("transform",
                             lambda: self.inner.transform(spec, handoff_manifest, work_dir))

    def validate(self, spec: PortfolioSpec, transformation_manifest: str,
                 work_dir: Path) -> StepResult:
        return self._observe("validate",
                             lambda: self.inner.validate(spec, transformation_manifest, work_dir))

    def stamp_provenance(self, spec: PortfolioSpec, validated_csv: str,
                         out_dir: Path) -> StepResult:
        return self._observe("stamp",
                             lambda: self.inner.stamp_provenance(spec, validated_csv, out_dir))

    def assemble(self, stamped_paths: Sequence[str], out_dir: Path, client_id: str,
                 target: str, *, regime: Optional[str] = None) -> StepResult:
        return self._observe("assemble",
                             lambda: self.inner.assemble(stamped_paths, out_dir,
                                                         client_id, target, regime=regime))

    def route_mi(self, central_canonical: str) -> StepResult:
        return self._observe("route", lambda: self.inner.route_mi(central_canonical))

    def project(self, central_canonical: str, out_dir: Path, regime: str) -> StepResult:
        return self._observe("project",
                             lambda: self.inner.project(central_canonical, out_dir, regime))


# --------------------------------------------------------------------------- #
# Decision extraction from onboarding artefacts
# --------------------------------------------------------------------------- #

def _find_artifact(root: Path, name: str) -> Optional[Path]:
    if not root.exists():
        return None
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def _load_llm_recommendations(root: Path) -> Dict[str, Dict[str, Any]]:
    p = _find_artifact(root, LLM_RECS_FILE)
    if not p:
        return {}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    recs = doc.get("recommendations") or doc.get("decisions") or []
    out: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        if isinstance(r, dict) and r.get("decision_id"):
            out[str(r["decision_id"])] = r
    return out


def extract_mapping_decisions(work_dir: Path, workflow: WorkflowRun) -> List[DecisionRequired]:
    """Read the run's pending onboarding decision artefacts and produce
    self-contained operator decisions. LLM recommendations are attached as
    ADVISORY only (provenance ``llm``) and only when the deterministic backstop
    did not mark the item unsafe."""
    out: List[DecisionRequired] = []
    llm = _load_llm_recommendations(work_dir)

    # 1. Target-first decisions template (pending Gate 4 decisions).
    dpath = _find_artifact(work_dir, DECISIONS_FILE)
    if dpath is not None:
        try:
            doc = yaml.safe_load(dpath.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            doc = {}
        for d in doc.get("decisions") or []:
            if not isinstance(d, dict):
                continue
            if str(d.get("status", "pending")).lower() == "approved":
                continue
            did = str(d.get("decision_id") or new_id("dec"))
            target = str(d.get("target_field") or d.get("canonical_field") or "")
            dtype = str(d.get("decision_type") or "")
            options = []
            for a in (d.get("options") or d.get("available_actions")
                      or d.get("actions") or []):
                # The coverage matrix labels the action "map_source_column";
                # the apply step calls it "provide_source_mapping". Offer the
                # name the engine will actually execute, or the operator picks
                # an option that silently applies to nothing.
                a = _ENGINE_ACTION.get(str(a), str(a))
                options.append({"value": a, "label": _action_label(a)})
            if not any(o["value"] == "defer" for o in options):
                options.append({"value": "defer",
                                "label": _action_label("defer")})
            rec: Dict[str, Any] = {}
            advisory = llm.get(did)
            unsafe = _looks_unsafe(d)
            if advisory and not unsafe:
                rec = {"source": "llm",
                       "value": str(advisory.get("selected_action")
                                    or advisory.get("recommended_action") or ""),
                       "confidence": advisory.get("confidence"),
                       "checked": True}
            elif d.get("recommended_action"):
                rec = {"source": "deterministic",
                       "value": str(d.get("recommended_action")), "checked": True}
            friendly = _friendly(target)
            if dtype == "missing_required_target":
                title = f"The report needs '{friendly}'"
                question = (f"The report requires '{friendly}', but Trakt could "
                            "not find it in the files. How should it be treated?")
            else:
                title = (f"Decide how to treat '{friendly}'" if target
                         else "A data decision needs your confirmation")
                question = (f"The field '{friendly}' could not be filled "
                            "automatically. How should Trakt treat it?"
                            if target else
                            "Trakt needs your decision on how to treat part "
                            "of this data.")
            evidence: List[Dict[str, Any]] = []
            if d.get("issue") or d.get("evidence_summary"):
                evidence.append({"label": "What Trakt found", "kind": "text",
                                 "data": {"issue": d.get("issue", ""),
                                          "detail": d.get("evidence_summary", "")}})
            out.append(DecisionRequired(
                decision_id=f"{workflow.workflow_id}_{did}",
                kind=(KIND_FIELD_MAPPING if "mapping" in dtype
                      else KIND_TRANSFORMATION),
                title=title,
                question=question,
                blocking=bool(d.get("blocking", True)),
                recommendation=rec, options=options, evidence=evidence,
                subject={"artefact": "target_first_decisions",
                         "decision_id": did, "target_field": target,
                         "decision_type": dtype}))

    # 2. Mapping review queue (source-column -> canonical mapping reviews).
    qpath = _find_artifact(work_dir, REVIEW_QUEUE_FILE)
    if qpath is not None:
        try:
            q = json.loads(qpath.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            q = {}
        for group in _QUEUE_DECISION_GROUPS:
            for item in (q.get(group) or []):
                if not isinstance(item, dict):
                    continue
                src = str(item.get("source_column") or item.get("column") or "")
                tgt = str(item.get("canonical_field") or item.get("target") or "")
                if not src:
                    continue
                did = f"{workflow.workflow_id}_map_{_slug(src)}"
                options = [{"value": tgt, "label": f"Yes — it is {_friendly(tgt)}"}] if tgt else []
                for alt in (item.get("candidates") or [])[:3]:
                    v = str(alt.get("canonical_field") if isinstance(alt, dict) else alt)
                    if v and v != tgt:
                        options.append({"value": v, "label": f"It is {_friendly(v)}"})
                options.append({"value": "__ignore__", "label": "Ignore this column"})
                out.append(DecisionRequired(
                    decision_id=did, kind=KIND_FIELD_MAPPING,
                    title=f"Confirm where '{src}' belongs",
                    question=(f"The file column '{src}' looks like "
                              f"{_friendly(tgt)}. Is that right?" if tgt else
                              f"Trakt could not match the file column '{src}'. "
                              "What is it?"),
                    blocking=True,
                    recommendation=({"source": ("llm" if item.get("mapping_source") == "llm"
                                                else "deterministic"),
                                     "value": tgt,
                                     "confidence": item.get("confidence"),
                                     "checked": True} if tgt else {}),
                    options=options,
                    subject={"artefact": "mapping_review_queue",
                             "source_column": src, "candidate": tgt,
                             "group": group}))

    # 3. Which column IS the loan? Several columns in a lender's tape are unique
    # per row — an account reference, a policy number, a customer identifier —
    # and only one of them keys the loan. Gate 1 records the ambiguity rather
    # than settling it by column order; this is where a person settles it.
    out.extend(_loan_key_decisions(work_dir, workflow))
    return out


def _loan_key_decisions(work_dir: Optional[Path],
                        workflow: WorkflowRun) -> List[DecisionRequired]:
    if not work_dir:
        return []
    path = _find_artifact(Path(work_dir), TAPE_GAPS_FILE)
    if path is None:
        return []
    try:
        import csv
        with path.open(encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.DictReader(fh)
                    if r.get("issue_type") == "ambiguous_loan_key"]
    except Exception:  # noqa: BLE001
        return []
    out: List[DecisionRequired] = []
    for r in rows:
        selected = str(r.get("source_column") or "")
        # The description carries the candidates in parentheses; the file column
        # carries the one that was used.
        import re
        m = re.search(r"\(([^)]*)\)", str(r.get("description") or ""))
        candidates = [c.strip() for c in (m.group(1).split(",") if m else [])
                      if c.strip()]
        if selected and selected not in candidates:
            candidates.insert(0, selected)
        if len(candidates) < 2:
            continue
        options = [{"value": c,
                    "label": (f"{c} — the one Trakt used" if c == selected
                              else c)} for c in candidates]
        out.append(DecisionRequired(
            decision_id=f"{workflow.workflow_id}_loankey_{_slug(selected)}",
            kind=KIND_FIELD_MAPPING,
            title="Confirm which column identifies the loan",
            question=("More than one column in this file is unique for every "
                      "row, so more than one could be the loan reference. "
                      "Which one is it? If Trakt groups by the borrower "
                      "instead of the loan, a borrower's separate loans are "
                      "merged and every figure in the report changes."),
            blocking=False,
            recommendation=({"source": "deterministic", "value": selected,
                             "checked": True} if selected else {}),
            options=options,
            evidence=[{"label": "What Trakt found", "kind": "text",
                       "data": {"file": r.get("source_file", ""),
                                "candidates": ", ".join(candidates),
                                "used": selected}}],
            allowed_scopes=["portfolio", "client"],
            default_scope="portfolio",
            subject={"artefact": "central_tape_gaps",
                     "source_column": selected,
                     "canonical_field": "loan_identifier"}))
    return out


def _coverage_evidence(handoff: Dict[str, Any],
                       summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gate 1's target-coverage numbers, shown as the Review Centre's evidence.

    Straight from the handoff manifest — no recomputation, so what the operator
    reads is what Gate 2 will act on.
    """
    if not handoff:
        return []
    rows = [
        ("Decisions still open", handoff.get("operator_decision_pending_count")),
        ("Of those, blocking", handoff.get("blocking_decision_count")),
        ("Approved decisions applied", handoff.get("approved_decision_applied_count")),
        ("Target fields with no disposition",
         summary.get("unresolved_disposition_count")),
        ("Blocking the data checks", summary.get("blocking_for_validation_count")),
    ]
    data = {label: value for label, value in rows if value is not None}
    if not data:
        return []
    return [{"label": "What Gate 1 found", "kind": "table", "data": data}]


def _looks_unsafe(d: Dict[str, Any]) -> bool:
    s = str(d.get("backstop_status") or d.get("safety") or "").upper()
    return s in ("UNSAFE", "BLOCKED")


#: Gate 1 publishes its own verdict in the handoff manifest. These are the keys
#: the Review Centre reports; OCC never recomputes them.
HANDOFF_FILE = "24_onboarding_handoff_manifest.json"


def read_handoff(work_dir: Optional[Path],
                 onboard_readiness: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Gate 1's own readiness verdict, as Gate 1 wrote it.

    The Operations Control Centre reports what the Onboarding Agent found; it
    does not form a second opinion about mapping success. Returns ``{}`` when
    there is no manifest, which callers must treat as "nothing claimed" rather
    than "everything fine".
    """
    path: Optional[Path] = None
    ref = (onboard_readiness or {}).get("mi_handoff") or \
        (onboard_readiness or {}).get("handoff_manifest")
    if ref:
        cand = Path(str(ref))
        if cand.exists():
            path = cand
    if path is None and work_dir:
        path = _find_artifact(Path(work_dir), HANDOFF_FILE)
    if path is None:
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return doc if isinstance(doc, dict) else {}


def handoff_ready(handoff: Dict[str, Any]) -> bool:
    """True only when Gate 1 SAYS the package is ready for Gate 2.

    Absent manifest -> not ready. Gate 2 enforces the same contract itself; this
    is the reporting side of that decision, so it must fail the same way.
    """
    return bool(handoff.get("ready_for_transformation_validation"))


def _workflow_summary_status(work_dir: Optional[Path]) -> str:
    """The onboarding agent's own status verdict (READY / NEEDS_CONFIRMATION /
    NEEDS_CONFIGURATION / BLOCKED / FAILED) from its 40_ summary, if present."""
    if not work_dir:
        return ""
    p = _find_artifact(Path(work_dir), "40_operator_workflow_summary.json")
    if not p:
        return ""
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("status") or "")
    except Exception:  # noqa: BLE001
        return ""


NEEDS_CONFIGURATION_SENTENCE = (
    "The regulatory report needs additional set-up inside Trakt before this "
    "can continue. This is not something in your files — please contact your "
    "Trakt administrator.")


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


_REGIME_CODE_NAMES: Optional[Dict[str, str]] = None


def _regime_code_names() -> Dict[str, str]:
    """Reverse lookup of regime codes (e.g. ``RREL2``) to canonical field
    names, read once from the EXISTING field registry (read-only)."""
    global _REGIME_CODE_NAMES
    if _REGIME_CODE_NAMES is not None:
        return _REGIME_CODE_NAMES
    out: Dict[str, str] = {}
    try:
        reg = yaml.safe_load(
            Path("config/system/fields_registry.yaml").read_text(
                encoding="utf-8")) or {}
        for name, spec in (reg.get("fields") or {}).items():
            for rm in (spec or {}).get("regime_mapping", {}).values():
                code = (rm or {}).get("code")
                if code:
                    out.setdefault(str(code), str(name))
    except Exception:  # noqa: BLE001 — labels degrade gracefully
        pass
    _REGIME_CODE_NAMES = out
    return out


def _friendly(field: str) -> str:
    """Operator-friendly name for a canonical field OR a regime code."""
    f = str(field or "").strip()
    if not f:
        return "this field"
    import re
    if re.fullmatch(r"[A-Z]{3,5}\d{1,3}", f):     # ESMA-style code
        f = _regime_code_names().get(f, "")
        if not f:
            return "a required regulatory field"
    return f.replace("_", " ").strip() or "this field"


#: Coverage-matrix action label -> the action name the apply step supports.
_ENGINE_ACTION = {
    "map_source_column": ACTION_PROVIDE_SOURCE_MAPPING,
    "confirm_ND_code": "confirm_default_or_nd",
    "set_derivation_or_default": "configure_static_value",
    "accept_mapping": "confirm_selected",
    "ignore": "mark_not_applicable",
}


def _action_label(action: str) -> str:
    labels = {
        "accept_mapping": "Accept the suggested match",
        "confirm_selected": "Accept the suggested match",
        "choose_alternative": "Choose a different match",
        "map_source_column": "Point it at a column in the files",
        "provide_source_mapping": "Point it at a column in the files",
        "set_derivation_or_default": "Use a standard or calculated value",
        "configure_static_value": "Use a fixed value",
        "confirm_ND_code": "Record it as 'no data' (allowed by the regulator)",
        "confirm_default_or_nd": "Use the standard value",
        "mark_not_applicable": "Mark as not applicable",
        "merge_or_reconcile": "Combine the overlapping sources",
        "treat_source_value_as_null": "Treat the value as empty",
        "mark_unavailable": "This information is not in the files",
        "use_default": "Use the standard value",
        "defer": "Decide later",
        "reject_recommendation": "Reject the suggestion",
        "ignore": "Ignore this field",
    }
    return labels.get(action, str(action).replace("_", " ").capitalize())


# --------------------------------------------------------------------------- #
# RunState -> stage statuses + Governed Agent Results
# --------------------------------------------------------------------------- #

def _stage_status_for_step(status: str) -> str:
    return {STEP_PENDING: ST_WAITING, STEP_RUNNING: ST_RUNNING,
            STEP_DONE: ST_COMPLETED, STEP_HALTED: ST_NEEDS_REVIEW,
            STEP_FAILED: ST_BLOCKED}.get(status, ST_WAITING)


def translate_run_state(workflow: WorkflowRun, state: RunState,
                        work_dir: Optional[Path]) -> Dict[str, GovernedAgentResult]:
    """Translate the orchestrator's RunState into one Governed Agent Result per
    operator stage. Purely read-only over the orchestrator's persisted state."""
    results: Dict[str, GovernedAgentResult] = {}
    p = state.portfolios[0] if state.portfolios else None
    wid = workflow.workflow_id

    def gar(stage: str, status: str, summary: str, *, why: str = "",
            decisions: Optional[List[DecisionRequired]] = None,
            warnings: Optional[List[str]] = None,
            blockers: Optional[List[str]] = None,
            inputs: Optional[List[str]] = None,
            outputs: Optional[List[str]] = None,
            evidence: Optional[List[Dict[str, Any]]] = None) -> GovernedAgentResult:
        g = GovernedAgentResult(
            run_id=wid, stage=stage, status=status, summary=summary,
            why_it_matters=why, decisions_required=decisions or [],
            warnings=warnings or [], blockers=blockers or [],
            evidence=evidence or [],
            input_references=inputs or [], output_references=outputs or [],
            agent_version="trakt-engine", configuration_version="",
            started_at=state.created_at, completed_at=now_iso())
        results[stage] = g
        return g

    # Received — the delivery is registered by definition once a run exists.
    files = [f.get("name", "") for f in (workflow.delivery.get("files") or [])]
    gar(STAGE_RECEIVED, ST_COMPLETED,
        f"{len(files)} file{'s' if len(files) != 1 else ''} received and registered.",
        evidence=[{"label": "Files in this delivery", "kind": "list",
                   "data": files}],
        inputs=[workflow.delivery.get("input_path", "")])

    if p is None:
        return results

    onboard = p.step("onboard")
    ob_status = _stage_status_for_step(onboard.status)

    # Gate 1's OWN verdict. A completed onboard step does not mean the mapping
    # is settled: Gate 1 routinely finishes, writes its tape, and declares
    # ready_for_transformation_validation=false with a queue of blocking
    # decisions. Reading the manifest is what lets the Review Centre report
    # what Gate 1 found instead of asserting that everything matched.
    handoff = read_handoff(work_dir, onboard.readiness)
    hs = handoff.get("target_contract_completion_summary") or {}
    blocking_decisions = int(handoff.get("blocking_decision_count") or 0)
    pending_decisions = int(handoff.get("operator_decision_pending_count") or 0)
    mapping_settled = (onboard.status == STEP_DONE
                       and (handoff_ready(handoff) if handoff else True)
                       and blocking_decisions == 0)

    # Understanding.
    if onboard.status == STEP_DONE:
        loan_count = (onboard.readiness or {}).get("loan_count")
        gar(STAGE_UNDERSTANDING, ST_COMPLETED,
            (f"Trakt read the files and recognised {loan_count} loans."
             if loan_count else "Trakt read and understood the files."),
            outputs=[onboard.output_path or ""],
            inputs=[p.input])
    elif onboard.status in (STEP_HALTED, STEP_FAILED):
        decisions = extract_mapping_decisions(work_dir, workflow) if work_dir else []
        if decisions or _workflow_summary_status(work_dir) == "NEEDS_CONFIGURATION":
            gar(STAGE_UNDERSTANDING, ST_COMPLETED,
                "Trakt read the files. Some items need attention before it "
                "can continue.")
        else:
            gar(STAGE_UNDERSTANDING, ST_BLOCKED,
                "Trakt could not fully read these files.",
                why="The files may be missing an expected listing, or a step "
                    "did not finish cleanly.",
                blockers=language.humanise_blockers(onboard.blockers))
    else:
        gar(STAGE_UNDERSTANDING, ob_status,
            "Trakt is reading the files." if ob_status == ST_RUNNING
            else "Waiting to read the files.")

    # Mapping.
    mapping_why = ("Matching each file column to the right meaning is what "
                   "makes the reports correct.")
    if mapping_settled:
        applied = int(handoff.get("approved_decision_applied_count") or 0)
        gar(STAGE_MAPPING, ST_COMPLETED,
            ("Every field was matched, using the "
             f"{applied} decision{'s' if applied != 1 else ''} you have already "
             "approved." if applied else
             "Every field was matched automatically."),
            why=mapping_why,
            evidence=_coverage_evidence(handoff, hs))
    elif onboard.status == STEP_DONE:
        # Gate 1 finished but says the package is NOT ready. Report its numbers
        # and offer its queue; Gate 2 will refuse the package until they are
        # answered, so hiding them just strands the run.
        decisions = extract_mapping_decisions(work_dir, workflow) if work_dir else []
        n = len(decisions) or blocking_decisions or pending_decisions
        gar(STAGE_MAPPING,
            ST_NEEDS_REVIEW if decisions else ST_BLOCKED,
            (f"Trakt matched most columns, but {n} "
             f"need{'s' if n == 1 else ''} your decision before the figures "
             "can be checked."),
            why=mapping_why,
            decisions=decisions,
            evidence=_coverage_evidence(handoff, hs),
            blockers=([] if decisions else language.humanise_blockers(
                ["onboarding handoff not ready for transformation/validation"])))
    elif onboard.status in (STEP_HALTED, STEP_FAILED):
        decisions = extract_mapping_decisions(work_dir, workflow) if work_dir else []
        if decisions:
            n = len(decisions)
            gar(STAGE_MAPPING, ST_NEEDS_REVIEW,
                f"Most fields were matched automatically. "
                f"{n} item{'s' if n != 1 else ''} require your confirmation.",
                why="Matching each file column to the right meaning is what "
                    "makes the reports correct.",
                decisions=decisions)
        elif _workflow_summary_status(work_dir) == "NEEDS_CONFIGURATION":
            gar(STAGE_MAPPING, ST_BLOCKED,
                "Additional set-up is needed inside Trakt.",
                why="Some regulatory fields have no configured treatment yet. "
                    "This is a set-up task, not a problem with your files.",
                blockers=[NEEDS_CONFIGURATION_SENTENCE])
        else:
            gar(STAGE_MAPPING, ST_BLOCKED if onboard.status == STEP_FAILED
                else ST_NEEDS_REVIEW,
                "The files could not be matched automatically.",
                blockers=language.humanise_blockers(onboard.blockers))
    else:
        gar(STAGE_MAPPING, ST_WAITING, "Waiting for the files to be read.")

    # Validation (transform + validate steps; lean MI runs skip them).
    tr, va = p.step("transform"), p.step("validate")
    steps_present = tr.status != STEP_PENDING or va.status != STEP_PENDING \
        or workflow.outcome == OUTCOME_MI_ANNEX2
    if va.status == STEP_DONE:
        gar(STAGE_VALIDATION, ST_COMPLETED, "All checks passed.",
            outputs=[va.output_path or ""])
    elif va.status == STEP_HALTED:
        gar(STAGE_VALIDATION, ST_NEEDS_REVIEW,
            "Some figures did not pass Trakt's checks and need your review.",
            why="Publishing figures that fail checks could mislead the "
                "report's readers.",
            decisions=[DecisionRequired(
                decision_id=f"{wid}_validation_exception",
                kind=KIND_VALIDATION_EXCEPTION,
                title="Review the flagged checks",
                question="Some checks did not pass. Do you want Trakt to "
                         "continue and prepare the report anyway?",
                blocking=True,
                options=[{"value": "proceed", "label": "Continue — I accept "
                                                       "the flagged items"},
                         {"value": "stop", "label": "Stop — this needs fixing "
                                                    "at source"}],
                default_scope="file",
                subject={"artefact": "validation_halt"})],
            blockers=language.humanise_blockers(va.blockers))
    elif tr.status == STEP_HALTED and not mapping_settled:
        # Gate 2 refused an unready Gate 1 package. That is the mapping queue
        # talking, not a data-quality problem — say so, and leave the decision
        # to the mapping stage rather than raising a second, competing one.
        gar(STAGE_VALIDATION, ST_WAITING,
            "The checks are waiting for the mapping decisions above.",
            why="Trakt will not check figures whose meaning is still unsettled.")
    elif va.status == STEP_FAILED or tr.status in (STEP_HALTED, STEP_FAILED):
        src = va if va.status == STEP_FAILED else tr
        gar(STAGE_VALIDATION, ST_BLOCKED,
            "The data could not be fully checked.",
            blockers=language.humanise_blockers(src.blockers))
    elif tr.status == STEP_RUNNING or va.status == STEP_RUNNING:
        gar(STAGE_VALIDATION, ST_RUNNING, "Trakt is checking the data.")
    elif onboard.status == STEP_DONE and not steps_present:
        gar(STAGE_VALIDATION, ST_COMPLETED,
            "Standard checks were applied while reading the files.")
    else:
        gar(STAGE_VALIDATION, ST_WAITING, "Waiting for mapping to finish.")

    # The regulatory delivery stages (regulatory_config, projection,
    # delivery_prep, xml_delivery) are OWNED by the engine's governed Annex 2
    # chain, which runs the proven route after assembly — they are not derived
    # from orchestrator state and are not emitted here.

    # Assembly (stamp + assemble + route).
    st, asm = p.step("stamp"), state.assemble
    if asm.status == STEP_DONE:
        rows = (asm.readiness or {}).get("total_rows")
        gar(STAGE_ASSEMBLY, ST_COMPLETED,
            (f"The combined report data is ready ({rows} records)."
             if rows else "The combined report data is ready."),
            outputs=[state.central_canonical_path or ""])
    elif asm.status in (STEP_HALTED, STEP_FAILED) or st.status in (STEP_HALTED, STEP_FAILED):
        src = asm if asm.status in (STEP_HALTED, STEP_FAILED) else st
        gar(STAGE_ASSEMBLY, ST_BLOCKED,
            "Trakt could not put the final data together.",
            blockers=language.humanise_blockers(src.blockers))
    elif st.status == STEP_RUNNING or asm.status == STEP_RUNNING:
        gar(STAGE_ASSEMBLY, ST_RUNNING, "Putting the final data together.")
    else:
        gar(STAGE_ASSEMBLY, ST_WAITING, "Waiting for earlier steps.")

    # Publication — never automatic; READY only when everything upstream done.
    # For the Annex 2 outcome the delivery chain (engine-owned) decides when
    # publication becomes ready — orchestrator completion is not enough.
    if workflow.outcome == OUTCOME_MI_ANNEX2:
        gar(STAGE_PUBLICATION, ST_WAITING,
            "Waiting for the regulatory delivery steps.")
    elif state.status == STEP_DONE:
        gar(STAGE_PUBLICATION, ST_READY,
            "The report is prepared and waiting for your approval to publish.",
            why="Nothing is shared until you approve it.",
            decisions=[DecisionRequired(
                decision_id=f"{wid}_publish",
                kind=KIND_PUBLICATION,
                title="Approve publication",
                question="Publish this report as the latest official version?",
                blocking=True,
                options=[{"value": "publish", "label": "Publish"},
                         {"value": "hold", "label": "Hold — not yet"}],
                default_scope="file",
                subject={"artefact": "publication"})])
    else:
        gar(STAGE_PUBLICATION, ST_WAITING, "Waiting for all steps to finish.")

    return results
