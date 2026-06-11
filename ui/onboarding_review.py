"""
ui/onboarding_review.py

Onboarding Workbench — Streamlit review interface.

Allows an analyst to:
  1. Review config bootstrap questions.
  2. Review field mapping recommendations.
  3. Review enum mapping recommendations.
  4. Submit decisions, persist learned mappings, re-run the agent.
  5. See the updated onboarding status.

Run:
    streamlit run ui/onboarding_review.py

Then enter the run directory path in the sidebar, or pass via URL:
    ?run_dir=out/onboarding/ob_20250610_123456_abc

The page is self-contained: it writes all decision files, persists
learning, and calls run_onboarding_agent() automatically after submit.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — ensure project root is on sys.path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from agents.onboarding_schemas import (
    ConfigBootstrapResult,
    EnumReviewItem,
    MappingReviewItem,
    OnboardingResult,
)
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
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Trakt · Onboarding Workbench",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_NAVY = "#0B1F3B"
_GREEN = "#1A7F4B"
_AMBER = "#D97706"
_RED = "#B91C1C"
_GREY = "#6B7280"
_LIGHT = "#F3F4F6"

_STATUS_COLOUR = {
    "ready_for_validation": _GREEN,
    "review_required": _AMBER,
    "blocked": _RED,
    "failed": _RED,
}
_STATUS_LABEL = {
    "ready_for_validation": "✅ Ready for Validation",
    "review_required": "⚠️ Review Required",
    "blocked": "🚫 Blocked",
    "failed": "❌ Failed",
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  .status-badge {
    display:inline-block; padding:4px 14px; border-radius:12px;
    color:white; font-weight:600; font-size:14px; margin-bottom:8px;
  }
  .metric-row { display:flex; gap:24px; margin:12px 0; flex-wrap:wrap; }
  .metric-box {
    background:#F3F4F6; border-radius:8px; padding:12px 18px;
    min-width:130px; text-align:center;
  }
  .metric-box .val { font-size:22px; font-weight:700; color:#0B1F3B; }
  .metric-box .lbl { font-size:12px; color:#6B7280; margin-top:2px; }
  .blocker-tag {
    background:#FEE2E2; color:#991B1B; padding:1px 8px;
    border-radius:6px; font-size:11px; font-weight:600;
  }
  .ok-tag {
    background:#D1FAE5; color:#065F46; padding:1px 8px;
    border-radius:6px; font-size:11px; font-weight:600;
  }
  .review-tag {
    background:#FEF3C7; color:#92400E; padding:1px 8px;
    border-radius:6px; font-size:11px; font-weight:600;
  }
  .narrative-box {
    background:#F8FAFC; border-left:4px solid #0B1F3B;
    padding:12px 16px; border-radius:0 8px 8px 0;
    font-size:14px; line-height:1.6; margin:12px 0;
  }
  .section-header {
    font-size:16px; font-weight:700; color:#0B1F3B;
    margin:20px 0 8px 0; border-bottom:2px solid #E5E7EB;
    padding-bottom:6px;
  }
  .all-clear {
    background:#D1FAE5; color:#065F46; padding:10px 16px;
    border-radius:8px; font-size:13px; margin:8px 0;
  }
  .conf-high { color:#1A7F4B; font-weight:600; }
  .conf-med  { color:#D97706; font-weight:600; }
  .conf-low  { color:#B91C1C; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Run directory loading helpers
# ---------------------------------------------------------------------------

def _load_run_dir(run_dir: str) -> Optional[Path]:
    p = Path(run_dir).expanduser()
    if not p.exists():
        st.error(f"Run directory not found: `{p}`")
        return None
    return p


def _load_result(run_dir: Path) -> Optional[OnboardingResult]:
    result_path = run_dir / "onboarding_result.json"
    if not result_path.exists():
        st.error(f"`onboarding_result.json` not found in `{run_dir}`.\n\nRun the Onboarding Agent first.")
        return None
    try:
        return OnboardingResult.from_json(result_path)
    except Exception as exc:
        st.error(f"Could not load `onboarding_result.json`: {exc}")
        return None


def _load_run_info(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "run_info.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_canonical_fields(registry_path: str) -> List[str]:
    try:
        import yaml
        data = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8")) or {}
        return sorted(data.get("fields", {}).keys())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Confidence badge helper
# ---------------------------------------------------------------------------

def _conf_badge(conf: float) -> str:
    pct = f"{conf:.0%}"
    if conf >= 0.90:
        return f'<span class="conf-high">{pct}</span>'
    elif conf >= 0.70:
        return f'<span class="conf-med">{pct}</span>'
    return f'<span class="conf-low">{pct}</span>'


# ---------------------------------------------------------------------------
# Section 1: Run summary
# ---------------------------------------------------------------------------

def _render_summary(result: OnboardingResult) -> None:
    st.markdown('<div class="section-header">Run Summary</div>', unsafe_allow_html=True)

    cb = result.config_bootstrap
    col1, col2 = st.columns([3, 2])
    with col1:
        status_colour = _STATUS_COLOUR.get(result.status, _GREY)
        status_label = _STATUS_LABEL.get(result.status, result.status)
        st.markdown(
            f'<span class="status-badge" style="background:{status_colour}">{status_label}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Run ID: `{result.run_id}`")
        if cb:
            st.write(f"**Asset class:** {cb.detected_asset_class or '—'}  "
                     f"| **Regime:** {cb.selected_regime or '—'}  "
                     f"| **Detection confidence:** {cb.detected_asset_confidence:.0%}")

    with col2:
        # Metrics row
        st.markdown(
            f"""<div class="metric-row">
              <div class="metric-box">
                <div class="val">{result.mapped_fields_count}/{result.total_input_fields}</div>
                <div class="lbl">Fields mapped</div>
              </div>
              <div class="metric-box">
                <div class="val">{result.review_fields_count}</div>
                <div class="lbl">For review</div>
              </div>
              <div class="metric-box">
                <div class="val">{result.unmapped_mandatory_count}</div>
                <div class="lbl">Unmapped mandatory</div>
              </div>
              <div class="metric-box">
                <div class="val">{result.enum_success_rate:.0%}</div>
                <div class="lbl">Enum success</div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    if result.narrative_summary:
        st.markdown(
            f'<div class="narrative-box">{result.narrative_summary}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section 2: Config questions
# ---------------------------------------------------------------------------

def _render_config_questions(
    result: OnboardingResult,
    q_state: Dict[str, Dict[str, Any]],
) -> None:
    all_questions = (
        (result.blocker_questions or []) + (result.user_questions or [])
    )
    # Also pull from bootstrap config_questions
    cb = result.config_bootstrap
    if cb and cb.config_questions:
        seen_ids = {q.get("question_id") for q in all_questions}
        for q in cb.config_questions:
            if q.get("question_id") not in seen_ids:
                all_questions.append(q)

    st.markdown('<div class="section-header">Config Questions</div>', unsafe_allow_html=True)

    if not all_questions:
        st.markdown('<div class="all-clear">✅ No config questions — all values resolved.</div>',
                    unsafe_allow_html=True)
        return

    blocking = [q for q in all_questions if q.get("blocking")]
    advisory = [q for q in all_questions if not q.get("blocking")]

    def _render_question(q: Dict[str, Any], is_blocking: bool) -> None:
        qid = q.get("question_id", "")
        label = q.get("question", qid)
        why = q.get("why_needed", "")
        suggested = str(q.get("suggested_answer", "") or "")
        options = q.get("options") or []

        tag = '<span class="blocker-tag">REQUIRED</span>' if is_blocking else '<span class="review-tag">ADVISORY</span>'
        with st.expander(f"{tag} {label}", expanded=is_blocking):
            if why:
                st.caption(f"Why needed: {why}")
            if options:
                current = q_state.get(qid, {}).get("answer", suggested) or suggested
                try:
                    idx = options.index(current) if current in options else 0
                except ValueError:
                    idx = 0
                answer = st.selectbox(
                    "Select value",
                    options=options,
                    index=idx,
                    key=f"q_select_{qid}",
                )
            else:
                answer = st.text_input(
                    "Your answer",
                    value=q_state.get(qid, {}).get("answer", suggested),
                    placeholder=suggested or "Enter value...",
                    key=f"q_text_{qid}",
                )
            approved = st.checkbox(
                "Approve this answer",
                value=q_state.get(qid, {}).get("approved", bool(suggested and not is_blocking)),
                key=f"q_approve_{qid}",
            )
            comments = st.text_input(
                "Comments (optional)",
                value=q_state.get(qid, {}).get("comments", ""),
                key=f"q_comment_{qid}",
            )
            # Update state
            q_state[qid] = {"answer": answer, "approved": approved, "comments": comments}

    if blocking:
        for q in blocking:
            _render_question(q, is_blocking=True)
    if advisory:
        with st.expander(f"Advisory questions ({len(advisory)})"):
            for q in advisory:
                _render_question(q, is_blocking=False)


# ---------------------------------------------------------------------------
# Section 3: Field mapping review
# ---------------------------------------------------------------------------

def _render_mapping_review(
    result: OnboardingResult,
    map_state: Dict[str, Dict[str, Any]],
    canonical_fields: List[str],
) -> None:
    st.markdown('<div class="section-header">Field Mapping Review</div>', unsafe_allow_html=True)

    review_items = [i for i in result.mapping_review_items if i.requires_review]
    clean_count = len(result.mapping_review_items) - len(review_items)

    if clean_count > 0:
        st.markdown(
            f'<div class="all-clear">✅ {clean_count} field(s) mapped automatically with high confidence — no review needed.</div>',
            unsafe_allow_html=True,
        )

    if not review_items:
        if not result.mapping_review_items:
            st.info("No mapping data available.")
        return

    for item in review_items:
        key = item.raw_field
        tag = '<span class="blocker-tag">BLOCKER</span>' if item.blocker else '<span class="review-tag">REVIEW</span>'
        label_text = (
            f"{item.raw_field}  →  "
            f"{item.suggested_canonical_field or '(unmapped)'}"
        )
        with st.expander(f"{tag} {label_text}", expanded=item.blocker):
            col_a, col_b = st.columns([3, 2])
            with col_a:
                conf_html = _conf_badge(item.confidence)
                st.markdown(
                    f"**Raw field:** `{item.raw_field}`  &nbsp; "
                    f"**Suggested:** `{item.suggested_canonical_field or '—'}`  &nbsp; "
                    f"**Confidence:** {conf_html}  &nbsp; "
                    f"**Source:** `{item.mapping_source}`",
                    unsafe_allow_html=True,
                )
                if item.reason:
                    st.caption(item.reason)
                if item.sample_values:
                    st.caption(f"Sample values: {', '.join(str(v) for v in item.sample_values[:5])}")

            with col_b:
                current_state = map_state.get(key, {})
                action = st.radio(
                    "Action",
                    ["Approve suggestion", "Override mapping", "Ignore"],
                    index=current_state.get("action_idx", 0),
                    key=f"map_action_{key}",
                    horizontal=True,
                )
                action_idx = ["Approve suggestion", "Override mapping", "Ignore"].index(action)

                override_field = current_state.get("override_field", item.suggested_canonical_field or "")
                if action == "Override mapping":
                    if canonical_fields:
                        cf_options = [""] + canonical_fields
                        cur_idx = 0
                        if override_field in cf_options:
                            cur_idx = cf_options.index(override_field)
                        override_field = st.selectbox(
                            "Select canonical field",
                            options=cf_options,
                            index=cur_idx,
                            key=f"map_override_{key}",
                        )
                    else:
                        override_field = st.text_input(
                            "Enter canonical field name",
                            value=override_field,
                            key=f"map_override_text_{key}",
                        )

                comments = st.text_input(
                    "Comments",
                    value=current_state.get("comments", ""),
                    key=f"map_comment_{key}",
                )

                # Determine final selected field
                if action == "Approve suggestion":
                    final_field = item.suggested_canonical_field
                    approved = True
                elif action == "Override mapping":
                    final_field = override_field or None
                    approved = bool(final_field)
                else:  # Ignore
                    final_field = None
                    approved = False

                map_state[key] = {
                    "action_idx": action_idx,
                    "override_field": override_field,
                    "comments": comments,
                    "approved": approved,
                    "selected_canonical_field": final_field,
                }


# ---------------------------------------------------------------------------
# Section 4: Enum review
# ---------------------------------------------------------------------------

def _render_enum_review(
    result: OnboardingResult,
    enum_state: Dict[str, Dict[str, Any]],
) -> None:
    st.markdown('<div class="section-header">Enum Mapping Review</div>', unsafe_allow_html=True)

    review_items = [i for i in result.enum_review_items if i.requires_review]
    resolved_count = result.enum_mapped_count

    if resolved_count > 0:
        st.markdown(
            f'<div class="all-clear">✅ {resolved_count} enum field(s) resolved automatically.</div>',
            unsafe_allow_html=True,
        )

    if not review_items:
        if not result.enum_review_items:
            st.info("No enum mapping data available.")
        return

    # Group by field name
    by_field: Dict[str, List[EnumReviewItem]] = {}
    for item in review_items:
        by_field.setdefault(item.field_name, []).append(item)

    for field_name, items in by_field.items():
        with st.expander(f"**{field_name}** — {len(items)} value(s) for review", expanded=any(i.blocker for i in items)):
            for item in items:
                enum_key = f"{field_name}|{item.raw_value}"
                tag = '<span class="blocker-tag">BLOCKER</span>' if item.blocker else '<span class="review-tag">REVIEW</span>'
                st.markdown(
                    f"{tag} Raw: `{item.raw_value}`  →  Suggested: `{item.suggested_value or '—'}`"
                    f"  &nbsp; {_conf_badge(item.confidence)}"
                    f"  &nbsp; count: {item.sample_count}",
                    unsafe_allow_html=True,
                )
                current_state = enum_state.get(enum_key, {})

                col_a, col_b = st.columns([2, 3])
                with col_a:
                    approved = st.checkbox(
                        "Approve",
                        value=current_state.get("approved", item.suggested_value is not None),
                        key=f"enum_approve_{enum_key}",
                    )
                with col_b:
                    selected = st.text_input(
                        "Canonical value",
                        value=current_state.get("selected_value", item.suggested_value or ""),
                        key=f"enum_val_{enum_key}",
                    )

                enum_state[enum_key] = {
                    "approved": approved,
                    "selected_value": selected.strip() or None,
                    "field_name": field_name,
                    "raw_value": item.raw_value,
                }
                st.divider()


# ---------------------------------------------------------------------------
# Submit handler
# ---------------------------------------------------------------------------

def _collect_decisions(
    result: OnboardingResult,
    q_state: Dict[str, Dict[str, Any]],
    map_state: Dict[str, Dict[str, Any]],
    enum_state: Dict[str, Dict[str, Any]],
) -> ReviewSubmission:
    """Read widget state and build a ReviewSubmission."""
    all_questions = (
        (result.blocker_questions or []) + (result.user_questions or [])
    )
    cb = result.config_bootstrap
    if cb and cb.config_questions:
        seen_ids = {q.get("question_id") for q in all_questions}
        for q in cb.config_questions:
            if q.get("question_id") not in seen_ids:
                all_questions.append(q)

    question_answers = [
        QuestionAnswer(
            question_id=q.get("question_id", ""),
            answer=q_state.get(q.get("question_id", ""), {}).get("answer", ""),
            approved=q_state.get(q.get("question_id", ""), {}).get("approved", False),
            comments=q_state.get(q.get("question_id", ""), {}).get("comments", ""),
        )
        for q in all_questions
    ]

    mapping_decisions = [
        MappingDecision(
            raw_field=item.raw_field,
            approved=map_state.get(item.raw_field, {}).get("approved", False),
            selected_canonical_field=map_state.get(item.raw_field, {}).get("selected_canonical_field"),
            comments=map_state.get(item.raw_field, {}).get("comments", ""),
        )
        for item in result.mapping_review_items
        if item.requires_review
    ]

    enum_decisions = []
    for item in result.enum_review_items:
        if not item.requires_review:
            continue
        key = f"{item.field_name}|{item.raw_value}"
        state = enum_state.get(key, {})
        enum_decisions.append(EnumDecision(
            field_name=item.field_name,
            raw_value=item.raw_value,
            approved=state.get("approved", False),
            selected_value=state.get("selected_value"),
            comments="",
        ))

    return ReviewSubmission(
        run_id=result.run_id,
        question_answers=question_answers,
        mapping_decisions=mapping_decisions,
        enum_decisions=enum_decisions,
    )


def _write_decision_files(
    submission: ReviewSubmission,
    run_dir: Path,
    run_info: Dict[str, Any],
) -> Dict[str, Path]:
    """Write questionnaire_answers, mapping_overrides, enum_overrides, submission JSONs."""
    run_id = submission.run_id

    # Questionnaire answers (format expected by run_onboarding_agent)
    qa_path = run_dir / f"{run_id}_questionnaire_answers.json"
    qa_path.write_text(
        json.dumps(build_questionnaire_answers_json(submission.question_answers), indent=2),
        encoding="utf-8",
    )

    # Mapping overrides
    mo_path = run_dir / f"{run_id}_approved_mapping_overrides.json"
    mo_path.write_text(
        json.dumps(build_mapping_overrides_json(submission.mapping_decisions), indent=2),
        encoding="utf-8",
    )

    # Enum overrides
    eo_path = run_dir / f"{run_id}_approved_enum_overrides.json"
    eo_path.write_text(
        json.dumps(build_enum_overrides_json(submission.enum_decisions), indent=2),
        encoding="utf-8",
    )

    # Full submission
    sub_path = run_dir / f"{run_id}_review_submission.json"
    submission.to_json(sub_path)

    return {"qa": qa_path, "mo": mo_path, "eo": eo_path, "sub": sub_path}


def _run_persistence(
    submission: ReviewSubmission,
    run_info: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, int]:
    """Persist approved decisions to alias/enum/config files."""
    counts: Dict[str, int] = {"aliases": 0, "enums": 0, "config": 0}
    session_id = f"workbench_{run_info.get('run_id', '')}"

    aliases_dir = Path(run_info.get("aliases_dir", ""))
    if aliases_dir.exists():
        counts["aliases"] = persist_mapping_decisions(
            submission.mapping_decisions, aliases_dir, session_id
        )

    enum_synonyms_path = Path(run_info.get("schema_registry_path", "")).parent / "enum_synonyms_confirmed.yaml"
    if not enum_synonyms_path.parent.exists():
        enum_synonyms_path = run_dir.parent.parent / "config" / "system" / "enum_synonyms_confirmed.yaml"
    counts["enums"] = persist_enum_decisions(
        submission.enum_decisions, enum_synonyms_path, session_id=session_id
    )

    draft_config = Path(run_dir / f"{run_info.get('run_id', '')}_draft_config.yaml")
    approved_config = Path(run_dir / f"{run_info.get('run_id', '')}_approved_config.yaml")
    counts["config"] = persist_config_answers(
        submission.question_answers, draft_config, approved_config, session_id
    )

    return counts


def _rerun_agent(
    run_info: Dict[str, Any],
    run_dir: Path,
) -> OnboardingResult:
    """Re-run the Onboarding Agent with approved answers and return new result."""
    from agents.onboarding_agent import run_onboarding_agent

    run_id = run_info.get("run_id", "")
    qa_path = run_dir / f"{run_id}_questionnaire_answers.json"
    approved_config = run_dir / f"{run_id}_approved_config.yaml"

    # Use approved config if it was written, else fall back to original client config
    client_config = str(approved_config) if approved_config.exists() else run_info.get("client_config_path", "")

    return run_onboarding_agent(
        raw_tape_path=run_info.get("raw_tape_path", ""),
        run_id=run_id,
        client_config_path=client_config or None,
        schema_registry_path=run_info.get("schema_registry_path") or None,
        aliases_dir=run_info.get("aliases_dir") or None,
        enum_mapping_path=run_info.get("enum_mapping_path") or None,
        output_dir=str(Path(run_info.get("output_dir", "out/onboarding"))),
        questionnaire_answers_path=str(qa_path) if qa_path.exists() else None,
        llm_enabled=run_info.get("llm_enabled", False),
    )


# ---------------------------------------------------------------------------
# Section 5: Submit + re-run
# ---------------------------------------------------------------------------

def _render_submit_section(
    result: OnboardingResult,
    run_dir: Path,
    run_info: Dict[str, Any],
    q_state: Dict[str, Any],
    map_state: Dict[str, Any],
    enum_state: Dict[str, Any],
) -> None:
    st.markdown('<div class="section-header">Submit Decisions</div>', unsafe_allow_html=True)

    review_count = (
        len(result.blocker_questions or [])
        + len(result.user_questions or [])
        + sum(1 for i in result.mapping_review_items if i.requires_review)
        + sum(1 for i in result.enum_review_items if i.requires_review)
    )

    if result.status == "ready_for_validation":
        st.markdown('<div class="all-clear">✅ All items resolved. No review required.</div>',
                    unsafe_allow_html=True)
        _render_validation_handoff(result)
        return

    st.write(f"**{review_count} item(s)** across config, mapping, and enum sections.")

    if st.button("Submit Decisions & Re-run Agent", type="primary", use_container_width=True):
        with st.spinner("Collecting decisions…"):
            submission = _collect_decisions(result, q_state, map_state, enum_state)
            decision_paths = _write_decision_files(submission, run_dir, run_info)

        st.success(f"Decisions written: {len(submission.question_answers)} answers, "
                   f"{len(submission.mapping_decisions)} mapping, "
                   f"{len(submission.enum_decisions)} enum decisions.")

        with st.spinner("Persisting learning (aliases, enums, config)…"):
            counts = _run_persistence(submission, run_info, run_dir)
        st.info(f"Persisted: {counts['aliases']} alias(es), "
                f"{counts['enums']} enum synonym(s), "
                f"{counts['config']} config value(s).")

        if not run_info.get("raw_tape_path"):
            st.error("Cannot re-run: raw tape path not found in run_info.json.")
            return

        with st.spinner("Re-running Onboarding Agent…"):
            try:
                new_result = _rerun_agent(run_info, run_dir)
                st.session_state["wb_new_result"] = new_result
                st.session_state["wb_submitted"] = True
            except Exception as exc:
                st.error(f"Re-run failed: {exc}")
                st.code(traceback.format_exc())
                return

        st.rerun()


# ---------------------------------------------------------------------------
# Section 6: Results after re-run
# ---------------------------------------------------------------------------

def _render_results_screen(
    original: OnboardingResult,
    new_result: OnboardingResult,
    submission: Optional[ReviewSubmission] = None,
) -> None:
    st.markdown("---")
    st.markdown('<div class="section-header">Re-run Results</div>', unsafe_allow_html=True)

    status_colour = _STATUS_COLOUR.get(new_result.status, _GREY)
    status_label = _STATUS_LABEL.get(new_result.status, new_result.status)
    st.markdown(
        f'<span class="status-badge" style="background:{status_colour}">{status_label}</span>',
        unsafe_allow_html=True,
    )

    # Resolution summary
    q_orig = len((original.blocker_questions or []) + (original.user_questions or []))
    q_new = len((new_result.blocker_questions or []) + (new_result.user_questions or []))
    q_resolved = max(0, q_orig - q_new)

    map_orig = sum(1 for i in original.mapping_review_items if i.requires_review)
    map_new = sum(1 for i in new_result.mapping_review_items if i.requires_review)
    map_resolved = max(0, map_orig - map_new)

    enum_orig = sum(1 for i in original.enum_review_items if i.requires_review)
    enum_new = sum(1 for i in new_result.enum_review_items if i.requires_review)
    enum_resolved = max(0, enum_orig - enum_new)

    col1, col2, col3 = st.columns(3)
    col1.metric("Config resolved", f"{q_resolved}/{q_orig}")
    col2.metric("Mapping resolved", f"{map_resolved}/{map_orig}")
    col3.metric("Enum resolved", f"{enum_resolved}/{enum_orig}")

    if new_result.narrative_summary:
        st.markdown(
            f'<div class="narrative-box">{new_result.narrative_summary}</div>',
            unsafe_allow_html=True,
        )

    # Remaining blockers
    remaining = [b for b in (new_result.blocker_questions or []) if b.get("blocking")]
    if remaining:
        st.warning(f"**{len(remaining)} blocker(s) remain:**")
        for b in remaining:
            st.write(f"- `{b.get('field', '')}`: {b.get('question', '')}")

    if new_result.proceed_to_validation:
        _render_validation_handoff(new_result)

    # Reload button
    if st.button("← Review again", use_container_width=False):
        for k in ("wb_submitted", "wb_new_result"):
            st.session_state.pop(k, None)
        st.rerun()


def _render_validation_handoff(result: OnboardingResult) -> None:
    """Part 7 — Validation Agent handoff hook."""
    st.markdown("---")
    st.success("**Proceed to Validation Agent: YES**")
    st.write("The following files are ready for the Validation Agent:")
    if result.canonical_draft_path:
        st.code(result.canonical_draft_path, language=None)
    if result.approved_config_path:
        st.code(result.approved_config_path, language=None)

    # Validation Agent integration point
    if st.button("▶ Run Validation Agent", type="primary", use_container_width=False):
        st.info("Validation Agent integration point — not yet implemented.")
        # TODO Phase 2:
        # from agents.validation_agent import run_validation_agent
        # validation_result = run_validation_agent(
        #     canonical_csv_path=result.canonical_draft_path,
        #     config_path=result.approved_config_path,
        # )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🔍 Trakt · Onboarding Workbench")
    st.caption("Review field mappings, config questions, and enum decisions — then let the agent continue automatically.")

    # ---- Sidebar: run directory input ----
    with st.sidebar:
        st.header("Load Run")

        # URL query param takes priority
        qp_run_dir = st.query_params.get("run_dir", "")
        default_run_dir = qp_run_dir or st.session_state.get("wb_run_dir_input", "")

        run_dir_input = st.text_input(
            "Run directory",
            value=default_run_dir,
            placeholder="out/onboarding/ob_20250610_...",
            help="Path to the run output directory containing onboarding_result.json",
            key="wb_run_dir_input",
        )
        load_btn = st.button("Load", use_container_width=True)

        if load_btn and run_dir_input:
            st.session_state["wb_run_dir"] = run_dir_input
            # Reset state when loading a new run
            for k in ("wb_submitted", "wb_new_result", "wb_q_state", "wb_map_state", "wb_enum_state"):
                st.session_state.pop(k, None)

        st.divider()
        st.caption("Trakt Onboarding Workbench v1")

    # ---- Load run directory ----
    run_dir_str = st.session_state.get("wb_run_dir", run_dir_input or qp_run_dir)
    if not run_dir_str:
        st.info("Enter a run directory path in the sidebar and click **Load**.")
        st.markdown(
            "**Example:**\n"
            "```\nout/onboarding/ob_20250610_123456_abc\n```\n"
            "This directory should contain `onboarding_result.json`.",
        )
        return

    run_dir = _load_run_dir(run_dir_str)
    if not run_dir:
        return

    result = _load_result(run_dir)
    if not result:
        return

    run_info = _load_run_info(run_dir)
    canonical_fields = _load_canonical_fields(run_info.get("schema_registry_path", ""))

    # ---- Initialise session state for review decisions ----
    if "wb_q_state" not in st.session_state:
        st.session_state["wb_q_state"] = {}
    if "wb_map_state" not in st.session_state:
        st.session_state["wb_map_state"] = {}
    if "wb_enum_state" not in st.session_state:
        st.session_state["wb_enum_state"] = {}

    q_state: Dict[str, Any] = st.session_state["wb_q_state"]
    map_state: Dict[str, Any] = st.session_state["wb_map_state"]
    enum_state: Dict[str, Any] = st.session_state["wb_enum_state"]

    # ---- If already submitted, show results ----
    if st.session_state.get("wb_submitted") and "wb_new_result" in st.session_state:
        _render_summary(result)
        _render_results_screen(result, st.session_state["wb_new_result"])
        return

    # ---- Main review UI ----
    _render_summary(result)
    st.divider()
    _render_config_questions(result, q_state)
    st.divider()
    _render_mapping_review(result, map_state, canonical_fields)
    st.divider()
    _render_enum_review(result, enum_state)
    st.divider()
    _render_submit_section(result, run_dir, run_info, q_state, map_state, enum_state)


if __name__ == "__main__":
    main()
