"""apps.blob_trigger_app.ops_advice — the "what do I do next?" advisory.

Turns an event manifest into a single, unambiguous operator instruction: which
of {approve, promote, rerun, fix_data_supply, fix_mapping, investigate, none}
comes next, the EXACT command to run, and the follow-up commands after it. Pure
(no storage / no Azure) so the router can embed it in every terminal manifest and
the ops CLI can reproduce it from a persisted record.
"""

from __future__ import annotations

from typing import Any, Dict

# Operator actions (the manifest ``next_action.action`` vocabulary).
ACT_APPROVE = "approve"
ACT_PROMOTE = "promote"
ACT_RERUN = "rerun"
ACT_FIX_DATA = "fix_data_supply"
ACT_FIX_MAPPING = "fix_mapping"
ACT_RESOLVE_DECISIONS = "resolve_decisions"
ACT_INVESTIGATE = "investigate"
ACT_NONE = "none"

# Audit labels this module reasons over (kept local to avoid importing router).
_EVT_NEW_SOURCE_PENDING = "new_source_pending_review"
_EVT_SCHEMA_DRIFT_PENDING = "schema_drift_pending_review"
_EVT_INCOMPLETE_PACK_PENDING = "incomplete_pack_pending_review"
_EVT_KNOWN_SOURCE_HALTED = "known_source_halted"
_EVT_FAILED = "failed"

OPS = "python -m apps.blob_trigger_app.ops"


def next_operator_action(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the next operator step for a terminal event manifest.

    Always includes the identifiers an operator needs (``approval_id``,
    ``run_id``, ``source_portfolio_id``, ``pack_key``) so the advisory is
    self-contained. ``command`` is the exact next command; ``then`` lists the
    commands that follow once it succeeds.
    """
    ev = manifest.get("event_decision")
    status = manifest.get("status")
    approval_id = manifest.get("approval_id")
    pack_key = manifest.get("pack_key")
    run_id = manifest.get("orchestrator_run_id")
    spid = manifest.get("source_portfolio_id")
    diag = manifest.get("orchestrator_diagnostics") or {}
    val_errors = diag.get("validation_errors") or []
    gap = diag.get("registry_gap_count") or 0
    recs = diag.get("mapping_recommendations") or []
    hr = diag.get("handoff_readiness") or {}
    ref = run_id or pack_key

    base = {"approval_id": approval_id, "run_id": run_id,
            "source_portfolio_id": spid, "pack_key": pack_key}

    def out(action, summary, command, then=None):
        return {**base, "action": action, "summary": summary,
                "command": command, "then": list(then or [])}

    if ev in (_EVT_NEW_SOURCE_PENDING, _EVT_SCHEMA_DRIFT_PENDING):
        why = ("New source — review the proposed mapping"
               if ev == _EVT_NEW_SOURCE_PENDING
               else "Schema drift — re-map for the new schema")
        return out(
            ACT_APPROVE,
            f"{why}, then approve → promote → rerun.",
            f"{OPS} approve {approval_id} --mapping-id <MAPPING_ID> "
            f"--mapping-config-path <PATH>",
            then=[f"{OPS} promote {approval_id}", f"{OPS} rerun {pack_key}"])

    if ev == _EVT_INCOMPLETE_PACK_PENDING:
        return out(
            ACT_FIX_DATA,
            "Incomplete pack — upload the missing expected files, then re-fire the pack.",
            f"{OPS} rerun {pack_key}",
            then=[])

    if ev == _EVT_KNOWN_SOURCE_HALTED:
        if val_errors:
            return out(
                ACT_FIX_DATA,
                "Validation halted the run — fix the flagged data issues and rerun, "
                "or rerun with --force-publish as an explicit break-glass override.",
                f"{OPS} rerun {pack_key}",
                then=[f"{OPS} rerun {pack_key} --force-publish   # break-glass only"])

        blocking_decisions = hr.get("blocking_decisions") or []
        blocking_count = int(hr.get("blocking_decision_count") or 0)
        failed_gates = hr.get("failed_readiness_gates") or []
        not_ready = (bool(hr) and not hr.get("ready_for_transformation_validation"))

        # (1) ACTUAL mapping work — mapping recommendations to review OR registry
        # gaps. ONLY here do we say fix_mapping.
        if recs or gap > 0:
            what = (f"{len(recs)} mapping recommendation(s)" if recs
                    else f"{gap} registry gap(s)")
            return out(
                ACT_FIX_MAPPING,
                f"Onboarding handoff has {what} to resolve — review the "
                f"recommendations, resolve them, then rerun.",
                f"{OPS} show-recommendations {ref}",
                then=[f"{OPS} rerun {pack_key}"])

        # (2) Blocking operator DECISIONS that are not mapping (e.g. a missing
        # required run-context field like reporting_date). Name them explicitly.
        if blocking_decisions or blocking_count > 0:
            fields = [d.get("target_field") for d in blocking_decisions if d.get("target_field")]
            named = (", ".join(fields) if fields else "; ".join(failed_gates) or
                     f"{blocking_count} blocking decision(s)")
            return out(
                ACT_RESOLVE_DECISIONS,
                f"Onboarding handoff is not ready — {blocking_count or len(fields)} "
                f"blocking operator decision(s) to resolve: {named}. These are not "
                f"mapping recommendations; supply the required value/decision, then "
                f"rerun. Inspect the full readiness with show-handoff.",
                f"{OPS} show-handoff {ref}",
                then=[f"{OPS} rerun {pack_key}"])

        # (3) CONTRADICTORY: not ready, yet no registry gaps, no mapping
        # recommendations, and no blocking decisions. Say so explicitly — this is a
        # readiness metadata mismatch / a readiness flag that was not persisted, NOT
        # a mapping problem.
        if not_ready:
            detail = (f" Failed readiness gate(s): {'; '.join(failed_gates)}."
                      if failed_gates else "")
            return out(
                ACT_INVESTIGATE,
                "Onboarding handoff reports not ready, but there are no registry "
                "gaps, no mapping recommendations, and no blocking decisions to fix — "
                "this is a readiness metadata mismatch or a readiness flag that was "
                f"not persisted, not a mapping problem.{detail} Inspect the handoff.",
                f"{OPS} show-handoff {ref}",
                then=[f"{OPS} rerun {pack_key}"])

        # (4) No handoff readiness captured and nothing else actionable — inspect
        # and rerun (the safe default).
        return out(
            ACT_RERUN,
            "Run halted — inspect diagnostics and rerun once resolved.",
            f"{OPS} rerun {pack_key}", then=[])

    if ev == _EVT_FAILED or status == "failed":
        return out(
            ACT_INVESTIGATE,
            "Run failed — inspect run diagnostics / run_state.json.",
            f"{OPS} show {ref}", then=[])

    if status == "processed":
        return out(ACT_NONE, "Processed — no operator action required.", None)

    return out(ACT_NONE, "No operator action required.", None)
