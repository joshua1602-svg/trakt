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
    blocking = diag.get("blocking_decisions") or []
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
        if gap or blocking:
            return out(
                ACT_FIX_MAPPING,
                "Onboarding handoff is incomplete (unresolved mapping decisions / "
                "registry gaps) — review the recommendations, resolve them, then rerun.",
                f"{OPS} show-recommendations {ref}",
                then=[f"{OPS} rerun {pack_key}"])
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
