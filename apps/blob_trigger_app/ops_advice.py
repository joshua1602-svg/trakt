"""apps.blob_trigger_app.ops_advice — the "what do I do next?" advisory.

Turns an event manifest into a single, specific operator instruction. The action
is driven by the FAILED GATE (never a generic "rerun" when a gate has unresolved
issues), from the controlled vocabulary:

    inspect_onboarding · inspect_transform · inspect_validation ·
    inspect_assembler · inspect_projection · approve_mapping ·
    resolve_llm_recommendations · rerun · force_publish_break_glass · none

Pure (no storage / no Azure) so the router can embed it in every terminal
manifest and the ops CLI can reproduce it from a persisted record.
"""

from __future__ import annotations

from typing import Any, Dict

# Operator actions (the manifest ``next_action.action`` vocabulary).
ACT_INSPECT_ONBOARDING = "inspect_onboarding"
ACT_INSPECT_TRANSFORM = "inspect_transform"
ACT_INSPECT_VALIDATION = "inspect_validation"
ACT_INSPECT_ASSEMBLER = "inspect_assembler"
ACT_INSPECT_PROJECTION = "inspect_projection"
ACT_APPROVE_MAPPING = "approve_mapping"
ACT_RESOLVE_LLM = "resolve_llm_recommendations"
ACT_RERUN = "rerun"
ACT_FORCE_PUBLISH = "force_publish_break_glass"
ACT_NONE = "none"

# gate name → inspect action + the specific show command.
_GATE_ACTION = {
    "onboarding": (ACT_INSPECT_ONBOARDING, "show-handoff"),
    "transform": (ACT_INSPECT_TRANSFORM, "show-transform"),
    "validation": (ACT_INSPECT_VALIDATION, "show-validation"),
    "assembler": (ACT_INSPECT_ASSEMBLER, "show-gate"),
    "projection": (ACT_INSPECT_PROJECTION, "show-gate"),
    "stamp": (ACT_INSPECT_ONBOARDING, "show-gate"),
}

# Audit labels this module reasons over (kept local to avoid importing router).
_EVT_NEW_SOURCE_PENDING = "new_source_pending_review"
_EVT_SCHEMA_DRIFT_PENDING = "schema_drift_pending_review"
_EVT_INCOMPLETE_PACK_PENDING = "incomplete_pack_pending_review"
_EVT_KNOWN_SOURCE_HALTED = "known_source_halted"
_EVT_FAILED = "failed"

OPS = "python -m apps.blob_trigger_app.ops"


def next_operator_action(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the next operator step for a terminal event manifest. Includes the
    identifiers an operator needs; ``command`` is the exact next command; ``then``
    lists the commands that follow."""
    ev = manifest.get("event_decision")
    status = manifest.get("status")
    approval_id = manifest.get("approval_id")
    pack_key = manifest.get("pack_key")
    run_id = manifest.get("orchestrator_run_id")
    spid = manifest.get("source_portfolio_id")
    diag = manifest.get("orchestrator_diagnostics") or {}
    run_summary = diag.get("run_summary") or {}
    llm = manifest.get("llm") or {}
    ref = run_id or pack_key

    base = {"approval_id": approval_id, "run_id": run_id,
            "source_portfolio_id": spid, "pack_key": pack_key,
            "failed_gate": run_summary.get("failed_gate")}

    def out(action, summary, command, then=None):
        return {**base, "action": action, "summary": summary,
                "command": command, "then": list(then or [])}

    # ---- new source / schema drift → review + approve the mapping --------- #
    if ev in (_EVT_NEW_SOURCE_PENDING, _EVT_SCHEMA_DRIFT_PENDING):
        why = ("New source" if ev == _EVT_NEW_SOURCE_PENDING else "Schema drift")
        if llm.get("recommendations_present"):
            return out(
                ACT_RESOLVE_LLM,
                f"{why} — LLM produced ADVISORY mapping recommendations (deterministic "
                f"registry remains the source of truth). Review them, then approve → "
                f"promote → rerun.",
                f"{OPS} show-llm {ref}",
                then=[f"{OPS} approve {approval_id} --mapping-id <MAPPING_ID> "
                      f"--mapping-config-path <PATH>",
                      f"{OPS} promote {approval_id}", f"{OPS} rerun {pack_key}"])
        return out(
            ACT_APPROVE_MAPPING,
            f"{why} — review the proposed mapping, then approve → promote → rerun.",
            f"{OPS} approve {approval_id} --mapping-id <MAPPING_ID> "
            f"--mapping-config-path <PATH>",
            then=[f"{OPS} promote {approval_id}", f"{OPS} rerun {pack_key}"])

    if ev == _EVT_INCOMPLETE_PACK_PENDING:
        return out(
            ACT_RERUN,
            "Incomplete pack — upload the missing expected files, then re-fire the pack.",
            f"{OPS} rerun {pack_key}", then=[])

    # ---- a gate halted/failed → inspect THAT gate specifically ------------ #
    failed_gate = run_summary.get("failed_gate")
    if ev == _EVT_KNOWN_SOURCE_HALTED or status in ("halted", "failed"):
        if failed_gate == "transform":
            return _transform_action(out, diag, ref, pack_key)
        if failed_gate == "validation":
            return _validation_action(out, diag, ref, pack_key)
        if failed_gate in ("assembler", "projection"):
            act, _ = _GATE_ACTION[failed_gate]
            return out(
                act,
                f"Gate '{failed_gate}' failed — inspect its diagnostics, resolve the "
                f"issue, then rerun.",
                f"{OPS} show-gate {ref} {failed_gate}",
                then=[f"{OPS} rerun {pack_key}"])
        if failed_gate in ("onboarding", "stamp"):
            return _onboarding_action(out, diag, llm, ref, pack_key)

    if status == "processed":
        return out(ACT_NONE, "Processed — no operator action required.", None)

    # No gate pinned — inspect the run then rerun (not treated as a gate issue).
    return out(ACT_RERUN, "Run did not complete — inspect the run, then rerun.",
               f"{OPS} show {ref}", then=[f"{OPS} rerun {pack_key}"])


def _transform_action(out, diag, ref, pack_key):
    tr = diag.get("transform_readiness") or {}
    n_issues = int(tr.get("issue_count") or 0)
    n_block = int(tr.get("blocking_issue_count") or 0)
    fields = tr.get("affected_fields") or []
    named = ", ".join(fields[:8]) + ("…" if len(fields) > 8 else "")
    return out(
        ACT_INSPECT_TRANSFORM,
        f"Gate 2 transform is not ready_for_validation — {n_issues} issue(s), "
        f"{n_block} blocking; affected fields: {named or 'n/a'}. The onboarding "
        f"handoff was ready, so this is a transformation/data-quality problem, not "
        f"mapping. Inspect the transform issues, fix the data/derivations, then rerun.",
        f"{OPS} show-transform {ref}",
        then=[f"{OPS} rerun {pack_key}"])


def _validation_action(out, diag, ref, pack_key):
    vr = diag.get("validation_readiness") or {}
    n_issues = int(vr.get("issue_count") or 0)
    n_block = int(vr.get("blocking_issue_count") or 0)
    mand = vr.get("mandatory_field_failures") or []
    parse = (vr.get("numeric_parse_failures") or []) + (vr.get("date_parse_failures") or [])
    detail = []
    if mand:
        detail.append(f"mandatory-field failures: {', '.join(mand[:6])}")
    if parse:
        detail.append(f"parse failures: {', '.join(parse[:6])}")
    return out(
        ACT_INSPECT_VALIDATION,
        f"Gate 3 validation is not ready_for_publish — {n_issues} issue(s), "
        f"{n_block} blocking. " + ("; ".join(detail) + ". " if detail else "")
        + "Fix the flagged data/derivations and rerun; --force-publish is an "
        "explicit break-glass only.",
        f"{OPS} show-validation {ref}",
        then=[f"{OPS} rerun {pack_key}",
              f"{OPS} rerun {pack_key} --force-publish   # break-glass only"])


def _onboarding_action(out, diag, llm, ref, pack_key):
    hr = diag.get("handoff_readiness") or {}
    gap = int(diag.get("registry_gap_count") or 0)
    recs = diag.get("mapping_recommendations") or []
    blocking = hr.get("blocking_decisions") or []
    failed_gates = hr.get("failed_readiness_gates") or []
    # LLM advisory recs available → resolve them first (still operator-approved).
    if llm.get("recommendations_present"):
        return out(
            ACT_RESOLVE_LLM,
            "Onboarding handoff is not ready and LLM produced ADVISORY mapping "
            "recommendations (deterministic registry remains the source of truth). "
            "Review + approve/reject them, then rerun.",
            f"{OPS} show-llm {ref}", then=[f"{OPS} rerun {pack_key}"])
    if recs or gap > 0:
        what = (f"{len(recs)} mapping recommendation(s)" if recs else f"{gap} registry gap(s)")
        return out(
            ACT_INSPECT_ONBOARDING,
            f"Onboarding handoff has {what} to resolve — review, resolve, then rerun.",
            f"{OPS} show-handoff {ref}", then=[f"{OPS} rerun {pack_key}"])
    if blocking:
        fields = [d.get("target_field") for d in blocking if d.get("target_field")]
        return out(
            ACT_INSPECT_ONBOARDING,
            f"Onboarding handoff is not ready — {len(fields) or len(blocking)} blocking "
            f"operator decision(s) to resolve: {', '.join(fields) or 'see show-handoff'}. "
            f"Supply the required value/decision, then rerun.",
            f"{OPS} show-handoff {ref}", then=[f"{OPS} rerun {pack_key}"])
    detail = (f" Failed readiness gate(s): {'; '.join(failed_gates)}." if failed_gates else "")
    return out(
        ACT_INSPECT_ONBOARDING,
        "Onboarding handoff reports not ready with no registry gaps, no mapping "
        "recommendations, and no blocking decisions — likely a readiness metadata "
        f"mismatch or a flag that was not persisted.{detail} Inspect the handoff.",
        f"{OPS} show-handoff {ref}", then=[f"{OPS} rerun {pack_key}"])
