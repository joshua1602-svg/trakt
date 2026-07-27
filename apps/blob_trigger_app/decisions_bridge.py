"""apps.blob_trigger_app.decisions_bridge — the CLI approve→rerun→promote bridge.

Replicates the Codespaces CLI loop in the headless Azure path:

    onboarding (LLM target advisor on) → 34_target_first_decisions.yaml (pending)
      + 36_target_first_llm_recommendations.json (advisory)
    → operator ACCEPTS the advised recs (accept_target_advice) → an APPROVED
      34_target_first_decisions_approved.yaml
    → rerun onboarding applying that decisions file (deterministic-apply)
    → promote it as the source's active mapping → future monthly packs run
      deterministically with NO LLM.

The accepted decisions are **never auto-applied**: they are written, persisted,
and only take effect on an explicit ``ops rerun``. Deterministic mapping/registry
remains the production source of truth; the LLM advisor only proposes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .layout import Layout
from .storage import Storage

# The onboarding decision artefacts accept_target_advice consumes, and the
# accepted output it produces.
_DECISIONS_NAME = "34_target_first_decisions.yaml"
_RECS_NAME = "36_target_first_llm_recommendations.json"
_COVERAGE_JSON = "28a_target_coverage_matrix.json"
_APPROVED_NAME = "34_target_first_decisions_approved.yaml"
#: Source-value normalisation candidates + approved rules for the pack.
_SOURCE_VALUE_RULES_NAME = "28d_source_value_rules.json"
_INPUT_NAMES = (_DECISIONS_NAME, _RECS_NAME, _COVERAGE_JSON)
# Additionally captured so the RESOLVED source→canonical mapping (from the LLM
# mapping resolver) is durably preserved alongside the target-first decisions —
# the operator can inspect it, and nothing the resolver produced is lost when the
# ephemeral run scratch is reclaimed. Persisted; not required by accept_target_advice.
_RESOLVER_NAMES = (
    "22_llm_mapping_suggestions.json",
    "31_llm_mapping_review.json",
    "31_llm_mapping_review.csv",
    "28a_target_coverage_matrix.csv",
    # Pending source-value normalisation candidates + the rules already in force.
    # Persisted so `ops approve-source-value` can read them after the ephemeral
    # run scratch is reclaimed.
    _SOURCE_VALUE_RULES_NAME,
)
_PERSIST_NAMES = tuple(dict.fromkeys(_INPUT_NAMES + _RESOLVER_NAMES))


def _project_dir_from_manifest(manifest: Dict[str, Any]) -> Optional[Path]:
    """Locate the onboarding project dir for this pack's source.

    Prefers the invoker-supplied ``onboarding_project_dirs`` (populated on ANY
    outcome, incl. a SUCCESSFUL new-source run), keyed by source_portfolio_id;
    falls back to the parent of the 28a coverage matrix path from the halt
    diagnostics (which only exist on a non-done run).
    """
    dirs = manifest.get("onboarding_project_dirs") or {}
    if isinstance(dirs, dict) and dirs:
        pid = manifest.get("source_portfolio_id")
        candidate = dirs.get(pid) if pid else None
        for d in ([candidate] if candidate else []) + list(dirs.values()):
            if d and Path(d).exists():
                return Path(d)
    hr = ((manifest.get("orchestrator_diagnostics") or {}).get("handoff_readiness") or {})
    cov = hr.get("target_coverage_matrix_path")
    if cov and Path(cov).parent.exists():
        return Path(cov).parent
    return None


def persist_decision_inputs(storage: Storage, layout: Layout,
                            manifest: Dict[str, Any]) -> Dict[str, str]:
    """Copy the onboarding decision artefacts (34 pending / 36 recs / 28a) AND the
    resolved-mapping artefacts (22/31) into ``trakt-state/runs/{pack_key}/
    onboarding/`` so ``approve-recommendations`` can run accept_target_advice —
    and the resolved mapping is reusable — after the Azure run scratch is
    reclaimed. Runs on a SUCCESSFUL new-source onboarding run too, not only a halt."""
    pack_key = manifest.get("pack_key")
    pdir = _project_dir_from_manifest(manifest)
    if not pack_key or pdir is None:
        return {}
    out: Dict[str, str] = {}
    for name in _PERSIST_NAMES:
        src = pdir / name
        if src.exists():
            uri = layout.run_onboarding_uri(pack_key, name)
            storage.upload_file(str(src), uri)
            out[name] = uri
    return out


def approve_recommendations(storage: Storage, layout: Layout, pack_key: str, *,
                            approved_by: str = "", min_confidence: float = 0.0,
                            now: Optional[str] = None) -> Dict[str, Any]:
    """Accept the advised LLM recommendations into an APPROVED decisions file.

    Downloads the persisted 34/36/28a for the pack, runs the same
    ``accept_target_advice`` the CLI uses, uploads the approved
    ``34_target_first_decisions_approved.yaml`` to trakt-state, and returns the
    acceptance summary + the approved-decisions URI. Never auto-applies.
    """
    from engine.onboarding_agent import accept_target_advice as _ata

    have = {n: layout.run_onboarding_uri(pack_key, n) for n in _INPUT_NAMES}
    missing = [n for n, uri in have.items() if not storage.exists(uri)]
    if _DECISIONS_NAME in missing or _RECS_NAME in missing:
        return {"error": f"onboarding decision inputs not found for {pack_key} "
                         f"(missing: {missing}). Run onboarding with the LLM advisor first.",
                "approved": 0, "pending": 0, "skipped": []}

    tmp = Path(tempfile.mkdtemp(prefix="ops_accept_"))
    for name, uri in have.items():
        if storage.exists(uri):
            storage.download_file(uri, tmp / name)

    summary = _ata.accept_target_advice(
        tmp, out_path=tmp / _APPROVED_NAME, approved_by=(approved_by or "ops"),
        min_confidence=min_confidence, now=now)
    if summary.get("error"):
        return summary

    approved_uri = layout.run_onboarding_uri(pack_key, _APPROVED_NAME)
    storage.upload_file(str(tmp / _APPROVED_NAME), approved_uri)
    summary["approved_decisions_uri"] = approved_uri
    return summary


def source_value_rules_uri(layout: Layout, pack_key: str) -> str:
    return layout.run_onboarding_uri(pack_key, _SOURCE_VALUE_RULES_NAME)


def load_source_value_rules(storage: Storage, layout: Layout,
                            pack_key: str) -> Dict[str, Any]:
    """The pack's persisted 28d document (candidates + rules already in force)."""
    import json

    uri = source_value_rules_uri(layout, pack_key)
    if not storage.exists(uri):
        return {}
    try:
        return json.loads(storage.read_text(uri)) or {}
    except Exception:  # noqa: BLE001 — a malformed artefact is "no candidates"
        return {}


def approve_source_value(
    storage: Storage, layout: Layout, pack_key: str, *,
    canonical_field: str, source_column: str, source_value: str,
    action: str = "", approved_by: str = "", now: Optional[str] = None,
    scope: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Approve ONE detected source-value normalisation for this pack.

    Reads the pack's pending candidates, matches exactly one (failing closed
    otherwise — see :func:`source_value_rules.match_candidate`), and merges the
    approved decision into the pack-scoped
    ``34_target_first_decisions_approved.yaml`` using the existing generic
    ``treat_source_value_as_null`` action.

    No mapping id is involved: a source-value treatment is not a source mapping,
    and inventing one would put a fabricated mapping into the registry on
    promotion.

    Returns the approval summary; raises
    :class:`source_value_rules.CandidateMatchError` when the request does not
    identify exactly one in-scope candidate.
    """
    import yaml as _yaml

    from engine.onboarding_agent import source_value_rules as _svr

    doc = load_source_value_rules(storage, layout, pack_key)
    if not doc:
        raise _svr.CandidateMatchError(
            f"no source-value candidates recorded for {pack_key} at "
            f"{source_value_rules_uri(layout, pack_key)} — run onboarding for "
            f"this pack first")

    candidate = _svr.match_candidate(
        doc.get("candidates_pending", []) or [],
        canonical_field=canonical_field, source_column=source_column,
        source_value=source_value, action=(action or _svr.ACTION_TREAT_AS_NULL),
        scope=scope)
    decision = _svr.decision_from_candidate(
        candidate, approved_by=approved_by, approved_at=(now or ""))

    # Merge into the pack's approved decisions, preserving anything already
    # accepted there (mapping recommendations included — this must not disturb
    # the existing approval path).
    uri = approved_decisions_uri(layout, pack_key)
    approved_doc: Dict[str, Any] = {}
    if storage.exists(uri):
        try:
            approved_doc = _yaml.safe_load(storage.read_text(uri)) or {}
        except Exception:  # noqa: BLE001
            approved_doc = {}
    if not isinstance(approved_doc, dict):
        approved_doc = {}
    approved_doc.setdefault("version", 1)
    approved_doc.setdefault("client_id", candidate.get("client_id", ""))
    approved_doc.setdefault("mode", doc.get("mode", ""))
    approved_doc.setdefault("supported_actions", list(_SUPPORTED_ACTIONS()))
    decisions = list(approved_doc.get("decisions", []) or [])

    replaced = False
    for i, existing in enumerate(decisions):
        if not isinstance(existing, dict):
            continue
        if (existing.get("decision_type") == _svr.DECISION_TYPE
                and str(existing.get("canonical_field", "")).strip().lower()
                == str(decision["canonical_field"]).strip().lower()
                and str(existing.get("source_column", "")).strip().lower()
                == str(decision["source_column"]).strip().lower()
                and str(existing.get("source_value", "")).strip().lower()
                == str(decision["source_value"]).strip().lower()):
            decisions[i] = decision       # re-approving is idempotent
            replaced = True
            break
    if not replaced:
        decisions.append(decision)
    approved_doc["decisions"] = decisions

    storage.write_text(uri, _yaml.safe_dump(approved_doc, sort_keys=False))
    return {
        "pack_key": pack_key,
        "approved_decisions_uri": uri,
        "decision": decision,
        "candidate": candidate,
        "already_approved": replaced,
        "decision_count": len(decisions),
    }


def _SUPPORTED_ACTIONS():
    from engine.onboarding_agent import target_first_decisions as _tfd
    return _tfd.SUPPORTED_ACTIONS


def approved_decisions_uri(layout: Layout, pack_key: str) -> str:
    return layout.run_onboarding_uri(pack_key, _APPROVED_NAME)


def has_approved_decisions(storage: Storage, layout: Layout, pack_key: str) -> bool:
    return storage.exists(approved_decisions_uri(layout, pack_key))


def localise_approved_decisions(storage: Storage, layout: Layout, pack_key: str,
                                dest_dir: str) -> Optional[str]:
    """Download the accepted decisions file to a local path (for a rerun that
    applies them). Returns the local path, or None if none accepted yet."""
    uri = approved_decisions_uri(layout, pack_key)
    if not storage.exists(uri):
        return None
    dest = Path(dest_dir) / _APPROVED_NAME
    storage.download_file(uri, dest)
    return str(dest)
