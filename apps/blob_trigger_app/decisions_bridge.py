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
_INPUT_NAMES = (_DECISIONS_NAME, _RECS_NAME, _COVERAGE_JSON)


def _project_dir_from_manifest(manifest: Dict[str, Any]) -> Optional[Path]:
    """The onboarding project dir = the parent of the 28a coverage matrix path."""
    hr = ((manifest.get("orchestrator_diagnostics") or {}).get("handoff_readiness") or {})
    cov = hr.get("target_coverage_matrix_path")
    if not cov:
        return None
    p = Path(cov).parent
    return p if p.exists() else None


def persist_decision_inputs(storage: Storage, layout: Layout,
                            manifest: Dict[str, Any]) -> Dict[str, str]:
    """Copy the onboarding decision artefacts (34 pending / 36 recs / 28a) into
    ``trakt-state/runs/{pack_key}/onboarding/`` so ``approve-recommendations`` can
    run accept_target_advice after the Azure run scratch is reclaimed."""
    pack_key = manifest.get("pack_key")
    pdir = _project_dir_from_manifest(manifest)
    if not pack_key or pdir is None:
        return {}
    out: Dict[str, str] = {}
    for name in _INPUT_NAMES:
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
