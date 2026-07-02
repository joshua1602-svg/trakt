"""apps.blob_trigger_app.persistence — durable artifact persistence facade.

Bundles a :class:`~apps.blob_trigger_app.storage.Storage` backend and a
:class:`~apps.blob_trigger_app.layout.Layout` and exposes the production
persistence operations the router needs: persist event manifests, pending
approvals, accepted per-portfolio canonicals, central platform canonicals
(latest + period), and regime outputs. Also the MI/Regime **locators**.

When the router is given a :class:`ProductionPersistence`, final artifacts are
uploaded to the durable store (Blob in Azure, filesystem locally/tests). When it
is ``None``, the router keeps its prior local-scratch-only behaviour, so existing
tests are unaffected.
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import approvals as _approvals
from . import run_records as _run_records
from .layout import Layout
from .source_registry import SourceRegistry
from .storage import Storage

logger = logging.getLogger("trakt.blob_trigger.persistence")


def _persist_step(op: str, uri: Optional[str] = None):
    """Log the full traceback + target URI of a failing persistence op, re-raise."""
    logger.error("PERSISTENCE FAILED op=%s uri=%s\n%s", op, uri, traceback.format_exc())


def _head_lines(path: str, n: int) -> List[str]:
    """First ``n`` lines of a text file (bounded canonical preview)."""
    out: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            out.append(line)
    return out


@dataclass
class ProductionPersistence:
    storage: Storage
    layout: Layout

    # -- registry ---------------------------------------------------------- #
    def load_registry(self) -> SourceRegistry:
        uri = self.layout.registry_uri
        try:
            return SourceRegistry(uri, storage=self.storage)
        except Exception:
            _persist_step("load_registry", uri)
            raise

    # -- event manifests --------------------------------------------------- #
    def persist_event_manifest(self, manifest: Dict[str, Any]) -> str:
        uri = self.layout.event_uri(manifest["event_id"])
        try:
            self.storage.write_text(uri, json.dumps(manifest, indent=2, default=str))
        except Exception:
            _persist_step("persist_event_manifest", uri)
            raise
        return uri

    # -- operator run ledger ----------------------------------------------- #
    def persist_run_record(self, manifest: Dict[str, Any]) -> Optional[str]:
        """Durably record a terminal run outcome for the operator CLI, including a
        durable copy of the onboarding handoff manifest + target coverage matrix so
        ops can inspect them after the Azure run scratch is reclaimed."""
        if not manifest.get("is_pack_marker"):
            return None
        if manifest.get("status") not in _run_records.OPERATOR_STATUSES:
            return None
        uri = None
        try:
            uri = self.layout.run_uri(manifest.get("pack_key") or "unknown")
            record = _run_records.build_run_record(manifest)
            record["handoff_artifacts"] = self._persist_handoff_artifacts(manifest)
            record["transform_artifacts"] = self._persist_transform_artifacts(manifest)
            # Generic per-gate diagnostics + artefacts (all gates).
            record["gate_artifacts"] = self._persist_gate_diagnostics(manifest)
            # Onboarding decision artefacts (34/36/28a) for the approve→rerun bridge.
            try:
                from . import decisions_bridge as _db
                record["onboarding_decision_inputs"] = _db.persist_decision_inputs(
                    self.storage, self.layout, manifest)
            except Exception as exc:  # noqa: BLE001 — never fail the event on this
                manifest.setdefault("persist_errors", []).append(f"decision_inputs: {exc}")
            return _run_records.write_run_record(self.storage, self.layout, record)
        except Exception:
            _persist_step("persist_run_record", uri)
            raise

    # -- generic per-gate diagnostics + artefacts -------------------------- #
    def _persist_gate_diagnostics(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Write each gate's standard diagnostics + its source artefacts under
        ``trakt-state/runs/{pack_key}/gates/{gate}/`` (diagnostics.json + a preview
        of any produced canonical). Enriches each gate's persisted_artifact_uris."""
        pack_key = manifest.get("pack_key")
        gates = (manifest.get("orchestrator_diagnostics") or {}).get("gates") or []
        if not pack_key or not gates:
            return {}
        out: Dict[str, Any] = {}
        for g in gates:
            name = g.get("gate_name") or "gate"
            uris: Dict[str, Any] = {}
            # Copy the gate's source artefacts FIRST so their URIs are recorded in
            # the diagnostics.json we write below.
            for label, path in (g.get("source_artifact_paths") or {}).items():
                if not path or label == "transformed_canonical_path":
                    continue
                try:
                    if Path(path).exists():
                        auri = self.layout.gate_artifact_uri(pack_key, name, Path(path).name)
                        self.storage.upload_file(str(path), auri)
                        uris[label] = auri
                except Exception as exc:  # noqa: BLE001
                    manifest.setdefault("persist_errors", []).append(f"gate_art:{name}: {exc}")
            # typed/canonical output preview (first 20 rows) if the gate produced one.
            canon = (g.get("source_artifact_paths") or {}).get("transformed_canonical_path")
            if canon and Path(str(canon)).exists():
                try:
                    preview = "".join(_head_lines(str(canon), 21))
                    puri = self.layout.gate_artifact_uri(pack_key, name, "canonical_preview.csv")
                    self.storage.write_text(puri, preview)
                    uris["canonical_preview"] = puri
                except Exception as exc:  # noqa: BLE001
                    manifest.setdefault("persist_errors", []).append(f"gate_preview:{name}: {exc}")
            g.setdefault("persisted_artifact_uris", {}).update(uris)
            # Write the standard diagnostics.json LAST (now carrying artefact URIs).
            try:
                duri = self.layout.gate_diagnostics_uri(pack_key, name)
                self.storage.write_text(duri, json.dumps(g, indent=2, default=str))
                uris["diagnostics"] = duri
            except Exception as exc:  # noqa: BLE001
                manifest.setdefault("persist_errors", []).append(f"gate_diag:{name}: {exc}")
            out[name] = uris
        return out

    def load_gate_diagnostics(self, pack_key: str, gate_name: str) -> Optional[Dict[str, Any]]:
        uri = self.layout.gate_diagnostics_uri(pack_key, gate_name)
        if not self.storage.exists(uri):
            return None
        try:
            return json.loads(self.storage.read_text(uri))
        except Exception:  # noqa: BLE001
            return None

    def list_gate_names(self, pack_key: str) -> List[str]:
        names: List[str] = []
        for u in self.storage.list(self.layout.gates_prefix(pack_key)):
            if u.endswith("/diagnostics.json"):
                names.append(u.rsplit("/", 2)[-2])
        return sorted(set(names))

    def gates_folder_exists(self, pack_key: str) -> bool:
        return bool(self.storage.list(self.layout.gates_prefix(pack_key)))

    # -- LLM advisory recommendations -------------------------------------- #
    def persist_llm_recommendations(self, pack_key: str,
                                    recommendations: List[Dict[str, Any]],
                                    meta: Dict[str, Any], now: str) -> str:
        from . import llm_recommendations as _llm
        doc = _llm.build_recommendations_doc(recommendations, meta, pack_key=pack_key, now=now)
        uri = self.layout.llm_recommendations_uri(pack_key)
        self.storage.write_text(uri, json.dumps(doc, indent=2, default=str))
        return uri

    def load_llm_recommendations(self, pack_key: str) -> Optional[Dict[str, Any]]:
        uri = self.layout.llm_recommendations_uri(pack_key)
        if not self.storage.exists(uri):
            return None
        try:
            return json.loads(self.storage.read_text(uri))
        except Exception:  # noqa: BLE001
            return None

    # -- governance artifact (auto-approve audit trail) -------------------- #
    def persist_governance_artifact(self, pack_key: str,
                                    doc: Dict[str, Any]) -> str:
        """Write the auto-approval governance artifact (materiality evidence +
        old→new fingerprint + re-pin outcome) durably. Returns the URI."""
        uri = self.layout.governance_uri(pack_key)
        self.storage.write_text(uri, json.dumps(doc, indent=2, default=str))
        return uri

    def load_governance_artifact(self, pack_key: str) -> Optional[Dict[str, Any]]:
        uri = self.layout.governance_uri(pack_key)
        if not self.storage.exists(uri):
            return None
        try:
            return json.loads(self.storage.read_text(uri))
        except Exception:  # noqa: BLE001
            return None

    # -- accepted-decisions bridge (CLI approve→rerun→promote) ------------- #
    def approve_recommendations(self, pack_key: str, **kw) -> Dict[str, Any]:
        from . import decisions_bridge as _db
        return _db.approve_recommendations(self.storage, self.layout, pack_key, **kw)

    def has_approved_decisions(self, pack_key: str) -> bool:
        from . import decisions_bridge as _db
        return _db.has_approved_decisions(self.storage, self.layout, pack_key)

    def approved_decisions_uri(self, pack_key: str) -> str:
        from . import decisions_bridge as _db
        return _db.approved_decisions_uri(self.layout, pack_key)

    def localise_approved_decisions(self, pack_key: str, dest_dir: str) -> Optional[str]:
        from . import decisions_bridge as _db
        return _db.localise_approved_decisions(self.storage, self.layout, pack_key, dest_dir)

    def _persist_transform_artifacts(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Copy the Gate 2 (transform) output artefacts — the transformation
        manifest + issues — into trakt-state/runs/{pack_key}/ so ``ops
        show-transform`` works after the Azure run scratch is reclaimed."""
        pack_key = manifest.get("pack_key")
        tr = ((manifest.get("orchestrator_diagnostics") or {})
              .get("transform_readiness") or {})
        if not pack_key or not tr:
            return {}
        artifacts: Dict[str, Any] = {}
        tm = tr.get("transformation_manifest")
        if tm:
            try:
                out = self.layout.run_artifact_uri(pack_key, "transformation_manifest.json")
                self.storage.write_text(out, json.dumps(tm, indent=2, default=str))
                artifacts["transformation_manifest_uri"] = out
            except Exception as exc:  # noqa: BLE001 — never fail the event on this
                manifest.setdefault("persist_errors", []).append(f"transform_manifest: {exc}")
        for key, name in (("transformation_issues_json", "transformation_issues.json"),
                          ("transformation_issues_csv", "transformation_issues.csv")):
            src = tr.get(key)
            try:
                if src and Path(src).exists():
                    out = self.layout.run_artifact_uri(pack_key, name)
                    self.storage.upload_file(str(src), out)
                    artifacts.setdefault("transformation_issues_uris", []).append(out)
            except Exception as exc:  # noqa: BLE001
                manifest.setdefault("persist_errors", []).append(f"transform_issues: {exc}")
        return artifacts

    def _persist_handoff_artifacts(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Copy the onboarding handoff manifest + target coverage matrix into
        trakt-state (durable) so ``ops show-handoff`` works after scratch is gone."""
        pack_key = manifest.get("pack_key")
        hr = ((manifest.get("orchestrator_diagnostics") or {})
              .get("handoff_readiness") or {})
        if not pack_key or not hr:
            return {}
        artifacts: Dict[str, Any] = {}
        hm = hr.get("handoff_manifest")
        if hm:
            try:
                out = self.layout.run_artifact_uri(pack_key, "handoff_manifest.json")
                self.storage.write_text(out, json.dumps(hm, indent=2, default=str))
                artifacts["handoff_manifest_uri"] = out
            except Exception as exc:  # noqa: BLE001 — never fail the event on this
                manifest.setdefault("persist_errors", []).append(f"handoff_manifest: {exc}")
        cov = hr.get("target_coverage_matrix_path")
        if cov:
            base = Path(cov)
            for src, name in ((base, "target_coverage_matrix.csv"),
                              (base.with_suffix(".json"), "target_coverage_matrix.json")):
                try:
                    if src.exists():
                        out = self.layout.run_artifact_uri(pack_key, name)
                        self.storage.upload_file(str(src), out)
                        artifacts.setdefault("target_coverage_matrix_uris", []).append(out)
                except Exception as exc:  # noqa: BLE001
                    manifest.setdefault("persist_errors", []).append(f"coverage_matrix: {exc}")
        return artifacts

    def load_run_record(self, pack_key: str) -> Optional[Dict[str, Any]]:
        return _run_records.load_run_record(self.storage, self.layout, pack_key)

    def list_halted_runs(self) -> List[Dict[str, Any]]:
        return _run_records.list_halted(self.storage, self.layout)

    def resolve_run_record(self, ref: str) -> Optional[Dict[str, Any]]:
        return _run_records.resolve(self.storage, self.layout, ref)

    # -- approvals --------------------------------------------------------- #
    def write_pending_approval(self, **kw) -> Dict[str, Any]:
        uri = self.layout.approvals_prefix()
        try:
            return _approvals.write_pending(self.storage, self.layout, **kw)
        except Exception:
            _persist_step("write_pending_approval", uri)
            raise

    # -- accepted per-portfolio canonical ---------------------------------- #
    def persist_accepted(self, client_id: str, source_portfolio_id: str,
                         local_path: str) -> Optional[str]:
        if not local_path or not Path(local_path).exists():
            return None
        uri = self.layout.accepted_uri(client_id, source_portfolio_id)
        try:
            return self.storage.upload_file(local_path, uri)
        except Exception:
            _persist_step("persist_accepted", uri)
            raise

    # -- central platform canonical (latest + period) ---------------------- #
    def persist_platform(self, client_id: str, period: str,
                         local_path: str) -> Dict[str, Optional[str]]:
        if not local_path or not Path(local_path).exists():
            return {"latest": None, "period": None}
        uri = self.layout.platform_latest_uri(client_id)
        try:
            latest = self.storage.upload_file(local_path, uri)
            uri = self.layout.platform_period_uri(client_id, period)
            period_uri = self.storage.upload_file(local_path, uri)
        except Exception:
            _persist_step("persist_platform", uri)
            raise
        return {"latest": latest, "period": period_uri}

    # -- weekly pipeline snapshot (React MI pipeline view) ----------------- #
    def persist_pipeline(self, client_id: str, period: str,
                         local_path: str) -> Dict[str, Optional[str]]:
        """Publish a weekly pipeline extract to the durable store: the period copy,
        the stable ``latest`` copy the MI API reads, and a JSON pointer (mirrors
        analytics.blob_storage.register_latest_pipeline_snapshot's shape). Returns
        the written URIs."""
        if not local_path or not Path(local_path).exists():
            return {"latest": None, "period": None, "pointer": None}
        uri = self.layout.pipeline_latest_csv_uri(client_id)
        try:
            latest = self.storage.upload_file(local_path, uri)
            uri = self.layout.pipeline_period_csv_uri(client_id, period)
            period_uri = self.storage.upload_file(local_path, uri)
            uri = self.layout.pipeline_latest_pointer_uri(client_id)
            pointer = self.storage.write_text(uri, json.dumps({
                "blob_name": latest, "period": period,
                "source_file": latest, "registered_period": period,
            }, indent=2))
        except Exception:
            _persist_step("persist_pipeline", uri)
            raise
        return {"latest": latest, "period": period_uri, "pointer": pointer}

    def pipeline_latest_uri(self, client_id: str) -> str:
        return self.layout.pipeline_latest_csv_uri(client_id)

    def pipeline_latest_path(self, client_id: str,
                             download_to: Optional[str] = None) -> Optional[str]:
        """Resolve the latest pipeline extract to a local path (on-disk for the
        filesystem backend; a downloaded copy for Blob). ``None`` when absent."""
        uri = self.layout.pipeline_latest_csv_uri(client_id)
        if not self.storage.exists(uri):
            return None
        local = self.storage._local_path(uri)
        if Path(str(local)).exists():
            return str(local)
        if download_to:
            return str(self.storage.download_file(uri, download_to))
        return None

    # -- regime outputs ---------------------------------------------------- #
    def persist_regime_dir(self, client_id: str, period: str,
                           local_dir: str) -> List[str]:
        """Upload every file the regime projector wrote (template-clean ESMA
        output + provenance companion) under the regime prefix for this period."""
        base = Path(local_dir)
        if not base.exists():
            return []
        prefix = self.layout.regime_prefix(client_id, period)
        out: List[str] = []
        uri = prefix
        try:
            for f in sorted(base.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(base).as_posix()
                    uri = f"{prefix}/{rel}"
                    out.append(self.storage.upload_file(str(f), uri))
        except Exception:
            _persist_step("persist_regime_dir", uri)
            raise
        return out

    # -- MI locator -------------------------------------------------------- #
    def mi_latest_platform_uri(self, client_id: str) -> str:
        return self.layout.platform_latest_uri(client_id)

    def mi_latest_platform_path(self, client_id: str,
                                download_to: Optional[str] = None) -> Optional[str]:
        """Resolve the latest platform canonical for MI. Returns a local path:
        the on-disk path for filesystem storage, or a downloaded copy for Blob."""
        uri = self.mi_latest_platform_uri(client_id)
        if not self.storage.exists(uri):
            return None
        local = self.storage._local_path(uri)  # filesystem backend: real path
        if Path(str(local)).exists():
            return str(local)
        if download_to:
            return str(self.storage.download_file(uri, download_to))
        return None

    @classmethod
    def from_env(cls, storage: Storage) -> "ProductionPersistence":
        return cls(storage=storage, layout=Layout.from_env())
