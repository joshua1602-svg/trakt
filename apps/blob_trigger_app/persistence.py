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
            return _run_records.write_run_record(self.storage, self.layout, record)
        except Exception:
            _persist_step("persist_run_record", uri)
            raise

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
