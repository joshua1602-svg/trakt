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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import approvals as _approvals
from .layout import Layout
from .source_registry import SourceRegistry
from .storage import Storage


@dataclass
class ProductionPersistence:
    storage: Storage
    layout: Layout

    # -- registry ---------------------------------------------------------- #
    def load_registry(self) -> SourceRegistry:
        return SourceRegistry(self.layout.registry_uri, storage=self.storage)

    # -- event manifests --------------------------------------------------- #
    def persist_event_manifest(self, manifest: Dict[str, Any]) -> str:
        uri = self.layout.event_uri(manifest["event_id"])
        self.storage.write_text(uri, json.dumps(manifest, indent=2, default=str))
        return uri

    # -- approvals --------------------------------------------------------- #
    def write_pending_approval(self, **kw) -> Dict[str, Any]:
        return _approvals.write_pending(self.storage, self.layout, **kw)

    # -- accepted per-portfolio canonical ---------------------------------- #
    def persist_accepted(self, client_id: str, source_portfolio_id: str,
                         local_path: str) -> Optional[str]:
        if not local_path or not Path(local_path).exists():
            return None
        return self.storage.upload_file(
            local_path, self.layout.accepted_uri(client_id, source_portfolio_id))

    # -- central platform canonical (latest + period) ---------------------- #
    def persist_platform(self, client_id: str, period: str,
                         local_path: str) -> Dict[str, Optional[str]]:
        if not local_path or not Path(local_path).exists():
            return {"latest": None, "period": None}
        latest = self.storage.upload_file(
            local_path, self.layout.platform_latest_uri(client_id))
        period_uri = self.storage.upload_file(
            local_path, self.layout.platform_period_uri(client_id, period))
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
        for f in sorted(base.rglob("*")):
            if f.is_file():
                rel = f.relative_to(base).as_posix()
                out.append(self.storage.upload_file(str(f), f"{prefix}/{rel}"))
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
