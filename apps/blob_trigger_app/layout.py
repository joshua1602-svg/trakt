"""apps.blob_trigger_app.layout — persistent artifact layout (configurable).

Defines WHERE durable artifacts live (registry, approvals, event manifests,
accepted per-portfolio canonicals, central platform canonicals, regime outputs,
MI outputs). Container names and the registry URI are configurable via app
settings; nothing is hardcoded at call sites.

Default layout (all overridable):
    trakt-state/registry/source_registry.yaml          source registry
    trakt-state/approvals/{approval_id}.json           pending approvals
    trakt-state/events/{event_id}.json                 event manifests
    processed-v2/accepted/{client}/{pid}_canonical_typed.csv
    processed-v2/platform/{client}/latest/platform_canonical_typed.csv
    processed-v2/platform/{client}/{period}/platform_canonical_typed.csv
    processed-v2/regime/{client}/{period}/
    processed-v2/mi/{client}/
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .storage import BLOB_SCHEME, join_uri

PLATFORM_CANONICAL_NAME = "platform_canonical_typed.csv"


@dataclass(frozen=True)
class Layout:
    state_container: str = "trakt-state"
    processed_container: str = "processed-v2"
    raw_container: str = "raw-v2"
    registry_uri: str = "blob://trakt-state/registry/source_registry.yaml"

    @classmethod
    def from_env(cls) -> "Layout":
        state = os.environ.get("TRAKT_STATE_CONTAINER", "trakt-state")
        processed = os.environ.get("TRAKT_PROCESSED_CONTAINER", "processed-v2")
        raw = os.environ.get("TRAKT_RAW_CONTAINER", "raw-v2")
        registry = os.environ.get(
            "TRAKT_SOURCE_REGISTRY_URI",
            f"{BLOB_SCHEME}{state}/registry/source_registry.yaml")
        return cls(state_container=state, processed_container=processed,
                   raw_container=raw, registry_uri=registry)

    # -- state container --------------------------------------------------- #
    def _state(self, *parts: str) -> str:
        return join_uri(f"{BLOB_SCHEME}{self.state_container}", *parts)

    def approvals_prefix(self) -> str:
        return self._state("approvals")

    def approval_uri(self, approval_id: str) -> str:
        return self._state("approvals", f"{approval_id}.json")

    def events_prefix(self) -> str:
        return self._state("events")

    def event_uri(self, event_id: str) -> str:
        return self._state("events", f"{event_id}.json")

    def runs_prefix(self) -> str:
        return self._state("runs")

    def run_uri(self, pack_key: str) -> str:
        """Operator-facing run record — one per reporting pack (keyed on the
        durable pack_key so reruns update the same record)."""
        return self._state("runs", f"{pack_key}.json")

    # -- processed container ----------------------------------------------- #
    def _processed(self, *parts: str) -> str:
        return join_uri(f"{BLOB_SCHEME}{self.processed_container}", *parts)

    def accepted_uri(self, client_id: str, source_portfolio_id: str) -> str:
        return self._processed("accepted", client_id,
                               f"{source_portfolio_id}_canonical_typed.csv")

    def accepted_prefix(self, client_id: str) -> str:
        return self._processed("accepted", client_id)

    def platform_latest_uri(self, client_id: str) -> str:
        return self._processed("platform", client_id, "latest", PLATFORM_CANONICAL_NAME)

    def platform_latest_dir(self, client_id: str) -> str:
        return self._processed("platform", client_id, "latest")

    def platform_period_uri(self, client_id: str, period: str) -> str:
        return self._processed("platform", client_id, period, PLATFORM_CANONICAL_NAME)

    def regime_prefix(self, client_id: str, period: str) -> str:
        return self._processed("regime", client_id, period)

    def mi_prefix(self, client_id: str) -> str:
        return self._processed("mi", client_id)
