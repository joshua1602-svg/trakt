"""apps.blob_trigger_app.source_registry — file-backed source registry.

A simple, deterministic registry of known sources (one record per
client/source_portfolio/dataset/frequency). File-backed (JSON or YAML) so it is
trivial to inspect now and to migrate to Azure Table / Cosmos later — the lookup
contract (:meth:`SourceRegistry.lookup`) stays the same.
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("trakt.blob_trigger.source_registry")

STATUS_ACTIVE = "active"
STATUS_PENDING_REVIEW = "pending_review"
STATUS_RETIRED = "retired"


@dataclass
class SourceRecord:
    client_id: str
    source_portfolio_id: str
    dataset: str
    frequency: str
    source_portfolio_type: Optional[str] = None
    source_system: Optional[str] = None          # lender / servicer where known
    approved_mapping_id: Optional[str] = None
    mapping_config_path: Optional[str] = None
    expected_schema_fingerprint: Optional[str] = None
    expected_columns: List[str] = field(default_factory=list)
    last_successful_run_id: Optional[str] = None
    last_successful_reporting_period: Optional[str] = None
    regime_required: bool = False                # funded books needing ESMA output
    status: str = STATUS_PENDING_REVIEW

    @property
    def key(self) -> str:
        return f"{self.client_id}/{self.source_portfolio_id}/{self.dataset}/{self.frequency}"

    @property
    def has_approved_mapping(self) -> bool:
        return bool(self.approved_mapping_id) and self.status == STATUS_ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class SourceRegistry:
    """File- or Blob-backed source registry.

    With no ``storage``, reads/writes a local filesystem path (unchanged
    behaviour). With a ``storage`` backend, ``path`` may be a ``blob://`` URI (or
    any storage URI) and load/save go through the storage abstraction — so the
    registry persists durably in Azure Blob in production.
    """

    def __init__(self, path: str | Path, *, storage: Any = None):
        self.storage = storage
        # Keep the original URI/string when storage-backed (blob:// is not a Path).
        self.uri = str(path)
        self.path = Path(path) if storage is None else None
        self._records: Dict[str, SourceRecord] = {}
        self.load()

    def _is_yaml(self) -> bool:
        return self.uri.lower().rsplit(".", 1)[-1] in ("yaml", "yml")

    # -- persistence -------------------------------------------------------- #
    def load(self) -> None:
        self._records = {}
        if self.storage is not None:
            if not self.storage.exists(self.uri):
                return
            raw = self.storage.read_text(self.uri)
        else:
            if not self.path.exists():
                return
            raw = self.path.read_text(encoding="utf-8")
        if not raw.strip():
            return
        if self._is_yaml():
            import yaml
            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw)
        for rec in (data.get("sources") or []):
            r = SourceRecord.from_dict(rec)
            self._records[r.key] = r

    def save(self) -> None:
        try:
            data = {"sources": [r.to_dict() for r in self._records.values()]}
            if self._is_yaml():
                import yaml
                text = yaml.safe_dump(data, sort_keys=True)
            else:
                text = json.dumps(data, indent=2, sort_keys=True)
            if self.storage is not None:
                self.storage.write_text(self.uri, text)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(text, encoding="utf-8")
        except Exception:  # noqa: BLE001 — log uri + traceback, re-raise
            logger.error("REGISTRY SAVE FAILED uri=%s storage=%s\n%s",
                         self.uri, type(self.storage).__name__ if self.storage else "filesystem",
                         traceback.format_exc())
            raise

    # -- lookup / mutation -------------------------------------------------- #
    def lookup(self, client_id: str, source_portfolio_id: str,
               dataset: str, frequency: str) -> Optional[SourceRecord]:
        return self._records.get(
            f"{client_id}/{source_portfolio_id}/{dataset}/{frequency}")

    def seen_client(self, client_id: str) -> bool:
        return any(r.client_id == client_id for r in self._records.values())

    def seen_portfolio(self, client_id: str, source_portfolio_id: str) -> bool:
        return any(r.client_id == client_id and r.source_portfolio_id == source_portfolio_id
                   for r in self._records.values())

    def upsert(self, record: SourceRecord) -> SourceRecord:
        self._records[record.key] = record
        self.save()
        return record

    def records(self) -> List[SourceRecord]:
        return list(self._records.values())
