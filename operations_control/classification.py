"""operations_control.classification — delivery classification.

Uses the EXISTING source registry and schema fingerprinting (the same machinery
the blob trigger routes with) to decide whether a delivery is:

  * a completely new client            -> new_client
  * a new/changed schema for a known client -> new_portfolio (secondary onboarding)
  * a recognised recurring delivery    -> recurring
  * an earlier period of a recognised source -> backfill

The suggestion is exactly that — the operator can override it before the
workflow starts, and both the suggestion and any override are persisted in the
audit record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from apps.blob_trigger_app.schema_fingerprint import fingerprint_pack
from apps.blob_trigger_app.source_registry import SourceRegistry

from .contracts import WF_BACKFILL, WF_NEW_CLIENT, WF_NEW_PORTFOLIO, WF_RECURRING
from .language import CLASSIFICATION_SENTENCES

DATA_EXTS = (".csv", ".xlsx", ".xls", ".xlsm")


@dataclass
class Classification:
    suggested: str
    fingerprint: str
    reason: str                      # internal (audit) — not rendered
    sentence: str                    # operator-facing explanation

    def to_dict(self):
        return {"suggested": self.suggested, "fingerprint": self.fingerprint,
                "reason": self.reason, "sentence": self.sentence}


def fingerprint_delivery(input_path: str) -> str:
    """Fingerprint a delivery directory (or single file) using the existing
    structural pack fingerprint — headers only, never values."""
    p = Path(input_path)
    files = ([p] if p.is_file()
             else sorted(f for f in p.rglob("*")
                         if f.is_file() and f.suffix.lower() in DATA_EXTS))
    if not files:
        return ""
    return fingerprint_pack(files).fingerprint


def classify_delivery(
    registry: SourceRegistry, *,
    client_id: str,
    portfolio_id: str,
    dataset: str = "funded",
    frequency: str = "monthly",
    fingerprint: str = "",
    reporting_period: str = "",
) -> Classification:
    """Suggest the workflow type for a delivery from registry state alone."""

    def _make(kind: str, reason: str) -> Classification:
        return Classification(suggested=kind, fingerprint=fingerprint,
                              reason=reason,
                              sentence=CLASSIFICATION_SENTENCES[kind])

    if not registry.seen_client(client_id):
        return _make(WF_NEW_CLIENT, "client not present in source registry")

    if not registry.seen_portfolio(client_id, portfolio_id):
        return _make(WF_NEW_PORTFOLIO,
                     "client known but portfolio not present in source registry")

    record = registry.lookup(client_id, portfolio_id, dataset, frequency)
    if record is None or not record.has_approved_mapping:
        return _make(WF_NEW_PORTFOLIO,
                     "portfolio known but no active approved mapping for this "
                     "dataset/frequency")

    if fingerprint and record.expected_schema_fingerprint \
            and fingerprint != record.expected_schema_fingerprint:
        # A new schema for an already-approved client is a SECONDARY onboarding,
        # never treated as recurring reporting.
        return _make(WF_NEW_PORTFOLIO,
                     "schema fingerprint differs from the approved source record "
                     "(schema drift -> secondary onboarding)")

    last = record.last_successful_reporting_period or ""
    if reporting_period and last and reporting_period < last:
        return _make(WF_BACKFILL,
                     f"reporting period {reporting_period} precedes last "
                     f"successful period {last}")

    return _make(WF_RECURRING,
                 "active source record with matching schema fingerprint")


def open_source_registry(storage=None) -> SourceRegistry:
    """The production source registry via the existing layout (read here;
    written only through the existing promote path)."""
    from apps.blob_trigger_app.layout import Layout
    from apps.blob_trigger_app.storage import open_storage
    layout = Layout.from_env()
    return SourceRegistry(layout.registry_uri, storage=storage or open_storage())
