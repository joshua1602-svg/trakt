"""apps.blob_trigger_app.repin — (re)pin a source's schema fingerprint + role schemas.

The registry is the single source of truth for whether an uploaded pack routes
``deterministic`` (recurring, approved) vs ``schema_drift`` / ``source_onboarding``
(new / changed). That decision hinges on two pinned fields on the ACTIVE
:class:`SourceRecord`:

    * ``expected_schema_fingerprint`` — the header-first fingerprint of a
      representative pack;
    * ``file_role_schemas`` — the approved ``{role: [columns]}`` header signatures
      so future months are classified HEADER-FIRST (robust to filename churn).

A brand-new registry (or one with placeholder ``sha256:<fill…>`` fingerprints)
never routes ``deterministic``. This module pins both fields from ONE representative
real pack per source — the deterministic equivalent of promoting a pack through
the ops path — so the next identical upload is a clean ``deterministic`` run.

Usage (local filesystem copy of a pack folder)::

    python -m apps.blob_trigger_app.repin \\
        --pack-dir /path/to/ERE/direct/funded/monthly/direct_001/2025-11-30 \\
        --client ERE --book-type direct --dataset funded --frequency monthly \\
        --pid direct_001 --regime-required \\
        --mapping-config-path config/client/mappings/ere_direct_funded_monthly.yaml

    # Pin from a folder ALREADY in blob storage (uses the storage abstraction):
    python -m apps.blob_trigger_app.repin --blob-prefix \\
        raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30 --client ERE ...

The pinned registry is written to ``--registry`` (default: the durable
``TRAKT_SOURCE_REGISTRY_URI``). It re-uses the exact same fingerprinting the Event
Grid handler uses (:func:`schema_fingerprint.fingerprint_pack`) so what you pin is
what the trigger will later match.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .layout import Layout
from .schema_fingerprint import SchemaInfo, fingerprint_pack
from .source_registry import STATUS_ACTIVE, SourceRecord, SourceRegistry
from .storage import Storage, is_blob_uri, open_storage

_MARKER_DEFAULT = "_READY.json"
_TABULAR_EXT = (".csv", ".xlsx", ".xls", ".xlsm")


def _local_pack_files(pack_dir: str, marker: str) -> List[str]:
    base = Path(pack_dir)
    if not base.exists():
        raise FileNotFoundError(f"pack dir does not exist: {pack_dir}")
    out = [str(p) for p in sorted(base.iterdir())
           if p.is_file() and p.name != marker and p.suffix.lower() in _TABULAR_EXT]
    if not out:
        raise ValueError(f"no tabular data files found in {pack_dir} (marker {marker!r} excluded)")
    return out


def _download_blob_pack(storage: Storage, prefix: str, dest: Path,
                        marker: str) -> List[str]:
    """Download every tabular data file under a blob prefix into ``dest`` and
    return the local paths (marker excluded)."""
    prefix_uri = prefix if is_blob_uri(prefix) else f"blob://{prefix}"
    out: List[str] = []
    for uri in storage.list(prefix_uri):
        name = uri.rsplit("/", 1)[-1]
        if name == marker or Path(name).suffix.lower() not in _TABULAR_EXT:
            continue
        local = dest / name
        storage.download_file(uri, local)
        out.append(str(local))
    if not out:
        raise ValueError(f"no tabular data files found under {prefix_uri!r}")
    return sorted(out)


def repin_source(
    registry: SourceRegistry,
    *,
    client_id: str,
    source_portfolio_id: str,
    dataset: str,
    frequency: str,
    data_files: List[str],
    source_book_type: Optional[str] = None,
    regime_required: Optional[bool] = None,
    approved_mapping_id: Optional[str] = None,
    mapping_config_path: Optional[str] = None,
    source_system: Optional[str] = None,
    reuse_existing_roles: bool = False,
) -> Dict[str, Any]:
    """Compute the header-first fingerprint + role schemas of ``data_files`` and
    pin them onto an ACTIVE :class:`SourceRecord`. Returns a summary dict.

    Fails closed (raises) on an ambiguous role conflict — two files resolving to
    the same logical role — so a bad representative pack never pins a fingerprint
    the trigger cannot reproduce.
    """
    existing = registry.lookup(client_id, source_portfolio_id, dataset, frequency)
    # For a fresh pin, classify with NO approved schemas so roles come from the
    # built-in keyword/fallback rules — exactly what the Event Grid handler does
    # the FIRST time it sees an unknown source (role_schemas_for_pack → None).
    # That guarantees the pinned fingerprint is what the trigger will recompute.
    role_hints = (getattr(existing, "file_role_schemas", None) or None) if reuse_existing_roles else None
    alias_hints = (getattr(existing, "file_role_aliases", None) or None) if existing else None
    schema: SchemaInfo = fingerprint_pack(data_files, role_schemas=role_hints,
                                          aliases=alias_hints)
    if schema.ambiguous_role_conflict:
        raise ValueError(
            "ambiguous_role_conflict: two files resolved to the same logical role "
            f"{schema.conflicting_roles} — cannot pin. Diagnostics: {schema.role_diagnostics}")

    rec = existing or SourceRecord(
        client_id=client_id, source_portfolio_id=source_portfolio_id,
        dataset=dataset, frequency=frequency)
    if source_book_type:
        rec.source_portfolio_type = source_book_type
    if source_system:
        rec.source_system = source_system
    if regime_required is not None:
        rec.regime_required = bool(regime_required)
    rec.expected_schema_fingerprint = schema.fingerprint
    rec.expected_columns = list(schema.columns)
    # THE header-first signatures future months are matched against.
    rec.file_role_schemas = {role: list(cols) for role, cols in
                             (schema.sheet_columns or {}).items()}
    rec.approved_mapping_id = (approved_mapping_id or rec.approved_mapping_id
                               or f"{source_portfolio_id}_{dataset}_{frequency}_pinned_v1")
    if mapping_config_path:
        rec.mapping_config_path = mapping_config_path
    rec.mapping_version = int(getattr(rec, "mapping_version", 0) or 0) + 1
    rec.status = STATUS_ACTIVE
    registry.upsert(rec)
    return {
        "key": rec.key,
        "status": rec.status,
        "expected_schema_fingerprint": rec.expected_schema_fingerprint,
        "roles": sorted(rec.file_role_schemas.keys()),
        "role_column_counts": {r: len(c) for r, c in rec.file_role_schemas.items()},
        "approved_mapping_id": rec.approved_mapping_id,
        "regime_required": rec.regime_required,
        "data_files": [Path(f).name for f in data_files],
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m apps.blob_trigger_app.repin",
        description="(Re)pin a source's schema fingerprint + role schemas from a "
                    "representative pack so future identical uploads run deterministically.")
    p.add_argument("--pack-dir", default=None, help="Local folder of a representative pack.")
    p.add_argument("--blob-prefix", default=None,
                   help="Blob prefix of a representative pack (e.g. raw-v2/ERE/.../2025-11-30).")
    p.add_argument("--client", required=True)
    p.add_argument("--book-type", default=None, choices=[None, "direct", "acquired"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--frequency", required=True)
    p.add_argument("--pid", required=True, help="source_portfolio_id")
    p.add_argument("--regime-required", action="store_true", default=None)
    p.add_argument("--no-regime", dest="regime_required", action="store_false")
    p.add_argument("--approved-mapping-id", default=None)
    p.add_argument("--mapping-config-path", default=None)
    p.add_argument("--source-system", default=None)
    p.add_argument("--marker", default=_MARKER_DEFAULT)
    p.add_argument("--registry", default=None,
                   help="Registry path/URI (default: TRAKT_SOURCE_REGISTRY_URI / layout).")
    p.add_argument("--local-root", default=None,
                   help="Local root emulating blob containers (local/dev).")
    args = p.parse_args(argv)

    if not args.pack_dir and not args.blob_prefix:
        p.error("one of --pack-dir or --blob-prefix is required")

    storage = open_storage(local_root=args.local_root)
    layout = Layout.from_env()
    registry = SourceRegistry(args.registry or layout.registry_uri, storage=storage)

    if args.pack_dir:
        data_files = _local_pack_files(args.pack_dir, args.marker)
        summary = repin_source(
            registry, client_id=args.client, source_portfolio_id=args.pid,
            dataset=args.dataset, frequency=args.frequency, data_files=data_files,
            source_book_type=args.book_type, regime_required=args.regime_required,
            approved_mapping_id=args.approved_mapping_id,
            mapping_config_path=args.mapping_config_path, source_system=args.source_system)
    else:
        with tempfile.TemporaryDirectory(prefix="repin_pack_") as td:
            data_files = _download_blob_pack(storage, args.blob_prefix, Path(td), args.marker)
            summary = repin_source(
                registry, client_id=args.client, source_portfolio_id=args.pid,
                dataset=args.dataset, frequency=args.frequency, data_files=data_files,
                source_book_type=args.book_type, regime_required=args.regime_required,
                approved_mapping_id=args.approved_mapping_id,
                mapping_config_path=args.mapping_config_path, source_system=args.source_system)

    print("PINNED " + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
