"""trakt_core.lineage — the compact, per-field provenance index.

Answers, for one canonical field of one snapshot: *where did this come from, how
was it mapped, was it validated, and under what version?*

Per field, NOT per cell
-----------------------
The distinction is the whole design. A canonical tape carries 66-130 columns and
may carry millions of rows; storing an envelope per *value* would multiply the
dataset by its own width and buy nothing, because mapping, transformation and
validation are properties of the **field within a snapshot**, not of the
individual cell. So a 130-column tape has a 130-entry index, and
``explain_value`` composes the answer at request time from that index plus the
one row it already has.

Built at ingestion, read at request time
----------------------------------------
``engine.gate_2_transform.lineage_tracker`` writes ``lineage_index.json``
alongside the ``field_lineage.json`` it already produced. Reading it is a single
small JSON parse, cached on content identity; the previous alternative was
parsing the whole ``field_lineage.json`` document on every lookup.

An index is bound to exactly one ``(tenant_id, snapshot_id)`` and
:meth:`LineageIndex.assert_binding` refuses any other. That is not defensive
tidiness: provenance from the wrong snapshot is worse than no provenance,
because it is confidently wrong.

No FastAPI, no pandas — this is a contract, and the pipeline, the API and a CLI
all read it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from . import config_cache
from .errors import ErrorCode, TraktError

#: Written next to the canonical output, beside ``field_lineage.json``.
LINEAGE_INDEX_NAME = "lineage_index.json"

SCHEMA_VERSION = "1.0.0"

#: How a canonical field came to hold a value.
ORIGIN_MAPPED = "mapped"          # mapped from a source column
ORIGIN_DERIVED = "derived"        # produced by a declared derivation rule
ORIGIN_DEFAULTED = "defaulted"    # a configured product/asset default
ORIGIN_RUN_METADATA = "run_metadata"   # stamped from run-level metadata
ORIGIN_UNKNOWN = "unknown"

KNOWN_ORIGINS = frozenset({ORIGIN_MAPPED, ORIGIN_DERIVED, ORIGIN_DEFAULTED,
                           ORIGIN_RUN_METADATA, ORIGIN_UNKNOWN})


@dataclass(frozen=True)
class LineageEntry:
    """Provenance for ONE canonical field of ONE snapshot."""

    canonical_field: str
    origin: str = ORIGIN_UNKNOWN
    #: The source column this field was mapped from, where it was mapped.
    source_field: Optional[str] = None
    #: The input the source column came from (file name — never a full path).
    source_dataset: Optional[str] = None
    #: How the mapping was resolved: the semantic-alignment tier name, or
    #: ``llm_confirmed`` for an operator-confirmed suggestion.
    mapping_method: Optional[str] = None
    mapping_confidence: Optional[float] = None
    #: Which alias/config version resolved it, so an answer can cite it.
    mapping_version: Optional[str] = None
    #: The declared derivation rule, for ``origin == derived``.
    derivation_rule: Optional[str] = None
    #: ``passed`` | ``failed`` | ``not_validated``.
    validation_status: str = "not_validated"
    validation_rules: Tuple[str, ...] = ()
    validation_error_count: Optional[int] = None
    #: For a calculated field: how, and under which version.
    calculation_method: Optional[str] = None
    calculation_version: Optional[str] = None
    #: Canonical unit / format as declared by the field registry.
    unit: Optional[str] = None
    format: Optional[str] = None
    present_in_run: bool = True
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["validation_rules"] = list(self.validation_rules)
        return out

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "LineageEntry":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in (doc or {}).items() if k in known}
        kwargs["validation_rules"] = tuple(kwargs.get("validation_rules") or ())
        return cls(**kwargs)


@dataclass(frozen=True)
class LineageIndex:
    """Every field's provenance for ONE snapshot of ONE tenant."""

    tenant_id: str
    snapshot_id: Optional[str] = None
    content_hash: Optional[str] = None
    reporting_date: Optional[str] = None
    source_dataset: Optional[str] = None
    generated_at: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    entries: Dict[str, LineageEntry] = field(default_factory=dict)

    def get(self, canonical_field: str) -> Optional[LineageEntry]:
        return self.entries.get(str(canonical_field))

    def fields(self) -> List[str]:
        return sorted(self.entries)

    def assert_binding(self, *, tenant_id: str,
                       snapshot_id: Optional[str] = None,
                       request_id: Optional[str] = None) -> None:
        """Refuse an index that does not belong to this request.

        Provenance that silently crosses a tenant or a snapshot is the one
        failure this module exists to make impossible, so it raises rather than
        degrading to "no provenance".
        """
        if str(self.tenant_id) != str(tenant_id):
            raise TraktError(
                ErrorCode.TENANT_MISMATCH,
                "The lineage index does not belong to this tenant.",
                request_id=request_id)
        if snapshot_id and self.snapshot_id and str(self.snapshot_id) != str(snapshot_id):
            raise TraktError(
                ErrorCode.SNAPSHOT_STALE,
                "The lineage index describes a different snapshot from the one "
                "this value was read from.",
                request_id=request_id,
                details={"index_snapshot": self.snapshot_id,
                         "value_snapshot": snapshot_id})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "snapshot_id": self.snapshot_id,
            "content_hash": self.content_hash,
            "reporting_date": self.reporting_date,
            "source_dataset": self.source_dataset,
            "generated_at": self.generated_at,
            "fields": {k: v.to_dict() for k, v in sorted(self.entries.items())},
        }

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "LineageIndex":
        return cls(
            tenant_id=str((doc or {}).get("tenant_id") or ""),
            snapshot_id=(doc or {}).get("snapshot_id"),
            content_hash=(doc or {}).get("content_hash"),
            reporting_date=(doc or {}).get("reporting_date"),
            source_dataset=(doc or {}).get("source_dataset"),
            generated_at=(doc or {}).get("generated_at"),
            schema_version=str((doc or {}).get("schema_version") or SCHEMA_VERSION),
            entries={k: LineageEntry.from_dict(v)
                     for k, v in ((doc or {}).get("fields") or {}).items()},
        )

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True),
                        encoding="utf-8")
        return path


def load_lineage_index(path: str | Path, *,
                       request_id: Optional[str] = None) -> Optional[LineageIndex]:
    """Load an index, cached on its content identity. ``None`` when absent.

    Absence is a legitimate state — a snapshot produced before the index existed,
    or one built outside the pipeline — and callers report it as *unavailable
    provenance* rather than inventing any.
    """
    p = Path(path)

    def _load() -> Optional[LineageIndex]:
        if not p.exists():
            return None
        try:
            return LineageIndex.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - an unreadable index is absent provenance
            return None

    return config_cache.get_or_load("lineage_index", p, _load)


# --------------------------------------------------------------------------- #
# Building — from the artefacts the pipeline already produces
# --------------------------------------------------------------------------- #
def _mapping_from_source(source: Any) -> Tuple[Optional[str], Optional[str],
                                               Optional[float]]:
    """``(source_field, mapping_method, confidence)`` from a field_lineage
    ``source`` block, whose shape ``lineage_tracker`` already normalises."""
    if not isinstance(source, Mapping):
        return None, None, None
    raw_columns = source.get("raw_columns") or []
    methods = source.get("methods") or []
    source_field = str(raw_columns[0]) if raw_columns else None
    method = confidence = None
    if isinstance(methods, list) and methods:
        first = methods[0]
        if isinstance(first, Mapping):
            method = first.get("method")
            confidence = first.get("confidence")
    return source_field, (str(method) if method else None), confidence


def build_index_from_field_lineage(
    field_lineage: Mapping[str, Any],
    *,
    tenant_id: str,
    snapshot_id: Optional[str] = None,
    content_hash: Optional[str] = None,
    reporting_date: Optional[str] = None,
    source_dataset: Optional[str] = None,
    generated_at: Optional[str] = None,
    derivations: Optional[Mapping[str, Any]] = None,
    registry_fields: Optional[Mapping[str, Any]] = None,
    run_metadata_fields: Iterable[str] = (),
) -> LineageIndex:
    """Compact a ``field_lineage.json`` document into a per-field index.

    Reads what the pipeline already emits rather than instrumenting it again, so
    the index cannot drift from the lineage it summarises.
    """
    run_metadata = {str(f) for f in run_metadata_fields}
    entries: Dict[str, LineageEntry] = {}

    for name, block in ((field_lineage or {}).get("fields") or {}).items():
        block = block or {}
        source_field, method, confidence = _mapping_from_source(block.get("source"))
        status = block.get("field_status") or {}
        present = bool(status.get("present_in_run", True))
        quality = block.get("quality") or {}
        spec = block.get("declared_spec") or {}
        registry_spec = (registry_fields or {}).get(name) or {}

        derivation = None
        origin = ORIGIN_UNKNOWN
        if name in run_metadata:
            origin = ORIGIN_RUN_METADATA
        elif derivations and name in derivations:
            origin = ORIGIN_DERIVED
            rule = derivations[name]
            derivation = (rule.get("rule") if isinstance(rule, Mapping)
                          else str(rule))
        elif source_field:
            origin = ORIGIN_MAPPED

        errors = quality.get("error_count") if isinstance(quality, Mapping) else None
        if not present:
            validation_status = "not_validated"
        elif errors is None:
            validation_status = "not_validated"
        elif int(errors) == 0:
            validation_status = "passed"
        else:
            validation_status = "failed"

        entries[str(name)] = LineageEntry(
            canonical_field=str(name),
            origin=origin,
            source_field=source_field,
            source_dataset=source_dataset,
            mapping_method=method,
            mapping_confidence=confidence,
            mapping_version=(field_lineage or {}).get("mapping_version"),
            derivation_rule=derivation,
            validation_status=validation_status,
            validation_error_count=(int(errors) if errors is not None else None),
            unit=registry_spec.get("unit"),
            format=registry_spec.get("format") or spec.get("format"),
            present_in_run=present,
        )

    return LineageIndex(
        tenant_id=str(tenant_id), snapshot_id=snapshot_id,
        content_hash=content_hash, reporting_date=reporting_date,
        source_dataset=source_dataset, generated_at=generated_at,
        entries=entries)
