"""mi_agent.business_semantics — the Business Semantics Registry v2 loader.

``config/business_semantics_registry.yaml`` is the governed business-semantics
layer over the canonical field registry (schema version 2, registry version
0.2.0). Until now it was only read by its build script and its validation test:
nothing in the runtime consumed it. This module is that consumer — ONE typed,
cached loader, so a workflow reads governed metadata rather than inferring field
meaning from names.

What it is
----------
A read-only projection of the committed YAML into frozen dataclasses:

    load_business_semantics()  ──►  BusinessSemanticsRegistry
                                      ├── .metadata      (version, schema_version, taxonomy)
                                      ├── .entries       {canonical_field: BusinessSemanticsEntry}
                                      └── .for_workflow(WORKFLOW_PERIOD_CHANGE)

What it is NOT
--------------
* not a second field registry — every entry keys a canonical field that exists in
  ``config/system/fields_registry.yaml`` (asserted by the existing registry test);
* not a place to curate semantics — curation lives in
  ``scripts/build_business_semantics_registry.py`` and is regenerated from there;
* not aware of any workflow. It carries metadata; interpretation belongs to the
  consuming workflow.

Source-specific temporality overrides
-------------------------------------
The v2 review records that ``allocated_losses`` is classified ``cumulative``
under the ESMA "losses to date" interpretation, while an individual source may
report it as a period figure. Rather than special-casing a client inside a
workflow, an override is *governed data*: an optional
``config/business_semantics_source_overrides.yaml`` maps a source identifier to
per-field metadata overrides, and :meth:`BusinessSemanticsRegistry.for_source`
returns a registry view with them applied. A future override therefore changes
configuration, not workflow code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml

logger = logging.getLogger("mi_agent.business_semantics")

_REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = _REPO_ROOT / "config" / "business_semantics_registry.yaml"
SOURCE_OVERRIDES_PATH = (
    _REPO_ROOT / "config" / "business_semantics_source_overrides.yaml")

#: The schema version this loader understands. A registry declaring a different
#: major schema is refused rather than silently misread.
SUPPORTED_SCHEMA_VERSION = 2

# --------------------------------------------------------------------------- #
# Controlled taxonomy — mirrors the registry's own ``metadata.taxonomy`` block.
# Duplicated as constants so consuming code names them; validated against the
# YAML by tests/test_business_semantics_loader.py.
# --------------------------------------------------------------------------- #
ROLE_MEASURE = "measure"
ROLE_DIMENSION = "dimension"
ROLE_DERIVED_INPUT = "derived_input"
ROLE_SUPPORTING_ATTRIBUTE = "supporting_attribute"

TEMPORALITY_POINT_IN_TIME = "point_in_time"
TEMPORALITY_PERIOD_FLOW = "period_flow"
TEMPORALITY_CUMULATIVE = "cumulative"
TEMPORALITY_STATIC_BASELINE = "static_baseline"

AGG_SUM = "sum"
AGG_AVERAGE = "average"
AGG_WEIGHTED_AVERAGE = "weighted_average"
AGG_SHARE = "share"
AGG_DISTRIBUTION = "distribution"

DIRECTION_HIGHER_IS_BETTER = "higher_is_better"
DIRECTION_HIGHER_IS_WORSE = "higher_is_worse"
DIRECTION_LOWER_IS_BETTER = "lower_is_better"
DIRECTION_LOWER_IS_WORSE = "lower_is_worse"
DIRECTION_NEUTRAL = "neutral"
DIRECTION_CONTEXT_DEPENDENT = "context_dependent"

WORKFLOW_PERIOD_CHANGE = "period_change"
WORKFLOW_PORTFOLIO_COMPARISON = "portfolio_comparison"
WORKFLOW_RANKING = "ranking"
WORKFLOW_MONITORING = "monitoring"

ASSET_CROSS_ASSET = "cross_asset"

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

#: Confidence ordering, for a deterministic selection preference.
CONFIDENCE_RANK: Mapping[str, int] = {
    CONFIDENCE_HIGH: 0, CONFIDENCE_MEDIUM: 1, CONFIDENCE_LOW: 2}

#: Fields a source override may legitimately restate. Deliberately narrow: an
#: override exists to correct a *source convention* (how a figure is reported),
#: not to re-curate what a field means.
OVERRIDABLE_KEYS: frozenset = frozenset({
    "temporality", "default_aggregation", "weight_field", "share_basis",
    "confidence", "rationale"})


class BusinessSemanticsError(RuntimeError):
    """The registry could not be loaded or is not the supported schema."""


def _tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class BusinessSemanticsEntry:
    """One governed field's business semantics. Read-only by construction."""

    field: str
    source_field: str
    display_name: str
    analytical_concept: str
    analytical_role: str
    temporality: str
    categories: Tuple[str, ...]
    workflow_tags: Tuple[str, ...]
    directionality: str
    default_aggregation: str
    weight_field: Optional[str]
    share_basis: Optional[str]
    portfolio_comparability: str
    supports_materiality_assessment: bool
    asset_applicability: Tuple[str, ...]
    confidence: str
    rationale: str
    #: Set when a governed source override restated part of this entry.
    overridden_from: Optional[Mapping[str, Any]] = None

    # -- predicates a consumer commonly branches on ------------------------- #
    def has_workflow_tag(self, tag: str) -> bool:
        return tag in self.workflow_tags

    def applies_to_asset(self, asset_classes: Sequence[str]) -> bool:
        """True when this entry may be used for the given portfolio asset classes.

        ``cross_asset`` applies generally. An asset-specific entry applies only
        when the portfolio actually contains that asset class. An empty
        ``asset_classes`` (the portfolio's asset class is unknown) admits only
        ``cross_asset`` entries — the conservative reading, so an equity-release
        field is never reported for an unidentified book.
        """
        if ASSET_CROSS_ASSET in self.asset_applicability:
            return True
        return any(a in self.asset_applicability for a in asset_classes)

    @property
    def is_overridden(self) -> bool:
        return self.overridden_from is not None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "field": self.field,
            "source_field": self.source_field,
            "display_name": self.display_name,
            "analytical_concept": self.analytical_concept,
            "analytical_role": self.analytical_role,
            "temporality": self.temporality,
            "categories": list(self.categories),
            "workflow_tags": list(self.workflow_tags),
            "directionality": self.directionality,
            "default_aggregation": self.default_aggregation,
            "weight_field": self.weight_field,
            "share_basis": self.share_basis,
            "portfolio_comparability": self.portfolio_comparability,
            "supports_materiality_assessment": self.supports_materiality_assessment,
            "asset_applicability": list(self.asset_applicability),
            "confidence": self.confidence,
            "rationale": self.rationale,
        }
        if self.overridden_from is not None:
            data["overridden_from"] = dict(self.overridden_from)
        return data

    @classmethod
    def from_yaml(cls, field: str, data: Mapping[str, Any]) -> "BusinessSemanticsEntry":
        return cls(
            field=field,
            source_field=str(data.get("source_field") or field),
            display_name=str(data.get("display_name") or field),
            analytical_concept=str(data.get("analytical_concept") or ""),
            analytical_role=str(data.get("analytical_role") or ""),
            temporality=str(data.get("temporality") or ""),
            categories=_tuple(data.get("categories")),
            workflow_tags=_tuple(data.get("workflow_tags")),
            directionality=str(data.get("directionality") or ""),
            default_aggregation=str(data.get("default_aggregation") or ""),
            weight_field=_clean(data.get("weight_field")),
            share_basis=_clean(data.get("share_basis")),
            portfolio_comparability=str(data.get("portfolio_comparability") or ""),
            supports_materiality_assessment=bool(
                data.get("supports_materiality_assessment")),
            asset_applicability=_tuple(data.get("asset_applicability")),
            confidence=str(data.get("confidence") or ""),
            rationale=str(data.get("rationale") or ""))


@dataclass(frozen=True)
class BusinessSemanticsRegistry:
    """The loaded registry: metadata, taxonomy and typed entries."""

    version: str
    schema_version: int
    source_registry: str
    entries: Mapping[str, BusinessSemanticsEntry]
    taxonomy: Mapping[str, Tuple[str, ...]]
    #: The governed source identifier whose overrides were applied, if any.
    source_id: Optional[str] = None

    # -- lookup ------------------------------------------------------------- #
    def get(self, field: str) -> Optional[BusinessSemanticsEntry]:
        return self.entries.get(field)

    def __contains__(self, field: object) -> bool:
        return field in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def fields(self) -> Tuple[str, ...]:
        return tuple(sorted(self.entries))

    def for_workflow(self, tag: str) -> Tuple[BusinessSemanticsEntry, ...]:
        """Every entry carrying ``tag`` in ``workflow_tags``, name-ordered.

        Deterministic ordering matters: it is the tie-break of the downstream
        field-selection policy, so the same registry always yields the same
        candidate list.
        """
        return tuple(self.entries[f] for f in sorted(self.entries)
                     if self.entries[f].has_workflow_tag(tag))

    def by_concept(self, concept: str, *, tag: Optional[str] = None
                   ) -> Tuple[BusinessSemanticsEntry, ...]:
        pool = self.for_workflow(tag) if tag else tuple(
            self.entries[f] for f in sorted(self.entries))
        return tuple(e for e in pool if e.analytical_concept == concept)

    def concepts(self, *, tag: Optional[str] = None) -> Tuple[str, ...]:
        pool = self.for_workflow(tag) if tag else tuple(self.entries.values())
        return tuple(sorted({e.analytical_concept for e in pool if e.analytical_concept}))

    # -- governed source overrides ------------------------------------------ #
    def for_source(self, source_id: Optional[str],
                   overrides: Optional[Mapping[str, Any]] = None
                   ) -> "BusinessSemanticsRegistry":
        """A registry view with ``source_id``'s governed overrides applied.

        Returns ``self`` when there is nothing to apply, so the common path
        allocates nothing. ``overrides`` is injectable for tests; production
        reads ``config/business_semantics_source_overrides.yaml``.
        """
        if not source_id:
            return self
        table = overrides if overrides is not None else load_source_overrides()
        per_field = (table or {}).get(source_id) or {}
        if not per_field:
            return self

        updated: Dict[str, BusinessSemanticsEntry] = dict(self.entries)
        for field, changes in per_field.items():
            entry = updated.get(field)
            if entry is None or not isinstance(changes, Mapping):
                logger.info("source override for unknown field %r ignored", field)
                continue
            applied = {k: v for k, v in changes.items() if k in OVERRIDABLE_KEYS}
            rejected = sorted(set(changes) - OVERRIDABLE_KEYS)
            if rejected:
                logger.warning(
                    "source override for %s.%s ignored non-overridable keys: %s",
                    source_id, field, ", ".join(rejected))
            if not applied:
                continue
            before = {k: getattr(entry, k) for k in applied}
            updated[field] = replace(
                entry, **applied,
                overridden_from={"source_id": source_id, "previous": before})
        return replace(self, entries=updated, source_id=source_id)

    def to_provenance(self) -> Dict[str, Any]:
        """The audit stamp identifying exactly which registry produced a result."""
        return {
            "registry": "business_semantics_registry",
            "registry_version": self.version,
            "schema_version": self.schema_version,
            "source_registry": self.source_registry,
            "entry_count": len(self.entries),
            "source_id": self.source_id,
            "overridden_fields": sorted(
                f for f, e in self.entries.items() if e.is_overridden),
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_registry(data: Mapping[str, Any]) -> BusinessSemanticsRegistry:
    """Build a registry from already-loaded YAML. Pure; used by tests."""
    metadata = data.get("metadata") or {}
    schema_version = metadata.get("schema_version")
    if int(schema_version or 0) != SUPPORTED_SCHEMA_VERSION:
        raise BusinessSemanticsError(
            f"Business Semantics Registry schema_version {schema_version!r} is not "
            f"supported (this loader reads schema version "
            f"{SUPPORTED_SCHEMA_VERSION}).")

    raw_fields = data.get("fields") or {}
    if not isinstance(raw_fields, Mapping) or not raw_fields:
        raise BusinessSemanticsError(
            "Business Semantics Registry contains no fields.")

    entries = {name: BusinessSemanticsEntry.from_yaml(name, spec)
               for name, spec in raw_fields.items()}
    taxonomy = {k: _tuple(v) for k, v in (metadata.get("taxonomy") or {}).items()}
    return BusinessSemanticsRegistry(
        version=str(metadata.get("version") or "unknown"),
        schema_version=int(schema_version),
        source_registry=str(metadata.get("source_registry") or ""),
        entries=entries, taxonomy=taxonomy)


@lru_cache(maxsize=4)
def _load_cached(path_text: str) -> BusinessSemanticsRegistry:
    path = Path(path_text)
    if not path.exists():
        raise BusinessSemanticsError(
            f"Business Semantics Registry not found at {path}")
    return parse_registry(_read_yaml(path))


def load_business_semantics(path: Optional[Path | str] = None
                            ) -> BusinessSemanticsRegistry:
    """The governed Business Semantics Registry, cached per path.

    Raises :class:`BusinessSemanticsError` when the registry is missing or
    declares an unsupported schema version — a workflow that depends on governed
    metadata must fail explicitly rather than fall back to name inference.
    """
    return _load_cached(str(Path(path) if path else REGISTRY_PATH))


@lru_cache(maxsize=4)
def _load_overrides_cached(path_text: str) -> Mapping[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    data = _read_yaml(path)
    sources = data.get("sources")
    return dict(sources) if isinstance(sources, Mapping) else {}


def load_source_overrides(path: Optional[Path | str] = None) -> Mapping[str, Any]:
    """Governed per-source semantic overrides. ``{}`` when none are configured."""
    return _load_overrides_cached(
        str(Path(path) if path else SOURCE_OVERRIDES_PATH))


def reset_cache() -> None:
    """Drop the loader caches. For tests that write a temporary registry."""
    _load_cached.cache_clear()
    _load_overrides_cached.cache_clear()


def entries_for(fields: Iterable[str], *,
                registry: Optional[BusinessSemanticsRegistry] = None
                ) -> Tuple[Tuple[str, Optional[BusinessSemanticsEntry]], ...]:
    """``((field, entry|None), …)`` for an explicit field list, order preserved."""
    reg = registry or load_business_semantics()
    return tuple((f, reg.get(f)) for f in fields)
