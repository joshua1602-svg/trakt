"""trakt_core.resource — WHAT a permissionable thing is, and exactly which rows it means.

Stage 2 of the multi-tenant work. This module answers one question and refuses to
answer the other:

    ANSWERS      "What is this resource, and which existing Trakt data
                  population does it deterministically represent?"
    DOES NOT     "Who is allowed to reach it?"

Nothing here grants anything. A resource in the catalogue is reachable by exactly
the same callers as before this module existed; the entitlement stage is what
turns these identifiers into access.

Why a resource layer at all
---------------------------
``source_portfolio_id`` is the only genuinely governed resource identifier Trakt
has: it is stamped at onboarding and **mandatory on every onboarded row**
(``config/system/fields_registry.yaml``). Everything else a funder would want to
be scoped to is currently something weaker:

  * **SPV** is a segmentation *dimension*. ``spv_id`` is a RESERVED **OPTIONAL**
    column at the snapshot layer (``snapshot.model.RESERVED_OPTIONAL_LOAN_COLUMNS``)
    and is absent from the canonical fields registry entirely. A client may
    simply not supply it.
  * **funded / pipeline / forecast** are *views* — ``mi_agent_api.workspace.VIEWS``
    — chosen freely per request, and ``resolve_active_view`` will even infer one
    from the words in a question.
  * **Presentation scope** (``trakt_core.portfolio.resolve_scope`` +
    ``mi_agent.portfolio_scope.apply_scope``) deliberately **fails open**: an
    unrecognised context widens to Total, and a frame with no provenance column
    is returned unfiltered. That is defensible for a display lens and
    catastrophic as a security filter.

So a funder entitlement cannot currently be *written down*, because there is
nothing stable to name. This module supplies the name and the deterministic
expansion, and does so without touching any of the above.

The three types
---------------
``ResourceRef``        the stable identity — ``(tenant_id, kind, resource_id)``.
                       Hashable, comparable, parseable from a canonical string.
``ResourceRecord``     the catalogue entry: what the resource is made of.
``ResolvedResource``   the expansion — the concrete, deterministic predicate over
                       existing canonical fields, plus whether the data can
                       actually carry it.

Two rules the design exists to enforce
--------------------------------------
**A resource never widens.** ``ResolvedResource.predicate()`` cannot return an
empty filter unless the record explicitly set ``whole_tenant_book``. A record
that declares no constraint at all is rejected at load time, so whole-book access
is impossible to reach *by omission* — only by writing it down.

**Missing attribution is a refusal, not a fallback.** :func:`check_attribution`
compares the fields a predicate needs against the fields the data actually has
and raises ``RESOURCE_NOT_PARTITIONABLE`` when they are missing. It never
degrades to "no filter", which is precisely what ``apply_scope`` does today and
why that function must not become the entitlement filter.

Data loading
------------
Lookup and expansion are pure metadata: no pandas, no storage, no dataframe.
:func:`check_attribution` takes the available field names as an argument rather
than discovering them, so the caller decides when data is touched and this module
stays importable anywhere ``trakt_core`` is.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from . import config_cache
from .errors import ErrorCode, TraktError
from .portfolio import (
    FIELD_PORTFOLIO_ID,
    PortfolioRecord,
    PortfolioScope,
    CONTEXT_KIND_PORTFOLIO,
)

logger = logging.getLogger("trakt_core.resource")

DEFAULT_RESOURCE_CONFIG = "config/resources.yaml"
RESOURCE_CONFIG_ENV = "TRAKT_RESOURCES_CONFIG"

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

#: A Trakt portfolio reference (``portfolio_id`` in the snapshot layer), or a
#: named grouping of books the tenant treats as one portfolio.
KIND_PORTFOLIO = "portfolio"
#: One onboarded source cohort — the ``source_portfolio_id`` provenance axis.
#: The strongest resource kind, because the field is mandatory on every row.
KIND_SOURCE_PORTFOLIO = "source_portfolio"
#: A special-purpose vehicle. Permissionable ONLY where it can be attributed
#: deterministically; see :data:`ATTRIBUTION_UNPARTITIONABLE`.
KIND_SPV = "spv"
#: A warehouse facility. Structurally identical to an SPV in this model — it is a
#: distinct kind because the two are distinct things commercially and an
#: entitlement should say which one it means.
KIND_FACILITY = "facility"
#: A whole population of the tenant's book (``pipeline``, ``funded``) with no
#: portfolio narrowing.
KIND_POPULATION = "population"

RESOURCE_KINDS: Tuple[str, ...] = (
    KIND_PORTFOLIO, KIND_SOURCE_PORTFOLIO, KIND_SPV, KIND_FACILITY, KIND_POPULATION,
)

#: Populations, mirroring ``mi_agent_api.workspace.VIEWS``. A drift test in
#: ``tests/test_governance_resource_model.py`` keeps the two aligned; they are
#: duplicated as literals so ``trakt_core`` stays free of the API package.
POPULATION_FUNDED = "funded"
POPULATION_PIPELINE = "pipeline"
POPULATION_FORECAST = "forecast"
POPULATIONS: Tuple[str, ...] = (POPULATION_FUNDED, POPULATION_PIPELINE,
                                POPULATION_FORECAST)

#: How a resource is narrowed. Derived from what the record declares — never
#: hand-written in config — so an operator cannot mis-declare a resource into
#: being wider than its constraints.
ATTRIBUTION_PROVENANCE = "provenance"        # source_portfolio_id
ATTRIBUTION_SPV_FIELD = "spv_field"          # spv_id (optionally + provenance)
ATTRIBUTION_VIEW = "view"                    # population only
ATTRIBUTION_WHOLE_BOOK = "whole_book"        # explicit, opt-in only
ATTRIBUTION_UNPARTITIONABLE = "unpartitionable"

#: Canonical field carrying the SPV / facility segmentation key, where a client
#: supplies one. Optional in the data by design — hence the refusal path.
FIELD_SPV_ID = "spv_id"

#: A resource id is a plain slug — anchored, no separators — so a catalogue key
#: can never traverse a storage path or inject a container segment. Same rule as
#: ``trakt_core.tenancy``'s portfolio selector.
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _slug(value: Any) -> Optional[str]:
    text = _clean(value)
    return text.lower() if text else None


def _valid_token(value: str) -> bool:
    return bool(_SAFE_TOKEN.match(value or "")) and ".." not in value


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResourceRef:
    """The stable identity of one permissionable resource.

    Frozen and therefore hashable, so a later entitlement grant can hold a set of
    these and membership is an ``in`` test rather than a string comparison
    someone has to remember to normalise. Normalisation happens once, here:
    ``ERE/SPV/Spv_I`` and ``ere/spv/spv_i`` are the same ref.

    ``tenant_id`` is part of the identity, not context around it. Two tenants may
    each have an ``spv_i`` and they are different resources; no lookup can
    confuse them because the tenant is inside the key.
    """

    tenant_id: str
    kind: str
    resource_id: str

    def __post_init__(self) -> None:
        tenant_id = _clean(self.tenant_id)
        kind = _slug(self.kind)
        resource_id = _slug(self.resource_id)
        if not tenant_id:
            raise ValueError("ResourceRef requires a tenant_id")
        if not resource_id:
            raise ValueError("ResourceRef requires a resource_id")
        if kind not in RESOURCE_KINDS:
            raise ValueError(
                f"ResourceRef kind must be one of {RESOURCE_KINDS}, got {kind!r}")
        if not _valid_token(tenant_id) or not _valid_token(resource_id):
            raise ValueError(
                "ResourceRef tenant_id and resource_id must be plain slugs")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "resource_id", resource_id)

    def __str__(self) -> str:
        return self.key

    @property
    def key(self) -> str:
        """``"{tenant}/{kind}/{resource_id}"`` — the canonical string form.

        Stable across processes and releases: safe to write into a config file,
        an audit line or a future grant.
        """
        return f"{self.tenant_id}/{self.kind}/{self.resource_id}"

    @classmethod
    def parse(cls, text: str) -> "ResourceRef":
        parts = [p for p in str(text or "").split("/")]
        if len(parts) != 3:
            raise ValueError(
                "A resource reference must be '{tenant}/{kind}/{resource_id}', "
                f"got {text!r}")
        return cls(tenant_id=parts[0], kind=parts[1], resource_id=parts[2])

    def to_dict(self) -> Dict[str, Any]:
        return {"tenant_id": self.tenant_id, "kind": self.kind,
                "resource_id": self.resource_id, "key": self.key}


# --------------------------------------------------------------------------- #
# Catalogue entries
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResourceRecord:
    """What a resource is made of, as declared in the catalogue.

    The constraints are *conjunctive*: a record carrying both
    ``source_portfolio_ids`` and ``spv_id`` means rows in those books AND in that
    SPV. A record must carry at least one constraint, or set
    ``whole_tenant_book``; :class:`ResourceCatalogue` rejects one that carries
    neither, so "no constraints" can never be reached by accident.
    """

    ref: ResourceRef
    display_label: Optional[str] = None
    #: Books this resource covers, on the mandatory provenance axis.
    source_portfolio_ids: Tuple[str, ...] = ()
    #: The SPV / facility key, where the client supplies ``spv_id``.
    spv_id: Optional[str] = None
    #: Immutable population constraint — ``funded`` / ``pipeline`` / ``forecast``.
    #: ``None`` means the resource does not constrain the population.
    #:
    #: Population is modelled as a CONSTRAINT rather than a child resource or a
    #: mutable attribute, and that is the load-bearing choice here. As a child
    #: resource, "SPV I funded" would need two refs intersected at request time —
    #: an intersection someone can forget. As a mutable attribute, the request
    #: could change it, and today the request genuinely can (``datasetContext``,
    #: and ``resolve_active_view`` reading the question text). Baking it into the
    #: record makes "SPV I / funded" one identifier that cannot be widened to
    #: pipeline by anything a caller says.
    population: Optional[str] = None
    #: Explicitly declares that this resource CANNOT be safely partitioned from
    #: the data, so every expansion refuses. The honest representation of a
    #: partial-book SPV whose rows carry no attribution.
    unpartitionable: bool = False
    #: Operator-facing explanation shown with the refusal.
    unpartitionable_reason: Optional[str] = None
    #: Opt-in whole-tenant access. Explicit because it is the one shape that
    #: carries no narrowing at all.
    whole_tenant_book: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        ids = tuple(sorted({_clean(p) for p in (self.source_portfolio_ids or ())}
                           - {None}))  # type: ignore[operator]
        object.__setattr__(self, "source_portfolio_ids", ids)
        object.__setattr__(self, "spv_id", _clean(self.spv_id))
        population = _slug(self.population)
        if population is not None and population not in POPULATIONS:
            raise ValueError(
                f"population must be one of {POPULATIONS}, got {population!r}")
        object.__setattr__(self, "population", population)
        object.__setattr__(self, "unpartitionable", bool(self.unpartitionable))
        object.__setattr__(self, "whole_tenant_book", bool(self.whole_tenant_book))
        object.__setattr__(self, "enabled", bool(self.enabled))
        for pid in ids:
            if not _valid_token(pid):
                raise ValueError(
                    f"source_portfolio_id {pid!r} is not a valid identifier")
        if self.spv_id is not None and not _valid_token(self.spv_id):
            raise ValueError(f"spv_id {self.spv_id!r} is not a valid identifier")

    @property
    def label(self) -> str:
        return self.display_label or self.ref.resource_id

    @property
    def attribution(self) -> str:
        """How this resource narrows the data. Derived, never declared.

        Deriving rather than reading an ``attribution:`` field from config is
        deliberate: an operator who wrote ``attribution: view`` on a record that
        also names source portfolios would have declared something weaker than
        the record actually is, and the mismatch would be invisible.
        """
        if self.unpartitionable:
            return ATTRIBUTION_UNPARTITIONABLE
        if self.spv_id is not None:
            return ATTRIBUTION_SPV_FIELD
        if self.source_portfolio_ids:
            return ATTRIBUTION_PROVENANCE
        if self.whole_tenant_book:
            return ATTRIBUTION_WHOLE_BOOK
        if self.population is not None:
            return ATTRIBUTION_VIEW
        # Unreachable through the catalogue, which rejects this at load.
        return ATTRIBUTION_UNPARTITIONABLE

    @property
    def constrains_anything(self) -> bool:
        return bool(self.source_portfolio_ids or self.spv_id or self.population
                    or self.whole_tenant_book or self.unpartitionable)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.ref.to_dict(),
            "display_label": self.label,
            "source_portfolio_ids": list(self.source_portfolio_ids),
            "spv_id": self.spv_id,
            "population": self.population,
            "attribution": self.attribution,
            "whole_tenant_book": self.whole_tenant_book,
            "unpartitionable": self.unpartitionable,
            "enabled": self.enabled,
        }


# --------------------------------------------------------------------------- #
# Expansion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolvedResource:
    """A resource expanded into the concrete data population it represents.

    This is the shape a later entitlement check consumes. The point of it is that
    a caller never supplies a raw dataset identifier: it names a resource, and
    the server produces the predicate.
    """

    ref: ResourceRef
    owning_tenant_id: str
    #: Books in scope on the mandatory provenance axis. Empty means "this
    #: resource does not narrow by book" — NOT "every book".
    source_portfolio_ids: Tuple[str, ...] = ()
    spv_id: Optional[str] = None
    population: Optional[str] = None
    attribution: str = ATTRIBUTION_UNPARTITIONABLE
    display_label: Optional[str] = None
    whole_tenant_book: bool = False
    unpartitionable_reason: Optional[str] = None

    @property
    def partitionable(self) -> bool:
        return self.attribution != ATTRIBUTION_UNPARTITIONABLE

    @property
    def required_fields(self) -> FrozenSet[str]:
        """Canonical columns the predicate needs in order to be applied.

        A view-only or whole-book resource needs none: the population selects
        which frame is loaded rather than filtering rows within one.
        """
        needed = set()
        if self.source_portfolio_ids:
            needed.add(FIELD_PORTFOLIO_ID)
        if self.spv_id is not None:
            needed.add(FIELD_SPV_ID)
        return frozenset(needed)

    def predicate(self) -> Dict[str, Any]:
        """The deterministic filter this resource means.

        Never empty for anything except an explicit ``whole_tenant_book``
        resource. That is the invariant standing between this model and the
        fail-open behaviour it replaces: there is no input to this function that
        silently produces "match everything".
        """
        if not self.partitionable:
            raise TraktError(
                ErrorCode.RESOURCE_NOT_PARTITIONABLE,
                self.unpartitionable_reason
                or "This resource cannot be separated from the rest of the book "
                   "with the attribution the data carries.",
                details={"resource": self.ref.key})
        out: Dict[str, Any] = {}
        if self.population is not None:
            out["view"] = self.population
        if self.source_portfolio_ids:
            out[FIELD_PORTFOLIO_ID] = list(self.source_portfolio_ids)
        if self.spv_id is not None:
            out[FIELD_SPV_ID] = self.spv_id
        if not out and not self.whole_tenant_book:
            # Defence in depth: the catalogue rejects an unconstrained record at
            # load, so reaching here means a record was hand-built. Refuse
            # rather than hand back a filter that matches the whole tenant.
            raise TraktError(
                ErrorCode.RESOURCE_NOT_PARTITIONABLE,
                "This resource declares no constraint and is not marked as the "
                "whole tenant book, so it cannot be expanded safely.",
                details={"resource": self.ref.key})
        return out

    def to_portfolio_scope(self) -> Optional[PortfolioScope]:
        """A :class:`~trakt_core.portfolio.PortfolioScope` for the book narrowing.

        Provided so a later stage can hand this to the services that already take
        a ``scope`` (``risk_limits.compute_risk_limits``, ``apply_scope``) without
        inventing a second scope type.

        **This is not a security filter on its own, and must not be treated as
        one.** ``mi_agent.portfolio_scope.apply_scope`` returns the frame
        UNCHANGED when it carries no ``source_portfolio_id`` column, so a scope
        alone can fail open exactly the way this module exists to prevent. It is
        safe only after :func:`check_attribution` has confirmed the field is
        present, and it carries no SPV or population constraint — those live in
        :meth:`predicate` and have to be applied too.

        Returns ``None`` when this resource does not narrow by book. ``None``
        means "no book narrowing available here", never "all books": returning a
        Total scope would be the fail-open bug.
        """
        if not self.source_portfolio_ids:
            return None
        return PortfolioScope(
            context_id=self.ref.resource_id,
            context_kind=CONTEXT_KIND_PORTFOLIO,
            label=self.display_label or self.ref.resource_id,
            portfolio_ids=tuple(self.source_portfolio_ids),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.ref.to_dict(),
            "owning_tenant_id": self.owning_tenant_id,
            "display_label": self.display_label,
            "source_portfolio_ids": list(self.source_portfolio_ids),
            "spv_id": self.spv_id,
            "population": self.population,
            "attribution": self.attribution,
            "partitionable": self.partitionable,
            "whole_tenant_book": self.whole_tenant_book,
            "required_fields": sorted(self.required_fields),
        }


@dataclass(frozen=True)
class AttributionCheck:
    """Whether the data can actually carry a resource's predicate."""

    resolved: ResolvedResource
    available_fields: FrozenSet[str]
    missing_fields: FrozenSet[str]

    @property
    def ok(self) -> bool:
        return self.resolved.partitionable and not self.missing_fields


def check_attribution(resolved: ResolvedResource,
                      available_fields: Iterable[str]) -> AttributionCheck:
    """Confirm the fields a resource needs are present, or raise.

    The whole point of this function is what it does **not** do. When the
    attribution column is missing, ``apply_scope`` today returns every row; this
    raises ``RESOURCE_NOT_PARTITIONABLE``. A dataset that cannot express the
    boundary is a reason to refuse, never a reason to serve more.

    ``available_fields`` is passed in rather than discovered so this module never
    loads data and stays importable without pandas.
    """
    fields = frozenset(str(f) for f in (available_fields or ()))
    if not resolved.partitionable:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            resolved.unpartitionable_reason
            or "This resource cannot be separated from the rest of the book with "
               "the attribution the data carries.",
            details={"resource": resolved.ref.key})
    missing = frozenset(resolved.required_fields - fields)
    if missing:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_PARTITIONABLE,
            "The governed dataset does not carry the attribution this resource "
            "needs, so its boundary cannot be enforced.",
            details={"resource": resolved.ref.key,
                     "missing_fields": sorted(missing)})
    return AttributionCheck(resolved=resolved, available_fields=fields,
                            missing_fields=frozenset())


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #
class ResourceCatalogue:
    """Every resource this deployment can name, indexed by :class:`ResourceRef`.

    Rejects at construction, deterministically:

      * a duplicate ``(tenant, kind, resource_id)``;
      * a record declaring no constraint and not marked ``whole_tenant_book``.

    Both are ambiguity or accidental-widening rather than a preference, and both
    are cheaper to find at load than in a grant six months later.
    """

    def __init__(self, records: Iterable[ResourceRecord], *, configured: bool) -> None:
        ordered = sorted(records, key=lambda r: r.ref.key)
        by_key: Dict[str, ResourceRecord] = {}
        for record in ordered:
            if record.ref.key in by_key:
                raise TraktError(
                    ErrorCode.INVALID_INPUT,
                    "The resource catalogue declares the same resource more "
                    "than once.",
                    details={"resource": record.ref.key})
            if not record.constrains_anything:
                raise TraktError(
                    ErrorCode.INVALID_INPUT,
                    "A resource must narrow the data by book, SPV or population, "
                    "or be explicitly declared as the whole tenant book. A "
                    "resource with no constraint would silently mean everything.",
                    details={"resource": record.ref.key})
            by_key[record.ref.key] = record
        self._by_key = by_key
        #: True when an explicit catalogue config was loaded.
        self.configured = bool(configured)

    # -- lookup ------------------------------------------------------------- #
    def get(self, ref: ResourceRef) -> Optional[ResourceRecord]:
        """A pure lookup. Does not apply ``enabled`` and does not raise."""
        return self._by_key.get(ref.key)

    def resolve(self, ref: ResourceRef, *,
                request_id: Optional[str] = None) -> ResolvedResource:
        """Expand ``ref`` into its data population, or raise.

        Fails closed on every unhappy path: unknown, disabled and unpartitionable
        all raise rather than returning a permissive default. There is no branch
        that returns "everything" for an input the catalogue did not recognise.

        Pure metadata — no dataset is opened. Whether the data can carry the
        result is :func:`check_attribution`'s question.
        """
        record = self._by_key.get(ref.key)
        if record is None:
            raise TraktError(
                ErrorCode.RESOURCE_NOT_REGISTERED,
                "That resource is not registered with Trakt.",
                request_id=request_id, details={"resource": ref.key})
        if not record.enabled:
            raise TraktError(
                ErrorCode.RESOURCE_DISABLED,
                "That resource is currently disabled.",
                request_id=request_id, details={"resource": ref.key})
        return ResolvedResource(
            ref=record.ref,
            owning_tenant_id=record.ref.tenant_id,
            source_portfolio_ids=record.source_portfolio_ids,
            spv_id=record.spv_id,
            population=record.population,
            attribution=record.attribution,
            display_label=record.label,
            whole_tenant_book=record.whole_tenant_book,
            unpartitionable_reason=record.unpartitionable_reason,
        )

    # -- enumeration -------------------------------------------------------- #
    def refs(self, *, tenant_id: Optional[str] = None,
             kind: Optional[str] = None,
             enabled_only: bool = True) -> Tuple[ResourceRef, ...]:
        """Registered refs in a stable (key-sorted) order."""
        tenant = _clean(tenant_id)
        wanted_kind = _slug(kind)
        return tuple(
            record.ref for key in sorted(self._by_key)
            for record in (self._by_key[key],)
            if (tenant is None or record.ref.tenant_id == tenant)
            and (wanted_kind is None or record.ref.kind == wanted_kind)
            and (record.enabled or not enabled_only))

    def records(self) -> Tuple[ResourceRecord, ...]:
        return tuple(self._by_key[k] for k in sorted(self._by_key))

    def __len__(self) -> int:
        return len(self._by_key)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"ResourceCatalogue(configured={self.configured}, "
                f"resources={len(self._by_key)})")


def resources_from_portfolio_records(
    tenant_id: str,
    records: Iterable[PortfolioRecord],
    *,
    population: Optional[str] = None,
) -> List[ResourceRecord]:
    """One ``source_portfolio`` resource per governed portfolio.

    The bridge to the existing model: every ``source_portfolio_id`` the
    :class:`~trakt_core.portfolio.PortfolioRegistry` knows about is already a
    deterministic, mandatory-on-every-row boundary, so it is a resource without
    anyone having to write it into a config file.

    Pure — it takes records the caller already built and opens nothing.
    """
    out: List[ResourceRecord] = []
    for record in records:
        pid = _clean(record.portfolio_id)
        if not pid or not _valid_token(pid):
            continue
        out.append(ResourceRecord(
            ref=ResourceRef(tenant_id=tenant_id, kind=KIND_SOURCE_PORTFOLIO,
                            resource_id=pid),
            display_label=record.display_label,
            source_portfolio_ids=(pid,),
            population=population,
        ))
    return sorted(out, key=lambda r: r.ref.key)


def _record_from_spec(spec: Mapping[str, Any]) -> ResourceRecord:
    ids = list(spec.get("source_portfolio_ids") or [])
    single = spec.get("source_portfolio_id")
    if single:
        ids.append(single)
    return ResourceRecord(
        ref=ResourceRef(
            tenant_id=spec.get("tenant_id"),
            kind=spec.get("kind"),
            resource_id=spec.get("resource_id") or spec.get("id"),
        ),
        display_label=spec.get("display_label"),
        source_portfolio_ids=tuple(ids),
        spv_id=spec.get("spv_id"),
        population=spec.get("population"),
        unpartitionable=bool(spec.get("unpartitionable") or False),
        unpartitionable_reason=spec.get("unpartitionable_reason"),
        whole_tenant_book=bool(spec.get("whole_tenant_book") or False),
        enabled=bool(spec.get("enabled", True)),
    )


def load_resource_catalogue(
    config_path: Optional[str | Path] = None,
) -> ResourceCatalogue:
    """Load ``config/resources.yaml`` if present, else an empty catalogue.

    An absent file yields an empty, unconfigured catalogue in which every
    :meth:`ResourceCatalogue.resolve` raises ``RESOURCE_NOT_REGISTERED``. That is
    the correct default and costs nothing today: no route consults the catalogue
    yet, so an empty one changes no behaviour, and when the entitlement stage
    arrives "no catalogue" must mean "no resources", never "all of them".

    A file that exists but cannot be trusted — unparseable, duplicated, or
    declaring an unconstrained resource — also yields an empty catalogue, for the
    same reason ``load_organisation_registry`` does: degrade closed, and never
    raise, so a bad file cannot stop the process starting.
    """
    path = Path(config_path
                or os.environ.get(RESOURCE_CONFIG_ENV)
                or DEFAULT_RESOURCE_CONFIG)
    return config_cache.get_or_load(
        "resources", path, lambda: _load_resource_catalogue(path))


def _load_resource_catalogue(path: Path) -> ResourceCatalogue:
    """The uncached load. Behaviour is unchanged from before caching existed."""
    if not path.exists():
        return ResourceCatalogue((), configured=False)

    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        specs = list((raw or {}).get("resources") or [])
        records = [_record_from_spec(dict(s or {})) for s in specs]
    except Exception:  # noqa: BLE001 - a bad config fails closed, never raises
        logger.exception(
            "resource catalogue %s could not be parsed; no resources are "
            "registered until it is fixed", path)
        return ResourceCatalogue((), configured=True)

    try:
        return ResourceCatalogue(records, configured=True)
    except TraktError:
        logger.exception(
            "resource catalogue %s is invalid; no resources are registered "
            "until it is fixed", path)
        return ResourceCatalogue((), configured=True)
