"""trakt_core.entitlement — WHO may do WHAT to WHICH resource.

The third and final identity axis. The three now compose into the relationship
the target architecture asked for:

    organisation  ×  resource  ×  capability

    trakt_core.organisation   WHO IS ASKING       (Stage 1)
    trakt_core.resource       WHAT IS THE THING   (Stage 2)
    trakt_core.entitlement    MAY THEY?           (this module)
    trakt_core.tenancy        WHOSE DATA IS IT    (unchanged throughout)

A grant is server-side configuration and nothing else. There is no code path
through which a caller creates, widens or influences one: grants are read from
``config/entitlements.yaml``, validated against the organisation registry and the
resource catalogue at load, resolved **once** at context construction, and frozen
onto the :class:`~trakt_core.context.ExecutionContext`.

Explicit grants, never inferred roles
-------------------------------------
An organisation's *type* — ``warehouse_funder``, ``investor`` — informs nothing
at runtime. It may seed a default when an operator writes the config, but the
security decision comes from a written grant naming a resource. "Funder A funds
SPV I" is a commercial fact; "funder_a may run risk:read against ERE/spv/spv_i"
is an authorisation fact, and only the second one is consulted. Collapsing the
two would make every funder relationship an implicit access grant.

Capabilities are per resource, not per organisation
---------------------------------------------------
The warehouse case requires it: Funder A holds ``mi:query`` and ``risk:read``
against SPV I and *nothing at all* against SPV II. A single organisation-level
capability set could not express that without also deciding it applied
everywhere, so a resolved entitlement is a mapping::

    ResourceRef -> frozenset(capabilities)

Two axes, both enforced
-----------------------
``ExecutionContext.scopes`` says what this caller may do *at all* (channel and
role). A grant says what this organisation may do *to one resource*. They use the
same vocabulary (:data:`~trakt_core.context.KNOWN_CAPABILITIES`) so nothing has
to be translated, and :func:`authorise_resource_access` requires both. The
effective permission is the intersection, which is the conservative direction.

Dormant until configured
------------------------
With no ``config/entitlements.yaml`` the store is unconfigured, every resolution
yields ``None``, and ``ExecutionContext.entitlements`` stays ``None``. Every
existing caller — React, Copilot, decks, internal jobs — is unchanged, because
they authorise through :func:`trakt_core.tenancy.authorise_portfolio_access`,
which this module does not touch.

Crucially, ``None`` is not "unlimited". :func:`authorise_resource_access` refuses
a context with no entitlements, so absence of configuration can never become
absence of restriction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple, Union,
)

from .context import KNOWN_CAPABILITIES, ExecutionContext
from .errors import ErrorCode, TraktError
from .organisation import OrganisationRegistry, load_organisation_registry
from .resource import (
    ResolvedResource,
    ResourceCatalogue,
    ResourceRef,
    load_resource_catalogue,
)

logger = logging.getLogger("trakt_core.entitlement")

DEFAULT_ENTITLEMENT_CONFIG = "config/entitlements.yaml"
ENTITLEMENT_CONFIG_ENV = "TRAKT_ENTITLEMENTS_CONFIG"

#: A grant is live only in this status. Anything else — notably ``proposed`` —
#: is inert at runtime.
#:
#: This is the seam the OCC Agent will later write through: it may propose a
#: grant by adding a ``status: proposed`` entry, and a human turns it into
#: ``status: active``. The runtime never asks the OCC Agent anything; it reads
#: the file. That keeps authorisation deterministic and keeps an agent off the
#: critical path of an access decision.
STATUS_ACTIVE = "active"
STATUS_PROPOSED = "proposed"
STATUS_REVOKED = "revoked"
KNOWN_GRANT_STATUSES: FrozenSet[str] = frozenset({
    STATUS_ACTIVE, STATUS_PROPOSED, STATUS_REVOKED})


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# --------------------------------------------------------------------------- #
# The grant
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grant:
    """One organisation's permission over one resource.

    Immutable, and validated by :class:`EntitlementStore` against the
    organisation registry and resource catalogue before it can be part of a
    store — so a grant that exists is a grant whose subject, object and verbs all
    resolved.
    """

    organisation_id: str
    resource_ref: ResourceRef
    capabilities: FrozenSet[str] = field(default_factory=frozenset)
    status: str = STATUS_ACTIVE
    enabled: bool = True
    #: Recorded for governance review, never consulted for an access decision.
    #: ``source`` distinguishes an operator-written grant from one an agent
    #: proposed; ``approved_by`` / ``approved_at`` are the human sign-off.
    source: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    note: Optional[str] = None

    def __post_init__(self) -> None:
        organisation_id = _clean(self.organisation_id)
        if not organisation_id:
            raise ValueError("Grant requires an organisation_id")
        object.__setattr__(self, "organisation_id", organisation_id.lower())
        if not isinstance(self.resource_ref, ResourceRef):
            raise ValueError("Grant requires a ResourceRef")
        caps = frozenset(str(c).strip().lower() for c in (self.capabilities or ())
                         if _clean(c))
        object.__setattr__(self, "capabilities", caps)
        status = (_clean(self.status) or STATUS_ACTIVE).lower()
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "enabled", bool(self.enabled))

    @property
    def active(self) -> bool:
        """Whether this grant confers anything right now.

        Both switches must be on. ``status`` carries the approval lifecycle
        (a proposal is not an approval); ``enabled`` is the operational
        on/off that keeps a registration and its history while withdrawing it.
        """
        return self.enabled and self.status == STATUS_ACTIVE

    @property
    def key(self) -> Tuple[str, str]:
        """The uniqueness key: one grant per (organisation, resource)."""
        return (self.organisation_id, self.resource_ref.key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "organisation_id": self.organisation_id,
            "resource": self.resource_ref.key,
            "capabilities": sorted(self.capabilities),
            "status": self.status,
            "enabled": self.enabled,
            "source": self.source,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
        }


# --------------------------------------------------------------------------- #
# Resolved state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolvedEntitlements:
    """One organisation's permissions, frozen for the life of a request.

    Held as a sorted tuple of pairs rather than a dict so the whole value is
    genuinely immutable — a frozen dataclass wrapping a dict would still let a
    holder mutate the mapping and quietly widen its own authority mid-execution.
    """

    organisation_id: str
    #: ``((ResourceRef, frozenset(capabilities)), …)`` sorted by resource key.
    entries: Tuple[Tuple[ResourceRef, FrozenSet[str]], ...] = ()
    #: Deterministic fingerprint of exactly these grants.
    version: str = ""

    @property
    def permitted_resources(self) -> FrozenSet[ResourceRef]:
        return frozenset(ref for ref, _caps in self.entries)

    def capabilities_for(self, resource_ref: ResourceRef) -> FrozenSet[str]:
        for ref, caps in self.entries:
            if ref == resource_ref:
                return caps
        return frozenset()

    def allows(self, resource_ref: ResourceRef, capability: str) -> bool:
        return str(capability) in self.capabilities_for(resource_ref)

    def is_empty(self) -> bool:
        return not self.entries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "organisation_id": self.organisation_id,
            "version": self.version,
            "resources": {ref.key: sorted(caps) for ref, caps in self.entries},
        }


def _fingerprint(payload: Any) -> str:
    """A stable content hash. ``hashlib``, not ``hash()``: the value goes into
    audit lines and later into cache keys, so it has to be identical across
    processes and runs."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "ent_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class AuthorisedResource:
    """Proof that a context may perform one capability against one resource.

    The counterpart to :class:`~trakt_core.tenancy.AuthorisedPortfolio`, and it
    exists for the same reason: a downstream service should require *this* rather
    than a resource id string, so there is no code path from a caller-supplied
    name to data that skipped the check.
    """

    resource: ResolvedResource
    capability: str
    organisation_id: str
    tenant_id: str
    entitlement_version: Optional[str] = None

    @property
    def ref(self) -> ResourceRef:
        return self.resource.ref

    def predicate(self) -> Dict[str, Any]:
        """The governed filter this authorisation permits. Delegates to the
        Stage 2 resource, so the never-empty invariant holds here too."""
        return self.resource.predicate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource.to_dict(),
            "capability": self.capability,
            "organisation_id": self.organisation_id,
            "tenant_id": self.tenant_id,
            "entitlement_version": self.entitlement_version,
        }


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #
class EntitlementStore:
    """Every grant this deployment honours.

    Validation happens at construction, against the real registries, so an
    unresolvable grant cannot sit in the store waiting to fail confusingly at
    request time. A grant is rejected when:

      * its organisation is not registered, or is disabled;
      * its resource is not in the catalogue, or is disabled;
      * its resource is **unpartitionable** — the data cannot enforce that
        boundary, so granting it would mean granting more than it names;
      * it names a capability outside :data:`KNOWN_CAPABILITIES`;
      * it duplicates another grant's ``(organisation, resource)``.

    Duplicates are **rejected rather than merged**. Unioning two grants for the
    same pair would silently produce access neither entry asked for, which is
    precisely the "contradictory grants must not widen access" case.
    """

    def __init__(
        self,
        grants: Iterable[Grant],
        *,
        configured: bool,
        organisations: Optional[OrganisationRegistry] = None,
        resources: Optional[ResourceCatalogue] = None,
        validate: bool = True,
    ) -> None:
        ordered = sorted(grants, key=lambda g: g.key)
        by_key: Dict[Tuple[str, str], Grant] = {}
        for grant in ordered:
            if grant.key in by_key:
                raise TraktError(
                    ErrorCode.INVALID_INPUT,
                    "The entitlement configuration grants the same organisation "
                    "the same resource more than once. Merging them would widen "
                    "access beyond what either entry states.",
                    details={"organisation_id": grant.organisation_id,
                             "resource": grant.resource_ref.key})
            if validate:
                self._validate(grant, organisations=organisations, resources=resources)
            by_key[grant.key] = grant

        self._by_key = by_key
        self._by_organisation: Dict[str, List[Grant]] = {}
        for grant in (by_key[k] for k in sorted(by_key)):
            self._by_organisation.setdefault(grant.organisation_id, []).append(grant)
        #: True when an explicit entitlement config was loaded.
        self.configured = bool(configured)

    # -- validation --------------------------------------------------------- #
    @staticmethod
    def _validate(grant: Grant, *, organisations: Optional[OrganisationRegistry],
                  resources: Optional[ResourceCatalogue]) -> None:
        unknown_caps = sorted(grant.capabilities - KNOWN_CAPABILITIES)
        if unknown_caps:
            raise TraktError(
                ErrorCode.INVALID_INPUT,
                "The entitlement configuration names a capability Trakt does "
                "not recognise.",
                details={"organisation_id": grant.organisation_id,
                         "resource": grant.resource_ref.key,
                         "unknown_capabilities": unknown_caps})
        if not grant.capabilities:
            raise TraktError(
                ErrorCode.INVALID_INPUT,
                "A grant must name at least one capability. An empty grant "
                "conveys nothing and is more likely a mistake than an intent.",
                details={"organisation_id": grant.organisation_id,
                         "resource": grant.resource_ref.key})

        if organisations is not None and organisations.configured:
            record = organisations.get(grant.organisation_id)
            if record is None:
                raise TraktError(
                    ErrorCode.INVALID_INPUT,
                    "The entitlement configuration grants access to an "
                    "organisation that is not registered.",
                    details={"organisation_id": grant.organisation_id})
            if not record.enabled:
                raise TraktError(
                    ErrorCode.INVALID_INPUT,
                    "The entitlement configuration grants access to a disabled "
                    "organisation.",
                    details={"organisation_id": grant.organisation_id})

        if resources is not None:
            record = resources.get(grant.resource_ref)
            if record is None:
                raise TraktError(
                    ErrorCode.INVALID_INPUT,
                    "The entitlement configuration grants a resource that is "
                    "not in the resource catalogue.",
                    details={"resource": grant.resource_ref.key})
            if not record.enabled:
                raise TraktError(
                    ErrorCode.INVALID_INPUT,
                    "The entitlement configuration grants a disabled resource.",
                    details={"resource": grant.resource_ref.key})
            if record.unpartitionable:
                # The SPV II case. The resource names a real commercial thing
                # that the data cannot separate, so a grant over it would hand
                # over whatever the enclosing book turns out to be.
                raise TraktError(
                    ErrorCode.INVALID_INPUT,
                    "This resource cannot be separated from the rest of the "
                    "book with the attribution the data carries, so it cannot "
                    "be granted to anyone.",
                    details={"resource": grant.resource_ref.key,
                             "reason": record.unpartitionable_reason})

    # -- lookup ------------------------------------------------------------- #
    def grants_for(self, organisation_id: Optional[str], *,
                   active_only: bool = True) -> Tuple[Grant, ...]:
        key = (_clean(organisation_id) or "").lower()
        if not key:
            return ()
        return tuple(g for g in self._by_organisation.get(key, ())
                     if g.active or not active_only)

    def resolve(self, organisation_id: Optional[str]) -> Optional[ResolvedEntitlements]:
        """Freeze one organisation's permissions for a request.

        ``None`` means the entitlement model is **not in play** — no config was
        loaded. It does not mean "no restrictions": the authorisation helper
        refuses a context whose entitlements are ``None``.

        A registered organisation with no grants resolves to an EMPTY set, which
        is a real answer and refuses everything.
        """
        if not self.configured:
            return None
        organisation = (_clean(organisation_id) or "").lower()
        if not organisation:
            return None
        entries = tuple(
            (g.resource_ref, g.capabilities)
            for g in sorted(self.grants_for(organisation),
                            key=lambda g: g.resource_ref.key))
        return ResolvedEntitlements(
            organisation_id=organisation,
            entries=entries,
            version=_fingerprint([
                {"resource": ref.key, "capabilities": sorted(caps)}
                for ref, caps in entries]),
        )

    # -- enumeration -------------------------------------------------------- #
    @property
    def version(self) -> str:
        """Fingerprint of the whole store, for deployment-level diagnostics."""
        return _fingerprint([g.to_dict() for g in self.grants()])

    def grants(self) -> Tuple[Grant, ...]:
        return tuple(self._by_key[k] for k in sorted(self._by_key))

    def organisation_ids(self) -> FrozenSet[str]:
        return frozenset(self._by_organisation)

    def __len__(self) -> int:
        return len(self._by_key)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"EntitlementStore(configured={self.configured}, "
                f"grants={len(self._by_key)})")


# --------------------------------------------------------------------------- #
# The authorisation decision
# --------------------------------------------------------------------------- #
def authorise_resource_access(
    context: ExecutionContext,
    resource: Union[str, ResourceRef],
    required_capability: str,
    *,
    catalogue: Optional[ResourceCatalogue] = None,
) -> AuthorisedResource:
    """Authorise ``required_capability`` against ``resource`` for ``context``.

    The order of the checks is part of the design:

      1. **scope** — does this caller hold the capability at all?
      2. **parse** — a caller-supplied string becomes a :class:`ResourceRef` or
         is refused. Malformed input is never coerced into something broader.
      3. **entitlement present** — a context with none is refused. Absence of
         configuration is not absence of restriction.
      4. **membership** — is the ref in the frozen permitted set?
      5. **capability** — is it granted *against that resource*?
      6. **tenant** — does the resource belong to the tenant this deployment
         serves?
      7. **catalogue** — only now is the resource expanded.

    Steps 4 and 5 run **before** step 7 deliberately: an ungranted resource is
    refused identically whether or not it exists, so a caller cannot enumerate
    another organisation's resources by comparing error codes.

    No dataframe is touched. There is no branch that returns a permissive default
    and none that widens an unrecognised resource to Total — every failure path
    raises.
    """
    # 1. The channel/role gate. Cheap, and it means a caller who was never given
    #    a capability cannot reach the entitlement lookup at all.
    context.require_scope(required_capability)

    # 2. Caller input becomes a checked identity, or nothing.
    if isinstance(resource, ResourceRef):
        ref = resource
    else:
        try:
            ref = ResourceRef.parse(str(resource))
        except ValueError as exc:
            raise TraktError(
                ErrorCode.INVALID_INPUT,
                "That is not a valid resource reference.",
                request_id=context.request_id) from exc

    capability = str(required_capability).strip().lower()

    # 3. No entitlements resolved -> no resource access. Never a fallback.
    if context.entitlements is None:
        raise TraktError(
            ErrorCode.ENTITLEMENTS_NOT_RESOLVED,
            "This request carries no resolved entitlements, so no resource can "
            "be authorised for it.",
            request_id=context.request_id)

    # 4/5. Membership and capability, both against the FROZEN set. Note these
    #      precede the catalogue lookup: "not granted" and "does not exist" are
    #      indistinguishable to the caller.
    granted = context.entitlements.capabilities_for(ref)
    if not granted:
        raise TraktError(
            ErrorCode.RESOURCE_NOT_AUTHORISED,
            "This organisation is not authorised for that resource.",
            request_id=context.request_id, details={"resource": ref.key})
    if capability not in granted:
        raise TraktError(
            ErrorCode.CAPABILITY_NOT_GRANTED,
            "This organisation is not authorised to perform that action on the "
            "resource.",
            request_id=context.request_id,
            details={"resource": ref.key, "capability": capability})

    # 6. Defence in depth: a grant must not reach a tenant this deployment does
    #    not serve, whatever the config says.
    if ref.tenant_id != context.tenant_id:
        raise TraktError(
            ErrorCode.TENANT_MISMATCH,
            "That resource belongs to a different tenant.",
            request_id=context.request_id, details={"resource": ref.key})

    # 7. Expand. The catalogue validated this resource at grant-load time, so a
    #    miss here means the config changed under a live context — refuse.
    cat = catalogue if catalogue is not None else load_resource_catalogue()
    resolved = cat.resolve(ref, request_id=context.request_id)

    return AuthorisedResource(
        resource=resolved,
        capability=capability,
        organisation_id=context.entitlements.organisation_id,
        tenant_id=context.tenant_id,
        entitlement_version=context.entitlement_version,
    )


def permitted_resources_for(
    context: ExecutionContext,
    *,
    capability: Optional[str] = None,
) -> Tuple[ResourceRef, ...]:
    """The resources this context may reach, optionally filtered by capability.

    The read-only counterpart to :func:`authorise_resource_access`, for a caller
    that needs to *offer* choices rather than check one — a resource picker, or
    the tool that will later tell an AI client what it is allowed to ask about.
    Returning the closed set is what keeps such a client from having to guess an
    identifier and be refused.
    """
    if context.entitlements is None:
        return ()
    wanted = str(capability).strip().lower() if capability else None
    return tuple(
        ref for ref, caps in context.entitlements.entries
        if wanted is None or wanted in caps)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _grant_from_spec(spec: Mapping[str, Any]) -> Grant:
    raw_resource = spec.get("resource") or spec.get("resource_ref")
    if isinstance(raw_resource, Mapping):
        ref = ResourceRef(tenant_id=raw_resource.get("tenant_id"),
                          kind=raw_resource.get("kind"),
                          resource_id=raw_resource.get("resource_id"))
    else:
        ref = ResourceRef.parse(str(raw_resource or ""))
    return Grant(
        organisation_id=spec.get("organisation_id"),
        resource_ref=ref,
        capabilities=frozenset(spec.get("capabilities") or ()),
        status=spec.get("status") or STATUS_ACTIVE,
        enabled=bool(spec.get("enabled", True)),
        source=spec.get("source"),
        approved_by=spec.get("approved_by"),
        approved_at=spec.get("approved_at"),
        note=spec.get("note"),
    )


def load_entitlement_store(
    config_path: Optional[str | Path] = None,
    *,
    organisations: Optional[OrganisationRegistry] = None,
    resources: Optional[ResourceCatalogue] = None,
) -> EntitlementStore:
    """Load ``config/entitlements.yaml`` if present, else a dormant store.

    No file means an unconfigured store: :meth:`EntitlementStore.resolve` returns
    ``None`` and the entitlement model stays out of the way entirely. That is the
    current ERE shape.

    A file that exists but cannot be trusted — unparseable, or carrying any
    invalid grant — yields a **configured but empty** store, so every
    organisation resolves to an empty permitted set and everything is refused.
    The whole file is rejected rather than partially applied: applying the
    grants that happened to parse would mean a typo silently changes who can see
    what, which is worse than a visible outage of the new surface. It never
    raises, so a bad file cannot stop the process starting or disturb the legacy
    path.
    """
    path = Path(config_path
                or os.environ.get(ENTITLEMENT_CONFIG_ENV)
                or DEFAULT_ENTITLEMENT_CONFIG)
    if not path.exists():
        return EntitlementStore((), configured=False)

    orgs = organisations if organisations is not None else load_organisation_registry()
    cat = resources if resources is not None else load_resource_catalogue()

    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        specs = list((raw or {}).get("entitlements") or [])
        grants = [_grant_from_spec(dict(s or {})) for s in specs]
    except Exception:  # noqa: BLE001 - a bad config fails closed, never raises
        logger.exception(
            "entitlement config %s could not be parsed; NO grants are active "
            "until it is fixed", path)
        return EntitlementStore((), configured=True)

    try:
        return EntitlementStore(grants, configured=True,
                                organisations=orgs, resources=cat)
    except TraktError:
        logger.exception(
            "entitlement config %s is invalid; NO grants are active until it "
            "is fixed", path)
        return EntitlementStore((), configured=True)


def resolve_entitlements(
    organisation_id: Optional[str],
    *,
    store: Optional[EntitlementStore] = None,
) -> Optional[ResolvedEntitlements]:
    """Resolve one organisation's frozen entitlement state, or ``None``.

    ``None`` when there is no organisation (the channel carries no directory
    identity) or no entitlement config — the two ways today's deployments stay
    on the legacy path.
    """
    if not _clean(organisation_id):
        return None
    entitlement_store = store if store is not None else load_entitlement_store()
    return entitlement_store.resolve(organisation_id)
