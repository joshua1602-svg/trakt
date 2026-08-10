"""trakt_core.organisation — WHO IS ASKING, as distinct from whose data is served.

The distinction this module exists to make
------------------------------------------
``trakt_core.tenancy`` answers *whose data is this*: a Trakt tenant owns a set of
portfolios, and :func:`~trakt_core.tenancy.authorise_portfolio_access` decides
whether a context may read one. That model has exactly one party in it, because
until now the party asking and the party owning the data were always the same:
one deployment, one client, one Microsoft directory.

They are not always the same. A warehouse funder, an investor or a servicer signs
in to *its own* Microsoft directory and asks about *someone else's* book. Two
different facts are involved and the codebase previously had a name for only one:

  * **Trakt tenant** (``ExecutionContext.tenant_id``) — the data owner. Still
    deployment configuration. Nothing in this module changes it.
  * **Organisation** (this module) — the external party making the request,
    resolved from the Microsoft Entra directory their token was issued by.

    Microsoft Entra directory (validated ``tid``)  ──►  Organisation
                                                        (NOT a Trakt tenant)

Stage 1 establishes the first mapping only. An organisation carries no
entitlements, no resources and no capabilities yet; resolving one has no effect
on which data is served. It is identity, recorded for audit, and the foundation
the entitlement model will later hang off.

Two operating modes
-------------------
``configured is False`` — **compatibility mode.** No ``config/organisations.yaml``
exists, so this is the historical single-client deployment. No organisation
identity is claimed and :meth:`OrganisationRegistry.resolve_microsoft_tenant`
returns ``None``. Every existing caller behaves exactly as it did before.

``configured is True`` — **organisation mode.** A config was loaded, so the
deployment has declared which directories it serves. An unregistered or disabled
directory is refused. There is deliberately no permissive fallback in this mode:
that would make the control decorative.

Why a broken config fails closed here but falls back in ``tenancy``
-------------------------------------------------------------------
:func:`~trakt_core.tenancy.load_tenant_registry` deliberately degrades to
single-tenant on an unreadable file, because a broken tenancy file must not take
the API down and the dataset resolver still scopes reads to the context tenant.
The same degradation here would be a *silent widening*: a deployment that had
declared "these five directories" would quietly go back to accepting anything the
token layer accepted. So a config file that exists but cannot be trusted yields a
**strict registry with no organisations**, which refuses every resolution. The
process still starts, and paths that do not resolve an organisation (the React
Easy Auth path) are unaffected.

Configuration
-------------
``config/organisations.yaml`` (optional; see ``config/organisations.example.yaml``)::

    organisations:
      - organisation_id: ere
        display_name: Equity Release Europe
        organisation_type: originator
        microsoft_tenant_ids: ["<ere-directory-guid>"]
        enabled: true

No database in this pass, matching the existing ``config/tenancy.yaml`` and
``config/client/portfolio_registry.yaml`` convention.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from .errors import ErrorCode, TraktError

logger = logging.getLogger("trakt_core.organisation")

DEFAULT_ORGANISATION_CONFIG = "config/organisations.yaml"
ORGANISATION_CONFIG_ENV = "TRAKT_ORGANISATIONS_CONFIG"

# --------------------------------------------------------------------------- #
# Organisation types. Descriptive only — nothing branches on these in Stage 1.
# They exist so the registry records what an organisation *is* at the point it
# is registered, rather than that being inferred later from its entitlements.
# --------------------------------------------------------------------------- #
ORG_TYPE_ORIGINATOR = "originator"
ORG_TYPE_WAREHOUSE_FUNDER = "warehouse_funder"
ORG_TYPE_INVESTOR = "investor"
ORG_TYPE_SERVICER = "servicer"
ORG_TYPE_OPERATOR = "operator"
ORG_TYPE_UNKNOWN = "unknown"

KNOWN_ORGANISATION_TYPES: FrozenSet[str] = frozenset({
    ORG_TYPE_ORIGINATOR, ORG_TYPE_WAREHOUSE_FUNDER, ORG_TYPE_INVESTOR,
    ORG_TYPE_SERVICER, ORG_TYPE_OPERATOR, ORG_TYPE_UNKNOWN,
})


def normalise_directory_id(value: Any) -> Optional[str]:
    """A Microsoft Entra directory (tenant) id in comparable form.

    Directory ids are GUIDs and are case-insensitive, so every comparison in this
    module — registration, lookup and the token check — goes through here. A
    blank or missing value is ``None`` rather than ``""``, so "no directory" can
    never accidentally match a record registered with an empty string.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def normalise_organisation_id(value: Any) -> Optional[str]:
    """An organisation id in comparable form. Same rules as a directory id."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


@dataclass(frozen=True)
class OrganisationRecord:
    """One external organisation Trakt will accept requests from.

    Immutable, like every other identity contract in ``trakt_core``: a resolved
    organisation cannot be edited mid-execution to widen what a caller is.
    """

    organisation_id: str
    display_name: Optional[str] = None
    #: One of :data:`KNOWN_ORGANISATION_TYPES`. Recorded, never branched on yet.
    organisation_type: str = ORG_TYPE_UNKNOWN
    #: Microsoft Entra directory ids this organisation signs in from. Plural
    #: because one organisation may legitimately hold more than one directory
    #: (an acquisition, a separate test directory), and because assuming
    #: one-directory-per-organisation is the assumption this module exists to
    #: remove.
    microsoft_tenant_ids: FrozenSet[str] = field(default_factory=frozenset)
    #: False disables the organisation without deleting its registration, so an
    #: access decision has a record of what was withdrawn and when.
    enabled: bool = True

    def __post_init__(self) -> None:
        organisation_id = normalise_organisation_id(self.organisation_id)
        if not organisation_id:
            raise ValueError("OrganisationRecord requires an organisation_id")
        object.__setattr__(self, "organisation_id", organisation_id)
        object.__setattr__(self, "organisation_type",
                           (str(self.organisation_type or ORG_TYPE_UNKNOWN)
                            .strip().lower() or ORG_TYPE_UNKNOWN))
        directories = {normalise_directory_id(d) for d in (self.microsoft_tenant_ids or ())}
        directories.discard(None)
        object.__setattr__(self, "microsoft_tenant_ids", frozenset(directories))
        object.__setattr__(self, "enabled", bool(self.enabled))

    @property
    def label(self) -> str:
        return self.display_name or self.organisation_id

    def owns_directory(self, microsoft_tenant_id: Any) -> bool:
        directory = normalise_directory_id(microsoft_tenant_id)
        return directory is not None and directory in self.microsoft_tenant_ids

    def to_dict(self) -> Dict[str, Any]:
        """Non-sensitive projection. A directory id is the caller's own public
        identifier, not a credential, so it is safe to surface for operations."""
        return {
            "organisation_id": self.organisation_id,
            "display_name": self.display_name,
            "organisation_type": self.organisation_type,
            "microsoft_tenant_ids": sorted(self.microsoft_tenant_ids),
            "enabled": self.enabled,
        }


class OrganisationRegistry:
    """The organisations this deployment will accept requests from.

    Construction is the point at which an ambiguous registration is rejected: a
    directory id claimed by two organisations means "who is asking" has two
    answers, and a lookup that silently picked one would be an authorisation bug
    the moment entitlements arrive. Detection is deterministic — records are
    indexed in sorted organisation order, so the same file always names the same
    conflicting pair.
    """

    def __init__(self, records: Iterable[OrganisationRecord] | Mapping[str, OrganisationRecord],
                 *, configured: bool) -> None:
        if isinstance(records, Mapping):
            ordered = [records[k] for k in sorted(records)]
        else:
            ordered = sorted(records, key=lambda r: r.organisation_id)

        by_id: Dict[str, OrganisationRecord] = {}
        by_directory: Dict[str, OrganisationRecord] = {}
        for record in ordered:
            existing = by_id.get(record.organisation_id)
            if existing is not None:
                raise TraktError(
                    ErrorCode.INVALID_INPUT,
                    "The organisation registry declares the same organisation "
                    "more than once.",
                    details={"organisation_id": record.organisation_id})
            by_id[record.organisation_id] = record
            for directory in sorted(record.microsoft_tenant_ids):
                clash = by_directory.get(directory)
                if clash is not None:
                    raise TraktError(
                        ErrorCode.INVALID_INPUT,
                        "The same Microsoft directory is registered to more than "
                        "one organisation, so the caller's organisation cannot be "
                        "determined.",
                        details={"microsoft_tenant_id": directory,
                                 "organisation_ids": sorted(
                                     {clash.organisation_id, record.organisation_id})})
                by_directory[directory] = record

        self._by_id = by_id
        self._by_directory = by_directory
        #: True when an explicit organisation config was loaded — i.e. this
        #: deployment is in organisation mode and unknown directories are refused.
        #: False is the historical single-client deployment.
        self.configured = bool(configured)

    # -- lookup ------------------------------------------------------------- #
    def get(self, organisation_id: Any) -> Optional[OrganisationRecord]:
        key = normalise_organisation_id(organisation_id)
        return self._by_id.get(key) if key else None

    def for_microsoft_tenant(self, microsoft_tenant_id: Any) -> Optional[OrganisationRecord]:
        """The organisation registered for a directory, or ``None``.

        A pure lookup: it does not apply the ``enabled`` flag and does not raise.
        Use :meth:`resolve_microsoft_tenant` for an access decision.
        """
        directory = normalise_directory_id(microsoft_tenant_id)
        return self._by_directory.get(directory) if directory else None

    def resolve_microsoft_tenant(self, microsoft_tenant_id: Any, *,
                                 request_id: Optional[str] = None
                                 ) -> Optional[OrganisationRecord]:
        """The organisation for a **validated** Microsoft directory id.

        ``None`` means compatibility mode — no organisation config is loaded, so
        this deployment makes no claim about who is asking and every existing
        caller keeps its current behaviour.

        In organisation mode this raises rather than returning ``None``: an
        unregistered or disabled directory is a refusal, not an unknown.

        The caller must pass a directory id that came from a **signature-verified**
        token. This method does no token validation and cannot tell a verified
        claim from an asserted one.
        """
        if not self.configured:
            return None

        directory = normalise_directory_id(microsoft_tenant_id)
        if directory is None:
            raise TraktError(
                ErrorCode.ORGANISATION_NOT_REGISTERED,
                "This deployment serves registered organisations only, and the "
                "request carries no Microsoft directory identity.",
                request_id=request_id)

        record = self._by_directory.get(directory)
        if record is None:
            raise TraktError(
                ErrorCode.ORGANISATION_NOT_REGISTERED,
                "This Microsoft directory is not registered with Trakt.",
                request_id=request_id,
                details={"microsoft_tenant_id": directory})
        if not record.enabled:
            raise TraktError(
                ErrorCode.ORGANISATION_DISABLED,
                "This organisation's access to Trakt is currently disabled.",
                request_id=request_id,
                details={"organisation_id": record.organisation_id})
        return record

    # -- enumeration -------------------------------------------------------- #
    def organisation_ids(self) -> FrozenSet[str]:
        return frozenset(self._by_id)

    def microsoft_tenant_ids(self, *, enabled_only: bool = True) -> FrozenSet[str]:
        """Every registered directory id, for the token layer's allow-list."""
        return frozenset(
            directory for directory, record in self._by_directory.items()
            if record.enabled or not enabled_only)

    def records(self) -> Tuple[OrganisationRecord, ...]:
        """Registered organisations in a stable (id-sorted) order."""
        return tuple(self._by_id[k] for k in sorted(self._by_id))

    def __len__(self) -> int:
        return len(self._by_id)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"OrganisationRegistry(configured={self.configured}, "
                f"organisations={len(self._by_id)})")


def _record_from_spec(spec: Mapping[str, Any]) -> OrganisationRecord:
    """One config entry -> one record. Tolerates the singular spelling because a
    one-directory organisation is the common case and ``microsoft_tenant_id`` is
    the obvious thing for an operator to write."""
    directories: List[Any] = list(spec.get("microsoft_tenant_ids") or [])
    single = spec.get("microsoft_tenant_id")
    if single:
        directories.append(single)
    status = spec.get("enabled")
    if status is None:
        # ``status: active | disabled`` is accepted as an alternative spelling,
        # matching the source registry's vocabulary.
        raw_status = str(spec.get("status") or "").strip().lower()
        status = raw_status not in ("disabled", "inactive", "suspended", "false")
    return OrganisationRecord(
        organisation_id=spec.get("organisation_id") or spec.get("id"),
        display_name=spec.get("display_name"),
        organisation_type=spec.get("organisation_type") or ORG_TYPE_UNKNOWN,
        microsoft_tenant_ids=frozenset(
            d for d in (normalise_directory_id(x) for x in directories) if d),
        enabled=bool(status),
    )


def _parse_organisations(raw: Any) -> List[OrganisationRecord]:
    """Accept either a list of entries or a mapping keyed by organisation id."""
    block = (raw or {}).get("organisations") if isinstance(raw, Mapping) else None
    if not block:
        return []
    if isinstance(block, Mapping):
        specs = []
        for organisation_id, spec in block.items():
            merged = dict(spec or {})
            merged.setdefault("organisation_id", organisation_id)
            specs.append(merged)
    else:
        specs = [dict(s or {}) for s in block]
    return [_record_from_spec(s) for s in specs]


def load_organisation_registry(
    config_path: Optional[str | Path] = None,
) -> OrganisationRegistry:
    """Load ``config/organisations.yaml`` if present, else compatibility mode.

    Fail-closed, in the specific sense set out in the module docstring: a file
    that exists but cannot be trusted (unparseable, or declaring an ambiguous
    directory) yields an organisation-mode registry with **no** organisations,
    which refuses every resolution. It never degrades to compatibility mode,
    because that would silently re-open a deployment that had declared itself
    closed. It also never raises, so a bad file cannot stop the process starting
    and take the React path down with it.
    """
    path = Path(config_path
                or os.environ.get(ORGANISATION_CONFIG_ENV)
                or DEFAULT_ORGANISATION_CONFIG)
    if not path.exists():
        return OrganisationRegistry((), configured=False)

    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        records = _parse_organisations(raw)
    except Exception:  # noqa: BLE001 - an unreadable config must fail closed, not raise
        logger.exception(
            "organisation config %s could not be parsed; running in organisation "
            "mode with NO registered organisations, so every organisation "
            "resolution will be refused until it is fixed", path)
        return OrganisationRegistry((), configured=True)

    try:
        return OrganisationRegistry(records, configured=True)
    except TraktError:
        logger.exception(
            "organisation config %s is ambiguous; running in organisation mode "
            "with NO registered organisations, so every organisation resolution "
            "will be refused until it is fixed", path)
        return OrganisationRegistry((), configured=True)
