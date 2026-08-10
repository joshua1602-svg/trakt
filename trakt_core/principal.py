"""trakt_core.principal — WHICH PERSON, and which organisation they belong to.

Stage 1 resolves an organisation from the Microsoft directory a token came from:
one directory, one organisation, everybody in it. That is right for deciding
*which company is asking* and wrong as the only control, because a warehouse
funder is not one person. When someone leaves, the directory is still the
funder's directory; nothing in Trakt knew that individual existed, so nothing in
Trakt can stop serving them.

This module adds the missing binding, and only that:

    validated (tid, oid)  ──►  PrincipalBinding  ──►  organisation_id

Deliberately not an IAM product
-------------------------------
No groups, no roles, no per-user grants, no password or session anything.
Entitlements stay at the **organisation** level, which is where they belong: a
funder's twelve analysts should not need twelve copies of the same grant, and
duplicating them is how entitlement sets drift apart. A principal answers one
question — *is this individual still a member in good standing of that
organisation?* — and its ``status`` is an independent kill switch:

    organisation entitled  AND  principal active   ->  the organisation's grants
    organisation entitled  AND  principal disabled ->  nothing

Keyed on stable Microsoft identity
----------------------------------
The key is ``(microsoft_tenant_id, microsoft_object_id)`` — the directory and the
Entra **object id**, both of which are stable and neither of which the user
controls. Display name and email are stored for the operator's benefit and are
never consulted: a display name is user-editable in most directories, and an
email address is reassignable, so treating either as identity is an
authorisation bug. This mirrors the reasoning already applied in
``trakt_notifications.recipients``, which binds a Teams recipient the same way.

Per-directory enforcement, so Stage 1 keeps working
---------------------------------------------------
Registering principals must not silently lock out a directory nobody has
enumerated yet. Enforcement is therefore **opt-in per directory, by having
registered anybody at all**:

  * a directory with **no** bindings resolves exactly as it did in Stage 1 —
    the organisation comes from the directory and no individual is checked;
  * a directory with **one or more** bindings is enforced — an unregistered or
    disabled object id in it is refused.

So the ERE directory keeps working untouched while Funder A's directory becomes
principal-gated the moment its first user is added, and turning enforcement on
for a directory is a visible act (registering someone) rather than a flag
somebody has to remember.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from .errors import ErrorCode, TraktError
from .organisation import normalise_directory_id, normalise_organisation_id

logger = logging.getLogger("trakt_core.principal")

DEFAULT_PRINCIPAL_CONFIG = "config/principals.yaml"
PRINCIPAL_CONFIG_ENV = "TRAKT_PRINCIPALS_CONFIG"

#: A principal in good standing.
PRINCIPAL_ACTIVE = "active"
#: Access withdrawn. The binding is kept so the person stays attributable in the
#: audit trail and can be restored without being re-identified from scratch.
PRINCIPAL_DISABLED = "disabled"
KNOWN_PRINCIPAL_STATUSES: FrozenSet[str] = frozenset({
    PRINCIPAL_ACTIVE, PRINCIPAL_DISABLED})


def normalise_object_id(value: Any) -> Optional[str]:
    """An Entra object id in comparable form. GUIDs are case-insensitive."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


@dataclass(frozen=True)
class PrincipalBinding:
    """One individual, bound to one organisation by stable Microsoft identity."""

    organisation_id: str
    microsoft_tenant_id: str
    microsoft_object_id: str
    #: Operator-facing only. Never consulted for an access decision — see the
    #: module docstring on why an email or display name must not be identity.
    display_name: Optional[str] = None
    email: Optional[str] = None
    status: str = PRINCIPAL_ACTIVE

    def __post_init__(self) -> None:
        organisation_id = normalise_organisation_id(self.organisation_id)
        directory = normalise_directory_id(self.microsoft_tenant_id)
        object_id = normalise_object_id(self.microsoft_object_id)
        if not organisation_id:
            raise ValueError("PrincipalBinding requires an organisation_id")
        if not directory:
            raise ValueError("PrincipalBinding requires a microsoft_tenant_id")
        if not object_id:
            raise ValueError("PrincipalBinding requires a microsoft_object_id")
        status = (str(self.status or PRINCIPAL_ACTIVE).strip().lower()
                  or PRINCIPAL_ACTIVE)
        if status not in KNOWN_PRINCIPAL_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(KNOWN_PRINCIPAL_STATUSES)}, "
                f"got {status!r}")
        object.__setattr__(self, "organisation_id", organisation_id)
        object.__setattr__(self, "microsoft_tenant_id", directory)
        object.__setattr__(self, "microsoft_object_id", object_id)
        object.__setattr__(self, "status", status)

    @property
    def key(self) -> Tuple[str, str]:
        """The identity key: a directory plus an object id within it."""
        return (self.microsoft_tenant_id, self.microsoft_object_id)

    @property
    def active(self) -> bool:
        return self.status == PRINCIPAL_ACTIVE

    @property
    def label(self) -> str:
        return self.display_name or self.microsoft_object_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "organisation_id": self.organisation_id,
            "microsoft_tenant_id": self.microsoft_tenant_id,
            "microsoft_object_id": self.microsoft_object_id,
            "display_name": self.display_name,
            "email": self.email,
            "status": self.status,
        }


class PrincipalRegistry:
    """The individuals this deployment recognises, keyed by (directory, oid).

    Rejects a duplicate ``(microsoft_tenant_id, microsoft_object_id)`` at
    construction: the same person bound to two organisations makes "who is
    asking" ambiguous, exactly as a directory claimed by two organisations does.
    """

    def __init__(self, bindings: Iterable[PrincipalBinding], *,
                 configured: bool) -> None:
        ordered = sorted(bindings, key=lambda b: b.key)
        by_key: Dict[Tuple[str, str], PrincipalBinding] = {}
        for binding in ordered:
            clash = by_key.get(binding.key)
            if clash is not None:
                raise TraktError(
                    ErrorCode.INVALID_INPUT,
                    "The same Microsoft principal is bound to more than one "
                    "organisation, so the caller's organisation cannot be "
                    "determined.",
                    details={"microsoft_tenant_id": binding.microsoft_tenant_id,
                             "microsoft_object_id": binding.microsoft_object_id,
                             "organisation_ids": sorted(
                                 {clash.organisation_id,
                                  binding.organisation_id})})
            by_key[binding.key] = binding
        self._by_key = by_key
        #: Directories that have at least one binding — the ones enforcement is
        #: switched on for. Derived, so turning it on is an act (registering
        #: somebody) rather than a flag.
        self._enforced = frozenset(d for d, _oid in by_key)
        self.configured = bool(configured)

    # -- lookup ------------------------------------------------------------- #
    def get(self, microsoft_tenant_id: Any,
            microsoft_object_id: Any) -> Optional[PrincipalBinding]:
        """A pure lookup: ignores ``status``, never raises."""
        directory = normalise_directory_id(microsoft_tenant_id)
        object_id = normalise_object_id(microsoft_object_id)
        if not directory or not object_id:
            return None
        return self._by_key.get((directory, object_id))

    def enforced_for(self, microsoft_tenant_id: Any) -> bool:
        """Whether individuals are checked for this directory."""
        directory = normalise_directory_id(microsoft_tenant_id)
        return bool(self.configured and directory and directory in self._enforced)

    def resolve(self, microsoft_tenant_id: Any, microsoft_object_id: Any, *,
                request_id: Optional[str] = None) -> Optional[PrincipalBinding]:
        """The binding for a **validated** identity, or ``None`` when this
        directory is not principal-gated.

        ``None`` means "no individual check applies here" — the Stage 1
        behaviour, preserved. It never means "allowed": the caller still has to
        resolve an organisation the ordinary way.

        For an enforced directory an unregistered or disabled individual raises.
        """
        if not self.enforced_for(microsoft_tenant_id):
            return None

        binding = self.get(microsoft_tenant_id, microsoft_object_id)
        if binding is None:
            raise TraktError(
                ErrorCode.PRINCIPAL_NOT_REGISTERED,
                "This user is not registered with Trakt.",
                request_id=request_id,
                details={"microsoft_tenant_id":
                         normalise_directory_id(microsoft_tenant_id)})
        if not binding.active:
            raise TraktError(
                ErrorCode.PRINCIPAL_DISABLED,
                "This user's access to Trakt has been withdrawn.",
                request_id=request_id,
                details={"organisation_id": binding.organisation_id})
        return binding

    # -- enumeration -------------------------------------------------------- #
    def for_organisation(self, organisation_id: Any, *,
                         active_only: bool = False) -> Tuple[PrincipalBinding, ...]:
        wanted = normalise_organisation_id(organisation_id)
        return tuple(
            self._by_key[k] for k in sorted(self._by_key)
            if self._by_key[k].organisation_id == wanted
            and (self._by_key[k].active or not active_only))

    def organisation_for(self, microsoft_tenant_id: Any,
                         microsoft_object_id: Any) -> Optional[str]:
        binding = self.get(microsoft_tenant_id, microsoft_object_id)
        return binding.organisation_id if binding else None

    def bindings(self) -> Tuple[PrincipalBinding, ...]:
        return tuple(self._by_key[k] for k in sorted(self._by_key))

    def enforced_directories(self) -> FrozenSet[str]:
        return self._enforced if self.configured else frozenset()

    def __len__(self) -> int:
        return len(self._by_key)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"PrincipalRegistry(configured={self.configured}, "
                f"principals={len(self._by_key)})")


def _binding_from_spec(spec: Mapping[str, Any]) -> PrincipalBinding:
    return PrincipalBinding(
        organisation_id=spec.get("organisation_id"),
        microsoft_tenant_id=spec.get("microsoft_tenant_id"),
        microsoft_object_id=(spec.get("microsoft_object_id")
                             or spec.get("entra_object_id")
                             or spec.get("object_id")),
        display_name=spec.get("display_name"),
        email=spec.get("email"),
        status=spec.get("status") or (
            PRINCIPAL_ACTIVE if spec.get("enabled", True) else PRINCIPAL_DISABLED),
    )


def load_principal_registry(
    config_path: Optional[str | Path] = None,
) -> PrincipalRegistry:
    """Load ``config/principals.yaml`` if present, else a dormant registry.

    No file means no directory is principal-gated and Stage 1 resolution is
    entirely unchanged.

    A file that exists but cannot be trusted yields a **configured but empty**
    registry. Empty means no directory has bindings, so nothing is enforced and
    organisation resolution still works — deliberately different from the
    organisation registry, where an untrusted file refuses everything. The
    reason is the direction of the failure: an unreadable *organisation* file
    means Trakt cannot tell who is asking and must refuse, whereas an unreadable
    *principal* file means Trakt has lost a narrowing control it did not have at
    all before Stage 3.5. Failing every request closed on the second would be an
    outage caused by a feature that is meant to be additive, and the loud error
    log plus the refusal to enforce is the honest state.
    """
    path = Path(config_path
                or os.environ.get(PRINCIPAL_CONFIG_ENV)
                or DEFAULT_PRINCIPAL_CONFIG)
    if not path.exists():
        return PrincipalRegistry((), configured=False)

    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        specs = list((raw or {}).get("principals") or [])
        bindings = [_binding_from_spec(dict(s or {})) for s in specs]
    except Exception:  # noqa: BLE001 - a bad config must never raise upward
        logger.exception(
            "principal config %s could not be parsed; NO directory is "
            "principal-gated until it is fixed", path)
        return PrincipalRegistry((), configured=True)

    try:
        return PrincipalRegistry(bindings, configured=True)
    except TraktError:
        logger.exception(
            "principal config %s is ambiguous; NO directory is principal-gated "
            "until it is fixed", path)
        return PrincipalRegistry((), configured=True)
