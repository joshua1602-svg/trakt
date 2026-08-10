"""Structured access changes — a proposal an operator can read, not a YAML diff.

Why a change set rather than editing the files
-----------------------------------------------
The OCC already has a generic "draft the YAML, validate it, activate it" path for
its system/regime/asset packages. That is the right mechanism and this module
reuses it — but it is the wrong *interface* for access, for two reasons:

* An agent that emits YAML emits arbitrary YAML. Reviewing "is this diff safe"
  is a harder job than reviewing "ADD_GRANT funder_a -> ERE/spv/spv_i, four read
  capabilities", and the harder job is the one an approver will do badly at 5pm.
* An access change is usually several coordinated edits across three files. A
  diff shows the edits; a change set shows the *intent*, which is what an
  operator is actually being asked to agree to.

So the OCC Agent produces :class:`AccessChangeSet` — a typed, enumerable list of
actions — and this module turns it into new file contents deterministically.
The agent never writes YAML and never writes to the store.

Every action is total and pure
------------------------------
:func:`apply_change_set` takes a document and returns a **new** one; it does not
mutate its input, touch the filesystem, or reach for storage. An action that
cannot be applied raises rather than silently doing nothing, so a change set is
either wholly applicable or refused — no partial application, which is the
property that keeps "add organisation + add grant" from leaving a grant pointing
at nothing.

Applying is not approving. The result of :func:`apply_change_set` is a candidate
that still has to pass :mod:`~operations_control.access_admin.validation` and be
confirmed by a human before it becomes active configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..contracts import canonical_json, new_id, now_iso, stable_hash
from .document import AccessDocument, _resource_key

# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
ADD_ORGANISATION = "ADD_ORGANISATION"
DISABLE_ORGANISATION = "DISABLE_ORGANISATION"
ENABLE_ORGANISATION = "ENABLE_ORGANISATION"
ADD_MICROSOFT_TENANT = "ADD_MICROSOFT_TENANT"

ADD_PRINCIPAL = "ADD_PRINCIPAL"
DISABLE_PRINCIPAL = "DISABLE_PRINCIPAL"
ENABLE_PRINCIPAL = "ENABLE_PRINCIPAL"

ADD_RESOURCE = "ADD_RESOURCE"
UPDATE_RESOURCE = "UPDATE_RESOURCE"

ADD_GRANT = "ADD_GRANT"
REVOKE_GRANT = "REVOKE_GRANT"
SET_GRANT_CAPABILITIES = "SET_GRANT_CAPABILITIES"
DISABLE_GRANT = "DISABLE_GRANT"
ENABLE_GRANT = "ENABLE_GRANT"

ACTIONS: Tuple[str, ...] = (
    ADD_ORGANISATION, DISABLE_ORGANISATION, ENABLE_ORGANISATION,
    ADD_MICROSOFT_TENANT,
    ADD_PRINCIPAL, DISABLE_PRINCIPAL, ENABLE_PRINCIPAL,
    ADD_RESOURCE, UPDATE_RESOURCE,
    ADD_GRANT, REVOKE_GRANT, SET_GRANT_CAPABILITIES,
    DISABLE_GRANT, ENABLE_GRANT,
)

#: Actions that materially change who can reach data. Every one of them requires
#: human confirmation — which is every action here, deliberately: there is no
#: access change small enough to apply unattended.
MATERIAL_ACTIONS: Tuple[str, ...] = ACTIONS

#: Who produced a proposal. An agent-sourced proposal can never be confirmed by
#: the agent itself; see ``service.AccessAdminService.confirm``.
SOURCE_OPERATOR = "operator"
SOURCE_OCC_AGENT = "occ_agent"

#: Proposal lifecycle.
PROPOSAL_DRAFT = "draft"
PROPOSAL_CONFIRMED = "confirmed"
PROPOSAL_REJECTED = "rejected"


class ChangeSetError(ValueError):
    """A change set that cannot be applied to the document it was built against.

    A plain error rather than a ``TraktError``: this is an administrative
    authoring fault surfaced to an operator in the OCC, not a governed runtime
    refusal on a data request.
    """


@dataclass(frozen=True)
class AccessChange:
    """One proposed action.

    ``payload`` is deliberately a free-form mapping rather than a per-action
    dataclass: the actions differ enough that fourteen dataclasses would be
    noise, and every field is validated where it is consumed — by the applier
    for shape and by the Stage 1-3 registries for meaning.
    """

    action: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        action = str(self.action or "").strip().upper()
        if action not in ACTIONS:
            raise ChangeSetError(
                f"{self.action!r} is not an access action. Known actions: "
                f"{', '.join(ACTIONS)}")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "payload", dict(self.payload or {}))

    def require(self, *keys: str) -> Tuple[Any, ...]:
        missing = [k for k in keys if not str(self.payload.get(k) or "").strip()]
        if missing:
            raise ChangeSetError(
                f"{self.action} needs {', '.join(missing)}.")
        return tuple(self.payload.get(k) for k in keys)

    def describe(self) -> str:
        """One line an operator can read without knowing the schema."""
        p = self.payload
        if self.action == DISABLE_ORGANISATION:
            return (f"{self.action} {p.get('organisation_id')} "
                    "(also disables all of its grants)")
        if self.action == ENABLE_ORGANISATION:
            return (f"{self.action} {p.get('organisation_id')} "
                    "(its grants stay disabled until re-enabled individually)")
        if self.action == ADD_ORGANISATION:
            return f"{self.action} {p.get('organisation_id')}"
        if self.action == ADD_MICROSOFT_TENANT:
            return (f"{self.action} {p.get('microsoft_tenant_id')} "
                    f"-> {p.get('organisation_id')}")
        if self.action in (ADD_PRINCIPAL, DISABLE_PRINCIPAL, ENABLE_PRINCIPAL):
            who = p.get("display_name") or p.get("microsoft_object_id")
            return f"{self.action} {who} ({p.get('organisation_id') or '—'})"
        if self.action in (ADD_RESOURCE, UPDATE_RESOURCE):
            return (f"{self.action} {p.get('tenant_id')}/{p.get('kind')}/"
                    f"{p.get('resource_id')}")
        caps = p.get("capabilities")
        suffix = f" [{', '.join(sorted(caps))}]" if caps else ""
        return (f"{self.action} {p.get('organisation_id')} "
                f"-> {p.get('resource')}{suffix}")

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "payload": dict(self.payload),
                "reason": self.reason, "description": self.describe()}


@dataclass(frozen=True)
class AccessChangeSet:
    """An ordered, atomic set of access changes awaiting human confirmation."""

    changes: Tuple[AccessChange, ...]
    proposed_by: str
    source: str = SOURCE_OPERATOR
    title: str = ""
    rationale: str = ""
    change_set_id: str = ""
    proposed_at: str = ""

    def __post_init__(self) -> None:
        if not self.changes:
            raise ChangeSetError("A change set must contain at least one change.")
        if not str(self.proposed_by or "").strip():
            raise ChangeSetError("A change set must record who proposed it.")
        object.__setattr__(self, "changes", tuple(self.changes))
        if not self.change_set_id:
            object.__setattr__(self, "change_set_id", new_id("acs"))
        if not self.proposed_at:
            object.__setattr__(self, "proposed_at", now_iso())

    @property
    def requires_confirmation(self) -> bool:
        """Always true. Kept as a property because the reviewer reads it, and
        because a future non-material action should have to justify itself
        against this rather than quietly slipping past."""
        return any(c.action in MATERIAL_ACTIONS for c in self.changes)

    @property
    def fingerprint(self) -> str:
        """Content hash of the proposed actions.

        Confirmation quotes this back, so an operator cannot approve one
        proposal and have a different one applied.
        """
        return stable_hash(canonical_json(
            [{"action": c.action, "payload": dict(c.payload)} for c in self.changes]))

    def describe(self) -> List[str]:
        return [c.describe() for c in self.changes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_set_id": self.change_set_id,
            "title": self.title,
            "rationale": self.rationale,
            "proposed_by": self.proposed_by,
            "source": self.source,
            "proposed_at": self.proposed_at,
            "fingerprint": self.fingerprint,
            "requires_confirmation": self.requires_confirmation,
            "changes": [c.to_dict() for c in self.changes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AccessChangeSet":
        return cls(
            changes=tuple(AccessChange(action=c.get("action"),
                                       payload=c.get("payload") or {},
                                       reason=c.get("reason") or "")
                          for c in (data.get("changes") or [])),
            proposed_by=data.get("proposed_by") or "",
            source=data.get("source") or SOURCE_OPERATOR,
            title=data.get("title") or "",
            rationale=data.get("rationale") or "",
            change_set_id=data.get("change_set_id") or "",
            proposed_at=data.get("proposed_at") or "",
        )


# --------------------------------------------------------------------------- #
# Applying
# --------------------------------------------------------------------------- #
def _norm(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _norm(value).lower()


def _capabilities(change: AccessChange) -> List[str]:
    caps = change.payload.get("capabilities") or []
    if isinstance(caps, str):
        caps = [c.strip() for c in caps.split(",")]
    out = sorted({_lower(c) for c in caps if _norm(c)})
    if not out:
        raise ChangeSetError(
            f"{change.action} needs at least one capability. A grant that "
            "conveys nothing is more likely a mistake than an intent.")
    return out


def apply_change_set(document: AccessDocument,
                     change_set: AccessChangeSet) -> AccessDocument:
    """Apply every change, in order, onto a copy of ``document``.

    All or nothing: the first inapplicable change raises and the caller keeps the
    original document untouched. That is what makes "add the organisation, then
    grant it a resource" safe to propose as one unit — there is no state in which
    the grant landed and the organisation did not.
    """
    draft = document.copy()
    for index, change in enumerate(change_set.changes, start=1):
        try:
            _APPLIERS[change.action](draft, change)
        except ChangeSetError as exc:
            raise ChangeSetError(f"change {index} ({change.action}): {exc}") from exc
    return draft


# -- organisations ---------------------------------------------------------- #
def _add_organisation(doc: AccessDocument, change: AccessChange) -> None:
    (organisation_id,) = change.require("organisation_id")
    organisation_id = _lower(organisation_id)
    if doc.organisation(organisation_id) is not None:
        raise ChangeSetError(
            f"organisation {organisation_id!r} already exists. Use "
            f"{ADD_MICROSOFT_TENANT} or {ENABLE_ORGANISATION} instead.")
    directories = change.payload.get("microsoft_tenant_ids") or []
    single = change.payload.get("microsoft_tenant_id")
    if single:
        directories = list(directories) + [single]
    doc.organisations.append({
        "organisation_id": organisation_id,
        "display_name": _norm(change.payload.get("display_name")) or None,
        "organisation_type": _lower(change.payload.get("organisation_type")) or "unknown",
        "microsoft_tenant_ids": sorted({_lower(d) for d in directories if _norm(d)}),
        "enabled": bool(change.payload.get("enabled", True)),
    })


def _set_organisation_enabled(doc: AccessDocument, change: AccessChange,
                              enabled: bool) -> None:
    """Switch an organisation on or off.

    Disabling **cascades to that organisation's grants**, and enabling does not.
    The asymmetry is deliberate on both sides:

    * Cascading the disable is what makes this usable as a kill switch. Stage 3
      treats a grant naming a disabled organisation as invalid configuration, so
      without the cascade an operator would have to revoke every grant by hand
      before switching a counterparty off — exactly the wrong amount of work for
      the one operation that needs to be fast. The grants are inert either way
      (a disabled organisation cannot resolve a context at all), so this changes
      no access; it keeps the configuration coherent while doing it.

    * Not cascading the enable is the fail-closed direction. Some of those
      grants may have been switched off individually and long before the
      organisation was; silently restoring them would widen access nobody asked
      to restore. Re-enabling an organisation therefore leaves its grants off
      until an operator names the ones they want back.
    """
    (organisation_id,) = change.require("organisation_id")
    row = doc.organisation(organisation_id)
    if row is None:
        raise ChangeSetError(f"no such organisation: {_lower(organisation_id)!r}")
    row["enabled"] = enabled
    if not enabled:
        wanted = _lower(organisation_id)
        for grant in doc.entitlements:
            if _lower(grant.get("organisation_id")) == wanted:
                grant["enabled"] = False


def _add_microsoft_tenant(doc: AccessDocument, change: AccessChange) -> None:
    organisation_id, directory = change.require("organisation_id",
                                                "microsoft_tenant_id")
    row = doc.organisation(organisation_id)
    if row is None:
        raise ChangeSetError(f"no such organisation: {_lower(organisation_id)!r}")
    directories = sorted({*(_lower(d) for d in (row.get("microsoft_tenant_ids") or [])),
                          _lower(directory)})
    row["microsoft_tenant_ids"] = directories


# -- principals ------------------------------------------------------------- #
def _add_principal(doc: AccessDocument, change: AccessChange) -> None:
    organisation_id, directory, object_id = change.require(
        "organisation_id", "microsoft_tenant_id", "microsoft_object_id")
    if doc.organisation(organisation_id) is None:
        raise ChangeSetError(
            f"no such organisation: {_lower(organisation_id)!r}. Add the "
            "organisation before adding its people.")
    if doc.principal(directory, object_id) is not None:
        raise ChangeSetError(
            "that Microsoft principal is already registered. Use "
            f"{ENABLE_PRINCIPAL} to restore access.")
    doc.principals.append({
        "organisation_id": _lower(organisation_id),
        "microsoft_tenant_id": _lower(directory),
        "microsoft_object_id": _lower(object_id),
        "display_name": _norm(change.payload.get("display_name")) or None,
        "email": _norm(change.payload.get("email")) or None,
        "status": "active",
    })


def _set_principal_status(doc: AccessDocument, change: AccessChange,
                          status: str) -> None:
    directory, object_id = change.require("microsoft_tenant_id",
                                          "microsoft_object_id")
    row = doc.principal(directory, object_id)
    if row is None:
        raise ChangeSetError("no such registered Microsoft principal")
    row["status"] = status


# -- resources -------------------------------------------------------------- #
_RESOURCE_FIELDS = ("display_label", "source_portfolio_ids", "spv_id",
                    "population", "unpartitionable", "unpartitionable_reason",
                    "whole_tenant_book", "enabled")


def _resource_row(change: AccessChange) -> Dict[str, Any]:
    tenant_id, kind, resource_id = change.require("tenant_id", "kind",
                                                  "resource_id")
    row: Dict[str, Any] = {"tenant_id": _norm(tenant_id), "kind": _lower(kind),
                           "resource_id": _lower(resource_id)}
    for key in _RESOURCE_FIELDS:
        if key in change.payload:
            row[key] = change.payload[key]
    return row


def _add_resource(doc: AccessDocument, change: AccessChange) -> None:
    row = _resource_row(change)
    reference = f"{row['tenant_id']}/{row['kind']}/{row['resource_id']}"
    if doc.resource(reference) is not None:
        raise ChangeSetError(
            f"resource {reference} already exists. Use {UPDATE_RESOURCE}.")
    doc.resources.append(row)


def _update_resource(doc: AccessDocument, change: AccessChange) -> None:
    incoming = _resource_row(change)
    reference = (f"{incoming['tenant_id']}/{incoming['kind']}/"
                 f"{incoming['resource_id']}")
    existing = doc.resource(reference)
    if existing is None:
        raise ChangeSetError(f"no such resource: {reference}")
    for key in _RESOURCE_FIELDS:
        if key in change.payload:
            existing[key] = change.payload[key]
    # Supplying real attribution is how a resource stops being unpartitionable,
    # so clearing the flag has to be possible — but only by naming it, never as
    # a side effect of touching some other field.
    if change.payload.get("unpartitionable") is False:
        existing["unpartitionable"] = False
        existing.pop("unpartitionable_reason", None)


# -- grants ----------------------------------------------------------------- #
def _add_grant(doc: AccessDocument, change: AccessChange) -> None:
    organisation_id, resource = change.require("organisation_id", "resource")
    if doc.organisation(organisation_id) is None:
        raise ChangeSetError(f"no such organisation: {_lower(organisation_id)!r}")
    if doc.resource(resource) is None:
        raise ChangeSetError(
            f"no such resource: {_norm(resource)!r}. Resources are resolved from "
            "the catalogue, never from the wording of a request.")
    if doc.grant(organisation_id, resource) is not None:
        raise ChangeSetError(
            f"{_lower(organisation_id)} already has a grant on "
            f"{_resource_key(resource)}. Use {SET_GRANT_CAPABILITIES} to change "
            "what it permits.")
    doc.entitlements.append({
        "organisation_id": _lower(organisation_id),
        "resource": _norm(resource),
        "capabilities": _capabilities(change),
        "status": _lower(change.payload.get("status")) or "active",
        "enabled": bool(change.payload.get("enabled", True)),
        "source": _norm(change.payload.get("source")) or None,
        "note": _norm(change.payload.get("note")) or None,
    })


def _existing_grant(doc: AccessDocument, change: AccessChange) -> Dict[str, Any]:
    organisation_id, resource = change.require("organisation_id", "resource")
    row = doc.grant(organisation_id, resource)
    if row is None:
        raise ChangeSetError(
            f"{_lower(organisation_id)} has no grant on {_resource_key(resource)}")
    return row


def _revoke_grant(doc: AccessDocument, change: AccessChange) -> None:
    row = _existing_grant(doc, change)
    doc.entitlements.remove(row)


def _set_grant_capabilities(doc: AccessDocument, change: AccessChange) -> None:
    row = _existing_grant(doc, change)
    row["capabilities"] = _capabilities(change)


def _set_grant_enabled(doc: AccessDocument, change: AccessChange,
                       enabled: bool) -> None:
    _existing_grant(doc, change)["enabled"] = enabled


_APPLIERS = {
    ADD_ORGANISATION: _add_organisation,
    DISABLE_ORGANISATION: lambda d, c: _set_organisation_enabled(d, c, False),
    ENABLE_ORGANISATION: lambda d, c: _set_organisation_enabled(d, c, True),
    ADD_MICROSOFT_TENANT: _add_microsoft_tenant,
    ADD_PRINCIPAL: _add_principal,
    DISABLE_PRINCIPAL: lambda d, c: _set_principal_status(d, c, "disabled"),
    ENABLE_PRINCIPAL: lambda d, c: _set_principal_status(d, c, "active"),
    ADD_RESOURCE: _add_resource,
    UPDATE_RESOURCE: _update_resource,
    ADD_GRANT: _add_grant,
    REVOKE_GRANT: _revoke_grant,
    SET_GRANT_CAPABILITIES: _set_grant_capabilities,
    DISABLE_GRANT: lambda d, c: _set_grant_enabled(d, c, False),
    ENABLE_GRANT: lambda d, c: _set_grant_enabled(d, c, True),
}
