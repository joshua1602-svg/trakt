"""Deterministic validation of a candidate access configuration.

The rule this module follows: **it re-implements nothing.** Every check is
performed by constructing the real Stage 1-3 registries from the candidate
document and letting them refuse it. So an access change is accepted here if and
only if the runtime would have accepted the resulting configuration — there is
no second, drifting copy of "what a valid grant looks like".

That gives, for free and without restating any of it:

  * duplicate directories across organisations, duplicate organisation ids
    (``OrganisationRegistry``);
  * duplicate ``(directory, object id)`` principals (``PrincipalRegistry``);
  * duplicate resources, and resources declaring no constraint at all
    (``ResourceCatalogue``);
  * grants naming an unregistered or disabled organisation, an unknown, disabled
    or **unpartitionable** resource, an unrecognised capability, or the same
    (organisation, resource) twice (``EntitlementStore``).

Two additional checks belong here rather than at runtime, because they are about
the *administration* of access rather than a single request:

  * a principal bound to an organisation the document does not contain;
  * a principal bound to an organisation whose directory list does not include
    the directory the principal signs in from — which would produce a person who
    can never authenticate, or worse, one whose organisation is decided by two
    disagreeing records.

Nothing here writes, activates, or decides who approves. It returns problems.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from trakt_core.entitlement import EntitlementStore, Grant
from trakt_core.errors import TraktError
from trakt_core.organisation import (
    OrganisationRegistry,
    _record_from_spec as _organisation_from_spec,
    normalise_directory_id,
)
from trakt_core.principal import PrincipalRegistry, _binding_from_spec
from trakt_core.resource import (
    ResourceCatalogue,
    ResourceRef,
    _record_from_spec as _resource_from_spec,
)

from .document import ACCESS_FILES, AccessDocument


def _problem(area: str, message: str, **detail: Any) -> Dict[str, Any]:
    return {"area": area, "message": message, "detail": detail}


def validate_document(document: AccessDocument) -> List[Dict[str, Any]]:
    """Every problem with ``document``, or an empty list.

    Each registry is built independently so one broken area does not mask
    another: an operator fixing a bad grant should not have to fix it three
    times to discover a separate bad principal.
    """
    problems: List[Dict[str, Any]] = []

    organisations = _build_organisations(document, problems)
    resources = _build_resources(document, problems)
    principals = _build_principals(document, problems)
    _check_principal_consistency(document, organisations, problems)
    _build_entitlements(document, organisations, resources, problems)

    return problems


def _build_organisations(document: AccessDocument,
                         problems: List[Dict[str, Any]]
                         ) -> Optional[OrganisationRegistry]:
    try:
        records = [_organisation_from_spec(dict(row))
                   for row in document.organisations]
    except Exception as exc:  # noqa: BLE001 - a malformed row is a problem, not a crash
        problems.append(_problem("organisations", str(exc)))
        return None
    try:
        return OrganisationRegistry(records, configured=True)
    except TraktError as exc:
        problems.append(_problem("organisations", exc.message, **exc.details))
        return None


def _build_resources(document: AccessDocument,
                     problems: List[Dict[str, Any]]) -> Optional[ResourceCatalogue]:
    try:
        records = [_resource_from_spec(dict(row)) for row in document.resources]
    except Exception as exc:  # noqa: BLE001
        problems.append(_problem("resources", str(exc)))
        return None
    try:
        return ResourceCatalogue(records, configured=True)
    except TraktError as exc:
        problems.append(_problem("resources", exc.message, **exc.details))
        return None


def _build_principals(document: AccessDocument,
                      problems: List[Dict[str, Any]]) -> Optional[PrincipalRegistry]:
    try:
        bindings = [_binding_from_spec(dict(row)) for row in document.principals]
    except Exception as exc:  # noqa: BLE001
        problems.append(_problem("principals", str(exc)))
        return None
    try:
        return PrincipalRegistry(bindings, configured=True)
    except TraktError as exc:
        problems.append(_problem("principals", exc.message, **exc.details))
        return None


def _check_principal_consistency(document: AccessDocument,
                                 organisations: Optional[OrganisationRegistry],
                                 problems: List[Dict[str, Any]]) -> None:
    """A principal must belong to an organisation that exists and owns their
    directory.

    Not a runtime check because at runtime the two are resolved independently and
    a mismatch is refused there too — but catching it at proposal time is the
    difference between an administrator seeing "Jane's directory is not on
    Funder A" now and a user seeing an inexplicable refusal in three weeks.
    """
    if organisations is None:
        return
    for row in document.principals:
        organisation_id = str(row.get("organisation_id") or "").strip().lower()
        directory = normalise_directory_id(row.get("microsoft_tenant_id"))
        who = (row.get("display_name") or row.get("microsoft_object_id") or "?")
        record = organisations.get(organisation_id)
        if record is None:
            problems.append(_problem(
                "principals",
                f"{who} is bound to organisation '{organisation_id}', which is "
                "not registered.",
                organisation_id=organisation_id))
            continue
        if directory and not record.owns_directory(directory):
            problems.append(_problem(
                "principals",
                f"{who} signs in from a Microsoft directory that is not "
                f"registered to '{organisation_id}'. Add the directory to the "
                "organisation, or bind the person to the organisation that owns "
                "it.",
                organisation_id=organisation_id,
                microsoft_tenant_id=directory))


def _build_entitlements(document: AccessDocument,
                        organisations: Optional[OrganisationRegistry],
                        resources: Optional[ResourceCatalogue],
                        problems: List[Dict[str, Any]]) -> None:
    grants: List[Grant] = []
    for row in document.entitlements:
        try:
            grants.append(Grant(
                organisation_id=row.get("organisation_id"),
                resource_ref=ResourceRef.parse(str(row.get("resource") or "")),
                capabilities=frozenset(row.get("capabilities") or ()),
                status=row.get("status") or "active",
                enabled=bool(row.get("enabled", True)),
                source=row.get("source"),
                approved_by=row.get("approved_by"),
                approved_at=row.get("approved_at"),
                note=row.get("note"),
            ))
        except Exception as exc:  # noqa: BLE001
            problems.append(_problem(
                "entitlements", str(exc),
                organisation_id=row.get("organisation_id"),
                resource=row.get("resource")))
    if organisations is None or resources is None:
        # The upstream registry already failed; validating grants against a
        # half-built world would produce misleading follow-on problems.
        return
    try:
        EntitlementStore(grants, configured=True, organisations=organisations,
                         resources=resources)
    except TraktError as exc:
        problems.append(_problem("entitlements", exc.message, **exc.details))


def validate_access_files(files: Mapping[str, Any]) -> List[str]:
    """Validate a ``ConfigPackageStore`` version's files. Returns flat strings.

    This is the hook :func:`operations_control.configuration.packages.validate_version`
    calls for the access layer, so the store's existing "a version must be
    validated before it can be activated" gate covers access configuration with
    no second activation path.
    """
    unexpected = sorted(set(files) - set(ACCESS_FILES))
    problems = [f"{rel}: not part of the access configuration" for rel in unexpected]
    try:
        document = AccessDocument.from_files(files)
    except Exception as exc:  # noqa: BLE001 - unparseable YAML is a problem, not a crash
        return problems + [f"access configuration could not be read: {exc}"]
    return problems + [f"{p['area']}: {p['message']}" for p in validate_document(document)]
