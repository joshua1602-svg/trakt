"""trakt_core.errors — the machine-readable error taxonomy.

One error type, a stable code vocabulary, and a single table mapping each code
to its category, retryability and HTTP status. Adapters map from that table
rather than inventing their own status codes, so React, Copilot and any future
caller classify the same failure the same way.

Design notes:

  * ``code`` is the contract. Messages are for humans and may be reworded;
    callers must branch on ``code``.
  * ``retryable`` tells an autonomous caller whether waiting could help. It is a
    property of the code, not of the call site.
  * ``details`` is for safe, structured context (a required scope, a portfolio
    id). Never put stack traces, credentials, storage paths or another tenant's
    identifiers in it — :meth:`TraktError.to_dict` is serialised to clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class ErrorCategory:
    """Coarse grouping, useful for metrics and for agent-side triage."""

    AUTHENTICATION = "authentication"
    AUTHORISATION = "authorisation"
    INPUT = "input"
    POLICY = "policy"
    DATA = "data"
    CAPABILITY = "capability"
    INFRASTRUCTURE = "infrastructure"


class ErrorCode:
    """Stable error codes. Values are part of the external contract."""

    # -- authentication ---------------------------------------------------- #
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    AUTHENTICATION_INVALID = "AUTHENTICATION_INVALID"
    AUTHENTICATION_NOT_CONFIGURED = "AUTHENTICATION_NOT_CONFIGURED"

    # -- authorisation ----------------------------------------------------- #
    PERMISSION_DENIED = "PERMISSION_DENIED"
    SCOPE_MISSING = "SCOPE_MISSING"
    TENANT_MISMATCH = "TENANT_MISMATCH"
    PORTFOLIO_NOT_AUTHORISED = "PORTFOLIO_NOT_AUTHORISED"
    #: The caller's Microsoft directory is not registered as an organisation on a
    #: deployment that serves registered organisations only.
    ORGANISATION_NOT_REGISTERED = "ORGANISATION_NOT_REGISTERED"
    #: The organisation is registered but its access has been withdrawn.
    ORGANISATION_DISABLED = "ORGANISATION_DISABLED"
    #: The named resource is not in the resource catalogue.
    RESOURCE_NOT_REGISTERED = "RESOURCE_NOT_REGISTERED"
    #: The resource is registered but currently switched off.
    RESOURCE_DISABLED = "RESOURCE_DISABLED"
    #: The resource is registered but the data carries no attribution that can
    #: separate it from the rest of the book. Refused rather than widened.
    RESOURCE_NOT_PARTITIONABLE = "RESOURCE_NOT_PARTITIONABLE"
    #: This organisation holds no grant over the named resource. Deliberately
    #: also returned for a resource that does not exist, so a caller cannot
    #: enumerate another organisation's resources by comparing error codes.
    RESOURCE_NOT_AUTHORISED = "RESOURCE_NOT_AUTHORISED"
    #: The organisation may reach the resource, but not with this capability.
    CAPABILITY_NOT_GRANTED = "CAPABILITY_NOT_GRANTED"
    #: The request carries no resolved entitlements, so no resource can be
    #: authorised for it. Absence of configuration is not absence of restriction.
    ENTITLEMENTS_NOT_RESOLVED = "ENTITLEMENTS_NOT_RESOLVED"

    # -- input ------------------------------------------------------------- #
    INVALID_INPUT = "INVALID_INPUT"

    # -- capability / question --------------------------------------------- #
    UNSUPPORTED_QUESTION = "UNSUPPORTED_QUESTION"
    AMBIGUOUS_QUESTION = "AMBIGUOUS_QUESTION"
    NO_MATCHING_RECORDS = "NO_MATCHING_RECORDS"
    CALCULATION_FAILED = "CALCULATION_FAILED"

    # -- data / policy ------------------------------------------------------ #
    DATA_SOURCE_NOT_APPROVED = "DATA_SOURCE_NOT_APPROVED"
    DATA_SOURCE_UNAVAILABLE = "DATA_SOURCE_UNAVAILABLE"
    DATASET_MANIFEST_MISSING = "DATASET_MANIFEST_MISSING"
    SNAPSHOT_STALE = "SNAPSHOT_STALE"

    # -- artefacts ---------------------------------------------------------- #
    ARTEFACT_NOT_FOUND = "ARTEFACT_NOT_FOUND"
    ARTEFACT_NOT_APPROVED = "ARTEFACT_NOT_APPROVED"
    ARTEFACT_INCOMPLETE = "ARTEFACT_INCOMPLETE"

    # -- infrastructure ------------------------------------------------------ #
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class _CodeSpec:
    category: str
    retryable: bool
    http_status: int


#: code -> (category, retryable, http status). The single mapping table; both
#: the FastAPI and Copilot adapters read it, so classification cannot diverge.
_CODES: Dict[str, _CodeSpec] = {
    ErrorCode.AUTHENTICATION_REQUIRED: _CodeSpec(ErrorCategory.AUTHENTICATION, False, 401),
    ErrorCode.AUTHENTICATION_INVALID: _CodeSpec(ErrorCategory.AUTHENTICATION, False, 401),
    # Not the caller's fault and not fixable by retrying the same way, but the
    # deployment may be repaired — 503 so a client agent backs off rather than
    # treating it as a permanent denial.
    ErrorCode.AUTHENTICATION_NOT_CONFIGURED: _CodeSpec(ErrorCategory.AUTHENTICATION, True, 503),

    ErrorCode.PERMISSION_DENIED: _CodeSpec(ErrorCategory.AUTHORISATION, False, 403),
    ErrorCode.SCOPE_MISSING: _CodeSpec(ErrorCategory.AUTHORISATION, False, 403),
    ErrorCode.TENANT_MISMATCH: _CodeSpec(ErrorCategory.AUTHORISATION, False, 403),
    ErrorCode.PORTFOLIO_NOT_AUTHORISED: _CodeSpec(ErrorCategory.AUTHORISATION, False, 403),
    # Not retryable by the caller: an operator has to register or re-enable the
    # organisation. 403 rather than 401 — the token was valid, the organisation
    # behind it is not accepted.
    ErrorCode.ORGANISATION_NOT_REGISTERED: _CodeSpec(ErrorCategory.AUTHORISATION, False, 403),
    ErrorCode.ORGANISATION_DISABLED: _CodeSpec(ErrorCategory.AUTHORISATION, False, 403),
    ErrorCode.RESOURCE_NOT_REGISTERED: _CodeSpec(ErrorCategory.AUTHORISATION, False, 403),
    ErrorCode.RESOURCE_DISABLED: _CodeSpec(ErrorCategory.AUTHORISATION, False, 403),
    # Not retryable and not the caller's fault: the boundary cannot be enforced
    # from the data, so serving anything would mean serving too much.
    ErrorCode.RESOURCE_NOT_PARTITIONABLE: _CodeSpec(ErrorCategory.AUTHORISATION, False, 403),
    ErrorCode.RESOURCE_NOT_AUTHORISED: _CodeSpec(ErrorCategory.AUTHORISATION, False, 403),
    ErrorCode.CAPABILITY_NOT_GRANTED: _CodeSpec(ErrorCategory.AUTHORISATION, False, 403),
    ErrorCode.ENTITLEMENTS_NOT_RESOLVED: _CodeSpec(ErrorCategory.AUTHORISATION, False, 403),

    ErrorCode.INVALID_INPUT: _CodeSpec(ErrorCategory.INPUT, False, 400),

    ErrorCode.UNSUPPORTED_QUESTION: _CodeSpec(ErrorCategory.CAPABILITY, False, 200),
    ErrorCode.AMBIGUOUS_QUESTION: _CodeSpec(ErrorCategory.CAPABILITY, False, 200),
    ErrorCode.NO_MATCHING_RECORDS: _CodeSpec(ErrorCategory.CAPABILITY, False, 200),
    ErrorCode.CALCULATION_FAILED: _CodeSpec(ErrorCategory.CAPABILITY, False, 200),

    ErrorCode.DATA_SOURCE_NOT_APPROVED: _CodeSpec(ErrorCategory.POLICY, False, 503),
    ErrorCode.DATA_SOURCE_UNAVAILABLE: _CodeSpec(ErrorCategory.DATA, True, 503),
    ErrorCode.DATASET_MANIFEST_MISSING: _CodeSpec(ErrorCategory.DATA, False, 503),
    ErrorCode.SNAPSHOT_STALE: _CodeSpec(ErrorCategory.DATA, True, 200),

    ErrorCode.ARTEFACT_NOT_FOUND: _CodeSpec(ErrorCategory.DATA, False, 404),
    ErrorCode.ARTEFACT_NOT_APPROVED: _CodeSpec(ErrorCategory.POLICY, False, 403),
    ErrorCode.ARTEFACT_INCOMPLETE: _CodeSpec(ErrorCategory.DATA, True, 409),

    ErrorCode.STORAGE_UNAVAILABLE: _CodeSpec(ErrorCategory.INFRASTRUCTURE, True, 503),
    ErrorCode.INTERNAL_ERROR: _CodeSpec(ErrorCategory.INFRASTRUCTURE, True, 500),
}

_UNKNOWN = _CodeSpec(ErrorCategory.INFRASTRUCTURE, False, 500)


def _spec(code: str) -> _CodeSpec:
    return _CODES.get(code, _UNKNOWN)


def category_for(code: str) -> str:
    return _spec(code).category


def is_retryable(code: str) -> bool:
    return _spec(code).retryable


def http_status_for(code: str) -> int:
    """The HTTP status an adapter should use for ``code``.

    ``200`` for the capability-level outcomes (unsupported / ambiguous / no
    records / calculation failure): the request was well-formed and correctly
    authorised, and the governed answer *is* "we did not compute one". Callers
    read ``status`` and ``error.code`` from the envelope. This preserves the
    existing React and Copilot behaviour, both of which return 200 with
    ``ok=false`` for those cases today.
    """
    return _spec(code).http_status


def known_codes() -> Dict[str, _CodeSpec]:
    return dict(_CODES)


class TraktError(Exception):
    """A governed failure with a stable, machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id
        self.details: Dict[str, Any] = dict(details or {})
        self.__cause__ = cause

    # -- derived ------------------------------------------------------------ #
    @property
    def category(self) -> str:
        return category_for(self.code)

    @property
    def retryable(self) -> bool:
        return is_retryable(self.code)

    @property
    def http_status(self) -> int:
        return http_status_for(self.code)

    def to_dict(self) -> Dict[str, Any]:
        """The client-safe serialisation. Contains no stack trace, no storage
        path, no credential and no cross-tenant identifier."""
        payload: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "category": self.category,
            "retryable": self.retryable,
        }
        if self.request_id:
            payload["request_id"] = self.request_id
        if self.details:
            payload["details"] = dict(self.details)
        return payload

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TraktError(code={self.code!r}, message={self.message!r})"
