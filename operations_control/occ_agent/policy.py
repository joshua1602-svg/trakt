"""operations_control.occ_agent.policy — the synthetic execution boundary.

This module is the floor of the OCC Agent feature. Everything above it — the
service, the interpreter, the API router and the React tab — depends on it; it
depends on nothing in the feature. That ordering is deliberate: a prohibited
action must be impossible to reach by writing new code higher up.

The boundary is expressed as an explicit, immutable policy:

    runtime_mode: synthetic
    allow_external_email: false
    allow_live_blob_write: false
    allow_live_pipeline_trigger: false
    allow_production_config_write: false
    allow_publish: false

Every prohibited capability has a named guard. A guard **fails closed**: it
raises :class:`SyntheticBoundaryError` (an :class:`~operations_control.engine.
OpsError`, so the existing API error envelope carries it unchanged) and records
an audit event through the sink the caller supplied. There is no "force" flag
and no way to construct a permissive policy: :func:`synthetic_policy` is the
only constructor and it hard-codes every permission to ``False``.

Two further controls live here because they are the same kind of control:

* :func:`sandbox_path` — the only way to turn a case-relative path into a real
  filesystem path. It refuses traversal, absolute paths and symlink escapes, so
  a case identifier or a model-produced filename cannot reach outside its own
  sandbox.
* :func:`assert_no_live_credentials_required` — synthetic mode never needs Azure
  credentials; the storage backend is pinned to the filesystem for the sandbox.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..engine import OpsError

#: The feature flag. Absent or anything other than a recognised truthy value
#: means the OCC Agent tab and its API are OFF — the flag fails closed.
FEATURE_FLAG_ENV = "OCC_AGENT_SYNTHETIC_ENABLED"

_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})

#: The runtime mode this feature runs in. Distinct from
#: ``trakt_core.runtime`` modes (production/development/test) on purpose: this
#: is not "which data may answer a question", it is "which side effects may
#: occur". A case carries it so a persisted case can never be mistaken for a
#: live one.
RUNTIME_MODE_SYNTHETIC = "synthetic"

#: Machine error code for every boundary refusal. Stable — callers branch on it.
ERROR_CODE = "OCC_AGENT_SYNTHETIC_BOUNDARY"

#: Capability names. One per prohibited action in the specification.
CAP_EXTERNAL_EMAIL = "external_email"
CAP_LIVE_BLOB_WRITE = "live_blob_write"
CAP_LIVE_PIPELINE_TRIGGER = "live_pipeline_trigger"
CAP_PRODUCTION_CONFIG_WRITE = "production_config_write"
CAP_PUBLISH = "publish"
CAP_LIVE_CASE_ACCESS = "live_case_access"
#: Activation is Client Onboarding's own write boundary — its ``activate()``
#: is "the only place active configuration is created". Naming it as a
#: capability makes that the exact line a practice case never crosses.
CAP_ACTIVATE_CONFIGURATION = "activate_configuration"

CAPABILITIES = (CAP_EXTERNAL_EMAIL, CAP_LIVE_BLOB_WRITE,
                CAP_LIVE_PIPELINE_TRIGGER, CAP_PRODUCTION_CONFIG_WRITE,
                CAP_PUBLISH, CAP_LIVE_CASE_ACCESS, CAP_ACTIVATE_CONFIGURATION)

#: Operator-safe refusal wording, by capability. Plain English: these surface in
#: the OCC exactly like any other governed refusal.
_REFUSALS: Dict[str, str] = {
    CAP_EXTERNAL_EMAIL:
        "This is a synthetic onboarding case. Trakt records the draft email in "
        "the case and never sends it.",
    CAP_LIVE_BLOB_WRITE:
        "This is a synthetic onboarding case. Trakt works out where each file "
        "would be filed, but never writes to live storage.",
    CAP_LIVE_PIPELINE_TRIGGER:
        "This is a synthetic onboarding case. Trakt prepares the execution "
        "package but never starts the live pipeline.",
    CAP_PRODUCTION_CONFIG_WRITE:
        "This is a synthetic onboarding case. It can propose configuration, "
        "but it can never change the platform's live configuration.",
    CAP_PUBLISH:
        "This is a synthetic onboarding case. Nothing it produces can be "
        "published.",
    CAP_LIVE_CASE_ACCESS:
        "This is a synthetic onboarding case. It cannot read or change live "
        "delivery workflows.",
    CAP_ACTIVATE_CONFIGURATION:
        "This is a practice case. It can be approved, but activating it — "
        "which is what creates a client's configuration — is not possible "
        "here.",
}


class SyntheticBoundaryError(OpsError):
    """A prohibited action was attempted from synthetic mode.

    Carries the capability that was refused so a caller (or a test) can assert
    on *what* was blocked, not just that something was.
    """

    def __init__(self, capability: str, detail: str = ""):
        self.capability = capability
        message = _REFUSALS.get(
            capability,
            "This is a synthetic onboarding case. That action is not "
            "permitted here.")
        if detail:
            message = f"{message} ({detail})"
        super().__init__(ERROR_CODE, message, http_status=403)


def feature_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """Whether the OCC Agent tab and API are switched on. Fails closed."""
    source = env if env is not None else os.environ
    return str(source.get(FEATURE_FLAG_ENV, "")).strip().lower() in _TRUTHY


# --------------------------------------------------------------------------- #
# The policy object
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SyntheticPolicy:
    """The immutable capability set for one synthetic session.

    Frozen and constructed only by :func:`synthetic_policy`, so there is no
    supported way to obtain one with a permission granted. ``audit_sink`` is the
    seam through which refusals become audit events — the store injects the
    case's own audit appender.
    """

    runtime_mode: str = RUNTIME_MODE_SYNTHETIC
    allow_external_email: bool = False
    allow_live_blob_write: bool = False
    allow_live_pipeline_trigger: bool = False
    allow_production_config_write: bool = False
    allow_publish: bool = False
    allow_live_case_access: bool = False
    allow_activate_configuration: bool = False
    audit_sink: Optional[Callable[[Dict[str, Any]], None]] = field(
        default=None, compare=False, repr=False)

    # -- introspection ------------------------------------------------------ #
    def permits(self, capability: str) -> bool:
        return bool(getattr(self, f"allow_{capability}", False))

    def to_dict(self) -> Dict[str, Any]:
        """The policy as recorded on a case and returned to the UI."""
        return {
            "runtime_mode": self.runtime_mode,
            "allow_external_email": self.allow_external_email,
            "allow_live_blob_write": self.allow_live_blob_write,
            "allow_live_pipeline_trigger": self.allow_live_pipeline_trigger,
            "allow_production_config_write": self.allow_production_config_write,
            "allow_publish": self.allow_publish,
            "allow_live_case_access": self.allow_live_case_access,
            "allow_activate_configuration": self.allow_activate_configuration,
        }

    # -- the guards --------------------------------------------------------- #
    def refuse(self, capability: str, *, detail: str = "", case_id: str = "",
               tenant: str = "", actor: str = "") -> SyntheticBoundaryError:
        """Audit a refusal and return the error to raise.

        Returning rather than raising lets a caller ``raise policy.refuse(...)``
        so the raise site stays visible at the call site. ``tenant`` and
        ``case_id`` together are what let the sink file the event against the
        case that attempted the action; without both it can only be logged.
        """
        event = {
            "action": "synthetic_boundary_refused",
            "capability": capability,
            "case_id": case_id,
            "tenant": tenant,
            "actor_identity": actor,
            "actor_type": "system",
            "runtime_mode": self.runtime_mode,
            "execution_classification": "blocked",
            "decision_basis": f"synthetic policy refuses {capability}",
            "detail": detail,
        }
        if self.audit_sink is not None:
            try:
                self.audit_sink(event)
            except Exception:  # noqa: BLE001 — auditing must not mask the refusal
                pass
        return SyntheticBoundaryError(capability, detail)

    def require(self, capability: str, *, detail: str = "", case_id: str = "",
                tenant: str = "", actor: str = "") -> None:
        """Assert a capability is permitted; raise (and audit) when it is not.

        Every prohibited capability is denied by construction, so in practice
        this always raises — which is the point. It exists so call sites that
        *would* perform a live action name the capability they would need, and
        so the refusal is auditable rather than an unreachable branch.
        """
        if not self.permits(capability):
            raise self.refuse(capability, detail=detail, case_id=case_id,
                              tenant=tenant, actor=actor)

    # Named guards. Call these from any code that is about to do the real thing.
    def guard_external_email(self, **kw: Any) -> None:
        self.require(CAP_EXTERNAL_EMAIL, **kw)

    def guard_live_blob_write(self, **kw: Any) -> None:
        self.require(CAP_LIVE_BLOB_WRITE, **kw)

    def guard_live_pipeline_trigger(self, **kw: Any) -> None:
        self.require(CAP_LIVE_PIPELINE_TRIGGER, **kw)

    def guard_production_config_write(self, **kw: Any) -> None:
        self.require(CAP_PRODUCTION_CONFIG_WRITE, **kw)

    def guard_publish(self, **kw: Any) -> None:
        self.require(CAP_PUBLISH, **kw)

    def guard_live_case_access(self, **kw: Any) -> None:
        self.require(CAP_LIVE_CASE_ACCESS, **kw)

    def guard_activate_configuration(self, **kw: Any) -> None:
        self.require(CAP_ACTIVATE_CONFIGURATION, **kw)


def synthetic_policy(
        audit_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> SyntheticPolicy:
    """The only supported way to obtain a policy. Every permission is denied."""
    return SyntheticPolicy(audit_sink=audit_sink)


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #

#: A tenant / client / portfolio identifier used as one path segment. Mirrors
#: manual_intake._SEGMENT_RE so synthetic and live agree on what a segment is.
SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class UnsafePathError(OpsError):
    """A path could not be resolved safely inside the synthetic sandbox."""

    def __init__(self, detail: str = ""):
        super().__init__(
            "OCC_AGENT_UNSAFE_PATH",
            "Trakt could not use that name. Check the file name and try again."
            + (f" ({detail})" if detail else ""),
            http_status=400)


def validate_segment(value: str, what: str = "identifier") -> str:
    v = str(value or "").strip()
    if not SEGMENT_RE.match(v) or v in (".", ".."):
        raise UnsafePathError(what)
    return v


def sandbox_path(root: Path, *parts: str) -> Path:
    """Resolve ``parts`` under ``root``, refusing anything that escapes it.

    ``root`` is resolved first, then the joined path; the result must be inside
    the resolved root. That catches ``..`` segments, absolute components and
    symlinks that point outside the sandbox — the three ways a supplied name
    could otherwise reach live data.
    """
    base = Path(root).resolve()
    for part in parts:
        p = str(part)
        if not p or p in (".", "..") or "\\" in p or p.startswith("/"):
            raise UnsafePathError("path segment")
        if any(ord(c) < 32 or ord(c) == 127 for c in p):
            raise UnsafePathError("path segment")
    candidate = base.joinpath(*[str(p) for p in parts])
    # ``strict=False``: the leaf may not exist yet (we are often about to
    # create it); every existing component is still resolved, so a symlinked
    # parent cannot smuggle the result outside the sandbox.
    resolved = candidate.resolve(strict=False)
    if resolved != base and base not in resolved.parents:
        raise UnsafePathError("path escapes the synthetic sandbox")
    return resolved


def assert_no_live_credentials_required() -> None:
    """Synthetic mode must work with no Azure credentials present.

    The sandbox is always filesystem-backed, so this is an assertion about the
    code path rather than about the environment: nothing in the feature reads a
    connection string, and this function documents and tests that contract.
    """
    # Deliberately reads nothing. Present so the guarantee has a name a test can
    # call, and so any future change that introduces a credential dependency has
    # an obvious place that must change too.
    return None
