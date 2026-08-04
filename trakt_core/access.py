"""trakt_core.access — the governed identity→access directory.

The platform (Azure Static Web Apps / App Service Easy Auth, or a validated
Entra bearer token) answers *who signed in*. It does not answer *what that
person may see here*. This module owns the second question, and Trakt owns the
answer.

Why this exists
---------------
Authorisation previously came from Static Web Apps' invitation roles: a user
was invited, accepted, and the accepted invitation attached ``client`` or
``operator`` to the identity SWA forwards. That mechanism proved unreliable in
this deployment — invitation acceptance did not create a role record and
``az staticwebapp users list`` stayed empty — so every signed-in user arrived
with no MI role and was refused.

More importantly it was the wrong place for the decision. Access to a tenant's
book is a Trakt governance concern, versioned and reviewable alongside the rest
of the configuration, not a per-user click in someone else's portal.

The split this module enforces
------------------------------
  * **Authentication** — the platform's job. Still required, still trusted only
    when :func:`mi_agent_api.identity.platform_auth_status` says the header
    cannot be forged. This module never authenticates anyone.
  * **Authorisation** — this directory's job. A verified identity is looked up
    here, and the result decides tenant, role, portfolio contexts and whether
    the account is live at all.

This is not a second identity system. There is still exactly one identity — the
one the platform verified — and exactly one trusted context type
(:class:`~trakt_core.context.ExecutionContext`). This module only supplies the
facts that context is built from, which previously came from a deployment-wide
environment variable and an unreliable role claim.

What a caller can and cannot influence
--------------------------------------
Nothing here is caller-supplied. The lookup key is the verified subject from
the platform; ``tenant_id``, ``platform_role`` and ``portfolio_contexts`` all
come from configuration. A caller presenting a principal that claims
``operator`` for another tenant gets whatever this directory says about its
*verified subject*, which is normally "not provisioned".

Fail closed
-----------
An identity that is not in the directory is refused, and so is one that is
present but disabled. There is no implicit grant: adding a user is a deliberate
configuration change. The one exception is the explicit local-development
bypass (``MI_AGENT_AUTH_ENABLED=false``), which already short-circuits
authentication itself and never applies in Azure.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from .errors import ErrorCode, TraktError

logger = logging.getLogger("trakt_core.access")

#: Platform roles. Deliberately two: "the customer" and "us". Finer-grained
#: permission lives in scopes, not in a proliferation of roles.
ROLE_CLIENT = "client"
ROLE_OPERATOR = "operator"

KNOWN_ROLES: FrozenSet[str] = frozenset({ROLE_CLIENT, ROLE_OPERATOR})

DEFAULT_ACCESS_CONFIG = "config/access.yaml"
ACCESS_CONFIG_ENV = "TRAKT_ACCESS_CONFIG"

#: The message an unprovisioned but authenticated caller receives. Fixed text:
#: the React error surface and the deployment runbook both quote it, and it is
#: what distinguishes "we do not know you" from "something broke".
NOT_PROVISIONED_MESSAGE = (
    "Your account is signed in but has not been provisioned for Trakt access."
)
DISABLED_MESSAGE = (
    "Your Trakt access has been disabled. Contact your Trakt administrator."
)


def _norm(value: Any) -> str:
    """Directory keys compare case-insensitively.

    Email local-parts are case-insensitive in every practical directory, Entra
    returns object ids in either case, and a mapping that silently failed on
    ``Admin@…`` versus ``admin@…`` would be a support incident, not a security
    control.
    """
    return str(value or "").strip().lower()


@dataclass(frozen=True)
class AccessGrant:
    """What one verified identity is allowed to do.

    Frozen: a grant is resolved once per request and must not be widened by
    anything downstream.
    """

    #: The directory key that matched, normalised. Used for audit, not display.
    subject: str
    tenant_id: str
    platform_role: str
    #: Portfolio contexts this identity may select. **Empty means unrestricted
    #: within the tenant** — the common case, and deliberately the default, so
    #: that adding a user does not require enumerating every context that
    #: exists today and re-editing the file when a new one appears.
    portfolio_contexts: FrozenSet[str] = field(default_factory=frozenset)
    enabled: bool = True
    #: Human label for logs and the ``/me`` payload. Never an access decision.
    display_name: Optional[str] = None
    #: Where this grant came from, for the deployment check and audit.
    source: str = ""

    @property
    def is_operator(self) -> bool:
        return self.platform_role == ROLE_OPERATOR

    @property
    def is_client(self) -> bool:
        return self.platform_role == ROLE_CLIENT

    @property
    def restricted_to_contexts(self) -> bool:
        return bool(self.portfolio_contexts)

    def allows_context(self, context_id: Optional[str]) -> bool:
        """Whether this grant may select ``context_id``.

        ``None`` is always allowed: it means "the caller named no context", the
        route's own default, not a widening of access.
        """
        if context_id is None or not str(context_id).strip():
            return True
        if not self.portfolio_contexts:
            return True
        return _norm(context_id) in self.portfolio_contexts

    def to_public(self) -> Dict[str, Any]:
        """The non-sensitive projection safe to echo to a signed-in caller."""
        return {
            "tenantId": self.tenant_id,
            "role": self.platform_role,
            "isOperator": self.is_operator,
            "portfolioContexts": sorted(self.portfolio_contexts) or None,
            "user": self.display_name,
        }


class AccessDirectory:
    """The resolved set of grants this deployment recognises."""

    def __init__(self, grants: Dict[str, AccessGrant], *, source: str,
                 configured: bool) -> None:
        self._grants = dict(grants)
        #: Where the directory was read from ("defaults" when no file existed).
        self.source = source
        #: True when a real config file supplied at least one grant. False means
        #: nobody is provisioned, which is a deployment fault worth surfacing on
        #: /health rather than a silent "deny everyone".
        self.configured = configured

    def resolve(self, *identifiers: Optional[str]) -> Optional[AccessGrant]:
        """The grant for the first identifier that matches, or ``None``.

        Several identifiers are accepted because the two principal shapes carry
        different things: SWA sends ``userId`` (an Entra object id) plus
        ``userDetails`` (usually the email), while an Easy Auth claims list may
        carry ``preferred_username`` and ``sub``. Any of them may be the
        directory key, so all are tried.
        """
        for candidate in identifiers:
            key = _norm(candidate)
            if key and key in self._grants:
                return self._grants[key]
        return None

    def subjects(self) -> FrozenSet[str]:
        return frozenset(self._grants)

    def __len__(self) -> int:
        return len(self._grants)


def _grant_from_spec(spec: Dict[str, Any], *, source: str
                     ) -> Tuple[List[str], Optional[AccessGrant]]:
    """One config entry -> (directory keys, grant). Invalid entries are dropped."""
    identities = spec.get("identities") or spec.get("identity") or []
    if isinstance(identities, str):
        identities = [identities]
    keys = [_norm(i) for i in identities if _norm(i)]
    if not keys:
        logger.warning("access entry with no identities ignored: %r", spec)
        return [], None

    tenant_id = str(spec.get("tenant_id") or spec.get("tenant") or "").strip()
    if not tenant_id:
        logger.warning("access entry %s has no tenant_id — ignored", keys[0])
        return [], None

    role = _norm(spec.get("role") or spec.get("platform_role"))
    if role not in KNOWN_ROLES:
        logger.warning("access entry %s has unknown role %r — ignored", keys[0], role)
        return [], None

    contexts = spec.get("portfolio_contexts")
    if isinstance(contexts, str):
        contexts = [contexts]
    context_set = frozenset(_norm(c) for c in (contexts or []) if _norm(c))

    enabled = spec.get("enabled", True)
    if isinstance(enabled, str):
        enabled = _norm(enabled) not in ("false", "0", "no", "off")

    return keys, AccessGrant(
        subject=keys[0],
        tenant_id=tenant_id,
        platform_role=role,
        portfolio_contexts=context_set,
        enabled=bool(enabled),
        display_name=spec.get("display_name") or None,
        source=source,
    )


_CACHE: Dict[str, Tuple[Optional[int], AccessDirectory]] = {}
_CACHE_LOCK = threading.Lock()


#: The repo-root-anchored location, independent of the working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ANCHORED_DEFAULT = _REPO_ROOT / DEFAULT_ACCESS_CONFIG


def _config_path(path: Optional[str | Path] = None) -> Path:
    """Where to read the directory from.

    An explicit argument or ``TRAKT_ACCESS_CONFIG`` wins. Otherwise prefer the
    path anchored to the repository root over the working-directory-relative
    one that ``tenancy`` uses: a missing file here denies *everyone*, so
    resolving it must not depend on which directory the process happened to
    start in. The relative form is still honoured when it exists, so a
    deployment that overlays its own config in the working directory keeps
    working.
    """
    explicit = path or os.environ.get(ACCESS_CONFIG_ENV)
    if explicit:
        return Path(explicit)
    relative = Path(DEFAULT_ACCESS_CONFIG)
    if relative.exists():
        return relative
    return _ANCHORED_DEFAULT


def load_access_directory(path: Optional[str | Path] = None, *,
                          refresh: bool = False) -> AccessDirectory:
    """Load the directory, cached on the file's modification time.

    Keyed on ``mtime_ns`` rather than a TTL so that editing the file takes
    effect on the next request without a restart, while a steady state costs one
    ``stat`` per request instead of a YAML parse. A missing or unreadable file
    yields an EMPTY directory — that denies everyone, which is the correct
    failure direction for an authorisation source. Contrast
    ``insight_config``/``tenancy``, where a bad file falls back to permissive
    defaults because those govern presentation, not access.
    """
    resolved = _config_path(path)
    key = str(resolved)
    try:
        stamp: Optional[int] = resolved.stat().st_mtime_ns
    except OSError:
        stamp = None

    if not refresh:
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]

    grants: Dict[str, AccessGrant] = {}
    configured = False
    if stamp is not None:
        try:
            import yaml
            raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
            for spec in (raw.get("principals") or []):
                keys, grant = _grant_from_spec(spec or {}, source=key)
                if grant is None:
                    continue
                for k in keys:
                    if k in grants:
                        logger.warning(
                            "access identity %s appears more than once — the "
                            "first entry wins", k)
                        continue
                    grants[k] = grant
            configured = bool(grants)
        except Exception:  # noqa: BLE001 - a bad file must deny, not crash
            logger.exception(
                "access config %s could not be parsed — no identity is "
                "provisioned until it is fixed", resolved)
            grants, configured = {}, False
    else:
        logger.warning("access config %s not found — no identity is provisioned",
                       resolved)

    directory = AccessDirectory(
        grants, source=key if stamp is not None else "missing", configured=configured)
    with _CACHE_LOCK:
        _CACHE[key] = (stamp, directory)
    return directory


def reset_cache() -> None:
    """Drop the cached directory (tests, and after an operator edits the file)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def directory_status(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Whether the directory is usable, for ``/health`` and a deployment check.

    Returned as data rather than raised so a misconfigured deployment is
    diagnosable from outside without having to authenticate first.
    """
    directory = load_access_directory(path)
    return {
        "configured": directory.configured,
        "source": directory.source,
        "provisioned_identities": len(directory),
    }


def resolve_grant(
    *identifiers: Optional[str],
    path: Optional[str | Path] = None,
    directory: Optional[AccessDirectory] = None,
) -> AccessGrant:
    """The grant for a verified identity, or raise the governed 403.

    ``identifiers`` must all come from a *verified* principal. Passing anything
    a caller supplied would defeat the entire mechanism.
    """
    directory = directory if directory is not None else load_access_directory(path)
    grant = directory.resolve(*identifiers)
    if grant is None:
        raise TraktError(
            ErrorCode.ACCESS_NOT_PROVISIONED,
            NOT_PROVISIONED_MESSAGE,
            details={"provisioned": False},
        )
    if not grant.enabled:
        raise TraktError(
            ErrorCode.ACCESS_DISABLED,
            DISABLED_MESSAGE,
            details={"provisioned": True, "enabled": False},
        )
    return grant


def authorise_context(grant: AccessGrant, context_id: Optional[str]) -> None:
    """Raise unless ``grant`` may select ``context_id``.

    Separate from :func:`trakt_core.tenancy.authorise_portfolio_access`, which
    gates the *portfolio* (which book). This gates the *context* (which
    workspace lens over that book). A deployment may restrict a user to one
    origination channel without restricting the portfolio itself.
    """
    if grant.allows_context(context_id):
        return
    raise TraktError(
        ErrorCode.PERMISSION_DENIED,
        "This account is not authorised for the requested portfolio context.",
        details={"portfolio_context": str(context_id)},
    )


def scopes_for(grant: AccessGrant) -> FrozenSet[str]:
    """The scope set a grant carries.

    Both roles currently hold the same read-only MI scopes; the split exists so
    an operator-only capability can be gated later without revisiting every call
    site. Imported lazily to keep this module free of a context import cycle.
    """
    from .context import DEFAULT_MI_SCOPES

    return DEFAULT_MI_SCOPES


def iter_grants(directory: AccessDirectory) -> Iterable[Tuple[str, AccessGrant]]:
    """Directory contents, for a deployment check or an admin listing."""
    for subject in sorted(directory.subjects()):
        grant = directory.resolve(subject)
        if grant is not None:
            yield subject, grant
