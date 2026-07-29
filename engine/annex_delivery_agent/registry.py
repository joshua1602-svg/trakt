"""
engine.annex_delivery_agent.registry
====================================

Report-type resolution.

The registry is the seam that makes this a multi-annex agent rather than an
Annex 2 program with room for more ``if`` statements: the agent asks the
registry for a provider by ``(annex_id, version)`` and never learns which annex
it got.

Registration is explicit. There is no directory scan and no "any module in
``providers/`` is live" rule — an annex becomes production-visible only when
someone adds it to :func:`default_registry`, which is a reviewable change.
Tests register stub providers into their own :class:`AnnexRegistry` instance,
so a test provider can never leak into the production registry.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional, Tuple

from trakt_core.errors import ErrorCode, TraktError

from .contracts import AnnexDefinition

#: A provider factory. Lazy so that registering an annex does not import pandas,
#: lxml or a 9.6 MB workbook until that annex is actually run.
ProviderFactory = Callable[[], "AnnexProviderLike"]

#: Structural alias: the registry does not depend on the provider ABC, so a
#: test double needs no inheritance.
AnnexProviderLike = object

#: Sentinel meaning "whatever version this deployment currently supports".
LATEST = "latest"


class AnnexRegistry:
    """Thread-safe map of ``(annex_id, version) -> provider factory``."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._factories: Dict[Tuple[str, str], ProviderFactory] = {}
        self._definitions: Dict[Tuple[str, str], AnnexDefinition] = {}
        self._default_versions: Dict[str, str] = {}

    # -- registration ------------------------------------------------------- #
    def register(self, definition: AnnexDefinition, factory: ProviderFactory, *,
                 default_version: bool = True) -> None:
        """Register one annex at one version.

        ``default_version=True`` makes this the version resolved when a caller
        names an annex without a version. The last registration wins, which is
        how a deployment moves from 1.3.0 to 1.3.1 without touching callers.
        """
        key = (_norm(definition.annex_id), _norm(definition.report_version))
        with self._lock:
            self._factories[key] = factory
            self._definitions[key] = definition
            if default_version or definition.annex_id not in self._default_versions:
                self._default_versions[_norm(definition.annex_id)] = _norm(definition.report_version)

    def unregister(self, annex_id: str, version: Optional[str] = None) -> None:
        with self._lock:
            aid = _norm(annex_id)
            keys = [k for k in self._factories
                    if k[0] == aid and (version is None or k[1] == _norm(version))]
            for k in keys:
                self._factories.pop(k, None)
                self._definitions.pop(k, None)
            if not any(k[0] == aid for k in self._factories):
                self._default_versions.pop(aid, None)

    # -- lookup ------------------------------------------------------------- #
    def annex_ids(self) -> List[str]:
        with self._lock:
            return sorted({k[0] for k in self._factories})

    def versions(self, annex_id: str) -> List[str]:
        aid = _norm(annex_id)
        with self._lock:
            return sorted(k[1] for k in self._factories if k[0] == aid)

    def definitions(self) -> List[AnnexDefinition]:
        with self._lock:
            return sorted(self._definitions.values(),
                          key=lambda d: (d.annex_id, d.report_version))

    def supports(self, annex_id: str, version: Optional[str] = None) -> bool:
        try:
            self._resolve_key(annex_id, version)
        except TraktError:
            return False
        return True

    def definition(self, annex_id: str, version: Optional[str] = None) -> AnnexDefinition:
        key = self._resolve_key(annex_id, version)
        with self._lock:
            return self._definitions[key]

    def resolve(self, annex_id: str, version: Optional[str] = None):
        """Return a fresh provider instance for ``(annex_id, version)``.

        Raises :class:`~trakt_core.errors.TraktError` with an operator-readable
        message when the annex or the version is not supported — the caller
        asked a reasonable question and deserves a reasonable answer, not a
        ``KeyError``.
        """
        key = self._resolve_key(annex_id, version)
        with self._lock:
            factory = self._factories[key]
        return factory()

    # -- internals ---------------------------------------------------------- #
    def _resolve_key(self, annex_id: str, version: Optional[str]) -> Tuple[str, str]:
        aid = _norm(annex_id)
        with self._lock:
            known = sorted({k[0] for k in self._factories})
            if aid not in known:
                raise TraktError(
                    ErrorCode.UNSUPPORTED_QUESTION,
                    f"'{annex_id}' is not a report this service can prepare. "
                    f"Supported reports: {', '.join(known) or 'none configured'}.",
                    details={"annex_id": annex_id, "supported": known})

            requested = _norm(version) if version else None
            if requested in (None, "", LATEST):
                resolved = self._default_versions.get(aid)
                if resolved is None:  # pragma: no cover - defensive
                    raise TraktError(
                        ErrorCode.UNSUPPORTED_QUESTION,
                        f"No version of '{annex_id}' is currently configured.",
                        details={"annex_id": annex_id})
                return (aid, resolved)

            if (aid, requested) not in self._factories:
                available = sorted(k[1] for k in self._factories if k[0] == aid)
                raise TraktError(
                    ErrorCode.UNSUPPORTED_QUESTION,
                    f"Version '{version}' of '{annex_id}' is not supported by this "
                    f"service. Supported version(s): {', '.join(available)}.",
                    details={"annex_id": annex_id, "requested_version": version,
                             "supported_versions": available})
            return (aid, requested)


def _norm(value: str) -> str:
    return str(value or "").strip()


# --------------------------------------------------------------------------- #
# The production registry
# --------------------------------------------------------------------------- #

_default: Optional[AnnexRegistry] = None
_default_lock = threading.Lock()


def default_registry() -> AnnexRegistry:
    """The process-wide registry of annexes this deployment can prepare.

    **This function is the production allow-list.** Adding Annex 3 or Annex 9
    means adding a provider module and one ``register(...)`` line here — not
    changing the agent, the lifecycle, the persistence layer or the API.
    """
    global _default
    if _default is not None:
        return _default
    with _default_lock:
        if _default is None:
            registry = AnnexRegistry()
            _register_builtin(registry)
            _default = registry
    return _default


def _register_builtin(registry: AnnexRegistry) -> None:
    from .providers.annex2 import ANNEX2_DEFINITION, Annex2Provider

    registry.register(ANNEX2_DEFINITION, Annex2Provider)

    # Annex 3 and Annex 9 are deliberately absent. No authoritative normaliser,
    # builder or schema exists for them in this repository, and a placeholder
    # registration would let an operator start a run that cannot complete.


def reset_default_registry() -> None:
    """Test hook: drop the cached production registry."""
    global _default
    with _default_lock:
        _default = None


__all__ = ["AnnexRegistry", "default_registry", "reset_default_registry", "LATEST"]
