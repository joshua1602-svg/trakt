"""apps.blob_trigger_app.assembler_refresh — refresh the central platform canonical.

After a funded portfolio pack processes successfully, the Orchestrator has
produced an accepted, provenance-stamped canonical for *that* portfolio. To keep
the **central platform canonical** current across portfolios (direct_001 +
acquired_001 + acquired_002 …) we publish each portfolio's latest accepted
canonical into a stable per-client store and re-run the **Assembler Agent** over
that store. The Assembler consolidates the *latest accepted canonical per
source_portfolio_id* — it never reprocesses raw files.

**The store must be hydrated from durable storage first.** The local store lives
under the run output root, which in the managed service is an ephemeral container
scratch (``/tmp/trakt/blob_trigger``) reclaimed between runs. Publishing only the
portfolio this run processed therefore left the Assembler discovering exactly one
canonical, and the "platform" tape was a copy of whichever portfolio ran last —
silently dropping every other accepted portfolio. Durable
``accepted/{client}/*_canonical_typed.csv`` is the real per-client store, so it is
pulled down before assembly and the current run's canonical is published over its
own entry.

The seam is injectable (like the Orchestrator invoker) so the router can be
unit-tested without running the real Assembler.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("trakt.blob_trigger.assembler_refresh")

#: Discovery-matching name for a portfolio's accepted canonical in the store.
CANONICAL_SUFFIX = "_canonical_typed.csv"

# A refresher publishes one portfolio's accepted canonical, then rebuilds the
# central platform canonical, returning {central_canonical_path, assembler_run_id,
# portfolios, ...}.
AssemblerRefresher = Callable[..., Dict[str, Any]]


def accepted_store_dir(accepted_root: str | Path, client_id: str) -> Path:
    return Path(accepted_root) / client_id


def publish_accepted_canonical(
    accepted_root: str | Path, *, client_id: str, source_portfolio_id: str,
    canonical_path: str | Path,
) -> Path:
    """Copy a portfolio's accepted canonical into the per-client store under a
    stable, discovery-matching name (``{pid}_canonical_typed.csv``).

    One file per portfolio (overwritten each run) keeps the Assembler's
    latest-per-portfolio selection unambiguous.
    """
    store = accepted_store_dir(accepted_root, client_id)
    store.mkdir(parents=True, exist_ok=True)
    dest = store / f"{source_portfolio_id}_canonical_typed.csv"
    shutil.copyfile(str(canonical_path), str(dest))
    return dest


def portfolio_id_from_accepted_uri(uri: str) -> str:
    """The source_portfolio_id an accepted canonical's path encodes.

    Path-derived identity is a DISCOVERY key only — it says which file to pull,
    never which portfolio the rows belong to. Row-level attribution always comes
    from the canonical ``source_portfolio_id`` column, which the Assembler
    requires as provenance and never overwrites.
    """
    name = str(uri or "").rsplit("/", 1)[-1]
    return name[: -len(CANONICAL_SUFFIX)] if name.endswith(CANONICAL_SUFFIX) else ""


def hydrate_accepted_store(
    persistence: Any, accepted_root: str | Path, client_id: str, *,
    registry: Any = None,
) -> Dict[str, Any]:
    """Pull every ACTIVE accepted canonical for the client into the local store.

    Without this the Assembler only ever sees the portfolio the current run
    published, because the local store is an ephemeral scratch directory.

    A portfolio is excluded ONLY when the source registry explicitly says it is
    retired. An unknown portfolio (no registry record — e.g. accepted before the
    registry existed) is INCLUDED: dropping rows because a lookup missed would be
    a silent data loss, whereas including them is visible and correctable.

    Returns ``{hydrated, excluded, errors}`` for the manifest, so what was pulled
    and what was skipped is always reportable.
    """
    from .source_registry import STATUS_RETIRED

    store = accepted_store_dir(accepted_root, client_id)
    store.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"hydrated": [], "excluded": [], "errors": []}
    if persistence is None:
        return report

    try:
        prefix = persistence.layout.accepted_prefix(client_id)
        uris = [u for u in persistence.storage.list(prefix)
                if u.endswith(CANONICAL_SUFFIX)]
    except Exception as exc:  # noqa: BLE001 — never fail the pack on a store read
        report["errors"].append(f"list_accepted: {exc}")
        return report

    for uri in sorted(uris):
        pid = portfolio_id_from_accepted_uri(uri)
        if not pid:
            report["errors"].append(f"unparseable_accepted_uri: {uri}")
            continue
        if registry is not None and _is_retired(registry, client_id, pid, STATUS_RETIRED):
            report["excluded"].append({"source_portfolio_id": pid,
                                       "reason": "registry_status_retired"})
            continue
        try:
            persistence.storage.download_file(uri, store / f"{pid}{CANONICAL_SUFFIX}")
            report["hydrated"].append(pid)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"download {pid}: {exc}")
    return report


def _is_retired(registry: Any, client_id: str, source_portfolio_id: str,
                retired_status: str) -> bool:
    """True only when the registry EXPLICITLY marks this portfolio retired."""
    try:
        records = registry.records_for_portfolio(client_id, source_portfolio_id)
    except Exception:  # noqa: BLE001 — an unreadable registry never drops data
        return False
    if not records:
        return False
    # Retired only when every record for the portfolio says so; one active
    # dataset/frequency is enough to keep it in the platform tape.
    return all(str(getattr(r, "status", "")) == retired_status for r in records)


def default_assembler_refresher(
    *,
    client_id: str,
    source_portfolio_id: str,
    canonical_path: str,
    accepted_root: str,
    platform_out_dir: str,
    target: str = "mi",
    run_regime: bool = False,
    regime: Optional[str] = None,
    persistence: Any = None,
    registry: Any = None,
) -> Dict[str, Any]:
    """Publish the accepted canonical and rebuild the central platform canonical.

    Assembles ALL of the client's active accepted portfolios, not just the one
    this run processed: the durable ``accepted/{client}/`` store is hydrated into
    the local store first, then this run's canonical is published over its own
    entry. Without the hydration step the platform tape is whatever portfolio ran
    last (see the module docstring).

    Uses the MI pipeline for the cross-portfolio consolidation (the per-pack
    Orchestrator run already handled any Regime projection for that portfolio);
    the central canonical is the deliverable here.
    """
    if not canonical_path or not Path(canonical_path).exists():
        return {"skipped": "no accepted canonical to assemble",
                "central_canonical_path": None, "portfolios": []}
    from engine.assembler_agent import run_assembler_agent

    # 1. Every other portfolio already accepted for this client.
    hydration = hydrate_accepted_store(
        persistence, accepted_root, client_id, registry=registry)

    # 2. This run's portfolio, published over its own entry — never over another's.
    publish_accepted_canonical(
        accepted_root, client_id=client_id,
        source_portfolio_id=source_portfolio_id, canonical_path=canonical_path)

    store = accepted_store_dir(accepted_root, client_id)
    portfolios = _portfolios_in_store(store)
    if len(portfolios) == 1 and hydration.get("errors"):
        # One portfolio plus a store-read failure is the shape of the defect this
        # hydration exists to prevent, so say so rather than publishing quietly.
        logger.warning(
            "platform assembly for %s sees only %s — accepted-store hydration "
            "reported %s", client_id, portfolios, hydration["errors"])

    result = run_assembler_agent(
        store, platform_out_dir, client_id=client_id, pipeline="mi")
    return {
        "central_canonical_path": str(result.central_canonical_path),
        "assembler_run_id": (result.manifest or {}).get("assembler_run_id"),
        "portfolios": portfolios,
        "store_dir": str(store),
        "accepted_hydration": hydration,
    }


def _portfolios_in_store(store: Path) -> List[str]:
    try:
        return sorted({p.name[: -len(CANONICAL_SUFFIX)]
                       for p in store.glob(f"*{CANONICAL_SUFFIX}")})
    except Exception:  # noqa: BLE001
        return []
