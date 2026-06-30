"""apps.blob_trigger_app.assembler_refresh — refresh the central platform canonical.

After a funded portfolio pack processes successfully, the Orchestrator has
produced an accepted, provenance-stamped canonical for *that* portfolio. To keep
the **central platform canonical** current across portfolios (direct_001 +
acquired_001 + acquired_002 …) we publish each portfolio's latest accepted
canonical into a stable per-client store and re-run the **Assembler Agent** over
that store. The Assembler consolidates the *latest accepted canonical per
source_portfolio_id* — it never reprocesses raw files.

The seam is injectable (like the Orchestrator invoker) so the router can be
unit-tested without running the real Assembler.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
) -> Dict[str, Any]:
    """Publish the accepted canonical and rebuild the central platform canonical.

    Uses the MI pipeline for the cross-portfolio consolidation (the per-pack
    Orchestrator run already handled any Regime projection for that portfolio);
    the central canonical is the deliverable here.
    """
    if not canonical_path or not Path(canonical_path).exists():
        return {"skipped": "no accepted canonical to assemble",
                "central_canonical_path": None, "portfolios": []}
    from engine.assembler_agent import run_assembler_agent

    publish_accepted_canonical(
        accepted_root, client_id=client_id,
        source_portfolio_id=source_portfolio_id, canonical_path=canonical_path)

    store = accepted_store_dir(accepted_root, client_id)
    result = run_assembler_agent(
        store, platform_out_dir, client_id=client_id, pipeline="mi")
    portfolios = []
    try:
        portfolios = sorted(
            {p.stem.replace("_canonical_typed", "")
             for p in store.glob("*_canonical_typed.csv")})
    except Exception:  # noqa: BLE001
        pass
    return {
        "central_canonical_path": str(result.central_canonical_path),
        "assembler_run_id": (result.manifest or {}).get("assembler_run_id"),
        "portfolios": portfolios,
        "store_dir": str(store),
    }
