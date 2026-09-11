"""The production funded catalogue, exposed as a `SnapshotStore`. Read-only.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT. It is an ADAPTER and nothing
else: every fact it returns is read from the catalogue production already owns
(`datasets.snapshot_index`) and every frame it loads comes from the loader
production already uses for a dated run (`datasets._resolve_run_dataframe`).

It creates no catalogue. There is no date list in this file, no YAML of
reporting periods, no environment variable naming snapshots, and no directory
walk — the on-disk walk that does exist is `snapshots.discover_snapshots`,
reached THROUGH `snapshot_index`, which is where production's own resolution
order between blob, disk and the loaded canonical already lives. Two lists of a
book's months could disagree, and the one a dropdown offers must be the one the
temporal runtime can resolve.

WHY THE ADAPTER LIVES IN THE API LAYER. `snapshot.store.SnapshotStore` is the
storage-neutral interface and `plan_temporal_runtime` is deterministic; neither
may learn about blobs, onboarding roots or platform canonicals. That knowledge
belongs where it already is — `mi_agent_api.datasets` — so the adapter sits
beside it and hands the temporal runtime the interface it already consumes.

SCOPE IS THE CLIENT, AND IT IS A PROPERTY OF THE OBJECT. A store built for a
request is BOUND to that request's client: any other client is refused by
`list_snapshots`, `get_snapshot` and therefore `load_loans`, whatever id a caller
supplies.

    Bound scope was not the first design and the cross-portfolio control is what
    found it. `get_snapshot` derived the client from the SNAPSHOT ID and then
    listed that client's runs, so a store built for one tenant would resolve and
    load another tenant's run if simply handed its id. Nothing on the temporal
    path does that — the runtime only ever asks for the request's own client —
    but "no caller happens to do it" is not a boundary, and a store handed
    around is exactly the object a future caller would hand an id to.

APPROVAL IS NOT INVENTED HERE. The production catalogue records `run_id`,
`reporting_date` and size for each run and carries no per-run approval or status
field; `evaluate_source_approval` governs the ACTIVE source, not each historical
run. So this adapter applies the scoping that exists (client, funded route,
a resolvable reporting date) and does not manufacture a status the estate does
not keep. A run the catalogue does not list is not selectable, which is the only
approval semantics available today.

WRITING IS REFUSED. The MI query path reads history; it does not publish it.
`register_snapshot` raises rather than silently succeeding.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from snapshot.model import SnapshotHeader, SnapshotNotFoundError, parse_date
from snapshot.store import RegistrationResult, SnapshotStore

logger = logging.getLogger("mi_agent_api.governed_snapshot_store")

#: The one route this store serves. The funded book is the only population
#: slice 2 admits, and the pipeline extracts are a different catalogue with a
#: different cadence — naming the route keeps a pipeline run structurally
#: unreachable from a funded temporal question.
FUNDED_ROUTE = "funded"

#: Snapshot identity: the production portfolioId form, `client/run`. Not a new
#: identifier — it is what `/mi/snapshot`, the dropdowns and
#: `_resolve_query_frame` already use to name a dated cut.
_SEPARATOR = "/"


def snapshot_id_for(client_id: str, run_id: str) -> str:
    return f"{client_id}{_SEPARATOR}{run_id}"


def split_snapshot_id(snapshot_id: str) -> tuple:
    client_id, _, run_id = str(snapshot_id).partition(_SEPARATOR)
    return client_id, run_id


class GovernedFundedSnapshotStore(SnapshotStore):
    """`SnapshotStore` over the production funded catalogue. Read-only.

    `index_provider` and `frame_loader` are injected so this class holds no
    import-time dependency on the API's data layer and can be exercised against
    a production-shaped index in a test. Production passes the real two.
    """

    def __init__(self, index_provider: Any, frame_loader: Any,
                 output_root: Optional[str] = None,
                 client_id: Optional[str] = None) -> None:
        self._index_provider = index_provider
        self._frame_loader = frame_loader
        self._output_root = output_root
        #: The one client this store may ever speak for. None leaves it
        #: unbound, which is for tooling that legitimately spans tenants; the
        #: serving path always binds it.
        self._client_id = str(client_id) if client_id else None

    def _in_scope(self, client_id: Any) -> bool:
        return (self._client_id is None
                or str(client_id) == self._client_id)

    # -- storage primitives ------------------------------------------------ #

    def register_snapshot(self, header: SnapshotHeader,
                          frame: pd.DataFrame) -> RegistrationResult:
        raise NotImplementedError(
            "the MI query path reads governed history and does not publish it; "
            "registration belongs to the onboarding/publishing pipeline")

    def list_snapshots(self, client_id: str, route: Optional[str] = None,
                       cadence: Optional[str] = None,
                       since: Any = None, until: Any = None
                       ) -> List[SnapshotHeader]:
        """Every governed funded run for THIS client, as headers.

        Scope is applied before anything is returned: the client must match, the
        route must be the funded one, and a run with no resolvable reporting
        date is dropped — a period that cannot be named cannot be selected, and
        guessing one would be the substitution the whole contract refuses.
        """
        if route is not None and route != FUNDED_ROUTE:
            return []
        if not self._in_scope(client_id):
            logger.warning("a store bound to one client was asked for another; "
                           "returning nothing")
            return []
        try:
            index = self._index_provider() or {}
        except Exception:                                            # noqa: BLE001
            logger.warning("the governed snapshot catalogue could not be read",
                           exc_info=True)
            return []

        headers: List[SnapshotHeader] = []
        for portfolio in (index.get("portfolios") or ()):
            if str(portfolio.get("client_id")) != str(client_id):
                continue                       # SCOPE: another client, never
            for run in (portfolio.get("runs") or ()):
                run_id = run.get("run_id")
                reporting_date = run.get("reporting_date")
                if not run_id or not reporting_date:
                    continue
                if parse_date(reporting_date) is None:
                    continue
                headers.append(SnapshotHeader(
                    client_id=str(client_id), route=FUNDED_ROUTE,
                    reporting_date=str(reporting_date),
                    source_file_id=snapshot_id_for(client_id, run_id),
                    snapshot_id=snapshot_id_for(client_id, run_id),
                    cadence=_cadence_of(index, portfolio),
                    cut_off_date=str(reporting_date),
                    source_file_name=str(index.get("source") or ""),
                    row_count=_int_or_none(run.get("loan_count")),
                    metadata={"run_id": str(run_id),
                              "catalogue_source": index.get("source")}))

        if cadence is not None:
            headers = [h for h in headers if h.cadence == cadence]
        since_date, until_date = parse_date(since), parse_date(until)
        if since_date is not None:
            headers = [h for h in headers
                       if parse_date(h.reporting_date) >= since_date]
        if until_date is not None:
            headers = [h for h in headers
                       if parse_date(h.reporting_date) <= until_date]
        return headers

    def get_snapshot(self, snapshot_id: str) -> SnapshotHeader:
        client_id, run_id = split_snapshot_id(snapshot_id)
        if not client_id or not run_id:
            raise SnapshotNotFoundError(f"malformed snapshot id {snapshot_id!r}")
        if not self._in_scope(client_id):
            raise SnapshotNotFoundError(
                f"{snapshot_id!r} belongs to another client; this store is "
                f"bound to {self._client_id!r}")
        for header in self.list_snapshots(client_id, route=FUNDED_ROUTE):
            if header.snapshot_id == snapshot_id:
                return header
        raise SnapshotNotFoundError(
            f"{snapshot_id!r} is not a governed funded run for {client_id!r}")

    def load_loans(self, snapshot_id: str) -> pd.DataFrame:
        """The run's prepared frame, from production's own dated-run loader.

        `get_snapshot` runs FIRST and is not an optimisation: it is the scope
        check. A snapshot id the catalogue does not list never reaches the
        loader, so a caller cannot name a path, another client's run or an
        unlisted cut and have it loaded.
        """
        header = self.get_snapshot(snapshot_id)
        client_id, run_id = split_snapshot_id(snapshot_id)
        try:
            frame, _report = self._frame_loader(client_id, run_id,
                                                self._output_root)
        except Exception as exc:                                     # noqa: BLE001
            raise SnapshotNotFoundError(
                f"the governed run {snapshot_id!r} could not be loaded: "
                f"{type(exc).__name__}: {exc}") from exc
        if frame is None or not len(frame):
            raise SnapshotNotFoundError(
                f"the governed run {snapshot_id!r} ({header.reporting_date}) "
                f"carries no rows")
        return frame


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cadence_of(index: Dict[str, Any], portfolio: Dict[str, Any]) -> Optional[str]:
    """The cadence the CATALOGUE declares, or None when it declares none.

    The funded catalogue is a monthly reporting cut in every deployment shape —
    dated platform canonicals and onboarding central tapes alike are published
    per reporting month — and `cadence` is how the temporal contract learns that
    a grain is honourable. It is read from the catalogue where the catalogue
    states it and defaults to `monthly` for the funded route only, which is a
    statement about the funded publishing rhythm rather than about any book.

    A deployment whose funded catalogue is not monthly would need this to come
    from the catalogue itself; that is a catalogue change, not a temporal one,
    and the temporal runtime already refuses a grain the cadence does not match.
    """
    stated = portfolio.get("cadence") or index.get("cadence")
    return str(stated) if stated else "monthly"


def build_store(datasets_module: Any = None, client_id: Optional[str] = None
                ) -> Optional[GovernedFundedSnapshotStore]:
    """The production store for ONE client, or None. Never raises.

    None means the temporal path is unavailable for this request and the legacy
    envelope serves — the same fail-closed outcome as an empty catalogue. A
    store that could not be built is never approximated with the current frame.

    `client_id` binds the scope. A caller that omits it gets an unbound store,
    which the serving path never does.
    """
    try:
        if datasets_module is None:
            from mi_agent_api import datasets as datasets_module   # type: ignore
        return GovernedFundedSnapshotStore(
            index_provider=datasets_module.snapshot_index,
            frame_loader=datasets_module._resolve_run_dataframe,
            output_root=datasets_module._onboarding_output_root(),
            client_id=client_id)
    except Exception:                                                # noqa: BLE001
        logger.warning("the governed snapshot store could not be built; the "
                       "temporal path is unavailable for this request",
                       exc_info=True)
        return None
