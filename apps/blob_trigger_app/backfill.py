"""apps.blob_trigger_app.backfill — process historical packs already in the containers.

Event Grid is NOT retroactive: any monthly funded pack or weekly pipeline file
uploaded BEFORE the BlobCreated subscription existed never fired, and Event Grid
will never replay it. This CLI enumerates those folders and drives the SAME
decision core a live event would (:func:`router.handle_blob_event`) — per folder,
chronologically — so history is processed exactly once:

    * recurring "significantly the same" packs AUTO-APPROVE (approval policy);
    * new sources / material changes route to one-click ``pending_review``;
    * durable idempotency (the trakt-state run ledger) means a re-run is a no-op
      unless ``--force``.

It uses the storage abstraction, so it works against a local filesystem copy of
the container tree (``TRAKT_STORAGE_BACKEND=file``) or real Azure Blob.

It is also the SCOPED RERUN path for a single pack. Unlike ``ops rerun`` — which
replays a run record's ``input_dir`` and therefore depends on an ephemeral run
scratch that Azure has usually reclaimed — backfill re-fetches the pack's source
files from the container every time, so it can always localise the source. A
scoped rerun additionally localises the pack's APPROVED target-first decisions
from ``runs/{pack_key}/onboarding/34_target_first_decisions_approved.yaml`` and
hands them to the router, so onboarding applies the approved mappings BEFORE
readiness and the handoff package are computed.

Usage::

    python -m apps.blob_trigger_app.backfill --container raw-v2            # process
    python -m apps.blob_trigger_app.backfill --container raw-v2 --dry-run  # plan only
    python -m apps.blob_trigger_app.backfill --container raw-v2 --force    # reprocess

    # scoped rerun of one pack, applying its approved decisions
    python -m apps.blob_trigger_app.backfill --container raw-v2 --force \\
        --pack-key ERE_acquired_001_funded_ad_hoc_2026-06-30
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import approval_policy as _ap
from . import router as R
from .layout import Layout
from .path_parser import PathParseError, parse_blob_path
from .persistence import ProductionPersistence
from .schema_fingerprint import fingerprint_pack
from .source_registry import SourceRegistry
from .storage import Storage, join_uri, open_storage

logger = logging.getLogger("trakt.blob_trigger.backfill")

_TABULAR_EXT = (".csv", ".xlsx", ".xls", ".xlsm")
#: The historical scope the task targets: monthly funded + weekly pipeline.
DEFAULT_SCOPE: Tuple[Tuple[str, str], ...] = (("funded", "monthly"), ("pipeline", "weekly"))
#: The approved target-first decisions artefact, persisted per pack at
#: ``runs/{pack_key}/onboarding/34_target_first_decisions_approved.yaml``.
APPROVED_DECISIONS_NAME = "34_target_first_decisions_approved.yaml"


class MissingApprovedDecisions(RuntimeError):
    """Raised when a run demanded approved decisions the pack does not have."""


@dataclass
class PackFolder:
    prefix: str                       # blob://container/client/book/dataset/freq/pid/period
    client_id: str
    source_book_type: Optional[str]
    dataset: str
    frequency: str
    source_portfolio_id: str
    reporting_period: str
    data_file_uris: List[str] = field(default_factory=list)
    marker_uri: Optional[str] = None

    @property
    def marker_blob_path(self) -> str:
        return f"{self.prefix}/{ '_READY.json' }".replace("blob://", "")

    def sort_key(self) -> Tuple[str, str, str]:
        # Chronological: oldest reporting period first, then source, then dataset.
        return (self.reporting_period, self.source_portfolio_id, self.dataset)

    @property
    def pack_key(self) -> str:
        """The durable pack key — identical to the one the router derives, so a
        scoped rerun addresses exactly the run record the live event created."""
        return R.pack_key_for(
            client_id=self.client_id, source_portfolio_id=self.source_portfolio_id,
            dataset=self.dataset, frequency=self.frequency,
            reporting_period=self.reporting_period)


@dataclass(frozen=True)
class PackSelector:
    """A scope narrowing the backfill to specific pack(s).

    Selecting explicitly is a deliberate operator instruction, so a selected pack
    is NOT filtered out by the historical ``scope`` default (which only covers
    monthly funded + weekly pipeline). That default exists to bound a bulk sweep
    of history; it must not silently drop the one pack an operator named — an
    ad-hoc acquired pack (``funded/ad_hoc``) is outside it by construction.
    """
    pack_key: str = ""
    client_id: str = ""
    source_portfolio_id: str = ""
    dataset: str = ""
    frequency: str = ""
    reporting_period: str = ""

    def is_empty(self) -> bool:
        return not any((self.pack_key, self.client_id, self.source_portfolio_id,
                        self.dataset, self.frequency, self.reporting_period))

    def matches(self, pf: "PackFolder") -> bool:
        if self.is_empty():
            return True
        if self.pack_key and pf.pack_key != self.pack_key:
            return False
        for attr in ("client_id", "source_portfolio_id", "dataset",
                     "frequency", "reporting_period"):
            wanted = getattr(self, attr)
            if wanted and getattr(pf, attr) != wanted:
                return False
        return True

    def describe(self) -> str:
        parts = [f"{k}={v}" for k, v in (
            ("pack_key", self.pack_key), ("client_id", self.client_id),
            ("source_portfolio_id", self.source_portfolio_id),
            ("dataset", self.dataset), ("frequency", self.frequency),
            ("reporting_period", self.reporting_period)) if v]
        return ", ".join(parts) or "all in-scope packs"


def scan_folders(storage: Storage, container: str, *,
                 marker: str = "_READY.json",
                 scope: Optional[Tuple[Tuple[str, str], ...]] = None,
                 selector: Optional[PackSelector] = None,
                 ) -> "Tuple[List[PackFolder], List[Dict[str, str]]]":
    """Return ``(packs, skipped)``. ``packs`` are complete in-scope pack folders
    (each period folder with ≥1 tabular data file), sorted chronologically.
    ``skipped`` records every folder that carried tabular data but was NOT
    enumerated — a path that did not match the convention (e.g. a period folder
    ``2025_09_08`` under an old separator) or a dataset/frequency outside ``scope`` —
    with a reason, so backfill never silently drops a pack.

    ``selector`` narrows the sweep to specific pack(s). An explicit selection also
    lifts the dataset/frequency ``scope`` filter for the packs it names, because
    the operator has said which pack they mean — see :class:`PackSelector`.
    """
    selector = selector or PackSelector()
    scoped = not selector.is_empty()
    scope_set = set(scope or DEFAULT_SCOPE)
    groups: Dict[str, PackFolder] = {}
    markers: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    for uri in storage.list(f"blob://{container}/"):
        name = uri.rsplit("/", 1)[-1]
        folder = uri.rsplit("/", 1)[0]
        if name == marker:
            markers[folder] = uri
            continue
        if Path(name).suffix.lower() not in _TABULAR_EXT:
            continue
        try:
            parsed = parse_blob_path(uri, container)
        except PathParseError as exc:
            skipped.setdefault(folder, f"path_parse_error: {exc}")
            continue
        pf = groups.get(folder)
        if pf is None:
            pf = PackFolder(
                prefix=folder, client_id=parsed.client_id,
                source_book_type=parsed.source_book_type, dataset=parsed.dataset,
                frequency=parsed.frequency, source_portfolio_id=parsed.source_portfolio_id,
                reporting_period=parsed.reporting_period)
        if scoped:
            # An explicitly selected pack is processed on the operator's say-so;
            # the historical dataset/frequency default does not apply to it.
            if not selector.matches(pf):
                skipped.setdefault(folder, f"not_selected: {selector.describe()}")
                continue
        elif (parsed.dataset, parsed.frequency) not in scope_set:
            skipped.setdefault(folder, f"out_of_scope: {parsed.dataset}/{parsed.frequency}")
            continue
        groups.setdefault(folder, pf)
        groups[folder].data_file_uris.append(uri)
    for folder, pf in groups.items():
        pf.data_file_uris.sort()
        pf.marker_uri = markers.get(folder)
    # A folder that ended up enumerated is not "skipped".
    skips = [{"folder": f, "reason": r} for f, r in skipped.items() if f not in groups]
    return sorted(groups.values(), key=lambda p: p.sort_key()), skips


def enumerate_packs(storage: Storage, container: str, *,
                    marker: str = "_READY.json",
                    scope: Optional[Tuple[Tuple[str, str], ...]] = None
                    ) -> List[PackFolder]:
    """The in-scope pack folders, chronologically (see :func:`scan_folders`)."""
    return scan_folders(storage, container, marker=marker, scope=scope)[0]


def _read_marker_meta(storage: Storage, pf: PackFolder) -> Dict[str, Any]:
    if not pf.marker_uri:
        return {}
    try:
        raw = storage.read_text(pf.marker_uri)
        return json.loads(raw) if raw.strip() else {}
    except Exception:  # noqa: BLE001 — a malformed marker never blocks backfill
        return {}


def _plan_route(registry: SourceRegistry, pf: PackFolder, fingerprint,
                sheet_columns: Dict[str, List[str]]) -> str:
    """Mirror the router's routing decision WITHOUT side effects (for --dry-run)."""
    rec = registry.lookup(pf.client_id, pf.source_portfolio_id, pf.dataset, pf.frequency)
    if rec is None or not rec.has_approved_mapping:
        return "new_source→pending_review"
    if rec.expected_schema_fingerprint == fingerprint:
        return "deterministic"
    if not (rec.file_role_schemas and sheet_columns):
        return "schema_drift→pending_review"
    res = _ap.classify(old_role_schemas=rec.file_role_schemas or {},
                       new_role_schemas=sheet_columns,
                       old_fingerprint=rec.expected_schema_fingerprint,
                       new_fingerprint=fingerprint)
    return "auto_approve" if res.auto_approvable else "material→pending_review"


def localise_approved_decisions(
    persistence: ProductionPersistence, pack_key: str, dest_dir: str,
    *, required: bool = False,
) -> Tuple[Optional[str], str]:
    """Fetch this pack's APPROVED target-first decisions for a deterministic rerun.

    Reads exactly one artefact — ``runs/{pack_key}/onboarding/
    34_target_first_decisions_approved.yaml`` — for THIS pack key. If the pack has
    no approved decisions the result is ``(None, reason)``: there is deliberately
    no fallback to another pack's decisions, to the pending (unapproved) 34_ file,
    or to a registry mapping, because applying someone else's approved mapping is
    worse than applying none.

    Returns ``(local_path_or_None, status)``. With ``required=True`` an absent
    artefact raises :class:`MissingApprovedDecisions` rather than quietly running
    without it.
    """
    try:
        present = persistence.has_approved_decisions(pack_key)
    except Exception as exc:  # noqa: BLE001 — a state-store read must not crash the sweep
        if required:
            raise MissingApprovedDecisions(
                f"could not read approved decisions for {pack_key}: {exc}") from exc
        return None, f"lookup_failed: {exc}"
    if not present:
        msg = (f"no approved decisions for {pack_key} at "
               f"{persistence.approved_decisions_uri(pack_key)}")
        if required:
            raise MissingApprovedDecisions(
                msg + " — run `ops approve-recommendations` for this pack first")
        return None, "absent"
    local = persistence.localise_approved_decisions(pack_key, dest_dir)
    if not local or not Path(local).exists():
        msg = f"approved decisions for {pack_key} could not be localised"
        if required:
            raise MissingApprovedDecisions(msg)
        return None, "localise_failed"
    return str(local), "applied"


def run_backfill(
    storage: Storage,
    persistence: ProductionPersistence,
    registry: SourceRegistry,
    *,
    container: str,
    scope: Optional[Tuple[Tuple[str, str], ...]] = None,
    selector: Optional[PackSelector] = None,
    dry_run: bool = False,
    force: bool = False,
    limit: Optional[int] = None,
    marker: str = "_READY.json",
    out_dir: Optional[str] = None,
    apply_approved_decisions: bool = True,
    require_approved_decisions: bool = False,
    orchestrator_invoker: Optional[Callable[..., Dict[str, Any]]] = None,
    assembler_refresher: Optional[Callable[..., Dict[str, Any]]] = None,
    regime_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Enumerate + drive every in-scope historical pack through the router. Returns
    one result dict per folder (dry-run: the PLAN; real: the event manifest tail).

    ``selector`` narrows the sweep to specific pack(s) — this is the proven rerun
    path, because the pack's source files are re-fetched from the container here
    rather than assumed to still be on an ephemeral run scratch.

    When a pack has APPROVED target-first decisions in the state store they are
    localised and passed to the router as ``applied_decisions_path``, so the
    approved mappings are applied by onboarding BEFORE readiness and the handoff
    are computed. ``require_approved_decisions`` turns an absent artefact into a
    loud failure instead of a silent unapproved run.
    """
    from .runtime_paths import ensure_output_root
    out_dir = out_dir or ensure_output_root()
    if regime_runner is None:
        from .regime_runner import default_regime_runner
        regime_runner = default_regime_runner

    packs, skipped = scan_folders(storage, container, marker=marker, scope=scope,
                                  selector=selector)
    for s in skipped:
        logger.warning("BACKFILL SKIPPED %s — %s", s["folder"], s["reason"])
    if limit:
        packs = packs[:limit]
    results: List[Dict[str, Any]] = []
    for i, pf in enumerate(packs, 1):
        marker_meta = _read_marker_meta(storage, pf)
        if force:
            marker_meta = {**marker_meta, "force_reprocess": True}
        with tempfile.TemporaryDirectory(prefix="backfill_pack_") as scratch:
            # The pack's SOURCE files and any control artefacts live in separate
            # directories: onboarding inventories the source directory, so a
            # decisions artefact sitting among the source files would be
            # inventoried as source. Both are cleaned up with the scratch.
            td = str(Path(scratch) / "pack")
            Path(td).mkdir(parents=True, exist_ok=True)
            decisions_dir = Path(scratch) / "decisions"
            decisions_dir.mkdir(parents=True, exist_ok=True)
            local = []
            for uri in pf.data_file_uris:
                dest = Path(td) / uri.rsplit("/", 1)[-1]
                try:
                    storage.download_file(uri, dest)
                    local.append(str(dest))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("backfill: download failed %s: %s", uri, exc)
            if not local:
                results.append({"prefix": pf.prefix, "status": "skipped_no_data"})
                continue
            role_schemas = R.role_schemas_for_pack(registry, pf.marker_blob_path, container)
            aliases = R.aliases_for_pack(registry, pf.marker_blob_path, container)
            schema = fingerprint_pack(local, role_schemas=role_schemas, aliases=aliases)
            pack_names = [Path(f).name for f in local]

            # Approved target-first decisions for THIS pack, applied by onboarding
            # before readiness is calculated (a deterministic-apply rerun).
            decisions_path, decisions_status = (None, "not_requested")
            if apply_approved_decisions:
                decisions_path, decisions_status = localise_approved_decisions(
                    persistence, pf.pack_key, str(decisions_dir),
                    required=require_approved_decisions)
                if decisions_path:
                    marker_meta = {**marker_meta,
                                   "applied_decisions_path": decisions_path}

            if dry_run:
                plan = _plan_route(registry, pf, schema.fingerprint,
                                   dict(schema.sheet_columns or {}))
                if decisions_path:
                    plan = "approved_decisions→deterministic"
                row = {"n": i, "prefix": pf.prefix, "period": pf.reporting_period,
                       "dataset": pf.dataset, "frequency": pf.frequency,
                       "source_portfolio_id": pf.source_portfolio_id,
                       "pack_key": pf.pack_key,
                       "approved_decisions": decisions_status,
                       "fingerprint": schema.fingerprint, "planned_route": plan,
                       "data_files": pack_names}
                logger.info("PLAN %s %s/%s %s → %s", pf.reporting_period, pf.dataset,
                            pf.frequency, pf.source_portfolio_id, plan)
                results.append(row)
                continue

            primary = str(sorted(local)[0])
            manifest = R.handle_blob_event(
                pf.marker_blob_path, registry=registry, out_dir=out_dir,
                container=container, pack_marker=marker,
                input_dir_override=td, local_input_path=primary,
                schema_info=schema, marker_metadata=marker_meta,
                pack_files=pack_names, persistence=persistence,
                regime_runner=regime_runner,
                **({"orchestrator_invoker": orchestrator_invoker} if orchestrator_invoker else {}),
                **({"assembler_refresher": assembler_refresher} if assembler_refresher else {}))
            row = {"n": i, "prefix": pf.prefix, "period": pf.reporting_period,
                   "dataset": pf.dataset, "frequency": pf.frequency,
                   "source_portfolio_id": pf.source_portfolio_id,
                   "pack_key": pf.pack_key,
                   "approved_decisions": decisions_status,
                   "applied_decisions_path": decisions_path or "",
                   "status": manifest.get("status"),
                   "decision": manifest.get("decision"),
                   "event_decision": manifest.get("event_decision"),
                   "auto_approved": bool(manifest.get("auto_approved")),
                   "run_id": manifest.get("orchestrator_run_id"),
                   "central_canonical_uri": manifest.get("central_canonical_uri")}
            logger.info("BACKFILL %s %s/%s %s → status=%s decision=%s%s",
                        pf.reporting_period, pf.dataset, pf.frequency,
                        pf.source_portfolio_id, row["status"], row["decision"],
                        " (auto-approved)" if row["auto_approved"] else "")
            results.append(row)
    # Surface folders that carried data but could not be enumerated (a path that
    # did not match the convention) so they are never silently dropped.
    for s in skipped:
        if s["reason"].startswith("path_parse_error"):
            results.append({"prefix": s["folder"], "status": "skipped_unparseable",
                            "reason": s["reason"]})
    return results


def _parse_scope(values: Optional[List[str]]) -> Optional[Tuple[Tuple[str, str], ...]]:
    if not values:
        return None
    out = []
    for v in values:
        ds, _, fr = v.partition(":")
        if ds and fr:
            out.append((ds.strip(), fr.strip()))
    return tuple(out) or None


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        prog="python -m apps.blob_trigger_app.backfill",
        description="Process historical monthly funded + weekly pipeline packs "
                    "already in the containers (Event Grid never replays them).")
    p.add_argument("--container", default=None,
                   help="Raw container (default: TRAKT_RAW_CONTAINER / layout).")
    p.add_argument("--dry-run", action="store_true", help="Enumerate + plan; process nothing.")
    p.add_argument("--force", action="store_true", help="Reprocess even if already processed.")
    p.add_argument("--limit", type=int, default=None, help="Process at most N packs.")
    p.add_argument("--scope", action="append", default=None,
                   help="dataset:frequency pair to include (repeatable). "
                        "Default: funded:monthly and pipeline:weekly.")
    p.add_argument("--registry", default=None, help="Registry path/URI override.")
    p.add_argument("--local-root", default=None,
                   help="Local root emulating blob containers (local/dev).")
    p.add_argument("--pack-key", default=None,
                   help="Process only this pack, e.g. "
                        "ERE_acquired_001_funded_ad_hoc_2026-06-30. An explicit "
                        "selection ignores the dataset/frequency --scope default.")
    p.add_argument("--client-id", default=None, help="Select packs for one client.")
    p.add_argument("--source-portfolio-id", default=None,
                   help="Select packs for one source portfolio, e.g. acquired_001.")
    p.add_argument("--dataset", default=None, help="Select one dataset (funded/pipeline).")
    p.add_argument("--frequency", default=None,
                   help="Select one frequency (monthly/weekly/ad_hoc/…).")
    p.add_argument("--period", default=None, help="Select one reporting period.")
    p.add_argument("--no-approved-decisions", action="store_true",
                   help="Do NOT apply the pack's approved target-first decisions.")
    p.add_argument("--require-approved-decisions", action="store_true",
                   help="Fail if a selected pack has no approved decisions "
                        "artefact (never runs unapproved by accident).")
    args = p.parse_args(argv)

    selector = PackSelector(
        pack_key=args.pack_key or "", client_id=args.client_id or "",
        source_portfolio_id=args.source_portfolio_id or "",
        dataset=args.dataset or "", frequency=args.frequency or "",
        reporting_period=args.period or "")

    storage = open_storage(local_root=args.local_root)
    layout = Layout.from_env()
    persistence = ProductionPersistence(storage, layout)
    registry = SourceRegistry(args.registry or layout.registry_uri, storage=storage)
    container = args.container or layout.raw_container

    if not selector.is_empty():
        print(f"BACKFILL SCOPE: {selector.describe()}")
    try:
        results = run_backfill(
            storage, persistence, registry, container=container,
            scope=_parse_scope(args.scope), selector=selector,
            dry_run=args.dry_run, force=args.force, limit=args.limit,
            apply_approved_decisions=not args.no_approved_decisions,
            require_approved_decisions=args.require_approved_decisions)
    except MissingApprovedDecisions as exc:
        print(f"\n✗ {exc}")
        return 2

    unparseable = [r for r in results if r.get("status") == "skipped_unparseable"]
    packs = [r for r in results if r.get("status") != "skipped_unparseable"]
    processed = [r for r in packs if r.get("status") == "processed"]
    auto = [r for r in packs if r.get("auto_approved")]
    pending = [r for r in packs if r.get("status") == "pending_review"]
    already = [r for r in packs if r.get("status") == "already_processed"]
    print(f"\nBACKFILL {'PLAN' if args.dry_run else 'COMPLETE'}: {len(packs)} pack(s)")
    if not args.dry_run:
        print(f"  processed={len(processed)}  auto_approved={len(auto)}  "
              f"pending_review(one-click)={len(pending)}  already_processed={len(already)}")
    for r in packs:
        if args.dry_run:
            print(f"  [{r.get('n')}] {r['period']} {r['dataset']}/{r['frequency']} "
                  f"{r['source_portfolio_id']} → {r['planned_route']}")
        else:
            dec = r.get("approved_decisions", "")
            print(f"  [{r.get('n')}] {r.get('period')} {r.get('dataset')}/{r.get('frequency')} "
                  f"{r.get('source_portfolio_id')} → {r.get('status')}/{r.get('decision')}"
                  + (f"  [approved decisions: {dec}]" if dec == "applied" else ""))
    if unparseable:
        print(f"\n  ⚠ SKIPPED {len(unparseable)} folder(s) that did not match the path "
              f"convention (fix the folder name, then re-run):")
        for r in unparseable:
            print(f"    - {r['prefix']}  [{r['reason']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
