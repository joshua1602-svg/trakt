"""regulatory_watch.discovery.cli — run one Stage 1B discovery pass.

    discover -> dedupe -> classify -> report

Weekly is sufficient. There is **no scheduler in this package**: the repository
already has one (``function_app.py`` registers Azure Functions timer triggers),
so Stage 2 wires a weekly trigger to this entrypoint rather than Stage 1B
growing infrastructure of its own.

Offline by default. With ``--fixture-dir`` the pass reads payloads from local
files, which is how the tests and the demonstration run. Live retrieval
requires both ``retrieval_enabled: true`` in the allowlist and ``--live``, and
a failed fetch is reported as a discovery failure, never as "nothing new".

Exit codes::

    0  discovery completed (DISCOVERY_OK or DISCOVERY_PARTIAL)
    2  a gating source could not be checked (DISCOVERY_FAILED)
    3  the source allowlist or state store is unusable
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .contracts import (
    DISCOVERY_FAILED,
    SOURCE_FETCH_FAILED,
    SourceOutcome,
)
from .pipeline import SourcePayload, run_discovery
from .report import write_reports
from .retrieval import fetch_source, sha256_bytes
from .sources import DEFAULT_SOURCES_PATH, SourceConfigError, load_publication_sources
from .state import SeenStore, StateStoreError

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STATE_PATH = (REPO_ROOT / "output" / "regulatory_watch"
                      / "discovery" / "seen_publications.json")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fixture_payloads(sources, fixture_dir: Path,
                      retrieved_at: str) -> List[SourcePayload]:
    """Read each source's payload from ``<fixture_dir>/<source_id>.*``.

    A source with no fixture is reported as FETCH_FAILED, exactly as a live
    fetch failure would be — so an incomplete fixture set cannot masquerade as
    a clean sweep.
    """
    out: List[SourcePayload] = []
    for source in sources.sources:
        matches = sorted(fixture_dir.glob(f"{source.source_id}.*"))
        if not matches:
            out.append(SourcePayload(source, None, SourceOutcome(
                source_id=source.source_id, criticality=source.criticality,
                outcome=SOURCE_FETCH_FAILED, retrieved_at=retrieved_at,
                detail=f"no fixture for '{source.source_id}' in {fixture_dir}")))
            continue
        payload = matches[0].read_bytes()
        out.append(SourcePayload(source, payload, SourceOutcome(
            source_id=source.source_id, criticality=source.criticality,
            outcome="OK", retrieved_at=retrieved_at,
            content_sha256=sha256_bytes(payload),
            detail=f"fixture: {matches[0].name}")))
    return out


def _live_payloads(sources, retrieved_at: str) -> List[SourcePayload]:
    import urllib.request

    def fetcher(url: str) -> bytes:
        request = urllib.request.Request(
            url, headers={"User-Agent": "trakt-regulatory-watch/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()

    out: List[SourcePayload] = []
    for source in sources.sources:
        payload, outcome = fetch_source(source, sources, retrieved_at, fetcher)
        out.append(SourcePayload(source, payload, outcome))
    return out


def cmd_discover(args: argparse.Namespace) -> int:
    try:
        sources = load_publication_sources(
            Path(args.sources) if args.sources else None)
    except SourceConfigError as exc:
        print(f"SOURCE_CONFIG_ERROR: {exc}", file=sys.stderr)
        return 3

    discovery_date = args.discovery_date or _now()
    state_path = Path(args.state or DEFAULT_STATE_PATH)
    try:
        store = SeenStore.load(state_path)
    except StateStoreError as exc:
        print(f"STATE_STORE_ERROR: {exc}", file=sys.stderr)
        return 3

    if args.fixture_dir:
        payloads = _fixture_payloads(sources, Path(args.fixture_dir),
                                     discovery_date)
    elif args.live:
        payloads = _live_payloads(sources, discovery_date)
    else:
        # Neither fixtures nor an explicit --live: report every source as
        # unchecked rather than inventing an empty sweep.
        payloads = [SourcePayload(s, None, SourceOutcome(
            source_id=s.source_id, criticality=s.criticality,
            outcome=SOURCE_FETCH_FAILED, retrieved_at=discovery_date,
            detail="no --fixture-dir and --live not requested"))
            for s in sources.sources]

    stage1a_reports: Dict[str, Path] = {}
    if args.stage1a_report:
        for pair in args.stage1a_report:
            development_id, _, path = pair.partition("=")
            if path:
                stage1a_reports[development_id] = Path(path)

    result = run_discovery(sources, payloads, store, discovery_date,
                           stage1a_reports=stage1a_reports,
                           commit_state=not args.no_commit_state)
    if not args.no_commit_state:
        store.save()

    paths = write_reports(result.report, Path(args.out_dir))
    report = result.report
    print(f"status={report.discovery_status} "
          f"sources={report.sources_checked} failed={report.sources_failed} "
          f"new={report.new_publications} "
          f"relevant={report.potentially_relevant} "
          f"review={report.review_required_count} "
          f"technical={report.technical_spec_change_candidates}")
    print(f"json -> {paths['json']}")
    print(f"markdown -> {paths['markdown']}")
    if not args.no_commit_state:
        print(f"state -> {state_path} ({len(store)} publication(s) known)")
    return 2 if report.discovery_status == DISCOVERY_FAILED else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regulatory_watch.discovery",
        description="ESMA Annex 2 publication discovery (Stage 1B, "
                    "observational)")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES_PATH),
                        help="upstream publication allowlist YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser(
        "discover", help="discover -> dedupe -> classify -> report")
    discover.add_argument("--out-dir", required=True)
    discover.add_argument("--state", default=None,
                          help="publication state store (dedupe memory)")
    discover.add_argument("--fixture-dir", default=None,
                          help="read source payloads from local files")
    discover.add_argument("--live", action="store_true",
                          help="fetch the allowlisted sources over the network "
                               "(also requires retrieval_enabled in the "
                               "allowlist)")
    discover.add_argument("--discovery-date", default=None,
                          help="ISO timestamp stamped on the run "
                               "(set for reproducible output)")
    discover.add_argument("--stage1a-report", action="append", default=[],
                          metavar="DEVELOPMENT_ID=PATH",
                          help="quote an existing Stage 1A report into a "
                               "development record")
    discover.add_argument("--no-commit-state", action="store_true",
                          help="do not update the dedupe store (dry run)")
    discover.set_defaults(func=cmd_discover)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
