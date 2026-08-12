"""regulatory_watch.cli — run the Stage 1 ESMA Annex 2 watch.

Two commands:

``snapshot``
    Hash and preserve every allowlisted artefact that has a vendored local
    copy, and write the normalized Annex 2 spec beside it. This is how a
    baseline is captured; the JSON spec is what a later run compares against.

``compare``
    Compare a baseline spec (or a baseline pair of artefacts) against a
    candidate, map the deltas onto the live Trakt Annex 2 implementation and
    write the JSON + Markdown reports.

Everything runs offline from local artefacts. Retrieval is a separate, opt-in
concern (:mod:`regulatory_watch.retrieval`) and is never exercised here by
default.

Nothing this CLI does writes to an active Annex 2 configuration.

Examples::

    python -m regulatory_watch.cli snapshot \\
        --out-dir output/regulatory_watch/baseline

    python -m regulatory_watch.cli compare \\
        --baseline-spec output/regulatory_watch/baseline/annex2_spec.json \\
        --candidate-workbook <workbook.xlsx> --candidate-schema <schema.xsd> \\
        --out-dir output/regulatory_watch/run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from . import PARSER_VERSION
from .annex2_spec import SpecParseError, parse_annex2_spec
from .comparator import ComparatorError, compare
from .contracts import (
    ARTEFACT_DISCLOSURE_WORKBOOK,
    ARTEFACT_XML_SCHEMA,
    NormalizedSpec,
    SourceSnapshot,
)
from .impact import TraktImplementationIndex, assess
from .manifest import DEFAULT_MANIFEST_PATH, load_manifest
from .report import build_report, write_reports
from .snapshots import SnapshotStore, snapshot_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]

SPEC_FILENAME = "annex2_spec.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_spec(path: Path) -> NormalizedSpec:
    return NormalizedSpec.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8")))


def _write_spec(spec: NormalizedSpec, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True,
                   ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _artefact_paths(manifest, repo_root: Path):
    workbooks = manifest.of_type(ARTEFACT_DISCLOSURE_WORKBOOK)
    schemas = manifest.of_type(ARTEFACT_XML_SCHEMA)
    if not workbooks or not schemas:
        raise SystemExit("manifest must declare one disclosure workbook and "
                         "one XML schema for Stage 1")
    return (repo_root / workbooks[0].local_path,
            repo_root / schemas[0].local_path)


def _build_spec(workbook: Path, schema: Path, spec_version: str,
                snapshots: Sequence[SourceSnapshot]) -> NormalizedSpec:
    return parse_annex2_spec(workbook, schema, spec_version=spec_version,
                             snapshots=snapshots)


# --------------------------------------------------------------------------- #
# snapshot
# --------------------------------------------------------------------------- #

def cmd_snapshot(args: argparse.Namespace) -> int:
    manifest = load_manifest(Path(args.manifest) if args.manifest else None)
    out_dir = Path(args.out_dir)
    store = SnapshotStore(root=Path(args.snapshot_root or
                                    (out_dir / "snapshots")),
                          regime=manifest.regime)
    retrieved_at = args.retrieved_at or _now()

    snapshots = snapshot_manifest(store, manifest.artefacts, REPO_ROOT,
                                  retrieved_at, PARSER_VERSION)
    failed = [s for s in snapshots if not s.ok]

    workbook, schema = _artefact_paths(manifest, REPO_ROOT)
    try:
        spec = _build_spec(workbook, schema, args.spec_version, snapshots)
    except SpecParseError as exc:
        print(f"SOURCE_CHECK_FAILED: {exc}", file=sys.stderr)
        return 2

    spec_path = _write_spec(spec, out_dir / SPEC_FILENAME)
    print(f"snapshotted {len(snapshots) - len(failed)}/{len(snapshots)} "
          f"artefact(s); {len(spec.fields)} Annex 2 fields normalized")
    for s in failed:
        print(f"  SOURCE_CHECK_FAILED {s.artefact_id}: {s.retrieval_detail}")
    print(f"spec -> {spec_path}")
    return 0


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #

def cmd_compare(args: argparse.Namespace) -> int:
    manifest = load_manifest(Path(args.manifest) if args.manifest else None)
    out_dir = Path(args.out_dir)
    retrieved_at = args.retrieved_at or _now()
    store = SnapshotStore(root=Path(args.snapshot_root or
                                    (out_dir / "snapshots")),
                          regime=manifest.regime)

    # -- baseline ---------------------------------------------------------- #
    if args.baseline_spec:
        baseline = _load_spec(Path(args.baseline_spec))
    else:
        workbook, schema = _artefact_paths(manifest, REPO_ROOT)
        snaps = snapshot_manifest(store, manifest.artefacts, REPO_ROOT,
                                  retrieved_at, PARSER_VERSION)
        try:
            baseline = _build_spec(workbook, schema, args.baseline_version,
                                   snaps)
        except SpecParseError as exc:
            print(f"SOURCE_CHECK_FAILED (baseline): {exc}", file=sys.stderr)
            return 2

    # -- candidate ---------------------------------------------------------- #
    if args.candidate_spec:
        candidate = _load_spec(Path(args.candidate_spec))
    else:
        if not (args.candidate_workbook and args.candidate_schema):
            raise SystemExit("compare needs --candidate-spec, or both "
                             "--candidate-workbook and --candidate-schema")
        cand_workbook = Path(args.candidate_workbook)
        cand_schema = Path(args.candidate_schema)
        cand_snaps: List[SourceSnapshot] = []
        for artefact, path in ((manifest.of_type(ARTEFACT_DISCLOSURE_WORKBOOK)[0],
                                cand_workbook),
                               (manifest.of_type(ARTEFACT_XML_SCHEMA)[0],
                                cand_schema)):
            cand_snaps.append(store.snapshot_local(artefact, path,
                                                   retrieved_at,
                                                   PARSER_VERSION))
        try:
            candidate = _build_spec(cand_workbook, cand_schema,
                                    args.candidate_version, cand_snaps)
        except SpecParseError as exc:
            print(f"SOURCE_CHECK_FAILED (candidate): {exc}", file=sys.stderr)
            return 2

    # -- compare + impact ---------------------------------------------------- #
    try:
        deltas = compare(baseline, candidate)
    except ComparatorError as exc:
        print(f"COMPARISON_REFUSED: {exc}", file=sys.stderr)
        return 3

    index = TraktImplementationIndex.load(REPO_ROOT)
    impacts = assess(deltas, index=index)

    report = build_report(
        baseline, candidate, deltas, impacts,
        generated_at=args.generated_at or _now(),
        evidence={
            "source_manifest": manifest.path,
            "allowlisted_sources": [a.artefact_id for a in manifest.artefacts],
            "declared_without_local_copy":
                [a.artefact_id for a in manifest.declared_without_local_copy()],
            "trakt_configs_read":
                ["config/system/fields_registry.yaml",
                 "config/system/esma_code_order.yaml",
                 "config/system/esma_model_structure.yaml",
                 "config/system/enum_mapping.yaml",
                 "config/regime/annex2_delivery_rules.yaml",
                 "config/regime/annex2_field_universe.yaml",
                 "config/delivery/annex2_field_xsd_path_map.yaml"],
            "mutating_writes": "none — Stage 1 is observational",
        })
    paths = write_reports(report, out_dir)
    print(f"outcome={report.outcome} source={report.source_status} "
          f"spec={report.spec_status} deltas={report.regulatory_delta_count} "
          f"impacts={report.implementation_impact_count}")
    print(f"json -> {paths['json']}")
    print(f"markdown -> {paths['markdown']}")
    return 0


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regulatory_watch",
        description="ESMA Annex 2 Regulatory Watch (Stage 1, observational)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH),
                        help="source allowlist YAML")
    parser.add_argument("--snapshot-root", default=None,
                        help="immutable snapshot store root")
    parser.add_argument("--retrieved-at", default=None,
                        help="ISO timestamp recorded on snapshots "
                             "(defaults to now; set for reproducible output)")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot",
                          help="hash local artefacts and normalize the spec")
    snap.add_argument("--out-dir", required=True)
    snap.add_argument("--spec-version", default="local-baseline")
    snap.set_defaults(func=cmd_snapshot)

    cmp_ = sub.add_parser("compare", help="compare two Annex 2 specs")
    cmp_.add_argument("--out-dir", required=True)
    cmp_.add_argument("--baseline-spec", default=None,
                      help="normalized baseline spec JSON")
    cmp_.add_argument("--baseline-version", default="local-baseline")
    cmp_.add_argument("--candidate-spec", default=None)
    cmp_.add_argument("--candidate-workbook", default=None)
    cmp_.add_argument("--candidate-schema", default=None)
    cmp_.add_argument("--candidate-version", default="candidate")
    cmp_.add_argument("--generated-at", default=None)
    cmp_.set_defaults(func=cmd_compare)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
