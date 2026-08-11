"""tests/test_regulatory_watch_end_to_end.py — manifest, snapshots, reports, CLI.

Walks the whole Stage 1 flow offline:

    authoritative snapshot -> normalized Annex 2 spec -> comparator -> impact
    -> JSON + Markdown report

and pins the two properties the watch is judged on: the report is a stable,
UI-agnostic data contract, and the run is observational (no active Annex 2
configuration is written).

Run: python -m pytest tests/test_regulatory_watch_end_to_end.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch import PARSER_VERSION, REPORT_CONTRACT_VERSION  # noqa: E402
from regulatory_watch.comparator import compare                  # noqa: E402
from regulatory_watch.contracts import (                         # noqa: E402
    ARTEFACT_DISCLOSURE_WORKBOOK,
    ARTEFACT_XML_SCHEMA,
    CHANGE_TYPES,
    IMPACT_STATUSES,
    REGULATORY_SPEC_CHANGED,
    SOURCE_CHECK_FAILED,
    SOURCE_CHANGED,
    SOURCE_CHANGED_SPEC_UNCHANGED,
    SPEC_CHANGED,
    SPEC_UNCHANGED,
    WATCH_INCONCLUSIVE,
)
from regulatory_watch.impact import TraktImplementationIndex, assess  # noqa: E402
from regulatory_watch.manifest import load_manifest              # noqa: E402
from regulatory_watch.report import (                            # noqa: E402
    build_report, to_json, to_markdown, write_reports,
)
from regulatory_watch.snapshots import (                         # noqa: E402
    SnapshotStore, sha256_file, snapshot_manifest,
)
from tests.helpers import regulatory_watch as fx                 # noqa: E402


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def test_the_shipped_manifest_is_valid_and_esma_annex2_only():
    manifest = load_manifest()
    assert manifest.regime == "ESMA_Annex2"
    assert manifest.authority == "ESMA"
    assert manifest.artefacts
    assert all(a.authority == "ESMA" for a in manifest.artefacts)
    assert all(a.regime == "ESMA_Annex2" for a in manifest.artefacts)
    assert all(u.startswith("https://www.esma.europa.eu/")
               for u in manifest.allowed_urls)


def test_the_manifest_declares_the_two_artefacts_the_normalizer_needs():
    manifest = load_manifest()
    assert len(manifest.of_type(ARTEFACT_DISCLOSURE_WORKBOOK)) == 1
    assert len(manifest.of_type(ARTEFACT_XML_SCHEMA)) == 1


def test_retrieval_is_off_by_default():
    assert load_manifest().retrieval_enabled is False


def test_every_vendored_local_path_exists():
    manifest = load_manifest()
    for artefact in manifest.local_artefacts():
        assert (_REPO / artefact.local_path).exists(), artefact.artefact_id


def test_artefacts_without_a_local_copy_are_declared_not_hidden():
    manifest = load_manifest()
    unvendored = manifest.declared_without_local_copy()
    assert unvendored, ("the technical reporting instructions are not vendored; "
                        "they must still be declared so the gap is reported")
    assert all(a.source_url for a in unvendored)


# --------------------------------------------------------------------------- #
# Snapshot store
# --------------------------------------------------------------------------- #

def test_snapshotting_is_content_addressed_and_idempotent(tmp_path):
    manifest = load_manifest()
    store = SnapshotStore(root=tmp_path / "snap", regime=manifest.regime)
    artefact = manifest.of_type(ARTEFACT_XML_SCHEMA)[0]
    source = _REPO / artefact.local_path

    first = store.snapshot_local(artefact, source, "2026-01-01T00:00:00Z",
                                 PARSER_VERSION)
    second = store.snapshot_local(artefact, source, "2026-01-02T00:00:00Z",
                                  PARSER_VERSION)
    assert first.ok and second.ok
    assert first.sha256 == second.sha256 == sha256_file(source)
    assert first.snapshot_path == second.snapshot_path
    assert Path(first.snapshot_path).exists()
    assert Path(first.snapshot_path).read_bytes() == source.read_bytes()
    assert len(store.list_snapshots(artefact.artefact_id)) == 1


def test_different_bytes_land_in_a_different_immutable_snapshot(tmp_path):
    manifest = load_manifest()
    store = SnapshotStore(root=tmp_path / "snap", regime=manifest.regime)
    artefact = manifest.of_type(ARTEFACT_XML_SCHEMA)[0]

    a = tmp_path / "a.xsd"
    a.write_text("<xs:schema>A</xs:schema>", encoding="utf-8")
    b = tmp_path / "b.xsd"
    b.write_text("<xs:schema>B</xs:schema>", encoding="utf-8")

    first = store.snapshot_local(artefact, a, "2026-01-01T00:00:00Z",
                                 PARSER_VERSION)
    second = store.snapshot_local(artefact, b, "2026-01-02T00:00:00Z",
                                  PARSER_VERSION)
    assert first.sha256 != second.sha256
    assert first.snapshot_path != second.snapshot_path
    # History preserved: the earlier snapshot is untouched.
    assert Path(first.snapshot_path).read_text(encoding="utf-8") == \
        "<xs:schema>A</xs:schema>"
    assert len(store.list_snapshots(artefact.artefact_id)) == 2


def test_snapshotting_the_manifest_reports_unvendored_sources_as_failed(
        tmp_path):
    manifest = load_manifest()
    store = SnapshotStore(root=tmp_path / "snap", regime=manifest.regime)
    snaps = snapshot_manifest(store, manifest.artefacts, _REPO,
                              "2026-01-01T00:00:00Z", PARSER_VERSION)
    assert len(snaps) == len(manifest.artefacts)
    failed = [s for s in snaps if not s.ok]
    assert failed, "an unvendored artefact must be SOURCE_CHECK_FAILED"
    assert all(s.retrieval_status == SOURCE_CHECK_FAILED for s in failed)


# --------------------------------------------------------------------------- #
# Report contract
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def index():
    return TraktImplementationIndex.load(_REPO)


def _run(baseline_snaps, candidate_snaps, mutate, index, generated_at):
    rows = fx.load_rows()
    codelists = fx.load_codelists()
    baseline = fx.build_spec(rows, codelists, spec_version="baseline-A",
                             snapshots=baseline_snaps)
    cand_rows, cand_codelists = mutate(rows, codelists)
    candidate = fx.build_spec(cand_rows, cand_codelists,
                              spec_version="candidate-B",
                              snapshots=candidate_snaps)
    deltas = compare(baseline, candidate)
    impacts = assess(deltas, index=index)
    return build_report(baseline, candidate, deltas, impacts,
                        generated_at=generated_at,
                        evidence={"scenario": "unit"})


def test_report_shape_matches_the_published_contract(index):
    report = _run([fx.snapshot("a" * 64)], [fx.snapshot("b" * 64)],
                  lambda r, c: (fx.drop_code(r, "RREL73"), c),
                  index, "2026-01-01T00:00:00Z")
    payload = json.loads(to_json(report))
    for key in ("regime", "baseline", "candidate", "source_status",
                "regulatory_delta_count", "implementation_impact_count",
                "deltas"):
        assert key in payload
    assert payload["regime"] == "ESMA_Annex2"
    assert payload["contract_version"] == REPORT_CONTRACT_VERSION
    assert payload["source_status"] == SOURCE_CHANGED
    assert payload["spec_status"] == SPEC_CHANGED
    assert payload["outcome"] == REGULATORY_SPEC_CHANGED
    assert payload["regulatory_delta_count"] == 1
    assert payload["implementation_impact_count"] > 0
    assert payload["deltas"][0]["change_type"] == "FIELD_REMOVED"
    assert payload["deltas"][0]["delta_id"] == "RREL73:FIELD_REMOVED"


def test_report_is_byte_reproducible(index):
    def build():
        return to_json(_run([fx.snapshot("a" * 64)], [fx.snapshot("b" * 64)],
                            lambda r, c: (fx.drop_code(r, "RREL73"), c),
                            index, "2026-01-01T00:00:00Z"))
    assert build() == build()


def test_report_is_ui_agnostic(index):
    """The data contract must not carry presentation concerns."""
    payload = json.loads(to_json(
        _run([fx.snapshot("a" * 64)], [fx.snapshot("b" * 64)],
             lambda r, c: (fx.drop_code(r, "RREL73"), c), index,
             "2026-01-01T00:00:00Z")))
    text = json.dumps(payload).lower()
    for forbidden in ("<div", "<span", "class=", "colour", "css", "html"):
        assert forbidden not in text


def test_every_delta_and_impact_uses_the_published_vocabularies(index):
    report = _run([fx.snapshot("a" * 64)], [fx.snapshot("b" * 64)],
                  lambda r, c: (fx.drop_code(r, "RREL73"), c), index,
                  "2026-01-01T00:00:00Z")
    assert all(d["change_type"] in CHANGE_TYPES for d in report.deltas)
    assert all(i["status"] in IMPACT_STATUSES for i in report.impacts)
    assert all(d["evidence"] for d in report.deltas)


def test_source_change_without_spec_change_is_reported_as_such(index):
    report = _run([fx.snapshot("a" * 64)], [fx.snapshot("b" * 64)],
                  lambda r, c: (r, c), index, "2026-01-01T00:00:00Z")
    assert report.source_status == SOURCE_CHANGED
    assert report.spec_status == SPEC_UNCHANGED
    assert report.outcome == SOURCE_CHANGED_SPEC_UNCHANGED
    assert report.regulatory_delta_count == 0
    markdown = to_markdown(report)
    assert "not, by itself, a regulatory change" in markdown


def test_a_failed_source_check_is_visible_in_both_outputs(index):
    report = _run([fx.snapshot("a" * 64)], [fx.failed_snapshot()],
                  lambda r, c: (r, c), index, "2026-01-01T00:00:00Z")
    assert report.source_status == SOURCE_CHECK_FAILED
    assert report.outcome == WATCH_INCONCLUSIVE
    markdown = to_markdown(report)
    assert SOURCE_CHECK_FAILED in markdown
    assert "not** evidence that the regime is current" in markdown


def test_markdown_contains_the_eight_required_sections(index):
    markdown = to_markdown(
        _run([fx.snapshot("a" * 64)], [fx.snapshot("b" * 64)],
             lambda r, c: (fx.drop_code(r, "RREL73"), c), index,
             "2026-01-01T00:00:00Z"))
    for heading in (
            "## 1. Baseline source / version",
            "## 2. Candidate source / version",
            "## 3. Did the authoritative source content change?",
            "## 4. Did the parsed Annex 2 specification change?",
            "## 5. Regulatory deltas",
            "## 6. Likely Trakt implementation impact",
            "## 7. Unresolved parser / review items",
            "## 8. Evidence and provenance"):
        assert heading in markdown, heading
    assert "observational" in markdown


def test_write_reports_emits_both_files(tmp_path, index):
    report = _run([fx.snapshot("a" * 64)], [fx.snapshot("b" * 64)],
                  lambda r, c: (fx.drop_code(r, "RREL73"), c), index,
                  "2026-01-01T00:00:00Z")
    paths = write_reports(report, tmp_path)
    assert paths["json"].exists() and paths["markdown"].exists()
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["regime"] == \
        "ESMA_Annex2"
    assert paths["markdown"].read_text(encoding="utf-8").startswith("# ESMA")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_snapshot_then_compare(tmp_path):
    from regulatory_watch.cli import main

    base_dir = tmp_path / "baseline"
    assert main(["--retrieved-at", "2026-01-01T00:00:00Z",
                 "snapshot", "--out-dir", str(base_dir),
                 "--spec-version", "vendored-1.3.1"]) == 0
    spec_path = base_dir / "annex2_spec.json"
    assert spec_path.exists()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["regime"] == "ESMA_Annex2"
    assert len(spec["fields"]) == 104

    run_dir = tmp_path / "run"
    assert main(["--retrieved-at", "2026-01-01T00:00:00Z",
                 "compare", "--out-dir", str(run_dir),
                 "--baseline-spec", str(spec_path),
                 "--candidate-spec", str(spec_path),
                 "--generated-at", "2026-01-01T00:00:00Z"]) == 0
    payload = json.loads(
        (run_dir / "esma_annex2_regulatory_watch.json").read_text("utf-8"))
    assert payload["regulatory_delta_count"] == 0
    # One allowlisted source is not vendored, so the run is explicitly
    # inconclusive rather than "no change".
    assert payload["source_status"] == SOURCE_CHECK_FAILED
    assert payload["outcome"] == WATCH_INCONCLUSIVE


def test_cli_refuses_a_corrupt_candidate_workbook(tmp_path):
    from regulatory_watch.cli import main
    corrupt = tmp_path / "corrupt.xlsx"
    corrupt.write_bytes(b"not a workbook")
    rc = main(["compare", "--out-dir", str(tmp_path / "out"),
               "--baseline-version", "x",
               "--candidate-workbook", str(corrupt),
               "--candidate-schema", str(_REPO / "DRAFT1auth.099.001.04_1.3.0.xsd")])
    assert rc == 2, "a corrupt source must be a failure, never 'no change'"


# --------------------------------------------------------------------------- #
# Stage 1 boundary: observational only
# --------------------------------------------------------------------------- #

WATCHED = (
    "config/system/fields_registry.yaml",
    "config/system/esma_code_order.yaml",
    "config/system/esma_model_structure.yaml",
    "config/system/enum_mapping.yaml",
    "config/regime/annex2_delivery_rules.yaml",
    "config/regime/annex2_field_universe.yaml",
    "config/delivery/annex2_field_xsd_path_map.yaml",
    "DRAFT1auth.099.001.04_1.3.0.xsd",
)


def test_a_full_watch_run_writes_nothing_outside_its_output_directory(tmp_path):
    from regulatory_watch.cli import main
    before = {p: hashlib.sha256((_REPO / p).read_bytes()).hexdigest()
              for p in WATCHED}
    base_dir = tmp_path / "baseline"
    main(["--retrieved-at", "2026-01-01T00:00:00Z", "snapshot",
          "--out-dir", str(base_dir)])
    main(["--retrieved-at", "2026-01-01T00:00:00Z", "compare",
          "--out-dir", str(tmp_path / "run"),
          "--baseline-spec", str(base_dir / "annex2_spec.json"),
          "--candidate-spec", str(base_dir / "annex2_spec.json"),
          "--generated-at", "2026-01-01T00:00:00Z"])
    after = {p: hashlib.sha256((_REPO / p).read_bytes()).hexdigest()
             for p in WATCHED}
    assert before == after


def test_the_watch_package_never_writes_to_config():
    """Static guard: no module in regulatory_watch/ writes under config/."""
    import re
    package = _REPO / "regulatory_watch"
    writers = re.compile(r"\.(write_text|write_bytes|open)\s*\(")
    for module in sorted(package.glob("*.py")):
        text = module.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if writers.search(line) and "config/" in line:
                raise AssertionError(f"{module.name}:{line_no}: {line.strip()}")
