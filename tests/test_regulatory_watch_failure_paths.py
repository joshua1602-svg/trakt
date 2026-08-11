"""tests/test_regulatory_watch_failure_paths.py — the watch must fail closed.

Every scenario here is a way the watch could go wrong in production. In none of
them may the system conclude that the active Annex 2 regime is current: it must
either fail loudly (``SpecParseError`` / ``ManifestError``), surface
``SOURCE_CHECK_FAILED``, or downgrade the comparison to
``SPEC_UNDETERMINED`` / ``review-required``.

Covered: source unavailable, corrupted workbook, unexpected source structure,
duplicate field code, parser ambiguity, unsupported regulatory structure,
unreadable manifest, out-of-scope authority, and retrieval failure.

Run: python -m pytest tests/test_regulatory_watch_failure_paths.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch import PARSER_VERSION                    # noqa: E402
from regulatory_watch.annex2_spec import (                     # noqa: E402
    SpecParseError, extract_codelists, extract_rows,
)
from regulatory_watch.comparator import compare, spec_status   # noqa: E402
from regulatory_watch.contracts import (                       # noqa: E402
    NO_CHANGE_DETECTED,
    REVIEW_REQUIRED,
    SOURCE_CHECK_FAILED,
    SOURCE_MISSING,
    SPEC_UNCHANGED,
    SPEC_UNDETERMINED,
    SourceArtefact,
    WATCH_INCONCLUSIVE,
    overall_status,
    source_status_for,
)
from regulatory_watch.manifest import (                        # noqa: E402
    ManifestError, load_manifest,
)
from regulatory_watch.retrieval import retrieve                # noqa: E402
from regulatory_watch.snapshots import SnapshotStore           # noqa: E402
from tests.helpers import regulatory_watch as fx               # noqa: E402


@pytest.fixture(scope="module")
def rows():
    return fx.load_rows()


@pytest.fixture(scope="module")
def codelists():
    return fx.load_codelists()


# --------------------------------------------------------------------------- #
# 1. Source unavailable
# --------------------------------------------------------------------------- #

def test_missing_local_artefact_is_source_check_failed(tmp_path):
    store = SnapshotStore(root=tmp_path / "snap", regime="ESMA_Annex2")
    artefact = SourceArtefact(artefact_id="missing", authority="ESMA",
                              regime="ESMA_Annex2",
                              artefact_type="xml_schema",
                              local_path="does/not/exist.bin")
    snap = store.snapshot_local(artefact, tmp_path / "does/not/exist.bin",
                                "2026-01-01T00:00:00Z", PARSER_VERSION)
    assert snap.retrieval_status == SOURCE_CHECK_FAILED
    assert not snap.ok
    assert snap.sha256 == ""


def test_a_failed_source_check_can_never_report_no_change(rows, codelists):
    baseline = fx.build_spec(rows, codelists, spec_version="A",
                             snapshots=[fx.snapshot("a" * 64)])
    candidate = fx.build_spec(rows, codelists, spec_version="B",
                              snapshots=[fx.failed_snapshot()])
    deltas = compare(baseline, candidate)
    src = source_status_for(baseline.snapshots, candidate.snapshots)
    spec = spec_status(deltas, baseline, candidate)
    assert deltas == []
    assert src == SOURCE_CHECK_FAILED
    assert spec == SPEC_UNCHANGED           # the parsed slice really is equal
    outcome = overall_status(src, spec)
    assert outcome == WATCH_INCONCLUSIVE
    assert outcome != NO_CHANGE_DETECTED


def test_a_missing_counterpart_artefact_is_not_unchanged(rows, codelists):
    a = [fx.snapshot("a" * 64, "esma_annex2_message_workbook"),
         fx.snapshot("b" * 64, "esma_annex2_xml_schema")]
    b = [fx.snapshot("a" * 64, "esma_annex2_message_workbook")]
    assert source_status_for(a, b) == SOURCE_MISSING
    assert overall_status(source_status_for(a, b),
                          SPEC_UNCHANGED) == WATCH_INCONCLUSIVE


def test_retrieval_without_a_fetcher_fails_closed(tmp_path):
    manifest = load_manifest()
    store = SnapshotStore(root=tmp_path, regime=manifest.regime)
    snap = retrieve(manifest.artefacts[0], manifest, store,
                    "2026-01-01T00:00:00Z", PARSER_VERSION, fetcher=None)
    assert snap.retrieval_status == SOURCE_CHECK_FAILED
    assert "retrieval disabled" in snap.retrieval_detail


def test_retrieval_refuses_a_url_outside_the_allowlist(tmp_path):
    manifest = dataclasses.replace(load_manifest(), retrieval_enabled=True)
    store = SnapshotStore(root=tmp_path, regime=manifest.regime)
    rogue = dataclasses.replace(manifest.artefacts[0],
                                source_url="https://example.invalid/annex2.xlsx")

    def fetcher(url):                       # pragma: no cover - must not run
        raise AssertionError(f"fetcher must not be called for {url}")

    snap = retrieve(rogue, manifest, store, "2026-01-01T00:00:00Z",
                    PARSER_VERSION, fetcher=fetcher)
    assert snap.retrieval_status == SOURCE_CHECK_FAILED
    assert "not in allowlist" in snap.retrieval_detail


def test_a_raising_fetcher_is_source_check_failed(tmp_path):
    manifest = dataclasses.replace(load_manifest(), retrieval_enabled=True)
    store = SnapshotStore(root=tmp_path, regime=manifest.regime)

    def fetcher(url):
        raise TimeoutError("connection timed out")

    snap = retrieve(manifest.artefacts[0], manifest, store,
                    "2026-01-01T00:00:00Z", PARSER_VERSION, fetcher=fetcher)
    assert snap.retrieval_status == SOURCE_CHECK_FAILED
    assert "TimeoutError" in snap.retrieval_detail


# --------------------------------------------------------------------------- #
# 2. Corrupted source
# --------------------------------------------------------------------------- #

def test_corrupted_workbook_raises(tmp_path):
    corrupt = tmp_path / "corrupt.xlsx"
    corrupt.write_bytes(b"PK\x03\x04 not really a workbook at all")
    with pytest.raises(SpecParseError) as exc:
        extract_rows(corrupt)
    assert "could not be opened" in str(exc.value)


def test_corrupted_schema_raises(tmp_path):
    corrupt = tmp_path / "corrupt.xsd"
    corrupt.write_text("<xs:schema><unclosed>", encoding="utf-8")
    with pytest.raises(SpecParseError):
        extract_codelists(corrupt)


def test_absent_workbook_raises(tmp_path):
    with pytest.raises(SpecParseError) as exc:
        extract_rows(tmp_path / "nope.xlsx")
    assert "not found" in str(exc.value)


# --------------------------------------------------------------------------- #
# 3. Unexpected source structure
# --------------------------------------------------------------------------- #

def _write_workbook(path: Path, header, data_rows):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "DRAFT1auth.099.001.04"
    ws.append([]), ws.append([]), ws.append([])
    ws.append(list(header))
    for row in data_rows:
        ws.append(list(row))
    wb.save(path)


def test_missing_required_column_raises(tmp_path):
    path = tmp_path / "reshaped.xlsx"
    _write_workbook(path, ["RTS Field code", "XML TAG", "PATH"],
                    [["RREL1", "<ScrtstnIdr>", "/Document/ScrtstnRpt/ScrtstnIdr"]])
    with pytest.raises(SpecParseError) as exc:
        extract_rows(path)
    assert "missing required columns" in str(exc.value)


def test_missing_sheet_raises(tmp_path):
    import openpyxl
    path = tmp_path / "wrongsheet.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "Something Else"
    wb.save(path)
    with pytest.raises(SpecParseError) as exc:
        extract_rows(path)
    assert "no sheet" in str(exc.value)


def test_no_annex2_rows_in_scope_raises(tmp_path):
    header = ["Field code", "Item type", "Template",
              "Performing/Non Performing", "RTS Field code", "RTS Field name",
              "Validation rule", "Pattern", "XML TAG", "MULTIPLICITY",
              "DATA TYPE / CODE", "MESSAGE ITEM DEFINITION", "PATH"]
    path = tmp_path / "otherannex.xlsx"
    _write_workbook(path, header, [
        ["1", "E", "CRE", "PRF/NPRF", "CREL1", "Unique Identifier", "", "",
         "<ScrtstnIdr>", "[1..1]", "Max35Text", "definition",
         "/Document/ScrtstnRpt/ScrtstnIdr"],
    ])
    with pytest.raises(SpecParseError) as exc:
        extract_rows(path)
    assert "no Annex 2 rows" in str(exc.value)


# --------------------------------------------------------------------------- #
# 4. Duplicate field code / parser ambiguity
# --------------------------------------------------------------------------- #

def test_duplicate_code_on_disjoint_paths_is_ambiguous_not_guessed(
        rows, codelists):
    """The same code published under two unrelated trees must not be resolved."""
    element = fx.element_row(rows, "RREL24")
    duplicate = dataclasses.replace(
        element,
        row_number=max(r.row_number for r in rows) + 1,
        path="/Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/"
             "ScrtstnRpt/DplctBranch/MtrtyDt")
    spec = fx.build_spec(list(rows) + [duplicate], codelists,
                         spec_version="dup")
    field = spec.fields["RREL24"]
    assert field.xml_path == "UNKNOWN"
    assert {"xml_path", "xml_tag", "mandatory",
            "multiplicity"} <= set(field.unresolved)
    warnings = {w["warning"] for w in spec.parse_warnings if w["code"] == "RREL24"}
    assert "AMBIGUOUS_ELEMENT_NODE" in warnings


def test_ambiguity_downgrades_the_delta_and_blocks_spec_unchanged(rows,
                                                                  codelists):
    baseline = fx.build_spec(rows, codelists, spec_version="A")
    element = fx.element_row(rows, "RREL24")
    duplicate = dataclasses.replace(
        element,
        row_number=max(r.row_number for r in rows) + 1,
        path="/Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/"
             "ScrtstnRpt/DplctBranch/MtrtyDt")
    candidate = fx.build_spec(list(rows) + [duplicate], codelists,
                              spec_version="B")
    deltas = compare(baseline, candidate)
    path_deltas = [d for d in deltas if d.code == "RREL24"]
    assert path_deltas, "an ambiguous code must still produce a delta"
    assert all(d.confidence == REVIEW_REQUIRED for d in path_deltas)
    assert spec_status(deltas, baseline, candidate) == SPEC_UNDETERMINED


def test_unresolved_attribute_blocks_a_clean_unchanged_verdict(rows,
                                                               codelists):
    """No delta + an unresolved attribute must NOT read as SPEC_UNCHANGED."""
    element = fx.element_row(rows, "RREL24")
    mutated = fx.edit_rows(rows, "RREL24", lambda r: r.path == element.path,
                           multiplicity="one or more")
    a = fx.build_spec(mutated, codelists, spec_version="A")
    b = fx.build_spec(mutated, codelists, spec_version="B")
    assert compare(a, b) == []
    assert "mandatory" in a.fields["RREL24"].unresolved
    assert spec_status([], a, b) == SPEC_UNDETERMINED


# --------------------------------------------------------------------------- #
# 5. Unsupported new regulatory structure
# --------------------------------------------------------------------------- #

def test_unknown_nd_justification_vocabulary_is_not_read_as_no_nd(rows,
                                                                  codelists):
    """A new ND vocabulary the schema does not define must not become "[]"."""
    nd_rows = fx.nd_leaf_rows(rows, "RREL24")
    assert nd_rows
    mutated = fx.edit_rows(
        rows, "RREL24", lambda r: r.path == nd_rows[0].path,
        data_type="NoDataAllowedJustification7Code")
    spec = fx.build_spec(mutated, codelists, spec_version="new-structure")
    field = spec.fields["RREL24"]
    assert field.nd_allowed is None, "unknown ND vocabulary must not be []"
    assert "nd_allowed" in field.unresolved
    assert any(w["warning"] == "UNKNOWN_ND_JUSTIFICATION_TYPE"
               for w in spec.parse_warnings)


def test_nd_branch_without_a_typed_leaf_is_unsupported_not_absent(rows,
                                                                  codelists):
    nd_rows = fx.nd_leaf_rows(rows, "RREL36")
    mutated = fx.edit_rows(rows, "RREL36",
                           lambda r: r.path == nd_rows[0].path, data_type="")
    spec = fx.build_spec(mutated, codelists, spec_version="new-structure")
    field = spec.fields["RREL36"]
    assert field.nd_allowed is None
    assert "nd_allowed" in field.unresolved
    assert any(w["warning"] == "UNSUPPORTED_ND_STRUCTURE"
               for w in spec.parse_warnings)


def test_declared_enumeration_with_no_schema_values_is_unresolved(rows,
                                                                  codelists):
    leaf = fx.value_leaf_row(rows, "RREC9")
    mutated = fx.edit_rows(rows, "RREC9", lambda r: r.path == leaf.path,
                           data_type="PropertyType9Code")
    spec = fx.build_spec(mutated, codelists, spec_version="new-structure")
    field = spec.fields["RREC9"]
    assert field.enum_values is None
    assert "enum_values" in field.unresolved
    assert any(w["warning"] == "UNRESOLVED_ENUMERATION"
               for w in spec.parse_warnings)


def test_a_field_with_no_typed_value_leaf_is_reported(rows, codelists):
    leaf = fx.value_leaf_row(rows, "RREL67")
    mutated = fx.edit_rows(rows, "RREL67",
                           lambda r: "NoDataOptn" not in r.path, data_type="")
    spec = fx.build_spec(mutated, codelists, spec_version="new-structure")
    field = spec.fields["RREL67"]
    assert field.data_type == "UNKNOWN"
    assert {"data_type", "format_pattern"} <= set(field.unresolved)
    assert leaf.data_type, "sanity: the unmutated fixture had a typed leaf"


# --------------------------------------------------------------------------- #
# 6. Manifest failures
# --------------------------------------------------------------------------- #

def test_absent_manifest_raises(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "nope.yaml")


def test_malformed_manifest_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("manifest: [this, is, not, a, mapping]\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_manifest_rejects_a_non_esma_authority(tmp_path):
    path = tmp_path / "fca.yaml"
    path.write_text(
        "manifest:\n  manifest_id: x\n  regime: ESMA_Annex2\n"
        "  authority: FCA\nartefacts:\n"
        "  - artefact_id: a\n    authority: FCA\n    regime: ESMA_Annex2\n"
        "    artefact_type: xml_schema\n", encoding="utf-8")
    with pytest.raises(ManifestError) as exc:
        load_manifest(path)
    assert "unsupported authority" in str(exc.value)


def test_manifest_rejects_a_non_https_url(tmp_path):
    path = tmp_path / "http.yaml"
    path.write_text(
        "manifest:\n  manifest_id: x\n  regime: ESMA_Annex2\n"
        "  authority: ESMA\nartefacts:\n"
        "  - artefact_id: a\n    authority: ESMA\n    regime: ESMA_Annex2\n"
        "    artefact_type: xml_schema\n"
        "    source_url: http://www.esma.europa.eu/x\n", encoding="utf-8")
    with pytest.raises(ManifestError) as exc:
        load_manifest(path)
    assert "https" in str(exc.value)


def test_manifest_rejects_duplicate_artefact_ids(tmp_path):
    path = tmp_path / "dupe.yaml"
    entry = ("  - artefact_id: a\n    authority: ESMA\n    regime: ESMA_Annex2\n"
             "    artefact_type: xml_schema\n")
    path.write_text(
        "manifest:\n  manifest_id: x\n  regime: ESMA_Annex2\n"
        "  authority: ESMA\nartefacts:\n" + entry + entry, encoding="utf-8")
    with pytest.raises(ManifestError) as exc:
        load_manifest(path)
    assert "duplicate artefact_id" in str(exc.value)


def test_manifest_rejects_an_unknown_artefact_type(tmp_path):
    path = tmp_path / "type.yaml"
    path.write_text(
        "manifest:\n  manifest_id: x\n  regime: ESMA_Annex2\n"
        "  authority: ESMA\nartefacts:\n"
        "  - artefact_id: a\n    authority: ESMA\n    regime: ESMA_Annex2\n"
        "    artefact_type: press_release\n", encoding="utf-8")
    with pytest.raises(ManifestError) as exc:
        load_manifest(path)
    assert "unsupported artefact_type" in str(exc.value)
