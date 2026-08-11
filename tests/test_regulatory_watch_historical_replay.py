"""tests/test_regulatory_watch_historical_replay.py — replay against real ESMA history.

The repository vendors ONE version of the ESMA Annex 2 artefacts (workbook
v1.3.1 / schema 1.3.0). It does **not** contain a second, earlier version, so a
true "parse version N, parse version N+1, diff them" replay cannot be run here
and no such difference is invented.

What the repository DOES contain is ESMA's own published amendment history,
shipped inside the workbook as the ``Change log - Summary`` and
``Change log - XML path`` sheets. That is authority-published evidence of what
actually changed between Annex 2 source versions, and it is used here for three
things:

1. the change log parses deterministically;
2. every amendment type ESMA has actually used maps onto a change type the
   comparator can express (vocabulary coverage);
3. one amendment (v1.3.0 #78, an element addition whose new XML paths ESMA
   publishes in full) is reverse-applied to build a genuine pre-1.3.0 baseline,
   and the comparator is run against it — without the comparator being aware
   that the baseline was reconstructed.

Amendments whose pre-change value ESMA does not publish are recorded as
non-reconstructible rather than guessed.

The harness is shaped so a real historical snapshot can be dropped in without
touching the comparator: any normalized-spec JSON can be passed as the
baseline.

Run: python -m pytest tests/test_regulatory_watch_historical_replay.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch.annex2_spec import parse_annex2_spec       # noqa: E402
from regulatory_watch.changelog import (                         # noqa: E402
    ESMA_AMENDMENT_TYPE_MAP, extract_changelog, reconstruct_prior_spec,
    vocabulary_coverage,
)
from regulatory_watch.comparator import compare, spec_status     # noqa: E402
from regulatory_watch.contracts import (                         # noqa: E402
    CHANGE_TYPES, FIELD_ADDED, NormalizedSpec, SPEC_UNCHANGED,
)
from tests.helpers import regulatory_watch as fx                 # noqa: E402

_ARTEFACTS_PRESENT = fx.REAL_WORKBOOK.exists() and fx.REAL_SCHEMA.exists()
requires_artefacts = pytest.mark.skipif(
    not _ARTEFACTS_PRESENT,
    reason="vendored ESMA workbook/schema not present in this checkout")


@pytest.fixture(scope="module")
def entries():
    if not _ARTEFACTS_PRESENT:
        pytest.skip("vendored ESMA artefacts not present")
    return extract_changelog(fx.REAL_WORKBOOK)


@pytest.fixture(scope="module")
def real_spec():
    if not _ARTEFACTS_PRESENT:
        pytest.skip("vendored ESMA artefacts not present")
    return parse_annex2_spec(fx.REAL_WORKBOOK, fx.REAL_SCHEMA,
                             spec_version="auth.099.001.04-wb1.3.1")


# --------------------------------------------------------------------------- #
# 1. The published change log parses deterministically
# --------------------------------------------------------------------------- #

@requires_artefacts
def test_changelog_parses(entries):
    assert entries, "the workbook ships a published change log"
    versions = {e.version for e in entries}
    assert {"1.1.0", "1.2.0", "1.3.0", "1.3.1"} <= versions


@requires_artefacts
def test_changelog_parse_is_deterministic():
    a = [e.to_dict() for e in extract_changelog(fx.REAL_WORKBOOK)]
    b = [e.to_dict() for e in extract_changelog(fx.REAL_WORKBOOK)]
    assert a == b


@requires_artefacts
def test_changelog_carries_dates_and_amendment_ids(entries):
    dated = [e for e in entries if e.amendment_date]
    assert dated
    assert all(len(e.amendment_date) == 10 for e in dated)
    assert any(e.amendment_id for e in entries)


@requires_artefacts
def test_changelog_identifies_amendments_touching_annex2_codes(entries):
    annex2 = [e for e in entries if e.touches_annex2]
    assert annex2, "ESMA history includes amendments naming RREL/RREC codes"
    codes = {c for e in annex2 for c in e.annex2_codes}
    # The v1.3.1 validation-rule deletions name these Annex 2 codes.
    assert {"RREL24", "RREL36", "RREL73"} <= codes


# --------------------------------------------------------------------------- #
# 2. Vocabulary coverage: can the comparator express real ESMA changes?
# --------------------------------------------------------------------------- #

@requires_artefacts
def test_every_real_esma_amendment_type_maps_to_a_comparator_change_type(
        entries):
    coverage = vocabulary_coverage(entries)
    assert coverage["entries"] > 0
    assert coverage["unmapped_amendment_types"] == {}, (
        "an amendment type ESMA has actually used cannot be expressed by the "
        "comparator: " + json.dumps(coverage["unmapped_amendment_types"]))


def test_the_amendment_type_map_only_targets_published_change_types():
    for types in ESMA_AMENDMENT_TYPE_MAP.values():
        assert set(types) <= set(CHANGE_TYPES)


@requires_artefacts
def test_the_v1_3_1_validation_rule_deletion_is_expressible(entries):
    """ESMA v1.3.1 deleted validation rules on RREL24 / RREL36 / RREL73."""
    from regulatory_watch.contracts import VALIDATION_RULE_CHANGED
    v131 = [e for e in entries if e.version == "1.3.1" and e.touches_annex2]
    assert v131, "the 1.3.1 amendment naming Annex 2 codes must be parsed"
    entry = v131[0]
    assert {"RREL24", "RREL36", "RREL73"} <= set(entry.annex2_codes)
    assert VALIDATION_RULE_CHANGED in entry.mapped_change_types


@requires_artefacts
def test_the_v1_3_0_element_addition_is_expressible(entries):
    addition = [e for e in entries
                if e.version == "1.3.0"
                and "Gross Annual Rental" in e.description]
    assert addition, "the v1.3.0 element addition must be parsed"
    assert FIELD_ADDED in addition[0].mapped_change_types
    assert addition[0].xml_paths, "ESMA publishes the new paths in full"


# --------------------------------------------------------------------------- #
# 3. Reconstructed historical baseline -> comparator
# --------------------------------------------------------------------------- #

@requires_artefacts
def test_reconstruction_reports_what_it_cannot_reverse(entries, real_spec):
    result = reconstruct_prior_spec(real_spec, entries, target_version="1.3.0",
                                    prior_version_label="pre-1.3.0")
    assert result.not_reconstructible, (
        "amendments whose pre-change value ESMA does not publish must be "
        "reported, not silently skipped")
    for item in result.not_reconstructible:
        assert item["reason"]
        assert item["xsd_amendment_types"]


@requires_artefacts
def test_the_element_addition_is_reverse_applied(entries, real_spec):
    result = reconstruct_prior_spec(real_spec, entries, target_version="1.3.0",
                                    prior_version_label="pre-1.3.0")
    applied = [a for a in result.applied
               if "Gross Annual Rental" in a["description"]]
    assert applied, "the element addition must be reverse-applied"
    assert applied[0]["added_paths"], "with the ESMA-published new paths"


@requires_artefacts
def test_replay_v1_3_0_addition_lands_outside_the_annex2_performing_scope(
        entries, real_spec):
    """The real v1.3.0 addition is a NON-performing residential element.

    ESMA added ``GrssAnlRntlIncm`` under ``ResdtlRealEsttLn/NonPrfrmgLn`` and
    published it with no RTS field code. Trakt's Annex 2 pathway is the
    performing branch, so the correct answer for this real amendment is:
    the authoritative source changed, and the Annex 2 specification Trakt
    projects did not. This is the third outcome Stage 1 must be able to reach,
    proven against a real regulatory change rather than a synthetic one.
    """
    result = reconstruct_prior_spec(real_spec, entries, target_version="1.3.0",
                                    prior_version_label="pre-1.3.0")
    addition = next(a for a in result.applied
                    if "Gross Annual Rental" in a["description"])
    assert addition["codes_removed_from_prior_baseline"] == []
    assert addition["in_annex2_scope"] is False
    for path in addition["added_paths"]:
        assert "NonPrfrmgLn" in path

    prior = result.spec
    deltas = compare(prior, real_spec)
    assert deltas == [], [d.to_dict() for d in deltas]
    assert spec_status(deltas, prior, real_spec) == SPEC_UNCHANGED


@requires_artefacts
def test_reconstruction_leaves_the_parsed_spec_untouched(entries, real_spec):
    before = real_spec.to_dict()
    reconstruct_prior_spec(real_spec, entries, target_version="1.3.0",
                           prior_version_label="pre-1.3.0")
    assert real_spec.to_dict() == before


@requires_artefacts
def test_reconstruction_is_deterministic(entries, real_spec):
    a = reconstruct_prior_spec(real_spec, entries, "1.3.0", "pre-1.3.0")
    b = reconstruct_prior_spec(real_spec, entries, "1.3.0", "pre-1.3.0")
    assert a.to_dict() == b.to_dict()
    assert a.spec.to_dict() == b.spec.to_dict()


# --------------------------------------------------------------------------- #
# 4. The harness accepts an inserted historical snapshot unchanged
# --------------------------------------------------------------------------- #

def test_a_historical_spec_can_be_inserted_without_touching_the_comparator(
        tmp_path):
    """Drop-in path for a real version-N snapshot when one becomes available.

    A normalized spec is plain JSON. Writing one to disk and loading it as the
    baseline requires no comparator change — which is what makes a future
    genuine historical replay a data task, not a code task.
    """
    rows = fx.load_rows()
    codelists = fx.load_codelists()
    historical = fx.build_spec(fx.drop_code(rows, "RREL73"), codelists,
                               spec_version="historical-vN")
    path = tmp_path / "annex2_spec_vN.json"
    path.write_text(json.dumps(historical.to_dict(), indent=2, sort_keys=True),
                    encoding="utf-8")

    loaded = NormalizedSpec.from_dict(
        json.loads(path.read_text(encoding="utf-8")))
    assert loaded.to_dict() == historical.to_dict()

    current = fx.build_spec(rows, codelists, spec_version="current-vN+1")
    deltas = compare(loaded, current)
    assert [(d.code, d.change_type) for d in deltas] == [("RREL73", FIELD_ADDED)]
    assert deltas[0].source_version_a == "historical-vN"
    assert deltas[0].source_version_b == "current-vN+1"


def test_replaying_the_same_pair_twice_is_identical(tmp_path):
    rows = fx.load_rows()
    codelists = fx.load_codelists()
    a = fx.build_spec(fx.drop_code(rows, "RREL73"), codelists,
                      spec_version="vN")
    b = fx.build_spec(rows, codelists, spec_version="vN+1")
    first = [d.to_dict() for d in compare(a, b)]
    second = [d.to_dict() for d in compare(a, b)]
    assert first == second
