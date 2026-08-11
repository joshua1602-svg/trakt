"""tests/test_regulatory_watch_comparator.py — synthetic mutation proof.

Stage 1 is only complete if it can prove change detection *now*, without
waiting for a real ESMA amendment. Each scenario below starts from the
committed slice of the REAL ESMA v1.3.1 workbook, applies exactly one
controlled mutation at the raw-source-row level, re-normalizes and asserts the
comparator reports **only** the intended delta.

Mutating raw rows rather than the normalized spec matters: it proves the
normalizer surfaces the change as well, so a scenario cannot pass against a
parser that is blind to the attribute.

Run: python -m pytest tests/test_regulatory_watch_comparator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch.comparator import (            # noqa: E402
    ComparatorError, compare, spec_status,
)
from regulatory_watch.contracts import (             # noqa: E402
    DETERMINISTIC,
    ENUM_CHANGED,
    FIELD_ADDED,
    FIELD_DESCRIPTION_CHANGED,
    FIELD_REMOVED,
    FORMAT_CHANGED,
    MANDATORY_STATUS_CHANGED,
    MULTIPLICITY_CHANGED,
    ND_PERMISSION_CHANGED,
    ORDER_CHANGED,
    SEV_HIGH,
    SEV_INFORMATIONAL,
    SOURCE_CHANGED,
    SOURCE_CHANGED_SPEC_UNCHANGED,
    SOURCE_UNCHANGED,
    SPEC_CHANGED,
    SPEC_UNCHANGED,
    VALIDATION_RULE_CHANGED,
    XML_PATH_CHANGED,
    overall_status,
    source_status_for,
)
from tests.helpers import regulatory_watch as fx     # noqa: E402


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def rows():
    return fx.load_rows()


@pytest.fixture(scope="module")
def codelists():
    return fx.load_codelists()


@pytest.fixture(scope="module")
def baseline(rows, codelists):
    return fx.build_spec(rows, codelists, spec_version="A")


def _only(deltas, code, change_type):
    """Assert exactly one delta, of the expected kind, on the expected code."""
    assert len(deltas) == 1, [d.to_dict() for d in deltas]
    delta = deltas[0]
    assert (delta.code, delta.change_type) == (code, change_type)
    return delta


# --------------------------------------------------------------------------- #
# 0. Control: identical inputs produce no delta, repeatably
# --------------------------------------------------------------------------- #

def test_identical_specs_produce_no_delta(baseline, rows, codelists):
    candidate = fx.build_spec(rows, codelists, spec_version="B")
    deltas = compare(baseline, candidate)
    assert deltas == []
    assert spec_status(deltas, baseline, candidate) == SPEC_UNCHANGED


def test_repeated_runs_are_identical(rows, codelists):
    first = compare(fx.build_spec(rows, codelists, spec_version="A"),
                    fx.build_spec(fx.drop_code(rows, "RREL73"), codelists,
                                  spec_version="B"))
    second = compare(fx.build_spec(rows, codelists, spec_version="A"),
                     fx.build_spec(fx.drop_code(rows, "RREL73"), codelists,
                                   spec_version="B"))
    assert [d.to_dict() for d in first] == [d.to_dict() for d in second]


def test_normalization_is_deterministic(rows, codelists):
    a = fx.build_spec(rows, codelists, spec_version="X").to_dict()
    b = fx.build_spec(list(reversed(list(rows))), codelists,
                      spec_version="X").to_dict()
    assert a == b, "row iteration order must not affect the normalized spec"


# --------------------------------------------------------------------------- #
# 1. FIELD_ADDED
# --------------------------------------------------------------------------- #

def test_field_added(baseline, rows, codelists):
    mutated = fx.add_code(rows, "RREL24", "RREL999",
                          label="Synthetic Added Field", tag="SynthDt")
    candidate = fx.build_spec(mutated, codelists, spec_version="B")
    delta = _only(compare(baseline, candidate), "RREL999", FIELD_ADDED)
    assert delta.old_value is None
    assert delta.new_value["label"] == "Synthetic Added Field"
    assert delta.severity == SEV_HIGH
    assert delta.confidence == DETERMINISTIC
    assert delta.evidence, "an added field must carry source provenance"


# --------------------------------------------------------------------------- #
# 2. FIELD_REMOVED
# --------------------------------------------------------------------------- #

def test_field_removed(baseline, rows, codelists):
    candidate = fx.build_spec(fx.drop_code(rows, "RREL73"), codelists,
                              spec_version="B")
    delta = _only(compare(baseline, candidate), "RREL73", FIELD_REMOVED)
    assert delta.new_value is None
    assert delta.old_value["label"] == "Allocated Losses"
    assert delta.severity == SEV_HIGH


def test_removing_one_code_from_a_shared_cell_keeps_the_other(baseline, rows,
                                                              codelists):
    # RREL1 and RREC1 share one workbook cell. Dropping one must not drop both.
    candidate = fx.build_spec(fx.drop_code(rows, "RREC1"), codelists,
                              spec_version="B")
    _only(compare(baseline, candidate), "RREC1", FIELD_REMOVED)
    assert "RREL1" in candidate.fields


# --------------------------------------------------------------------------- #
# 3. ND permission changed
# --------------------------------------------------------------------------- #

def test_nd_permission_withdrawn(baseline, rows, codelists):
    candidate = fx.build_spec(fx.remove_nd_branch(rows, "RREL24"), codelists,
                              spec_version="B")
    delta = _only(compare(baseline, candidate), "RREL24",
                  ND_PERMISSION_CHANGED)
    assert delta.old_value["nd_allowed"] == ["ND5"]
    assert delta.new_value["nd_allowed"] == []
    assert delta.confidence == DETERMINISTIC


def test_nd_permission_widened_via_justification_vocabulary(baseline, rows,
                                                            codelists):
    # The authority widening NoDataAllowedJustification3Code from {ND5} is a
    # regulatory ND change even though no workbook row moved.
    mutated = fx.mutate_codelist(codelists, "NoDataAllowedJustification3Code",
                                 add=["ND1"])
    candidate = fx.build_spec(rows, mutated, spec_version="B")
    deltas = compare(baseline, candidate)
    assert deltas, "widening an ND vocabulary must be detected"
    assert {d.change_type for d in deltas} == {ND_PERMISSION_CHANGED}
    affected = {d.code for d in deltas}
    assert "RREL24" in affected
    for delta in deltas:
        assert "ND1" in delta.new_value["nd_allowed"]


# --------------------------------------------------------------------------- #
# 4. FORMAT_CHANGED
# --------------------------------------------------------------------------- #

def test_format_changed(baseline, rows, codelists):
    leaf = fx.value_leaf_row(rows, "RREL2")
    mutated = fx.edit_rows(rows, "RREL2", lambda r: r.path == leaf.path,
                           pattern="xs:string|maxLength: 64|minLength: 1|")
    candidate = fx.build_spec(mutated, codelists, spec_version="B")
    delta = _only(compare(baseline, candidate), "RREL2", FORMAT_CHANGED)
    assert "maxLength: 64" in delta.new_value["format_pattern"]
    assert "format_pattern" in delta.new_value


def test_data_type_change_is_a_format_change(baseline, rows, codelists):
    leaf = fx.value_leaf_row(rows, "RREL24")
    mutated = fx.edit_rows(rows, "RREL24", lambda r: r.path == leaf.path,
                           data_type="ISODateTime")
    candidate = fx.build_spec(mutated, codelists, spec_version="B")
    delta = _only(compare(baseline, candidate), "RREL24", FORMAT_CHANGED)
    assert delta.old_value["data_type"] == "ISODate"
    assert delta.new_value["data_type"] == "ISODateTime"


# --------------------------------------------------------------------------- #
# 5. ENUM_CHANGED
# --------------------------------------------------------------------------- #

def test_enum_value_added(baseline, rows, codelists):
    mutated = fx.mutate_codelist(codelists, "PropertyType1Code", add=["HOUS"])
    candidate = fx.build_spec(rows, mutated, spec_version="B")
    delta = _only(compare(baseline, candidate), "RREC9", ENUM_CHANGED)
    assert "HOUS" in delta.new_value["enum_values"]
    assert "HOUS" not in delta.old_value["enum_values"]


def test_enum_value_removed(baseline, rows, codelists):
    mutated = fx.mutate_codelist(codelists, "InterestRateType2Code",
                                 remove=["FXRL"])
    candidate = fx.build_spec(rows, mutated, spec_version="B")
    delta = _only(compare(baseline, candidate), "RREL42", ENUM_CHANGED)
    assert "FXRL" in delta.old_value["enum_values"]
    assert "FXRL" not in delta.new_value["enum_values"]


# --------------------------------------------------------------------------- #
# 6. MULTIPLICITY_CHANGED
# --------------------------------------------------------------------------- #

def test_multiplicity_changed_without_touching_mandatory(baseline, rows,
                                                         codelists):
    element = fx.element_row(rows, "RREC13")
    mutated = fx.edit_rows(rows, "RREC13", lambda r: r.path == element.path,
                           multiplicity="[1..n]")
    candidate = fx.build_spec(mutated, codelists, spec_version="B")
    delta = _only(compare(baseline, candidate), "RREC13",
                  MULTIPLICITY_CHANGED)
    assert delta.old_value["multiplicity"] == "[1..1]"
    assert delta.new_value["multiplicity"] == "[1..n]"


def test_mandatory_status_changed(baseline, rows, codelists):
    element = fx.element_row(rows, "RREL36")
    mutated = fx.edit_rows(rows, "RREL36", lambda r: r.path == element.path,
                           multiplicity="[0..1]")
    candidate = fx.build_spec(mutated, codelists, spec_version="B")
    deltas = compare(baseline, candidate)
    kinds = {(d.code, d.change_type) for d in deltas}
    assert kinds == {("RREL36", MANDATORY_STATUS_CHANGED),
                     ("RREL36", MULTIPLICITY_CHANGED)}, kinds
    mandatory = next(d for d in deltas
                     if d.change_type == MANDATORY_STATUS_CHANGED)
    assert mandatory.old_value["mandatory"] == "mandatory"
    assert mandatory.new_value["mandatory"] == "optional"
    assert mandatory.severity == SEV_HIGH


# --------------------------------------------------------------------------- #
# 7. XML_PATH_CHANGED
# --------------------------------------------------------------------------- #

def test_xml_path_changed(baseline, rows, codelists):
    old = fx.element_row(rows, "RREL42").path
    new = old.rsplit("/", 1)[0] + "/IntrstRateTp"
    mutated = fx.move_code_path(rows, "RREL42", old, new)
    candidate = fx.build_spec(mutated, codelists, spec_version="B")
    delta = _only(compare(baseline, candidate), "RREL42", XML_PATH_CHANGED)
    assert delta.old_value["xml_path"] == old
    assert delta.new_value["xml_path"] == new
    assert delta.new_value["xml_tag"] == "IntrstRateTp"
    assert delta.severity == SEV_HIGH


# --------------------------------------------------------------------------- #
# 8. ORDER_CHANGED
# --------------------------------------------------------------------------- #

def test_order_changed_flags_only_the_moved_code(baseline, rows, codelists):
    mutated = fx.reorder_code(rows, "RREL69", new_row_number=1)
    candidate = fx.build_spec(mutated, codelists, spec_version="B")
    deltas = compare(baseline, candidate)
    order_deltas = [d for d in deltas if d.change_type == ORDER_CHANGED]
    assert [d.code for d in order_deltas] == ["RREL69"], \
        "only the moved code may be reported as re-ordered"
    assert all(d.change_type == ORDER_CHANGED for d in deltas)
    assert order_deltas[0].old_value["position"] != \
        order_deltas[0].new_value["position"]


def test_adding_a_field_does_not_report_order_changes(baseline, rows,
                                                      codelists):
    mutated = fx.add_code(rows, "RREL24", "RREL998", label="Synthetic",
                          tag="SynthTwo")
    candidate = fx.build_spec(mutated, codelists, spec_version="B")
    deltas = compare(baseline, candidate)
    assert [d.change_type for d in deltas] == [FIELD_ADDED], \
        "an insertion must not shift every following code into ORDER_CHANGED"


# --------------------------------------------------------------------------- #
# 9. Wording-only change
# --------------------------------------------------------------------------- #

def test_description_only_change_is_informational(baseline, rows, codelists):
    element = fx.element_row(rows, "RREL6")
    mutated = fx.edit_rows(
        rows, "RREL6", lambda r: r.path == element.path,
        definition="The data cut-off date for this data submission "
                   "(clarified wording).")
    candidate = fx.build_spec(mutated, codelists, spec_version="B")
    delta = _only(compare(baseline, candidate), "RREL6",
                  FIELD_DESCRIPTION_CHANGED)
    assert delta.severity == SEV_INFORMATIONAL
    assert "clarified wording" in delta.new_value["description"]


def test_label_change_is_a_description_change(baseline, rows, codelists):
    mutated = fx.edit_rows(rows, "RREL67", lambda r: bool(r.rts_field_name),
                           rts_field_name="Arrears Balance (amount)")
    candidate = fx.build_spec(mutated, codelists, spec_version="B")
    delta = _only(compare(baseline, candidate), "RREL67",
                  FIELD_DESCRIPTION_CHANGED)
    assert delta.new_value["label"] == "Arrears Balance (amount)"


# --------------------------------------------------------------------------- #
# 10. Source bytes changed, parsed specification unchanged
# --------------------------------------------------------------------------- #

def test_source_hash_change_alone_is_not_a_regulatory_change(rows, codelists):
    baseline = fx.build_spec(rows, codelists, spec_version="A",
                             snapshots=[fx.snapshot("a" * 64)])
    candidate = fx.build_spec(rows, codelists, spec_version="B",
                              snapshots=[fx.snapshot("b" * 64)])
    deltas = compare(baseline, candidate)
    assert deltas == []
    src = source_status_for(baseline.snapshots, candidate.snapshots)
    spec = spec_status(deltas, baseline, candidate)
    assert src == SOURCE_CHANGED
    assert spec == SPEC_UNCHANGED
    assert overall_status(src, spec) == SOURCE_CHANGED_SPEC_UNCHANGED


def test_identical_bytes_report_source_unchanged(rows, codelists):
    baseline = fx.build_spec(rows, codelists, spec_version="A",
                             snapshots=[fx.snapshot("c" * 64)])
    candidate = fx.build_spec(rows, codelists, spec_version="B",
                              snapshots=[fx.snapshot("c" * 64)])
    assert source_status_for(baseline.snapshots,
                             candidate.snapshots) == SOURCE_UNCHANGED


# --------------------------------------------------------------------------- #
# 11. VALIDATION_RULE_CHANGED
# --------------------------------------------------------------------------- #

def test_validation_rule_changed(baseline, rows, codelists):
    element = fx.element_row(rows, "RREL12")
    mutated = fx.edit_rows(rows, "RREL12", lambda r: r.path == element.path,
                           validation_rule="1) Must be a valid NUTS3 code")
    candidate = fx.build_spec(mutated, codelists, spec_version="B")
    delta = _only(compare(baseline, candidate), "RREL12",
                  VALIDATION_RULE_CHANGED)
    assert "NUTS3" in " ".join(delta.new_value["validation_rules"])


def test_validation_rule_deleted(baseline, rows, codelists):
    # ESMA v1.3.1 amendment 79 deleted validation rules on RREL24/36/73. This
    # is the shape of that change: rule text present, then gone.
    with_rule = fx.edit_rows(rows, "RREL73",
                             lambda r: r.path == fx.element_row(rows,
                                                                "RREL73").path,
                             validation_rule="Must be greater than or equal to 0")
    old = fx.build_spec(with_rule, codelists, spec_version="A")
    new = fx.build_spec(rows, codelists, spec_version="B")
    delta = _only(compare(old, new), "RREL73", VALIDATION_RULE_CHANGED)
    assert delta.old_value["validation_rules"] == \
        ["Must be greater than or equal to 0"]
    assert delta.new_value["validation_rules"] == []


# --------------------------------------------------------------------------- #
# Comparator refusals
# --------------------------------------------------------------------------- #

def test_comparator_refuses_across_parser_versions(baseline, rows, codelists):
    candidate = fx.build_spec(rows, codelists, spec_version="B")
    candidate.parser_version = "annex2-normalizer/9.9.9"
    with pytest.raises(ComparatorError):
        compare(baseline, candidate)


def test_comparator_refuses_across_regimes(baseline, rows, codelists):
    candidate = fx.build_spec(rows, codelists, spec_version="B")
    candidate.regime = "ESMA_Annex12"
    with pytest.raises(ComparatorError):
        compare(baseline, candidate)


def test_every_change_type_has_a_severity():
    from regulatory_watch.contracts import CHANGE_TYPES, SEVERITY_BY_CHANGE_TYPE
    assert set(CHANGE_TYPES) == set(SEVERITY_BY_CHANGE_TYPE)


def test_spec_change_dominates_a_failed_source_check(rows, codelists):
    """A confirmed change is reported even when another source is unchecked."""
    from regulatory_watch.contracts import (REGULATORY_SPEC_CHANGED,
                                            SOURCE_CHECK_FAILED)
    assert overall_status(SOURCE_CHECK_FAILED, SPEC_CHANGED) == \
        REGULATORY_SPEC_CHANGED
