"""tests/test_p1_lens_narrowing_declared.py — a lens narrowing must be legible.

The defect this pins, in one sentence: an analytical plan that narrowed two
populations through the portfolio lens, measured both correctly, and then
refused the question on the ground that neither narrowing had been applied.

Nothing was wrong with the measurement. The two questions

    "Did Direct or Acquired add more balance during the last month?"
    "Which of the Direct and Acquired books drove more of the
     month-on-month balance increase?"

each produced £105.0m → £117.4m for Direct and £44.5m → £54.7m for Acquired,
matching the independent truth to the penny, and both were refused.

The narrowing was declared only as display text — ``"portfolio lens = Direct
(direct_001)"`` — which names a lens and never a field. The one channel every
receipt reader consults for "what did this plan narrow to" is built by splitting
field-named predicates apart, so the lens clause was skipped as unparseable and
the narrowing became invisible to everything downstream. STATED, NOT CARRIED.

Three properties, and the middle one is the one that would have caught it:

1. the primitive declares the lens narrowing field-named;
2. both producers of the lens predicate go THROUGH that primitive — there were
   two, the second under a comment asserting it matched the first, and that
   assertion held right up until the two had to say more than text;
3. the field-named declaration reaches both receipt readers that decide whether
   a narrowing or a breakdown was applied.

Run: python -m pytest tests/test_p1_lens_narrowing_declared.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import execution_receipt as receipt  # noqa: E402
from mi_agent import portfolio_lens as lens_mod  # noqa: E402
from mi_workflows.analytical import contract, populations as pops, route  # noqa: E402


class _Lens:
    """A resolved lens, as the governed registry hands one back."""

    def __init__(self, name, label, ids):
        self.name, self.label = name, label
        self.filters = {lens_mod.SOURCE_ID_FIELD: list(ids)}


class _Finding:
    def __init__(self, population):
        self.population = population


def _ref(term, label, ids, rows):
    """A population ref built the way `apply` and `period_movement` build one."""
    spec = pops.PopulationSpec(key=term, label=label,
                               kind=pops.KIND_PROVENANCE, lens_term=term)
    text, narrowed_on = pops.lens_narrowing(spec, _Lens(term, label, ids))
    return contract.PopulationRef(key=term, label=label, predicate=text,
                                  rows=rows, narrowed_on=narrowed_on)


# --------------------------------------------------------------------------- #
# 1. The primitive
# --------------------------------------------------------------------------- #
def test_the_lens_narrowing_is_declared_with_the_field_it_scoped():
    text, narrowed_on = pops.lens_narrowing(
        pops.PopulationSpec(key="direct", label="Direct",
                            kind=pops.KIND_PROVENANCE, lens_term="direct"),
        _Lens("direct", "Direct", ["direct_001"]))
    # The reader-facing text is unchanged by any of this.
    assert text == "portfolio lens = Direct (direct_001)"
    # And the same filter, said so a machine can read it.
    assert narrowed_on == ((lens_mod.SOURCE_TYPE_FIELD, "direct"),)


def test_a_population_with_no_lens_declares_no_lens_narrowing():
    spec = pops.PopulationSpec(key="total", label="the whole book",
                               kind=pops.KIND_TOTAL)
    assert pops.lens_narrowing(spec) == (None, ())


# --------------------------------------------------------------------------- #
# 2. One producer, not two
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", ["populations", "executors"])
def test_no_module_composes_the_lens_predicate_itself(module):
    """The text is composed in exactly one place: `lens_narrowing`.

    A second composition is not a style complaint. It is how this defect
    survived a fix: the field-named channel was added to one of them, the
    capability that actually runs these two questions used the other, and the
    answers stayed refused with the fix in the tree.
    """
    source = (_REPO_ROOT / "mi_workflows" / "analytical"
              / f"{module}.py").read_text()
    # The RESOLVED lens's label — the predicate form. `populations.apply`
    # separately records `portfolio lens = {spec.lens_term}` on the
    # UNAVAILABLE list, which is the opposite claim (a scope that was not
    # applied) and is deliberately not matched here.
    composed = re.findall(r'f"portfolio lens = \{lens\.label\}', source)
    expected = 1 if module == "populations" else 0
    assert len(composed) == expected, (
        f"{module}.py composes the lens predicate {len(composed)} times; "
        f"expected {expected} — call populations.lens_narrowing instead")


# --------------------------------------------------------------------------- #
# 3. It reaches the readers that decide
# --------------------------------------------------------------------------- #
def test_a_lens_narrowing_reaches_the_plans_narrowing_declaration():
    entries = route.narrowed_entries([_Finding(_ref("direct", "Direct",
                                                    ["direct_001"], 441))])
    assert entries == [{"field": lens_mod.SOURCE_TYPE_FIELD, "value": "direct",
                        "rows": 441, "dataset": "funded"}]


def test_a_narrowing_that_selected_no_rows_is_not_a_narrowing():
    assert route.narrowed_entries(
        [_Finding(_ref("direct", "Direct", ["direct_001"], 0))]) == []


def test_the_receipt_reads_a_single_lens_population_as_a_narrowing():
    """Q22C's reader: `lost_narrowing "Direct"` against the plan's declaration."""
    plan = {"narrowedTo": route.narrowed_entries(
        [_Finding(_ref("direct", "Direct", ["direct_001"], 441))])}
    assert receipt._analytical_narrowed_to(
        plan, receipt.RequestedFacet(kind=receipt.KIND_LOST_NARROWING,
                                     label="Direct"))
    # And still refuses a population the plan did not narrow to.
    assert not receipt._analytical_narrowed_to(
        plan, receipt.RequestedFacet(kind=receipt.KIND_LOST_NARROWING,
                                     label="Acquired"))


def test_the_receipt_reads_two_lens_populations_as_a_breakdown():
    """Q22B's reader: `grouping_dimension "direct or acquired"`.

    Two populations of one field ARE the breakdown. One is a filter, and the
    threshold is load-bearing — `_two_or_more_populations` owns that rule and
    this only checks the lens now reaches it.
    """
    plan = {"narrowedTo": route.narrowed_entries([
        _Finding(_ref("direct", "Direct", ["direct_001"], 441)),
        _Finding(_ref("acquired", "Acquired", ["acquired_001"], 199))])}
    assert receipt._two_or_more_populations(plan) == {lens_mod.SOURCE_TYPE_FIELD}


def test_one_lens_population_is_a_filter_and_not_a_breakdown():
    plan = {"narrowedTo": route.narrowed_entries(
        [_Finding(_ref("direct", "Direct", ["direct_001"], 441))])}
    assert receipt._two_or_more_populations(plan) == set()
