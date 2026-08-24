"""tests/test_dataset_ownership.py — the question decides the dataset.

The product rule this pins:

    Natural-language MI is self-contained. The user's question determines the
    analytical dataset. The active React/workspace tab must not silently change
    dataset semantics.

Funded / Pipeline / Forecast are DATASET semantics.
Direct / Acquired / a named SPV are POPULATION semantics.
The two axes are independent and this file asserts that they stay so.

Before this rule, six of fourteen probe questions were TAB-SENSITIVE — the same
sentence served from a different dataset depending on the tab, including
"the balance by seasoning segment excluding pipeline cases" served from the
pipeline on the pipeline tab: the question ruled the pipeline out in words and
the tab put it back.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from mi_agent_api import workspace as ws
from question_interpretation.projection import project
from question_interpretation.schema import (PROV_CALLER_CONTEXT, PROV_DEFAULT,
                                            PROV_EXPLICIT_USER)

_REPO = Path(__file__).resolve().parent.parent

#: Every tab value a caller can be on, plus "no tab at all".
TABS = (None, "", "funded", "pipeline", "forecast", "nonsense")

FUNDED = (
    "What is the funded balance?",
    "What is the funded loan count?",
    "What is the acquired funded balance?",
    "What is the direct funded balance?",
)
PIPELINE = (
    "How many cases are there?",
    "How many applications are there?",
    "How many KFIs are there?",
    "How many offers are there?",
    "What is the pipeline amount?",
)
FORECAST = (
    "Forecast application volumes next quarter",
    "Forecast case completions over the next three months",
    "Forecast funded volumes for the next quarter",
    "How much of the forecast comes from pipeline?",
)

MATRIX = ([(q, "funded") for q in FUNDED]
          + [(q, "pipeline") for q in PIPELINE]
          + [(q, "forecast") for q in FORECAST])


# --------------------------------------------------------------------------- #
# The matrix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,expected", MATRIX)
def test_the_question_alone_determines_the_dataset(question, expected):
    assert ws.resolve_dataset(question) == expected, question


@pytest.mark.parametrize("question,expected", MATRIX)
def test_the_dataset_is_identical_on_every_tab(question, expected):
    """THE tab-independence invariant.

    Asserted through `resolve_active_view`, which is the entry point every
    caller still holds — so a tab reaching the decision by the old route would
    fail here even though `resolve_dataset` has no parameter to reach it by.
    """
    seen = {tab: ws.resolve_active_view(question, tab) for tab in TABS}
    assert set(seen.values()) == {expected}, (question, seen)


def test_the_tab_argument_is_inert():
    """`dataset_context` survives for callers and has no semantic effect.

    Pinned explicitly because an accepted-and-ignored parameter is exactly the
    kind of thing that quietly becomes an input again.
    """
    for question in ("amount by region", "What is the total balance?"):
        answers = {ws.resolve_active_view(question, tab) for tab in TABS}
        assert answers == {"funded"}, question


def test_a_question_naming_no_dataset_takes_the_governed_default():
    assert ws.resolve_dataset("What is the total balance?") == "funded"
    assert ws.resolve_dataset("amount by region") == "funded"
    assert ws.resolve_dataset("") == "funded"
    assert ws.resolve_dataset(None) == "funded"


def test_a_disclaimed_tape_word_does_not_select_it():
    """B21, preserved through the consolidation and no longer overridable.

    This is the case the tab used to reverse: the sentence rules the pipeline
    out, and on the pipeline tab it was served from the pipeline anyway.
    """
    q = "What is the balance by seasoning segment excluding pipeline cases?"
    assert ws.resolve_dataset(q) == "funded"
    assert {ws.resolve_active_view(q, tab) for tab in TABS} == {"funded"}


def test_forecast_beats_the_pipeline_vocabulary():
    """The precedence the retired second owner had backwards.

    `_dataset_for` tested its tape vocabulary BEFORE any forecast reading, so
    "Forecast application volumes next quarter" was `pipeline` to it while the
    active view was `forecast`.
    """
    for q in ("Forecast application volumes next quarter",
              "How much of the forecast comes from pipeline?",
              "Forecast case completions over the next three months"):
        assert ws.resolve_dataset(q) == "forecast", q


def test_the_forecast_reading_is_the_word_forecast_and_nothing_wider():
    """A LIMIT, pinned so it is a decision rather than an oversight.

    "Show projected completions from the pipeline" resolves PIPELINE, not
    FORECAST. `projected`, `expected` and `anticipated` are not in the dataset
    owner's vocabulary and were deliberately not added: that wider reading is
    `mi_workflows.analytical.intent`'s ``REQ_FORECAST``, which exists to decide
    REFUSABILITY and moves 59 of the 882 corpus questions when used to select a
    dataset — including "top brokers by expected funded amount" to forecast.

    Widening this is a product decision with its own blast radius, not a
    tidy-up, and it is out of scope here.
    """
    assert ws.resolve_dataset("Show projected completions from the pipeline") \
        == "pipeline"
    assert ws.resolve_dataset("What is the expected funded balance?") == "funded"


# --------------------------------------------------------------------------- #
# Population is a different axis
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,scope_word", [
    ("What is the acquired book balance?", "acquired"),
    ("What is the direct book balance?", "direct"),
    ("What is the acquired funded balance?", "acquired"),
    ("What is the direct funded balance?", "direct"),
])
def test_population_scope_never_chooses_the_dataset(question, scope_word):
    """A provenance word selects a POPULATION within the funded book.

    It must never select a tape. Asserted on the contract, not only on the
    resolver, so a leak through the projection would be caught too.
    """
    assert ws.resolve_dataset(question) == "funded", question
    qi = project(question, semantics={})
    assert qi.dataset.dataset == "funded"
    assert scope_word in (qi.source_scope.raw_text or "").lower() \
        or qi.source_scope.scope is not None, \
        f"the scope claim lost {scope_word!r}: {qi.source_scope.as_dict()}"


def test_the_dataset_axis_ignores_the_scope_vocabulary():
    """No provenance word appears in the owner's EXECUTABLE source.

    The docstring names them at length — explaining why they are absent is the
    point of it — so the docstring is stripped before looking.
    """
    tree = ast.parse(inspect.getsource(ws.resolve_dataset).strip())
    fn = tree.body[0]
    body = fn.body[1:] if ast.get_docstring(fn) else fn.body
    src = "\n".join(ast.unparse(node) for node in body).lower()
    for word in ("acquired", "direct", "spv", "book"):
        assert word not in src, f"{word!r} reached the dataset decision"


# --------------------------------------------------------------------------- #
# The contract is the handoff
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,expected", MATRIX)
def test_the_contract_carries_the_owners_answer_on_every_tab(question, expected):
    for tab in TABS:
        qi = project(question, semantics={}, caller_dataset=tab)
        assert qi.dataset.dataset == expected, (question, tab)
        assert qi.dataset.source == "mi_agent_api.workspace.resolve_dataset"


def test_the_caller_context_provenance_is_no_longer_reachable():
    """Two cases now, not three: the question said so, or the default applied.

    A property worth asserting rather than believing — `PROV_CALLER_CONTEXT` on
    this axis WAS the tab dependence.
    """
    for question, _ in MATRIX:
        for tab in TABS:
            qi = project(question, semantics={}, caller_dataset=tab)
            assert qi.dataset.provenance != PROV_CALLER_CONTEXT
            assert qi.dataset.provenance in (PROV_EXPLICIT_USER, PROV_DEFAULT)


# --------------------------------------------------------------------------- #
# One owner, structurally
# --------------------------------------------------------------------------- #
def test_the_resolver_cannot_be_handed_a_tab():
    """Not "it ignores the tab" — it has nowhere to put one."""
    params = list(inspect.signature(ws.resolve_dataset).parameters)
    assert params == ["question"], params


def test_the_second_owner_is_gone():
    from mi_agent_api import chat_routing
    assert not hasattr(chat_routing, "_dataset_for")
    assert not hasattr(chat_routing, "_PIPELINE_WORDS")


def test_no_production_module_re_decides_the_dataset_from_raw_text():
    """The tape vocabulary exists in exactly one production place.

    A grep-shaped test on purpose: the failure mode this guards is a THIRD
    reader appearing, and a third reader is invisible to any behavioural test
    until its vocabulary happens to disagree.
    """
    artefacts = {w.strip().lower() for w in ws.PIPELINE_ARTEFACTS}
    owner = _REPO / "mi_agent_api" / "workspace.py"
    holders = []
    for path in _REPO.rglob("*.py"):
        rel = path.relative_to(_REPO).as_posix()
        if "test" in rel or rel.startswith(("migration_phase0/", "docs/")):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                continue
            vals = {e.value.strip().lower() for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            # EXACT set equality, not containment. `intent._PIPELINE_TERMS` is a
            # superset and is a legitimate different reader — it decides
            # refusability, never the dataset. Only an exact copy of THIS
            # vocabulary is a second dataset owner.
            if vals == artefacts:
                holders.append(rel)
    assert holders == [owner.relative_to(_REPO).as_posix()], (
        "the pipeline tape vocabulary must live in exactly one production "
        "module; found it in %s" % sorted(set(holders)))
