"""One owner for LEVEL versus MOVEMENT, and nothing else may decide it.

The estate had FIVE components inferring the distinction independently. Measured
over the 882-question corpus they disagreed on 30 questions and NO reader was a
superset of any other:

    A  period_change.recognition.has_change_language      17
    B  llm_query_parser._COMPARE_TRIGGER_RE               21
    C  spec.temporal_mode == "compare"                     5
    D  interpreter.deterministic's compare branch         20
    E  concentration_query's compare gate                  0
    union                                                 30

Reader A missed "How did the balance change since last month?" — the most
canonical movement question in the estate — because its vocabulary carried
"changed", "change in" and "has changed" but not the bare verb.

These tests pin the owner's semantics and, structurally, its singularity.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from question_interpretation.lexical import (LEVEL, MOVEMENT, is_movement_question,
                                             temporal_aspect)

_REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Semantics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "How did the balance change since last month?",
    "Which region grew the most balance since last month?",
    "Compare October and November funded balance.",
    "How has lending to North West changed compared with Scotland?",
    "What is the month-on-month movement?",
    "Are we originating different types of loans now compared with a few months ago?",
])
def test_a_change_over_time_is_a_movement(question):
    assert temporal_aspect(question).verdict == MOVEMENT, question


@pytest.mark.parametrize("question", [
    # A LEVEL, however superlative.
    "Which region has the largest balance?",
    "top 5 brokers by balance",
    "which Broker has the smallest balance",
    # Against a FORECAST — the second operand is a plan, not an earlier date.
    "compare current funded balance to expected funded",
    "Compare current weighted pipeline forecast with run-rate extrapolation.",
    # Against a THRESHOLD.
    "What is the largest geographic concentration versus limit?",
    # Two POPULATIONS of one snapshot. The `seasoning` owner's axis, not this one.
    "How does the front book compare with our older lending from a risk perspective?",
    "Compare the credit profile of the front book with our seasoned loans.",
    "How different is the risk profile of recent originations versus the back book?",
    # A SERIES of levels is not a two-point movement.
    "balance by month over the last 6 months",
])
def test_these_are_levels_and_the_reason_matters(question):
    assert temporal_aspect(question).verdict == LEVEL, question


def test_a_movement_says_what_made_it_one():
    """Evidence is returned so a receipt can show WHY, and so a verdict with no
    evidence is checkably a level."""
    assert temporal_aspect("Which region has the largest balance?").evidence == ()
    assert "since_period" in temporal_aspect(
        "Which region grew the most since last month?").evidence


def test_a_bare_and_between_two_periods_is_not_a_comparison():
    """"show pipeline by stage for October and November" asks for two LEVELS
    side by side. Only an explicit comparison verb makes `and` a comparison."""
    assert not is_movement_question(
        "Show pipeline by stage for October and November.")
    assert is_movement_question("Compare October and November funded balance.")


# --------------------------------------------------------------------------- #
# Singularity — the structural half
# --------------------------------------------------------------------------- #
#: Every module that used to decide this, and now must only ask.
_DELEGATED = (
    "mi_agent/period_change/recognition.py",
    "mi_agent/llm_query_parser.py",
    "mi_agent/interpreter/deterministic.py",
    "mi_agent_api/concentration_query.py",
    "question_interpretation/projection.py",
)

#: Words that only appear in a change-language vocabulary. A module that defines
#: a collection or pattern containing several of them is deciding the question
#: again, whatever it calls the variable.
_CHANGE_WORDS = ("grew", "grown", "increased", "decreased", "declined",
                 "moved", "movement", "changed", "shrunk", "rose", "fell")


#: Vocabularies that CONTAIN change words without DECIDING the aspect, each with
#: the reason it is exempt. The exemption is by name and must be argued for: a
#: new entry here is a claim that a vocabulary does not decide level versus
#: movement, and that claim is reviewable in a way a lowered threshold is not.
_NOT_ASPECT_VOCABULARY = {
    ("mi_agent/period_change/recognition.py", "OVERVIEW_MARKERS"):
        "decides portfolio-overview MODE once the question is already known to "
        "be about a change; the aspect gate has run before it",
    ("mi_agent/period_change/recognition.py", "BRIDGE_MARKERS"):
        "decides whether the answer includes the balance BRIDGE, same position "
        "in the sequence",
    ("mi_agent/period_change/recognition.py", "TREND_MARKERS"):
        "a SERIES is not a two-point movement and is a separate decline reason",
}


def _vocabulary_assignments(path: Path):
    """Module-level names bound to a literal collection, with their strings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = ([node.target] if isinstance(node, ast.AnnAssign)
                   else node.targets)
        name = next((t.id for t in targets if isinstance(t, ast.Name)), None)
        if name is None or node.value is None:
            continue
        literals = [n.value for n in ast.walk(node.value)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        if literals:
            yield name, literals


def test_no_other_module_defines_a_change_vocabulary():
    """The singularity rule, enforced rather than asserted in a comment.

    A module-level collection carrying three or more distinct change words IS a
    level-versus-movement vocabulary, whatever the variable is called. The two
    that legitimately contain such words without deciding the aspect are
    exempted BY NAME above, with the reason written down.
    """
    offenders = {}
    for rel in _DELEGATED:
        for name, literals in _vocabulary_assignments(_REPO / rel):
            if (rel, name) in _NOT_ASPECT_VOCABULARY:
                continue
            found = {w for lit in literals for w in _CHANGE_WORDS
                     if w in lit.lower()}
            if len(found) >= 3:
                offenders[f"{rel}:{name}"] = sorted(found)
    assert not offenders, (
        "these define their own LEVEL/MOVEMENT vocabulary again; "
        "`question_interpretation.lexical` is the owner: " + repr(offenders))


def test_every_exemption_still_exists_and_is_still_exempt():
    """An exemption for a vocabulary that has been deleted or renamed is a stale
    licence. It must be removed with the thing it excused."""
    for (rel, name), reason in _NOT_ASPECT_VOCABULARY.items():
        names = {n for n, _ in _vocabulary_assignments(_REPO / rel)}
        assert name in names, f"{rel}:{name} is exempted but no longer exists"
        assert reason.strip(), f"{rel}:{name} is exempted with no reason"


def test_every_delegated_module_asks_the_owner():
    """Delegation is not just the absence of a vocabulary — the module has to
    actually consult the owner, or it has silently stopped deciding at all."""
    for rel in _DELEGATED:
        source = (_REPO / rel).read_text(encoding="utf-8")
        assert ("is_movement_question" in source
                or "temporal_aspect" in source), (
            f"{rel} neither owns the distinction nor asks the owner for it")


def test_the_retired_vocabularies_are_gone():
    """Dead vocabulary left in place is a second owner waiting to be re-used."""
    rec = (_REPO / "mi_agent/period_change/recognition.py").read_text()
    par = (_REPO / "mi_agent/llm_query_parser.py").read_text()
    assert "CHANGE_MARKERS: Tuple" not in rec
    assert "COMPARISON_PERIOD_MARKERS: Tuple" not in rec
    assert "_COMPARE_TRIGGER_RE = re.compile" not in par
    # TREND_MARKERS stays: a series is not a two-point movement.
    assert "TREND_MARKERS" in rec
