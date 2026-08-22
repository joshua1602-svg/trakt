"""The instruction corpus, run against the real interpreter.

Every reading defect this feature has had was found the same way: in
production, one screenshot at a time, after a client-facing document had
already been built from a wrong answer. The fixes were structural; the process
was reactive. A rule-based reader without a corpus cannot tell you that today's
fix broke yesterday's sentence — and one of these entries records a defect that
NO screenshot ever showed, found only by testing a phrasing nobody had sent.

``corpus/instructions.yaml`` is the corpus. Adding a sentence needs no Python:
paste it, say why it is there, and assert what matters. Assertions are partial
by design — an entry that pins one field is a good entry, and far better than
leaving the sentence out because pinning all of it felt like work.

Two kinds of assertion, and the second is the one that catches regressions:

``expect``  what must be read.
``reject``  substrings that must NOT appear in a value. A defect is pinned by
            naming the WRONG answer, because the right answer alone does not
            stop a different wrong answer taking its place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from operations_control.occ_agent.interpretation import (
    DeterministicInterpreter, Interpretation,
)
from operations_control.onboarding.catalogue import catalogue

CORPUS = Path(__file__).with_name("corpus") / "instructions.yaml"

#: Every key an entry may assert, and how to read it out of an interpretation.
#: Listed once, so an unknown key in the YAML is a loud failure rather than a
#: silently ignored expectation — a typo'd key that quietly asserts nothing is
#: worse than no entry at all.
READERS: Dict[str, Any] = {
    "client_name": lambda i: _step(i, "client").get("client_name", ""),
    "jurisdiction": lambda i: _step(i, "client").get("jurisdiction", ""),
    "products": lambda i: _step(i, "reporting").get("products", []),
    "legal_name": lambda i: _first(i, "entities").get("legal_name", ""),
    "roles": lambda i: _first(i, "entities").get("roles", []),
    "asset_class": lambda i: _first(i, "portfolios").get("asset_class", ""),
    "portfolio_name": lambda i: _first(i, "portfolios").get("display_name", ""),
    "portfolio_id": lambda i: _first(i, "portfolios").get("portfolio_id", ""),
    "streams": lambda i: sorted(i.streams),
    "cadence": lambda i: str(i.delivery.get("cadence") or ""),
    "stream_cadence": lambda i: {k: str(v.get("cadence") or "")
                                 for k, v in i.stream_delivery.items()},
    "reporting_period": lambda i: i.reporting_period,
    "unrecognised": lambda i: list(i.unrecognised),
    # Which reads Trakt is not certain of. Pinning this is how the corpus
    # protects the grading: a change that quietly promotes a guess to a
    # certainty fails here.
    "uncertain": lambda i: sorted(i.confidence),
}


def _step(interpretation: Interpretation, step: str) -> Dict[str, Any]:
    return dict((interpretation.steps or {}).get(step) or {})


def _first(interpretation: Interpretation, step: str) -> Dict[str, Any]:
    """The first item of a repeatable section, or an empty one."""
    items = _step(interpretation, step).get(step) or []
    return dict(items[0]) if items else {}


def _entries() -> List[Dict[str, Any]]:
    loaded = yaml.safe_load(CORPUS.read_text(encoding="utf-8")) or []
    assert isinstance(loaded, list), "the corpus is a list of entries"
    return loaded


ENTRIES = _entries()
IDS = [str(e.get("name") or index) for index, e in enumerate(ENTRIES)]


@pytest.fixture(scope="module")
def interpreter() -> DeterministicInterpreter:
    return DeterministicInterpreter(cat=catalogue())


# --------------------------------------------------------------------------- #
# The corpus itself has to be well formed, or it protects nothing
# --------------------------------------------------------------------------- #

def test_the_corpus_is_not_empty():
    assert len(ENTRIES) >= 5


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_every_entry_says_why_it_is_there(entry):
    """An entry nobody can explain is one nobody dares change."""
    assert str(entry.get("why") or "").strip(), entry.get("name")
    assert str(entry.get("instruction") or "").strip(), entry.get("name")


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_every_assertion_key_is_one_the_corpus_can_read(entry):
    """A typo'd key would assert nothing, quietly. Fail loudly instead."""
    for key in (entry.get("expect") or {}):
        assert key in READERS, f"{entry.get('name')}: unknown key {key!r}"
    for key in (entry.get("reject") or {}):
        assert key in READERS, f"{entry.get('name')}: unknown key {key!r}"


def test_entry_names_are_unique():
    names = [e.get("name") for e in ENTRIES]
    assert len(names) == len(set(names)), names


# --------------------------------------------------------------------------- #
# The corpus, read
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_the_instruction_is_read_as_the_corpus_says(entry, interpreter):
    read = interpreter.interpret_instruction(entry["instruction"])

    for key, expected in (entry.get("expect") or {}).items():
        actual = READERS[key](read)
        if key in ("streams", "uncertain") and isinstance(expected, list):
            expected = sorted(expected)
        assert actual == expected, (
            f"{entry['name']}: {key} was {actual!r}, expected {expected!r}")

    for key, forbidden in (entry.get("reject") or {}).items():
        actual = str(READERS[key](read)).lower()
        for fragment in forbidden:
            assert str(fragment).lower() not in actual, (
                f"{entry['name']}: {key} is {actual!r}, which must not "
                f"contain {fragment!r}")


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_every_instruction_produces_a_valid_interpretation(entry, interpreter):
    """Whatever is read must be catalogue-shaped, whether or not it is right.

    This is the cheapest assertion in the file and the one most likely to catch
    a bad change: a reader that starts inventing fields fails here on every
    entry at once.
    """
    read = interpreter.interpret_instruction(entry["instruction"])
    read.validate(catalogue())


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_reading_an_instruction_twice_gives_the_same_answer(entry, interpreter):
    """Determinism is the property the whole governance argument rests on."""
    first = interpreter.interpret_instruction(entry["instruction"])
    second = interpreter.interpret_instruction(entry["instruction"])
    assert first.to_dict() == second.to_dict()
