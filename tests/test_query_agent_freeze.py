#!/usr/bin/env python3
"""tests/test_query_agent_freeze.py — the deck cannot move the Query Agent.

The presentation selector decides which dimensions earn a panel. That is a
DISPLAY decision, and it was rewritten to rank by information content rather
than by a hand-written preference order. The MI Query Agent answers questions
about the same book, and it must be unaffected by that: a question that
returned 66.9% joint borrower share before the selector changed has to return
66.9% after it, because the number was never the selector's to decide.

Nothing enforced that separation except the fact that nobody had wired the two
together yet. These tests make it structural, in both directions:

  * no module on the Query Agent's own path imports the shared presentation
    semantics, or anything under the deck builder — so a selector change has
    no route by which it could reach an answer;

  * the presentation module owns no query vocabulary of its own, so it cannot
    quietly grow into a second, disagreeing recogniser.

The list below is the freeze list stated in the brief — recogniser, parser,
routing, executor, vocabulary, capability resolution — resolved to the actual
modules that implement each.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: The frozen surface, by the role each module plays.
FROZEN = {
    "recogniser": (
        "mi_agent_api/recogniser_registry.py",
    ),
    "parser": (
        "mi_agent/llm_query_parser.py",
        "mi_agent/parsed_question.py",
        "mi_agent/mi_query_spec.py",
    ),
    "routing": (
        "mi_agent_api/chat_routing.py",
    ),
    "executor": (
        "mi_agent/mi_query_executor.py",
        "mi_agent/mi_query_validator.py",
    ),
    "vocabulary": (
        "mi_agent/business_semantics.py",
        "mi_agent/semantic_resolver.py",
    ),
    "capability resolution": (
        "mi_agent/mi_query_contract.py",
    ),
}

#: What the Query Agent must not reach for. ``presentation`` is the selector
#: this sprint rewrote; ``mi_agent_pptx`` is the deck that consumes it.
DISPLAY_ONLY = ("mi_agent_api.presentation", "mi_agent_pptx")


def _module_paths():
    for role, files in FROZEN.items():
        for name in files:
            yield role, name, _ROOT / name


def _imported_names(path: Path) -> set:
    """Every module name ``path`` imports, however it imports it."""
    names = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
                names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_the_freeze_list_resolves_to_real_modules():
    """A freeze that names a file which has moved protects nothing."""
    missing = [name for _role, name, path in _module_paths() if not path.exists()]
    assert not missing, missing


@pytest.mark.parametrize(
    "role,name",
    [(role, name) for role, name, _ in _module_paths()],
    ids=[f"{role}:{Path(name).stem}" for role, name, _ in _module_paths()],
)
def test_a_frozen_query_module_cannot_see_the_display_layer(role, name):
    """Catches: a future "just read the selector here" shortcut.

    If the executor could ask the presentation layer which dimensions are
    interesting, then changing what a slide shows would change what a question
    answers. It cannot, and this is why.
    """
    imported = _imported_names(_ROOT / name)
    reached = sorted(
        n for n in imported
        if any(n == d or n.startswith(d + ".") for d in DISPLAY_ONLY)
    )
    assert not reached, (
        f"the {role} ({name}) imports display-layer code: {reached}. "
        "A presentation change would then be able to move an MI Query answer."
    )


def test_the_selector_carries_no_query_vocabulary():
    """The reverse direction: presentation must not become a second parser.

    It ranks dimensions the caller hands it. It does not decide what a
    question means, so it has no business importing the semantics registry or
    the spec the Query Agent parses into.
    """
    imported = _imported_names(_ROOT / "mi_agent_api" / "presentation.py")
    owned_by_query = sorted(
        n for n in imported
        if n.startswith("mi_agent.") or n == "mi_agent"
        or "query" in n.lower() or "semantic" in n.lower()
    )
    assert not owned_by_query, owned_by_query


#: What a deck slide may legitimately borrow from the Query Agent's package.
#: Both are SHARED GOVERNED CONTRACTS rather than query routes — they parse
#: nothing, route nothing, and answer nothing:
#:
#:   * ``mi_query_validator.load_mi_semantics`` reads the governed semantics
#:     registry, the dimension vocabulary both surfaces are defined against.
#:     Reading it is the parity link item 19 asks for.
#:   * ``portfolio_scope`` resolves which portfolios a request may see. The
#:     deck must apply the same scope the API does, or a pack would show a
#:     book the requester cannot query.
REGISTRY_READS = ("mi_agent.mi_query_validator", "mi_agent.portfolio_scope")

#: Everything that actually turns a question into an answer.
QUERY_PATH = (
    "mi_agent.llm_query_parser",
    "mi_agent.mi_query_executor",
    "mi_agent.mi_query_harness",
    "mi_agent.parsed_question",
    "mi_agent.semantic_resolver",
    "mi_agent_api.chat_routing",
    "mi_agent_api.recogniser_registry",
)


def test_the_deck_is_composed_from_the_engine_not_from_parsed_questions():
    """The pack's content comes from engine outputs, never from a Query answer.

    Catches a deck slide that starts asking the Query Agent for its numbers,
    which would put question parsing on the critical path of a generated pack
    and make this freeze unenforceable — a parser change would silently become
    a deck change.

    The permitted crossings are governed shared contracts rather than routes; ``test_the_deck_reads_shared_contracts_only``
    pins the full set.
    """
    builder = _ROOT / "mi_agent_pptx"
    offenders = {}
    for path in sorted(builder.glob("*.py")):
        reached = sorted(
            n for n in _imported_names(path)
            if any(n == q or n.startswith(q + ".") for q in QUERY_PATH)
        )
        if reached:
            offenders[path.name] = reached
    assert not offenders, offenders


def test_the_deck_reads_shared_contracts_only():
    """Pins the legitimate crossings, so a new one has to be argued.

    The deck reads the governed dimension vocabulary and the governed
    portfolio scope. If some other symbol from the Query package appears here,
    it is new and it needs justifying — that is the point of failing on it.
    """
    builder = _ROOT / "mi_agent_pptx"
    crossings = {}
    for path in sorted(builder.glob("*.py")):
        reached = sorted(
            n for n in _imported_names(path)
            if n.split(".")[0] == "mi_agent"
            and not any(n == r or n.startswith(r + ".") for r in REGISTRY_READS)
        )
        if reached:
            crossings[path.name] = reached
    assert not crossings, crossings
