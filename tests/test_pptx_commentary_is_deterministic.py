#!/usr/bin/env python3
"""tests/test_pptx_commentary_is_deterministic.py — the pack's own claim, pinned.

Every generated pack states, on its methodology page:

    Commentary is generated deterministically from those figures.
    No language model is used in its production.

That is a product claim made to a funder, and it must be enforced by something
other than intention. The production deck path — POST /mi/decks/generate ->
pptx_stage -> mi_agent_pptx.cli -> DeckBuilder — must contain no route by which
model-written text could reach a slide.

``mi_agent_pptx/insight_resolver.py`` DOES carry an LLM-strapline path. It is v1
dead code: nothing on the production path imports it. That is exactly the kind
of fact that stops being true quietly, so it is asserted rather than assumed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: Every module the production deck build actually executes.
PRODUCTION_MODULES = (
    "cli.py", "deck.py", "mi_api.py", "composition.py", "insights.py",
    "movement.py", "watchlist.py", "cohorts.py", "concentration.py",
    "forecast_accuracy.py", "materiality_bridge.py", "render.py",
    "chart_resolver.py", "preflight.py", "deck_context.py",
)

#: Modules that carry an LLM path and must stay OFF the production path.
QUARANTINED = ("insight_resolver.py", "pptx_builder.py")


def _imports(path: Path) -> set:
    """Every module name imported by ``path``, however it is imported."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(a.name for a in node.names)
    return names


def test_no_production_deck_module_imports_an_llm_capable_module():
    """Catches: the dead LLM strapline path being wired back in."""
    package = _ROOT / "mi_agent_pptx"
    offenders = []
    for name in PRODUCTION_MODULES:
        path = package / name
        if not path.exists():
            continue
        for imported in _imports(path):
            for banned in QUARANTINED:
                stem = banned[:-3]
                if imported.endswith(stem):
                    offenders.append(f"{name} imports {imported}")
    assert not offenders, offenders


def test_the_deck_path_reaches_no_model_provider():
    """No production deck module may import a model SDK."""
    package = _ROOT / "mi_agent_pptx"
    banned = ("anthropic", "openai", "cohere", "mistralai", "google.generativeai")
    offenders = []
    for name in PRODUCTION_MODULES:
        path = package / name
        if not path.exists():
            continue
        for imported in _imports(path):
            if any(imported == b or imported.startswith(b + ".") for b in banned):
                offenders.append(f"{name} imports {imported}")
    assert not offenders, offenders


def test_the_stage_that_generates_a_deck_reaches_no_model_provider():
    for module in ("mi_agent_api/pptx_stage.py", "mi_agent_api/deck_generation.py"):
        path = _ROOT / module
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for banned in ("import anthropic", "import openai", "from anthropic",
                       "from openai"):
            assert banned not in text, f"{module}: {banned}"


def test_the_claim_on_the_page_is_the_claim_the_code_supports():
    """The methodology sentence and the enforcement must not drift apart."""
    deck = (_ROOT / "mi_agent_pptx" / "deck.py").read_text(encoding="utf-8")
    assert "No language model is used in its production." in deck
    assert "Commentary is generated deterministically from those figures." in deck
