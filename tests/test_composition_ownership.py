#!/usr/bin/env python3
"""tests/test_composition_ownership.py — one owner for composition arithmetic.

A portfolio type's SHARE of the book, and the OPENING balance its movement
decomposes, were each derived independently at four call sites in the
presentation layer: the composition bar, its legend, the comparison table and
the executive summary. Four sites, four chances to disagree, under a pack that
claims a single governed origin for all of them.

``mi_agent_api.portfolio_context`` now owns both. These tests pin the behaviour
AND the ownership — the second is what stops the arithmetic drifting back into a
renderer the next time a slide needs a percentage.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# --------------------------------------------------------------------------- #
# Behaviour.
# --------------------------------------------------------------------------- #

def test_a_share_of_nothing_is_undefined_not_zero():
    """Catches: rendering 0% for a book with no resolved balance.

    Zero asserts a measurement. ``None`` says one was not made, and the
    presentation layer decides what to print instead.
    """
    from mi_agent_api.portfolio_context import balance_share
    assert balance_share(30, 120) == pytest.approx(0.25)
    assert balance_share(30, 0) is None
    assert balance_share(30, None) is None
    assert balance_share(None, 120) is None
    assert balance_share(-10, 100) == pytest.approx(-0.1)


def test_opening_is_closing_less_movement():
    from mi_agent_api.portfolio_context import opening_from_movement
    assert opening_from_movement(120, 20) == pytest.approx(100.0)
    assert opening_from_movement(120, -20) == pytest.approx(140.0)
    assert opening_from_movement(120, None) is None
    assert opening_from_movement(None, 20) is None


def test_composition_reconciles_to_the_whole():
    """Shares sum to one, and each opening plus its movement returns its balance."""
    from mi_agent_api.portfolio_context import type_composition

    class _Slice:
        def __init__(self, t, b, m):
            self.portfolio_type, self.balance, self.balance_movement = t, b, m

    slices = [_Slice("direct", 60.0, 10.0), _Slice("acquired", 40.0, -5.0)]
    out = type_composition(100.0, slices)
    assert sum(v["share"] for v in out.values()) == pytest.approx(1.0)
    for v in out.values():
        assert v["opening"] + v["movement"] == pytest.approx(v["balance"])


# --------------------------------------------------------------------------- #
# Ownership.
# --------------------------------------------------------------------------- #

_PRESENTATION = ("mi_agent_pptx/deck.py", "mi_agent_pptx/insights.py")


#: A RATCHET, not a clean bill of health.
#:
#: These are the share-shaped divisions still living in the presentation layer.
#: Each is classified A — the engine should own it — and each belongs to a
#: DIFFERENT analytical domain from portfolio composition, so each is its own
#: migration rather than something to sweep up here:
#:
#:   deck.py     bridge leg share            -> evolution.funded_balance_movement
#:   deck.py     stratification spread       -> snapshots stratifications
#:   deck.py     geographic top-5 share      -> geo.exposure_by_itl3
#:   insights.py geographic top-5 / top share-> geo.exposure_by_itl3
#:   insights.py contributor share of moveme -> evolution.funded_bridge
#:
#: The count may only ever go DOWN. A new entry means share arithmetic was
#: written into a renderer again, and the test says so by name.
_KNOWN_UNMIGRATED_SHARES = 6


def test_composition_shares_are_no_longer_derived_in_the_presentation_layer():
    """Catches the arithmetic drifting back into a renderer.

    Scans for a division whose right operand names a total or whole — the shape
    all four removed composition call sites had. The remaining hits are counted
    against a ratchet and named in ``_KNOWN_UNMIGRATED_SHARES`` above.
    """
    offenders = []
    for rel in _PRESENTATION:
        tree = ast.parse((_ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
                continue
            right = node.right
            name = (right.id if isinstance(right, ast.Name) else
                    right.attr if isinstance(right, ast.Attribute) else "")
            if name in ("total", "total_bal", "total_balance", "total_move"):
                offenders.append(f"{rel}:{node.lineno}")

    assert len(offenders) <= _KNOWN_UNMIGRATED_SHARES, (
        "share arithmetic was written into the presentation layer again: "
        + ", ".join(offenders))
    # And none of them may be a COMPOSITION share — those have an owner now.
    composition = [o for o in offenders if "total_bal" in
                   (_ROOT / o.split(":")[0]).read_text(
                       encoding="utf-8").splitlines()[int(o.split(":")[1]) - 1]]
    assert not composition, composition


def test_the_deck_context_delegates_rather_than_computing():
    """``share_of`` / ``opening_of`` must call the governed owner."""
    source = (_ROOT / "mi_agent_pptx" / "deck_context.py").read_text(encoding="utf-8")
    assert "from mi_agent_api.portfolio_context import balance_share" in source
    assert "from mi_agent_api.portfolio_context import opening_from_movement" in source
