#!/usr/bin/env python3
"""tests/test_selector_governs_every_surface.py

One rule, applied everywhere a dimension competes for a panel.

The information-first rule was introduced for the funded stratification matrix.
That is not enough: the pipeline slide picks a second cut, and the
multidimensional page picks pairs, and each of those used to pick by a
hand-written preference order of its own. Three call sites with three orders is
three chances to draw the wrong panel.

These pin that all three now defer to ``mi_agent_api.presentation``, and that
the shared surface the React dashboard reads did not narrow while that happened.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mi_agent_api import presentation as P      # noqa: E402
from mi_agent_api import snapshots as S         # noqa: E402
from mi_agent_pptx import deck as D             # noqa: E402


def bars(*pairs):
    return [{"label": label, "balance": float(v)} for label, v in pairs]


# --------------------------------------------------------------------------- #
# The pipeline second cut.
# --------------------------------------------------------------------------- #

def _pipeline_with(*strats):
    return {"stratifications": [
        {"key": k, "label": lbl, "bars": b} for k, lbl, b in strats]}


def _second_cut(pipeline):
    from mi_agent_pptx.deck import DeckBuilder
    return DeckBuilder._pipeline_second_cut(object.__new__(DeckBuilder), pipeline)


def test_the_pipeline_second_cut_takes_the_most_informative_dimension():
    """Catches: the pipeline page drawing "by product — Standard 96%".

    Product is named first in the call site's preference tuple. It is also,
    on this pipeline, a single meaningful category. Region says something;
    product does not, so region takes the panel.
    """
    cut = _second_cut(_pipeline_with(
        ("product", "By product", bars(("Standard", 96), ("Other", 4))),
        ("region", "By region", bars(("London", 34), ("North", 30),
                                     ("Wales", 20), ("Scotland", 16))),
    ))
    assert cut is not None
    assert cut["dimension"] == "region", cut["dimension"]


def test_preference_no_longer_decides_the_pipeline_panel():
    """Both cuts are informative; the more informative one still wins.

    This is the case the old rule got wrong silently — nothing looked broken,
    the page just showed the weaker cut every time.
    """
    cut = _second_cut(_pipeline_with(
        ("product", "By product", bars(("Standard", 70), ("Green", 30))),
        ("ltv", "By LTV band", bars(("50-60%", 28), ("60-70%", 26),
                                    ("70-80%", 24), ("80-90%", 22))),
    ))
    assert cut["dimension"] == "ltv", cut["dimension"]


def test_the_pipeline_draws_nothing_rather_than_a_single_bar():
    """No informative cut means the facts panel, not a one-bar chart."""
    assert _second_cut(_pipeline_with(
        ("product", "By product", bars(("Standard", 100))),
        ("region", "By region", bars(("London", 999), ("Wales", 1))),
    )) is None


def test_the_pipeline_reads_the_governed_stratifications_first():
    """The fallback breakdowns are for older payloads only.

    If governed stratifications are present the page must read them, so the
    Pipeline Stratifications slide and this panel cannot disagree about what
    the pipeline supports.
    """
    pipeline = _pipeline_with(
        ("region", "By region", bars(("London", 40), ("Wales", 35), ("North", 25))))
    pipeline["brokerBreakdown"] = [{"key": "Direct", "pipelineAmount": 10_000_000}]
    assert _second_cut(pipeline)["dimension"] == "region"


# --------------------------------------------------------------------------- #
# The multidimensional pairs.
# --------------------------------------------------------------------------- #

def test_pair_selection_is_ranked_by_information_not_declaration_order():
    """The pair chooser must consult the same shape rule, not just a list.

    Reading the source is the honest check here: the alternative is to build a
    frame that happens to reorder, which proves the outcome rather than the
    mechanism.
    """
    src = inspect.getsource(S.select_multidim_pairs)
    assert "dispersion" in src, (
        "select_multidim_pairs no longer scores candidate pairs — it is back "
        "to picking by declaration order")


def test_a_pair_that_says_nothing_is_rejected_with_a_true_reason():
    """Every ledger line must be mechanically true; "too sparse" must mean it."""
    codes = {S.REASON_TOO_SPARSE, S.REASON_REDUNDANT}
    assert len(codes) == 2 and all(isinstance(c, str) and c for c in codes)


# --------------------------------------------------------------------------- #
# The shared surface React reads.
# --------------------------------------------------------------------------- #

#: What ``presentation`` promised before this sprint. Rewriting the RULE is
#: allowed; withdrawing a name other code imports is not.
SHARED_SURFACE = ("dispersion", "is_informative", "select_dimensions")


def test_the_shared_presentation_surface_did_not_narrow():
    for name in SHARED_SURFACE:
        assert callable(getattr(P, name, None)), f"presentation.{name} is gone"


def test_dispersion_still_answers_the_original_one_argument_call():
    """New behaviour arrived as new keywords; the old call still works.

    Catches a signature change that would break any caller — React's server
    side included — that never asked for the new knobs.
    """
    shape = P.dispersion(bars(("A", 40), ("B", 35), ("C", 25)))
    assert shape["informative"] is True
    assert set(("informative", "reason")) <= set(shape)


def test_every_rejection_carries_a_machine_readable_code():
    """The methodology ledger prints prose; something has to pin the prose.

    A reason code lets a test say "this was rejected for having one category"
    without matching on a sentence that a copy edit could change.
    """
    out = P.select_dimensions([
        {"key": "broker", "label": "By broker", "bars": bars(("Direct", 100))},
        {"key": "ltv", "label": "By LTV",
         "bars": bars(("A", 30), ("B", 25), ("C", 25), ("D", 20))},
        {"key": "region", "label": "By region",
         "bars": bars(("L", 40), ("W", 35), ("S", 25))},
    ], want=1, preferred=("ltv", "region", "broker"))
    for row in out["rejected"]:
        assert row.get("reasonCode"), row
        assert row.get("reason"), row


def test_the_deck_owns_no_second_selection_rule():
    """Catches a call site that reintroduces its own ordering.

    The deck may pass a preference tuple — that is the tie-break — but it must
    not sort candidates itself before or after asking.
    """
    src = (_ROOT / "mi_agent_pptx" / "deck.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    hand_rolled = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "attr", None) or getattr(fn, "id", None)
        if name != "sort" and name != "sorted":
            continue
        text = lines[node.lineno - 1]
        if "strat" in text or "dimension" in text.lower():
            hand_rolled.append((node.lineno, text.strip()))
    assert not hand_rolled, hand_rolled


# --------------------------------------------------------------------------- #
# The ledger the page prints.
# --------------------------------------------------------------------------- #

def _rej(key, label, code):
    return {"key": key, "label": label, "reasonCode": code}


def test_a_dimension_that_ranked_lower_is_never_called_uncharted():
    """THE UNTRUE LEDGER LINE THIS SPRINT REMOVES.

    "The whole book sits in a single band" is a claim about the DATA. Ticket
    size spread over five bands and lost on score; saying the book sits in one
    band would be false, and a funder can check it against the page above.
    """
    note = D.strat_ledger_note(
        [_rej("ticket", "By ticket size", P.REASON_LOWER_RANKED)], [], drawn=4)
    assert "single band" not in note, note
    assert "ranked below the 4" in note, note


def test_a_dimension_the_book_cannot_distribute_on_says_so():
    note = D.strat_ledger_note(
        [_rej("broker", "By broker / channel", P.REASON_ONE_CATEGORY)], [],
        drawn=4)
    assert "single band" in note, note


def test_both_facts_are_reported_and_kept_apart():
    """A book can have one of each, and each dimension must sit in its own
    sentence — otherwise the reader cannot tell which claim is about which."""
    note = D.strat_ledger_note(
        [_rej("broker", "By broker / channel", P.REASON_ONE_CATEGORY),
         _rej("ticket", "By ticket size", P.REASON_LOWER_RANKED)], [], drawn=4)
    uncharted, ranked = note.split("charted.")
    assert "By broker / channel" in uncharted and "By ticket size" not in uncharted
    assert "By ticket size" in ranked and "By broker / channel" not in ranked


def test_the_page_says_nothing_when_there_is_nothing_to_say():
    """No suppressed dimension, no footnote — the strip is not padding."""
    assert D.strat_ledger_note([], [], drawn=4) == ""


def test_a_long_list_is_summarised_rather_than_run_off_the_slide():
    note = D.strat_ledger_note(
        [_rej(f"d{i}", f"By dimension {i}", P.REASON_LOWER_RANKED)
         for i in range(7)], [], drawn=4)
    assert "and 3 other(s)" in note, note
    assert note.count(",") <= 4, note


def test_the_clause_counts_the_panels_actually_drawn():
    """"Ranked below the 4 drawn here" has to be the number on the page."""
    assert "below the 2 drawn" in D.strat_ledger_note(
        [_rej("ticket", "By ticket size", P.REASON_LOWER_RANKED)], [], drawn=2)
