#!/usr/bin/env python3
"""tests/test_final_pack_surfaces.py

What the final funder pack SAYS, checked on the pack a funder receives.

Every test builds a real deck through the React route — ``POST
/mi/decks/generate`` -> poll -> ``GET /mi/decks/download`` — and reads the
PowerPoint that comes back. Nothing calls the builder directly: a page that
renders under a direct call and not under the job service is a page that does
not exist.

These pin three corrections found by reading the printed pack page by page,
plus the two properties the sprint must not have broken on the way. Each states
the defect it would catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("pandas")
pytest.importorskip("pptx")

from tests.test_funder_pack_enhancement import (  # noqa: E402
    deck_text, generate_and_download, rich, simple, slide_titles)

__all__ = ["rich", "simple"]


@pytest.fixture()
def limited(rich):
    """The rich book WITH an operator-approved concentration configuration.

    Concentration limits are contractual: nothing in the tape implies them, and
    the deck refuses to invent one, so the page the headroom wording lives on
    does not render at all until an operator has approved a configuration
    through the store. This commits four regional limits the way the operations
    console does.
    """
    from apps.blob_trigger_app.storage import open_storage
    from mi_agent.concentration_tests.models import (
        ActiveConfiguration, ActiveTest, ApprovalRecord, SourceEvidence)
    from mi_agent.concentration_tests.store import ConcentrationStore
    import os

    client = os.environ["MI_AGENT_CLIENT_ID"]
    tests = [
        ActiveTest(metric_id="geo_region_share", threshold=thr, operator="max",
                   display_name=name, parameters={"regions": regions},
                   evidence=SourceEvidence(
                       source_reference="facility_agreement.pdf",
                       source_text="Schedule 8, clause 3.1"),
                   approval=ApprovalRecord(decision="approved",
                                           operator="Operator",
                                           decided_at="2026-01-10T09:00:00+00:00"))
        for name, thr, regions in (
            ("London concentration", 30.0, ["London"]),
            ("South East concentration", 45.0, ["South East"]),
            ("Wales concentration", 40.0, ["Wales"]),
            ("Scotland concentration", 50.0, ["Scotland"]))]
    ConcentrationStore(open_storage(), container="operations-control"
                       ).commit_configuration(
        client, ActiveConfiguration(client_id=client, version=1,
                                    activated_by="Operator",
                                    library_version="1.0.0", tests=tests))
    return rich


# --------------------------------------------------------------------------- #
# 1. HEADROOM IS A DISTANCE, NOT A SHARE.
# --------------------------------------------------------------------------- #

def test_1_percentage_headroom_is_stated_in_points(limited):
    """Catches: "London concentration at 47% utilisation (16.0% headroom)".

    Headroom on a percentage test is the DIFFERENCE between two percentages —
    30.0% limit less 14.0% current is 16.0 percentage points. Printed with a
    percent sign it landed on the same line as a 47% utilisation figure, where
    a reader has no way to tell the two apart and the natural reading of "16.0%
    of headroom" is a share of the headroom rather than the headroom itself.
    The engine's own answer text has always said pp; only the deck did not.
    """
    text = deck_text(generate_and_download())
    assert "headroom" in text.lower(), "no headroom is reported at all"
    for line in text.splitlines():
        low = line.lower()
        if "headroom" not in low or "%" not in line:
            continue
        # Percentage headroom must carry pp. A currency or count test states
        # headroom in its own unit and no percent sign appears with it.
        assert "pp" in low, f"headroom stated as a percentage: {line!r}"


def test_2_headroom_carries_its_unit_whatever_the_test_measures(limited):
    """Catches: a bare "16.0" under a Headroom column, and £2m printed raw.

    The column printed ``f"{headroom:.1f}"`` with no unit at all. On a
    percentage test that is an unlabelled 16.0 beside a 30.0% limit; on a
    currency test it is 2000000.0 beside limits formatted as money. The unit
    belongs to the test, so the formatter must take it from the test and never
    from the slide.
    """
    from mi_agent_pptx import concentration as C

    assert C.format_headroom(16.0, "pct") == "16.0pp"
    assert C.format_headroom(16.0, "percentage_points") == "16.0pp"
    assert C.format_headroom(None, "pct") == "—"
    # A currency headroom is a difference of pounds, which IS pounds — it keeps
    # the governed money formatting rather than being relabelled in points.
    money = C.format_headroom(2_000_000.0, "gbp")
    assert "pp" not in money and any(ch.isdigit() for ch in money), money
    assert C.format_headroom(12.0, "count") == "12"


# --------------------------------------------------------------------------- #
# 2. A STRAPLINE DESCRIBES THE PAGE IT IS ON.
# --------------------------------------------------------------------------- #

def test_3_stratifications_promise_movement_only_when_they_show_it(rich):
    """Catches: "Composition and period movement" over four panels of
    composition and no movement.

    The strapline was chosen from whether movement was AVAILABLE; the movement
    strip is drawn only where two panels leave room for it. On a four-panel
    matrix the page promised a view it had suppressed, and a reader looking for
    it found nothing — the worst kind of missing content, because the page
    itself said it was there.
    """
    import io

    import pptx as _pptx

    deck = _pptx.Presentation(io.BytesIO(generate_and_download()))
    found = False
    for slide in deck.slides:
        lines = [sh.text_frame.text.strip() for sh in slide.shapes
                 if sh.has_text_frame and sh.text_frame.text.strip()]
        if not lines or "Stratifications" not in lines[0]:
            continue
        found = True
        page = "\n".join(lines)
        if "Composition and period movement" in page:
            # It claimed movement, so movement must be stated in words on it.
            assert any(w in page.lower() for w in
                       ("increased", "decreased", "moved", "contributed")), page
    assert found, "no stratification page in the pack — the test proved nothing"


# --------------------------------------------------------------------------- #
# 3. AN EMPTY PANEL IS NOT A LAYOUT.
# --------------------------------------------------------------------------- #

def test_4_no_empty_observations_panel_beside_watch_items(rich):
    """Catches: four inches of empty box labelled OBSERVATIONS / None recorded.

    The watchlist slide sized its columns to whether there were WATCH items and
    never to whether there were OBSERVATIONS. With watch items and no
    observations a funder got the page that says what needs attention, half of
    it occupied by an empty container.
    """
    text = deck_text(generate_and_download())
    assert "None recorded." not in text, (
        "an observations panel was drawn with nothing in it")


def test_5_a_short_watch_list_still_says_what_was_checked(rich):
    """Catches: one watch item floating in an otherwise blank page.

    A reader cannot tell "one thing was flagged" from "only one thing was
    looked at". Where the stack leaves room the governed checks are named,
    which is the same list the all-clear branch has always printed.
    """
    import io

    import pptx as _pptx
    from mi_agent_pptx.deck import DeckBuilder

    deck = _pptx.Presentation(io.BytesIO(generate_and_download()))
    for slide in deck.slides:
        lines = [sh.text_frame.text.strip() for sh in slide.shapes
                 if sh.has_text_frame and sh.text_frame.text.strip()]
        if not lines or "Watch Items" not in lines[0]:
            continue
        page = "\n".join(lines)
        if "CHECKS PERFORMED" not in page:
            pytest.skip("this book's watch stack fills the band")
        for check in DeckBuilder.GOVERNED_CHECKS:
            assert check in page, f"check not named: {check}"
        return


# --------------------------------------------------------------------------- #
# 4. WHAT THE SPRINT MUST NOT HAVE BROKEN.
# --------------------------------------------------------------------------- #

def test_6_additive_balance_stratification_still_renders(rich):
    """Catches: the additivity contract reaching the deck and suppressing the
    stratification bars.

    ``is_additive_measure`` was introduced for the React and Copilot surfaces.
    The deck's bar lists are balance sums — genuinely additive — and must be
    unaffected.
    """
    titles = slide_titles(generate_and_download())
    assert "Funded Stratifications" in titles, titles


def test_7_the_route_still_returns_a_downloadable_pack(simple):
    """Catches: any of the above breaking generation itself.

    The assertion that matters is in ``generate_and_download``: 202 accepted, a
    job that completes with no failed gate, and a file that starts PK.
    """
    content = generate_and_download()
    assert len(content) > 100_000, "a real pack is not this small"
    titles = slide_titles(content)
    # One book, one period: cover, exec position, exec summary, key measures,
    # stratifications, watch items, methodology. Everything else needs history,
    # a second book or a pipeline, and is omitted with a stated reason rather
    # than drawn empty.
    assert len(titles) == 7, titles
    assert "Data and Methodology" in titles, titles
