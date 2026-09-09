"""RECOVERY GATE A + B — the geography surfaces, executed.

A. `/mi/geo/exposure` must stay available on a book that carries ITL3 CODES.
B. the `portfolio_summary` route must not raise on any governed geography shape.

Both were certified live BEFORE the G6 chooser consolidation. Neither is a new
capability, so a change that loses either is a regression by definition and this
file exists to say so before a deploy rather than after one.
"""
from __future__ import annotations

import pandas as pd
import pytest

from .conftest import (ITL3_CODES_RECOGNISED, ITL3_CODES_UNRECOGNISED,
                       POSTCODES_UNMATCHABLE, REGION_NAMES, funded_frame,
                       install_series, raised, route, snapshot)

#: The bank's own portfolio-summary phrasings (C15). Quoted because the failure
#: was observed on these three and on no other summary phrasing — not because
#: the gate is tuned to them: any summary wording exercises the same route.
SUMMARY_QUESTIONS = (
    "Give me a summary of the portfolio.",
    "Where does the book stand at the moment — headline numbers, please.",
    "Portfolio overview.",
)

#: Every governed geography shape a live book has been observed to take. The
#: live book at the failing acceptance reported `basis = postcode_derived`,
#: i.e. NO ladder-recognised ITL3 column at all — a shape no fixture carried.
GEOGRAPHY_SHAPES = {
    "collateral region names": {"collateral_geography": REGION_NAMES},
    "obligor region names": {"geographic_region_obligor": REGION_NAMES},
    "harmonised taxonomy only": {"canonical_region_reporting": REGION_NAMES},
    "itl3 codes the ladder accepts":
        {"geographic_region_collateral_itl3": ITL3_CODES_RECOGNISED},
    "itl3 codes the ladder rejects":
        {"geographic_region_collateral_itl3": ITL3_CODES_UNRECOGNISED},
    "postcodes only": {"property_post_code": ("BS1 4DJ", "M1 2AB", "LS1 5AA")},
    "postcodes that resolve to nothing, plus itl3 codes":
        {"property_post_code": POSTCODES_UNMATCHABLE,
         "geographic_region_collateral_itl3": ITL3_CODES_UNRECOGNISED},
    "no geography of any kind": {},
}


# --------------------------------------------------------------------------- #
# A — the ITL3 exposure surface
# --------------------------------------------------------------------------- #
class TestTheItl3SurfaceStaysAvailable:
    """A CODE COLUMN IS CHOSEN BECAUSE GOVERNANCE SAYS IT IS ONE.

    The ITL3 field is named `geographic_region_collateral_itl3`. What it carries
    is codes — `TLK12` — and a code is not a place name. Deciding whether the
    column counts as geography by asking whether its VALUES look like places
    makes availability a property of which codes this particular book happens to
    hold: the demo book's codes pass that test and the live book's do not, which
    is the whole distance between a working geography view and an unavailable
    one.
    """

    def test_a_book_carrying_itl3_codes_has_an_available_geo_surface(self):
        from mi_agent_api import geo as geo_mod

        frame = pd.DataFrame({
            "current_outstanding_balance": [100_000.0, 250_000.0, 90_000.0],
            "geographic_region_collateral_itl3": list(ITL3_CODES_UNRECOGNISED),
            "property_post_code": list(POSTCODES_UNMATCHABLE),
        })

        result = geo_mod.exposure_by_itl3(frame)

        assert result["available"] is True, (
            "the book carries a governed ITL3 code column and a balance column, "
            "so the exposure surface is answerable; it reported "
            f"{result.get('reason')!r} instead")
        assert result["basis"] == "collateral"
        assert result["total"] == pytest.approx(440_000.0)

    def test_availability_does_not_depend_on_which_codes_the_book_holds(self):
        """The SAME column, the SAME shape, different code values.

        This is the regression stated at its narrowest: if these two disagree,
        availability is being decided by the region ladder's vocabulary rather
        than by governance.
        """
        from mi_agent_api import geo as geo_mod

        def surface(codes):
            return geo_mod.exposure_by_itl3(pd.DataFrame({
                "current_outstanding_balance": [100_000.0, 250_000.0, 90_000.0],
                "geographic_region_collateral_itl3": list(codes),
                "property_post_code": list(POSTCODES_UNMATCHABLE),
            }))

        accepted = surface(ITL3_CODES_RECOGNISED)
        rejected = surface(ITL3_CODES_UNRECOGNISED)

        assert accepted["available"] == rejected["available"], (
            "availability moved with the code VALUES: "
            f"{ITL3_CODES_RECOGNISED} -> {accepted['available']}, "
            f"{ITL3_CODES_UNRECOGNISED} -> {rejected['available']}")
        assert accepted["basis"] == rejected["basis"]

    def test_the_obligor_book_is_measured_on_the_obligor_codes(self):
        """A book that carries only the obligor codes is still answerable."""
        from mi_agent_api import geo as geo_mod

        result = geo_mod.exposure_by_itl3(pd.DataFrame({
            "current_outstanding_balance": [100_000.0, 250_000.0],
            "geographic_region_obligor_itl3": list(ITL3_CODES_UNRECOGNISED[:2]),
        }))

        assert result["available"] is True, result.get("reason")
        assert result["basis"] == "obligor"


# --------------------------------------------------------------------------- #
# B — the portfolio summary route
# --------------------------------------------------------------------------- #
class TestThePortfolioSummaryRouteNeverRaises:
    """A CLAIMED ROUTE THAT RAISES LOSES THE ANSWER.

    `chat_routing` treats entering a handler as the claim: a route that breaks
    partway through does not hand the question on, it returns the governed
    execution-failure envelope. That is the right behaviour and it is also why a
    route exception is never a quiet degradation — it is a lost answer, every
    time, for every phrasing the route claims.

    KNOWN LIMITATION, STATED RATHER THAN HIDDEN. The live failure of Q043-Q045
    is NOT reproduced by any shape below. See the module note in
    `docs`-less form here: the matrix is the strongest offline statement of the
    invariant that could be built, and it currently PASSES on the build that
    fails live, so it does not yet gate that defect.
    """

    @pytest.mark.parametrize("shape", sorted(GEOGRAPHY_SHAPES))
    @pytest.mark.parametrize("question", SUMMARY_QUESTIONS)
    def test_the_route_answers_or_defers_but_never_breaks(
            self, monkeypatch, semantics, shape, question):
        geography = GEOGRAPHY_SHAPES[shape]
        early = funded_frame(rows=50, balance=8_903_225.07,
                             origination="2025-11-05", geography=geography)
        current = funded_frame(rows=958, balance=137_854_092.0,
                               origination="2026-06-10", geography=geography)
        install_series(monkeypatch, [snapshot("2025-11-30", early),
                                     snapshot("2026-06-30", current)])

        envelope = route(question, current, semantics)

        assert not raised(envelope), (
            f"the portfolio_summary route claimed {question!r} on a book whose "
            f"geography is {shape!r} and then failed while running")

    @pytest.mark.parametrize("early_shape", sorted(GEOGRAPHY_SHAPES))
    def test_the_route_survives_an_earlier_snapshot_of_another_shape(
            self, monkeypatch, semantics, early_shape):
        """The summary reads its region column from the EARLIEST frame and its
        regional exposure from the CURRENT one. A book whose early snapshot was
        prepared under a different schema therefore puts two different geography
        shapes through one answer.
        """
        early = funded_frame(rows=50, balance=8_903_225.07,
                             origination="2025-11-05",
                             geography=GEOGRAPHY_SHAPES[early_shape])
        current = funded_frame(rows=958, balance=137_854_092.0,
                               origination="2026-06-10",
                               geography={"collateral_geography": REGION_NAMES})
        install_series(monkeypatch, [snapshot("2025-11-30", early),
                                     snapshot("2026-06-30", current)])

        envelope = route("Portfolio overview.", current, semantics)

        assert not raised(envelope), (
            f"the summary failed while running when the earlier snapshot's "
            f"geography was {early_shape!r} and the current snapshot's was "
            f"collateral region names")
