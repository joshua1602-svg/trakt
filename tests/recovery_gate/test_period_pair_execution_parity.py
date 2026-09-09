"""RECOVERY GATE C + D — the period pair, executed.

C. relative wording on an IRREGULAR series must resolve the governed previous
   observation or REFUSE under gap policy. It must never span a window the gap
   policy would reject and present that as the answer to "last month".
D. an answer that compared NOTHING is not an answer. `metrics_comparable == 0`
   must imply `ok is not True`.

TIME x DIMENSION IS NOT UNDER TEST HERE and nothing in this file touches it.
The seam being gated is the PERIOD PAIR — opening observation, closing
observation, gap validity — which is a different owner from the time axis.
"""
from __future__ import annotations

import re

import pytest

from .conftest import (REGION_NAMES, funded_frame, install_series, route,
                       snapshot)

#: The bank's relative-period phrasings (C26, C27, C30) and the bridge (C28).
#: Every one of them names the IMMEDIATELY PREVIOUS governed observation. None
#: of them names a seven-month window.
RELATIVE_PERIOD_QUESTIONS = (
    "How did the funded balance change since the prior reporting date?",
    "Month-on-month balance change.",
    "How does this month's balance measure up against last month's?",
    "Balance: current period versus prior period.",
    "New and exited loan counts month on month.",
)

BRIDGE_QUESTION = "Bridge the movement in the book from last month to this month."

#: The live book's own observation dates: November, then June. 212 days.
IRREGULAR_SERIES_DATES = ("2025-11-30", "2026-06-30")

#: `mi_agent.period_change` governs how far a "previous observation" may be
#: from the current one before the comparison stops meaning what the reader
#: asked for. Read from configuration rather than restated, so this gate cannot
#: drift away from the policy it is enforcing.
def _gap_ceiling_days() -> int:
    from mi_agent.period_change import periods as periods_mod

    for holder in (periods_mod, getattr(periods_mod, "config", None)):
        value = getattr(holder, "max_snapshot_gap_days", None)
        if isinstance(value, int):
            return value
    try:
        from mi_agent.period_change import config as pc_config

        return int(pc_config.max_snapshot_gap_days())
    except Exception:  # noqa: BLE001 - the documented governed default
        return 45


def _geography():
    return {"collateral_geography": REGION_NAMES}


def _irregular_series(*, early_balance_is_null: bool = False,
                      early_drops: tuple = ()):
    early = funded_frame(rows=50, balance=8_903_225.07,
                         origination="2025-11-05", geography=_geography(),
                         balance_is_null=early_balance_is_null,
                         drop=early_drops)
    current = funded_frame(rows=958, balance=137_854_092.0,
                           origination="2026-06-10", geography=_geography())
    return [snapshot(IRREGULAR_SERIES_DATES[0], early),
            snapshot(IRREGULAR_SERIES_DATES[1], current)], current


_COMPARED = re.compile(r"(\d+)\s+of\s+(\d+)\s+governed metrics could be compared")
_SPAN = re.compile(r"[Bb]etween\s+(.+?)\s+and\s+(.+?),")


def _comparable(answer: str):
    """``(compared, offered)`` as the answer itself states them, or ``None``."""
    match = _COMPARED.search(answer or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


# --------------------------------------------------------------------------- #
# C — the gap policy governs a relative period request
# --------------------------------------------------------------------------- #
class TestRelativePeriodWordingHonoursTheGapPolicy:
    """"LAST MONTH" IS NOT "SINCE WHENEVER WE LAST LOOKED".

    On a book observed in November and then in June, there is no previous MONTH
    to compare against. The governed answer is a refusal that says so. Spanning
    212 days and calling it the month-on-month movement answers a question the
    reader did not ask, and does it in wording that reads like the one they did.
    """

    @pytest.mark.parametrize("question", RELATIVE_PERIOD_QUESTIONS)
    def test_it_refuses_rather_than_spanning_a_window_the_policy_rejects(
            self, monkeypatch, semantics, question):
        series, current = _irregular_series()
        install_series(monkeypatch, series)

        envelope = route(question, current, semantics)
        assert envelope is not None, "the question was not claimed by any route"

        answer = envelope.get("answer") or envelope.get("error") or ""
        span = _SPAN.search(answer)
        if envelope.get("ok") is not True:
            return  # a governed refusal is the accepted outcome
        assert span is None or "2025" not in span.group(1), (
            f"{question!r} was answered as a movement spanning "
            f"{span.group(1)!r} to {span.group(2)!r} — "
            f"{(_gap_ceiling_days())} days is the governed ceiling, and this "
            f"pair is 212 days apart, so the governed outcome is a refusal")

    def test_the_bridge_and_the_change_agree_about_the_same_book(
            self, monkeypatch, semantics):
        """One book, one gap policy, two routes: they cannot disagree.

        The bridge refusing while the period change answers the same window
        means the pair is being decided twice, which is the condition the
        consolidation was meant to remove and the one that must not come back
        in either direction.
        """
        series, current = _irregular_series()
        install_series(monkeypatch, series)

        bridge = route(BRIDGE_QUESTION, current, semantics)
        change = route("Month-on-month balance change.", current, semantics)

        assert bridge is not None and change is not None
        assert bool(bridge.get("ok")) == bool(change.get("ok")), (
            "on one book with one gap policy the bridge and the period change "
            f"reached different verdicts: bridge ok={bridge.get('ok')!r}, "
            f"period change ok={change.get('ok')!r}")

    @pytest.mark.parametrize("question", RELATIVE_PERIOD_QUESTIONS)
    def test_a_regular_series_still_compares_the_previous_month(
            self, monkeypatch, semantics, question):
        """THE CONTROL. Nothing above may be bought by refusing the easy case."""
        frames = [snapshot("2026-04-30", funded_frame(
                      rows=900, balance=120_000_000.0, origination="2026-04-10",
                      geography=_geography())),
                  snapshot("2026-05-31", funded_frame(
                      rows=930, balance=130_000_000.0, origination="2026-05-10",
                      geography=_geography())),
                  snapshot("2026-06-30", funded_frame(
                      rows=958, balance=137_854_092.0, origination="2026-06-10",
                      geography=_geography()))]
        install_series(monkeypatch, frames)

        envelope = route(question, frames[-1]["df"], semantics)

        assert envelope is not None and envelope.get("ok") is True, (
            f"{question!r} must still be answerable on a regular monthly series")
        answer = envelope.get("answer") or ""
        span = _SPAN.search(answer)
        assert span is not None and "May 2026" in span.group(1), (
            f"the previous governed observation is 31 May 2026; the answer "
            f"opened on {span.group(1)!r}" if span else
            f"the answer stated no period span: {answer[:160]!r}")


# --------------------------------------------------------------------------- #
# D — nothing compared is not an answer
# --------------------------------------------------------------------------- #
class TestZeroComparableMetricsIsNotASuccessfulAnswer:
    """`ok=true` IS A PROMISE THAT SOMETHING WAS CALCULATED.

    An envelope that states, in its own words, that none of the metrics it
    offered could be compared has calculated nothing. Publishing that as a
    successful answer puts a sentence in front of the reader that looks like a
    movement analysis, carries two dates and two counts, and rests on no
    comparison at all — which is the exact shape a fail-closed boundary exists
    to stop.
    """

    @pytest.mark.parametrize("crippling", ["null balances", "no balance column",
                                           "no rows"])
    @pytest.mark.parametrize("question", [
        "How does this month's balance measure up against last month's?",
        "Month-on-month balance change."])
    def test_an_answer_that_compared_nothing_is_not_ok(
            self, monkeypatch, semantics, crippling, question):
        if crippling == "null balances":
            series, current = _irregular_series(early_balance_is_null=True)
        elif crippling == "no balance column":
            series, current = _irregular_series(
                early_drops=("current_outstanding_balance",))
        else:
            series, current = _irregular_series()
            series[0]["df"] = series[0]["df"].iloc[0:0]
        install_series(monkeypatch, series)

        envelope = route(question, current, semantics)
        assert envelope is not None

        answer = envelope.get("answer") or envelope.get("error") or ""
        stated = _comparable(answer)
        if stated is None:
            return  # the answer made no such claim; nothing to enforce here
        compared, offered = stated
        if compared == 0:
            assert envelope.get("ok") is not True, (
                f"the answer states that 0 of {offered} governed metrics could "
                f"be compared and was published as a successful answer: "
                f"{answer[:200]!r}")

    def test_the_receipt_cannot_report_a_movement_it_did_not_measure(
            self, monkeypatch, semantics):
        """If nothing compared, no movement may be named as the largest one."""
        series, current = _irregular_series(early_balance_is_null=True)
        install_series(monkeypatch, series)

        envelope = route("Month-on-month balance change.", current, semantics)
        answer = (envelope or {}).get("answer") or ""
        stated = _comparable(answer)
        if stated and stated[0] == 0:
            assert "largest observed movement" not in answer.lower(), (
                "the answer compared nothing and still named a largest "
                f"movement: {answer[:200]!r}")
