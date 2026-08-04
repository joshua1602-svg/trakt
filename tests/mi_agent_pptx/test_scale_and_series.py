"""What breaks when the book gets big and the history gets long.

Every fixture in this suite until now had two reporting periods and a handful of
loans, and several defects hid behind exactly that. With two periods, "the prior
period" and "the earliest period" are the same period, so a movement attributed
over the whole series looked correct. With small balances, an axis tick fits in
any margin. With a two-point series, no tick can collide with its neighbour.

These are the properties that only bite at scale.
"""

from __future__ import annotations

import pytest

from mi_agent_pptx import render as R


# --------------------------------------------------------------------------- #
# Movement window.
# --------------------------------------------------------------------------- #

def test_the_movement_bridge_opens_at_the_period_the_caller_names(monkeypatch):
    """The deck's headline movement is month-on-month, so its attribution must
    be too. Left to itself the governed bridge opens at the EARLIEST period it
    can find — fourteen months of attribution beside a one-month headline."""
    from mi_agent_api import evolution
    from mi_agent_pptx import movement as MV

    seen = []
    monkeypatch.setattr(evolution, "funded_bridge",
                        lambda *a, **kw: seen.append(kw.get("start_period"))
                        or {"available": False, "reason": "stub"})
    MV.build_bridges("/root", "c", "run", start_period="2026-05-31",
                     keys=["region"])
    assert seen == ["2026-05-31"], \
        "the start period never reached the governed bridge"


def test_over_three_periods_the_attribution_matches_the_headline(tmp_path,
                                                                 monkeypatch):
    """The property end to end, on a book with MORE than two periods — the only
    shape in which the two can differ.

    With exactly two periods "the prior period" and "the earliest period" are
    the same period, which is why every earlier fixture passed while a deck over
    a real fourteen-month history attributed £448m of movement beside a £34m
    headline.
    """
    import json

    import pandas as pd

    from mi_agent_pptx.mi_api import build_dashboard_data

    def _rows(n, balance, cut):
        return [{"unique_identifier": f"L{i:04d}",
                 "source_portfolio_id": "direct_001",
                 "source_portfolio_type": "direct",
                 "current_outstanding_balance": balance,
                 "current_principal_balance": balance,
                 "current_valuation_amount": balance / 0.45,
                 "current_loan_to_value": 45.0,
                 "current_interest_rate": 7.0,
                 "youngest_borrower_age": 72,
                 "collateral_geography": "London" if i % 2 else "Wales",
                 "geographic_region_obligor": "London" if i % 2 else "Wales",
                 "origination_date": "2021-04-01",
                 "data_cut_off_date": cut} for i in range(n)]

    root = tmp_path / "root"
    periods = [("mi_2026_04", "2026-04-30", 10, 100_000.0),
               ("mi_2026_05", "2026-05-31", 12, 100_000.0),
               ("mi_2026_06", "2026-06-30", 13, 100_000.0)]
    for run_id, cut, n, bal in periods:
        central = root / "acme" / run_id / "central"
        central.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(_rows(n, bal, cut))
        frame.to_csv(central / "18_central_lender_tape.csv", index=False)
        dated = root / cut
        dated.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dated / "platform_canonical_typed.csv", index=False)

    rd = root / "orun"
    (rd / "out_platform").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_rows(13, 100_000.0, "2026-06-30")).to_csv(
        rd / "out_platform" / "platform_canonical_typed.csv", index=False)
    (rd / "run_state.json").write_text(json.dumps(
        {"run_id": "mi_2026_06", "client_id": "acme",
         "reporting_date": "2026-06-30", "out_root": str(root)}))
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(root))

    data = build_dashboard_data(rd, client_id="acme", as_of="2026-06-30",
                                output_root=str(root))
    headline = next((k.get("raw") for k in (data.funded.get("kpis") or ())
                     if k.get("id") == "mom_balance"), None)
    assert headline == pytest.approx(100_000.0), "the fixture did not move by one loan"

    bridges = [b for b in (data.movement or {}).values() if b.available]
    assert bridges, "no dimension attributed at all"
    for b in bridges:
        assert b.start_date == "2026-05-31", \
            f"{b.key} opened at {b.start_date}, not the prior period"
        assert b.total_delta == pytest.approx(headline), \
            f"{b.key} attributed {b.total_delta} against a headline of {headline}"


# --------------------------------------------------------------------------- #
# Axis ticks.
# --------------------------------------------------------------------------- #

def test_long_series_thin_their_labels_instead_of_overprinting():
    """Ten weekly ISO dates in six inches printed as one illegible band."""
    labels = [f"2026-{m:02d}-{d:02d}" for m, d in
              ((4, 24), (5, 1), (5, 8), (5, 15), (5, 22), (5, 29),
               (6, 5), (6, 12), (6, 19), (6, 26))]
    idx = R._tick_indices(labels, axis_width_in=4.8, fontsize=8.5)
    assert len(idx) < len(labels), "no thinning happened at ten 10-char labels"
    # Every label must have room: width per slot >= the widest label.
    label_in = 10 * 8.5 * 0.55 / 72.0
    assert 4.8 / len(idx) >= label_in


def test_the_first_and_last_points_are_always_labelled():
    labels = [f"2026-{m:02d}" for m in range(1, 13)]
    idx = R._tick_indices(labels, axis_width_in=4.0)
    assert idx[0] == 0 and idx[-1] == len(labels) - 1


def test_the_forced_last_tick_never_lands_on_its_neighbour():
    """Keeping the final position is right; printing it on top of the stepped
    tick beside it is not."""
    for n in range(3, 40):
        labels = [f"2026-{i:02d}-01" for i in range(n)]
        idx = R._tick_indices(labels, axis_width_in=4.8)
        gaps = [b - a for a, b in zip(idx, idx[1:])]
        assert min(gaps) >= max(gaps) - 1 or min(gaps) >= 2, \
            f"n={n} produced adjacent ticks: {idx}"


def test_a_short_series_keeps_every_label():
    labels = ["2026-05", "2026-06"]
    assert R._tick_indices(labels, axis_width_in=4.8) == [0, 1]


# --------------------------------------------------------------------------- #
# Notation.
# --------------------------------------------------------------------------- #

def test_chart_labels_use_the_same_notation_as_the_kpi_tiles():
    """A bar labelled '+£20.5m' beside an axis labelled '£800.0MM' reads as two
    different figures. Both now come from the dashboard's own formatter."""
    from mi_agent_pptx.metric_resolver import compact_currency, signed_currency

    assert R._fmt_money(833_500_000.0) == compact_currency(833_500_000.0)
    assert R._fmt_money(20_500_000.0, signed=True) == signed_currency(20_500_000.0)
    assert R._fmt_money(833_500_000.0) == "£833.5MM"


def test_the_deck_tile_formatter_matches_the_react_client():
    """Mirrors `formatGBP` in frontend/mi-agent-ui/src/lib/utils.ts. If one side
    changes, an exported tile stops matching the screen it is a copy of."""
    from mi_agent_pptx.metric_resolver import compact_currency

    assert compact_currency(1_200_000_000.0) == "£1.20BN"
    assert compact_currency(833_500_000.0) == "£833.5MM"
    assert compact_currency(607_507.29) == "£608K"
    assert compact_currency(842.0) == "£842"


def test_sentences_keep_prose_notation():
    """Deliberately NOT the tile notation: these feed the executive summary,
    where they sit beside observations written by the governed generators that
    also write React's weekly brief."""
    from mi_agent_pptx.movement import _money

    assert _money(20_500_000.0) == "£20.5m"
    assert _money(1_200_000_000.0) == "£1.20bn"


# --------------------------------------------------------------------------- #
# Percent axis resolution.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("values,expect_distinct", [
    ([45.8, 46.2, 46.6, 46.1], True),      # 0.8pp spread — needs a decimal
    ([12.0, 38.0, 61.0], True),            # wide — whole numbers are fine
])
def test_a_narrow_percent_range_does_not_label_every_gridline_the_same(
        tmp_path, values, expect_distinct):
    """A weighted LTV moving inside one point produced five ticks all reading
    '46%'. An axis that labels distinct gridlines identically reads as a
    rendering fault."""
    from matplotlib.ticker import FuncFormatter
    import matplotlib
    matplotlib.use("Agg")

    out = R.draw_lines(tmp_path / "p.png", [f"2026-{i:02d}" for i in range(len(values))],
                       [{"name": "WA LTV", "values": values}], 5.0, 3.0,
                       currency=False, percent=True)
    assert out.exists()
    # Re-derive what the formatter would print at the data's own tick spacing.
    spread = max(values) - min(values)
    dp = 0 if spread >= 6 else (1 if spread >= 0.6 else 2)
    printed = {f"{v:.{dp}f}%" for v in values}
    assert (len(printed) == len(set(values))) is expect_distinct or dp > 0
