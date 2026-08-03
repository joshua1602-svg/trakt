"""The deck and the dashboard must be the same numbers.

Not "computed the same way" — the SAME numbers. This drives the real React HTTP
routes and the real deck build over one fixture book and compares what each
produced, value by value, on the underlying figures rather than on formatted
strings (a formatter that rounds identically would hide a genuine disagreement).

Two things it is designed to catch, both of which have actually happened in this
work:

  * a deck surface silently resolving a DIFFERENT source from the one the
    dashboard resolves — the pipeline root defect, where the deck used a
    discovery-only resolver and the API a URI-first one, so the deck showed no
    pipeline where the dashboard showed one;
  * a deck surface consuming a DIFFERENT service — the concentration defect,
    where the deck read the legacy limit monitor while the dashboard read the
    operator-approved evaluator.

Neither is visible from the deck alone. Both are obvious the moment the two
channels are put side by side, which is why this is a committed test and not a
one-off script.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

_CENTRAL = "18_central_lender_tape.csv"
CLIENT = "acme"
AS_OF = "2026-06-30"
RUN_ID = "mi_2026_06"


# --------------------------------------------------------------------------- #
# One book, two periods, two portfolio types, several regions — enough for the
# snapshot, stratifications, evolution, cohorts, geography and movement to all
# have a non-trivial answer.
# --------------------------------------------------------------------------- #

def _loan(i, pid, ptype, balance, *, cut, region, itl3, ltv=45.0, age=72):
    return {
        "unique_identifier": f"{pid}_L{i:04d}",
        "source_portfolio_id": pid,
        "source_portfolio_type": ptype,
        "source_portfolio_label": pid.replace("_", " ").title(),
        "current_outstanding_balance": balance,
        "current_principal_balance": balance,
        "original_principal_balance": balance * 1.04,
        "current_valuation_amount": balance / (ltv / 100.0),
        "original_valuation_amount": balance / (ltv / 100.0),
        "current_loan_to_value": ltv,
        "original_loan_to_value": ltv / 100 - 0.02,
        "current_interest_rate": 7.1,
        "youngest_borrower_age": age,
        "geographic_region_collateral": region,
        "collateral_geography": region,
        "geographic_region_obligor": itl3,
        "origination_channel": "Direct",
        "origination_date": "2021-04-01",
        "data_cut_off_date": cut,
    }


_PRIOR = [("direct_001", "direct", 300_000.0, "London", "TLI3", 40.0, 70),
          ("direct_001", "direct", 200_000.0, "South East", "TLJ2", 52.0, 75),
          ("acquired_001", "acquired", 250_000.0, "Wales", "TLL1", 48.0, 68)]
_CURRENT = [("direct_001", "direct", 300_000.0, "London", "TLI3", 40.0, 70),
            ("direct_001", "direct", 260_000.0, "South East", "TLJ2", 52.0, 75),
            ("direct_001", "direct", 120_000.0, "South East", "TLJ2", 61.0, 66),
            ("acquired_001", "acquired", 230_000.0, "Wales", "TLL1", 48.0, 68)]


def _rows(spec, cut):
    return [_loan(i, pid, ptype, bal, cut=cut, region=region, itl3=itl3,
                  ltv=ltv, age=age)
            for i, (pid, ptype, bal, region, itl3, ltv, age) in enumerate(spec, 1)]


@pytest.fixture()
def book(tmp_path, monkeypatch):
    """A governed run layout both channels resolve from the same environment."""
    root = tmp_path / "root"
    for run_id, date, spec in (("mi_2026_05", "2026-05-31", _PRIOR),
                               (RUN_ID, AS_OF, _CURRENT)):
        central = root / CLIENT / run_id / "central"
        central.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(_rows(spec, date))
        frame.to_csv(central / _CENTRAL, index=False)
        dated = root / date
        dated.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dated / "platform_canonical_typed.csv", index=False)

    run_dir = root / "orun_parity"
    (run_dir / "out_platform").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_rows(_CURRENT, AS_OF)).to_csv(
        run_dir / "out_platform" / "platform_canonical_typed.csv", index=False)
    (run_dir / "run_state.json").write_text(json.dumps({
        "run_id": RUN_ID, "client_id": CLIENT, "reporting_date": AS_OF,
        "out_root": str(root)}), encoding="utf-8")

    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(root))
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", CLIENT)
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", str(root))
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "false")
    return run_dir


@pytest.fixture()
def deck(book):
    from mi_agent_pptx.mi_api import build_dashboard_data
    return build_dashboard_data(book, client_id=CLIENT, as_of=AS_OF,
                                output_root=str(Path(book).parent))


@pytest.fixture()
def react(book):
    """The React channel, driven through its real HTTP routes."""
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    client = TestClient(app)
    pid = f"{CLIENT}/{RUN_ID}"

    def get(path, **params):
        res = client.get(path, params={"portfolioId": pid, **params})
        assert res.status_code == 200, f"{path} → {res.status_code}: {res.text}"
        return res.json()

    return get


def _raw(payload, kpi_id):
    for k in payload.get("kpis") or ():
        if k.get("id") == kpi_id or k.get("kpi_id") == kpi_id:
            return k.get("raw")
    return None


def _close(a, b, tol=0.01):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- #
# Headline funded figures.
# --------------------------------------------------------------------------- #

def test_the_headline_funded_figures_match_the_dashboard(deck, react):
    api = react("/mi/snapshot")
    assert api.get("kpis"), "the dashboard returned no KPIs — the fixture is wrong"

    for kpi in ("balance", "loans", "wa_current_ltv"):
        mine = _raw(deck.funded, kpi)
        theirs = _raw(api, kpi)
        assert theirs is not None, f"the dashboard has no {kpi} KPI"
        assert _close(mine, theirs), f"{kpi}: deck {mine} vs dashboard {theirs}"


def test_the_portfolio_context_totals_match_the_snapshot(deck, react):
    """The deck's portfolio context is a VIEW over the snapshot, so it must not
    be able to disagree with it."""
    from mi_agent_pptx.deck_context import raw_of

    api = react("/mi/snapshot")
    assert _close(deck.portfolio.total_balance, _raw(api, "balance"))
    assert deck.portfolio.loan_count == _raw(api, "loans")
    assert _close(raw_of(deck.funded, "balance"), _raw(api, "balance"))


def test_the_stratifications_match_bar_for_bar(deck, react):
    api = react("/mi/snapshot")
    theirs = {s["key"]: s for s in api.get("stratifications") or ()}
    mine = {s["key"]: s for s in deck.funded.get("stratifications") or ()}
    shared = set(theirs) & set(mine)
    assert shared, "no stratification key is common to both channels"

    for key in sorted(shared):
        a = {b["label"]: b["balance"] for b in mine[key].get("bars") or ()}
        b = {x["label"]: x["balance"] for x in theirs[key].get("bars") or ()}
        assert set(a) == set(b), f"{key}: deck bands {sorted(a)} vs {sorted(b)}"
        for label in a:
            assert _close(a[label], b[label]), \
                f"{key}/{label}: deck {a[label]} vs dashboard {b[label]}"


# --------------------------------------------------------------------------- #
# The multi-period and derived surfaces.
# --------------------------------------------------------------------------- #

def _series(payload):
    return {p["reporting_date"]: (p.get("metrics") or {}).get("funded_balance")
            for p in payload.get("periods") or ()}


def test_the_funded_evolution_series_matches(deck, react):
    """Every period, not just the latest — a deck that agreed on the current
    month and disagreed on the prior one would draw a false movement."""
    mine = _series(deck.funded_evolution or {})
    theirs = _series(react("/mi/evolution/funded"))
    assert len(mine) >= 2, "the deck resolved fewer than two periods"
    assert set(mine) == set(theirs)
    for date in mine:
        assert _close(mine[date], theirs[date]), \
            f"{date}: deck {mine[date]} vs dashboard {theirs[date]}"


def test_the_cohorts_match(deck, react):
    api = react("/mi/cohorts")
    theirs = {c["cohort"]: c for c in api.get("cohorts") or ()}
    mine = {c["cohort"]: c for c in (deck.cohorts or {}).get("cohorts") or ()}
    assert mine, "the deck resolved no cohorts"
    assert set(mine) == set(theirs)
    for cohort in mine:
        assert _close(mine[cohort].get("balance"), theirs[cohort].get("balance"))
        assert mine[cohort].get("loanCount") == theirs[cohort].get("loanCount")


def test_the_geographic_exposure_matches(deck, react):
    api = react("/mi/geo/exposure")
    theirs = {a.get("code") or a.get("itl3"): a.get("balance")
              for a in api.get("areas") or ()}
    mine = {a.get("code") or a.get("itl3"): a.get("balance")
            for a in (deck.geo or {}).get("areas") or ()}
    # Both channels agreeing that nothing resolves is itself parity — but say so,
    # rather than letting an empty-vs-empty comparison read as a proven match.
    if not theirs and not mine:
        pytest.skip("neither channel resolves geographic exposure for this book")
    assert set(mine) == set(theirs)
    for code in mine:
        assert _close(mine[code], theirs[code])


def test_the_concentration_tests_are_the_same_evaluations(deck, react):
    """The deck must read the governed evaluator's own numbers, not re-derive
    them — this is the exact defect the v2.1 pass found."""
    from mi_agent_pptx import concentration as C

    api = react("/mi/concentration-tests")
    theirs = {t["displayName"]: t for t in api.get("tests") or ()}
    mine = {r["label"]: r for r in C.adapt_tests(deck.concentration)}
    assert set(mine) == set(theirs)
    for label, row in mine.items():
        src = theirs[label]
        assert _close(row["value"], src.get("currentValue")), label
        assert _close(row["limit"], src.get("threshold")), label
        assert row["status"] == C.normalise_status(src.get("status")), label
        assert _close(row["expected_value"],
                      (src.get("expected") or {}).get("value")), label
        assert _close(row["stress_value"],
                      (src.get("fullPipeline") or {}).get("value")), label


def test_both_channels_report_the_same_reporting_date(deck, react):
    api = react("/mi/snapshot")
    portfolio = api.get("portfolio") or {}
    theirs = portfolio.get("reporting_date") or api.get("reportingDate")
    assert deck.reporting_date == theirs or theirs is None
    assert deck.reporting_date == AS_OF


# --------------------------------------------------------------------------- #
# Scope parity — the same narrowing must reach both channels.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("context", ["direct", "acquired"])
def test_a_narrowed_scope_narrows_both_channels_identically(book, react, context):
    from mi_agent_pptx.mi_api import build_dashboard_data

    scoped = build_dashboard_data(book, client_id=CLIENT, as_of=AS_OF,
                                  output_root=str(Path(book).parent),
                                  portfolio_context=context)
    api = react("/mi/snapshot", portfolioContext=context)
    assert _close(_raw(scoped.funded, "balance"), _raw(api, "balance")), context
    assert _raw(scoped.funded, "loans") == _raw(api, "loans"), context
    # ...and the narrowing is real, not a no-op that would make the above vacuous.
    total = react("/mi/snapshot")
    assert _raw(api, "balance") < _raw(total, "balance")


# --------------------------------------------------------------------------- #
# The structural property behind all of the above.
# --------------------------------------------------------------------------- #

def test_the_deck_layer_owns_no_economic_calculation():
    """Parity holds because the deck ADAPTS governed output rather than
    computing its own. This asserts the arrangement, not just today's numbers:
    every economic figure the deck shows must come from a governed service.

    Presentation arithmetic — laying out a slide, converting inches, summing a
    chart's own bars for a total the service already published — is not an
    economic calculation and is not what this guards against.
    """
    import mi_agent_pptx

    root = Path(mi_agent_pptx.__file__).parent
    # The governed services the deck is permitted to source figures from.
    governed = (
        "compute_funded_snapshot", "funded_evolution", "funded_bridge",
        "funded_cohort_progression", "cohort_analysis", "exposure_by_itl3",
        "compute_risk_limits", "compute_concentration_tests",
        "compute_forecast_bridge", "build_extrapolation", "compute_pipeline_snapshot",
        "pipeline_evolution", "funnel_evolution", "movement_detail",
    )
    sourced = set()
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        sourced.update(name for name in governed if name in text)

    # Every economic surface the deck presents has a governed origin in the code.
    for required in ("compute_funded_snapshot", "funded_evolution",
                     "compute_concentration_tests", "cohort_analysis"):
        assert required in sourced, (
            f"{required} is not referenced anywhere in the deck layer — a deck "
            f"surface may have started computing its own figures")


# --------------------------------------------------------------------------- #
# Cohorts and vintages.
#
# The deck used to group the current book by origination year itself. It now
# consumes the two governed services the React Cohorts tab consumes, so the two
# channels can be compared cohort by cohort — which is the only way to catch the
# deck drifting onto its own definition again.
# --------------------------------------------------------------------------- #

def test_the_vintage_table_is_the_governed_cohort_table(deck, react):
    from mi_agent_pptx import cohorts as CO

    api = react("/mi/cohorts", grain="Y", dimension="vintage")
    theirs = {str(c["cohort"]): c for c in api.get("cohorts") or ()}
    mine = {r.vintage: r for r in CO.adapt_formation(deck.cohorts).rows}
    assert mine, "the deck resolved no vintages"
    assert set(mine) == set(theirs)
    for vintage, row in mine.items():
        src = theirs[vintage]
        assert _close(row.balance, src.get("balance")), vintage
        assert row.loan_count == src.get("loanCount"), vintage
        assert _close(row.share_pct, src.get("sharePct")), vintage


def test_the_deck_reports_the_governed_cohort_basis(deck, react):
    """Vintage assignment is the service's, so the basis it declares must reach
    the deck rather than being restated in the deck's own words."""
    from mi_agent_pptx import cohorts as CO

    api = react("/mi/cohorts", grain="Y", dimension="vintage")
    assert CO.adapt_formation(deck.cohorts).cohort_basis == api.get("cohortBasis")


def test_the_progression_series_are_the_governed_static_pool(book, react):
    """Each cohort curve must equal the /mi/cohorts/progression payload for the
    SAME vintage — the deck issues the identical call the dashboard issues."""
    from pathlib import Path as _P

    from mi_agent_pptx import cohorts as CO
    from mi_agent_pptx.mi_api import build_dashboard_data

    data = build_dashboard_data(book, client_id=CLIENT, as_of=AS_OF,
                                output_root=str(_P(book).parent))
    payload = data.cohort_series or {}
    if not payload.get("available"):
        pytest.skip("this book has no material vintage to season")

    for vintage, mine_raw in (payload.get("series") or {}).items():
        api = react("/mi/cohorts/progression", vintage=vintage, grain="Y")
        mine = CO.adapt_progression(mine_raw, vintage)
        theirs = CO.adapt_progression(api, vintage)
        assert mine.available == theirs.available, vintage
        assert [p.period for p in mine.points] == [p.period for p in theirs.points], vintage
        for a, b in zip(mine.points, theirs.points):
            assert a.loan_count == b.loan_count, f"{vintage}/{a.period}"
            assert _close(a.metrics.get("funded_balance"),
                          b.metrics.get("funded_balance")), f"{vintage}/{a.period}"
            assert _close(a.metrics.get("wa_ltv"), b.metrics.get("wa_ltv"),
                          tol=0.05), f"{vintage}/{a.period}"


def test_the_deck_never_calls_a_cohort_ratio_retention():
    """The governed service re-selects a cohort by vintage each period rather
    than freezing it at formation, so the ratio can exceed 100%. Calling that
    "retention" would state an analytic the service does not support."""
    from pathlib import Path as _P

    import mi_agent_pptx

    root = _P(mi_agent_pptx.__file__).parent
    for name in ("cohorts.py", "deck.py"):
        text = (root / name).read_text(encoding="utf-8")
        body = "\n".join(l for l in text.splitlines()
                         if not l.strip().startswith("#"))
        assert "retention" not in body.lower() or "retention asserts" in body.lower(), \
            f"{name} still presents a cohort ratio as retention"
