"""Currency resolution + money formatting (C2)."""
from __future__ import annotations

import pandas as pd

from mi_agent_api import currency as c


def _reset():
    c.set_currency("GBP")


def test_default_is_gbp():
    _reset()
    assert c.current_code() == "GBP"
    assert c.current_symbol() == "£"
    assert c.format_money(1_500_000) == "£1.5m"
    assert c.format_money(1_500_000, suffixes=("BN", "MM", "K")) == "£1.5MM"
    assert c.format_money(2_300_000_000) == "£2.30bn"
    assert c.format_money(750) == "£750"
    assert c.format_money(None) == "—"


def test_symbol_for_known_and_unknown():
    assert c.symbol_for("EUR") == "€"
    assert c.symbol_for("USD") == "$"
    assert c.symbol_for("gbp") == "£"          # case-insensitive
    assert c.symbol_for("PLN") == "PLN "       # unknown -> code + space
    assert c.symbol_for(None) == "£"


def test_resolves_currency_from_the_tape():
    _reset()
    df = pd.DataFrame({"exposure_currency_denomination": ["EUR", "EUR", "GBP"]})
    assert c.resolve_and_set(df) == "EUR"      # most common value wins
    assert c.current_symbol() == "€"
    assert c.format_money(2_000_000) == "€2.0m"
    _reset()


def test_falls_back_to_config_then_default():
    _reset()
    # No tape column -> client config.
    assert c.resolve_currency_code(pd.DataFrame({"x": [1]}),
                                   client_config={"currency": "usd"}) == "USD"
    # Nothing at all -> default GBP.
    assert c.resolve_currency_code(None) == "GBP"
    _reset()


def test_resolution_never_raises_on_bad_data():
    _reset()
    # All-null / blank currency column falls through to the default.
    df = pd.DataFrame({"exposure_currency_denomination": [None, "", "  "]})
    assert c.resolve_currency_code(df) == "GBP"
    _reset()


def test_signed_formatting():
    _reset()
    assert c.format_money(-2_500_000, signed=True, suffixes=("BN", "MM", "K")) == "-£2.5MM"
    assert c.format_money(2_500_000, signed=True, suffixes=("BN", "MM", "K")) == "+£2.5MM"
    assert c.format_money(-500_000, signed=True, suffixes=("BN", "MM", "K")) == "-£500K"


def test_risk_limit_threshold_parse_is_currency_agnostic():
    from mi_agent_api.risk_limits import _amount_threshold
    assert _amount_threshold({"source_snippet": "loans over £500,000 aggregate"}) == 500_000.0
    assert _amount_threshold({"source_snippet": "single exposure > EUR 750,000"}) == 750_000.0
    assert _amount_threshold({"source_snippet": "USD 1,250,000 cap"}) == 1_250_000.0
    assert _amount_threshold({"source_snippet": "€2,000,000 per loan"}) == 2_000_000.0
    assert _amount_threshold({"source_snippet": "no monetary limit here"}) is None
