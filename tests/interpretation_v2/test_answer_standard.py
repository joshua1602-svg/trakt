"""The answer standard (P0 design §18): one way every governed answer states a
figure, whichever path served it.

The 2026-09-29 spot check (438932ac) showed the one book's funded balance as
"£87.1MM" in a funded answer and "£87.1m" in a forecast answer, a pipeline
breakdown naming ten brokers where a funded one named three, "month(s)" and
"case(s)", and the forecast path hard-coding "£" where the platform resolves the
client's reporting currency. These tests hold the three paths to one standard.
"""
from __future__ import annotations

import pytest

from mi_agent import answer_standard as standard
from mi_agent import plan_serving_canary as canary
from mi_agent_api import adapters
from mi_agent_api import currency as currency_mod
from mi_agent_api import pipeline_contract


@pytest.mark.parametrize("value, words", [
    (87_100_000.0, "£87.1m"), (6_938_000.0, "£6.9m"), (865_900.0, "£866k"),
    (1_234_000_000.0, "£1.23bn"), (950.0, "£950")])
def test_money_in_an_answer_is_the_platforms_chat_convention(value, words):
    assert standard.money(value) == words


@pytest.mark.parametrize("value", [87_100_000.0, 865_900.0, 42_200.0, 0.0])
def test_every_answer_path_states_money_the_same_way(value):
    """Funded, pipeline and forecast sentences: one formatter, one result."""
    expected = standard.money(value)
    assert adapters._prose_value(value, "gbp") == expected
    assert canary._money(value) == expected
    assert pipeline_contract._gbp(value) == expected


def test_a_dashboard_tile_keeps_the_tile_convention():
    assert adapters._format_kpi_value(87_100_000.0, "gbp") == "£87.1MM"
    assert adapters._format_kpi_value(1_234.0, "gbp") == "£1K"


def test_money_is_in_the_clients_reporting_currency():
    import contextvars

    def _in_euros():
        currency_mod.set_currency("EUR")
        return (standard.money(87_100_000.0), canary._money(87_100_000.0),
                adapters._prose_value(87_100_000.0, "gbp"),
                adapters._format_kpi_value(87_100_000.0, "gbp"))

    # A copied context, as each request runs in its own: nothing leaks out.
    assert contextvars.copy_context().run(_in_euros) == (
        "€87.1m", "€87.1m", "€87.1m", "€87.1MM")
    assert standard.money(87_100_000.0) == "£87.1m"


@pytest.mark.parametrize("count, words", [
    (1, "1 month"), (12, "12 months"), (1330, "1,330 months")])
def test_a_count_agrees_with_its_noun(count, words):
    assert standard.plural(count, "month") == words


def test_a_breakdown_names_measure_axis_leaders_and_how_many():
    lead = standard.breakdown_lead(
        "Live pipeline amount", "broker",
        [("A", "£77.6m"), ("B", "£35.1m"), ("C", "£26.3m"), ("D", "£17.4m")],
        total=786)
    assert lead == ("Live pipeline amount by broker — largest: A £77.6m, "
                    "B £35.1m, C £26.3m, and 783 more (786 groups)")
    short = standard.breakdown_lead("Balance", "borrower type",
                                    [("joint", "£45.0m"), ("single", "£42.1m")],
                                    total=2)
    assert short == "Balance by borrower type — largest: joint £45.0m, single £42.1m (2 groups)"


def test_the_funded_breakdown_lead_is_the_standards():
    rows = [{"region": "South East", "current_outstanding_balance_sum": 19.5e6},
            {"region": "South West", "current_outstanding_balance_sum": 11.8e6},
            {"region": "Wales", "current_outstanding_balance_sum": 4.4e6},
            {"region": "London", "current_outstanding_balance_sum": 8.0e6}]
    spec = {"dimension": "region", "metric": "current_outstanding_balance",
            "aggregation": "sum"}
    lead = adapters._breakdown_lead(
        rows, {}, {"current_outstanding_balance": {"format": "gbp"}}, spec)
    assert "— largest: South East £19.5m, South West £11.8m, London £8.0m, " \
           "and 1 more (4 groups)." in lead


def test_no_answer_path_formats_money_itself():
    """The magnitude ladder lives in `mi_agent_api.currency` alone."""
    import inspect
    for module in (canary, pipeline_contract, standard):
        source = inspect.getsource(module)
        assert '(1e6, "m")' not in source, module.__name__
        assert "1e6:,.1f}" not in source, module.__name__
