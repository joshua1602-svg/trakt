"""P1N — governed MIN/MAX and two exposure-weighted measures.

Four capabilities, added through the P1M statistic-identity seam rather than
beside it:

  * ``min`` / ``max`` as governed statistics of a measure — distinct from a
    RANKING, which orders groups and returns the winning group;
  * exposure-weighted borrower age;
  * exposure-weighted months on book (portfolio seasoning).

The P1M invariant stays authoritative throughout: a requested statistic that is
not the executed statistic refuses, and that must hold for the new statistics
exactly as it does for the old ones.

Every figure is recomputed with pandas from the fixture. The agent is never its
own oracle.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo_platform import config as cfg  # noqa: E402
from mi_agent import statistic as S  # noqa: E402
from mi_agent.mi_query_executor import aggregate_series  # noqa: E402
from mi_agent.mi_query_spec import AGGREGATIONS  # noqa: E402

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
AGE = "youngest_borrower_age"
MOB = "months_on_book"
VAL = "current_valuation_amount"
SEG = "seasoning_segment"
SRC = "source_portfolio_type"


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #

def test_min_and_max_are_part_of_the_governed_aggregation_contract():
    assert {"min", "max"} <= AGGREGATIONS


def test_the_executor_computes_them_ignoring_nulls():
    frame = pd.DataFrame({"x": [3.0, 1.0, 5.0, None]})
    assert aggregate_series(frame, "x", "min") == 1.0
    assert aggregate_series(frame, "x", "max") == 5.0


# --------------------------------------------------------------------------- #
# Vocabulary — a superlative is not always a statistic
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question, expected", [
    ("What is the maximum loan balance?", "max"),
    ("What is the highest LTV?", "max"),
    ("What is the largest loan in the acquired book?", "max"),
    ("What is the minimum loan balance?", "min"),
    ("What is the lowest borrower age?", "min"),
    ("What is the smallest valuation?", "min"),
])
def test_a_superlative_names_the_extreme_statistic(question, expected):
    assert S.statistic_named(question) == expected


def test_an_explicit_statistic_beats_a_superlative_in_the_same_sentence():
    """"The highest AVERAGE LTV" asks for an average, ranked. Reading it as a
    maximum would refuse a question the product answers correctly."""
    assert S.statistic_named("Which region has the highest average LTV?",
                             grouped=True) == S.MEAN


def test_a_superlative_over_groups_is_a_ranking_not_a_statistic():
    """"Which region has the highest LTV" wants the winning region, not one
    extreme loan. The ranking facet owns it, so no statistic is read."""
    assert S.statistic_named("Which region has the highest LTV?",
                             grouped=True) is None
    # ... but the same words with no grouping ARE the extreme value.
    assert S.statistic_named("What is the highest LTV?", grouped=False) == "max"


def test_exposure_weighted_is_read_as_a_weighting_not_a_measure():
    assert S.statistic_named("What is the exposure-weighted borrower age?") == "weighted_avg"
    assert "exposure" not in S.mask_statistic_phrases(
        "What is the exposure-weighted borrower age?").lower()


def test_oldest_and_youngest_are_not_statistic_vocabulary():
    """The measure is literally named ``youngest_borrower_age``, so these words
    are the field's name rather than a statistic over it. Reported, not guessed."""
    assert S.statistic_named("What is the youngest borrower age?") is None
    assert S.statistic_named("Who is the oldest borrower?") is None


# --------------------------------------------------------------------------- #
# Statistic identity still holds for the NEW statistics (brief §8)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("requested, executed", [
    ("max", "avg"),
    ("max", "sum"),
    ("max", "weighted_avg"),
    ("max", "min"),
    ("min", "avg"),
    ("min", "max"),
    ("weighted_avg", "avg"),      # weighted borrower age vs simple mean
    ("weighted_avg", "median"),
])
def test_a_substituted_statistic_is_still_never_satisfied(requested, executed):
    assert not S.satisfies(requested, executed)


def test_the_new_statistics_satisfy_only_themselves():
    assert S.satisfies("max", "max")
    assert S.satisfies("min", "min")
    assert S.satisfies("weighted_avg", "weighted_avg")


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def governed_env():
    root = cfg.local_blob_root() / "processed" / "platform" / cfg.CLIENT_ID
    if not root.exists():
        pytest.skip("the demonstration platform has not been built")
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    logging.disable(logging.WARNING)
    from mi_agent_api import data_source

    data_source.reset_cache()
    yield
    logging.disable(logging.NOTSET)
    os.environ.clear()
    os.environ.update(before)
    data_source.reset_cache()


@pytest.fixture(scope="module")
def book(governed_env) -> pd.DataFrame:
    from mi_agent_api.data_source import get_dataframe

    return get_dataframe()


@pytest.fixture(scope="module")
def ask(governed_env):
    from trakt_core.context import ExecutionContext
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    cache: Dict[str, Dict[str, Any]] = {}

    def _ask(question: str) -> Dict[str, Any]:
        if question not in cache:
            cache[question] = (execute_governed_mi_query(
                MiQueryRequest(question=question), ctx).result or {})
        return cache[question]

    return _ask


def answer(envelope) -> str:
    return envelope.get("answer") or envelope.get("error") or ""


def kpi_values(envelope) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for artifact in envelope.get("artifacts") or []:
        if artifact.get("type") == "kpi":
            for entry in artifact.get("kpis") or []:
                out[entry.get("label") or entry.get("field")] = entry.get("rawValue")
    return out


def scalar(envelope) -> Optional[float]:
    for label, value in kpi_values(envelope).items():
        if label != "Loan" and value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def population(envelope) -> Optional[int]:
    loans = kpi_values(envelope).get("Loan")
    return None if loans is None else int(loans)


def weighted(frame: pd.DataFrame, column: str) -> float:
    clean = frame[[column, BALANCE]].dropna()
    return float((clean[column] * clean[BALANCE]).sum() / clean[BALANCE].sum())


# -- basic extremes --------------------------------------------------------- #

@pytest.mark.parametrize("question, column, how", [
    ("What is the maximum loan balance?", BALANCE, "max"),
    ("What is the minimum loan balance?", BALANCE, "min"),
    ("What is the maximum LTV?", LTV, "max"),
    ("What is the minimum LTV?", LTV, "min"),
    ("What is the maximum borrower age?", AGE, "max"),
    ("What is the minimum borrower age?", AGE, "min"),
    ("What is the maximum property valuation?", VAL, "max"),
    ("What is the minimum property valuation?", VAL, "min"),
    ("What is the maximum months on book?", MOB, "max"),
    ("What is the minimum months on book?", MOB, "min"),
])
def test_extremes_reconcile_to_the_fixture(ask, book, question, column, how):
    envelope = ask(question)
    assert envelope.get("ok") is True, answer(envelope)
    expected = float(getattr(book[column], how)())
    assert scalar(envelope) == pytest.approx(expected, rel=1e-9)
    assert population(envelope) == len(book)


def test_highest_stays_a_loan_level_ranking_and_maximum_is_the_statistic(ask, book):
    """The two governed readings of a superlative, kept apart.

    "Largest loan balance" is an established LOAN-LEVEL RANKING — a table of the
    biggest loans — and P1N is not allowed to eat it. So "highest LTV" keeps that
    table, whose leading row carries the true extreme, while "maximum LTV"
    resolves to the governed scalar statistic. Both are correct; they are
    different questions, and each keeps its own answer shape.
    """
    ranked = ask("What is the highest LTV?")
    assert ranked.get("ok") is True, answer(ranked)
    rows = []
    for artifact in ranked.get("artifacts") or []:
        rows = artifact.get("rows") or rows
    top = max(float(r[LTV]) for r in rows if r.get(LTV) is not None)
    assert top == pytest.approx(float(book[LTV].max()), rel=1e-9)

    scalar_form = ask("What is the maximum LTV?")
    assert scalar(scalar_form) == pytest.approx(float(book[LTV].max()), rel=1e-9)


def test_the_receipt_names_the_extreme_that_ran(ask):
    assert "maximum balance" in answer(ask("What is the maximum loan balance?")).lower()
    assert "minimum balance" in answer(ask("What is the minimum loan balance?")).lower()


# -- composition with filters (brief §10) ----------------------------------- #

def test_max_composes_with_an_age_threshold(ask, book):
    envelope = ask("What is the maximum loan balance for borrowers over 85?")
    over = book[book[AGE] > 85]
    assert envelope.get("ok") is True, answer(envelope)
    assert scalar(envelope) == pytest.approx(float(over[BALANCE].max()), rel=1e-9)
    assert population(envelope) == len(over)


def test_max_composes_with_a_balance_threshold_on_a_different_measure(ask, book):
    """The threshold defines the POPULATION; the measure stays LTV. It must not
    be reinterpreted as a filter on LTV, which would match nothing."""
    envelope = ask("What is the maximum LTV for loans over £500k?")
    over = book[book[BALANCE] > 500000]
    assert envelope.get("ok") is True, answer(envelope)
    assert scalar(envelope) == pytest.approx(float(over[LTV].max()), rel=1e-9)
    assert population(envelope) == len(over)


def test_an_unqualified_number_attaches_to_the_measure_and_refuses(ask):
    """"loans over 500000" with no currency marker is genuinely ambiguous. The
    threshold attaches to the requested measure, matches nothing, and the answer
    refuses rather than quietly widening to the whole book."""
    envelope = ask("What is the maximum LTV for loans over 500000?")
    assert envelope.get("ok") is False
    assert scalar(envelope) is None


# -- composition with governed populations (brief §9) ----------------------- #

@pytest.mark.parametrize("question, column, how, mask", [
    ("What is the maximum LTV in the back book?", LTV, "max", "back"),
    ("What is the maximum loan balance in the back book?", BALANCE, "max", "back"),
    ("What is the maximum loan balance in the acquired book?", BALANCE, "max", "acquired"),
    ("What is the minimum borrower age in the direct book?", AGE, "min", "direct"),
])
def test_extremes_are_computed_over_the_requested_population(
        ask, book, question, column, how, mask):
    frame = {"back": book[book[SEG] == "Back Book"],
             "direct": book[book[SRC] == "direct"],
             "acquired": book[book[SRC] == "acquired"]}[mask]
    envelope = ask(question)
    assert envelope.get("ok") is True, answer(envelope)
    assert scalar(envelope) == pytest.approx(float(getattr(frame[column], how)()), rel=1e-9)
    # P1L: the population is proven by the row count, not claimed.
    assert population(envelope) == len(frame)
    assert population(envelope) < len(book)


# -- exposure-weighted measures --------------------------------------------- #

def test_exposure_weighted_borrower_age(ask, book):
    envelope = ask("What is the exposure-weighted borrower age of the portfolio?")
    assert envelope.get("ok") is True, answer(envelope)
    assert scalar(envelope) == pytest.approx(weighted(book, AGE), rel=1e-9)
    # It is NOT the simple mean — that is the whole point of adding it.
    assert scalar(envelope) != pytest.approx(float(book[AGE].mean()), rel=1e-6)
    assert "weighted-average" in answer(envelope).lower()


def test_exposure_weighted_months_on_book(ask, book):
    envelope = ask("What is the exposure-weighted months on book?")
    assert envelope.get("ok") is True, answer(envelope)
    assert scalar(envelope) == pytest.approx(weighted(book, MOB), rel=1e-9)
    assert scalar(envelope) != pytest.approx(float(book[MOB].mean()), rel=1e-6)


def test_exposure_weighted_measures_compose_with_a_population(ask, book):
    back = book[book[SEG] == "Back Book"]
    envelope = ask("What is the exposure-weighted borrower age of the back book?")
    assert envelope.get("ok") is True, answer(envelope)
    assert scalar(envelope) == pytest.approx(weighted(back, AGE), rel=1e-9)
    assert population(envelope) == len(back)


def test_the_plain_average_of_those_measures_is_unchanged(ask, book):
    """``default_aggregation`` stays ``avg``: permitting a weighted average must
    not change what "average borrower age" means."""
    envelope = ask("What is the average borrower age?")
    assert scalar(envelope) == pytest.approx(float(book[AGE].mean()), rel=1e-9)
    assert scalar(envelope) != pytest.approx(weighted(book, AGE), rel=1e-6)


# -- P1M regression (brief §14) --------------------------------------------- #

def test_median_ltv_still_refuses(ask):
    envelope = ask("What is the median LTV?")
    assert envelope.get("ok") is False
    assert "median" in answer(envelope).lower()
    assert scalar(envelope) is None


def test_the_other_p1m_statistics_are_unchanged(ask, book):
    assert scalar(ask("What is the median loan balance?")) == pytest.approx(
        float(book[BALANCE].median()), rel=1e-9)
    assert scalar(ask("What is the median borrower age?")) == pytest.approx(
        float(book[AGE].median()), rel=1e-9)
    assert scalar(ask("What is the average LTV?")) == pytest.approx(
        weighted(book, LTV), rel=1e-9)
    assert scalar(ask("What is the total balance?")) == pytest.approx(
        float(book[BALANCE].sum()), rel=1e-9)


def test_the_largest_single_loan_exposure_capability_is_unchanged(ask, book):
    """P1F owns this question and must keep owning it — the new max statistic
    must not capture a routed exposure answer."""
    envelope = ask("What is our largest single-loan exposure and what "
                   "percentage of the portfolio is it?")
    assert envelope.get("ok") is True, answer(envelope)
    assert "largest single-loan" in answer(envelope).lower()


def test_min_max_were_not_added_to_count_metrics_or_dimensions():
    """The brief forbids adding extremes to every numeric-looking field."""
    import yaml

    registry = yaml.safe_load(
        (_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml").read_text())
    fields = registry["fields"]
    for key in ("number_of_leased_objects", "number_of_properties_at_data_cut_off_date",
                "collateral_geography", "account_status", "origination_date"):
        allowed = set((fields.get(key) or {}).get("allowed_aggregations") or [])
        assert not ({"min", "max"} & allowed), key


def test_the_two_weighted_measures_keep_their_governed_weight_and_default():
    import yaml

    registry = yaml.safe_load(
        (_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml").read_text())
    for key in (AGE, MOB):
        entry = registry["fields"][key]
        assert "weighted_avg" in entry["allowed_aggregations"]
        assert entry["weight_field"] == BALANCE
        assert entry["default_aggregation"] == "avg"
