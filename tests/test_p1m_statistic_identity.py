"""P1M — the requested STATISTIC must be the one that runs, or the answer refuses.

The defect these tests pin: "what is the median LTV?" returned 43.1562 — the
exposure-weighted average — against a true median of 39.6757, with ok=True and no
warning. Two independent routes produced it, so the bank exercises both: the
deterministic parser (which had no word for "median" and applied the field
default) and the LLM repair loop (which was allowed to satisfy a permission error
by changing the statistic).

The invariant under test is narrow and general:

    requested statistic -> governed permission -> executed statistic

with no path from an ungoverned request to a successful answer computed with a
neighbouring statistic.
"""

from __future__ import annotations

import pytest

from mi_agent import statistic as S
from mi_agent.llm_query_parser import (
    _statistic_not_permitted, resolve_statistic_role,
)
from mi_agent.mi_query_spec import MIQuerySpec


# --------------------------------------------------------------------------- #
# The identity relation itself (brief §10)
# --------------------------------------------------------------------------- #
# Every one of these is a substitution that shipped, or could have shipped, as a
# confident answer. The plausibility of the number is irrelevant to the verdict.

@pytest.mark.parametrize("requested, executed", [
    ("median", "weighted_avg"),      # the defect, exactly
    ("median", "avg"),
    ("median", "sum"),
    ("max", "avg"),
    ("min", "avg"),
    ("weighted_avg", "avg"),         # a simple mean is not a weighted mean
    ("sum", "count"),                # a total is not a count
    (S.MEAN, "sum"),                 # an average is not a total
    (S.MEAN, "median"),
])
def test_a_substituted_statistic_is_never_satisfied(requested, executed):
    assert not S.satisfies(requested, executed)


@pytest.mark.parametrize("requested, executed", [
    ("median", "median"),
    ("sum", "sum"),
    ("max", "max"),
    ("weighted_avg", "weighted_avg"),
    ("count", "count_distinct"),
])
def test_the_same_statistic_satisfies_itself(requested, executed):
    assert S.satisfies(requested, executed)


def test_a_plain_average_is_answered_by_the_fields_governed_mean():
    """The one deliberate relaxation, and the reason it is safe.

    "Average LTV" does not choose between a simple and an exposure-weighted mean
    — the registry does, and the house convention for a ratio measure is
    weighted. Treating that as a substitution would refuse a question the product
    answers correctly today.
    """
    assert S.satisfies(S.MEAN, "weighted_avg")
    assert S.satisfies(S.MEAN, "avg")
    # But naming the weighting is a specific request, satisfied only exactly.
    assert not S.satisfies("weighted_avg", "avg")


def test_an_unrequested_statistic_constrains_nothing():
    assert S.satisfies(None, "weighted_avg")


def test_a_multi_measure_answer_is_satisfied_by_any_executed_statistic():
    """P1E composition: several statistics legitimately run at once."""
    assert S.satisfied_by_any("median", ["sum", "count", "median"])
    assert not S.satisfied_by_any("median", ["sum", "count", "weighted_avg"])


# --------------------------------------------------------------------------- #
# Vocabulary — deliberately minimal (brief §2, §11)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question, expected", [
    ("What is the median LTV?", "median"),
    ("What is the median loan balance?", "median"),
    ("What is the average LTV?", S.MEAN),
    ("What is the mean loan size?", S.MEAN),
    ("What is the weighted average interest rate?", "weighted_avg"),
    ("weighted-average current LTV", "weighted_avg"),
])
def test_the_statistic_a_question_names(question, expected):
    assert S.statistic_named(question) == expected


@pytest.mark.parametrize("question", [
    "What is the total balance?",           # ambiguous: "total number" is a count
    "How many loans are in the book?",
    "Show me balance by region.",
    "Which region has the highest average LTV?",  # "highest" ranks, not a statistic
])
def test_statistics_that_cannot_be_denied_are_not_recognised(question):
    """Recognising these would buy no safety and would risk refusing sound work.

    Only a statistic a field registry can DENY can produce the substitution, and
    sum/count are permitted almost everywhere. "Highest" is the sharper case: in
    "which region has the highest average LTV" it names a ranking over groups,
    and reading it as a maximum over the measure would refuse a question the
    product answers correctly.
    """
    named = S.statistic_named(question)
    assert named in (None, S.MEAN)


def test_p1m_adds_no_new_statistics():
    """The brief forbids widening. Percentiles and dispersion stay ungoverned."""
    for word in ("percentile", "quartile", "standard deviation", "variance", "spread"):
        assert S.statistic_named(f"What is the {word} of LTV?") is None
    assert set(S.GOVERNED_STATISTICS) == {
        "sum", "count", "count_distinct", "avg", "weighted_avg",
        "median", "min", "max"}


# --------------------------------------------------------------------------- #
# Permission and resolution against a registry entry (brief §6)
# --------------------------------------------------------------------------- #

_LTV = {"allowed_aggregations": ["avg", "weighted_avg", "distribution"],
        "default_aggregation": "weighted_avg", "business_name": "Current LTV"}
_BALANCE = {"allowed_aggregations": ["sum", "avg", "median"],
            "default_aggregation": "sum", "business_name": "Balance"}


def test_a_statistic_the_registry_omits_is_not_permitted():
    assert not S.permitted_for("median", _LTV)
    assert S.permitted_for("median", _BALANCE)


def test_a_generic_mean_resolves_to_the_fields_own_governed_mean():
    assert S.concrete_for(S.MEAN, _LTV) == "weighted_avg"
    assert S.concrete_for(S.MEAN, _BALANCE) == "avg"


def test_an_unpermitted_statistic_resolves_to_nothing_rather_than_a_neighbour():
    """The whole defect in one assertion: no fallback to the default."""
    assert S.concrete_for("median", _LTV) is None


# --------------------------------------------------------------------------- #
# The parser seam (brief §4)
# --------------------------------------------------------------------------- #

_SEMANTICS = {"fields": {"current_loan_to_value": _LTV,
                         "current_outstanding_balance": _BALANCE}}


def test_the_requested_statistic_reaches_the_spec_instead_of_the_default():
    """Deterministic route to the defect: the request never reached governance."""
    spec = MIQuerySpec(intent="summary", chart_type="none",
                       metric="current_loan_to_value", aggregation="weighted_avg")
    assert resolve_statistic_role(spec, "What is the median LTV?", _SEMANTICS) == "median"
    # Carried forward as asked, so validation can refuse it. Before P1M the spec
    # still said weighted_avg and there was nothing left to refuse.
    assert spec.aggregation == "median"


def test_an_aggregation_that_already_satisfies_the_request_is_left_alone():
    """One-directional by design — this is what keeps "average LTV" correct."""
    spec = MIQuerySpec(intent="summary", chart_type="none",
                       metric="current_loan_to_value", aggregation="weighted_avg")
    assert resolve_statistic_role(spec, "What is the average LTV?", _SEMANTICS) == S.MEAN
    assert spec.aggregation == "weighted_avg"


def test_a_permitted_statistic_is_resolved_and_executed():
    spec = MIQuerySpec(intent="summary", chart_type="none",
                       metric="current_outstanding_balance", aggregation="sum")
    resolve_statistic_role(spec, "What is the median loan balance?", _SEMANTICS)
    assert spec.aggregation == "median"


_MEDIAN_DENIED = ("Aggregation 'median' not allowed for metric "
                  "'current_loan_to_value' (allowed: ['avg', 'distribution', "
                  "'weighted_avg'])")


def test_an_ungoverned_statistic_is_not_a_repairable_parse_error():
    """Why the LLM path shipped it: repair could only satisfy the error by
    asking for a different statistic, and the repaired spec then validated."""
    assert _statistic_not_permitted([_MEDIAN_DENIED], "median")
    assert not _statistic_not_permitted(
        ["Canonical column 'x' (for semantic field 'y') not present"], "median")


def test_one_ungoverned_statistic_is_enough_to_stop_repair():
    """`any`, not `all`: fixing the other errors would still leave a spec that
    can only become valid by substituting the statistic."""
    assert _statistic_not_permitted(
        ["some other validation problem", _MEDIAN_DENIED], "median")


def test_repair_is_allowed_when_the_model_invented_the_statistic():
    """The discriminator, and the reason a blanket rule was wrong.

    "Weighted ltv by region" asks for a weighted average. A model that returns a
    SUM on a percent metric has simply got it wrong, and a repair can only move
    the spec back towards what was asked for. Blocking repair there withheld a
    correct answer to protect a statistic the user never requested.
    """
    invented = ("Aggregation 'sum' not allowed for metric "
                "'current_loan_to_value' (allowed: ['avg', 'weighted_avg'])")
    assert not _statistic_not_permitted([invented], "weighted_avg")
    assert not _statistic_not_permitted([invented], S.MEAN)
    # ... but the same error blocks repair when the SUM is what was asked for.
    assert _statistic_not_permitted([invented], "sum")


def test_a_question_naming_no_statistic_never_blocks_repair():
    """No named statistic means no statistic a repair could move away from."""
    assert not _statistic_not_permitted([_MEDIAN_DENIED], None)


# --------------------------------------------------------------------------- #
# End to end, against the governed entrypoint (brief §7, §8, §9, §13)
# --------------------------------------------------------------------------- #
# Every expected figure is recomputed with pandas from the fixture. The agent is
# never its own oracle.

import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Optional  # noqa: E402

import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo_platform import config as cfg  # noqa: E402

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
AGE = "youngest_borrower_age"
RATE = "current_interest_rate"


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
    cache: Dict[Any, Dict[str, Any]] = {}

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
    """The one substantive figure an answer presents, ignoring the loan count."""
    for label, value in kpi_values(envelope).items():
        if label != "Loan" and value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def weighted(frame: pd.DataFrame, column: str) -> float:
    clean = frame[[column, BALANCE]].dropna()
    return float((clean[column] * clean[BALANCE]).sum() / clean[BALANCE].sum())


# -- the blocker itself ----------------------------------------------------- #

def test_median_ltv_refuses_rather_than_returning_the_weighted_average(ask, book):
    """The headline defect. 43.1562 was delivered for a true median of 39.6757."""
    envelope = ask("What is the median LTV?")
    assert envelope.get("ok") is False
    text = answer(envelope)
    # Names what was asked for, and says the governed statistic was NOT used.
    assert "median" in text.lower()
    assert "not substituted" in text.lower()
    # And no substitute reaches the reader by any other door.
    assert scalar(envelope) is None
    assert not [a for a in (envelope.get("artifacts") or [])
                if a.get("type") in ("kpi", "chart", "table")]
    assert f"{weighted(book, LTV):.4f}" not in text


def test_median_interest_rate_refuses_on_the_same_rule(ask):
    envelope = ask("What is the median interest rate?")
    assert envelope.get("ok") is False
    assert "median" in answer(envelope).lower()
    assert scalar(envelope) is None


def test_the_refusal_does_not_imply_a_substitute_executed(ask):
    """Brief §12: a refusal must not read as though something ran."""
    text = answer(ask("What is the median LTV?")).lower()
    assert "calculated:" not in text


# -- statistics that ARE governed stay correct (brief §7) -------------------- #

def test_median_loan_balance_returns_the_median_not_the_total(ask, book):
    """Before P1M this returned £1,964,886,258.21 for a £156,864.66 question."""
    envelope = ask("What is the median loan balance?")
    assert envelope.get("ok") is True
    assert scalar(envelope) == pytest.approx(float(book[BALANCE].median()), rel=1e-9)
    assert "median" in answer(envelope).lower()


def test_median_borrower_age_returns_the_median_not_the_mean(ask, book):
    """Before P1M this returned the mean, 71.3976, for a true median of 71.0."""
    envelope = ask("What is the median borrower age?")
    assert envelope.get("ok") is True
    assert scalar(envelope) == pytest.approx(float(book[AGE].median()), rel=1e-9)


def test_average_ltv_keeps_its_governed_weighted_definition(ask, book):
    """The regression P1M most had to avoid: "average" must not become a simple
    mean just because the statistic is now governed."""
    envelope = ask("What is the average LTV?")
    assert envelope.get("ok") is True
    assert scalar(envelope) == pytest.approx(weighted(book, LTV), rel=1e-9)
    assert scalar(envelope) != pytest.approx(float(book[LTV].mean()), rel=1e-6)
    assert "weighted-average" in answer(envelope).lower()


def test_weighted_average_interest_rate_is_unchanged(ask, book):
    envelope = ask("What is the weighted average interest rate?")
    assert envelope.get("ok") is True
    assert scalar(envelope) == pytest.approx(weighted(book, RATE), rel=1e-9)


def test_average_borrower_age_is_unchanged(ask, book):
    envelope = ask("What is the average borrower age?")
    assert envelope.get("ok") is True
    assert scalar(envelope) == pytest.approx(float(book[AGE].mean()), rel=1e-9)


def test_total_balance_still_sums(ask, book):
    envelope = ask("What is the total balance?")
    assert envelope.get("ok") is True
    assert scalar(envelope) == pytest.approx(float(book[BALANCE].sum()), rel=1e-9)


def test_loan_count_still_counts(ask, book):
    envelope = ask("How many loans are in the book?")
    assert envelope.get("ok") is True
    assert float(kpi_values(envelope).get("Loan")) == pytest.approx(len(book))


def test_extreme_value_questions_are_now_governed_statistics(ask, book):
    """Superseded by P1N, deliberately.

    At P1M these refused, and the refusal was correct for the reason the guard
    gave: no aggregation could express a single extreme, so any answer would have
    covered the whole book. P1N added governed ``min``/``max``, so the extreme is
    now the statistic that runs and the figure reconciles to the fixture. The
    P1M invariant is untouched — what changed is that the requested statistic
    became available, not that a substitution became permissible.
    """
    assert scalar(ask("What is the maximum LTV?")) == pytest.approx(
        float(book[LTV].max()), rel=1e-9)
    assert scalar(ask("What is the minimum LTV?")) == pytest.approx(
        float(book[LTV].min()), rel=1e-9)


# -- receipts name the statistic that ran (brief §12) ------------------------ #

@pytest.mark.parametrize("question, expected", [
    ("What is the median loan balance?", "median balance"),
    ("What is the average LTV?", "weighted-average current ltv"),
    ("What is the median borrower age?", "median borrower age"),
])
def test_the_receipt_identifies_the_executed_statistic(ask, question, expected):
    assert expected in answer(ask(question)).lower()


# -- multi-measure composition with P1E (brief §9) --------------------------- #

def test_an_ungoverned_statistic_cannot_hide_inside_a_multi_measure_question(ask, book):
    """"Median LTV and total balance" must not answer the balance alone and
    present itself as the whole answer."""
    envelope = ask("Give me median LTV and total loan balance.")
    if envelope.get("ok"):
        # Answering is permissible only if the median genuinely ran.
        assert "median" in answer(envelope).lower()
    else:
        text = answer(envelope).lower()
        assert "median" in text
    # Either way the weighted average must never stand in for the median.
    assert scalar(envelope) != pytest.approx(weighted(book, LTV), rel=1e-6)


def test_a_supported_multi_measure_question_is_unaffected(ask, book):
    envelope = ask("Give me balance, loan count and weighted average LTV.")
    assert envelope.get("ok") is True
    values = {v for v in kpi_values(envelope).values() if v is not None}
    assert any(float(v) == pytest.approx(float(book[BALANCE].sum()), rel=1e-9)
               for v in values)
    assert any(float(v) == pytest.approx(weighted(book, LTV), rel=1e-9)
               for v in values)
