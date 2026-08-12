#!/usr/bin/env python3
"""One registry, four portfolios, no asset-class branching.

The claim under test is narrow and checkable: **same registry + same analytics
+ different governed data = appropriate capabilities.** If any expectation in
this file needed an ``if asset_class ==`` in the resolver to come out right,
the architecture is not asset-agnostic and the test is the place that should
say so.

The other thing asserted throughout is that a refusal carries its reason. An
autonomous consumer must be able to tell four answers apart, and ``null``
collapses all of them:

    NOT_APPLICABLE       this portfolio has no such thing
    UNAVAILABLE          Trakt lacks an input
    ASSUMPTION_REQUIRED  it needs an unknown future variable
    MODEL_REQUIRED       it needs behavioural modelling

and a fifth that is deliberately not folded into UNAVAILABLE:

    METHODOLOGY_NOT_APPROVED
                         we hold the data and have not settled the definition
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core import config_cache
from trakt_core.capability import (
    ASSUMPTION_REQUIRED,
    AVAILABLE,
    METHODOLOGY_NOT_APPROVED,
    MODEL_REQUIRED,
    NOT_APPLICABLE,
    REASON_CONTINGENT_REPAYMENT,
    REASON_FUTURE_RATE_PATH,
    REASON_INSUFFICIENT_HISTORY,
    REASON_MISSING_INPUT,
    REASON_NO_COLLATERAL,
    STATES,
    UNAVAILABLE,
    describe_portfolio,
    load_registry,
    resolve_all,
    resolve_capability,
    summarise,
)

from tests.multi_asset_portfolios import (
    PORTFOLIOS,
    portfolio_a_equity_release,
    portfolio_b_fixed_amortising,
    portfolio_c_floating_conventional,
    portfolio_d_unsecured,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    config_cache.reset()
    yield
    config_cache.reset()


def _matrix(builder, periods: int = 3):
    shape = describe_portfolio(builder(), history_periods=periods)
    return {a.metric: a for a in resolve_all(shape)}


# =========================================================================== #
# The registry itself
# =========================================================================== #
def test_the_registry_loads_and_every_capability_is_well_formed():
    registry = load_registry()
    assert registry.capabilities
    ids = [c.id for c in registry.capabilities]
    assert len(ids) == len(set(ids)), "duplicate capability ids"
    for capability in registry.capabilities:
        assert capability.category in registry.categories, capability.id
        assert capability.description, capability.id
        assert capability.basis in ("observed", "contractual", "modelled"), (
            f"{capability.id} declares basis {capability.basis!r}; the "
            "observed/contractual/modelled boundary is the vocabulary that "
            "stops a measured CPR being read as a projected one")


def test_a_capability_that_can_never_be_available_says_so_without_a_portfolio():
    """`status_override` short-circuits resolution. A metric whose methodology
    is unowned, or which needs a model Trakt does not build, does not become
    available by finding a richer portfolio — so it must not be resolved
    against one."""
    registry = load_registry()
    # default_rate and cure_rate were here until the 2.5E close-out pass owned
    # their methodologies; they now resolve against the portfolio like any
    # other capability. expected_wal remains, because no portfolio makes a
    # behavioural model unnecessary.
    for metric_id, expected in (("expected_wal", MODEL_REQUIRED),):
        capability = registry.get(metric_id)
        assert capability is not None
        assert capability.status_override == expected
        assert capability.explanation, (
            f"{metric_id} must explain itself; an agent told only "
            f"{expected} will either invent a formula or drop the question")


def test_an_unknown_condition_type_fails_closed():
    """A typo in configuration must not publish a capability nothing checks."""
    from trakt_core.capability import Capability, Condition

    bogus = Capability(id="x", name="x", category="exposure",
                       conditions=(Condition(type="not_a_real_condition"),))
    outcome = resolve_capability(bogus, describe_portfolio(
        portfolio_b_fixed_amortising()))
    assert outcome.status == UNAVAILABLE
    assert outcome.reason_code == "UNKNOWN_CONDITION"


# =========================================================================== #
# Portfolio A — equity release
# =========================================================================== #
def test_erm_gets_collateral_metrics_and_no_contractual_life():
    """The two halves of the ERM answer. It IS collateralised, so LTV is a
    real capability. Its repayment is contingent, so a contractual WAL is not
    — and the reason must say contingency, not a missing field, because no
    field could be supplied to fix it."""
    matrix = _matrix(portfolio_a_equity_release)

    assert matrix["current_ltv"].status == AVAILABLE
    assert matrix["wa_current_ltv"].status == AVAILABLE
    assert matrix["valuation_age"].status == AVAILABLE

    wal = matrix["contractual_wal"]
    assert wal.status == NOT_APPLICABLE
    assert wal.reason_code == REASON_CONTINGENT_REPAYMENT
    assert "death, sale or long-term care" in wal.explanation
    assert "legal long-stop" in wal.explanation

    assert matrix["contractual_ytm"].status == NOT_APPLICABLE


def test_erm_expected_wal_is_model_required_and_stays_that_way():
    """The metric an ERM book would actually want, and the one Trakt must not
    produce. Registered so a consumer learns WHY, not because it is planned."""
    result = _matrix(portfolio_a_equity_release)["expected_wal"]
    assert result.status == MODEL_REQUIRED
    assert "mortality" in result.explanation
    assert "contractual_wal" in result.explanation, (
        "the refusal should point at the deterministic alternatives rather "
        "than leaving the consumer with nothing")


# =========================================================================== #
# Portfolio B — fixed-rate amortising
# =========================================================================== #
def test_a_fixed_rate_amortising_book_gets_both_cash_flow_metrics():
    matrix = _matrix(portfolio_b_fixed_amortising)
    assert matrix["contractual_wal"].status == AVAILABLE
    assert matrix["contractual_ytm"].status == AVAILABLE
    assert matrix["contractual_wal"].detail["amortisation_mix"] == {"FIXE": 1.0}


def test_the_only_portfolio_carrying_loss_evidence_is_the_only_one_that_gets_it():
    """Proves the states are genuinely resolved rather than stuck. B carries
    allocated_losses, default_amount and unscheduled principal; the others do
    not, and report UNAVAILABLE with the missing input named."""
    b = _matrix(portfolio_b_fixed_amortising)
    d = _matrix(portfolio_d_unsecured)

    for metric_id in ("observed_smm", "observed_cpr", "observed_loss_rate",
                      "observed_recovery_rate", "observed_loss_severity"):
        assert b[metric_id].status == AVAILABLE, metric_id
        assert d[metric_id].status == UNAVAILABLE, metric_id
        assert d[metric_id].reason_code == REASON_MISSING_INPUT
        assert d[metric_id].missing_inputs, (
            f"{metric_id} must name what is missing, or a client cannot be "
            "asked for it")


# =========================================================================== #
# Portfolio C — floating conventional
# =========================================================================== #
def test_a_floating_french_book_loses_both_metrics_and_says_it_is_the_rate():
    """FRXX fixes the TOTAL instalment, so an unknown rate makes the PRINCIPAL
    split unknown too — WAL falls with YTM here, unlike a floating bullet.
    The reason must be the rate path, not a missing field."""
    matrix = _matrix(portfolio_c_floating_conventional)

    for metric_id in ("contractual_wal", "contractual_ytm"):
        result = matrix[metric_id]
        assert result.status == ASSUMPTION_REQUIRED, metric_id
        assert result.reason_code == REASON_FUTURE_RATE_PATH
        assert "No rate path is assumed" in result.explanation or \
               "do not determine it" in result.explanation

    assert matrix["contractual_wal"].detail["amortisation_mix"] == {"FRXX": 1.0}


def test_a_floating_bullet_keeps_its_wal_where_a_floating_french_loan_does_not():
    """The distinction that would be lost if determinism were one flag instead
    of two. Same rate type, same everything else — only RREL35 differs."""
    french = portfolio_c_floating_conventional()
    bullet = portfolio_c_floating_conventional()
    bullet["amortisation_type"] = "BLLT"

    french_wal = resolve_capability(
        load_registry().get("contractual_wal"),
        describe_portfolio(french, history_periods=3))
    bullet_wal = resolve_capability(
        load_registry().get("contractual_wal"),
        describe_portfolio(bullet, history_periods=3))

    assert french_wal.status == ASSUMPTION_REQUIRED
    assert bullet_wal.status == AVAILABLE, (
        "a bullet's principal is one flow at maturity whatever the rate does")


# =========================================================================== #
# Portfolio D — unsecured
# =========================================================================== #
def test_an_unsecured_book_loses_only_the_metrics_that_need_a_value():
    """The asset-agnosticism proof. D is not a mortgage, has no valuation and
    no geography, and was not special-cased anywhere. Its credit and
    cash-flow capabilities survive intact."""
    matrix = _matrix(portfolio_d_unsecured)

    for metric_id in ("current_ltv", "wa_current_ltv", "high_ltv_exposure",
                      "valuation_age"):
        result = matrix[metric_id]
        assert result.status == NOT_APPLICABLE, metric_id
        assert result.reason_code == REASON_NO_COLLATERAL

    # And everything that does not need collateral is untouched.
    for metric_id in ("total_balance", "arrears_30_plus", "arrears_90_plus",
                      "wa_coupon", "contractual_wal", "contractual_ytm",
                      "cohort_comparison"):
        assert matrix[metric_id].status == AVAILABLE, metric_id


def test_no_collateral_is_not_applicable_rather_than_unavailable():
    """The distinction a client-facing consumer acts on. UNAVAILABLE invites a
    data request; NOT_APPLICABLE says no field would help, because a
    loan-to-VALUE on an unsecured loan has no denominator to supply."""
    result = _matrix(portfolio_d_unsecured)["current_ltv"]
    assert result.status == NOT_APPLICABLE
    assert result.status != UNAVAILABLE
    assert "unsecured exposure, not a data gap" in result.explanation


def test_missing_geography_is_not_applicable_too():
    result = _matrix(portfolio_d_unsecured)["geographic_concentration"]
    assert result.status == NOT_APPLICABLE
    assert result.reason_code == "NO_GEOGRAPHY"


# =========================================================================== #
# History
# =========================================================================== #
def test_one_snapshot_makes_period_metrics_unavailable_with_the_count():
    """"What was CPR over the last 12 months?" on a single snapshot is a
    missing-history answer, and the numbers make it actionable."""
    matrix = _matrix(portfolio_b_fixed_amortising, periods=1)

    for metric_id in ("observed_cpr", "observed_smm", "portfolio_history",
                      "arrears_transition"):
        result = matrix[metric_id]
        assert result.status == UNAVAILABLE, metric_id
        assert result.reason_code == REASON_INSUFFICIENT_HISTORY
        assert result.detail["required_periods"] == 2
        assert result.detail["available_periods"] == 1

    # And point-in-time capabilities are unaffected by having one snapshot.
    assert matrix["current_ltv"].status == AVAILABLE
    assert matrix["contractual_wal"].status == AVAILABLE


# =========================================================================== #
# The matrix as a whole
# =========================================================================== #
def test_every_state_in_the_model_is_reachable_from_real_data():
    """A state nothing can produce is a state nobody has thought through. All
    six are exercised by these four portfolios."""
    seen = set()
    for builder in PORTFOLIOS.values():
        seen.update(a.status for a in _matrix(builder).values())

    # METHODOLOGY_NOT_APPROVED is no longer reachable from these four
    # portfolios, because the close-out pass owned the two methodologies that
    # produced it. The state stays in the model — it is the right answer the
    # next time Trakt holds data for a metric whose definition is unsettled,
    # and removing it would guarantee the next such metric is mislabelled
    # UNAVAILABLE, sending someone to a client for a field already on the tape.
    expected = set(STATES) - {METHODOLOGY_NOT_APPROVED}
    assert seen == expected, f"unreached states: {expected - seen}"


def test_no_capability_ever_resolves_to_an_undeclared_state():
    for builder in PORTFOLIOS.values():
        for result in _matrix(builder).values():
            assert result.status in STATES, result


def test_every_refusal_carries_a_reason_code_and_an_explanation():
    """The load-bearing property. A consumer that receives a bare status can
    report that something is missing but not what to do about it."""
    for name, builder in PORTFOLIOS.items():
        for result in _matrix(builder).values():
            if result.status == AVAILABLE:
                continue
            assert result.reason_code, f"{name}/{result.metric} has no reason code"
            assert len(result.explanation) > 40, (
                f"{name}/{result.metric} explanation is too thin to act on")


def test_the_four_portfolios_genuinely_differ():
    """If the matrix were identical everywhere, the tests above would pass
    while proving nothing about whether availability is resolved at all."""
    matrices = {name: {k: v.status for k, v in _matrix(b).items()}
                for name, b in PORTFOLIOS.items()}
    distinct = {tuple(sorted(m.items())) for m in matrices.values()}
    assert len(distinct) == 4, "the portfolios must produce four different answers"

    summaries = {name: summarise(list(_matrix(b).values()))
                 for name, b in PORTFOLIOS.items()}
    assert summaries["D_unsecured"][NOT_APPLICABLE] > \
        summaries["B_fixed_amortising"].get(NOT_APPLICABLE, 0)


def test_the_resolver_contains_no_asset_class_branching():
    """The non-negotiable, asserted against the source rather than trusted.

    Availability is decided by economic conditions — fields, history,
    determinism — and never by a product label. A future asset class whose
    canonical tape carries the concepts must light up without code.
    """
    import ast
    import io
    import re
    import tokenize

    source = (_REPO_ROOT / "trakt_core" / "capability.py").read_text(
        encoding="utf-8")

    # Strip comments and docstrings before searching. The claim being tested
    # is about LOGIC, not prose: a docstring may legitimately explain that a
    # lifetime mortgage is why NOT_APPLICABLE exists, and asserting against
    # the explanation would push the reasoning out of the file that needs it.
    lines = source.splitlines()
    blank = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            blank.update(range(first.lineno - 1, (first.end_lineno or
                                                  first.lineno)))
    code = "\n".join("" if i in blank else line
                      for i, line in enumerate(lines))
    code = "".join(
        "" if tok.type == tokenize.COMMENT else tok.string
        for tok in tokenize.generate_tokens(io.StringIO(code).readline))

    for token in ("equity_release", "residential_mortgage",
                  "commercial_mortgage", "asset_class", "ERM"):
        assert not re.search(rf"\b{token}\b", code), (
            f"trakt_core/capability.py branches on {token!r}; availability "
            "must be resolved from economics, not from an asset label")

    # An explanation string may name a product — telling a human that a
    # lifetime mortgage is the usual case of contingent repayment is exactly
    # what makes a refusal useful. What must not exist is a DECISION made on
    # a product label. So assert the positive: the determinism branch is on
    # the regime's own codes.
    assert "RATE_INDEPENDENT_PRINCIPAL" in code and "RATE_FIXED_FOR_LIFE" in code
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Compare):
            rendered = ast.dump(node)
            for token in ("equity_release", "residential_mortgage",
                          "commercial_mortgage", "lifetime"):
                assert token not in rendered, (
                    f"a comparison in capability.py tests for {token!r}")


# =========================================================================== #
# MI Query — a refusal that explains itself is a successful answer
# =========================================================================== #
def _ask(question, builder, periods: int = 1):
    from mi_agent.mi_agent_workflow import _capability_explanation

    return _capability_explanation(question, builder(), periods)


def test_asking_an_erm_book_for_a_wal_gets_the_reason_not_a_dead_end():
    """Before this, MI Query answered "that measure does not exist here",
    which is wrong twice over: the measure exists, and the portfolio is the
    reason it does not apply. The reason is the answer."""
    answer = _ask("What is the contractual WAL of this portfolio?",
                  portfolio_a_equity_release)
    assert answer is not None
    assert "NOT_APPLICABLE" in answer
    assert "contingent on death, sale or long-term care" in answer
    assert "no other measure has been substituted" in answer


def test_a_generic_metric_that_can_answer_is_left_to_the_ordinary_query_path():
    """The capability layer must not become a second route to a number the
    ordinary path computes correctly. A field-plus-aggregation capability —
    balance, WA LTV, WA coupon — stays silent when it is available."""
    from trakt_core.capability import load_registry

    assert load_registry().get("total_balance").owned_kpi is False
    assert _ask("What is the total outstanding balance?",
                portfolio_b_fixed_amortising) is None


def test_an_owned_kpi_that_is_available_names_its_methodology_instead():
    """The precedence rule. An owned KPI is NOT computed by the ad-hoc query
    path, so when it is available the honest answer names the approved
    methodology and where it comes from — rather than falling through to "I
    couldn't map this question", which is what happened before the close-out
    pass traced the real code path."""
    answer = _ask("What is the contractual WAL of this portfolio?",
                  portfolio_b_fixed_amortising)
    assert answer is not None
    assert "CONTRACTUAL_WAL@v1" in answer
    assert "analytics_lib.contractual.portfolio_wal" in answer
    assert "generic aggregation" in answer


def test_asking_a_floating_book_for_a_yield_names_the_rate_path():
    answer = _ask("What is the YTM?", portfolio_c_floating_conventional)
    assert "ASSUMPTION_REQUIRED" in answer
    assert "No rate path is assumed" in answer


def test_a_multi_snapshot_metric_says_mi_query_is_the_wrong_path_for_it():
    """MI Query is a SINGLE-FRAME engine: one dataset in, one answer out. A
    metric measured across snapshots cannot be computed here whatever the
    deployment holds, so reporting "1 snapshot is available" would misreport
    a limitation of this path as a gap in the client's data — and would send
    someone to ask for history they already have."""
    answer = _ask("What was CPR over the last 12 months?",
                  portfolio_b_fixed_amortising, periods=1)
    assert "single dataset" in answer
    assert "OBSERVED_CPR@v2" in answer
    assert "analytics_lib.history.prepayment_rate" in answer
    assert "at least 2 governed snapshots" not in answer


def test_asking_an_unsecured_book_for_wa_ltv_is_not_applicable():
    answer = _ask("What is WA LTV?", portfolio_d_unsecured)
    assert "NOT_APPLICABLE" in answer
    assert "no value to weight a loan against" in answer


def test_asking_for_default_rate_now_gets_history_not_a_methodology_refusal():
    """Until the close-out pass this answered METHODOLOGY_NOT_APPROVED. The
    methodology is now owned — ESMA's own CDR definition — so the only thing
    standing between this portfolio and a default rate is a second snapshot,
    and the answer says exactly that."""
    answer = _ask("What is the default rate?", portfolio_b_fixed_amortising,
                  periods=1)
    assert "METHODOLOGY_NOT_APPROVED" not in answer
    assert "OBSERVED_DEFAULT_RATE_CDR@v1" in answer


def test_asking_for_default_rate_with_history_names_the_owned_methodology():
    answer = _ask("What is the default rate?", portfolio_b_fixed_amortising,
                  periods=3)
    assert "OBSERVED_DEFAULT_RATE_CDR@v1" in answer
    assert "analytics_lib.history.default_rate" in answer


def test_asking_for_cure_rate_names_its_methodology_too():
    answer = _ask("What is the cure rate?", portfolio_b_fixed_amortising,
                  periods=3)
    assert "OBSERVED_CURE_RATE@v1" in answer


def test_a_question_about_nothing_still_falls_through_to_the_generic_refusal():
    """The capability registry must not become a catch-all that dresses up
    every unparsed question as a known metric."""
    assert _ask("What is the unicorn ratio?", portfolio_b_fixed_amortising) is None


def test_expected_wal_is_matched_ahead_of_contractual_wal():
    """Longest-phrase matching. "expected weighted average life" contains
    "weighted average life", and answering the second would tell the user
    their modelled question was a contractual one."""
    from trakt_core.capability import match_capability

    assert match_capability("what is expected weighted average life").id == \
        "expected_wal"
    assert match_capability("what is the weighted average life").id == \
        "contractual_wal"


# =========================================================================== #
# One economic definition, one calculation — the drift guard
# =========================================================================== #
def test_the_weighted_average_helpers_scattered_across_consumers_still_agree():
    """Sprint 2.5E found TEN weighted-average helpers across the repository:
    in the PowerPoint resolver, the cohorts, snapshots and evolution API
    routes, three analytics modules, the history handler, the simulation
    oracle and the governed metric evaluator.

    Tested on a common input they currently AGREE — all use pairwise deletion,
    differing only in rounding — so this is not a live correctness defect and
    is reported as such rather than overstated. What it is, is ten places one
    convention can drift, and the drift would be invisible: a missing-data
    policy change in one of them would move a number on one screen and not
    another.

    Routing them to the shared evaluator is real work and deliberately NOT
    done here — it is unrelated refactoring, and ``simulation.reference_truth``
    must stay independent by design, since its whole purpose is to be an
    oracle that does not share code with what it checks. This test pins the
    agreement so a future divergence fails loudly instead of silently.
    """
    import pandas as pd

    from analytics.mi_prep import weighted_average as mi_prep_wa
    from mi_agent_api.cohorts import _weighted_avg as cohorts_wa
    from mi_agent_api.evolution import _weighted_avg as evolution_wa
    from mi_agent_api.snapshots import _weighted_average as snapshots_wa
    from mi_agent_pptx.metric_resolver import _weighted_avg as pptx_wa

    values = pd.Series([10.0, 20.0, None, 40.0])
    weights = pd.Series([100.0, 200.0, 300.0, None])
    # Pairwise deletion: rows missing EITHER term drop from both numerator and
    # denominator. (10x100 + 20x200) / (100+200) = 16.666..., by hand.
    expected = (10 * 100 + 20 * 200) / (100 + 200)
    assert expected == pytest.approx(16.666667, abs=1e-6)

    frame = pd.DataFrame({"x": values, "current_outstanding_balance": weights})

    for label, actual in (
            ("analytics.mi_prep", mi_prep_wa(values, weights)),
            ("mi_agent_api.cohorts", cohorts_wa(values, weights)),
            ("mi_agent_api.snapshots", snapshots_wa(values, weights)),
            ("mi_agent_pptx.metric_resolver", pptx_wa(values, weights)),
            ("mi_agent_api.evolution",
             evolution_wa(frame, "x", "current_outstanding_balance")),
    ):
        assert actual == pytest.approx(expected, abs=1e-4), (
            f"{label} disagrees with the pairwise-deletion convention every "
            "other weighted average in the repository uses")
