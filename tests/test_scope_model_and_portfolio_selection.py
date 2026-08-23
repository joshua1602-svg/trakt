"""Phase 1G — the source-scope model: base population, selection, provenance.

Phase 1F stopped because the contract carried WHICH scope was resolved and not
WHETHER THE USER ASKED FOR IT, and those decide opposite things when the
workspace carries a selection. This module pins the model that closes it, and
the property that has to survive it: **broad business population and specific
portfolio identity are separate concepts**.

    "the funded book"     base=funded    portfolios=()            unrestricted
    "the acquired book"   base=acquired  portfolios=(every acquired id)
    "SPV2"                base=funded    portfolios=('spv2',)

`SPV2` is deliberately NOT `base=acquired`. Which category a named portfolio
belongs to is a property of the PORTFOLIO, held by the registry, not of the
request — and reading a named portfolio as its whole category is a 3-of-5
overstatement on the fixture below.

THE FIXTURE EXISTS BECAUSE THE LIVE BOOK CANNOT SHOW THIS. It holds one
portfolio per category, so "the acquired book" and "the one acquired portfolio"
select identical rows and every mistake in this file is invisible on it.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: 1 direct + 2 acquired + 2 SPVs, and the SPVs are typed so that "SPV1" and
#: "the acquired book" are DIFFERENT answers.
RECORDS = (
    {"source_portfolio_id": "direct_a", "source_portfolio_type": "direct",
     "source_portfolio_label": "Direct Portfolio A"},
    {"source_portfolio_id": "acquired_a", "source_portfolio_type": "acquired",
     "source_portfolio_label": "Acquired Portfolio A"},
    {"source_portfolio_id": "acquired_b", "source_portfolio_type": "acquired",
     "source_portfolio_label": "Acquired Portfolio B"},
    {"source_portfolio_id": "spv1", "source_portfolio_type": "acquired",
     "source_portfolio_label": "SPV1"},
    {"source_portfolio_id": "spv2", "source_portfolio_type": "direct",
     "source_portfolio_label": "SPV2"},
)

#: Distinct balances per portfolio, so a category-vs-portfolio mistake is a
#: WRONG NUMBER rather than a wrong-looking id list. Powers of ten: any subset
#: sums to a value unique to that subset.
BALANCES = {"direct_a": 1_000.0, "acquired_a": 200.0, "acquired_b": 30.0,
            "spv1": 4.0, "spv2": 50_000.0}
FUNDED_TOTAL = sum(BALANCES.values())        # 51_234.0


@pytest.fixture(scope="module")
def env():
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    try:
        yield cfg.CLIENT_ID
    finally:
        os.environ.clear()
        os.environ.update(before)


@pytest.fixture(scope="module")
def registry(env):
    from trakt_core import portfolio as portfolio_mod
    return portfolio_mod.build_registry(RECORDS, client_id="phase1g")


@pytest.fixture(scope="module")
def semantics(env):
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.datasets import semantics_path
    return load_mi_semantics(semantics_path())


@pytest.fixture(scope="module")
def frame():
    """One row per portfolio, carrying the provenance contract."""
    import pandas as pd
    return pd.DataFrame([
        {"source_portfolio_id": pid, "current_outstanding_balance": bal}
        for pid, bal in BALANCES.items()
    ])


def claim_for(question, semantics, registry, caller=None):
    from question_interpretation import projection
    return projection.project(question, semantics=semantics, registry=registry,
                              caller_scope=caller).source_scope


def selected_balance(claim, frame):
    """What the CLAIM selects, summed. The numeric proof.

    An empty `portfolio_ids` on a funded base means UNRESTRICTED — the complete
    funded population — not "nothing selected". That distinction is the reason
    `base_population` exists.
    """
    if claim.state != "filled":
        return None
    if not claim.portfolio_ids:
        assert claim.base_population == "funded", claim.base_population
        return float(frame["current_outstanding_balance"].sum())
    subset = frame[frame["source_portfolio_id"].isin(list(claim.portfolio_ids))]
    return float(subset["current_outstanding_balance"].sum())


class TestTheMultiPortfolioFixtureMakesMistakesVisible:
    """§11. Every expectation is a distinct number."""

    def test_no_scope_named_is_the_complete_funded_population(
            self, semantics, registry, frame):
        claim = claim_for("Please provide a portfolio summary", semantics, registry)
        assert claim.base_population == "funded"
        assert claim.portfolio_ids == ()
        assert selected_balance(claim, frame) == FUNDED_TOTAL

    def test_the_funded_book_is_the_complete_funded_population(
            self, semantics, registry, frame):
        claim = claim_for("Summarise the funded book", semantics, registry)
        assert claim.base_population == "funded"
        assert selected_balance(claim, frame) == FUNDED_TOTAL

    def test_the_direct_book_is_the_direct_portfolios(
            self, semantics, registry, frame):
        claim = claim_for("Summarise the direct book", semantics, registry)
        assert claim.base_population == "direct"
        assert set(claim.portfolio_ids) == {"direct_a", "spv2"}
        assert selected_balance(claim, frame) == 51_000.0

    def test_the_acquired_book_is_every_acquired_portfolio(
            self, semantics, registry, frame):
        """Three of them, and the number says so: 234.0, not 200.0."""
        claim = claim_for("Summarise the acquired book", semantics, registry)
        assert claim.base_population == "acquired"
        assert set(claim.portfolio_ids) == {"acquired_a", "acquired_b", "spv1"}
        assert selected_balance(claim, frame) == 234.0

    def test_one_named_acquired_portfolio_is_only_that_one(
            self, semantics, registry, frame):
        claim = claim_for("Summarise Acquired Portfolio A", semantics, registry)
        assert claim.portfolio_ids == ("acquired_a",)
        assert selected_balance(claim, frame) == 200.0

    @pytest.mark.parametrize("question,pid,balance", [
        ("Summarise SPV1", "spv1", 4.0),
        ("Show SPV2", "spv2", 50_000.0),
    ])
    def test_a_named_spv_is_only_that_spv(self, semantics, registry, frame,
                                          question, pid, balance):
        claim = claim_for(question, semantics, registry)
        assert claim.portfolio_ids == (pid,)
        assert selected_balance(claim, frame) == balance

    def test_an_unknown_named_portfolio_refuses(self, semantics, registry, frame):
        claim = claim_for("Summarise SPV9", semantics, registry)
        assert claim.state == "unresolvable"
        assert claim.provenance == "unresolved"
        assert claim.portfolio_ids == ()
        assert selected_balance(claim, frame) is None


class TestCategoryAndPortfolioStayDistinct:
    """§8. The two mistakes this fixture exists to catch, each a wrong number."""

    def test_a_category_does_not_collapse_onto_one_portfolio(
            self, semantics, registry, frame):
        acquired = claim_for("Summarise the acquired book", semantics, registry)
        one = claim_for("Summarise Acquired Portfolio A", semantics, registry)
        assert selected_balance(acquired, frame) == 234.0
        assert selected_balance(one, frame) == 200.0
        assert acquired.portfolio_ids != one.portfolio_ids

    def test_a_named_portfolio_is_not_read_as_its_category(
            self, semantics, registry, frame):
        """SPV1 is typed `acquired` in the registry. Asking for SPV1 must not
        answer for the acquired book — 4.0, not 234.0 — and its base population
        is `funded`, because its category belongs to the portfolio and not to
        the request."""
        claim = claim_for("Summarise SPV1", semantics, registry)
        assert claim.base_population == "funded"
        assert claim.portfolio_ids == ("spv1",)
        assert selected_balance(claim, frame) == 4.0

    def test_funded_is_unrestricted_rather_than_an_enumeration(
            self, semantics, registry, frame):
        """A newly onboarded portfolio is inside "the funded book" without
        anything changing here. Enumerating today's members would silently
        exclude tomorrow's."""
        claim = claim_for("Summarise the funded book", semantics, registry)
        assert claim.portfolio_ids == ()
        assert claim.base_population == "funded"


class TestProvenanceDecidesPrecedence:
    """§5 and §6 — the fact Phase 1F stopped for."""

    def test_a_silent_question_defers_to_the_caller_context(
            self, semantics, registry, frame):
        claim = claim_for("Please provide a portfolio summary", semantics,
                          registry, caller="acquired")
        assert claim.provenance == "caller_context"
        assert claim.stated_by_user is False
        assert selected_balance(claim, frame) == 234.0

    def test_an_explicit_question_overrides_the_caller_context(
            self, semantics, registry, frame):
        for question in ("Summarise the funded book",
                         "portfolio summary across all portfolios"):
            claim = claim_for(question, semantics, registry, caller="acquired")
            assert claim.provenance == "explicit_user", question
            assert claim.stated_by_user is True, question
            assert selected_balance(claim, frame) == FUNDED_TOTAL, question

    def test_an_explicit_named_portfolio_overrides_the_caller_context(
            self, semantics, registry, frame):
        claim = claim_for("Show SPV2", semantics, registry, caller="acquired")
        assert claim.provenance == "explicit_user"
        assert selected_balance(claim, frame) == 50_000.0

    def test_no_question_scope_and_no_caller_context_is_funded(
            self, semantics, registry, frame):
        claim = claim_for("Please provide a portfolio summary", semantics, registry)
        assert claim.provenance == "default"
        assert selected_balance(claim, frame) == FUNDED_TOTAL

    def test_a_caller_selected_portfolio_is_honoured(
            self, semantics, registry, frame):
        """Measured before Phase 1G: `_SELECTABLE_COHORT_ID_RE` requires an
        underscore, so a workspace scoped to `spv1` fell through every branch
        and became Total — the whole book, 51,234.0, under a selection of 4.0."""
        claim = claim_for("Please provide a portfolio summary", semantics,
                          registry, caller="spv1")
        assert claim.provenance == "caller_context"
        assert claim.portfolio_ids == ("spv1",)
        assert selected_balance(claim, frame) == 4.0

    def test_the_two_total_readings_are_now_distinguishable(
            self, semantics, registry):
        """The Phase 1F blocker, in one assertion. Both resolve to `total`; the
        claims must differ, because the caller context decides one and not the
        other."""
        silent = claim_for("Please provide a portfolio summary", semantics, registry)
        explicit = claim_for("portfolio summary across all portfolios",
                             semantics, registry)
        assert silent.scope == explicit.scope == "total"
        assert silent.as_dict() != explicit.as_dict()
        assert silent.provenance == "default"
        assert explicit.provenance == "explicit_user"

    def test_an_unresolved_scope_is_never_funded(self, semantics, registry, frame):
        for caller in (None, "acquired", "spv1"):
            claim = claim_for("Summarise SPV9", semantics, registry, caller=caller)
            assert claim.state == "unresolvable", caller
            assert claim.portfolio_ids == (), caller


class TestAddingAPortfolioIsDataNotCode:
    """§7 and §12 — the architecture is data-driven, not branch-driven."""

    def test_a_newly_registered_portfolio_resolves_with_no_code_change(
            self, semantics):
        """SPV3 does not appear anywhere in `mi_agent` or
        `question_interpretation`. It resolves because the registry holds it."""
        from trakt_core import portfolio as portfolio_mod
        extended = portfolio_mod.build_registry(
            RECORDS + ({"source_portfolio_id": "spv3",
                        "source_portfolio_type": "acquired",
                        "source_portfolio_label": "SPV3"},),
            client_id="phase1g")
        claim = claim_for("Summarise SPV3", semantics, extended)
        assert claim.state == "filled"
        assert claim.portfolio_ids == ("spv3",)
        assert claim.base_population == "funded"

    def test_the_same_name_is_unresolvable_before_it_is_registered(
            self, semantics, registry):
        """The control. Without the registry entry SPV3 is a name MI cannot
        find — so the test above is measuring registration, not a coincidence."""
        claim = claim_for("Summarise SPV3", semantics, registry)
        assert claim.state == "unresolvable"

    def test_a_new_category_member_joins_its_category_automatically(
            self, semantics):
        """Registering SPV3 as acquired puts it in "the acquired book" with no
        change to any resolver: 234.0 becomes 238.0."""
        from trakt_core import portfolio as portfolio_mod
        extended = portfolio_mod.build_registry(
            RECORDS + ({"source_portfolio_id": "spv3",
                        "source_portfolio_type": "acquired",
                        "source_portfolio_label": "SPV3"},),
            client_id="phase1g")
        claim = claim_for("Summarise the acquired book", semantics, extended)
        assert set(claim.portfolio_ids) == {"acquired_a", "acquired_b",
                                            "spv1", "spv3"}

    def test_no_portfolio_name_is_hard_coded_in_the_resolvers(self):
        """The property that makes the two tests above mean something: adding
        SPV4 must need registry data, not a parser branch."""
        import re
        for module in ("mi_agent/portfolio_lens.py",
                       "question_interpretation/projection.py",
                       "question_interpretation/schema.py"):
            text = (_REPO / module).read_text(encoding="utf-8")
            code = "\n".join(line for line in text.splitlines()
                             if not line.lstrip().startswith("#"))
            assert not re.search(r"['\"]spv\d", code, re.I), module

    def test_the_naming_family_is_derived_from_the_registry(self, semantics):
        """`SPV9` refuses because the registry demonstrates an `spv<n>` family,
        not because `spv` appears in any resolver. A registry with no numbered
        family recognises no family member, and nothing changes for it."""
        from trakt_core import portfolio as portfolio_mod
        plain = portfolio_mod.build_registry(
            ({"source_portfolio_id": "alpha", "source_portfolio_type": "direct",
              "source_portfolio_label": "Alpha Book"},), client_id="phase1g")
        assert claim_for("Summarise SPV9", semantics, plain).scope == "total"

        pools = portfolio_mod.build_registry(
            ({"source_portfolio_id": "pool1", "source_portfolio_type": "direct",
              "source_portfolio_label": "Pool1"},
             {"source_portfolio_id": "pool2", "source_portfolio_type": "acquired",
              "source_portfolio_label": "Pool2"}), client_id="phase1g")
        assert claim_for("Summarise Pool7", semantics, pools).state == "unresolvable"


class TestVintageAndPortfolioAreSeparateAxes:
    """§13 — one of each, and neither overwrites the other."""

    def test_both_claims_survive_together(self, semantics, registry):
        qi_scope = claim_for("How has the 2025 vintage of SPV2 progressed?",
                             semantics, registry)
        assert qi_scope.portfolio_ids == ("spv2",)

        from question_interpretation import projection
        qi = projection.project("How has the 2025 vintage of SPV2 progressed?",
                                semantics=semantics, registry=registry)
        vintages = [p for p in qi.population if p.concept == "cohort_vintage"]
        assert [p.raw_text for p in vintages] == ["2025"]
        # ... and the portfolio claim is untouched by the vintage claim.
        assert qi.source_scope.portfolio_ids == ("spv2",)
        assert qi.source_scope.base_population == "funded"

    def test_a_vintage_creates_no_new_scope_value(self, semantics, registry):
        """The hierarchy this phase removes would have needed a scope value per
        vintage per portfolio. There is one scope claim and one population
        claim, and the scope vocabulary is unchanged."""
        from question_interpretation.schema import SOURCE_SCOPES
        assert set(SOURCE_SCOPES) == {"total", "direct", "acquired", "cohort"}

    @pytest.mark.xfail(strict=True, reason=(
        "Phase 1D defect, pre-registered and unchanged by 1G: `cohort_vintage` "
        "is set only when the question also carries a progression marker, so a "
        "POINT-IN-TIME vintage is dropped upstream of the contract. The contract "
        "represents both axes simultaneously — proved by the progression "
        "phrasing above — so this is an owner gap, not a structural one."))
    def test_a_point_in_time_vintage_is_also_carried(self, semantics, registry):
        from question_interpretation import projection
        qi = projection.project("Show the 2025 vintage for SPV2",
                                semantics=semantics, registry=registry)
        assert [p.raw_text for p in qi.population if p.concept == "cohort_vintage"]
