"""Phase 1E — is the identity the scope claim carries the GOVERNED one?

Phase 1A gave the contract a source-scope claim; Phase 1C gave it provenance;
Phase 1D found that the identity flowing into it was the STORAGE convention.
`acquired_001` is a blob folder name. It is not what the governed registry keys
on, not what React renders, and not what an MI analyst calls a book — so a
consumer that filtered on `portfolio_ids` would have been filtering on a name
the governed model does not hold.

These tests pin the claim's identity after 1E:

  * a book NAMED in the question resolves to its GOVERNED id;
  * the wording that asked and the governed label are carried SEPARATELY;
  * a name the registry does not hold is UNRESOLVABLE, never `total`;
  * without a registry the claim is exactly what it was before 1E.

The owner is still `mi_agent.portfolio_lens`. Nothing here matches a phrase.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from question_interpretation.schema import (  # noqa: E402
    FILLED, UNRESOLVABLE, SCOPE_ACQUIRED, SCOPE_COHORT, SCOPE_TOTAL,
    SourceScopeClaim,
)

FIXTURE_RECORDS = (
    {"source_portfolio_id": "alp_origination", "source_portfolio_type": "direct",
     "source_portfolio_label": "ALP Origination Book"},
    {"source_portfolio_id": "alp_acquired", "source_portfolio_type": "acquired",
     "source_portfolio_label": "ALP Acquired Back Book"},
    {"source_portfolio_id": "nbs_acquired", "source_portfolio_type": "acquired",
     "source_portfolio_label": "NBS Acquired Book"},
)


@pytest.fixture(scope="module")
def semantics():
    warnings.simplefilter("ignore")
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    return load_mi_semantics(semantics_path())


@pytest.fixture(scope="module")
def registry():
    from trakt_core import portfolio as portfolio_mod
    return portfolio_mod.build_registry(FIXTURE_RECORDS, client_id="phase1e")


def _scope(question, semantics, registry=None):
    from question_interpretation import projection
    return projection.project(question, semantics=semantics,
                              registry=registry).source_scope


class TestTheClaimCarriesGovernedIdentity:
    def test_a_named_book_resolves_to_its_governed_id(self, semantics, registry):
        claim = _scope("Summarise the NBS Acquired Book", semantics, registry)
        assert claim.state == FILLED
        assert claim.scope == SCOPE_COHORT
        assert claim.portfolio_ids == ("nbs_acquired",)

    def test_the_wording_and_the_governed_label_are_carried_separately(
            self, semantics, registry):
        """"the alp_acquired book" and "ALP Acquired Back Book" are the SAME
        portfolio said two ways. An audit of what was ASKED needs the first; an
        explanation of what was ANSWERED needs the second."""
        claim = _scope("Summarise the alp_acquired book", semantics, registry)
        assert claim.portfolio_ids == ("alp_acquired",)
        assert claim.portfolio_label == "ALP Acquired Back Book"
        assert claim.raw_text == "alp_acquired"
        # ... and the span points at the words that actually appear.
        assert claim.span is not None
        assert claim.span.text_of("Summarise the alp_acquired book") == "alp_acquired"

    def test_a_named_book_does_not_arrive_as_its_category(self, semantics, registry):
        """The 1D failure at the contract boundary: a consumer reading
        `scope=acquired` for a question about ONE acquired book would plan over
        both of them."""
        claim = _scope("Summarise the NBS Acquired Book", semantics, registry)
        assert claim.scope != SCOPE_ACQUIRED
        assert claim.portfolio_ids == ("nbs_acquired",)

    def test_a_category_still_arrives_as_a_category(self, semantics, registry):
        claim = _scope("Summarise the acquired book", semantics, registry)
        assert claim.state == FILLED
        assert claim.scope == SCOPE_ACQUIRED
        assert claim.portfolio_ids == ()


class TestAnUnheldNameIsUnresolvableNotTotal:
    @pytest.mark.parametrize("question,requested", [
        ("Summarise the Highgate Mortgages Book", "Highgate Mortgages Book"),
        ("Summarise the acquired_001 book", "acquired_001"),
    ])
    def test_an_unheld_name_is_unresolvable(self, semantics, registry,
                                            question, requested):
        claim = _scope(question, semantics, registry)
        assert claim.state == UNRESOLVABLE
        assert claim.scope is None
        assert claim.raw_text == requested
        assert requested in (claim.reason or "")

    @pytest.mark.parametrize("question", [
        "Summarise the Highgate Mortgages Book",
        "Summarise the acquired_001 book",
    ])
    def test_an_unheld_name_never_reads_as_no_narrowing(self, semantics,
                                                        registry, question):
        """UNRESOLVABLE and `scope=total` are the two readings this contract
        exists to keep apart. `narrows` is False for both — which is why a
        consumer must branch on `state`, and why this asserts `state` and not
        `narrows`."""
        claim = _scope(question, semantics, registry)
        assert claim.scope != SCOPE_TOTAL
        assert claim.state != FILLED


class TestTheSchemaRefusesAnIdentitylessCohort:
    def test_a_filled_cohort_claim_must_carry_a_governed_id(self):
        with pytest.raises(ValueError):
            SourceScopeClaim(state=FILLED, scope=SCOPE_COHORT)

    def test_a_cohort_claim_with_an_id_is_accepted(self):
        claim = SourceScopeClaim(state=FILLED, scope=SCOPE_COHORT,
                                 portfolio_ids=("nbs_acquired",))
        assert claim.narrows is True
        assert claim.as_dict()["portfolio_ids"] == ["nbs_acquired"]


class TestWithoutARegistryNothingChanged:
    """The registry parameter ADDS resolution. A caller that does not pass one
    must get precisely the pre-1E claim, or this phase changed answers by
    omission rather than by decision."""

    def test_a_named_book_without_a_registry_is_still_total(self, semantics):
        """"ALP Origination Book" carries no category vocabulary, so before 1E
        nothing in the question was recognised at all and the claim was Total.
        (A label that HAPPENS to contain a category word — "NBS Acquired Book" —
        resolved to that category, which is the collapse 1E closes; it is not
        the no-registry case this pins.)"""
        claim = _scope("Summarise the ALP Origination Book", semantics)
        assert claim.state == FILLED
        assert claim.scope == SCOPE_TOTAL

    def test_a_category_without_a_registry_is_unchanged(self, semantics):
        claim = _scope("Summarise the acquired book", semantics)
        assert claim.state == FILLED
        assert claim.scope == SCOPE_ACQUIRED
