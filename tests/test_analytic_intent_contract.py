"""The governed ANALYTIC claim, and the route entitlement that reads it.

The question this closes: `OperationClaim.type` says what KIND of answer is
wanted; it does not say whether the reader named an ANALYSIS. Measured before
this field existed, these two carried identical contracts —

    "Which region has the largest balance?"                 (generic ranking)
    "What is the largest geographic area concentration?"    (specialist)

— on every governed field: type, all four ordering values, modifiers, the
subject claim, the dimension claims and residue. So route ownership fell back to
wording tests inside the routing layer, and a plain ranked stratification of a
governed dimension was claimed by the ITL3 exposure engine, which cannot answer
it on a tape carrying region names.

`analytic` is generic by construction: it names a SHAPE OF ANALYSIS, not a
dimension family, and its vocabulary belongs to
`mi_workflows.concentration_analysis` — the module that already owned it.
"""

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mi_workflows.concentration_analysis import names_a_concentration_analytic
from question_interpretation.schema import (
    ANALYTIC_CONCENTRATION, ANALYTICS, OperationClaim, RANKING,
)

GENERIC = [
    "Which region has the largest balance?",
    "Which region has the smallest balance?",
    "What are the top three regions by balance?",
    "Which broker channel has the largest balance?",
    "Which product type has the largest balance?",
    "Show balance by region.",
    "Which region added the most balance since last month?",
    "What is the total funded balance?",
    "Top 5 regions by balance",
    "What is our exposure to Wales?",
]
SPECIALIST = [
    "Where is the book concentrated geographically?",
    "Show geographic exposure.",
    "Analyse geographic concentration.",
    "What is the largest geographic area concentration?",
    "Which area has the largest concentration?",
    "Where are we most exposed geographically?",
    "Show broker concentration.",
    "Show product concentration.",
    "Show concentration.",
]


class TestTheOwnerSeparatesTheShapes(unittest.TestCase):
    """The reading itself, before any route precedence is applied."""

    def test_a_ranked_stratification_names_no_analytic(self):
        for q in GENERIC:
            self.assertIsNone(names_a_concentration_analytic(q), q)

    def test_a_concentration_or_exposure_question_names_one(self):
        for q in SPECIALIST:
            self.assertIsNotNone(names_a_concentration_analytic(q), q)

    def test_the_reading_carries_no_precedence(self):
        """`is_concentration_question` answers "does the concentration ROUTE own
        this", which folds in who else might. Every geographic question is
        rejected by that test before its concentration language is ever read —
        which is why the analytic reading had to be separable."""
        from mi_workflows.concentration_analysis import is_concentration_question

        q = "Where is the book concentrated geographically?"
        self.assertIsNotNone(names_a_concentration_analytic(q))
        self.assertFalse(is_concentration_question(q)[0])

    def test_exposure_TO_a_value_is_not_an_exposure_analytic(self):
        """Bounded exactly as the mix construction is: a FAMILY word, not a
        value. "What is our exposure to Wales?" stays with the point-in-time
        path."""
        self.assertIsNone(names_a_concentration_analytic(
            "What is our exposure to Wales?"))


class TestTheContractCarriesIt(unittest.TestCase):
    def test_the_vocabulary_is_controlled(self):
        self.assertEqual(ANALYTICS, (ANALYTIC_CONCENTRATION,))
        with self.assertRaises(ValueError):
            OperationClaim(analytic="not-an-analytic")

    def test_none_means_no_analytic_was_named_not_a_negative(self):
        self.assertIsNone(OperationClaim().analytic)

    def test_it_is_orthogonal_to_the_operation_type(self):
        """A question may name a concentration analytic AND rank it. Both facts
        have to survive — "the LARGEST geographic area CONCENTRATION" is the
        case that forced the field to be separate from `type`."""
        claim = OperationClaim(state="filled", type=RANKING,
                               analytic=ANALYTIC_CONCENTRATION)
        self.assertEqual(claim.type, RANKING)
        self.assertEqual(claim.analytic, ANALYTIC_CONCENTRATION)
        self.assertEqual(claim.as_dict()["analytic"], ANALYTIC_CONCENTRATION)


class TestTheEntitlementRuleReadsTheContract(unittest.TestCase):
    """`_is_a_generic_ranking` decides ENTITLEMENT, and only from the contract."""

    @staticmethod
    def _interpretation(op_type, analytic, grouping=True):
        from question_interpretation.schema import (
            DimensionClaim, FILLED, GROUPING, QuestionInterpretation,
            UNRESOLVED_ROLE,
        )
        qi = QuestionInterpretation(question="q")
        qi.operation = OperationClaim(state=FILLED, type=op_type,
                                      analytic=analytic)
        qi.dimensions = [DimensionClaim(
            state=FILLED, candidate_concept="geographic_region_obligor",
            role=GROUPING if grouping else UNRESOLVED_ROLE)]
        return qi

    def _rule(self):
        from mi_agent_api.chat_routing import _is_a_generic_ranking
        return _is_a_generic_ranking

    def test_a_ranking_with_no_analytic_is_generic(self):
        self.assertTrue(self._rule()(self._interpretation(RANKING, None)))

    def test_a_ranking_that_names_an_analytic_is_the_specialists(self):
        self.assertFalse(self._rule()(
            self._interpretation(RANKING, ANALYTIC_CONCENTRATION)))

    def test_a_non_ranking_is_not_released(self):
        self.assertFalse(self._rule()(self._interpretation("amount", None)))

    def test_a_ranking_with_no_axis_is_not_released(self):
        """A ranking with no grouping dimension is not a stratification of
        anything, and the specialist route keeps it."""
        self.assertFalse(self._rule()(
            self._interpretation(RANKING, None, grouping=False)))

    def test_no_contract_releases_nothing(self):
        self.assertFalse(self._rule()(None))

    def test_the_rule_names_no_dimension_family(self):
        """Generic by construction: no geography, product or broker word may
        appear in the entitlement rule's executable body."""
        import ast
        import inspect

        src = inspect.getsource(self._rule())
        tree = ast.parse(src.lstrip())
        doc = ast.get_docstring(tree.body[0], clean=False) or ""
        literals = " ".join(
            n.value.lower() for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value != doc)
        for token in ("geograph", "region", "itl3", "postcode", "broker",
                      "product"):
            self.assertNotIn(token, literals, token)


if __name__ == "__main__":
    unittest.main()
