"""RECOVERY GATE E — "by balance" is a MEASURE, not always a WEIGHTING.

    "What does the loan-to-value distribution of the book look like by balance?"

The reader wants BALANCE distributed across LTV bands. The candidate build
answered the weighted-average LTV within each LTV band — a tautology, since the
mean LTV inside the 40-50% band is 0.45 — and published it as the distribution.
Its two sibling phrasings, "Show funded balance by LTV band" and "Split the
balance into LTV buckets", both measure balance, so the same book answers the
same question two ways depending on the wording.

WHAT THIS FILE DOES NOT DO. It states the rule about the COMPOSITION —
distribution noun + dimension + "by <measure>" — not about the sentence. No
question from the frozen bank is encoded as a special case; the bank is an exam.
The paraphrases below are here to prove the rule is stable across wording, which
is the property that was lost.
"""
from __future__ import annotations

import pytest

#: Measures that mean "the money", whichever spelling the registry resolves to.
BALANCE_MEASURES = frozenset({"current_outstanding_balance", "funded_balance",
                              "outstanding_balance", "balance"})

#: What "distribution ... by balance" must NOT come out as: the dimension's own
#: underlying field, weighted by the thing that should have been the measure.
LTV_MEASURE = "current_loan_to_value"


def _parse(question, semantics):
    from mi_agent.llm_query_parser import _deterministic_parse

    spec, _meta = _deterministic_parse(question, semantics)
    data = spec.to_dict()
    return data.get("metric"), data.get("aggregation"), data.get("dimension")


class TestADistributionByBalanceMeasuresBalance:
    """A DISTRIBUTION NAMES ITS AXIS AND ITS MEASURE SEPARATELY.

    "the LTV distribution ... by balance" names the axis (LTV bands) and the
    measure (balance). "on a balance-weighted basis, what LTV ..." names one
    measure (LTV) and how to average it (by balance). The two readings share the
    words "by balance" and mean opposite things, and the difference is whether a
    DIMENSION is present for the balance to be distributed across.
    """

    @pytest.mark.parametrize("question", [
        "What does the loan-to-value distribution of the book look like by balance?",
        "What does the LTV distribution look like by balance?",
        "What does the LTV distribution of the book look like by balance?",
    ])
    def test_the_measure_is_the_money_not_the_axis(self, semantics, question):
        metric, aggregation, dimension = _parse(question, semantics)

        assert metric != LTV_MEASURE or aggregation != "weighted_avg", (
            f"{question!r} parsed as the weighted-average of the very field the "
            f"bands are cut from (metric={metric!r}, aggregation="
            f"{aggregation!r}) — the mean LTV inside an LTV band is the band, "
            f"so this answer can only restate the axis")
        assert metric in BALANCE_MEASURES, (
            f"{question!r} asks for the distribution BY BALANCE; it parsed "
            f"metric={metric!r}, aggregation={aggregation!r}, "
            f"dimension={dimension!r}")

    @pytest.mark.parametrize("question", [
        "Show funded balance by LTV band.",
        "Split the balance into LTV buckets.",
    ])
    def test_the_sibling_phrasings_are_unchanged(self, semantics, question):
        """THE CONTROL. These two already read correctly and must stay that way."""
        metric, aggregation, dimension = _parse(question, semantics)

        assert metric in BALANCE_MEASURES
        assert aggregation == "sum"
        assert dimension == "ltv_bucket"

    @pytest.mark.parametrize("question", [
        "On a balance-weighted basis, what loan-to-value is the portfolio running at?",
        "What is the weighted average current LTV?",
        "Give me the book's weighted average LTV.",
    ])
    def test_a_genuine_weighting_qualifier_still_weights(self, semantics, question):
        """THE OTHER CONTROL. Recovering the distribution reading must not cost
        the weighted-average reading, which the same sprint fixed and which the
        live evidence recorded as an improvement."""
        metric, aggregation, _dimension = _parse(question, semantics)

        assert metric == LTV_MEASURE, (
            f"{question!r} names one measure — the LTV — and how to average it")
        assert aggregation == "weighted_avg"


class TestTheAnswerNeverRestatesItsOwnAxis:
    """A GROUPED ANSWER WHOSE CELLS ARE ITS OWN BUCKET MIDPOINTS SAYS NOTHING.

    Stated structurally so it holds for any dimension cut from a measure, not
    only LTV: if the grouped measure is the field the dimension is derived from
    and the aggregation is an average, the cells are the buckets.
    """

    @pytest.mark.parametrize("question", [
        "What does the loan-to-value distribution of the book look like by balance?",
    ])
    def test_the_grouped_measure_is_not_the_field_the_buckets_come_from(
            self, semantics, question):
        metric, aggregation, dimension = _parse(question, semantics)
        if dimension != "ltv_bucket":
            return
        assert not (metric == LTV_MEASURE and aggregation in
                    {"weighted_avg", "avg", "mean"}), (
            "the answer would group the LTV by LTV band and average it, so "
            "every cell is the midpoint of its own band")
