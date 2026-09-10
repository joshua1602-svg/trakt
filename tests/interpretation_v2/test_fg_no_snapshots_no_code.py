"""F, G · The model cannot state a snapshot, a date, or any executable code.

Time is stated SEMANTICALLY and resolved by the governed period contract against
a book's actual history. A model that could pin a snapshot would be deciding
which data an answer covers, from a prompt that contains no data.

Code is refused outright rather than stripped. A payload that tried to reach
past the boundary does not get a tidier second chance.
"""

from __future__ import annotations

import pytest

from mi_agent.interpretation_v2 import IntentParseError, parse_candidate_intent

from .conftest import intent_payload


# --------------------------------------------------------------------------- #
# F · snapshots and dates
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("label", [
    "2026-06-30", "30/06/2026", "20260630", "snapshot_17", "snapshot: abc",
    "as_of_2026",
])
def test_a_date_or_snapshot_in_a_period_label_refuses(label):
    payload = intent_payload(time={"form": "explicit_period", "labels": [label]})
    with pytest.raises(IntentParseError) as excinfo:
        parse_candidate_intent(payload)
    assert excinfo.value.code == "MODEL_OUTPUT_CONTAINS_PHYSICAL_BINDING"


def test_a_date_anywhere_in_the_payload_refuses():
    payload = intent_payload(
        filters=[{"concept": "origination_date", "comparator": "gte",
                  "value": "2026-01-01"}])
    with pytest.raises(IntentParseError) as excinfo:
        parse_candidate_intent(payload)
    assert excinfo.value.code == "MODEL_OUTPUT_CONTAINS_PHYSICAL_BINDING"


@pytest.mark.parametrize("label", ["last month", "April", "the quarter",
                                   "this year", "the last few months"])
def test_a_period_the_question_named_in_words_is_allowed(label):
    """The words are a LABEL, not a resolution. The compiler hands them to the
    governed period contract, which decides whether the book reaches that far."""
    payload = intent_payload(time={"form": "explicit_period", "labels": [label]})
    intent = parse_candidate_intent(payload)
    assert intent.time.labels == (label,)


def test_the_plan_carries_a_contract_not_a_snapshot(compiler):
    from .conftest import build_intent

    intent = build_intent(time={"form": "previous_reporting_period"})
    result = compiler.compile(intent)
    period = result.plan.period
    assert period.contract == "previous_governed_reporting_period"
    # Nothing on the period is a date. This package has no book to resolve one
    # against, and inventing one would be deciding what the answer covers.
    assert not any(char.isdigit() for char in period.contract)
    assert period.labels == ()


# --------------------------------------------------------------------------- #
# G · code, SQL and dataframe expressions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("payload_kwargs", [
    {"measures": [{"concept": "df['balance'].sum()"}]},
    {"measures": [{"concept": "import pandas"}]},
    {"dimensions": ["```python"]},
    {"filters": [{"concept": "balance", "comparator": "gt",
                  "value": "df.loc[df.ltv > 50]"}]},
    {"filters": [{"concept": "balance", "comparator": "eq",
                  "value": "select sum(x) from loans"}]},
    {"time": {"form": "current", "labels": ["lambda x: x"]}},
    {"comparison": {"kind": "population_pair", "left": "np.mean",
                    "right": "acquired"}},
])
def test_code_anywhere_in_a_binding_slot_refuses(payload_kwargs):
    with pytest.raises(IntentParseError) as excinfo:
        parse_candidate_intent(intent_payload(**payload_kwargs))
    assert excinfo.value.code == "MODEL_OUTPUT_CONTAINS_CODE"


def test_code_in_quoted_evidence_still_refuses():
    """A fenced code block is code wherever it appears."""
    payload = intent_payload(evidence=[{"claim": "x", "text": "```sql\nSELECT 1"}])
    with pytest.raises(IntentParseError) as excinfo:
        parse_candidate_intent(payload)
    assert excinfo.value.code == "MODEL_OUTPUT_CONTAINS_CODE"


def test_the_users_own_words_are_not_code_when_quoted_back():
    """"Count drawdown cases WHERE current LTV exceeds 50%" is a question.

    SQL keywords are also ordinary English. Refusing them in a slot that binds
    nothing would reject correct readings of ordinary sentences — which it did,
    on the first live run, before this distinction existed.
    """
    payload = intent_payload(
        evidence=[{"claim": "ltv predicate",
                   "text": "where current LTV exceeds 50%"}],
        ambiguity=[{"slot": "filters", "blocking": False,
                    "note": "read 'from April' as the period, not a join"}])
    intent = parse_candidate_intent(payload)
    assert intent.evidence[0].text == "where current LTV exceeds 50%"


def test_a_sql_keyword_in_a_binding_slot_still_refuses():
    payload = intent_payload(dimensions=["group by region"])
    with pytest.raises(IntentParseError) as excinfo:
        parse_candidate_intent(payload)
    assert excinfo.value.code == "MODEL_OUTPUT_CONTAINS_CODE"


def test_a_nested_predicate_object_is_not_a_filter_value():
    """A predicate tree is a query. The contract takes scalars and flat lists."""
    payload = intent_payload(
        filters=[{"concept": "balance", "comparator": "eq",
                  "value": {"op": "and", "args": [1, 2]}}])
    with pytest.raises(IntentParseError) as excinfo:
        parse_candidate_intent(payload)
    assert excinfo.value.code == "INTENT_SCHEMA_INVALID"


def test_the_generated_schema_forbids_extra_properties():
    """Strict structured generation cannot produce a binding key in the first
    place; the parser is the second layer, not the only one."""
    from mi_agent.interpretation_v2 import candidate_intent_json_schema

    def assert_closed(node, path="root"):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, (
                    f"{path} allows additional properties")
            for key, value in node.items():
                assert_closed(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                assert_closed(value, f"{path}[{index}]")

    assert_closed(candidate_intent_json_schema())
