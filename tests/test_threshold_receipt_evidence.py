"""tests/test_threshold_receipt_evidence.py — a threshold is certified by evidence.

`reconcile_routed_facets` stamped every KIND_THRESHOLD facet LOST
unconditionally, consulting nothing — on the same path where a sibling
KIND_POPULATION facet for the SAME predicate is stamped APPLIED from
`metadata.populationApplied`. Funded evolution narrowed correctly per period,
published the narrowing, and the answer refused anyway.

The threshold facet carries no field, operator or value structurally
(`field_key` is None for every comparator form), and the ledger carries only the
field — so the two cannot be matched against each other, and a rule that tried
would be parsing `label`. The proof used instead is an executor invariant:
`_apply_filters` applies EVERY `spec.filters` entry or raises, and appends each
field it narrowed on. Every governed predicate present in `applied`, none
`unavailable`, and at least as many predicates as thresholds ⇒ the narrowing ran.

Fail-closed everywhere else — that is what the negative controls pin.
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

from mi_agent import execution_receipt as R  # noqa: E402

LTV = "current_loan_to_value"
AGE = "youngest_borrower_age"


def _env(applied=(LTV,), unavailable=(), filters=None, spec=True):
    """An envelope shaped exactly like a routed evolution answer."""
    body = {"metadata": {"populationApplied": {
        "applied": [f"{a} (applied within each period)" for a in applied],
        "unavailable": list(unavailable), "rowsBefore": None, "rowsAfter": 1889}}}
    if spec:
        body["spec"] = {"filters": filters if filters is not None
                        else {LTV: {"op": "gt", "value": 50.0}}}
    return body


@pytest.fixture(scope="module")
def semantics():
    from migration_phase0.assurance_semantics import load_assurance_semantics
    return load_assurance_semantics()


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def test_a_proven_threshold_is_certified(semantics):
    assert R.threshold_execution_proven(_env(), semantics, 1) is True


def test_no_ledger_at_all_stays_lost(semantics):
    """NEGATIVE CONTROL: threshold requested, populationApplied absent."""
    assert R.threshold_execution_proven({"spec": {"filters": {LTV: {"op": "gt", "value": 50.0}}}},
                                        semantics, 1) is False
    assert R.threshold_execution_proven({}, semantics, 1) is False
    assert R.threshold_execution_proven(None, semantics, 1) is False


def test_evidence_for_a_different_field_stays_lost(semantics):
    """NEGATIVE CONTROL: the narrowing that ran is not the one asked for."""
    env = _env(applied=(AGE,), filters={LTV: {"op": "gt", "value": 50.0}})
    assert R.threshold_execution_proven(env, semantics, 1) is False


def test_an_unavailable_narrowing_stays_lost(semantics):
    """NEGATIVE CONTROL: a requested narrowing that could not be applied."""
    env = _env(applied=(LTV,), unavailable=("some_missing_column",))
    assert R.threshold_execution_proven(env, semantics, 1) is False


def test_a_spec_with_no_material_predicate_stays_lost(semantics):
    """NEGATIVE CONTROL: the threshold never resolved into a governed predicate,
    so there is nothing execution could have applied."""
    assert R.threshold_execution_proven(_env(filters={}), semantics, 1) is False
    assert R.threshold_execution_proven(_env(spec=False), semantics, 1) is False


def test_two_thresholds_one_predicate_stays_lost(semantics):
    """NEGATIVE CONTROL, and the subtlest: a question naming two bounds where
    only one resolved must not have the unresolved one certified by the
    resolved one's evidence."""
    env = _env(applied=(LTV,), filters={LTV: {"op": "gt", "value": 50.0}})
    assert R.threshold_execution_proven(env, semantics, 1) is True
    assert R.threshold_execution_proven(env, semantics, 2) is False


def test_two_thresholds_both_proven_are_certified(semantics):
    env = _env(applied=(LTV, AGE),
               filters={LTV: {"op": "gt", "value": 50.0},
                        AGE: {"op": "gt", "value": 75.0}})
    assert R.threshold_execution_proven(env, semantics, 2) is True


def test_two_predicates_only_one_applied_stays_lost(semantics):
    env = _env(applied=(LTV,),
               filters={LTV: {"op": "gt", "value": 50.0},
                        AGE: {"op": "gt", "value": 75.0}})
    assert R.threshold_execution_proven(env, semantics, 2) is False


def test_a_question_with_no_threshold_is_untouched(semantics):
    """NEGATIVE CONTROL: nothing requested, nothing certified."""
    assert R.threshold_execution_proven(_env(), semantics, 0) is False


def test_spec_presence_alone_is_never_evidence(semantics):
    """The spec says what was ASKED; only the ledger says what RAN."""
    import ast
    import inspect

    src = inspect.getsource(R.threshold_execution_proven)
    assert "populationApplied" in src

    # The rule cannot read the facet's wording because it is never given the
    # facet: its inputs are the envelope, the semantics and a count. Asserted
    # structurally rather than by searching the source text — an earlier cut of
    # this test searched for "label" and matched its own docstring.
    params = list(inspect.signature(R.threshold_execution_proven).parameters)
    assert params == ["envelope", "semantics", "threshold_count"], params

    body = ast.parse(inspect.getsource(R.threshold_execution_proven).lstrip())
    names = {n.attr for n in ast.walk(body) if isinstance(n, ast.Attribute)}
    for prose in ("label", "answer", "question", "reason"):
        assert prose not in names, prose


# --------------------------------------------------------------------------- #
# End to end, on the live governed book
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ask():
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    client = TestClient(app)

    def _ask(q):
        return client.post("/mi/query",
                           json={"question": q, "portfolioId": cfg.CLIENT_ID}).json()
    return _ask


def _series(r):
    charts = [a for a in (r.get("artifacts") or []) if a.get("type") == "chart"]
    return [(x.get("period"), x.get("value")) for x in ((charts[0].get("rows") if charts else []) or [])]


#: Pinned from the governed book. Every period differs, so a fixture where all
#: periods coincide cannot make these pass vacuously.
PINNED = {
    "Show funded balance evolution by month for loans above 50% LTV.":
        ([("2026-04", 432425355.78999996), ("2026-05", 450969362.11),
          ("2026-06", 472527483.38)], "1721, 1799, 1889"),
    "Show funded balance evolution by month for borrowers over 75.":
        ([("2026-04", 565452027.47), ("2026-05", 575304529.1700001),
          ("2026-06", 588411793.0699999)], "2648, 2682, 2722"),
    "Show funded balance evolution by month for loans above 200000.":
        ([("2026-04", 1031317551.54), ("2026-05", 1047465870.6299999),
          ("2026-06", 1064930912.2199999)], "3555, 3610, 3666"),
}


@pytest.mark.parametrize("question", sorted(PINNED))
def test_filtered_evolution_delivers_with_pinned_per_period_values(ask, question):
    expected_series, expected_rows = PINNED[question]
    r = ask(question)
    assert r.get("ok") is True, r.get("answer")
    assert _series(r) == expected_series
    note = next((n.get("note") for n in (r.get("sourceNotes") or [])
                 if n.get("field") == "filter"), "")
    assert expected_rows in note, note
    # Non-vacuous: the filtered population genuinely moves between periods.
    assert len({v for _p, v in expected_series}) == 3


def test_a_filtered_count_series_equals_the_declared_row_counts(ask):
    """Cross-check that the certified answer is the same computation the ledger
    describes — the receipt fix must not have touched the economics."""
    r = ask("Show loan count evolution by month for loans above 50% LTV.")
    assert r.get("ok") is True
    assert [v for _p, v in _series(r)] == [1721, 1799, 1889]


def test_unfiltered_evolution_is_unchanged(ask):
    r = ask("Show funded balance evolution by month.")
    assert r.get("ok") is True
    assert _series(r) == [("2026-04", 1932310991.2), ("2026-05", 1946827440.6),
                          ("2026-06", 1964886258.21)]


def test_a_geographic_scope_refusal_is_a_different_owner_and_still_refuses(ask):
    """Authorised blast excluded geography: it refuses through
    KIND_GEOGRAPHIC_SCOPE, not the threshold branch."""
    r = ask("Show funded balance evolution by month for London.")
    assert r.get("ok") is False
    assert "geographic scope" in (r.get("answer") or "")


# --------------------------------------------------------------------------- #
# The BRANCH, not just the rule
# --------------------------------------------------------------------------- #
# Mutation testing found this gap: replacing the branch condition with `if True`
# — restoring the unconditional stamp, in the opposite direction — left all
# sixteen tests above green, because every negative control exercised
# `threshold_execution_proven` directly and none went through
# `reconcile_routed_facets`. A control that cannot see the branch it guards is
# not a control.
def _threshold_facet(label="LTV over 50"):
    return R.RequestedFacet(kind=R.KIND_THRESHOLD, label=label)


def _reconcile(facets, envelope, semantics):
    return R.reconcile_routed_facets(list(facets), route="evolution",
                                     semantics=semantics, available_columns=(),
                                     envelope=envelope)


def test_branch_certifies_a_threshold_backed_by_evidence(semantics):
    out = _reconcile([_threshold_facet()], _env(), semantics)
    assert [f.status for f in out] == [R.APPLIED], [(f.label, f.status, f.reason) for f in out]


def test_branch_keeps_a_threshold_lost_without_any_ledger(semantics):
    env = {"spec": {"filters": {LTV: {"op": "gt", "value": 50.0}}}}
    out = _reconcile([_threshold_facet()], env, semantics)
    assert [f.status for f in out] == [R.LOST]
    assert "does not apply a value threshold" in out[0].reason


def test_branch_keeps_a_threshold_lost_on_wrong_field_evidence(semantics):
    env = _env(applied=(AGE,), filters={LTV: {"op": "gt", "value": 50.0}})
    out = _reconcile([_threshold_facet()], env, semantics)
    assert [f.status for f in out] == [R.LOST]


def test_branch_keeps_a_threshold_lost_when_a_narrowing_was_unavailable(semantics):
    env = _env(applied=(LTV,), unavailable=("missing_column",))
    out = _reconcile([_threshold_facet()], env, semantics)
    assert [f.status for f in out] == [R.LOST]


def test_branch_keeps_both_lost_when_only_one_bound_resolved(semantics):
    env = _env(applied=(LTV,), filters={LTV: {"op": "gt", "value": 50.0}})
    out = _reconcile([_threshold_facet("LTV over 50"),
                      _threshold_facet("borrower age over 75")], env, semantics)
    assert [f.status for f in out] == [R.LOST, R.LOST]


def test_branch_keeps_a_threshold_lost_with_no_spec_filters(semantics):
    out = _reconcile([_threshold_facet()], _env(filters={}), semantics)
    assert [f.status for f in out] == [R.LOST]
