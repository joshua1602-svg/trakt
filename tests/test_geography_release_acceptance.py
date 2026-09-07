#!/usr/bin/env python3
"""The release acceptance SCORER, proven to fail when it should.

A gate is only worth the failures it catches, and a scorer nobody has tried to
fool is a scorer that passes. Each test here takes a set of frozen response
envelopes that WOULD be accepted, breaks exactly one thing, and asserts the
named check goes red — so a future edit that loosens a check fails here rather
than in production, silently.

NOTHING HERE CALLS PRODUCTION. The envelopes are frozen representations of what
the deployed service publishes, and the scorer is driven with a dict-backed
`ask`. That is the point: the scorer must be testable without the thing it
scores, or it can only be tested by the outage it was built to prevent.

The figures are the ERE/2026-06-30 snapshot, so these also exercise the shipped
`geography_snapshot.json` contract rather than a convenient fixture of their own.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from due_diligence.evidence.mi_api_certification.certify_mi_api import (  # noqa: E402
    ACCEPTANCE_CHECKS,
    GEOGRAPHY_QUESTIONS,
    acceptance_verdict,
    geography_acceptance,
    load_snapshot,
)

PORTFOLIO = "ERE/2026-06-30"

#: The snapshot truth, restated here ONLY so the fixtures agree with it. The
#: scorer reads it from the shipped contract, not from this.
TOTAL = 1964886258.21
RECORDS = 11035
REGIONS = {
    "South East": (516214136.58, 2420),
    "London": (413804467.49, 1380),
    "Rest of UK": (970537197.11, 6665),
    "Scotland": (64330457.03, 570),
}

COLLATERAL_FIELD = "collateral_geography"
BORROWER_FIELD = "geographic_region_obligor"


def _geo(primary: str = "collateral", source: str = "asset_class_default",
         configured: Any = "__same__", asset_class: str = "equity_release"
         ) -> Dict[str, Any]:
    block: Dict[str, Any] = {"primaryBasis": primary, "basisSource": source,
                             "assetClass": asset_class,
                             "supportedBases": ["borrower", "collateral"]}
    if configured != "__same__":
        # The envelope publishes `configuredBasis` only when the question
        # overrode the configuration — which is exactly when there are two
        # bases to tell apart.
        block["configuredBasis"] = configured
        block["configuredBasisSource"] = "asset_class_default"
    return block


def _grouped(field: str, value_key: str, values: Dict[str, float],
             geo: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "answer": f"Here is the bar for your query, covering {len(values)} groups.",
        "metadata": {"portfolioId": PORTFOLIO, "geographyBasis": geo},
        "spec": {"dimension": field, "dimensions": [field], "filters": {},
                 "unavailable_filters": []},
        "executionSummary": {"population": RECORDS,
                             "dimensionsApplied": ["Region"]},
        "reconciliation": {"total_balance": TOTAL, "total_records": RECORDS,
                           "balance_after_filters": TOTAL,
                           "records_after_filters": RECORDS, "filters": {}},
        "artifacts": [{"type": "table",
                       "rows": [{field: name, value_key: value}
                                for name, value in values.items()]}],
    }


def _scalar(balance: float, count: int, geo: Dict[str, Any],
            filters: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "answer": f"Balance: £{balance:,.2f} · {count:,} loans.",
        "metadata": {"portfolioId": PORTFOLIO, "geographyBasis": geo},
        "spec": {"dimension": None, "dimensions": [],
                 "filters": filters or {}, "unavailable_filters": []},
        "executionSummary": {"population": count, "populationTotal": RECORDS},
        "reconciliation": {"total_balance": TOTAL, "total_records": RECORDS,
                           "balance_after_filters": balance,
                           "records_after_filters": count,
                           "filters": filters or {}},
        "artifacts": [{"type": "kpi", "kpis": [
            {"field": "loan_count", "label": "Loan", "rawValue": float(count)},
            {"field": "current_outstanding_balance_sum", "label": "Balance",
             "rawValue": balance}]}],
    }


def _baseline() -> Dict[str, Dict[str, Any]]:
    """Ten envelopes that SHOULD be accepted, keyed by question id."""
    balances = {name: value for name, (value, _) in REGIONS.items()}
    counts = {name: float(count) for name, (_, count) in REGIONS.items()}
    scots_balance, scots_count = REGIONS["Scotland"]

    envelopes = {
        "G01": _scalar(TOTAL, RECORDS, _geo()),
        "G02": _grouped(COLLATERAL_FIELD, "current_outstanding_balance_sum",
                        balances, _geo()),
        "G03": _scalar(scots_balance, scots_count, _geo(),
                       {COLLATERAL_FIELD: "Scotland"}),
        "G04": _scalar(scots_balance, scots_count, _geo(),
                       {COLLATERAL_FIELD: "Scotland"}),
        "G05": _grouped(COLLATERAL_FIELD, "loan_count", counts, _geo()),
        "G06": _grouped(COLLATERAL_FIELD, "current_outstanding_balance_sum",
                        balances,
                        _geo("collateral", "explicit_query", "collateral")),
        "G07": _grouped(BORROWER_FIELD, "current_outstanding_balance_sum",
                        balances,
                        _geo("borrower", "explicit_query", "collateral")),
        # The live book cannot apply the joint-borrower facet and SAYS SO. A
        # refusal that names what it dropped is correct behaviour; answering
        # over a broader population is the failure.
        "G08": {"ok": False,
                "answer": "I understood that you asked for joint borrower, but "
                          "that could not be applied to the calculation.",
                "metadata": {"portfolioId": PORTFOLIO, "geographyBasis": _geo()},
                "spec": {"dimension": COLLATERAL_FIELD, "filters": {},
                         "unavailable_filters":
                             ["borrower_structure is not in this dataset"]},
                "executionSummary": {}, "reconciliation": {}, "artifacts": []},
        "G09": None,   # built below: two measures on one grouping
        "G10": {"ok": False,
                "answer": "No loans in this book match that filter "
                          "('platinum'), so there is nothing to calculate. I "
                          "have not returned a whole-book figure in its place.",
                "metadata": {"portfolioId": PORTFOLIO, "geographyBasis": _geo()},
                "spec": {"dimension": COLLATERAL_FIELD, "filters": {},
                         "unavailable_filters": []},
                "executionSummary": {}, "reconciliation": {}, "artifacts": []},
    }
    two_measures = _grouped(COLLATERAL_FIELD, "current_outstanding_balance_sum",
                            balances, _geo())
    for row in two_measures["artifacts"][0]["rows"]:
        row["loan_count"] = REGIONS[row[COLLATERAL_FIELD]][1]
    envelopes["G09"] = two_measures
    return envelopes


def _ask(envelopes: Dict[str, Dict[str, Any]]) -> Callable[[str], Dict[str, Any]]:
    by_question = {question: envelopes[qid]
                   for qid, question in GEOGRAPHY_QUESTIONS}

    def ask(question: str) -> Dict[str, Any]:
        return copy.deepcopy(by_question[question])

    return ask


def _score(envelopes, **kwargs):
    return geography_acceptance(_ask(envelopes), portfolio_id=PORTFOLIO,
                                provenance=kwargs.pop("provenance", "YES"),
                                **kwargs)


# =========================================================================== #
# The baseline must pass, or every falsification below proves nothing
# =========================================================================== #
def test_the_frozen_release_envelopes_are_accepted():
    checks, rows, lines = _score(_baseline())
    failed = [name for name in ACCEPTANCE_CHECKS if not checks.get(name)]
    assert failed == [], "\n".join(lines)
    assert acceptance_verdict(checks) == ("YES", "all acceptance checks passed")
    assert len(rows) == len(GEOGRAPHY_QUESTIONS)


def test_the_shipped_snapshot_is_the_one_being_read():
    """The contract lives in the release harness and is keyed to the book."""
    snap = load_snapshot(PORTFOLIO)
    assert snap["expectedAssetClass"] == "equity_release"
    assert snap["expectedPrimaryBasis"] == "collateral"
    assert snap["totalBalance"] == TOTAL
    assert snap["regions"]["Scotland"] == {"loanCount": 570,
                                           "balance": 64330457.03}
    assert load_snapshot("SOMEONE_ELSE/2026-06-30") == {}


# =========================================================================== #
# One broken thing each
# =========================================================================== #
def test_generic_region_resolving_borrower_is_refused():
    """1. The whole architecture in one assertion: an unqualified "region" on
    an ERE book means the COLLATERAL geography, because that is what the ERM
    asset pack declares."""
    envelopes = _baseline()
    envelopes["G02"] = _grouped(
        BORROWER_FIELD, "current_outstanding_balance_sum",
        {name: value for name, (value, _) in REGIONS.items()},
        _geo("borrower", "asset_class_default"))
    checks, _, _ = _score(envelopes)
    assert checks["GENERIC_REGION_PASS"] is False
    assert acceptance_verdict(checks)[0] == "NO"


def test_an_unconfigured_basis_source_is_refused():
    """2. `unconfigured` is the exact symptom the config-ownership work fixed:
    the client file was named for the asset, so the lookup found nothing."""
    envelopes = _baseline()
    for qid in ("G02", "G05", "G09"):
        envelopes[qid]["metadata"]["geographyBasis"] = _geo(
            None, "unconfigured", asset_class=None)
    checks, _, _ = _score(envelopes)
    assert checks["GENERIC_REGION_PASS"] is False


def test_a_missing_or_wrong_asset_class_is_refused():
    """3. The asset class is what selects the pack. If the deployed service
    cannot name it, nothing downstream of it can be trusted."""
    envelopes = _baseline()
    for envelope in envelopes.values():
        block = envelope["metadata"]["geographyBasis"]
        block["assetClass"] = "auto_finance"
    checks, _, _ = _score(envelopes)
    assert checks["ASSET_CLASS_PASS"] is False

    envelopes = _baseline()
    for envelope in envelopes.values():
        envelope["metadata"]["geographyBasis"]["assetClass"] = None
    checks, _, _ = _score(envelopes)
    assert checks["ASSET_CLASS_PASS"] is False


def test_a_property_region_answer_that_is_not_explicit_is_refused():
    """4. "by property region" STATED a basis. An answer that reports it as the
    asset default has lost the fact that the question asked."""
    envelopes = _baseline()
    envelopes["G06"]["metadata"]["geographyBasis"] = _geo(
        "collateral", "asset_class_default")
    checks, _, _ = _score(envelopes)
    assert checks["EXPLICIT_PROPERTY_PASS"] is False


def test_a_borrower_region_answered_on_collateral_is_refused():
    """5. THE ONE THAT MATTERS MOST.

    The metadata says everything a passing run would say — stated basis
    honoured, source explicit_query — and the answer was measured on the
    collateral column anyway. A gate that read the metadata alone would call
    this a pass, which is why the scorer reads what was MEASURED.
    """
    envelopes = _baseline()
    envelopes["G07"] = _grouped(
        COLLATERAL_FIELD, "current_outstanding_balance_sum",
        {name: value for name, (value, _) in REGIONS.items()},
        _geo("borrower", "explicit_query", "collateral"))
    checks, rows, _ = _score(envelopes)
    assert checks["EXPLICIT_BORROWER_PASS"] is False
    borrower = next(r for r in rows if r["id"] == "G07")
    assert borrower["effectiveBasis"] == "borrower"
    assert borrower["measuredBasis"] == "collateral"


def test_a_governed_refusal_of_borrower_region_is_acceptable():
    """The counterpart: a book with no obligor geography must be free to say
    so. Refusing is not the failure — substituting is."""
    envelopes = _baseline()
    envelopes["G07"] = {
        "ok": False,
        "answer": "I understood that you asked for borrower region, but that "
                  "could not be applied to the calculation.",
        "metadata": {"portfolioId": PORTFOLIO,
                     "geographyBasis": _geo("borrower", "explicit_query",
                                            "collateral")},
        "spec": {"dimension": None, "filters": {},
                 "unavailable_filters": ["geographic_region_obligor is not in "
                                         "this dataset"]},
        "executionSummary": {}, "reconciliation": {}, "artifacts": []}
    checks, _, lines = _score(envelopes)
    assert checks["EXPLICIT_BORROWER_PASS"] is True, "\n".join(lines)


def test_a_scotland_cell_that_disagrees_with_the_scottish_question_is_refused():
    """6. One population computed two ways is one population."""
    envelopes = _baseline()
    envelopes["G03"] = _scalar(REGIONS["Scotland"][0] + 1_000_000.0,
                               REGIONS["Scotland"][1], _geo(),
                               {COLLATERAL_FIELD: "Scotland"})
    checks, _, _ = _score(envelopes)
    assert checks["NUMERICAL_RECONCILIATION_PASS"] is False


def test_a_scotland_count_that_disagrees_with_the_grouped_count_is_refused():
    """6b. The same identity for the cardinality."""
    envelopes = _baseline()
    envelopes["G04"] = _scalar(REGIONS["Scotland"][0], 999, _geo(),
                               {COLLATERAL_FIELD: "Scotland"})
    checks, _, _ = _score(envelopes)
    assert checks["NUMERICAL_RECONCILIATION_PASS"] is False


def test_regional_cells_that_do_not_sum_to_the_total_are_refused():
    """7. A breakdown that does not add up to the thing it breaks down is not
    a breakdown of it."""
    envelopes = _baseline()
    rows = envelopes["G02"]["artifacts"][0]["rows"]
    rows[0]["current_outstanding_balance_sum"] += 5_000_000.0
    checks, _, _ = _score(envelopes)
    assert checks["NUMERICAL_RECONCILIATION_PASS"] is False


def test_a_property_region_answer_that_disagrees_with_generic_region_is_refused():
    """ERE's configured basis IS collateral, so the two are the same question
    asked twice and must agree cell for cell."""
    envelopes = _baseline()
    envelopes["G06"]["artifacts"][0]["rows"][1][
        "current_outstanding_balance_sum"] = 1.0
    checks, _, _ = _score(envelopes)
    assert checks["NUMERICAL_RECONCILIATION_PASS"] is False


def test_a_confident_platinum_answer_is_refused():
    """8. "platinum" is a material qualifier no governed vocabulary claims. It
    must not quietly disappear leaving a whole-book figure behind."""
    envelopes = _baseline()
    envelopes["G10"] = _grouped(
        COLLATERAL_FIELD, "current_outstanding_balance_sum",
        {name: value for name, (value, _) in REGIONS.items()}, _geo())
    checks, _, _ = _score(envelopes)
    assert checks["SILENT_WRONG_SAFETY_PASS"] is False


def test_a_g09_answer_that_drops_a_requested_measure_is_refused():
    """Two outputs were asked for and two must come back."""
    envelopes = _baseline()
    for row in envelopes["G09"]["artifacts"][0]["rows"]:
        row.pop("loan_count")
    checks, _, _ = _score(envelopes)
    assert checks["NUMERICAL_RECONCILIATION_PASS"] is False


def test_a_g08_answer_that_silently_drops_a_facet_is_refused():
    """Answering "for joint borrowers with LTV over 50%" over a population that
    applied neither filter is the substitution this estate exists to stop."""
    envelopes = _baseline()
    envelopes["G08"] = _grouped(
        COLLATERAL_FIELD, "current_outstanding_balance_sum",
        {name: value for name, (value, _) in REGIONS.items()}, _geo())
    checks, _, _ = _score(envelopes)
    assert checks["NUMERICAL_RECONCILIATION_PASS"] is False


# =========================================================================== #
# Infrastructure is not a semantic verdict
# =========================================================================== #
def test_a_differing_deployed_sha_is_not_executable():
    """9. Certifying a build nobody can identify certifies nothing an operator
    can act on. Note the REASON: this is not "the geography is wrong"."""
    checks, _, _ = _score(_baseline(),
                          provenance="NO — serving abc123, expected def456")
    assert checks["DEPLOYED_PROVENANCE_PASS"] is False
    assert acceptance_verdict(checks) == ("NO", "NOT_EXECUTABLE")


def test_an_unestablished_deployed_sha_is_not_executable():
    checks, _, _ = _score(_baseline(),
                          provenance="NO — the service publishes no build stamp")
    assert acceptance_verdict(checks) == ("NO", "NOT_EXECUTABLE")


def test_an_auth_or_reachability_failure_is_not_a_semantic_failure():
    """10. Sending an operator to debug the geography configuration because a
    token expired is the wrong outcome, so it is a different verdict."""
    checks, _, _ = _score(_baseline(), reached="YES",
                          authorised="NO — HTTP 401")
    assert checks["AUTH_PASS"] is False
    assert acceptance_verdict(checks) == ("NO", "NOT_EXECUTABLE")

    checks, _, _ = _score(_baseline(), reached="NO — connection refused",
                          authorised="NOT REACHED")
    assert acceptance_verdict(checks) == ("NO", "NOT_EXECUTABLE")


def test_a_semantic_failure_is_not_reported_as_not_executable():
    """The converse, which matters just as much: a reachable, authorised,
    correctly-provenanced run that resolves the wrong basis must say so."""
    envelopes = _baseline()
    envelopes["G02"]["metadata"]["geographyBasis"] = _geo(
        "borrower", "asset_class_default")
    checks, _, _ = _score(envelopes)
    ready, reason = acceptance_verdict(checks)
    assert ready == "NO"
    assert reason != "NOT_EXECUTABLE"
    assert "GENERIC_REGION_PASS" in reason


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# =========================================================================== #
# The harness must run where the engine's dependencies do not
# =========================================================================== #
def test_the_field_basis_owner_loads_without_the_engines_dependencies():
    """The live certification workflow installs NOTHING.

    `certify-mi-api.yml` is a thin HTTP client by design — it sets up Python and
    runs the harness, with no `pip install`. The core and broad suites never
    imported the engine, so nothing had ever needed a dependency.

    Then the geography scorer imported `mi_agent.mi_geography` to learn which
    basis a column carries. That executes `mi_agent/__init__.py`, which imports
    the whole package, which imports `yaml` — and the acceptance run reached
    production, authenticated, confirmed the deployed commit, and died on
    `ModuleNotFoundError: No module named 'yaml'` before scoring one question.

    So the owner is loaded BY PATH. This test is the guard, and it has to run in
    a subprocess with `yaml` genuinely blocked: asserting anything in THIS
    process would prove nothing, because the test environment has yaml
    installed — which is exactly why the defect shipped.
    """
    import subprocess
    import textwrap

    program = textwrap.dedent(f'''
        import sys

        class BlockYaml:
            """Refuse `yaml` the way a runner without it would."""
            def find_spec(self, name, path=None, target=None):
                if name == "yaml" or name.startswith("yaml."):
                    raise ImportError("No module named 'yaml'")
                return None

        sys.meta_path.insert(0, BlockYaml())
        sys.path.insert(0, {str(_REPO_ROOT)!r})

        try:
            import yaml
            raise SystemExit("the block did not work; this proves nothing")
        except ImportError:
            pass

        from due_diligence.evidence.mi_api_certification.certify_mi_api import (
            basis_of_field_owner, _measured_basis, _measured_region_field)

        basis_of_field = basis_of_field_owner()
        assert basis_of_field is not None, "the owner did not load"
        assert basis_of_field("collateral_geography") == "collateral"
        assert basis_of_field("geographic_region_obligor") == "borrower"
        assert basis_of_field("not_a_region_at_all") is None

        # THE PATH THAT ACTUALLY RAN IN PRODUCTION. Asserting only on the
        # loader would leave this test green against the original defect,
        # because that defect was the direct package import inside
        # `_measured_region_field` — which is the function the scorer calls for
        # every question. Exercise the real call site.
        envelope = {{"spec": {{"dimension": "geographic_region_obligor",
                             "dimensions": ["geographic_region_obligor"],
                             "filters": {{}}}},
                    "artifacts": []}}
        assert _measured_region_field(envelope) == "geographic_region_obligor"
        assert _measured_basis(envelope) == "borrower"

        # By PATH, not as a package: importing the package is the bug.
        assert "mi_agent" not in sys.modules, "the mi_agent package was imported"
        print("OK")
    ''')
    result = subprocess.run([sys.executable, "-c", program],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    assert "OK" in result.stdout


def test_a_scorer_that_cannot_tell_what_was_measured_fails_loudly():
    """If the owner cannot be loaded at all, the gate must go red.

    Returning `measuredBasis: null` ten times and passing everything else would
    be the worst outcome available: a green run that never performed the one
    check it exists to perform.
    """
    import due_diligence.evidence.mi_api_certification.certify_mi_api as C

    saved = list(C._BASIS_OF_FIELD)
    C._BASIS_OF_FIELD.clear()
    C._BASIS_OF_FIELD.append(None)          # simulate a load failure
    try:
        checks, _, _ = _score(_baseline())
        assert checks["GENERIC_REGION_PASS"] is False
        assert acceptance_verdict(checks)[0] == "NO"
    finally:
        C._BASIS_OF_FIELD.clear()
        C._BASIS_OF_FIELD.extend(saved)
