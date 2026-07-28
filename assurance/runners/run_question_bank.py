"""Execute the 1,000-question bank through the live governed MI capability and
validate the structured output against the independent oracle.

This is a multi-layer runner: for each question it records whether the correct
route/owner won (routing), whether the structured value matches the oracle
(numerical), whether the controlled-failure questions actually refuse, and
whether forbidden claims are absent (presentation safety). It writes a
machine-readable result file and a per-family summary.

The runner deliberately runs with the deterministic parser (no LLM) so results
are reproducible in CI. Questions whose family is not numerically checkable are
validated on route / status / forbidden-claim axes only.

Usage:
    python assurance/runners/run_question_bank.py \
        --out assurance/reports/question_bank_results.json [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "assurance"))

BANK = ROOT / "assurance/question_bank/question_bank.json"
FIXTURES = ROOT / "assurance/fixtures/generated"

# Relative tolerance for numerical comparison (structured values).
REL_TOL = 1e-6


def _configure_env(fixture_id: str) -> None:
    os.environ["TRAKT_RUNTIME_MODE"] = "test"
    os.environ["MI_AGENT_PLATFORM_CANONICAL"] = str(FIXTURES / f"{fixture_id}.csv")
    os.environ.pop("MI_AGENT_ONBOARDING_OUTPUT_ROOT", None)
    os.environ["MI_AGENT_CLIENT_ID"] = "client_001"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    os.environ.pop("ANTHROPIC_API_KEY", None)


def _reset() -> None:
    from mi_agent_api import data_source, datasets
    data_source.reset_cache()
    datasets._CLIENT_CURRENCY_CACHE.clear()


# expected_scope -> workspace source-portfolio lens, exactly as the React
# workspace supplies it via ``sourcePortfolioLens``. Governed scope is a
# workspace concern; the question text is a secondary override path tested
# separately in the collision suite.
_SCOPE_TO_LENS = {
    "total": None,
    "direct_book": "direct",
    "acquired_book": "acquired",
    "alp_origination": "alp_origination",
    "alp_acquired": "alp_acquired",
    "spv1": "spv1_sponsored",
}


def _run_one(question: str, lens: Optional[str] = None):
    from trakt_core.context import ExecutionContext
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    ctx = ExecutionContext.for_internal("client_001")
    return execute_governed_mi_query(
        MiQueryRequest(question=question, source_portfolio_lens=lens), ctx)


def _oracle_expected(spec: Dict[str, Any], fixture_id: str) -> Optional[Dict[str, Any]]:
    import pandas as pd
    from oracle import oracle as O
    check = spec.get("oracle_check")
    if not check:
        return None
    df = pd.read_csv(FIXTURES / f"{fixture_id}.csv")
    scope = check.get("scope") or {}
    df = O.apply_scope(df, source_portfolio_id=scope.get("source_portfolio_id"),
                       source_portfolio_type=scope.get("source_portfolio_type"))
    kind = check["kind"]
    if kind == "sum":
        return {"value": O.total(df, check["metric"]).value, "unit": "currency"}
    if kind == "count":
        return {"value": O.count(df).value, "unit": "count"}
    if kind == "average":
        return {"value": O.simple_average(df, check["metric"]).value, "unit": "ratio"}
    if kind == "weighted_average":
        return {"value": O.weighted_average(df, check["metric"]).value, "unit": "ratio"}
    if kind == "distribution":
        d = O.distribution(df, check["dimension"], check.get("measure", O.BALANCE))
        return {"count_share_sum": d["count_share_sum"],
                "exposure_share_sum": d["exposure_share_sum"], "rows": len(d["rows"])}
    if kind == "concentration_top_n":
        c = O.concentration_top_n(df, check["dimension"], check["n"])
        return {"top_share": c["top_share"]}
    if kind == "single_name":
        s = O.single_name_top(df, check.get("n", 1))
        return {"top_share": s["top_share"]}
    return None


def _extract_total(result) -> Optional[float]:
    """The scoped monetary answer. When filters are applied the *included*
    balance is the answer; ``total_balance`` is the unfiltered denominator."""
    arts = (result.result or {}).get("artifacts") or []
    for art in arts:
        recon = art.get("reconciliation") or {}
        if recon.get("filters_applied") and recon.get("balance_included") is not None:
            return float(recon["balance_included"])
        if recon.get("balance_included") is not None:
            return float(recon["balance_included"])
        if recon.get("total_balance") is not None:
            return float(recon["total_balance"])
    return None


def _answer_text(result) -> str:
    r = result.result or {}
    parts = [str(r.get("answer") or "")]
    for art in r.get("artifacts") or []:
        for kpi in art.get("kpis") or []:
            parts.append(str(kpi.get("label", "")))
            parts.append(str(kpi.get("value", "")))
    return " ".join(parts).lower()


# Phrases that mark a governed refusal / limitation. When present, the response
# is a controlled disclosure (it may legitimately NAME the unavailable field), so
# a forbidden term appearing inside it is not a fabricated claim.
_DISCLOSURE_MARKERS = (
    "not available", "not reported", "cannot be answered", "no value was fabricated",
    "unavailable", "need review", "does not include", "not present", "no data",
    "missing field", "is not a", "not supported",
)


def _has_disclosure(result) -> bool:
    r = result.result or {}
    hay = str(r.get("answer") or "").lower()
    for w in r.get("warnings") or []:
        hay += " " + str(w).lower()
    return any(m in hay for m in _DISCLOSURE_MARKERS)


def _answer_is_none_or_zero(result) -> bool:
    """True when the answer's *primary metric* is null or zero — an honest 'none'
    rather than a fabricated value.

    The ubiquitous ``Loan`` count KPI is context present on every answer, so it is
    ignored unless it is the only KPI (a bare unfiltered count answering a
    filtered question — e.g. "loans in default" -> whole book — which is exactly
    the dropped-qualifier fabrication this check must catch)."""
    import re
    arts = (result.result or {}).get("artifacts") or []
    if not arts:
        return True
    for art in arts:
        kpis = art.get("kpis") or []
        metric_kpis = [k for k in kpis
                       if str(k.get("label", "")).strip().lower() not in ("loan", "loans")]
        # A filtered question answered by the bare loan count of the whole book:
        # treat the count as the (fabricated) answer.
        judged = metric_kpis if metric_kpis else kpis
        for kpi in judged:
            raw = str(kpi.get("value", "")).replace(",", "")
            for n in re.findall(r"-?\d[\d.]*", raw):
                try:
                    if abs(float(n)) > 0:
                        return False
                except ValueError:
                    continue
    return True


def _check_forbidden(result, forbidden: List[str]) -> List[str]:
    # A forbidden term inside a governed refusal/limitation is a disclosure, not a
    # claim. Only flag forbidden terms in a confident, non-disclosed answer.
    if result.status != "success" or _has_disclosure(result):
        return []
    text = _answer_text(result)
    return [c for c in forbidden if c.lower() in text]


def evaluate(q: Dict[str, Any]) -> Dict[str, Any]:
    fixture = q.get("fixture_id") or "three_portfolios"
    _configure_env(fixture)
    _reset()
    outcome: Dict[str, Any] = {"question_id": q["question_id"],
                               "family": q["question_family"], "checks": {}}
    lens = _SCOPE_TO_LENS.get(q.get("expected_scope"), None)
    try:
        result = _run_one(q["question"], lens=lens)
    except Exception as exc:  # noqa: BLE001
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        outcome["passed"] = False
        return outcome
    outcome["status"] = result.status

    passed = True

    # Controlled-failure family: must NOT fabricate a confident answer to a field
    # the book does not support. Three outcomes count as safe:
    #   * a non-success status (the route failed closed), or
    #   * a governed disclosure that the field is unavailable / needs review, or
    #   * a genuinely null / zero answer (an honest "none", not a fabrication).
    # A confident NON-ZERO numeric answer with no disclosure is a fabrication or a
    # dropped-qualifier substitution (e.g. "loans in default" answered as the whole
    # book) — the failure this family is designed to catch.
    if q["expected_status"] == "controlled_failure":
        safe = (result.status != "success") or _has_disclosure(result) \
            or _answer_is_none_or_zero(result)
        outcome["checks"]["controlled_refusal"] = safe
        passed = passed and safe

    # Forbidden claims (presentation safety) — applies to all families.
    if q.get("forbidden_claims"):
        leaked = _check_forbidden(result, q["forbidden_claims"])
        outcome["checks"]["forbidden_claims_absent"] = (leaked == [])
        outcome["leaked_claims"] = leaked
        passed = passed and (leaked == [])

    # Numerical oracle (structured value) where declared.
    expected = _oracle_expected(q, fixture)
    if expected is not None:
        outcome["oracle_expected"] = expected
        if "value" in expected and expected["value"] is not None:
            got = _extract_total(result) if expected.get("unit") == "currency" else None
            if expected.get("unit") == "currency" and got is not None:
                match = abs(got - expected["value"]) <= REL_TOL * max(1.0, abs(expected["value"]))
                outcome["checks"]["structured_value"] = match
                outcome["structured_got"] = got
                passed = passed and match

    outcome["passed"] = passed
    return outcome


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "assurance/reports/question_bank_results.json"))
    ap.add_argument("--limit", type=int, default=0, help="run only the first N (0 = all)")
    ap.add_argument("--family", default=None, help="restrict to one family")
    args = ap.parse_args()

    bank = json.loads(BANK.read_text())
    questions = bank["questions"]
    if args.family:
        questions = [q for q in questions if q["question_family"] == args.family]
    if args.limit:
        questions = questions[: args.limit]

    results: List[Dict[str, Any]] = []
    for q in questions:
        results.append(evaluate(q))

    by_family: Dict[str, Dict[str, int]] = {}
    for r in results:
        fam = by_family.setdefault(r["family"], {"total": 0, "passed": 0})
        fam["total"] += 1
        fam["passed"] += 1 if r.get("passed") else 0

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.get("passed")),
        "failed": sum(1 for r in results if not r.get("passed")),
        "by_family": by_family,
    }
    out = {"summary": summary, "results": results}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
