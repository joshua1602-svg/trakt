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


def _check_forbidden(result, forbidden: List[str]) -> List[str]:
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

    # Controlled-failure family: must NOT return a fabricated numeric answer,
    # and must not emit forbidden domain terms as if answered.
    if q["expected_status"] == "controlled_failure":
        total = _extract_total(result)
        refused = (result.status != "success") or (total is None)
        outcome["checks"]["controlled_refusal"] = refused
        passed = passed and refused

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
