#!/usr/bin/env python3
"""mi_agent/mi_calibration.py

Evaluation engine for the curated MI calibration bank
(``config/mi/golden_questions/ere_mi_calibration_250.yaml``).

Runs each curated question through the REAL deterministic MI path
(``run_mi_agent_query`` + ``adapt_workflow_result``) and checks the case's
declared EXPECTED semantic behaviour — metric resolution, dimensions resolved
AND applied, filters resolved AND applied, the dimension/filter invariants,
artifact type, reconciliation, and required columns. Refuse/clarify cases assert
the query is NOT answered with (narrower/unfiltered) data and that a reason is
surfaced. Pipeline/forecast cases are validated at parse level (their execution
needs the runtime chat-routing harness, not the funded fixture).

Shared by ``mi_agent/tests/test_mi_calibration_bank.py`` and
``scripts/mi_query_calibration.py`` (report). Deterministic and offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .mi_query_contract import canonical_of
from .mi_agent_workflow import run_mi_agent_query
from .llm_query_parser import _deterministic_parse

try:
    from mi_agent_api.adapters import adapt_workflow_result
except Exception:  # pragma: no cover
    adapt_workflow_result = None  # type: ignore

_BANK = (Path(__file__).resolve().parents[1] / "config" / "mi" /
         "golden_questions" / "ere_mi_calibration_250.yaml")


def load_bank(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    doc = yaml.safe_load((path or _BANK).read_text(encoding="utf-8"))
    return doc.get("questions", [])


@dataclass
class CalibrationResult:
    id: str
    category: str
    question: str
    ok: bool
    failures: List[str] = field(default_factory=list)
    known_gap: Optional[str] = None
    # observed, for the report / debugging
    observed_metric: Optional[str] = None
    observed_dims: List[str] = field(default_factory=list)
    observed_filters: List[str] = field(default_factory=list)
    observed_artifacts: List[str] = field(default_factory=list)
    status: str = "answer"

    @property
    def defect_class(self) -> Optional[str]:
        """The first failure's coarse class, for recurring-defect tallies."""
        if self.ok or not self.failures:
            return None
        f = self.failures[0].lower()
        if "metric" in f:
            return "metric-resolution"
        if "dimension" in f:
            return "dimension-invariant" if "invariant" in f else "dimension-applied"
        if "filter" in f:
            return "filter-invariant" if "invariant" in f else "filter-applied"
        if "artifact" in f:
            return "artifact-output"
        if "reconciliation" in f:
            return "reconciliation"
        if "answered with data" in f or "reason" in f:
            return "fail-closed"
        if "not ok" in f:
            return "execution"
        if "hallucinat" in f:
            return "parser-hallucination"
        if "warning" in f:
            return "warning"
        return "other"


def _artifact_tokens(artifacts: List[dict]) -> List[str]:
    out: List[str] = []
    for a in artifacts or []:
        if a.get("type") == "chart":
            out.append(f"chart:{a.get('chartType')}")
        else:
            out.append(a.get("type"))
    return out


def _artifact_compatible(expected: str, tokens: List[str]) -> bool:
    charts = {t.split(":", 1)[1] for t in tokens if t.startswith("chart:")}
    if expected == "none":
        return not ({"kpi", "table"} & set(tokens)) and not charts
    if expected == "kpi":
        return "kpi" in tokens
    if expected == "table":
        return "table" in tokens
    if expected in ("bar", "line", "heatmap", "treemap"):
        return expected in charts
    return True


def _run(q: str, df, semantics: dict, live_llm: bool):
    """Run one question, deterministically (default) or via the LLM parser. The
    deterministic dimension/filter invariant guards run either way, so a live LLM
    parse is held to the SAME fail-closed contract before execution."""
    if live_llm:
        return run_mi_agent_query(q, df, semantics, llm_enabled=True,
                                  parser_mode="llm", zero_cost_first=False)
    return run_mi_agent_query(q, df, semantics)


def evaluate_case(case: Dict[str, Any], df, semantics: dict,
                  live_llm: bool = False) -> CalibrationResult:
    q = case["question"]
    status = case.get("expected_status", "answer")
    execution = case.get("execution", "full")
    known_gap = case.get("known_gap")
    r = CalibrationResult(id=case["id"], category=case["category"], question=q,
                          ok=True, known_gap=known_gap, status=status)
    fails: List[str] = []

    if execution == "parse_only":
        spec, meta = _deterministic_parse(q, semantics, available_columns=set(df.columns))
        registry = set(semantics.get("fields", {}))
        halluc = [f for f in spec.referenced_fields() if f not in registry]
        if halluc:
            fails.append(f"parser hallucinated fields: {halluc}")
        em = case.get("expected_metric")
        if em and spec.metric != em:
            fails.append(f"metric {spec.metric} != expected {em}")
        r.observed_metric = spec.metric
        r.observed_dims = list(spec.dimensions or ([spec.dimension] if spec.dimension else []))
        r.failures = fails
        r.ok = not fails
        return r

    res = _run(q, df, semantics, live_llm)
    spec = res.get("spec") or {}
    di = res.get("dimension_invariant") or {}
    fi = res.get("filter_invariant") or {}
    qr = res.get("query_result")
    cols = [str(c) for c in qr.data.columns] if qr is not None else []
    ad = adapt_workflow_result(res, portfolio_id="client_001", as_of=None) \
        if adapt_workflow_result else {"artifacts": [], "reconciliation": None, "warnings": []}
    tokens = _artifact_tokens(ad.get("artifacts") or [])
    r.observed_metric = spec.get("metric")
    r.observed_dims = list(di.get("applied") or [])
    r.observed_filters = list(fi.get("applied_filters") or [])
    r.observed_artifacts = tokens

    if status in ("refuse", "clarify"):
        answered = bool(res.get("ok")) and qr is not None
        if answered:
            fails.append("answered with data (expected refuse/clarify)")
        reason = res.get("error") or (ad.get("warnings") or []) or res.get("warnings")
        if not reason:
            fails.append("no refusal/clarification reason surfaced")
        # No positive data artifact should be emitted.
        if {"kpi", "table"} & set(tokens) or any(t.startswith("chart:") for t in tokens):
            fails.append(f"emitted a data artifact on a refuse/clarify case: {tokens}")
        r.failures = fails
        r.ok = not fails
        return r

    # status == answer
    if not res.get("ok"):
        fails.append(f"not ok: {res.get('error')}")
        r.failures = fails
        r.ok = False
        return r

    em = case.get("expected_metric")
    if em is not None and spec.get("metric") != em:
        fails.append(f"metric {spec.get('metric')} != expected {em}")
    for d in case.get("expected_dimensions") or []:
        if d not in (di.get("applied") or []) and canonical_of(d, semantics) not in cols:
            fails.append(f"dimension {d} not applied")
    if case.get("expected_dimension_invariant_ok") and not di.get("ok", True):
        fails.append("dimension invariant not ok")
    for f in case.get("expected_filters") or []:
        if f not in (fi.get("applied_filters") or []):
            fails.append(f"filter {f} not applied")
    if case.get("expected_filter_invariant_ok") and not fi.get("ok", True):
        fails.append("filter invariant not ok")
    for c in case.get("expected_columns_include") or []:
        if c not in cols:
            fails.append(f"column {c} missing from result")
    if qr is not None and len(cols) < case.get("expected_min_columns", 1):
        fails.append(f"result has {len(cols)} cols < expected {case.get('expected_min_columns')}")
    if not _artifact_compatible(case.get("expected_artifact_type", "bar"), tokens):
        fails.append(f"artifact {tokens} incompatible with expected {case.get('expected_artifact_type')}")
    if case.get("expected_reconciliation") and not ad.get("reconciliation"):
        fails.append("reconciliation missing")
    for w in case.get("expected_warnings") or []:
        hay = [str(x).lower() for x in (ad.get("warnings") or [])]
        if res.get("error"):
            hay.append(str(res["error"]).lower())
        if not any(w.lower() in h for h in hay):
            fails.append(f"expected warning '{w}' not surfaced")

    r.failures = fails
    r.ok = not fails
    return r


def run_bank(df=None, semantics=None, path: Optional[Path] = None,
             live_llm: bool = False
             ) -> Tuple[List[CalibrationResult], Dict[str, Any]]:
    if semantics is None:
        from .mi_query_validator import load_mi_semantics
        semantics = load_mi_semantics(
            Path(__file__).resolve().parent / "mi_semantics_field_registry.yaml")
    if df is None:
        from .mi_query_harness import build_fixture
        df = build_fixture()
    cases = load_bank(path)
    results = [evaluate_case(c, df, semantics, live_llm=live_llm) for c in cases]
    return results, summarise_bank(results)


# Priority-1 fail-closed cases — the ones most worth running through the LLM
# parser to prove the deterministic invariants still hold after LLM parsing.
PRIORITY1_QUESTIONS = [
    "balance trend where LTV above 50%",
    "funded balance by month where LTV > 50%",
    "balance by region where LTV above 50%",
    "balance by region by borrower type by LTV bucket",
]


def llm_available() -> bool:
    import os
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def run_live_llm_priority(df=None, semantics=None) -> List[Dict[str, Any]]:
    """Run the Priority-1 cases through the LLM parser and assert the SAME
    fail-closed contract (no silent filter omission, no silent dimension
    truncation; both invariants enforced after LLM parsing). Requires an
    ANTHROPIC_API_KEY. Returns a per-question contract report."""
    if semantics is None:
        from .mi_query_validator import load_mi_semantics
        semantics = load_mi_semantics(
            Path(__file__).resolve().parent / "mi_semantics_field_registry.yaml")
    if df is None:
        from .mi_query_harness import build_fixture
        df = build_fixture()
    out: List[Dict[str, Any]] = []
    for q in PRIORITY1_QUESTIONS:
        res = _run(q, df, semantics, live_llm=True)
        di = res.get("dimension_invariant") or {}
        fi = res.get("filter_invariant") or {}
        parse_meta = (res.get("metadata") or {}).get("parse_metadata") or {}
        out.append({
            "question": q,
            "parser_mode": parse_meta.get("parser_mode") or res.get("parser_mode"),
            "ok": res.get("ok"),
            "dimension_invariant_ok": di.get("ok"),
            "filter_invariant_ok": fi.get("ok"),
            "applied_filters": fi.get("applied_filters"),
            "applied_dimensions": di.get("applied"),
            # The contract: invariants enforced, nothing silently dropped.
            "contract_ok": bool(di.get("ok")) and bool(fi.get("ok")),
        })
    return out


def summarise_bank(results: List[CalibrationResult]) -> Dict[str, Any]:
    by_cat: Dict[str, Dict[str, int]] = {}
    defects: Dict[str, int] = {}
    known_gaps: List[CalibrationResult] = []
    hard_failures: List[CalibrationResult] = []
    for r in results:
        c = by_cat.setdefault(r.category, {"total": 0, "passed": 0, "known_gap": 0})
        c["total"] += 1
        if r.known_gap:
            c["known_gap"] += 1
        if r.ok:
            c["passed"] += 1
        elif r.known_gap:
            known_gaps.append(r)
        else:
            hard_failures.append(r)
            dc = r.defect_class or "other"
            defects[dc] = defects.get(dc, 0) + 1
    total = len(results)
    passed = sum(1 for r in results if r.ok)
    flagged_known_gaps = [r for r in results if r.known_gap]
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "hard_failures": len(hard_failures),
        # A case flagged known_gap either currently fails (xfailed) or already
        # passes (the behaviour now meets the ideal). Report both so the count is
        # unambiguous and matches the pytest xfail total.
        "known_gaps": len(flagged_known_gaps),
        "known_gaps_xfailed": len([r for r in flagged_known_gaps if not r.ok]),
        "known_gaps_passing": len([r for r in flagged_known_gaps if r.ok]),
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "by_category": by_cat,
        "defects_by_class": defects,
        "hard_failure_cases": [(r.id, r.question, r.failures) for r in hard_failures],
        "known_gap_cases": [(r.id, r.question, r.known_gap) for r in
                            [x for x in results if x.known_gap]],
    }


if __name__ == "__main__":  # pragma: no cover
    results, summary = run_bank()
    import json
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("hard_failure_cases", "known_gap_cases")}, indent=2))
    for r in results:
        if not r.ok and not r.known_gap:
            print("HARD-FAIL", r.id, r.category, "|", r.failures, "|", r.question)
