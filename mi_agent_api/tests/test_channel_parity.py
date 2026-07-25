#!/usr/bin/env python3
"""mi_agent_api/tests/test_channel_parity.py

ANALYTICAL PARITY between the two presentation channels.

For the same authenticated tenant, client, portfolio context, question and
as-of date, ``POST /mi/query`` (React) and ``POST /v1/copilot/mi/query``
(Copilot) must invoke the same governed analytical service and produce the same
underlying interpreted intent, query specification, dataset, reporting date,
filters, metric, aggregation, dimensions, rows, numeric values, validation
result, reconciliation result, provenance, warnings and governed error state.

Differences are permitted ONLY in presentation: maximum row count, chart
rendering, interactive UI capability and signed artifact-link formatting.

The question set is a representative subset drawn from the EXISTING MI Agent
golden-question library (``config/mi/golden_questions/ere_mi_questions.yaml``),
covering every required category. Parity is asserted on structure and numbers —
never on prose alone.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import yaml
from fastapi.testclient import TestClient

from mi_agent_api import data_source
from mi_agent_api.app import app
from mi_agent_api.copilot_text import normalise_text

client = TestClient(app)

_QUESTION_BANK = (_REPO_ROOT / "config" / "mi" / "golden_questions"
                  / "ere_mi_questions.yaml")

#: Every category the parity requirement names, mapped to the golden-question
#: library's own category keys. ``None`` means the question is supplied inline
#: because the library has no dedicated category for it.
_REQUIRED_CATEGORIES = {
    "portfolio summary": "funded_kpi",
    "LTV distribution": "funded_breakdown_1d",
    "geography": "funded_breakdown_1d",
    "top-N": "ranking_concentration",
    "filtered": "funded_filtered_qa",
    "drill-through": "drill_through",
    "two-dimensional table": "funded_crosstab_2d",
    "filtered time series": "funded_evolution",
    "period comparison": "funded_evolution",
    "cohort": "vintage_cohort",
    "forecast": "forecast",
    "risk query": "risk_limits",
    "data-quality question": "data_quality",
    "unsupported concept": "nneg_er",
}

#: Questions with no library category of their own (WA LTV as a KPI, a
#: three-dimensional request that must fall back to a table, an explicit
#: geographic concentration, and an explicit period comparison).
_EXTRA_QUESTIONS = [
    ("wa_ltv", "What is the weighted average LTV?"),
    ("wa_interest_rate", "What is the weighted average interest rate?"),
    ("geography_concentration", "Where is the book most concentrated by region?"),
    ("ltv_distribution", "Show balance by LTV bucket."),
    ("three_dimension_fallback",
     "Show balance by age bucket, LTV bucket and collateral region."),
    ("period_comparison",
     "Compare funded balance this month against last month."),
    ("cohort_progression", "Show cohort progression by origination year."),
]


def _bank_questions(per_category: int = 2) -> List[tuple]:
    data = yaml.safe_load(_QUESTION_BANK.read_text(encoding="utf-8"))
    by_category: Dict[str, List[dict]] = {}
    for q in data.get("questions") or []:
        by_category.setdefault(q.get("category"), []).append(q)
    picked: List[tuple] = []
    seen = set()
    for label, category in _REQUIRED_CATEGORIES.items():
        for q in (by_category.get(category) or [])[:per_category]:
            key = q["question"]
            if key in seen:
                continue
            seen.add(key)
            picked.append((f"{label}:{q['id']}", key))
    for qid, question in _EXTRA_QUESTIONS:
        if question not in seen:
            seen.add(question)
            picked.append((qid, question))
    return picked


PARITY_QUESTIONS = _bank_questions()
#: Copilot caps supporting rows; parity on rows is asserted up to that cap.
_COPILOT_ROW_CAP = 50


def _demo_csv() -> Path:
    hits = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))
    assert hits, "expected the bundled synthetic demo canonical CSV"
    return hits[0]


@pytest.fixture(scope="module", autouse=True)
def governed_dataset():
    """One governed live dataset for BOTH channels, with the deterministic parser
    forced so the comparison is reproducible."""
    import os

    saved = {k: os.environ.get(k) for k in
             ("MI_AGENT_DATA_CSV", "MI_AGENT_DATA_CACHE_TTL", "MI_AGENT_LLM_PARSER",
              "ANTHROPIC_API_KEY", "TRAKT_COPILOT_AUTH_MODE", "MI_AGENT_CLIENT_ID")}
    os.environ["MI_AGENT_DATA_CSV"] = str(_demo_csv())
    os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ["TRAKT_COPILOT_AUTH_MODE"] = "disabled"
    os.environ["MI_AGENT_CLIENT_ID"] = "ERE"
    data_source.reset_cache()
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    data_source.reset_cache()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _react(question: str, **kw) -> Dict[str, Any]:
    body = {"question": question, "portfolioId": "ERE", **kw}
    r = client.post("/mi/query", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _copilot(question: str, **kw) -> Dict[str, Any]:
    body = {"question": question, **kw}
    r = client.post("/v1/copilot/mi/query", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _data_artifacts(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The analytical artifacts, in order — validation artifacts excluded (the
    Copilot channel surfaces validation as a typed field instead)."""
    return [a for a in (envelope.get("artifacts") or [])
            if a.get("type") in ("kpi", "table", "chart")]


def _numeric_cells(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Only the numeric cells of each row — the values that must be identical."""
    return [{k: v for k, v in row.items() if isinstance(v, (int, float))
             and not isinstance(v, bool)}
            for row in rows or []]


def _text_cells(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: normalise_text(v) for k, v in row.items() if isinstance(v, str)}
            for row in rows or []]


def _normalised(values: Optional[List[Any]]) -> List[str]:
    return [normalise_text(str(v)) for v in (values or [])]


# --------------------------------------------------------------------------- #
# Parity across the question bank
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,question", PARITY_QUESTIONS,
                         ids=[q[0] for q in PARITY_QUESTIONS])
def test_react_and_copilot_are_analytically_identical(label, question):
    react = _react(question)
    copilot = _copilot(question, portfolioId="ERE")

    # -- status / governed error state ------------------------------------- #
    assert copilot["ok"] == react["ok"], f"{label}: ok diverged"
    assert normalise_text(str(copilot.get("error") or "")) == \
        normalise_text(str(react.get("error") or "")), f"{label}: error diverged"

    # -- parsed query specification (metric, aggregation, dimensions,
    #    filters, top-N, ordering) ------------------------------------------ #
    assert (copilot.get("querySpec") or {}) == (react.get("spec") or {}), \
        f"{label}: parsed spec diverged"

    # -- dataset + reporting date ------------------------------------------ #
    react_meta = react.get("metadata") or {}
    assert copilot.get("datasetContext") == react_meta.get("datasetContext")
    assert copilot.get("reportingDate") == react_meta.get("asOfDate")
    assert copilot.get("selectedClient") == react_meta.get("selectedClient")
    assert copilot.get("selectedRun") == react_meta.get("selectedRun")

    # -- validation / reconciliation --------------------------------------- #
    assert (copilot.get("validation") or {}) == (react.get("validation") or {}), \
        f"{label}: validation diverged"
    assert (copilot.get("reconciliation") or {}) == \
        (react.get("reconciliation") or {}), f"{label}: reconciliation diverged"

    # -- provenance / warnings --------------------------------------------- #
    assert _normalised(copilot.get("warnings")) == _normalised(react.get("warnings")), \
        f"{label}: warnings diverged"
    assert len(copilot.get("sourceNotes") or []) == len(react.get("sourceNotes") or []), \
        f"{label}: provenance note count diverged"

    # -- artifacts: same kinds, same numbers, same rows (up to the cap) ----- #
    react_arts = _data_artifacts(react)
    copilot_arts = copilot.get("supportingValues") or []
    assert len(copilot_arts) == len(react_arts), f"{label}: artifact count diverged"
    for r_art, c_art in zip(react_arts, copilot_arts):
        kind = r_art["type"]
        assert c_art["kind"] == kind, f"{label}: artifact kind diverged"
        if kind == "kpi":
            r_kpis = [(normalise_text(str(k.get("label"))),
                       normalise_text(str(k.get("value"))))
                      for k in (r_art.get("kpis") or [])]
            c_kpis = [(normalise_text(str(k.get("label"))),
                       normalise_text(str(k.get("value"))))
                      for k in (c_art.get("kpis") or [])]
            assert c_kpis == r_kpis, f"{label}: KPI values diverged"
            continue
        r_rows = r_art.get("rows") or []
        c_rows = c_art.get("rows") or []
        assert c_art.get("totalRows") == len(r_rows), f"{label}: row total diverged"
        assert len(c_rows) == min(len(r_rows), _COPILOT_ROW_CAP)
        assert c_art.get("truncated") is (len(r_rows) > _COPILOT_ROW_CAP)
        capped = r_rows[:_COPILOT_ROW_CAP]
        assert _numeric_cells(c_rows) == _numeric_cells(capped), \
            f"{label}: numeric row values diverged"
        assert _text_cells(c_rows) == _text_cells(capped), \
            f"{label}: row labels diverged"

    # -- the answer text is the same answer, only text-normalised ---------- #
    assert copilot["answer"] == normalise_text(react["answer"]), \
        f"{label}: answer text diverged beyond normalisation"


# --------------------------------------------------------------------------- #
# Regression: Copilot must not report "no data" where React succeeds
# --------------------------------------------------------------------------- #
_POINT_IN_TIME_QUESTIONS = [
    "What is the total current balance?",
    "How many loans are in the book?",
    "What is the weighted average LTV?",
    "Where is the book most concentrated by region?",
    "Show balance by LTV bucket.",
    "Show balance by collateral region.",
    "Show the top 10 loans by current balance.",
    "How many borrowers are aged 75 or above?",
]

_DEGRADED_MARKERS = ("no funded frame", "can't resolve the funded book",
                     "missing run", "unavailable data", "no data is available")


@pytest.mark.parametrize("question", _POINT_IN_TIME_QUESTIONS)
def test_point_in_time_question_never_needs_a_run_id(question):
    """A point-in-time question with NO run id must succeed on BOTH channels and
    must never be downgraded to a run-scoped 'no funded frame' answer."""
    react = _react(question)
    copilot = _copilot(question)

    assert react["ok"] is True, f"React failed: {react.get('error')}"
    assert copilot["ok"] is True, f"Copilot failed where React succeeded: {copilot}"
    # Neither channel required a run for a point-in-time question.
    assert copilot["selectedRun"] is None
    assert (react.get("metadata") or {})["runRequired"] is False

    blob = " ".join([copilot["answer"], *(copilot.get("warnings") or [])]).lower()
    for marker in _DEGRADED_MARKERS:
        assert marker not in blob, f"Copilot degraded on '{question}': {blob}"


def test_geography_regression_matches_react_row_for_row():
    """The exact failure this refactor fixes: regional concentration with no run
    id. Both channels must return the same ranked exposure rows."""
    question = "Where is the book most concentrated by region?"
    react = _react(question)
    copilot = _copilot(question)

    react_rows = _data_artifacts(react)[0]["rows"]
    copilot_rows = (copilot["supportingValues"] or [{}])[0].get("rows")
    assert react_rows and copilot_rows
    assert _numeric_cells(copilot_rows) == _numeric_cells(react_rows[:_COPILOT_ROW_CAP])
    assert (react.get("metadata") or {})["route"] == "geo_exposure"


def test_unsupported_question_is_the_same_governed_limitation():
    question = "How many loans have negative equity guarantee?"
    react = _react(question)
    copilot = _copilot(question)
    assert react["ok"] is False and copilot["ok"] is False
    assert copilot["answer"] == normalise_text(react["answer"])
    # Neither channel invented a metric to fill the gap.
    assert not copilot["supportingValues"]
    assert not _data_artifacts(react)
