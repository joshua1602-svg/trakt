#!/usr/bin/env python3
"""build_demo_pack.py — generate the public landing-page demo pack.

Runs each allow-listed public-demo question through the REAL Trakt deterministic
MI engine, in-process, against the repository's bundled synthetic portfolio, and
writes the governed answers to ``landing-page/data/demo-pack.json``.

Nothing here re-implements portfolio maths. The chain is exactly the one the
React MI Agent and Microsoft 365 Copilot use:

    synthetic canonical CSV
      → mi_agent_api.funded_prep.prepare_funded_mi_dataset   (derived dimensions)
      → mi_agent.mi_agent_workflow.run_mi_agent_query        (parse→validate→execute→chart)
      → mi_agent_api.adapters.adapt_workflow_result          (governed envelope)
      → this script                                          (redact + shrink for public use)

Everything this script adds on top is *removal*: it drops the internal query
spec, diagnostics, engine identifiers, file paths and loan-level columns, and
keeps only aggregated display values.

Usage
-----
    python landing-page/scripts/build_demo_pack.py            # write the pack
    python landing-page/scripts/build_demo_pack.py --check    # fail if stale

The ``--check`` mode is what the test suite uses to prove the committed pack is
reproducible from the committed synthetic data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve()
_LANDING_ROOT = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from mi_agent.mi_agent_workflow import run_mi_agent_query  # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics  # noqa: E402
from mi_agent_api.adapters import adapt_workflow_result  # noqa: E402
from mi_agent_api.funded_prep import prepare_funded_mi_dataset  # noqa: E402

# --------------------------------------------------------------------------- #
# Repository inputs (all pre-existing; none created by the landing page)
# --------------------------------------------------------------------------- #
CANONICAL = _REPO_ROOT / (
    "synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv")
SEMANTICS = _REPO_ROOT / "mi_agent/mi_semantics_field_registry.yaml"
CLIENT_CONFIG = _REPO_ROOT / "synthetic_demo/config/config_client_SYNTHETIC_ERM.yaml"
MAPPING_REPORT = _REPO_ROOT / (
    "synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_header_mapping_report.json")
TRANSFORM_REPORT = _REPO_ROOT / (
    "synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_transform_report.json")
VALIDATION_SUMMARY = _REPO_ROOT / (
    "synthetic_demo/output/validation/SYNTHETIC_ERE_Portfolio_012026_field_summary.csv")
DELIVERY_REPORT = _REPO_ROOT / (
    "synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_delivery_report.json")

OUT = _LANDING_ROOT / "data" / "demo-pack.json"

#: Bumped whenever the pack's shape changes. The runtime asserts on it.
PACK_VERSION = 1

#: Hard ceiling on rows published for any single answer. The synthetic portfolio
#: is small, but the cap is structural, not incidental.
MAX_ROWS = 12

#: Columns that must never reach the public pack, whatever the engine produced.
#: (The engine only ever emits aggregates for these questions; this is defence
#: in depth, and it is asserted by a test.)
FORBIDDEN_COLUMNS = {
    "loan_identifier", "unique_identifier", "borrower_identifier", "postcode",
    "underlying_exposure_identifier", "original_underlying_exposure_identifier",
    "new_underlying_exposure_identifier", "original_obligor_identifier",
    "new_obligor_identifier", "original_collateral_identifier",
    "new_collateral_identifier", "youngest_borrower_age",
    "originator_legal_entity_identifier",
}


# --------------------------------------------------------------------------- #
# The public allow-list
# --------------------------------------------------------------------------- #
# Every entry is one supported public intent. `question` is the phrasing sent to
# the real engine; `phrases` are the visitor phrasings the runtime matcher will
# accept for it. Nothing outside this list is ever executed — the runtime never
# passes visitor text to the engine, because the engine is not deployed with the
# landing page at all.
INTENTS: List[Dict[str, Any]] = [
    {
        "id": "funded_balance",
        "artifactTitle": "Portfolio summary",
        "label": "What is the current funded portfolio balance?",
        "category": "portfolio_kpi",
        "question": "What is the current funded portfolio balance?",
        "primaryKpi": "outstanding balance",
        "phrases": [
            "current funded portfolio balance", "funded balance", "portfolio balance",
            "total balance", "current balance", "outstanding balance",
            "how big is the portfolio", "portfolio size", "aum", "book size",
        ],
        "narrative": (
            "The funded book stands at {kpi0} across {loans} exposures as at "
            "{as_of}. The figure is the sum of current outstanding balance on the "
            "governed canonical dataset — the same calculation the MI workspace "
            "and Microsoft 365 Copilot return for this portfolio."
        ),
        "followUps": ["region_exposure", "ltv_band", "wa_ltv"],
    },
    {
        "id": "loan_count",
        "artifactTitle": "Portfolio summary",
        "label": "How many loans are in the funded book?",
        "category": "portfolio_kpi",
        "question": "How many loans are in the funded book?",
        "primaryKpi": "exposures",
        "phrases": [
            "how many loans", "number of loans", "loan count", "how many exposures",
            "number of exposures", "how many accounts", "how many cases",
        ],
        "narrative": (
            "There are {kpi0} exposures in the funded book as at {as_of}, "
            "carrying {balance} of current outstanding balance."
        ),
        "followUps": ["funded_balance", "ticket_band", "channel"],
    },
    {
        "id": "wa_ltv",
        "artifactTitle": "Portfolio summary",
        "label": "What is the weighted average current LTV?",
        "category": "portfolio_kpi",
        "question": "What is the weighted average current LTV?",
        "primaryKpi": "loan to value",
        "phrases": [
            "weighted average current ltv", "weighted average ltv", "wa ltv",
            "average ltv", "what is the ltv", "loan to value",
        ],
        "narrative": (
            "Weighted average current LTV is {kpi0} as at {as_of}, balance-weighted "
            "across the funded book. Current LTV is derived from current "
            "outstanding balance over current valuation amount by the governed "
            "canonical transform, not restated here."
        ),
        "followUps": ["ltv_band", "funded_balance", "portfolio_risks"],
    },
    {
        "id": "wa_rate",
        "artifactTitle": "Portfolio summary",
        "label": "What is the weighted average interest rate?",
        "category": "portfolio_kpi",
        "question": "What is the weighted average interest rate?",
        "primaryKpi": "interest rate",
        "phrases": [
            "weighted average interest rate", "average interest rate", "wa rate",
            "what rate", "interest rate", "coupon", "yield",
        ],
        "narrative": (
            "Weighted average interest rate is {kpi0} as at {as_of}, "
            "balance-weighted across the funded book."
        ),
        "followUps": ["ltv_band", "funded_balance", "portfolio_risks"],
    },
    {
        "id": "region_exposure",
        "artifactTitle": "Current balance by region",
        "label": "Which regions have the highest exposure?",
        "category": "concentration",
        "question": "Which regions have the highest exposure?",
        "phrases": [
            "which regions have the highest exposure", "regional exposure",
            "exposure by region", "balance by region", "geographic concentration",
            "geographic exposure", "where is the portfolio", "by region",
            "regional concentration", "top regions", "geography",
        ],
        "narrative": (
            "{top_label} is the largest regional exposure at {top_value} "
            "({top_pct} of the funded book), ahead of {second_label} at "
            "{second_value}. Regions are derived from property postcode via the "
            "ITL/NUTS lookup in the governed transform, so the split is "
            "reproducible from source."
        ),
        "followUps": ["ltv_band", "portfolio_risks", "investor_report"],
    },
    {
        "id": "ltv_band",
        "artifactTitle": "Current balance by LTV band",
        "label": "Show the portfolio by LTV band.",
        "category": "stratification",
        "question": "Show current balance by LTV bucket",
        "phrases": [
            "show the portfolio by ltv band", "portfolio by ltv", "ltv band",
            "ltv bucket", "ltv distribution", "balance by ltv", "ltv breakdown",
            "stratify by ltv", "ltv stratification",
        ],
        "narrative": (
            "The book is concentrated in the {top_label} band, which holds "
            "{top_value} ({top_pct}). LTV bands are the governed bucket "
            "definitions from the shared analytics library, so the same bands "
            "appear in the investor pack and the regulatory stratifications."
        ),
        "followUps": ["wa_ltv", "region_exposure", "investor_report"],
    },
    {
        "id": "age_band",
        "artifactTitle": "Current balance by borrower age band",
        "label": "Show the portfolio by borrower age band.",
        "category": "stratification",
        "question": "Show balance by borrower age band",
        "phrases": [
            "portfolio by borrower age band", "by age band", "age band",
            "borrower age", "age distribution", "balance by age", "age bucket",
        ],
        "narrative": (
            "The largest age cohort is {top_label}, holding {top_value} "
            "({top_pct}) of the funded book. Borrower age drives redemption "
            "expectations on an equity release book, so it is a standing "
            "stratification in both investor and management reporting."
        ),
        "followUps": ["ltv_band", "funded_balance", "management_summary"],
    },
    {
        "id": "ticket_band",
        "artifactTitle": "Current balance by ticket size band",
        "label": "Show the portfolio by ticket size.",
        "category": "stratification",
        "question": "Show balance by ticket size band",
        "phrases": [
            "portfolio by ticket size", "ticket size", "ticket band", "loan size",
            "by loan size", "balance by ticket", "size distribution",
        ],
        "narrative": (
            "The {top_label} band is the largest by balance at {top_value} "
            "({top_pct})."
        ),
        "followUps": ["loan_count", "ltv_band", "investor_report"],
    },
    {
        "id": "channel",
        "artifactTitle": "Current balance by origination channel",
        "label": "Show the portfolio by origination channel.",
        "category": "stratification",
        "question": "Show balance by origination channel",
        "phrases": [
            "by origination channel", "origination channel", "by channel",
            "channel mix", "broker or direct", "distribution channel",
            "where did the loans come from", "introducer",
        ],
        "narrative": (
            "{top_label} is the largest origination channel at {top_value} "
            "({top_pct}) of the funded book."
        ),
        "followUps": ["ticket_band", "funded_balance", "management_summary"],
    },
]

#: Composite intents assembled from several engine runs plus governed pipeline
#: artefacts. Handled by dedicated builders below.
COMPOSITE_INTENT_IDS = ["portfolio_risks", "data_quality"]

#: Report actions. These return an in-page preview only — never a document, a
#: download URL, or a storage path.
REPORT_INTENT_IDS = ["management_summary", "investor_report"]

#: Questions the public demo deliberately refuses, each with an honest reason.
#: These are *not* failures — they are the governed "we will not invent this"
#: behaviour, and the page shows them off on purpose.
CONTROLLED_UNSUPPORTED: List[Dict[str, str]] = [
    {
        "id": "temporal_movement",
        "label": "How has the portfolio changed since last month?",
        "phrases": [
            "changed since last month", "since last month", "month on month",
            "month-on-month", "movement", "versus last month", "vs last month",
            "compared to last month", "trend", "over time", "evolution",
            "last quarter", "since december", "growth",
        ],
        "reason": (
            "Temporal comparison needs two governed reporting periods. The public "
            "demonstration publishes a single snapshot (30 November 2025), so "
            "there is no prior period to compare against and Trakt will not "
            "estimate one."
        ),
        "productionNote": (
            "In a client environment, Trakt holds a governed snapshot history and "
            "answers month-on-month, quarter-on-quarter and since-inception "
            "movement, with a reconciliation of every driver of the change."
        ),
    },
    {
        "id": "pipeline",
        "label": "Summarise the current pipeline.",
        "phrases": [
            "summarise the current pipeline", "summarize the pipeline", "pipeline",
            "applications", "new business", "funnel", "conversion", "offers",
            "forecast", "expected completions", "what will we fund",
        ],
        "reason": (
            "Pipeline analytics run on an application-stage dataset. The public "
            "demonstration publishes only the funded loan tape, so there is no "
            "pipeline to summarise."
        ),
        "productionNote": (
            "In a client environment, Trakt runs pipeline snapshots, the "
            "origination funnel, conversion analysis and expected-funding "
            "forecasts alongside the funded book."
        ),
    },
    {
        "id": "arrears",
        "label": "What are the arrears and default figures?",
        "phrases": [
            "arrears", "default", "delinquency", "npl", "non-performing",
            "impairment", "write-off", "loss", "recoveries", "in arrears",
        ],
        "reason": (
            "This synthetic equity release book carries no arrears or default "
            "balances, so an arrears answer would be meaningless rather than "
            "merely empty."
        ),
        "productionNote": (
            "Arrears, default, forbearance and loss analytics are standard Trakt "
            "portfolio analytics where the portfolio's own data supports them."
        ),
    },
    {
        "id": "loan_level",
        "label": "Show me individual loan records.",
        "phrases": [
            "individual loan", "loan level", "loan-level", "show me the loans",
            "list the loans", "borrower details", "customer details", "postcode",
            "loan identifier", "specific loan", "each loan", "raw data",
            "download the tape", "export the data", "csv",
        ],
        "reason": (
            "The public demonstration returns aggregated portfolio measures only. "
            "Exposure-level records are never exposed on this page, even for "
            "synthetic data."
        ),
        "productionNote": (
            "Exposure-level drill-through exists in the Trakt workspace, governed "
            "by role-based access within the client environment."
        ),
    },
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clean(value: Any) -> Any:
    """JSON-safe scalar, with NaN/Inf normalised to None and floats rounded."""
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int,)):
        return int(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    if hasattr(value, "item"):
        return _clean(value.item())
    return str(value)


def _fmt_currency(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"£{value:,.0f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def _artifacts(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [a for a in (envelope.get("artifacts") or []) if isinstance(a, dict)]


def _redact_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows[:MAX_ROWS]:
        out.append({k: _clean(v) for k, v in row.items()
                    if k not in FORBIDDEN_COLUMNS})
    return out


def _column_label(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _columns_from(art: Dict[str, Any],
                  rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise the adapter's column spec (list of dicts) or infer from rows."""
    raw = art.get("columns") or []
    hints = art.get("displayHints") or {}
    columns: List[Dict[str, Any]] = []
    if raw and isinstance(raw[0], dict):
        for col in raw:
            key = col.get("key")
            if not key or key in FORBIDDEN_COLUMNS:
                continue
            columns.append({
                "key": key,
                "label": col.get("label") or _column_label(key),
                "align": col.get("align") or "left",
                "format": col.get("format") or "text",
                "scale": col.get("scale"),
            })
    if not columns and rows:
        for key in rows[0]:
            if key in FORBIDDEN_COLUMNS:
                continue
            hint = hints.get(key) or {}
            numeric = isinstance(rows[0][key], (int, float))
            columns.append({
                "key": key,
                "label": _column_label(key),
                "align": "right" if numeric else "left",
                "format": hint.get("format") or ("number" if numeric else "text"),
                "scale": hint.get("scale"),
            })
    return columns


def _coverage(art: Dict[str, Any]) -> Optional[float]:
    rec = art.get("reconciliation")
    if not isinstance(rec, dict):
        return None
    return _clean(rec.get("coverage_by_balance_pct"))


def _public_artifact(art: Dict[str, Any],
                     kpi_labels: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Strip an engine artifact down to what the public page may render.

    Dropped: the internal MIQuerySpec, engine identifiers, diagnostics, artifact
    ids, creation timestamps, native chart types, warnings and every source path.
    Kept: aggregated rows, the column/format contract, and the balance-coverage
    figure from the engine's own reconciliation block.
    """
    kind = art.get("type")
    if kind == "kpi":
        kpis = []
        for item in art.get("kpis") or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            kpis.append({
                "label": kpi_labels.get(label, label),
                "value": str(item.get("value") or "").strip(),
            })
        if not kpis:
            return None
        return {"kind": "kpi", "title": art.get("title"), "kpis": kpis,
                "coverage": _coverage(art)}

    if kind in ("chart", "table"):
        rows = _redact_rows([r for r in (art.get("rows") or [])
                             if isinstance(r, dict)])
        if not rows:
            return None
        columns = _columns_from(art, rows)
        if not columns:
            return None
        public: Dict[str, Any] = {
            "kind": "chart" if kind == "chart" else "table",
            "title": art.get("title"),
            "columns": columns,
            "rows": rows,
            "totalRows": len(art.get("rows") or []),
            "coverage": _coverage(art),
        }
        if kind == "chart":
            public.update({
                "chartType": art.get("chartType") or "bar",
                "xKey": art.get("xKey") or columns[0]["key"],
                "valueKey": art.get("valueKey"),
                "valueFormat": art.get("valueFormat"),
            })
        return public
    return None


def _publish(envelope: Dict[str, Any],
             kpi_labels: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Public artifacts for one envelope, with the adapter's duplicate table
    folded into the chart it duplicates (the adapter emits both for every
    grouped result; the page renders one chart with a values table beneath it)."""
    labels = kpi_labels or {}
    publics = [p for p in (_public_artifact(a, labels) for a in _artifacts(envelope))
               if p]
    charts = [p for p in publics if p["kind"] == "chart"]
    if charts:
        keep = []
        for p in publics:
            if p["kind"] == "table" and any(c["rows"] == p["rows"] for c in charts):
                for chart in charts:
                    if chart["rows"] == p["rows"] and len(p["columns"]) > len(chart["columns"]):
                        chart["columns"] = p["columns"]
                continue
            keep.append(p)
        publics = keep
    return publics


def _kpi_values(publics: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    for art in publics:
        if art["kind"] == "kpi":
            return art["kpis"]
    return []


def _top_rows(publics: List[Dict[str, Any]]) -> tuple[Optional[str], Optional[str],
                                                      List[Dict[str, Any]]]:
    """(x key, value key, rows) of the first chart/table artifact."""
    for art in publics:
        if art["kind"] in ("chart", "table") and art["rows"]:
            keys = [c["key"] for c in art["columns"]]
            xkey = art.get("xKey") or keys[0]
            vkey = art.get("valueKey")
            if not vkey:
                vkey = next((k for k in keys
                             if k != xkey and isinstance(art["rows"][0].get(k),
                                                         (int, float))),
                            None)
            return xkey, vkey, art["rows"]
    return None, None, []


# --------------------------------------------------------------------------- #
# Engine invocation
# --------------------------------------------------------------------------- #
class Engine:
    """Thin holder for the prepared dataset + semantics registry."""

    def __init__(self) -> None:
        raw = pd.read_csv(CANONICAL, low_memory=False)
        self.frame, self.prep_report = prepare_funded_mi_dataset(raw)
        self.semantics = load_mi_semantics(SEMANTICS)
        self.total_balance = float(
            pd.to_numeric(self.frame["current_outstanding_balance"],
                          errors="coerce").fillna(0).sum())
        self.loan_count = int(len(self.frame))
        cutoff = self.frame.get("data_cut_off_date")
        values = sorted({str(v) for v in cutoff.dropna().unique()}) if cutoff is not None else []
        self.as_of = values[-1] if values else "2025-11-30"

    def run(self, question: str) -> Dict[str, Any]:
        workflow = run_mi_agent_query(question, self.frame, self.semantics)
        if not workflow.get("ok"):
            raise RuntimeError(
                f"engine refused {question!r}: {workflow.get('error')}")
        return adapt_workflow_result(workflow, portfolio_id="synthetic_demo",
                                     as_of=self.as_of)


def _as_of_display(iso: str) -> str:
    try:
        return pd.Timestamp(iso).strftime("%-d %B %Y")
    except Exception:  # pragma: no cover - platform strftime differences
        return iso


#: Presentation-only relabelling of engine KPI labels. Values are never touched.
KPI_LABELS = {
    "Loan": "Exposures",
    "Current Outstanding Balance": "Current outstanding balance",
    "Current Loan To Value Weighted": "Weighted average current LTV",
    "Current Interest Rate Weighted": "Weighted average interest rate",
}


def build_intent(engine: Engine, spec: Dict[str, Any],
                 as_of_display: str) -> Dict[str, Any]:
    envelope = engine.run(spec["question"])
    publics = _publish(envelope, KPI_LABELS)
    if not publics:
        raise RuntimeError(f"intent {spec['id']} produced no publishable artifact")

    # The engine titles an artifact with the question it answered, which reads
    # oddly in a report page. Presentation only — no value is touched.
    for art in publics:
        art["title"] = spec["artifactTitle"]

    kpis = _kpi_values(publics)
    xkey, vkey, rows = _top_rows(publics)

    subs: Dict[str, str] = {
        "as_of": as_of_display,
        "balance": _fmt_currency(engine.total_balance),
        "loans": f"{engine.loan_count:,}",
        "kpi0": _primary_kpi(kpis, spec.get("primaryKpi")),
    }
    if rows and xkey and vkey:
        first, second = rows[0], (rows[1] if len(rows) > 1 else rows[0])
        total = sum(float(r.get(vkey) or 0) for r in rows) or 1.0
        subs.update({
            "top_label": str(first.get(xkey)),
            "top_value": _fmt_currency(first.get(vkey)),
            "top_pct": _fmt_pct(100.0 * float(first.get(vkey) or 0) / total),
            "second_label": str(second.get(xkey)),
            "second_value": _fmt_currency(second.get(vkey)),
        })

    answer = spec["narrative"].format_map(_Defaults(subs))

    return {
        "id": spec["id"],
        "label": spec["label"],
        "category": spec["category"],
        "phrases": spec["phrases"],
        "answer": answer,
        "interpreted": envelope.get("interpreted"),
        "artifacts": publics,
        "followUps": spec.get("followUps", []),
    }


def _primary_kpi(kpis: List[Dict[str, str]], selector: Optional[str]) -> str:
    """The KPI this intent is actually about.

    The engine's summary artifact carries every measure it computed (an
    exposure count alongside the requested measure), so the narrative names the
    one the question asked for rather than whichever came first.
    """
    if not kpis:
        return "n/a"
    if selector:
        needle = selector.lower()
        for kpi in kpis:
            if needle in kpi["label"].lower():
                return kpi["value"]
    return kpis[-1]["value"]


class _Defaults(dict):
    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        return "n/a"


# --------------------------------------------------------------------------- #
# Composite intents
# --------------------------------------------------------------------------- #
def _concentration_table(engine: Engine, as_of_display: str) -> Dict[str, Any]:
    """Deterministic top-line concentration measures across three dimensions."""
    dims = [
        ("Region", "Which regions have the highest exposure?"),
        ("LTV band", "Show current balance by LTV bucket"),
        ("Borrower age band", "Show balance by borrower age band"),
    ]
    rows: List[Dict[str, Any]] = []
    detail: List[str] = []
    for label, question in dims:
        env = engine.run(question)
        publics = _publish(env, KPI_LABELS)
        xkey, vkey, drows = _top_rows(publics)
        if not (xkey and vkey and drows):
            continue
        total = sum(float(r.get(vkey) or 0) for r in drows) or 1.0
        top = drows[0]
        top_pct = 100.0 * float(top.get(vkey) or 0) / total
        top3 = sum(float(r.get(vkey) or 0) for r in drows[:3]) / total * 100.0
        rows.append({
            "dimension": label,
            "largest_segment": str(top.get(xkey)),
            "largest_share_pct": round(top_pct, 1),
            "top_3_share_pct": round(top3, 1),
            "segments": len(drows),
        })
        detail.append(f"{label.lower()} ({str(top.get(xkey))} at {_fmt_pct(top_pct)})")

    answer = (
        "The material concentrations as at {as_of} are {detail}. Nothing in this "
        "synthetic book breaches a limit, because no limit set is configured for "
        "the public demonstration — in a client environment these same measures "
        "are evaluated against the portfolio's own covenant and concentration "
        "limits, and breaches raise governed alerts."
    ).format(as_of=as_of_display, detail="; ".join(detail))

    return {
        "id": "portfolio_risks",
        "label": "What are the principal portfolio risks?",
        "category": "risk",
        "phrases": [
            "principal portfolio risks", "portfolio risks", "what are the risks",
            "risk", "concentration risk", "biggest risks", "risk profile",
            "limit breach", "covenant", "exposure risk",
        ],
        "answer": answer,
        "interpreted": "Concentration analysis · Region, LTV band, borrower age band",
        "artifacts": [{
            "kind": "table",
            "title": "Concentration by governed dimension",
            "columns": [
                {"key": "dimension", "label": "Dimension", "align": "left",
                 "format": "text", "scale": None},
                {"key": "largest_segment", "label": "Largest segment",
                 "align": "left", "format": "text", "scale": None},
                {"key": "largest_share_pct", "label": "Largest share",
                 "align": "right", "format": "pct", "scale": "percent_points"},
                {"key": "top_3_share_pct", "label": "Top 3 share",
                 "align": "right", "format": "pct", "scale": "percent_points"},
                {"key": "segments", "label": "Segments", "align": "right",
                 "format": "number", "scale": None},
            ],
            "rows": rows,
            "totalRows": len(rows),
            "coverage": 100.0,
        }],
        "followUps": ["region_exposure", "ltv_band", "data_quality"],
    }


def _data_quality(engine: Engine, as_of_display: str) -> Dict[str, Any]:
    """Governance evidence, read from the pipeline's own committed reports."""
    mapping = json.loads(MAPPING_REPORT.read_text(encoding="utf-8"))
    transform = json.loads(TRANSFORM_REPORT.read_text(encoding="utf-8"))
    delivery = json.loads(DELIVERY_REPORT.read_text(encoding="utf-8"))
    validation = pd.read_csv(VALIDATION_SUMMARY)

    mappings = mapping.get("mappings") or []
    exact = sum(1 for m in mappings if float(m.get("confidence") or 0) >= 1.0)
    fields = transform.get("fields") or {}
    parse_failures = sum(int(f.get("parse_failures") or 0)
                         for f in fields.values() if isinstance(f, dict))
    blocking = int((validation["materiality"] == "BLOCKING").sum()) \
        if "materiality" in validation else len(validation)

    rows = [
        {"gate": "1 · Semantic alignment",
         "measure": "Source headers mapped to canonical fields",
         "result": f"{len(mappings)} mapped, {exact} at full confidence"},
        {"gate": "— · Transform",
         "measure": "Typed fields with parse failures",
         "result": f"{parse_failures} of {len(fields)} fields"},
        {"gate": "2/3 · Validation",
         "measure": "Field-level exceptions raised",
         "result": f"{len(validation)} exception(s), {blocking} blocking"},
        {"gate": "5 · ESMA Annex 2 delivery",
         "measure": "Preflight on the projected regulatory output",
         "result": (f"{delivery.get('preflight', {}).get('status', 'n/a')} — "
                    f"{delivery.get('rows_in')} rows in / "
                    f"{delivery.get('rows_out')} out, "
                    f"{delivery.get('issues_total')} issues")},
    ]

    answer = (
        "Every figure on this page traces back through the same gated pipeline. "
        f"For the {as_of_display} cut, {len(mappings)} source headers were mapped "
        f"to canonical fields ({exact} at full confidence), the typed transform "
        f"recorded {parse_failures} parse failures, validation raised "
        f"{len(validation)} field-level exception(s) — both enumeration "
        "mismatches, held open rather than silently corrected — and the ESMA "
        "Annex 2 delivery preflight passed with "
        f"{delivery.get('issues_total')} issues. The exceptions are real, and "
        "Trakt reports them rather than resolving them for you."
    )

    return {
        "id": "data_quality",
        "label": "How do I know these numbers are right?",
        "category": "governance",
        "phrases": [
            "how do i know these numbers are right", "data quality", "validation",
            "provenance", "lineage", "audit", "can i trust", "how is this checked",
            "evidence", "where does the data come from", "source",
            "is this accurate", "governance",
        ],
        "answer": answer,
        "interpreted": "Governance evidence · pipeline gate reports for this run",
        "artifacts": [{
            "kind": "table",
            "title": "Pipeline evidence for this reporting cut",
            "columns": [
                {"key": "gate", "label": "Pipeline gate", "align": "left",
                 "format": "text", "scale": None},
                {"key": "measure", "label": "Measure", "align": "left",
                 "format": "text", "scale": None},
                {"key": "result", "label": "Result", "align": "left",
                 "format": "text", "scale": None},
            ],
            "rows": rows,
            "totalRows": len(rows),
            "coverage": None,
        }],
        "followUps": ["portfolio_risks", "funded_balance", "investor_report"],
    }


# --------------------------------------------------------------------------- #
# Report previews
# --------------------------------------------------------------------------- #
def _preview_page(title: str, subtitle: Optional[str],
                  intents: Dict[str, Dict[str, Any]],
                  intent_ids: List[str],
                  note: Optional[str] = None) -> Dict[str, Any]:
    """One preview page, assembled from already-computed intent artifacts.

    Several KPI intents share the engine's summary artifact (each carries the
    exposure count alongside its own measure), so the page would otherwise
    repeat "Exposures 36" once per intent. They are merged into a single KPI
    block, first occurrence wins, order preserved — a presentation change only.
    """
    blocks: List[Dict[str, Any]] = []
    kpis: List[Dict[str, str]] = []
    seen_labels: set[str] = set()
    coverage: Optional[float] = None
    kpi_slot: Optional[int] = None

    for iid in intent_ids:
        intent = intents.get(iid)
        if not intent:
            continue
        for art in intent["artifacts"]:
            if art["kind"] != "kpi":
                blocks.append(art)
                continue
            if kpi_slot is None:
                kpi_slot = len(blocks)
                blocks.append(None)  # placeholder, filled in below
            for kpi in art["kpis"]:
                if kpi["label"] in seen_labels:
                    continue
                seen_labels.add(kpi["label"])
                kpis.append(kpi)
            if coverage is None:
                coverage = art.get("coverage")

    if kpi_slot is not None:
        blocks[kpi_slot] = {"kind": "kpi", "title": None, "kpis": kpis,
                            "coverage": coverage}

    return {"title": title, "subtitle": subtitle, "blocks": blocks, "note": note}


def _reports(engine: Engine, intents: Dict[str, Dict[str, Any]],
             as_of_display: str) -> List[Dict[str, Any]]:
    """Report actions, previewed in-page. No document, no URL, no storage path.

    Page order mirrors the real deck definition in ``configs/pptx/investor_pack.yaml``
    for the slides this single-snapshot synthetic dataset can genuinely support.
    """
    scope = f"Synthetic Demo Lender · SYNTHETIC_ERE_Portfolio_012026 · as at {as_of_display}"

    investor = {
        "id": "investor_report",
        "label": "Generate the latest investor report.",
        "category": "report",
        "phrases": [
            "generate the latest investor report", "investor report",
            "investor pack", "investor deck", "funder report", "funding partner",
            "produce the investor pack", "quarterly investor",
        ],
        "documentTitle": "Investor & Funder MI Pack",
        "documentSubtitle": scope,
        "answer": (
            "The Investor & Funder MI Pack for "
            f"{as_of_display} is previewed below. In a client environment the same "
            "pack is generated as a branded PowerPoint from this identical "
            "governed dataset and delivered on schedule to each funding partner — "
            "the numbers in the deck are the numbers on this page, because both "
            "come from one calculation."
        ),
        "pages": [
            _preview_page("Executive summary", scope, intents,
                          ["funded_balance", "loan_count", "wa_ltv", "wa_rate"]),
            _preview_page("Funded stratifications — LTV & ticket size", None,
                          intents, ["ltv_band", "ticket_band"]),
            _preview_page("Funded stratifications — borrower age & channel", None,
                          intents, ["age_band", "channel"]),
            _preview_page("Geographic exposure", None, intents,
                          ["region_exposure"]),
            _preview_page("Methodology & data coverage", None, intents,
                          ["data_quality"],
                          note=("Pipeline evidence for the reporting cut. The "
                                "production pack also carries funded evolution, "
                                "vintage cohort progression, pipeline, "
                                "origination funnel, forecast bridge and risk "
                                "limit pages, which need reporting history and a "
                                "pipeline dataset that this public demonstration "
                                "does not publish.")),
        ],
        "followUps": ["management_summary", "data_quality", "region_exposure"],
    }

    management = {
        "id": "management_summary",
        "label": "Prepare a management summary.",
        "category": "report",
        "phrases": [
            "prepare a management summary", "management summary", "management report",
            "mi pack", "board report", "board pack", "exec summary",
            "executive summary", "summarise the portfolio", "summarize the portfolio",
            "monthly mi", "management information",
        ],
        "documentTitle": "Portfolio Management Summary",
        "documentSubtitle": scope,
        "answer": (
            f"The management summary for {as_of_display} is previewed below: "
            "headline KPIs, the concentration position and the governance "
            "evidence behind the figures. In a client environment this pack runs "
            "on a schedule and reaches finance, risk and leadership through "
            "whichever channel each team already uses."
        ),
        "pages": [
            _preview_page("Portfolio at a glance", scope, intents,
                          ["funded_balance", "loan_count", "wa_ltv", "wa_rate"]),
            _preview_page("Concentration position", None, intents,
                          ["portfolio_risks", "region_exposure"]),
            _preview_page("Governance evidence", None, intents, ["data_quality"],
                          note=("Every figure above is reproducible from the "
                                "governed canonical dataset for this reporting "
                                "cut.")),
        ],
        "followUps": ["investor_report", "portfolio_risks", "funded_balance"],
    }

    return [management, investor]


# --------------------------------------------------------------------------- #
# Pack assembly
# --------------------------------------------------------------------------- #
def build_pack() -> Dict[str, Any]:
    engine = Engine()
    as_of_display = _as_of_display(engine.as_of)

    intents: Dict[str, Dict[str, Any]] = {}
    for spec in INTENTS:
        intents[spec["id"]] = build_intent(engine, spec, as_of_display)

    for composite in (_concentration_table(engine, as_of_display),
                      _data_quality(engine, as_of_display)):
        intents[composite["id"]] = composite

    reports = _reports(engine, intents, as_of_display)

    delivery = json.loads(DELIVERY_REPORT.read_text(encoding="utf-8"))

    return {
        "packVersion": PACK_VERSION,
        "client": {
            "id": "synthetic_demo",
            "name": "Synthetic Demo Lender",
            "originator": "ERE Funding Limited",
            "description": (
                "A synthetic UK equity release lender used throughout the Trakt "
                "demonstration materials. Its book is a lifetime-mortgage "
                "portfolio with interest roll-up, funded through a warehouse "
                "facility and reported to ESMA Annex 2 for securitisation."
            ),
            "synthetic": True,
        },
        "portfolio": {
            "id": "SYNTHETIC_ERE_Portfolio_012026",
            "name": "Equity Release Portfolio",
            "assetClass": "UK equity release mortgages",
            "currency": "GBP",
            "country": "United Kingdom",
            "asOfDate": engine.as_of,
            "asOfDisplay": as_of_display,
            "loanCount": engine.loan_count,
            "totalBalance": round(engine.total_balance, 2),
            "totalBalanceDisplay": _fmt_currency(engine.total_balance),
            "regulatoryRegime": "ESMA Annex 2",
            "deliveryPreflight": (delivery.get("preflight") or {}).get("status"),
        },
        "provenance": {
            "sourceDataset": (
                "synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv"),
            "engine": [
                "mi_agent_api.funded_prep.prepare_funded_mi_dataset",
                "mi_agent.mi_agent_workflow.run_mi_agent_query",
                "mi_agent_api.adapters.adapt_workflow_result",
            ],
            "note": (
                "Every value in this pack was produced by the Trakt deterministic "
                "MI engine from the repository's synthetic canonical dataset. The "
                "landing page performs no portfolio calculation of its own."
            ),
        },
        "intents": list(intents.values()),
        "reports": reports,
        "unsupported": CONTROLLED_UNSUPPORTED,
    }


def _serialise(pack: Dict[str, Any]) -> str:
    return json.dumps(pack, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the committed pack matches a fresh build")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    pack = build_pack()
    text = _serialise(pack)
    out = Path(args.out)

    if args.check:
        if not out.exists():
            print(f"MISSING: {out}", file=sys.stderr)
            return 1
        if out.read_text(encoding="utf-8") != text:
            print(f"STALE: {out} differs from a fresh build. Re-run without "
                  f"--check.", file=sys.stderr)
            return 1
        print(f"OK: {out} is reproducible "
              f"({len(pack['intents'])} intents, {len(pack['reports'])} reports)")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} — {len(pack['intents'])} intents, "
          f"{len(pack['reports'])} reports, "
          f"{len(pack['unsupported'])} controlled-unsupported topics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
