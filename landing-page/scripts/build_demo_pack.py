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
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve()
_LANDING_ROOT = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from mi_agent.mi_agent_workflow import run_mi_agent_query  # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics  # noqa: E402
from mi_agent_api.adapters import adapt_workflow_result  # noqa: E402
from mi_agent_api.funded_prep import prepare_funded_mi_dataset  # noqa: E402

# --------------------------------------------------------------------------- #
# Repository inputs (all pre-existing; none created by the landing page)
# --------------------------------------------------------------------------- #
_MULTIBOOK = _REPO_ROOT / "synthetic_demo/output/multibook"

#: The current reporting period — every headline figure is as at this date.
CANONICAL = _MULTIBOOK / "platform_2026-06-30_canonical_typed.csv"
#: The prior period. Movement is computed by differencing the two governed
#: snapshots; no delta is ever stored.
CANONICAL_PRIOR = _MULTIBOOK / "platform_2026-05-31_canonical_typed.csv"

SEMANTICS = _REPO_ROOT / "mi_agent/mi_semantics_field_registry.yaml"
CLIENT_CONFIG = _REPO_ROOT / (
    "synthetic_demo/config/config_client_SYNTHETIC_MULTIBOOK.yaml")

#: Gate reports. Header mapping and transform are per book, so the governance
#: answer sums across the three.
MAPPING_REPORTS = sorted(_MULTIBOOK.glob("*_2026-06-30_header_mapping_report.json"))
TRANSFORM_REPORTS = sorted(_MULTIBOOK.glob("*_2026-06-30_transform_report.json"))
VALIDATION_SUMMARY = _MULTIBOOK / "validation/platform_2026-06-30_field_summary.csv"
DELIVERY_REPORT = _MULTIBOOK / (
    "platform_2026-06-30_ESMA_Annex2_delivery_report.json")
EXCEPTION_RECONCILIATION = _MULTIBOOK / (
    "platform_2026-06-30_ESMA_Annex2_exception_reconciliation.json")

# --------------------------------------------------------------------------- #
# Book identity — mirrors demo_platform.config and the generator.
# --------------------------------------------------------------------------- #
#: id -> (label, short label, balance-sheet status, one-line status detail)
BOOKS: Dict[str, Dict[str, str]] = {
    "alp_origination": {
        "label": "ALP Origination Book",
        "short": "Origination",
        "status": "warehoused",
        "detail": "Warehoused, destined for SPV2",
    },
    "alp_acquired": {
        "label": "ALP Acquired Back Book",
        "short": "Acquired",
        "status": "warehoused",
        "detail": "Warehoused, destined for SPV2",
    },
    "spv1_sponsored": {
        "label": "SPV1 Sponsored Securitisation",
        "short": "SPV1",
        "status": "sold",
        "detail": "Sold and derecognised; servicing and reporting retained",
    },
}

#: The warehoused platform: what sits on the sponsor's own balance sheet.
PLATFORM_BOOKS = ("alp_origination", "alp_acquired")
#: Everything the sponsor still reports on, including the deal it sold. Both
#: totals are legitimate; the demonstration names which is which rather than
#: silently picking one.
SPONSOR_BOOKS = tuple(BOOKS)

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
            "Aggregated across all three governed books, the funded position is "
            "{kpi0} over {loans} exposures as at {as_of}. This is the sponsor "
            "scope, which includes the sold SPV1 securitisation; ask for the "
            "balance by book to see the platform total on its own."
        ),
        "followUps": ["balance_by_book", "period_movement", "region_exposure"],
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
            "There are {kpi0} exposures across the three governed books as at "
            "{as_of}, carrying {balance} of current outstanding balance in "
            "aggregate."
        ),
        "followUps": ["balance_by_book", "ticket_band", "channel"],
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
# TODO(B4) — deliberately out of scope for this pass, recorded so they are not
# lost:
#
#   * Lineage trace as an ANSWER TYPE. "How do I know these numbers are right?"
#     currently returns gate-level counts. The richer answer walks one published
#     figure back through source header -> mapping -> transformation ->
#     validation -> output. The evidence already exists per book in
#     *_header_mapping_report.json and *_transform_report.json, and
#     engine/gate_2_transform/lineage_tracker.py is the production path; what is
#     missing is selecting a single figure and rendering its chain.
#
#   * Downloadable generated artefact from "Generate the latest investor report".
#     The preview is deliberately preview-only today — no document, no download
#     URL, no storage path — and that boundary is asserted by unit and E2E tests.
#     Producing a real artefact means deciding how a public visitor may receive
#     one, which is a hosting and access-control question, not a rendering one.

COMPOSITE_INTENT_IDS = ["portfolio_risks", "data_quality",
                        "balance_by_book", "period_movement",
                        "annex_exceptions"]

#: Report actions. These return an in-page preview only — never a document, a
#: download URL, or a storage path.
REPORT_INTENT_IDS = ["management_summary", "investor_report"]

#: Questions the public demo deliberately refuses, each with an honest reason.
#: These are *not* failures — they are the governed "we will not invent this"
#: behaviour, and the page shows them off on purpose.
CONTROLLED_UNSUPPORTED: List[Dict[str, str]] = [
    {
        "id": "loan_level",
        "label": "Show me individual loan records.",
        "phrases": [
            "individual loan records", "loan level", "loan-level", "show me the loans",
            "list the loans", "individual exposures", "borrower details",
            "customer data", "personal data", "raw data", "export the tape",
            "download the tape", "row level", "each loan",
        ],
        "reason": (
            "Trakt will not return exposure-level records to a public page. This "
            "is a governance boundary, not a data gap: the demonstration serves "
            "aggregated figures only, and no loan, borrower or collateral record "
            "leaves the governed dataset."
        ),
        "productionNote": (
            "In a client environment, exposure-level drill-through is available "
            "in the Trakt workspace, governed by role-based access within that "
            "client's own deployment."
        ),
    },
    {
        "id": "pipeline",
        "label": "Summarise the current pipeline.",
        "phrases": [
            "summarise the current pipeline", "summarize the pipeline", "pipeline",
            "applications", "new business pipeline", "funnel", "conversion", "offers",
            "forecast", "expected completions", "what will we fund",
        ],
        "reason": (
            "This demonstration holds funded loan tapes only, so it cannot "
            "calculate pipeline volumes, conversion or expected completions — "
            "there is no application-stage data behind it."
        ),
        "productionNote": (
            "In a production Trakt environment, this answer is generated from "
            "governed pipeline feeds: application-stage snapshots, the "
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
            "These three books carry no arrears, default or loss balances, so "
            "delinquency measures cannot be calculated — an answer here would be "
            "meaningless rather than merely empty."
        ),
        "productionNote": (
            "In a production Trakt environment, this answer is generated from "
            "governed servicing data: arrears, default, forbearance and loss "
            "analytics, where the portfolio's own data supports them."
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
class DemoSourceMismatch(RuntimeError):
    """The resolved dataset is not the one this landing page is pinned to."""


@dataclass(frozen=True)
class DemoSource:
    """An explicitly pinned demo dataset.

    Selection is deterministic and fails closed: the generator names one file,
    asserts its identity against the expectations below, and aborts rather than
    publishing figures from whatever dataset happened to resolve. There is no
    fallback path — in particular no fallback to whatever
    ``mi_agent_api.data_source`` would resolve as ``KIND_SYNTHETIC_DEMO``,
    because a silent substitution is exactly the failure this guards against.

    To repoint the landing page at a different governed portfolio, change this
    one object. Nothing else in the generator, and nothing at all in the
    TypeScript, needs to know which dataset is in use.
    """

    client_id: str
    client_name: str
    portfolio_id: str
    canonical_path: Path
    config_path: Path
    expected_currency: str
    expected_asset_class: str
    expected_reporting_date: str
    expected_min_balance: float
    expected_max_balance: float
    expected_min_exposures: int
    #: SHA-256 of the canonical CSV. Pins the exact bytes the pack was built
    #: from, so a silently edited dataset is caught even if the totals still
    #: fall inside the range above.
    expected_sha256: str

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        with self.canonical_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()


#: The portfolio this landing page publishes.
#:
#: NOTE ON PROVENANCE — read before changing.
#: This is the only governed portfolio dataset that exists in the repository.
#: A full search (every CSV with a balance column, every fixture, every mock,
#: git history, all branches, all untracked files) found no dataset between
#: £1.8bn and £2.0bn and no demo-video source material of any kind. The
#: largest figures elsewhere in the repository — £842.6MM and £0.97BN in
#: `frontend/mi-agent-ui/src/data/mockResponses.ts` — are hard-coded prose and
#: literal arrays inside `MockAgentClient`, flagged `mock: true`, with no rows
#: behind them; they are display copy for the workspace's offline mode, not a
#: portfolio. See `docs/implementation-note.md` § "Demo source provenance".
#:
#: If a larger governed portfolio is added to the repository, repoint the
#: landing page by editing this object and re-running the generator. The
#: assertions below will refuse anything that does not match.
DEMO_SOURCE = DemoSource(
    client_id="alderbridge_demo",
    client_name="Alderbridge Lending Platform",
    portfolio_id="ALP_Platform_202606",
    canonical_path=CANONICAL,
    config_path=CLIENT_CONFIG,
    expected_currency="GBP",
    expected_asset_class="equity_release",
    expected_reporting_date="2026-06-30",
    expected_min_balance=35_000_000.0,
    expected_max_balance=40_000_000.0,
    expected_min_exposures=100,
    expected_sha256="3a77cbaeb29e6bce2a23c469e1330e45643836c7d4d2e9ccf8c7cbf051c1b3d5",
)

#: The prior period, pinned the same way. Only its bytes and its balance are
#: asserted — every other guarantee comes from the current-period source.
PRIOR_SHA256 = "58fd505d0cfd0d8920e09a442c92f78c7a730f6f2473c5a777ecc2cbe9761b51"


def _mismatch(source: DemoSource, problems: List[str]) -> DemoSourceMismatch:
    try:
        shown = source.canonical_path.relative_to(_REPO_ROOT)
    except ValueError:
        shown = source.canonical_path
    return DemoSourceMismatch(
        "Landing-page demo source mismatch: resolved portfolio does not "
        "match the portfolio this landing page is pinned to.\n  - "
        + "\n  - ".join(problems)
        + f"\n\nPinned source: {source.client_name} / {source.portfolio_id}"
        f"\nResolved file: {shown}"
        "\n\nThe generator will not publish figures from an unverified "
        "dataset. Fix the dataset, or repoint DEMO_SOURCE deliberately.")


def _assert_source_file(source: DemoSource) -> None:
    """Identity checks that run BEFORE the dataset is parsed.

    Ordering matters: a substituted file must fail with a clear mismatch, not
    with whatever schema error the preparation step happens to raise first.
    """
    if not source.canonical_path.exists():
        raise _mismatch(source, ["the pinned canonical dataset does not exist"])
    if not source.config_path.exists():
        raise _mismatch(source, ["the pinned client config does not exist"])

    actual_sha = source.fingerprint()
    # A placeholder hash means "not yet pinned"; a real one is enforced.
    if set(source.expected_sha256) != {"0"} and actual_sha != source.expected_sha256:
        raise _mismatch(source, [
            f"canonical fingerprint {actual_sha[:16]}… != "
            f"{source.expected_sha256[:16]}… (the dataset was replaced or edited)"])


def _assert_source_identity(source: DemoSource, total_balance: float,
                            loan_count: int, as_of: str) -> None:
    """Identity checks that need the prepared dataset.

    Every check names what it expected and what it got, so a mismatch is
    diagnosable from the error alone.
    """
    problems: List[str] = []

    config = yaml.safe_load(source.config_path.read_text(encoding="utf-8")) or {}
    client = config.get("client") or {}
    portfolio = config.get("portfolio") or {}

    if client.get("client_id") != source.client_id:
        problems.append(
            f"client id {client.get('client_id')!r} != {source.client_id!r}")
    if client.get("display_name") != source.client_name:
        problems.append(
            f"client name {client.get('display_name')!r} != {source.client_name!r}")
    if portfolio.get("asset_class") != source.expected_asset_class:
        problems.append(
            f"asset class {portfolio.get('asset_class')!r} != "
            f"{source.expected_asset_class!r}")
    if portfolio.get("base_currency") != source.expected_currency:
        problems.append(
            f"currency {portfolio.get('base_currency')!r} != "
            f"{source.expected_currency!r}")
    if as_of != source.expected_reporting_date:
        problems.append(
            f"reporting date {as_of!r} != {source.expected_reporting_date!r}")
    if not (source.expected_min_balance <= total_balance <= source.expected_max_balance):
        problems.append(
            f"total balance {total_balance:,.2f} outside the expected range "
            f"{source.expected_min_balance:,.0f}–{source.expected_max_balance:,.0f}")
    if loan_count < source.expected_min_exposures:
        problems.append(
            f"exposure count {loan_count} below the expected minimum "
            f"{source.expected_min_exposures}")

    if problems:
        raise _mismatch(source, problems)


class Engine:
    """The pinned dataset, prepared exactly as the MI API prepares it."""

    def __init__(self, source: DemoSource = DEMO_SOURCE) -> None:
        self.source = source
        # Verified before a single row is parsed.
        _assert_source_file(source)
        raw = pd.read_csv(source.canonical_path, low_memory=False)
        self.frame, self.prep_report = prepare_funded_mi_dataset(raw)
        self.semantics = load_mi_semantics(SEMANTICS)
        self.total_balance = float(
            pd.to_numeric(self.frame["current_outstanding_balance"],
                          errors="coerce").fillna(0).sum())
        self.loan_count = int(len(self.frame))
        cutoff = self.frame.get("data_cut_off_date")
        values = sorted({str(v) for v in cutoff.dropna().unique()}) if cutoff is not None else []
        self.as_of = values[-1] if values else source.expected_reporting_date
        self.fingerprint = source.fingerprint()

        # Nothing below this line runs until the dataset is verified.
        _assert_source_identity(source, self.total_balance, self.loan_count,
                                self.as_of)

    def run(self, question: str) -> Dict[str, Any]:
        workflow = run_mi_agent_query(question, self.frame, self.semantics)
        if not workflow.get("ok"):
            raise RuntimeError(
                f"engine refused {question!r}: {workflow.get('error')}")
        return adapt_workflow_result(workflow, portfolio_id=self.source.client_id,
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


def _book_totals(frame: "pd.DataFrame") -> Dict[str, Dict[str, float]]:
    """Balance and exposure count per book, from the governed frame.

    Book identity is a column on every row because Gate 2 stamped it, so this
    is a group-by on the governed model rather than a filter invented here.
    """
    balances = pd.to_numeric(frame["current_outstanding_balance"],
                             errors="coerce").fillna(0)
    out: Dict[str, Dict[str, float]] = {}
    for book_id in BOOKS:
        mask = frame["source_portfolio_id"].astype(str) == book_id
        out[book_id] = {
            "balance": float(balances[mask].sum()),
            "loans": int(mask.sum()),
        }
    return out


def _balance_by_book(engine: Engine, as_of_display: str) -> Dict[str, Any]:
    """Funded balance per book, with BOTH governed totals.

    The sponsor originated SPV1 and then sold it, so it is off balance sheet —
    and the sponsor still services it, retains risk and reports to its
    noteholders. Two totals are therefore correct at once, and collapsing them
    into one would misstate whichever question was actually being asked. The
    answer gives both and says which is which.
    """
    totals = _book_totals(engine.frame)
    platform_balance = sum(totals[b]["balance"] for b in PLATFORM_BOOKS)
    platform_loans = sum(totals[b]["loans"] for b in PLATFORM_BOOKS)
    sponsor_balance = sum(totals[b]["balance"] for b in SPONSOR_BOOKS)
    sponsor_loans = sum(totals[b]["loans"] for b in SPONSOR_BOOKS)

    rows = []
    for book_id, meta in BOOKS.items():
        rows.append({
            "book": meta["label"],
            "status": meta["detail"],
            "exposures": f"{totals[book_id]['loans']:,}",
            "balance": _fmt_currency(totals[book_id]["balance"]),
        })
    rows.append({
        "book": "Platform total (warehoused)",
        "status": "On balance sheet",
        "exposures": f"{platform_loans:,}",
        "balance": _fmt_currency(platform_balance),
    })
    rows.append({
        "book": "Sponsor total (including SPV1)",
        "status": "Everything the sponsor reports on",
        "exposures": f"{sponsor_loans:,}",
        "balance": _fmt_currency(sponsor_balance),
    })

    answer = (
        f"Across three governed books as at {as_of_display}, the platform carries "
        f"{_fmt_currency(platform_balance)} over {platform_loans:,} exposures on "
        f"balance sheet. Including SPV1 — originated by the sponsor, securitised "
        f"and sold, with servicing, risk retention and investor reporting "
        f"retained — the sponsor reports on {_fmt_currency(sponsor_balance)} over "
        f"{sponsor_loans:,} exposures. Both totals are correct; they answer "
        "different questions, so Trakt returns both rather than choosing one."
    )

    return {
        "id": "balance_by_book",
        "label": "Show the funded balance by book.",
        "category": "portfolio_kpi",
        "phrases": [
            "funded balance by book", "balance by book", "by book", "each book",
            "split by book", "by portfolio", "by spv", "spv1", "spv 1",
            "which books", "how many books", "book breakdown", "per book",
            "origination book", "acquired book", "securitisation",
            "balance sheet", "sponsor total", "platform total",
        ],
        "answer": answer,
        "interpreted": "Balance by source portfolio; platform and sponsor scopes",
        "artifacts": [{
            "kind": "table",
            "title": "Funded balance by governed book",
            "columns": [
                {"key": "book", "label": "Book", "align": "left"},
                {"key": "status", "label": "Status", "align": "left"},
                {"key": "exposures", "label": "Exposures", "align": "right"},
                {"key": "balance", "label": "Balance", "align": "right"},
            ],
            "rows": rows,
        }],
        "followUps": ["period_movement", "annex_exceptions", "data_quality"],
    }


def _period_movement(engine: Engine, as_of_display: str) -> Dict[str, Any]:
    """Movement between the two governed snapshots, per book and in aggregate.

    Differenced from the two committed canonicals at build time. Nothing here is
    a stored delta: change the prior snapshot and this answer changes with it.
    """
    digest = hashlib.sha256(CANONICAL_PRIOR.read_bytes()).hexdigest()
    if digest != PRIOR_SHA256:
        raise DemoSourceMismatch(
            f"prior snapshot {CANONICAL_PRIOR.name} has SHA-256 {digest}, "
            f"expected {PRIOR_SHA256}. Movement would be computed against an "
            "unverified prior position."
        )

    prior_raw = pd.read_csv(CANONICAL_PRIOR, low_memory=False)
    prior_frame, _ = prepare_funded_mi_dataset(prior_raw)
    prior_date = sorted({str(v) for v in
                         prior_frame["data_cut_off_date"].dropna().unique()})[-1]
    prior_display = _as_of_display(prior_date)

    now = _book_totals(engine.frame)
    was = _book_totals(prior_frame)

    def _delta(current: float, previous: float) -> str:
        change = current - previous
        return f"{'+' if change >= 0 else '−'}{_fmt_currency(abs(change))}"

    def _pct(current: float, previous: float) -> str:
        return "n/a" if previous == 0 else f"{(current - previous) / previous * 100:+.1f}%"

    rows = []
    for book_id, meta in BOOKS.items():
        rows.append({
            "book": meta["label"],
            "exposures": f"{was[book_id]['loans']:,} → {now[book_id]['loans']:,}",
            "movement": _delta(now[book_id]["balance"], was[book_id]["balance"]),
            "pct": _pct(now[book_id]["balance"], was[book_id]["balance"]),
        })

    platform_now = sum(now[b]["balance"] for b in PLATFORM_BOOKS)
    platform_was = sum(was[b]["balance"] for b in PLATFORM_BOOKS)
    sponsor_now = sum(now[b]["balance"] for b in SPONSOR_BOOKS)
    sponsor_was = sum(was[b]["balance"] for b in SPONSOR_BOOKS)
    rows.append({
        "book": "Platform total (warehoused)",
        "exposures": (f"{sum(was[b]['loans'] for b in PLATFORM_BOOKS):,} → "
                      f"{sum(now[b]['loans'] for b in PLATFORM_BOOKS):,}"),
        "movement": _delta(platform_now, platform_was),
        "pct": _pct(platform_now, platform_was),
    })
    rows.append({
        "book": "Sponsor total (including SPV1)",
        "exposures": (f"{sum(was[b]['loans'] for b in SPONSOR_BOOKS):,} → "
                      f"{sum(now[b]['loans'] for b in SPONSOR_BOOKS):,}"),
        "movement": _delta(sponsor_now, sponsor_was),
        "pct": _pct(sponsor_now, sponsor_was),
    })

    added = now["alp_origination"]["loans"] - was["alp_origination"]["loans"]
    left = was["alp_acquired"]["loans"] - now["alp_acquired"]["loans"]

    answer = (
        f"Comparing the {as_of_display} cut with {prior_display}, the platform "
        f"moved {_delta(platform_now, platform_was)} to "
        f"{_fmt_currency(platform_now)} ({_pct(platform_now, platform_was)}). The "
        f"origination book added {added} exposures and the acquired book lost "
        f"{left}; the remainder is interest roll-up on a book that capitalises "
        f"rather than amortises. Including SPV1 the sponsor position moved "
        f"{_delta(sponsor_now, sponsor_was)} to {_fmt_currency(sponsor_now)}. Both "
        "figures are differenced from two governed snapshots — Trakt holds no "
        "stored deltas."
    )

    return {
        "id": "period_movement",
        "label": "How has the portfolio changed since last month?",
        "category": "temporal",
        "phrases": [
            "changed since last month", "since last month", "month on month",
            "month-on-month", "movement", "versus last month", "vs last month",
            "compared to last month", "trend", "over time", "evolution",
            "growth", "how has the portfolio changed", "period on period",
        ],
        "answer": answer,
        "interpreted": (
            f"Movement {prior_display} to {as_of_display}; two governed snapshots"),
        "artifacts": [{
            "kind": "table",
            "title": f"Movement {prior_display} to {as_of_display}",
            "columns": [
                {"key": "book", "label": "Book", "align": "left"},
                {"key": "exposures", "label": "Exposures", "align": "right"},
                {"key": "movement", "label": "Balance movement", "align": "right"},
                {"key": "pct", "label": "Change", "align": "right"},
            ],
            "rows": rows,
        }],
        "followUps": ["balance_by_book", "annex_exceptions", "region_exposure"],
    }


def _annex_exceptions(engine: Engine, as_of_display: str) -> Dict[str, Any]:
    """ESMA Annex 2 validation exceptions, with the reasoning behind each.

    Read from the Gate 4b reconciliation, which classifies every exception from
    either validation gate by its actual effect on the submission. A pass/fail
    verdict would say nothing a lender could act on; what costs them weeks is
    knowing which field, which book, what the regime requires, what the source
    actually contained and what would clear it.
    """
    recon = json.loads(EXCEPTION_RECONCILIATION.read_text(encoding="utf-8"))
    totals = recon.get("totals") or {}
    exceptions = recon.get("exceptions") or []

    label_for = {
        "BLOCKS_DELIVERY": "Blocks submission",
        "DEFAULTED_AT_DELIVERY": "Defaults to a no-data code",
        "RESOLVED_AT_PROJECTION": "Resolved at projection",
        "OUT_OF_REGIME_SCOPE": "Outside Annex 2 scope",
    }

    rows = []
    for item in exceptions:
        annex = (item.get("annex") or [{}])[0]
        code = annex.get("code") or "—"
        field = item.get("canonicalField") or item.get("businessRule") or "—"
        books = ", ".join(
            f"{b['bookLabel']} ({b['exposures']})" for b in item.get("byBook") or []
        ) or "—"
        observed = item.get("observedValues") or []
        rows.append({
            "disposition": label_for.get(item.get("disposition"),
                                         str(item.get("disposition"))),
            "field": f"{code} · {field}",
            "books": books,
            "found": str(observed[0])[:70] if observed else "no value supplied",
            "resolution": str(item.get("resolution", ""))[:150],
        })

    blocking = int(totals.get("blocksDelivery") or 0)
    defaulted = int(totals.get("defaultedAtDelivery") or 0)
    out_of_scope = int(totals.get("outOfRegimeScope") or 0)
    raised = int(totals.get("canonicalExceptions") or 0)

    answer = (
        f"The {as_of_display} Annex 2 preparation raised {raised} exceptions "
        f"across the three books. {blocking} would block the submission, "
        f"{defaulted} would pass while asserting a no-data code the source never "
        f"supplied, and {out_of_scope} are governed data-quality findings that do "
        "not reach the return. Each carries its Annex reference, the book and "
        "exposures affected, what the source contained and what would clear it. "
        "Trakt reports them; it does not resolve them for you."
    )

    return {
        "id": "annex_exceptions",
        "label": "Show the current Annex 2 validation exceptions.",
        "category": "governance",
        "phrases": [
            "annex 2 validation exceptions", "annex 2 exceptions", "validation exceptions",
            "annex exceptions", "regulatory exceptions", "esma exceptions",
            "what is blocking the submission", "submission blocked", "annex 2",
            "esma annex", "regulatory errors", "what would fail", "preflight",
            "can we submit", "submission ready", "exceptions",
        ],
        "answer": answer,
        "interpreted": "Annex 2 exception reconciliation; Gate 3 and Gate 4b",
        "artifacts": [{
            "kind": "table",
            "title": "ESMA Annex 2 exceptions, by effect on the submission",
            "columns": [
                {"key": "disposition", "label": "Effect", "align": "left"},
                {"key": "field", "label": "Annex field", "align": "left"},
                {"key": "books", "label": "Book (exposures)", "align": "left"},
                {"key": "found", "label": "Source value", "align": "left"},
                {"key": "resolution", "label": "What would clear it", "align": "left"},
            ],
            "rows": rows,
        }],
        "followUps": ["data_quality", "balance_by_book", "investor_report"],
    }


def _data_quality(engine: Engine, as_of_display: str) -> Dict[str, Any]:
    """Governance evidence, read from the pipeline's own committed reports."""
    # Three books, so three of each gate report. The governance answer sums
    # across them: the client onboarded three source schemas, not one.
    mappings = []
    for report in MAPPING_REPORTS:
        mappings.extend(json.loads(report.read_text(encoding="utf-8")).get("mappings") or [])
    fields: Dict[str, Any] = {}
    parse_failures = 0
    for report in TRANSFORM_REPORTS:
        book_fields = json.loads(report.read_text(encoding="utf-8")).get("fields") or {}
        fields.update(book_fields)
        parse_failures += sum(int(f.get("parse_failures") or 0)
                              for f in book_fields.values() if isinstance(f, dict))

    delivery = json.loads(DELIVERY_REPORT.read_text(encoding="utf-8"))
    recon = json.loads(EXCEPTION_RECONCILIATION.read_text(encoding="utf-8"))
    validation = pd.read_csv(VALIDATION_SUMMARY)

    exact = sum(1 for m in mappings if float(m.get("confidence") or 0) >= 1.0)
    blocking = int((validation["materiality"] == "BLOCKING").sum()) \
        if "materiality" in validation else len(validation)

    totals = recon.get("totals") or {}
    rows = [
        {"gate": "1 · Semantic alignment",
         "measure": "Source headers mapped to canonical fields, across three books",
         "result": f"{len(mappings)} mapped, {exact} at full confidence"},
        {"gate": "— · Transform",
         "measure": "Typed fields with parse failures",
         "result": f"{parse_failures} of {len(fields)} fields"},
        {"gate": "2/3 · Validation",
         "measure": "Field-level exceptions raised",
         "result": f"{len(validation)} exception(s) on the governed model"},
        {"gate": "4b · ESMA Annex 2 delivery",
         "measure": "Exceptions by effect on the submission",
         "result": (f"{totals.get('blocksDelivery', 0)} blocking, "
                    f"{totals.get('defaultedAtDelivery', 0)} no-data defaulted, "
                    f"{totals.get('outOfRegimeScope', 0)} outside regime scope")},
    ]

    answer = (
        "Every figure on this page traces back through the same gated pipeline. "
        f"For the {as_of_display} cut, {len(mappings)} source headers across three "
        f"books were mapped to canonical fields ({exact} at full confidence), the "
        f"typed transform recorded {parse_failures} parse failures, and validation "
        f"raised {len(validation)} exception(s). The Annex 2 reconciliation then "
        "classifies each one by whether it actually blocks the submission — "
        f"{totals.get('blocksDelivery', 0)} do. The exceptions are real, and Trakt "
        "reports them rather than resolving them for you."
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
        "followUps": ["annex_exceptions", "balance_by_book", "region_exposure"],
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
                      _data_quality(engine, as_of_display),
                      _balance_by_book(engine, as_of_display),
                      _period_movement(engine, as_of_display),
                      _annex_exceptions(engine, as_of_display)):
        intents[composite["id"]] = composite

    reports = _reports(engine, intents, as_of_display)

    delivery = json.loads(DELIVERY_REPORT.read_text(encoding="utf-8"))

    return {
        "packVersion": PACK_VERSION,
        # Identity of the dataset this pack was built from. The runtime
        # validates it against its own pinned expectations, so a pack built
        # from the wrong portfolio cannot be served even if it is committed.
        "source": {
            "clientId": engine.source.client_id,
            "portfolioId": engine.source.portfolio_id,
            "currency": engine.source.expected_currency,
            "assetClass": engine.source.expected_asset_class,
            "reportingDate": engine.as_of,
            "totalBalance": round(engine.total_balance, 2),
            "exposures": engine.loan_count,
            "canonicalSha256": engine.fingerprint,
        },
        "client": {
            "id": "alderbridge_demo",
            "name": "Alderbridge Lending Platform",
            "originator": "Alderbridge Lending Platform (synthetic)",
            # Deliberately empty. The page previously carried a paragraph of
            # scene-setting here; the scope header above it already states the
            # client, books, exposures, balance and date, which is what a reader
            # needs before an answer means anything.
            "description": "",
            "synthetic": True,
        },
        "portfolio": {
            "id": "ALP_Platform_202606",
            "name": "Three governed books",
            # Named generically on purpose: a single asset class tells a bridging
            # lender, an auto-loan buyer or a private-credit manager that the
            # product was not built for them.
            "assetClass": "UK asset-backed portfolio",
            "currency": "GBP",
            "country": "United Kingdom",
            "asOfDate": engine.as_of,
            "asOfDisplay": as_of_display,
            "loanCount": engine.loan_count,
            "totalBalance": round(engine.total_balance, 2),
            "totalBalanceDisplay": _fmt_currency(engine.total_balance),
            "regulatoryRegime": "ESMA Annex 2",
            "deliveryPreflight": (delivery.get("preflight") or {}).get("status"),
            "platformBalanceDisplay": _fmt_currency(
                sum(_book_totals(engine.frame)[b]["balance"] for b in PLATFORM_BOOKS)),
            "books": [
                {
                    "id": book_id,
                    "label": meta["label"],
                    "shortLabel": meta["short"],
                    "balanceSheetStatus": meta["status"],
                    "statusDetail": meta["detail"],
                    "balanceDisplay": _fmt_currency(
                        _book_totals(engine.frame)[book_id]["balance"]),
                }
                for book_id, meta in BOOKS.items()
            ],
            "priorAsOfDate": "2026-05-31",
        },
        "provenance": {
            "sourceDataset": (
                "synthetic_demo/output/multibook/platform_2026-06-30_canonical_typed.csv"),
            "priorDataset": (
                "synthetic_demo/output/multibook/platform_2026-05-31_canonical_typed.csv"),
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
