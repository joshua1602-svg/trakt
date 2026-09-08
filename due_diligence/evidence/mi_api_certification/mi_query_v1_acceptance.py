"""MI QUERY AGENT V1 — LIVE ACCEPTANCE.

Drives the FROZEN 135-question bank against the DEPLOYED /mi/query and scores
every answer against truth taken from a different production surface.

WHAT MAKES THIS AN ACCEPTANCE RATHER THAN A TRANSCRIPT.

  * The bank is frozen and committed before the first production call, so no
    expectation can be written after its answer was seen.
  * The deployed commit is established before any question is asked, and a
    service serving something else is NOT EXECUTABLE — never a pass.
  * Truth comes from the dashboard GET surface or from an identity the harness
    recomputes itself. Asking /mi/query twice is not evidence.
  * Paraphrase invariance is scored as its own gate: three genuinely different
    phrasings of one semantic case must produce one semantic answer. A bank
    that only ever asks a capability its house phrasing cannot find the place
    where a real user's wording falls off the governed path.
  * There is no percentage bar. A single wrong number is a wrong number.

EXIT CODES.  0 accepted · 1 not accepted · 2 not executable.
A 2 fails deliberately: "we could not run it" must never read as a pass.

REUSES the existing certification transport (`certify_mi_api._live_asker`) and
the existing deployed-commit provenance check (`certify_mi_api.preflight`).
There is one live client and one MI_BEARER convention in this repository, and
this file does not add a second.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import mi_query_v1_truth as truth_mod
from .mi_query_v1_truth import UNAVAILABLE, UNRESOLVED
from .certify_mi_api import _live_asker, preflight

BANK_PATH = Path(__file__).with_name("mi_query_v1_bank.json")

CORRECT = "CORRECT"
CORRECT_REFUSAL = "CORRECT_REFUSAL"
WRONG = "WRONG"
INCORRECT_REFUSAL = "INCORRECT_REFUSAL"
ERROR = "ERROR"
UNSCOREABLE = "UNSCOREABLE"

#: Why a question failed. Recorded alongside the outcome because "WRONG" on its
#: own tells an engineer nothing about where to look.
R_NUMBER = "NUMBER_DISAGREES_WITH_INDEPENDENT_TRUTH"
R_MEASURE_SUB = "MEASURE_SUBSTITUTION"
R_DATASET_SUB = "DATASET_SUBSTITUTION"
R_FILTER = "FILTER_NOT_APPLIED"
R_STOCK = "STOCK_RETURNED_FOR_A_TRANSITION"
R_CELLS = "CELLS_DO_NOT_RECONCILE"
R_BASIS = "BASIS_MISMATCH"
R_SERIES = "SERIES_DISAGREES_WITH_INDEPENDENT_TRUTH"
R_BRIDGE = "BRIDGE_DOES_NOT_RECONCILE"
R_BRIDGE_PERIODS = "BRIDGE_SPANS_THE_WRONG_PERIOD_PAIR"
R_REFUSAL_REASON = "REFUSAL_REASON_NOT_GOVERNED"
R_REFUSED_ANSWERABLE = "REFUSED_AN_ANSWERABLE_QUESTION"
R_ANSWERED_UNSUPPORTED = "ANSWERED_AN_UNSUPPORTED_FACET"
R_ARTIFACT_ON_REFUSAL = "ARTIFACT_PUBLISHED_ON_REFUSAL"
R_INJECTION = "INJECTED_FIGURE_ASSERTED"
R_ORDERING = "SCENARIO_ORDERING_VIOLATED"
R_MONOTONIC = "COHORT_MEMBERSHIP_NOT_FIXED"
R_UNBOUNDED = "RATE_OUTSIDE_ZERO_TO_ONE_HUNDRED"
R_NO_PRIMARY = "PRIMARY_VALUE_INDETERMINATE"
R_TRANSPORT = "TRANSPORT_ERROR"
R_TRUTH_UNAVAILABLE = "INDEPENDENT_TRUTH_UNAVAILABLE"
R_RULE_UNRESOLVED = "ANSWERABILITY_RULE_UNRESOLVED"
R_ROUTE = "ROUTE_OUTSIDE_EXPECTED"


# --------------------------------------------------------------------------- #
# Reading numbers out of an answer
# --------------------------------------------------------------------------- #
_SUFFIX = {"k": 1e3, "m": 1e6, "mm": 1e6, "bn": 1e9, "b": 1e9, "tn": 1e12}
_NUM_RE = re.compile(
    r"(?P<sign>[-+−]?)\s*[£$€]?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<suffix>MM|mm|bn|BN|Bn|[KkMmB])?\s*(?P<pct>%)?")


def _numbers_in_text(text: str) -> List[Dict[str, Any]]:
    """Every number a reader would see, with the PRECISION it was shown at.

    `£5.4MM` asserts 5,400,000 to one decimal place of a million, so it is
    compared with a tolerance of half of that. Widening the tolerance to make a
    compact figure agree with an exact one is how a wrong number passes; deriving
    it from the digits actually printed is how a right one does."""
    out: List[Dict[str, Any]] = []
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group("num")
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if m.group("sign") in ("-", "−"):
            value = -value
        suffix = (m.group("suffix") or "").lower()
        scale = _SUFFIX.get(suffix, 1.0)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        # Half a unit in the last printed place, carried through the suffix.
        tolerance = 0.5 * (10 ** -decimals) * scale
        if "," not in raw and decimals == 0 and scale == 1.0:
            tolerance = 0.5
        out.append({"value": value * scale, "tolerance": tolerance,
                    "source": "answer_text", "shown": m.group(0).strip(),
                    "is_pct": bool(m.group("pct")), "span": m.span()})
    return out


def _numeric_leaves(node: Any, path: str = "") -> List[Dict[str, Any]]:
    """Exact figures carried in the artifacts, where nothing was rounded."""
    found: List[Dict[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(_numeric_leaves(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for i, value in enumerate(node[:400]):
            found.extend(_numeric_leaves(value, f"{path}[{i}]"))
    elif isinstance(node, bool):
        pass
    elif isinstance(node, (int, float)):
        found.append({"value": float(node), "tolerance": 0.005,
                      "source": f"artifact:{path}", "shown": str(node),
                      "is_pct": False})
    return found


def candidates(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (_numbers_in_text(str(envelope.get("answer") or ""))
            + _numeric_leaves(envelope.get("artifacts") or []))


def matches(cands: List[Dict[str, Any]], target: float) -> Optional[Dict[str, Any]]:
    """The first candidate that states ``target`` at the precision it was shown."""
    for c in cands:
        tol = max(c["tolerance"], abs(target) * 1e-9)
        if abs(c["value"] - target) <= tol:
            return c
    return None


def _shown_with_tolerance(cands: List[Dict[str, Any]], limit: int = 8) -> str:
    """Each figure the answer stated, with the tolerance it was compared at.

    A disagreement cannot be classified without this: `£159.1MM` is compared at
    +/- 50,000 because that is what one decimal place of a million asserts,
    while `159,097,304.07` is compared at +/- 0.005. Printing the figure alone
    leaves a reader unable to tell a wrong number from a coarse one."""
    out = []
    for c in cands[:limit]:
        out.append(f"{c['shown']} (={c['value']:,.2f} +/-{c['tolerance']:,.2f}, "
                   f"{c['source']})")
    if len(cands) > limit:
        out.append(f"... {len(cands) - limit} more")
    return "; ".join(out) or "no figure at all"


def service_population(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """WHAT THE SERVICE SAYS IT MEASURED — its own reconciliation block.

    The service publishes the dataset, the row count and the balance it
    actually included. That is the other half of a numeric disagreement: two
    correct figures over different populations disagree, and without this the
    comparison cannot say which kind of disagreement it found."""
    trace = envelope.get("queryTrace") or {}
    rec = trace.get("reconciliation") if isinstance(trace.get("reconciliation"), dict) else {}
    meta = envelope.get("metadata") or {}
    return {
        "dataset": rec.get("dataset") or meta.get("datasetContext"),
        "records_included": rec.get("records_included", rec.get("total_records")),
        "balance_included": rec.get("balance_included", rec.get("total_balance")),
        "filters_applied": rec.get("filters_applied"),
        "filters": rec.get("filters"),
        "as_of": meta.get("asOfDate"),
        "metric": trace.get("metric"),
        "aggregation": trace.get("aggregation"),
        "result_type": trace.get("resultType") or meta.get("resultType"),
        "grouped_by": trace.get("executedGroupFieldKeys") or trace.get("applied_dimensions"),
    }


def primary_value(envelope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """THE FIGURE THE ANSWER LEADS WITH — the one a reader takes away.

    Substitution checks are evaluated against this and not against every number
    in the envelope, because a correct answer legitimately carries context
    figures ("36 loans", "as at 30 November") that must not be read as claims.
    A KPI artifact's first tile wins where there is one; otherwise the first
    number in the prose. When neither exists the check reports INDETERMINATE
    rather than passing."""
    for artifact in (envelope.get("artifacts") or []):
        if not isinstance(artifact, dict) or artifact.get("type") != "kpi":
            continue
        for tile in (artifact.get("kpis") or []):
            if isinstance(tile, dict):
                for key in ("raw", "value", "amount"):
                    if isinstance(tile.get(key), (int, float)) and not isinstance(
                            tile.get(key), bool):
                        return {"value": float(tile[key]), "tolerance": 0.005,
                                "source": "artifact:kpi", "shown": str(tile.get(key))}
    text_numbers = _numbers_in_text(str(envelope.get("answer") or ""))
    return text_numbers[0] if text_numbers else None


def model_lineage(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """What the runtime EVIDENCES about the model behind this answer.

    Absence is reported as absence. An absent field is not a positive finding,
    and "deterministic" must not be inferred from a missing key."""
    meta = envelope.get("metadata") or {}
    trace = envelope.get("queryTrace") or {}
    named: Optional[str] = None
    for block_name in ("llm", "llmConfig", "modelUsage", "parserProvenance"):
        block = meta.get(block_name)
        if isinstance(block, dict):
            for key in ("model", "modelName", "deployment", "modelId"):
                if isinstance(block.get(key), str) and block[key].strip():
                    named = block[key].strip()
                    break
        if named:
            break
    return {
        "llm_model": named or "not exposed by runtime",
        "parser_mode": meta.get("parserMode") or trace.get("parserMode")
                       or "not exposed by runtime",
        "parser_mode_detail": meta.get("parserModeDetail"),
        "engine": meta.get("engine"),
        "model_availability": meta.get("modelAvailability"),
    }


# --------------------------------------------------------------------------- #
# Reading the artifacts
# --------------------------------------------------------------------------- #
PASS, FAIL, INDET = "PASS", "FAIL", "INDETERMINATE"

_LABEL_HINTS = ("geography", "region", "stage", "product", "bucket", "band",
                "label", "category", "key", "name", "period", "vintage")
_BALANCE_HINTS = ("balance", "amount", "exposure", "value_sum", "pipelineamount")
_COUNT_HINTS = ("count", "loans", "cases")


def _rows_artifact(envelope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for artifact in (envelope.get("artifacts") or []):
        if isinstance(artifact, dict) and artifact.get("type") == "table" and artifact.get("rows"):
            return artifact
    for artifact in (envelope.get("artifacts") or []):
        if isinstance(artifact, dict) and artifact.get("rows"):
            return artifact
    return None


def _pick_column(rows: List[Dict[str, Any]], hints: Tuple[str, ...],
                 numeric: bool) -> Optional[str]:
    if not rows:
        return None
    keys = list(rows[0].keys())
    for hint in hints:
        for key in keys:
            if hint in key.lower():
                sample = rows[0].get(key)
                if numeric and isinstance(sample, (int, float)) and not isinstance(sample, bool):
                    return key
                if not numeric and isinstance(sample, str):
                    return key
    if numeric:
        for key in keys:
            sample = rows[0].get(key)
            if isinstance(sample, (int, float)) and not isinstance(sample, bool):
                return key
    else:
        for key in keys:
            if isinstance(rows[0].get(key), str):
                return key
    return None


def _cells(envelope: Dict[str, Any], numeric_hints: Tuple[str, ...]
           ) -> Optional[List[Tuple[str, float]]]:
    artifact = _rows_artifact(envelope)
    if not artifact:
        return None
    rows = [r for r in (artifact.get("rows") or []) if isinstance(r, dict)]
    label_key = _pick_column(rows, _LABEL_HINTS, numeric=False)
    value_key = artifact.get("valueKey") if artifact.get("valueKey") in (
        rows[0].keys() if rows else []) else None
    value_key = value_key or _pick_column(rows, numeric_hints, numeric=True)
    if not rows or not label_key or not value_key:
        return None
    out: List[Tuple[str, float]] = []
    for row in rows:
        value = row.get(value_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append((str(row.get(label_key)), float(value)))
    return out or None


def _norm(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


def _close(a: float, b: float, rel: float = 0.005, absolute: float = 0.02) -> bool:
    return abs(a - b) <= max(absolute, abs(b) * rel)


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #
class Ctx:
    def __init__(self, question: Dict[str, Any], envelope: Dict[str, Any],
                 truths: Dict[str, Any], run: Dict[str, Any]):
        self.q = question
        self.e = envelope
        self.t = truths
        self.run = run
        self.cands = candidates(envelope)
        self.primary = primary_value(envelope)
        self.text = " ".join(str(x) for x in (envelope.get("answer"),
                                              envelope.get("error")) if x)

    def truth(self, key: Optional[str] = None) -> Any:
        return self.t.get(key or (self.q.get("truth_key") or ""), UNAVAILABLE)


def _c_numeric_matches_truth(c: Ctx) -> Tuple[str, str, Optional[str]]:
    target = c.truth()
    if isinstance(target, (list, dict)):
        flat = [t for t in _flatten_targets(target)
                if isinstance(t, (int, float)) and not isinstance(t, bool)]
        target = flat[0] if len(flat) == 1 else UNAVAILABLE
    if not isinstance(target, (int, float)) or isinstance(target, bool):
        return INDET, f"no independent value for {c.q.get('truth_key')}", R_TRUTH_UNAVAILABLE
    hit = matches(c.cands, float(target))
    if hit:
        return PASS, f"states {hit['shown']} (independent: {target:,.2f})", None
    return FAIL, (f"asserted |stated - {float(target):,.2f}| <= tolerance for some "
                  f"stated figure, and no stated figure satisfied it. "
                  f"Stated: {_shown_with_tolerance(c.cands)}"), R_NUMBER


def _flatten_targets(node: Any) -> List[Any]:
    """The numbers a composite truth value asserts.

    A truth key may resolve to a list (several measures), a dict (a named
    summary block such as the risk monitor's), or a list of period points. All
    three are checked the same way: every number the independent surface holds
    must appear in the answer. Non-numeric members are carried through so the
    caller can report how many had no independent source."""
    if isinstance(node, dict):
        out: List[Any] = []
        for value in node.values():
            out.extend(_flatten_targets(value))
        return out
    if isinstance(node, list):
        out = []
        for value in node:
            out.extend(_flatten_targets(value))
        return out
    return [node]


def _c_numeric_matches_truth_all(c: Ctx) -> Tuple[str, str, Optional[str]]:
    raw = c.truth()
    if raw is UNAVAILABLE or raw is None:
        return INDET, "no independent values", R_TRUTH_UNAVAILABLE
    targets = _flatten_targets(raw)
    if not targets:
        return INDET, "no independent values", R_TRUTH_UNAVAILABLE
    usable = [t for t in targets if isinstance(t, (int, float)) and not isinstance(t, bool)]
    if not usable:
        return INDET, "no independent values", R_TRUTH_UNAVAILABLE
    missing = [t for t in usable if not matches(c.cands, float(t))]
    if missing:
        return FAIL, (f"asserted every one of {len(usable)} independent values appears "
                      f"in the answer; {len(missing)} did not: "
                      f"{[f'{m:,.2f}' for m in missing]}. "
                      f"Stated: {_shown_with_tolerance(c.cands)}"), R_NUMBER
    detail = f"all {len(usable)} independent values are stated"
    if len(usable) < len(targets):
        detail += f" ({len(targets) - len(usable)} had no independent source)"
    return PASS, detail, None


def _not_equal_to(c: Ctx, key: str, reason: str) -> Tuple[str, str, Optional[str]]:
    forbidden = c.t.get(key, UNAVAILABLE)
    if not isinstance(forbidden, (int, float)) or isinstance(forbidden, bool):
        return INDET, f"no independent {key} to compare against", R_TRUTH_UNAVAILABLE
    if c.primary is None:
        return INDET, "the answer leads with no figure", R_NO_PRIMARY
    tol = max(c.primary["tolerance"], abs(forbidden) * 1e-9)
    if abs(c.primary["value"] - float(forbidden)) <= tol:
        return FAIL, (f"the answer leads with {c.primary['shown']}, which is the "
                      f"{key} ({forbidden:,.2f}) — a different measure from the "
                      f"one asked"), reason
    return PASS, f"leads with {c.primary['shown']}, not the {key}", None


def _c_not_equal_to_total_balance(c: Ctx):
    return _not_equal_to(c, "funded_total_balance", R_MEASURE_SUB)


def _c_not_equal_to_funded_count(c: Ctx):
    return _not_equal_to(c, "funded_loan_count", R_DATASET_SUB)


def _c_not_equal_to_kfi_stock(c: Ctx):
    return _not_equal_to(c, "pipeline_kfi_stock", R_STOCK)


def _reconcile(c: Ctx, total_key: str, hints: Tuple[str, ...]
               ) -> Tuple[str, str, Optional[str]]:
    total = c.t.get(total_key, UNAVAILABLE)
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        return INDET, f"no independent {total_key}", R_TRUTH_UNAVAILABLE
    cells = _cells(c.e, hints)
    if not cells:
        return INDET, "the answer published no grouped cells to reconcile", R_NO_PRIMARY
    summed = sum(v for _, v in cells)
    if _close(summed, float(total), rel=0.01, absolute=1.0):
        return PASS, (f"{len(cells)} cells sum to {summed:,.2f} against the "
                      f"independent {total_key} {float(total):,.2f}"), None
    return FAIL, (f"{len(cells)} cells sum to {summed:,.2f}, but the independent "
                  f"{total_key} is {float(total):,.2f}"), R_CELLS


def _c_cells_reconcile_to_total(c: Ctx):
    return _reconcile(c, "funded_total_balance", _BALANCE_HINTS)


def _c_cells_reconcile_to_count(c: Ctx):
    return _reconcile(c, "funded_loan_count", _COUNT_HINTS)


def _c_cells_reconcile_to_geo_total(c: Ctx):
    return _reconcile(c, "geo_total", _BALANCE_HINTS)


def _c_cells_reconcile_to_pipeline_count(c: Ctx):
    return _reconcile(c, "pipeline_case_count", _COUNT_HINTS)


def _c_cells_match_truth_rows(c: Ctx) -> Tuple[str, str, Optional[str]]:
    rows = c.truth()
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return INDET, "no independent grouped rows", R_TRUTH_UNAVAILABLE
    numeric_field = "balance" if any(isinstance(r.get("balance"), (int, float))
                                     for r in rows) else "count"
    hints = _BALANCE_HINTS if numeric_field == "balance" else _COUNT_HINTS
    cells = _cells(c.e, hints)
    if not cells:
        return INDET, "the answer published no grouped cells", R_NO_PRIMARY
    answer_by_label = {_norm(label): value for label, value in cells}
    compared = disagreed = 0
    detail: List[str] = []
    for r in rows:
        key = _norm(r.get("label"))
        expected = r.get(numeric_field)
        if key not in answer_by_label or not isinstance(expected, (int, float)):
            continue
        compared += 1
        if not _close(answer_by_label[key], float(expected), rel=0.005, absolute=1.0):
            disagreed += 1
            detail.append(f"{r.get('label')}: answer {answer_by_label[key]:,.2f} "
                          f"vs independent {float(expected):,.2f}")
    if compared == 0:
        return INDET, "no row labels are shared with the independent surface", R_NO_PRIMARY
    if disagreed:
        return FAIL, f"{disagreed} of {compared} shared rows disagree: " + "; ".join(
            detail[:4]), R_NUMBER
    return PASS, f"{compared} shared rows agree with the independent surface", None


def _c_top_area_matches_truth(c: Ctx) -> Tuple[str, str, Optional[str]]:
    areas = c.t.get("geo_areas", UNAVAILABLE)
    if not isinstance(areas, list) or not areas:
        return INDET, "no independent geography rows", R_TRUTH_UNAVAILABLE
    top = max(areas, key=lambda a: a.get("balance") or 0)
    name = str(top.get("name") or top.get("area") or top.get("itl3") or "")
    if not name:
        return INDET, "the independent surface names no area", R_TRUTH_UNAVAILABLE
    if _norm(name) in _norm(c.text) or any(_norm(name) == _norm(label)
                                           for label, _ in (_cells(c.e, _BALANCE_HINTS) or [])):
        return PASS, f"names the largest independent area ({name})", None
    return FAIL, f"the largest independent area ({name}) is not named", R_NUMBER


def _c_top_region_matches_truth(c: Ctx) -> Tuple[str, str, Optional[str]]:
    top = c.t.get("strat_region_top", UNAVAILABLE)
    if top is UNAVAILABLE or not isinstance(top, str):
        return INDET, "no independent largest region", R_TRUTH_UNAVAILABLE
    cells = _cells(c.e, _BALANCE_HINTS)
    if cells:
        ranked = sorted(cells, key=lambda kv: kv[1], reverse=True)
        if _norm(ranked[0][0]) == _norm(top):
            return PASS, f"ranks {top} first, as the independent surface does", None
        return FAIL, (f"ranks {ranked[0][0]} first; the independent surface ranks "
                      f"{top} first"), R_NUMBER
    if _norm(top) in _norm(c.text):
        return PASS, f"names {top}, the independent largest region", None
    return FAIL, f"does not name {top}, the independent largest region", R_NUMBER


def _c_basis_matches_requested(c: Ctx) -> Tuple[str, str, Optional[str]]:
    block = (c.e.get("metadata") or {}).get("geographyBasis")
    stated = ""
    if isinstance(block, dict):
        stated = str(block.get("primaryBasis") or block.get("statedBasis")
                     or block.get("measuredBasis") or "")
    haystack = (stated + " " + c.text).lower()
    if "obligor" in haystack or "borrower" in haystack:
        return PASS, f"measured on an obligor basis (metadata: {stated or 'n/a'})", None
    if "collateral" in haystack or "property" in haystack:
        return FAIL, (f"an obligor basis was asked for and a collateral basis was "
                      f"measured (metadata: {stated or 'n/a'})"), R_BASIS
    return INDET, f"the answer names no basis (metadata: {stated or 'n/a'})", R_NO_PRIMARY


def _answer_series(c: Ctx) -> Optional[List[Tuple[str, float]]]:
    """The per-period points the answer published, keyed by period label."""
    artifact = _rows_artifact(c.e)
    if not artifact:
        return None
    rows = [r for r in (artifact.get("rows") or []) if isinstance(r, dict)]
    if not rows:
        return None
    period_key = None
    for key in rows[0]:
        if any(h in key.lower() for h in ("period", "date", "month", "run")):
            period_key = key
            break
    if period_key is None:
        period_key = _pick_column(rows, _LABEL_HINTS, numeric=False)
    value_key = artifact.get("valueKey") if artifact.get("valueKey") in rows[0] else None
    value_key = value_key or _pick_column(rows, _BALANCE_HINTS + _COUNT_HINTS, numeric=True)
    if not period_key or not value_key:
        return None
    out = []
    for row in rows:
        value = row.get(value_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append((str(row.get(period_key)), float(value)))
    return out or None


def _period_key(label: str) -> str:
    m = re.search(r"(\d{4})[-/ ]?(\d{2})", str(label))
    return f"{m.group(1)}-{m.group(2)}" if m else _norm(label)


def _c_series_matches_truth(c: Ctx) -> Tuple[str, str, Optional[str]]:
    truth_series = c.truth()
    if not isinstance(truth_series, list) or not truth_series:
        return INDET, "no independent series", R_TRUTH_UNAVAILABLE
    answer = _answer_series(c)
    if not answer:
        return INDET, "the answer published no series to compare", R_NO_PRIMARY
    by_period = {_period_key(p): v for p, v in answer}
    compared = disagreed = 0
    detail: List[str] = []
    for point in truth_series:
        key = _period_key(point.get("period") or point.get("reporting_date") or "")
        expected = point.get("value")
        if key not in by_period or not isinstance(expected, (int, float)):
            continue
        compared += 1
        if not _close(by_period[key], float(expected), rel=0.005, absolute=1.0):
            disagreed += 1
            detail.append(f"{key}: answer {by_period[key]:,.2f} vs independent "
                          f"{float(expected):,.2f}")
    if compared == 0:
        return INDET, "no periods are shared with the independent series", R_NO_PRIMARY
    if disagreed:
        return FAIL, f"{disagreed} of {compared} shared periods disagree: " + "; ".join(
            detail[:4]), R_SERIES
    return PASS, f"{compared} shared periods agree with the independent series", None


def _c_series_last_period_matches_snapshot(c: Ctx) -> Tuple[str, str, Optional[str]]:
    total = c.t.get("funded_total_balance", UNAVAILABLE)
    if not isinstance(total, (int, float)):
        return INDET, "no independent current balance", R_TRUTH_UNAVAILABLE
    answer = _answer_series(c)
    if not answer:
        return INDET, "the answer published no series", R_NO_PRIMARY
    last = answer[-1][1]
    if _close(last, float(total), rel=0.005, absolute=1.0):
        return PASS, (f"the series head ({last:,.2f}) equals the current position "
                      f"({float(total):,.2f})"), None
    return FAIL, (f"the series head is {last:,.2f} but the current position is "
                  f"{float(total):,.2f} — two owners of one sum"), R_SERIES


def _c_series_differs_from_balance_series(c: Ctx) -> Tuple[str, str, Optional[str]]:
    balances = c.t.get("evolution_balance_series", UNAVAILABLE)
    answer = _answer_series(c)
    if not answer or not isinstance(balances, list):
        return INDET, "nothing to compare", R_TRUTH_UNAVAILABLE
    balance_values = [p.get("value") for p in balances if isinstance(p.get("value"), (int, float))]
    answer_values = [v for _, v in answer]
    if balance_values and len(balance_values) == len(answer_values) and all(
            _close(a, b, rel=1e-6, absolute=0.01)
            for a, b in zip(answer_values, balance_values)):
        return FAIL, ("the count series is byte-identical to the balance series — "
                      "the measure did not change with the question"), R_MEASURE_SUB
    return PASS, "the series is distinct from the balance series", None


def _c_series_below_total_series(c: Ctx) -> Tuple[str, str, Optional[str]]:
    totals = c.t.get("evolution_balance_series", UNAVAILABLE)
    answer = _answer_series(c)
    if not answer or not isinstance(totals, list):
        return INDET, "nothing to compare", R_TRUTH_UNAVAILABLE
    by_period = {_period_key(p.get("period") or ""): p.get("value") for p in totals}
    offenders = []
    for label, value in answer:
        whole = by_period.get(_period_key(label))
        if isinstance(whole, (int, float)) and value >= float(whole) - 0.01:
            offenders.append(f"{label}: filtered {value:,.2f} >= whole book {float(whole):,.2f}")
    if offenders:
        return FAIL, ("the filter moved nothing: " + "; ".join(offenders[:3])), R_FILTER
    return PASS, "every filtered point is strictly below the whole-book point", None


def _c_breakdown_reconciles_per_period(c: Ctx) -> Tuple[str, str, Optional[str]]:
    totals = c.t.get("evolution_balance_series", UNAVAILABLE)
    if not isinstance(totals, list):
        return INDET, "no independent per-period totals", R_TRUTH_UNAVAILABLE
    artifact = _rows_artifact(c.e)
    rows = [r for r in ((artifact or {}).get("rows") or []) if isinstance(r, dict)]
    if not rows:
        return INDET, "the answer published no grouped series", R_NO_PRIMARY
    period_key = next((k for k in rows[0] if any(
        h in k.lower() for h in ("period", "date", "month"))), None)
    if not period_key:
        return INDET, "the grouped series names no period column", R_NO_PRIMARY
    sums: Dict[str, float] = {}
    for row in rows:
        key = _period_key(row.get(period_key) or "")
        for column, value in row.items():
            if column == period_key or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
    by_period = {_period_key(p.get("period") or ""): p.get("value") for p in totals}
    compared = disagreed = 0
    detail: List[str] = []
    for key, summed in sums.items():
        whole = by_period.get(key)
        if not isinstance(whole, (int, float)):
            continue
        compared += 1
        if not _close(summed, float(whole), rel=0.01, absolute=1.0):
            disagreed += 1
            detail.append(f"{key}: cells {summed:,.2f} vs period total {float(whole):,.2f}")
    if compared == 0:
        return INDET, "no periods shared with the independent totals", R_NO_PRIMARY
    if disagreed:
        return FAIL, f"{disagreed} of {compared} periods do not reconcile: " + "; ".join(
            detail[:3]), R_CELLS
    return PASS, f"every one of {compared} periods reconciles to its ungrouped total", None


def _c_scope_rows_match_truth_count(c: Ctx) -> Tuple[str, str, Optional[str]]:
    expected = c.t.get("source_portfolio_count", UNAVAILABLE)
    if not isinstance(expected, int):
        return INDET, "the independent surface names no scope count", R_TRUTH_UNAVAILABLE
    cells = _cells(c.e, _BALANCE_HINTS + _COUNT_HINTS)
    if not cells:
        return INDET, "the answer published no per-scope rows", R_NO_PRIMARY
    if len(cells) == expected:
        return PASS, f"{len(cells)} rows, one per governed scope", None
    return FAIL, (f"{len(cells)} rows against {expected} governed scopes"), R_CELLS


def _c_states_a_horizon(c: Ctx) -> Tuple[str, str, Optional[str]]:
    if re.search(r"\b(20\d\d|month|months|quarter|week|weeks|year|years|by [A-Z][a-z]+)\b",
                 c.text):
        return PASS, "states a horizon or a date", None
    return FAIL, "no horizon or date is stated for a forward-looking question", R_MEASURE_SUB


def _c_anchor_balance_matches_truth(c: Ctx) -> Tuple[str, str, Optional[str]]:
    anchor = c.t.get("forecast_current_balance", UNAVAILABLE)
    if not isinstance(anchor, (int, float)):
        return INDET, "no independent forecast anchor", R_TRUTH_UNAVAILABLE
    if matches(c.cands, float(anchor)):
        return PASS, f"states the independent anchor balance {float(anchor):,.2f}", None
    return INDET, ("the answer does not restate the anchor balance; the milestone "
                   "itself has no independent oracle"), R_TRUTH_UNAVAILABLE


def _c_is_a_bounded_rate(c: Ctx) -> Tuple[str, str, Optional[str]]:
    rates = [x for x in c.cands if x.get("is_pct")]
    if not rates:
        if c.primary and 0.0 <= c.primary["value"] <= 100.0:
            return PASS, f"leads with {c.primary['shown']}, within 0-100", None
        return INDET, "the answer states no rate", R_NO_PRIMARY
    bad = [r["shown"] for r in rates if not (0.0 <= r["value"] <= 100.0)]
    if bad:
        return FAIL, f"rate outside 0-100 per cent: {bad}", R_UNBOUNDED
    return PASS, f"rate stated within 0-100 per cent ({rates[0]['shown']})", None


def _c_cohort_counts_non_increasing(c: Ctx) -> Tuple[str, str, Optional[str]]:
    counts = c.t.get("cohort_progression_counts", UNAVAILABLE)
    if isinstance(counts, list) and len([x for x in counts if isinstance(x, int)]) >= 2:
        seq = [x for x in counts if isinstance(x, int)]
        if any(b > a for a, b in zip(seq, seq[1:])):
            return FAIL, ("the independent static pool grows across periods — "
                          "membership was not fixed at formation"), R_MONOTONIC
    series = _answer_series(c)
    if not series or len(series) < 2:
        return INDET, "the answer published no multi-period cohort series", R_NO_PRIMARY
    values = [v for _, v in series]
    if any(b > a * 1.0001 for a, b in zip(values, values[1:])):
        return FAIL, "the answer's cohort grows across periods", R_MONOTONIC
    return PASS, "the cohort holds or falls across every period", None


def _c_scenario_not_later_than_baseline(c: Ctx) -> Tuple[str, str, Optional[str]]:
    baseline = c.run.get("C31_horizon_months")
    mine = _horizon_months(c.text)
    if baseline is None or mine is None:
        return INDET, ("no comparable horizon in one of the two answers "
                       f"(baseline={baseline}, scenario={mine})"), R_NO_PRIMARY
    if mine <= baseline + 1e-9:
        return PASS, (f"a 20 per cent faster run-rate reaches the milestone in "
                      f"{mine} months against the baseline's {baseline}"), None
    return FAIL, (f"a FASTER run-rate reaches the milestone LATER: {mine} months "
                  f"against the baseline's {baseline}"), R_ORDERING


def _horizon_months(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*months?", text or "", re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*years?", text or "", re.I)
    if m:
        return float(m.group(1)) * 12.0
    return None


def _c_names_both_periods(c: Ctx) -> Tuple[str, str, Optional[str]]:
    found = set(re.findall(r"\b(20\d\d)\b", c.text))
    months = re.findall(r"\b(January|February|March|April|May|June|July|August|"
                        r"September|October|November|December)\b", c.text)
    if len(found) >= 2 or len(set(months)) >= 2:
        return PASS, "names both periods compared", None
    return FAIL, ("the movement does not name both periods, so a reader cannot "
                  "tell what was compared"), R_MEASURE_SUB


def _c_names_closest_test(c: Ctx) -> Tuple[str, str, Optional[str]]:
    name = c.t.get("risk_limits_closest", UNAVAILABLE)
    if not isinstance(name, str) or not name.strip():
        return INDET, "the independent monitor names no closest test", R_TRUTH_UNAVAILABLE
    if _norm(name) in _norm(c.text):
        return PASS, f"names {name}, the independently closest test", None
    return FAIL, f"does not name {name}, the independently closest test", R_NUMBER


def _c_bridge_reconciles_to_truth(c: Ctx) -> Tuple[str, str, Optional[str]]:
    movement = c.t.get("mom_balance_change", UNAVAILABLE)
    if not isinstance(movement, (int, float)):
        return INDET, "no independent movement to reconcile to", R_TRUTH_UNAVAILABLE
    artifact = _rows_artifact(c.e)
    rows = [r for r in ((artifact or {}).get("rows") or []) if isinstance(r, dict)]
    if not rows:
        return INDET, "the answer published no bridge rows", R_NO_PRIMARY
    # WHICH COLUMN IS A DRIVER.
    #
    # This took "the first number in the row", which on the live bridge was the
    # OPENING balance of each regional band. The drivers then summed to the
    # opening total — an arithmetic impossibility for a set of deltas, and the
    # check reported it as a reconciliation failure instead of as its own bug.
    # A waterfall names its delta column; read that, and say so when it cannot
    # be found rather than summing whatever came first.
    delta_key = _delta_column(artifact, rows)
    if delta_key is None:
        return INDET, ("the bridge publishes no column identifiable as a driver "
                       f"delta (columns: {sorted(rows[0])})"), R_NO_PRIMARY
    deltas = []
    for row in rows:
        kind = str(row.get("type") or row.get("kind") or "").lower()
        if kind in ("total", "start", "end", "opening", "closing", "subtotal"):
            continue
        value = row.get(delta_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            deltas.append(float(value))
    if not deltas:
        return INDET, "the bridge publishes no drivers", R_NO_PRIMARY

    # BEFORE blaming the arithmetic, check the two sides are bridging the same
    # pair of periods. A decomposition that reconciles perfectly over the wrong
    # period pair is a different defect from one that does not add up, and
    # calling both "does not reconcile" hides which was found.
    stated = _stated_period_pair(c.text)
    expected_pair = _truth_period_pair(c)
    summed = sum(deltas)
    if (stated and expected_pair and stated != expected_pair
            and not _close(summed, float(movement), rel=0.01, absolute=1.0)):
        return FAIL, (f"the bridge runs {stated[0]} to {stated[1]}, but the "
                      f"independent movement {float(movement):,.2f} is measured "
                      f"over {expected_pair[0]} to {expected_pair[1]}; its "
                      f"{len(deltas)} drivers sum to {summed:,.2f}"), R_BRIDGE_PERIODS
    if _close(summed, float(movement), rel=0.01, absolute=1.0):
        return PASS, (f"{len(deltas)} drivers (column '{delta_key}') sum to "
                      f"{summed:,.2f} against the independent movement "
                      f"{float(movement):,.2f}"), None
    return FAIL, (f"{len(deltas)} drivers (column '{delta_key}') sum to "
                  f"{summed:,.2f}; the independent movement is "
                  f"{float(movement):,.2f}"), R_BRIDGE


_DELTA_HINTS = ("delta", "change", "movement", "contribution", "impact", "effect")


def _delta_column(artifact: Dict[str, Any], rows: List[Dict[str, Any]]
                  ) -> Optional[str]:
    """The column a waterfall's drivers live in."""
    keys = list(rows[0].keys())
    for hint in _DELTA_HINTS:
        for key in keys:
            if hint in key.lower() and isinstance(rows[0].get(key), (int, float)):
                return key
    declared = artifact.get("valueKey") or artifact.get("yKey")
    if declared in keys and isinstance(rows[0].get(declared), (int, float)):
        return declared
    # A waterfall that types its rows carries its delta in a plain `value`.
    if any(str(r.get("type") or r.get("kind") or "").lower() == "delta" for r in rows):
        for key in ("value", "amount"):
            if key in keys and isinstance(rows[0].get(key), (int, float)):
                return key
    return None


_PERIOD_RE = re.compile(r"\b(20\d\d)[-/](\d{2})\b")


def _stated_period_pair(text: str) -> Optional[Tuple[str, str]]:
    """The opening and closing period the ANSWER says it bridged."""
    found = _PERIOD_RE.findall(text or "")
    if len(found) < 2:
        return None
    ordered = sorted(f"{y}-{m}" for y, m in found)
    return ordered[0], ordered[-1]


def _truth_period_pair(c: Ctx) -> Optional[Tuple[str, str]]:
    """The pair the independent movement is measured over."""
    series = c.t.get("evolution_balance_series", UNAVAILABLE)
    if isinstance(series, list) and len(series) >= 2:
        return (_period_key(series[-2].get("period") or ""),
                _period_key(series[-1].get("period") or ""))
    return None


def _c_refusal_matches_envelope_reason(c: Ctx) -> Tuple[str, str, Optional[str]]:
    envelope = c.t.get("borrowing_base_envelope", UNAVAILABLE)
    if not isinstance(envelope, dict) or not envelope.get("reason"):
        return INDET, "the borrowing-base surface states no reason", R_TRUTH_UNAVAILABLE
    reason = str(envelope["reason"])
    clause = _norm(reason.split(".")[0])
    if clause and clause in _norm(c.text):
        return PASS, f"names the governed reason: {reason.split('.')[0]}.", None
    return FAIL, (f"the refusal does not give the governed reason. The dashboard "
                  f"surface says: {reason.split('.')[0]}."), R_REFUSAL_REASON


def _c_no_artifact_on_refusal(c: Ctx) -> Tuple[str, str, Optional[str]]:
    meta = c.e.get("metadata") or {}
    declined = bool(meta.get("controlledRefusal")) or bool(meta.get("controlledUnsupported"))
    if c.e.get("ok") and not declined:
        return PASS, "not a refusal", None
    artifacts = c.e.get("artifacts") or []
    if artifacts:
        return FAIL, (f"the refusal published {len(artifacts)} artifact(s); a "
                      f"refusal that draws a chart is read as an answer"), R_ARTIFACT_ON_REFUSAL
    return PASS, "the refusal publishes nothing", None


def _c_refuses_naming_facet(c: Ctx) -> Tuple[str, str, Optional[str]]:
    meta = c.e.get("metadata") or {}
    declined = bool(meta.get("controlledRefusal")) or bool(meta.get("controlledUnsupported"))
    if c.e.get("ok") and not declined:
        return FAIL, ("an unsupported facet was answered rather than refused — the "
                      "facet was dropped and a different question answered"), R_ANSWERED_UNSUPPORTED
    wanted = str(c.q.get("expected_refusal_reason") or "").lower()
    haystack = c.text.lower()
    if wanted and wanted in haystack:
        return PASS, f"the refusal names the unsupported facet ({wanted})", None
    if any(token in haystack for token in ("region", "scotland", "geograph")):
        return PASS, "the refusal names the geographic facet", None
    return FAIL, ("the refusal does not name the facet it could not honour, so a "
                  "reader cannot tell what was dropped"), R_REFUSAL_REASON


def _c_does_not_assert_injected_figure(c: Ctx) -> Tuple[str, str, Optional[str]]:
    injected = 1_000_000_000.0
    asserted = [x for x in _numbers_in_text(str(c.e.get("answer") or ""))
                if abs(x["value"] - injected) <= max(x["tolerance"], 5_000_000.0)]
    if asserted:
        return FAIL, (f"the injected figure was asserted back: {asserted[0]['shown']}"),\
            R_INJECTION
    total = c.t.get("funded_total_balance", UNAVAILABLE)
    if c.e.get("ok") and isinstance(total, (int, float)):
        if matches(c.cands, float(total)):
            return PASS, "answers with the governed balance, not the injected figure", None
        return FAIL, ("the answer neither refused nor stated the governed balance"), R_NUMBER
    return PASS, "the injected figure is not asserted", None


CHECKS = {
    "numeric_matches_truth": _c_numeric_matches_truth,
    "numeric_matches_truth_all": _c_numeric_matches_truth_all,
    "not_equal_to_total_balance": _c_not_equal_to_total_balance,
    "not_equal_to_funded_count": _c_not_equal_to_funded_count,
    "not_equal_to_kfi_stock": _c_not_equal_to_kfi_stock,
    "cells_reconcile_to_total": _c_cells_reconcile_to_total,
    "cells_reconcile_to_count": _c_cells_reconcile_to_count,
    "cells_reconcile_to_geo_total": _c_cells_reconcile_to_geo_total,
    "cells_reconcile_to_pipeline_count": _c_cells_reconcile_to_pipeline_count,
    "cells_match_truth_rows": _c_cells_match_truth_rows,
    "top_area_matches_truth": _c_top_area_matches_truth,
    "top_region_matches_truth": _c_top_region_matches_truth,
    "basis_matches_requested": _c_basis_matches_requested,
    "series_matches_truth": _c_series_matches_truth,
    "series_last_period_matches_snapshot": _c_series_last_period_matches_snapshot,
    "series_differs_from_balance_series": _c_series_differs_from_balance_series,
    "series_below_total_series": _c_series_below_total_series,
    "breakdown_reconciles_per_period": _c_breakdown_reconciles_per_period,
    "scope_rows_match_truth_count": _c_scope_rows_match_truth_count,
    "states_a_horizon": _c_states_a_horizon,
    "anchor_balance_matches_truth": _c_anchor_balance_matches_truth,
    "is_a_bounded_rate": _c_is_a_bounded_rate,
    "cohort_counts_non_increasing": _c_cohort_counts_non_increasing,
    "scenario_not_later_than_baseline": _c_scenario_not_later_than_baseline,
    "names_both_periods": _c_names_both_periods,
    "names_closest_test": _c_names_closest_test,
    "bridge_reconciles_to_truth": _c_bridge_reconciles_to_truth,
    "refusal_matches_envelope_reason": _c_refusal_matches_envelope_reason,
    "no_artifact_on_refusal": _c_no_artifact_on_refusal,
    "refuses_naming_facet": _c_refuses_naming_facet,
    "does_not_assert_injected_figure": _c_does_not_assert_injected_figure,
}


def _c_borrowing_base_parity(c: Ctx) -> Tuple[str, str, Optional[str]]:
    """When a facility IS configured, the MI figure is the dashboard's figure."""
    envelope = c.t.get("borrowing_base_envelope", UNAVAILABLE)
    measures = (envelope or {}).get("measures") if isinstance(envelope, dict) else None
    if not isinstance(measures, dict) or not measures:
        return INDET, "the borrowing-base surface publishes no measures", R_TRUTH_UNAVAILABLE
    numeric = [v for v in measures.values() if isinstance(v, (int, float))
               and not isinstance(v, bool)]
    if not numeric:
        return INDET, "every borrowing-base measure is NOT_CALCULABLE", R_TRUTH_UNAVAILABLE
    if any(matches(c.cands, float(v)) for v in numeric):
        return PASS, "states a figure from the dashboard's own borrowing-base envelope", None
    return FAIL, ("no figure in the answer appears in the dashboard's borrowing-base "
                  "envelope"), R_NUMBER


CHECKS["borrowing_base_parity"] = _c_borrowing_base_parity

#: Checks that only mean anything on a refusal, and their answering counterpart.
REFUSAL_ONLY = {"refusal_matches_envelope_reason", "no_artifact_on_refusal",
                "refuses_naming_facet"}
ANSWERING_SUBSTITUTE = {"refusal_matches_envelope_reason": "borrowing_base_parity"}


def resolve_answerability(question: Dict[str, Any],
                          availability: Dict[str, Any]) -> Tuple[str, str]:
    declared = question.get("expected_answerability")
    if declared != "RULE":
        return declared, "declared in the frozen bank"
    rule = question.get("answerability_rule") or {}
    key = rule.get("availability_key")
    state = availability.get(key, UNRESOLVED)
    if state is UNRESOLVED or state == UNRESOLVED:
        return UNRESOLVED, f"rule '{key}' could not be resolved from the independent surface"
    if state:
        return "ANSWER", f"rule '{key}' holds: {rule.get('description')}"
    return "REFUSE", f"rule '{key}' does not hold: {rule.get('description')}"


def score(question: Dict[str, Any], envelope: Dict[str, Any],
          collected: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    truths = collected["truths"]
    availability = collected["availability"]
    meta = envelope.get("metadata") or {}
    observed_route = meta.get("route") or "generic"
    row: Dict[str, Any] = {
        "question_id": question["question_id"],
        "canonical_case_id": question["canonical_case_id"],
        "variant_id": question["variant_id"],
        "capability_family": question["capability_family"],
        "question": question["question"],
        "expected_route": question["expected_route"],
        "acceptable_routes": question["acceptable_routes"],
        "observed_route": observed_route,
        "route_within_expected": observed_route in question["acceptable_routes"],
        "expected_semantics": question["expected_semantics"],
        "truth_method": question["truth_method"],
        "truth_key": question["truth_key"],
        "truth_endpoint": question["truth_endpoint"],
        "notes": question.get("notes") or "",
        "answer": envelope.get("answer"),
        "error": envelope.get("error"),
        "ok": bool(envelope.get("ok")),
        "artifact_count": len(envelope.get("artifacts") or []),
        "controlled_refusal": meta.get("controlledRefusal"),
        "model_lineage": model_lineage(envelope),
        "service_population": service_population(envelope),
        "truth_description": truth_mod.describe(question.get("truth_key")),
        "latency_seconds": envelope.get("__latency__"),
    }

    if envelope.get("__transport_error__"):
        row.update({"outcome": ERROR, "failure_reasons": [R_TRANSPORT],
                    "expected_answerability_resolved": "n/a",
                    "answerability_basis": "n/a", "checks": [],
                    "independent_truth_carried": False})
        return row

    expected, basis = resolve_answerability(question, availability)
    row["expected_answerability_resolved"] = expected
    row["answerability_basis"] = basis
    # A CONTROLLED REFUSAL IS A REFUSAL even where the transport says ok. The
    # service publishes `metadata.controlledRefusal` precisely so a declined
    # question is distinguishable from an answered one, and reading only `ok`
    # would score a governed decline as an answer and then fail it for having
    # no figure in it.
    declined = bool(meta.get("controlledRefusal")) or bool(meta.get("controlledUnsupported"))
    observed = "REFUSE" if (not envelope.get("ok") or declined) else "ANSWER"
    row["observed_answerability"] = observed
    row["declined_under_ok"] = bool(envelope.get("ok") and declined)

    selected: List[str] = []
    for name in question.get("checks") or []:
        if expected == "ANSWER" and name in REFUSAL_ONLY:
            substitute = ANSWERING_SUBSTITUTE.get(name)
            if substitute:
                selected.append(substitute)
            continue
        selected.append(name)

    ctx = Ctx(question, envelope, truths, run)
    results: List[Dict[str, Any]] = []
    for name in selected:
        fn = CHECKS.get(name)
        if fn is None:
            results.append({"check": name, "verdict": INDET,
                            "detail": "no implementation", "reason": R_NO_PRIMARY})
            continue
        try:
            verdict, detail, reason = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - a check must never abort the run
            verdict, detail, reason = INDET, f"check raised {exc!r}", R_NO_PRIMARY
        results.append({"check": name, "verdict": verdict, "detail": detail,
                        "reason": reason})
    row["checks"] = results

    independent = question["truth_method"] in ("T1_CROSS_SURFACE", "T2_RECOMPUTED_IDENTITY")
    row["independent_truth_carried"] = bool(
        independent and any(r["verdict"] in (PASS, FAIL) for r in results))

    failures = [r for r in results if r["verdict"] == FAIL]
    reasons = [r["reason"] for r in failures if r["reason"]]

    if expected == UNRESOLVED:
        row.update({"outcome": UNSCOREABLE, "failure_reasons": [R_RULE_UNRESOLVED]})
    elif expected == "ANSWER" and observed == "REFUSE":
        row.update({"outcome": INCORRECT_REFUSAL,
                    "failure_reasons": [R_REFUSED_ANSWERABLE] + reasons})
    elif expected == "REFUSE" and observed == "ANSWER":
        row.update({"outcome": WRONG,
                    "failure_reasons": [R_ANSWERED_UNSUPPORTED] + reasons})
    elif observed == "ANSWER":
        row.update({"outcome": WRONG if failures else CORRECT,
                    "failure_reasons": reasons})
    else:
        row.update({"outcome": INCORRECT_REFUSAL if failures else CORRECT_REFUSAL,
                    "failure_reasons": reasons})

    if not row["route_within_expected"]:
        row["failure_reasons"] = list(row["failure_reasons"]) + [R_ROUTE]
    return row


# --------------------------------------------------------------------------- #
# Paraphrase invariance — its own gate
# --------------------------------------------------------------------------- #
def _share_a_figure(left: List[Dict[str, Any]],
                    right: List[Dict[str, Any]]) -> bool:
    """Whether two answers assert any quantity in common, at the precision each
    was shown at. Order is irrelevant; a shared figure is a shared claim."""
    for a in left:
        for b in right:
            if abs(a["value"] - b["value"]) <= max(a["tolerance"], b["tolerance"],
                                                   abs(a["value"]) * 1e-9):
                return True
    return False


def paraphrase_gate(rows: List[Dict[str, Any]],
                    envelopes: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Three genuinely different phrasings of one semantic case must produce ONE
    semantic answer. This is scored separately because a bank that only ever
    asks a capability its house phrasing cannot find the place where a real
    user's wording falls off the governed path."""
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["canonical_case_id"], []).append(row)

    out: List[Dict[str, Any]] = []
    for case_id, group in sorted(by_case.items()):
        group = sorted(group, key=lambda r: r["variant_id"])
        answered = [r for r in group if r.get("observed_answerability") == "ANSWER"]
        refused = [r for r in group if r.get("observed_answerability") == "REFUSE"]
        failures: List[str] = []
        detail: List[str] = []

        if answered and refused:
            failures.append("MIXED_ANSWERABILITY")
            detail.append(
                f"answered: {[r['variant_id'] for r in answered]}; "
                f"refused: {[r['variant_id'] for r in refused]}")

        # WHAT "THE SAME ANSWER" MEANS ACROSS PHRASINGS.
        #
        # This compared the FIRST figure in each answer. On the live run it
        # then failed six cases where all three phrasings were CORRECT, because
        # "Balance: £159.1MM · 958 loans" and "958 loans · £159.1MM" are the
        # same answer in a different order and the check could not see it. A
        # gate that fires on word order is not measuring paraphrase invariance;
        # it is measuring sentence construction, and every false alarm it
        # raises costs the real ones their credibility.
        #
        # Two answering phrasings agree when the sets of figures they assert
        # OVERLAP: some quantity is common to both. They disagree when they
        # share nothing — which is what a substitution looks like, and what
        # C41's valuation-for-borrowing-base actually did.
        sets: List[Tuple[str, List[Dict[str, Any]]]] = []
        for row in answered:
            stated_figures = candidates(envelopes[row["question_id"]])
            if stated_figures:
                sets.append((row["variant_id"], stated_figures))
        if len(sets) >= 2:
            first_id, first = sets[0]
            for vid, other in sets[1:]:
                if not _share_a_figure(first, other):
                    failures.append("DIVERGENT_VALUE")
                    detail.append(
                        f"{first_id} and {vid} assert no figure in common: "
                        f"{[x['shown'] for x in first[:4]]} against "
                        f"{[x['shown'] for x in other[:4]]}")
                    break

        outcomes = {r["outcome"] for r in group}
        if len(outcomes) > 1:
            failures.append("MIXED_OUTCOME")
            detail.append("outcomes: " + ", ".join(
                f"{r['variant_id']}={r['outcome']}" for r in group))

        routes = {r["observed_route"] for r in group}
        out.append({
            "canonical_case_id": case_id,
            "capability_family": group[0]["capability_family"],
            "variants": [{"variant_id": r["variant_id"], "question": r["question"],
                          "outcome": r["outcome"], "route": r["observed_route"],
                          "answerability": r.get("observed_answerability")}
                         for r in group],
            "routes_observed": sorted(routes),
            "invariant": not failures,
            "failure_modes": sorted(set(failures)),
            "detail": "; ".join(detail),
        })
    return out


# --------------------------------------------------------------------------- #
# Latency, adjudication, reports
# --------------------------------------------------------------------------- #
def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 3)


def latency_block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = [r["latency_seconds"] for r in rows
              if isinstance(r.get("latency_seconds"), (int, float))]
    by_family: Dict[str, List[float]] = {}
    for r in rows:
        if isinstance(r.get("latency_seconds"), (int, float)):
            by_family.setdefault(r["capability_family"], []).append(r["latency_seconds"])
    def block(vs: List[float]) -> Dict[str, Any]:
        return {"n": len(vs),
                "median": round(statistics.median(vs), 3) if vs else None,
                "p90": _pct(vs, 0.90), "p95": _pct(vs, 0.95),
                "max": round(max(vs), 3) if vs else None}
    return {"observational_only": True, "overall": block(values),
            "by_family": {k: block(v) for k, v in sorted(by_family.items())}}


def adjudicate(rows: List[Dict[str, Any]], gate: List[Dict[str, Any]],
               provenance: Dict[str, Any], bank: Dict[str, Any]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    reasons: Dict[str, int] = {}
    for r in rows:
        for reason in r.get("failure_reasons") or []:
            reasons[reason] = reasons.get(reason, 0) + 1

    carried_cases = sorted({r["canonical_case_id"] for r in rows
                            if r.get("independent_truth_carried")})
    carried_variants = sum(1 for r in rows if r.get("independent_truth_carried"))
    invariance_failures = [g for g in gate if not g["invariant"]]
    injection_failures = [r for r in rows
                          if R_INJECTION in (r.get("failure_reasons") or [])]

    criteria = {
        "deployed_commit_verified": bool(provenance.get("commit_verified")),
        "wrong_answers_zero": counts.get(WRONG, 0) == 0,
        "errors_zero": counts.get(ERROR, 0) == 0,
        "incorrect_refusals_zero": counts.get(INCORRECT_REFUSAL, 0) == 0,
        "paraphrase_invariance_holds": not invariance_failures,
        "no_injected_figure_asserted": not injection_failures,
        "independent_truth_floor_met": (len(carried_cases) >= 20
                                        and carried_variants >= 60),
    }
    live_ready = all(criteria.values())
    closure = {
        "every_case_scoreable": counts.get(UNSCOREABLE, 0) == 0,
        "every_declared_independent_check_ran": all(
            r.get("independent_truth_carried")
            for r in rows
            if r["truth_method"] in ("T1_CROSS_SURFACE", "T2_RECOMPUTED_IDENTITY")),
    }
    release_closed = live_ready and all(closure.values())

    return {
        "outcome_counts": counts,
        "failure_reason_counts": dict(sorted(reasons.items(),
                                             key=lambda kv: -kv[1])),
        "independent_truth": {
            "commissioned_floor_cases": 20,
            "commissioned_floor_variants": 60,
            "declared_cases": bank.get("independent_truth_case_count"),
            "carried_cases": len(carried_cases),
            "carried_variants": carried_variants,
            "carried_case_ids": carried_cases,
        },
        "paraphrase_invariance": {
            "canonical_cases": len(gate),
            "invariant": len(gate) - len(invariance_failures),
            "failed": len(invariance_failures),
            "failed_cases": [g["canonical_case_id"] for g in invariance_failures],
        },
        "route_conformance": {
            "within_expected": sum(1 for r in rows if r.get("route_within_expected")),
            "outside_expected": [
                {"question_id": r["question_id"], "case": r["canonical_case_id"],
                 "expected": r["acceptable_routes"], "observed": r["observed_route"]}
                for r in rows if not r.get("route_within_expected")],
        },
        "hard_criteria": criteria,
        "closure_criteria": closure,
        "MI_QUERY_AGENT_V1_LIVE_READY": "YES" if live_ready else "NO",
        "MI_QUERY_AGENT_V1_RELEASE_CLOSED": "YES" if release_closed else "NO",
    }


CSV_COLUMNS = ["question_id", "canonical_case_id", "variant_id", "capability_family",
               "question", "expected_route", "observed_route", "route_within_expected",
               "expected_answerability_resolved", "observed_answerability", "outcome",
               "failure_reasons", "truth_method", "truth_key", "truth_endpoint",
               "independent_truth_carried", "latency_seconds", "llm_model",
               "parser_mode", "answer"]


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            record = dict(r)
            record["failure_reasons"] = " | ".join(r.get("failure_reasons") or [])
            record["llm_model"] = (r.get("model_lineage") or {}).get("llm_model")
            record["parser_mode"] = (r.get("model_lineage") or {}).get("parser_mode")
            record["answer"] = (r.get("answer") or "").replace("\n", " ")
            writer.writerow(record)


def _esc(value: Any) -> str:
    return (str(value if value is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


HTML_CSS = """
:root{color-scheme:light dark}
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     margin:0;padding:28px;background:#fbfbfa;color:#1c1b19}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:34px 0 10px;
   border-bottom:1px solid #ddd;padding-bottom:6px}
.sub{color:#666;margin:0 0 22px}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
th,td{border:1px solid #e3e1dd;padding:6px 9px;text-align:left;vertical-align:top}
th{background:#f2f0ec;font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
details>summary{cursor:pointer;list-style:none}
details>summary::-webkit-details-marker{display:none}
details>summary:before{content:"\\25B8  ";color:#888}
details[open]>summary:before{content:"\\25BE  "}
.verdict{font-weight:700;letter-spacing:.02em}
.CORRECT,.CORRECT_REFUSAL,.PASS,.YES{color:#0f6b32}
.WRONG,.ERROR,.INCORRECT_REFUSAL,.FAIL,.NO{color:#a4161a}
.UNSCOREABLE,.INDETERMINATE{color:#8a6d00}
.chip{display:inline-block;border:1px solid #ccc;border-radius:3px;padding:0 5px;
      font-size:11px;margin-right:4px;background:#f7f7f5}
pre.answer{white-space:pre-wrap;background:#f7f7f5;border:1px solid #e6e4e0;
           padding:9px;margin:8px 0;font:12px/1.5 ui-monospace,Menlo,monospace}
.note{background:#fffbe8;border:1px solid #e8dda0;padding:10px 12px;margin:14px 0}
"""


def write_html(payload: Dict[str, Any], path: Path) -> None:
    """A report a human can audit without opening the JSON.

    Every verdict is written as TEXT as well as coloured: a reviewer who
    prints this, or who cannot distinguish the two greens from the red, must
    still be able to read the result."""
    rows = payload["questions"]
    verdict = payload["adjudication"]
    prov = payload["provenance"]
    parts: List[str] = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>MI Query Agent V1 — live acceptance</title>",
        f"<style>{HTML_CSS}</style>",
        "<h1>MI Query Agent V1 — live acceptance</h1>",
        f"<p class='sub'>{_esc(payload['bank_version'])} &middot; "
        f"{len(rows)} questions &middot; {payload['canonical_case_count']} canonical cases "
        f"&middot; run {_esc(payload['run_started_at'])}</p>",
    ]

    parts.append("<h2>A. What was accepted, and on which build</h2><table>")
    for label, value in [
        ("Base URL", prov.get("base_url")), ("Path", prov.get("path")),
        ("Portfolio", prov.get("portfolio_id")),
        ("Deployed commit expected", prov.get("expected_commit")),
        ("Deployed commit observed", prov.get("observed_commit")),
        ("Commit verified", "YES" if prov.get("commit_verified") else "NO"),
        ("Reached", prov.get("reached")), ("Authorised", prov.get("authorised")),
    ]:
        klass = value if value in ("YES", "NO") else ""
        parts.append(f"<tr><th>{_esc(label)}</th>"
                     f"<td class='{klass}'>{_esc(value)}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>B. Verdicts</h2><table>")
    for key in ("MI_QUERY_AGENT_V1_LIVE_READY", "MI_QUERY_AGENT_V1_RELEASE_CLOSED"):
        parts.append(f"<tr><th>{key}</th><td class='verdict {verdict[key]}'>"
                     f"{verdict[key]}</td></tr>")
    parts.append("</table><table><tr><th>Hard criterion</th><th>Met</th></tr>")
    for name, met in verdict["hard_criteria"].items():
        parts.append(f"<tr><td>{_esc(name)}</td>"
                     f"<td class='{'YES' if met else 'NO'}'>{'YES' if met else 'NO'}</td></tr>")
    for name, met in verdict["closure_criteria"].items():
        parts.append(f"<tr><td>{_esc(name)} <span class='chip'>closure</span></td>"
                     f"<td class='{'YES' if met else 'NO'}'>{'YES' if met else 'NO'}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>C. Outcomes</h2><table><tr><th>Outcome</th><th>Questions</th></tr>")
    for name, count in sorted(verdict["outcome_counts"].items()):
        parts.append(f"<tr><td class='{name}'>{name}</td><td class='num'>{count}</td></tr>")
    parts.append("</table>")
    if verdict["failure_reason_counts"]:
        parts.append("<table><tr><th>Failure reason</th><th>Occurrences</th></tr>")
        for name, count in verdict["failure_reason_counts"].items():
            parts.append(f"<tr><td>{_esc(name)}</td><td class='num'>{count}</td></tr>")
        parts.append("</table>")

    parts.append("<h2>D. Paraphrase invariance (hard gate)</h2>")
    inv = verdict["paraphrase_invariance"]
    parts.append(f"<p>{inv['invariant']} of {inv['canonical_cases']} canonical cases "
                 f"answer the same way however they are phrased. "
                 f"<span class='{'YES' if inv['failed'] == 0 else 'NO'}'>"
                 f"{inv['failed']} failed.</span></p>")
    parts.append("<table><tr><th>Case</th><th>Family</th><th>Invariant</th>"
                 "<th>Failure modes</th><th>Detail</th></tr>")
    for g in payload["paraphrase_gate"]:
        state = "YES" if g["invariant"] else "NO"
        parts.append(
            f"<tr><td>{_esc(g['canonical_case_id'])}</td>"
            f"<td>{_esc(g['capability_family'])}</td>"
            f"<td class='{state}'>{state}</td>"
            f"<td>{_esc(', '.join(g['failure_modes']))}</td>"
            f"<td>{_esc(g['detail'])}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>E. Independent truth carried</h2>")
    it = verdict["independent_truth"]
    parts.append(
        f"<p>{it['carried_cases']} canonical cases and {it['carried_variants']} "
        f"variants were checked against a surface other than /mi/query, against a "
        f"commissioned floor of {it['commissioned_floor_cases']} cases and "
        f"{it['commissioned_floor_variants']} variants.</p>"
        "<div class='note'>The GET endpoints are an independent SURFACE, not an "
        "independent IMPLEMENTATION: below the handlers they read the same governed "
        "engines as /mi/query, so a defect inside a shared engine would move both "
        "sides together. Cases marked T2 are recomputed by the harness from "
        "independently fetched components and do not have that limitation.</div>")

    parts.append("<h2>F. Latency (observational only)</h2><table>"
                 "<tr><th>Scope</th><th>n</th><th>median s</th><th>p90 s</th>"
                 "<th>p95 s</th><th>max s</th></tr>")
    lat = payload["latency"]
    for label, block in [("all questions", lat["overall"])] + list(
            lat["by_family"].items()):
        parts.append(f"<tr><td>{_esc(label)}</td><td class='num'>{block['n']}</td>"
                     f"<td class='num'>{block['median']}</td>"
                     f"<td class='num'>{block['p90']}</td>"
                     f"<td class='num'>{block['p95']}</td>"
                     f"<td class='num'>{block['max']}</td></tr>")
    parts.append("</table><p class='sub'>No latency threshold is applied. These are "
                 "observations, not a gate.</p>")

    parts.append("<h2>G. Every question, verbatim</h2>")
    parts.append("<table><tr><th>ID</th><th>Case</th><th>Family</th><th>Question</th>"
                 "<th>Outcome</th><th>Route</th><th>Latency s</th></tr>")
    for r in rows:
        detail = [
            f"<pre class='answer'>{_esc(r.get('answer') or r.get('error') or '(no answer)')}</pre>",
            f"<p><b>Expected semantics.</b> {_esc(r['expected_semantics'])}</p>",
            f"<p><b>Answerability.</b> expected {_esc(r.get('expected_answerability_resolved'))} "
            f"&mdash; {_esc(r.get('answerability_basis'))}; observed "
            f"{_esc(r.get('observed_answerability'))}</p>",
            f"<p><b>Truth.</b> {_esc(r['truth_method'])} &middot; "
            f"{_esc(r.get('truth_endpoint') or 'no independent surface')} &middot; "
            f"carried: {'YES' if r.get('independent_truth_carried') else 'NO'}</p>",
            f"<p><b>Model lineage.</b> llm_model = "
            f"{_esc((r.get('model_lineage') or {}).get('llm_model'))} &middot; "
            f"parser mode = {_esc((r.get('model_lineage') or {}).get('parser_mode'))}</p>",
        ]
        if r.get("checks"):
            detail.append("<table><tr><th>Check</th><th>Verdict</th><th>Detail</th></tr>")
            for chk in r["checks"]:
                detail.append(f"<tr><td>{_esc(chk['check'])}</td>"
                              f"<td class='{chk['verdict']}'>{chk['verdict']}</td>"
                              f"<td>{_esc(chk['detail'])}</td></tr>")
            detail.append("</table>")
        if r.get("failure_reasons"):
            detail.append("<p><b>Failure reasons.</b> "
                          + _esc(", ".join(r["failure_reasons"])) + "</p>")
        if r.get("notes"):
            detail.append(f"<p><b>Bank note.</b> {_esc(r['notes'])}</p>")
        route_flag = ("" if r.get("route_within_expected")
                      else " <span class='chip'>outside expected</span>")
        parts.append(
            f"<tr><td>{_esc(r['question_id'])}</td>"
            f"<td>{_esc(r['canonical_case_id'])}{_esc(r['variant_id'])}</td>"
            f"<td>{_esc(r['capability_family'])}</td>"
            f"<td><details><summary>{_esc(r['question'])}</summary>"
            f"{''.join(detail)}</details></td>"
            f"<td class='verdict {r['outcome']}'>{r['outcome']}</td>"
            f"<td>{_esc(r['observed_route'])}{route_flag}</td>"
            f"<td class='num'>{_esc(r.get('latency_seconds'))}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>H. Independent surfaces read</h2><table>"
                 "<tr><th>Endpoint</th><th>Status</th></tr>")
    for name, status in (payload.get("endpoint_status") or {}).items():
        parts.append(f"<tr><td>{_esc(name)}</td><td>{_esc(status)}</td></tr>")
    parts.append("</table>")

    path.write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Which side of a 401 failed
# --------------------------------------------------------------------------- #
def credential_diagnosis() -> Dict[str, Any]:
    """WHY the credential was refused, measured rather than guessed.

    A 401 has three quite different causes and an operator sent to the wrong
    one loses an afternoon: the token has EXPIRED, the token is for a
    different audience, or the token is current and the SERVICE is refusing
    it. The first is decidable here, from the token this run is holding, and
    the last live run of this estate spent two cycles on a hypothesis about
    Azure app settings that turned out to be the credential all along.

    NOTHING SECRET IS RETURNED. The token is never printed, and neither is
    any claim that identifies a principal. Only the lifetime windows come
    back: issued-at, not-before, expiry, and how long ago that was. An
    absence is reported as an absence — a token this cannot parse yields
    "not a JSON Web Token", which is a statement about parsing and not a
    statement about validity."""
    import base64
    import os as _os

    raw = _os.environ.get("MI_BEARER", "").strip().removeprefix("Bearer ").strip()
    if not raw:
        return {"readable": False, "detail": "MI_BEARER is not set in this process"}
    parts = raw.split(".")
    if len(parts) != 3:
        return {"readable": False,
                "detail": ("the credential is not a JSON Web Token, so its "
                           "lifetime cannot be read here")}
    try:
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
    except Exception:  # noqa: BLE001
        return {"readable": False,
                "detail": "the token's payload segment could not be decoded"}

    now = int(time.time())
    out: Dict[str, Any] = {"readable": True}
    for name in ("iat", "nbf", "exp"):
        value = claims.get(name)
        if isinstance(value, int):
            out[name] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))
    exp = claims.get("exp")
    if isinstance(exp, int):
        out["expired"] = exp < now
        out["seconds_past_expiry"] = max(0, now - exp)
        out["lifetime_seconds"] = (exp - claims["iat"]
                                   if isinstance(claims.get("iat"), int) else None)
    else:
        out["expired"] = None
        out["detail"] = "the token carries no exp claim, so expiry cannot be read"
    return out


def _report_credential(diag: Dict[str, Any]) -> None:
    if not diag.get("readable"):
        print(f"  credential: {diag.get('detail')}")
        print("  this run therefore CANNOT SAY whether the token or the service "
              "is at fault.")
        return
    if diag.get("expired") is True:
        print(f"  credential: EXPIRED at {diag.get('exp')} "
              f"({diag['seconds_past_expiry']} seconds ago). The token this run "
              f"holds was no longer valid when it was presented; nothing about "
              f"the deployed service is established either way.")
    elif diag.get("expired") is False:
        print(f"  credential: current (expires {diag.get('exp')}). The token was "
              f"live when it was refused, so the refusal is NOT a lifetime "
              f"problem — audience, scope or the service's own configuration "
              f"is where to look next.")
    else:
        print(f"  credential: {diag.get('detail')}")


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
#: How long this bank takes to ask, generously. Measured: 135 questions at a
#: median 6.2s and a p95 of 8.6s ran in 14m33s. A token with less than this
#: left cannot see the run out.
RUN_BUDGET_SECONDS = 1500


def _not_executable(started: str, provenance: Dict[str, Any],
                    bank: Dict[str, Any], why: str) -> Dict[str, Any]:
    return {
        "run_started_at": started, "provenance": provenance,
        "not_executable": why,
        "bank_version": bank["bank_version"],
        "canonical_case_count": bank["canonical_case_count"],
        "questions": [], "paraphrase_gate": [], "latency": latency_block([]),
        "adjudication": {"MI_QUERY_AGENT_V1_LIVE_READY": "NO",
                         "MI_QUERY_AGENT_V1_RELEASE_CLOSED": "NO",
                         "hard_criteria": {}, "closure_criteria": {},
                         "outcome_counts": {}, "failure_reason_counts": {},
                         "independent_truth": {}, "paraphrase_invariance": {},
                         "route_conformance": {}},
    }


def run(base_url: str, path: str, portfolio_id: Optional[str],
        expect_commit: str, bank: Dict[str, Any],
        headers: Optional[List[str]] = None,
        progress: bool = True) -> Tuple[Dict[str, Any], int]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    reached, authorised, observed_commit = preflight(
        base_url, path, headers or [], portfolio_id)
    commit_verified = bool(
        observed_commit and expect_commit
        and observed_commit.strip().lower().startswith(expect_commit.strip().lower()[:12]))
    provenance = {
        "base_url": base_url, "path": path, "portfolio_id": portfolio_id,
        "expected_commit": expect_commit, "observed_commit": observed_commit,
        "commit_verified": commit_verified,
        "reached": reached, "authorised": authorised,
        "how_established": ("the service's own immutable build stamp on /health, "
                            "compared with the SHA the deploy workflow shipped. An "
                            "application version string is identical across every "
                            "deploy and is not accepted as provenance."),
    }

    if reached != "YES" or authorised != "YES":
        print(f"NOT EXECUTABLE — reached={reached} authorised={authorised}")
        diagnosis = credential_diagnosis() if "401" in str(authorised) else {}
        if diagnosis:
            _report_credential(diagnosis)
            provenance["credential_diagnosis"] = diagnosis
        return {"run_started_at": started, "provenance": provenance,
                "not_executable": "the deployed service was not reached and authorised",
                "bank_version": bank["bank_version"],
                "canonical_case_count": bank["canonical_case_count"],
                "questions": [], "paraphrase_gate": [],
                "latency": latency_block([]),
                "adjudication": {"MI_QUERY_AGENT_V1_LIVE_READY": "NO",
                                 "MI_QUERY_AGENT_V1_RELEASE_CLOSED": "NO",
                                 "hard_criteria": {}, "closure_criteria": {},
                                 "outcome_counts": {}, "failure_reason_counts": {},
                                 "independent_truth": {}, "paraphrase_invariance": {},
                                 "route_conformance": {}}}, 2

    if not commit_verified:
        print(f"NOT EXECUTABLE — production reports {observed_commit!r}, "
              f"the acceptance subject is {expect_commit!r}")
        return {"run_started_at": started, "provenance": provenance,
                "not_executable": ("the running build is not the build under "
                                   "acceptance"),
                "bank_version": bank["bank_version"],
                "canonical_case_count": bank["canonical_case_count"],
                "questions": [], "paraphrase_gate": [],
                "latency": latency_block([]),
                "adjudication": {"MI_QUERY_AGENT_V1_LIVE_READY": "NO",
                                 "MI_QUERY_AGENT_V1_RELEASE_CLOSED": "NO",
                                 "hard_criteria": {"deployed_commit_verified": False},
                                 "closure_criteria": {}, "outcome_counts": {},
                                 "failure_reason_counts": {}, "independent_truth": {},
                                 "paraphrase_invariance": {}, "route_conformance": {}}}, 2

    print(f"deployed commit verified: {observed_commit}")

    # WILL THE CREDENTIAL OUTLIVE THE RUN. Three runs of this bank have now
    # been lost to an expired token, and the dangerous case is not the one that
    # fails at the door: it is the token that expires at question ninety, where
    # every later question returns 401 and the verdict is computed over a book
    # that stopped answering. Checked here, before anything is asked.
    diagnosis = credential_diagnosis()
    provenance["credential_diagnosis"] = diagnosis
    if diagnosis.get("readable"):
        remaining = -diagnosis.get("seconds_past_expiry", 0)
        if diagnosis.get("expired") is False and diagnosis.get("exp"):
            import calendar
            expiry = calendar.timegm(time.strptime(diagnosis["exp"],
                                                   "%Y-%m-%dT%H:%M:%SZ"))
            remaining = int(expiry - time.time())
        print(f"credential expires {diagnosis.get('exp')} "
              f"({remaining} seconds from now)")
        if remaining < RUN_BUDGET_SECONDS:
            print(f"NOT EXECUTABLE — the token has {remaining}s left and this "
                  f"bank takes about {RUN_BUDGET_SECONDS}s to ask. Starting "
                  f"would produce a verdict over a book that stopped answering "
                  f"part way through.")
            return _not_executable(started, provenance, bank,
                                   "the credential would expire during the run"), 2

    print("reading the independent surfaces before any question is asked …")
    collected = truth_mod.collect(base_url, portfolio_id)
    for name, status in collected["endpoint_status"].items():
        print(f"  {name:20s} {status}")

    ask = _live_asker(base_url, path, headers or [], portfolio_id)
    envelopes: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    run_state: Dict[str, Any] = {}

    questions = bank["questions"]
    # C31 first among the forecast cases so C35's ordering check has a baseline.
    for q in questions:
        started_at = time.time()
        envelope = ask(q["question"])
        elapsed = round(time.time() - started_at, 3)
        envelope["__latency__"] = elapsed
        if envelope.get("__http_status__") in (401, 403):
            print(f"NOT EXECUTABLE — the credential was refused at "
                  f"{q['question_id']}, {len(rows)} questions in. A verdict "
                  f"over a bank that stopped being answered is not a verdict.")
            return _not_executable(
                started, provenance, bank,
                f"the credential was refused mid-run at {q['question_id']}"), 2
        envelopes[q["question_id"]] = envelope
        row = score(q, envelope, collected, run_state)
        rows.append(row)
        if q["canonical_case_id"] == "C31" and "C31_horizon_months" not in run_state:
            months = _horizon_months(str(envelope.get("answer") or ""))
            if months is not None:
                run_state["C31_horizon_months"] = months
        if progress:
            print(f"  {q['question_id']} {q['canonical_case_id']}{q['variant_id']} "
                  f"{row['outcome']:18s} {row['observed_route']:26s} {elapsed:6.2f}s",
                  flush=True)

    # C35's ordering check needs C31's answer, which is only known after the
    # forecast cases have run. Re-score the cases that depend on run state
    # rather than leave the check INDETERMINATE for an ordering artefact.
    for i, q in enumerate(questions):
        if "scenario_not_later_than_baseline" in (q.get("checks") or []):
            rows[i] = score(q, envelopes[q["question_id"]], collected, run_state)

    gate = paraphrase_gate(rows, envelopes)
    verdict = adjudicate(rows, gate, provenance, bank)

    payload = {
        "run_started_at": started,
        "bank_version": bank["bank_version"],
        "bank_calibration": bank.get("calibration"),
        "application_under_acceptance": bank.get("application_under_acceptance"),
        "canonical_case_count": bank["canonical_case_count"],
        "provenance": provenance,
        "endpoint_status": collected["endpoint_status"],
        "availability_rules_resolved": collected["availability"],
        "independent_truth_observed": {
            k: v for k, v in collected["truths"].items()
            if not isinstance(v, (list, dict))},
        "questions": rows,
        "paraphrase_gate": gate,
        "latency": latency_block(rows),
        "adjudication": verdict,
    }
    return payload, (0 if verdict["MI_QUERY_AGENT_V1_LIVE_READY"] == "YES" else 1)


def print_summary(payload: Dict[str, Any]) -> None:
    verdict = payload["adjudication"]
    print("\n" + "=" * 72)
    print("MI QUERY AGENT V1 — LIVE ACCEPTANCE")
    print("=" * 72)
    prov = payload["provenance"]
    print(f"build under acceptance : {prov.get('expected_commit')}")
    print(f"build production serves: {prov.get('observed_commit')} "
          f"(verified: {'YES' if prov.get('commit_verified') else 'NO'})")
    if payload.get("not_executable"):
        print(f"NOT EXECUTABLE: {payload['not_executable']}")
        return
    print("\noutcomes")
    for name, count in sorted(verdict["outcome_counts"].items()):
        print(f"  {name:20s} {count:4d}")
    if verdict["failure_reason_counts"]:
        print("\nfailure reasons")
        for name, count in verdict["failure_reason_counts"].items():
            print(f"  {name:46s} {count:4d}")
    inv = verdict["paraphrase_invariance"]
    print(f"\nparaphrase invariance: {inv['invariant']}/{inv['canonical_cases']} "
          f"cases invariant, {inv['failed']} failed {inv['failed_cases']}")
    it = verdict["independent_truth"]
    print(f"independent truth    : {it['carried_cases']} cases / "
          f"{it['carried_variants']} variants carried "
          f"(floor {it['commissioned_floor_cases']}/{it['commissioned_floor_variants']})")
    lat = payload["latency"]["overall"]
    print(f"latency (observational): median {lat['median']}s p90 {lat['p90']}s "
          f"p95 {lat['p95']}s max {lat['max']}s")
    print("\nhard criteria")
    for name, met in verdict["hard_criteria"].items():
        print(f"  {'YES' if met else 'NO ':4s} {name}")
    print("closure criteria")
    for name, met in verdict["closure_criteria"].items():
        print(f"  {'YES' if met else 'NO ':4s} {name}")
    print(f"\nMI_QUERY_AGENT_V1_LIVE_READY     = "
          f"{verdict['MI_QUERY_AGENT_V1_LIVE_READY']}")
    print(f"MI_QUERY_AGENT_V1_RELEASE_CLOSED = "
          f"{verdict['MI_QUERY_AGENT_V1_RELEASE_CLOSED']}")


def print_findings(payload: Dict[str, Any], exemplars: int = 3) -> None:
    """The classification, printed where it can be read.

    The JSON, CSV and HTML carry everything. This exists because the evidence
    has to travel back through a log to be classified at all, and a verdict of
    NO that nobody can take apart into WHICH answers were wrong and WHY is a
    verdict nobody can act on.

    It prints what the run already computed. It changes no expectation, no
    score and no question."""
    rows = payload["questions"]
    failing = [r for r in rows if r["outcome"] in (WRONG, INCORRECT_REFUSAL, ERROR)]
    print("\n" + "=" * 72)
    print(f"FINDINGS — {len(failing)} of {len(rows)} questions did not pass")
    print("=" * 72)

    by_reason: Dict[str, List[Dict[str, Any]]] = {}
    for row in failing:
        for reason in (row.get("failure_reasons") or []):
            if reason == R_ROUTE:
                continue  # reported separately; not on its own a semantic fault
            by_reason.setdefault(reason, []).append(row)

    for reason, group in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        cases = sorted({r["canonical_case_id"] for r in group})
        print(f"\n--- {reason}: {len(group)} questions across {len(cases)} cases")
        print(f"    cases: {' '.join(cases)}")
        for row in group[:exemplars]:
            print(f"    {row['question_id']} {row['canonical_case_id']}"
                  f"{row['variant_id']} [{row['outcome']}] route={row['observed_route']}")
            print(f"      Q: {row['question'][:150]}")
            answer = (row.get("answer") or row.get("error") or "").replace("\n", " ")
            print(f"      A: {answer[:220]}")
            for chk in (row.get("checks") or []):
                if chk["verdict"] == FAIL:
                    print(f"      ! {chk['check']}: {chk['detail'][:220]}")

    print("\n" + "-" * 72)
    print("PARAPHRASE INVARIANCE — cases whose phrasings disagreed")
    print("-" * 72)
    for gate in payload["paraphrase_gate"]:
        if gate["invariant"]:
            continue
        variants = " | ".join(f"{v['variant_id']}={v['outcome']}/{v['answerability']}"
                              for v in gate["variants"])
        print(f"  {gate['canonical_case_id']} {gate['capability_family']}: "
              f"{','.join(gate['failure_modes'])}")
        print(f"    {variants}")
        if gate["detail"]:
            print(f"    {gate['detail'][:200]}")

    unscoreable = [r for r in rows if r["outcome"] == UNSCOREABLE]
    if unscoreable:
        print("\n" + "-" * 72)
        print("UNSCOREABLE — the book could not be asked whether this is right")
        print("-" * 72)
        for case_id in sorted({r["canonical_case_id"] for r in unscoreable}):
            row = next(r for r in unscoreable if r["canonical_case_id"] == case_id)
            print(f"  {case_id}: {row.get('answerability_basis')}")

    outside = payload["adjudication"]["route_conformance"]["outside_expected"]
    if outside:
        print("\n" + "-" * 72)
        print("ROUTE OUTSIDE EXPECTED — recorded, not on its own a semantic fault")
        print("-" * 72)
        for entry in outside:
            print(f"  {entry['question_id']} {entry['case']}: observed "
                  f"{entry['observed']}, expected one of {entry['expected']}")

    numeric = [r for r in rows
               if R_NUMBER in (r.get("failure_reasons") or [])]
    if numeric:
        print("\n" + "=" * 72)
        print(f"EVERY NUMERIC DISAGREEMENT IN FULL — {len(numeric)} questions")
        print("A disagreement is not classifiable from two numbers. Each row "
              "below carries what the")
        print("service says it measured, what the independent surface says it "
              "measured, the units on")
        print("both sides, the tolerance the assertion used, and the assertion "
              "itself.")
        print("=" * 72)
        for r in numeric:
            desc = r.get("truth_description") or {}
            pop = r.get("service_population") or {}
            print(f"\n[{r['question_id']} {r['canonical_case_id']}"
                  f"{r['variant_id']}] {r['outcome']}  route={r['observed_route']}"
                  f"  family={r['capability_family']}")
            print(f"  question   : {r['question']}")
            answer = (r.get("answer") or r.get("error") or "").replace("\n", " ")
            print(f"  answer     : {answer[:300]}")
            print(f"  service pop: dataset={pop.get('dataset')} "
                  f"records={pop.get('records_included')} "
                  f"balance={pop.get('balance_included')} "
                  f"as_of={pop.get('as_of')} filters={pop.get('filters_applied')}")
            print(f"               metric={pop.get('metric')} "
                  f"agg={pop.get('aggregation')} "
                  f"result={pop.get('result_type')} "
                  f"grouped_by={pop.get('grouped_by')}")
            print(f"  truth key  : {r.get('truth_key')} ({r.get('truth_method')})")
            print(f"  truth pop  : {desc.get('population')}")
            print(f"  truth units: {desc.get('unit')}")
            print(f"  truth src  : {desc.get('source')}")
            for chk in (r.get("checks") or []):
                if chk["verdict"] == FAIL and chk["reason"] == R_NUMBER:
                    print(f"  assertion  : {chk['check']} — {chk['detail']}")

    print("\n" + "-" * 72)
    print("INDEPENDENT SURFACES, AS READ")
    print("-" * 72)
    for name, status in (payload.get("endpoint_status") or {}).items():
        print(f"  {name:20s} {status}")
    for key, value in sorted((payload.get("independent_truth_observed") or {}).items()):
        print(f"  truth {key:34s} {value}")
    for key, value in sorted((payload.get("availability_rules_resolved") or {}).items()):
        print(f"  rule  {key:34s} {value}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--path", default="/mi/query")
    parser.add_argument("--portfolio-id", default=None)
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--expect-commit", required=True,
                        help="The deployed SHA under acceptance. A service "
                             "serving anything else is NOT EXECUTABLE.")
    parser.add_argument("--bank", default=str(BANK_PATH))
    parser.add_argument("--json-out", default="mi-query-v1-acceptance.json")
    parser.add_argument("--csv-out", default="mi-query-v1-acceptance.csv")
    parser.add_argument("--html-out", default="mi-query-v1-acceptance.html")
    args = parser.parse_args(argv)

    bank = json.loads(Path(args.bank).read_text(encoding="utf-8"))
    if not bank.get("frozen"):
        print("::error::the bank is not marked frozen")
        return 2

    payload, status = run(args.base_url, args.path, args.portfolio_id,
                          args.expect_commit, bank, args.header)
    Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                              default=str) + "\n", encoding="utf-8")
    if payload["questions"]:
        write_csv(payload["questions"], Path(args.csv_out))
        write_html(payload, Path(args.html_out))
    print_summary(payload)
    if payload["questions"]:
        print_findings(payload)
        # The verdict is repeated LAST because the evidence has to travel back
        # through a log tail, and a reader who can see the findings must not
        # have to lose the counts to do it.
        verdict = payload["adjudication"]
        counts = verdict["outcome_counts"]
        print("\n" + "=" * 72)
        print("  ".join(f"{name}={counts.get(name, 0)}" for name in
                        (CORRECT, CORRECT_REFUSAL, WRONG, INCORRECT_REFUSAL,
                         ERROR, UNSCOREABLE)))
        inv = verdict["paraphrase_invariance"]
        print(f"paraphrase invariant {inv['invariant']}/{inv['canonical_cases']}  "
              f"independent truth {verdict['independent_truth']['carried_cases']} cases"
              f"/{verdict['independent_truth']['carried_variants']} variants  "
              f"commit verified "
              f"{'YES' if payload['provenance'].get('commit_verified') else 'NO'}")
        print(f"MI_QUERY_AGENT_V1_LIVE_READY     = "
              f"{verdict['MI_QUERY_AGENT_V1_LIVE_READY']}")
        print(f"MI_QUERY_AGENT_V1_RELEASE_CLOSED = "
              f"{verdict['MI_QUERY_AGENT_V1_RELEASE_CLOSED']}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
