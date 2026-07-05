#!/usr/bin/env python3
"""
llm_query_parser.py

Translate a natural-language MI question into an :class:`MIQuerySpec`.

Two modes:
    * llm_enabled=False (default)  -> deterministic, offline pattern matcher.
      Safe for unit tests; no network, no API key required.
    * llm_enabled=True             -> optional, mockable Claude call.  The LLM
      is only ever shown the *semantic field catalogue* (field keys, display
      names, descriptions, roles, allowed chart roles / aggregations) — never
      raw data.  It must return STRICT JSON matching MIQuerySpec.  Generated
      content is parsed as data only; it is NEVER executed.

This module deliberately resolves field references against the *actual* MI
semantic registry (by role / format / keyword) rather than hard-coding field
names, because canonical field names differ across deployments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .mi_query_spec import MIQuerySpec
from .mi_query_validator import load_mi_semantics, validate_mi_query

# Cheap default model for NL->spec parsing.  Overridable via the `model` arg.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# --------------------------------------------------------------------------- #
# Field resolution helpers (work against whatever registry is loaded)
# --------------------------------------------------------------------------- #


def _fields(semantics: dict) -> Dict[str, dict]:
    return semantics.get("fields", {})


def _synonyms(entry: dict) -> List[str]:
    """The governed business synonyms for a field. The registry uses ``synonyms``;
    ``aliases`` is accepted as a fallback for forward-compatibility."""
    return list(entry.get("synonyms") or entry.get("aliases") or [])


def find_field(
    semantics: dict,
    role: Optional[str] = None,
    fmt: Optional[str] = None,
    keywords: Iterable[str] = (),
    exclude: Iterable[str] = (),
    prefer_tier: str = "core",
    strict: bool = False,
) -> Optional[str]:
    """Return the best-matching semantic field key, or None.

    Resolution preference (highest first):
      1. Keyword hit on a field at the preferred tier (default ``core``).
      2. Keyword hit on any field.
      3. Any field at the preferred tier matching role/format.
      4. Any field matching role/format.

    This means that when several fields could match the same user phrase the
    parser leans towards ``mi_tier: core``, falling back to ``extended`` only
    when nothing in core fits.

    When ``strict=True`` and keywords are supplied, only keyword hits are
    returned (steps 3-4 are disabled) — used by dimension resolution so an
    unrecognised term yields ``None`` rather than an arbitrary substitute.
    """
    items = _fields(semantics)
    exclude = set(exclude)
    keywords = tuple(k.lower() for k in keywords)

    def ok(key: str, entry: dict) -> bool:
        if key in exclude:
            return False
        if role and entry.get("role") != role:
            return False
        if fmt and entry.get("format") != fmt:
            return False
        return True

    def is_preferred(entry: dict) -> bool:
        return entry.get("mi_tier") == prefer_tier

    def primary_hit(key: str, entry: dict) -> bool:
        # A hit on the field key or its display/business name — the strong signal.
        hay = " ".join([key, str(entry.get("display_name", "")),
                        str(entry.get("business_name", ""))]).lower()
        return bool(keywords) and any(kw in hay for kw in keywords)

    def synonym_hit(key: str, entry: dict) -> bool:
        # A hit ONLY via the governed business synonyms ("customer age", "current
        # ltv", "exposure") — accepted, but RANKED BELOW a primary name hit so an
        # ambiguous keyword ("age") still prefers the field actually named for it
        # (youngest_borrower_age) over one that merely lists it as a synonym
        # (months_on_book / "loan age").
        hay = " ".join(_synonyms(entry)).lower()
        return bool(keywords) and any(kw in hay for kw in keywords)

    preferred_kw: Optional[str] = None
    fallback_kw: Optional[str] = None
    preferred_syn: Optional[str] = None
    fallback_syn: Optional[str] = None
    preferred_any: Optional[str] = None
    fallback_any: Optional[str] = None

    for key, entry in items.items():
        if not ok(key, entry):
            continue
        if primary_hit(key, entry):
            if is_preferred(entry):
                if preferred_kw is None:
                    preferred_kw = key
            elif fallback_kw is None:
                fallback_kw = key
        elif synonym_hit(key, entry):
            if is_preferred(entry):
                if preferred_syn is None:
                    preferred_syn = key
            elif fallback_syn is None:
                fallback_syn = key
        if is_preferred(entry):
            if preferred_any is None:
                preferred_any = key
        elif fallback_any is None:
            fallback_any = key

    if strict and keywords:
        return preferred_kw or fallback_kw or preferred_syn or fallback_syn
    return (preferred_kw or fallback_kw or preferred_syn or fallback_syn
            or preferred_any or fallback_any)


# Preferred balance/exposure fields (mirrors the executor's balance hierarchy).
_PREFERRED_BALANCE = ("current_outstanding_balance", "current_principal_balance",
                      "original_principal_balance")


def _concept_candidates(semantics: dict, role: Optional[str], fmt: Optional[str],
                        keywords: Tuple[str, ...]) -> List[str]:
    """All semantic keys matching role/format whose key/name mentions a keyword."""
    out: List[str] = []
    for key, entry in _fields(semantics).items():
        if role and entry.get("role") != role:
            continue
        if fmt and entry.get("format") != fmt:
            continue
        hay = " ".join([key, str(entry.get("display_name", "")),
                        str(entry.get("business_name", "")),
                        " ".join(_synonyms(entry))]).lower()
        if any(kw in hay for kw in keywords):
            out.append(key)
    return out


def _prefer_present(semantics: dict, default: Optional[str],
                    candidates: List[str], available_columns) -> Optional[str]:
    """Pick the field whose canonical column is actually present in the dataset.

    Keeps alias resolution CONSISTENT and avoids a first-attempt validation
    failure: when the registry default (e.g. ``youngest_borrower_age``) is absent
    from the data but a synonymous field is present, resolve to the present one.
    """
    if available_columns is None:
        return default
    cols = set(available_columns)
    ordered = ([default] if default else []) + [c for c in candidates if c != default]
    for key in ordered:
        if key and _fields(semantics).get(key, {}).get("canonical_field", key) in cols:
            return key
    return default


def _balance_metric(semantics, available_columns=None) -> Optional[str]:
    # Prefer the canonical balance hierarchy so "balance" resolves to the
    # primary exposure field rather than an alphabetically-earlier keyword hit
    # such as ``arrears_balance``.
    fields = _fields(semantics)
    default = next((k for k in _PREFERRED_BALANCE if k in fields), None) \
        or find_field(semantics, role="metric", fmt="currency",
                      keywords=("balance", "outstanding", "principal"))
    cand = list(_PREFERRED_BALANCE) + _concept_candidates(
        semantics, "metric", "currency", ("balance", "outstanding", "principal"))
    return _prefer_present(semantics, default, cand, available_columns)


def _ltv_metric(semantics, available_columns=None) -> Optional[str]:
    default = find_field(semantics, role="metric", fmt="percent",
                         keywords=("ltv", "loan_to_value"))
    cand = _concept_candidates(semantics, "metric", "percent", ("ltv", "loan_to_value"))
    return _prefer_present(semantics, default, cand, available_columns)


def _age_metric(semantics, available_columns=None) -> Optional[str]:
    default = find_field(semantics, role="metric", fmt="integer", keywords=("age",))
    cand = _concept_candidates(semantics, "metric", "integer", ("age",))
    return _prefer_present(semantics, default, cand, available_columns)


def _rate_metric(semantics, available_columns=None) -> Optional[str]:
    default = find_field(semantics, role="metric", fmt="percent", keywords=("rate",))
    cand = _concept_candidates(semantics, "metric", "percent", ("rate", "coupon"))
    return _prefer_present(semantics, default, cand, available_columns)


def _dimension(semantics, keywords=(), exclude=()) -> Optional[str]:
    # STRICT: only return a dimension on a genuine keyword hit — never fall back
    # to an arbitrary dimension (that is what caused broker -> account_status).
    return find_field(semantics, role="dimension", keywords=keywords,
                      exclude=exclude, strict=True)


def _default_weight(semantics, metric_key: Optional[str]) -> Optional[str]:
    if metric_key and metric_key in _fields(semantics):
        wf = _fields(semantics)[metric_key].get("weight_field")
        if wf:
            return wf
    return semantics.get("metadata", {}).get("default_weight_field")


# --------------------------------------------------------------------------- #
# Explicit dimension / metric vocabularies (NL term -> semantic field key)
# --------------------------------------------------------------------------- #
# Longer phrases first so e.g. "age bucket" matches before "age", and
# "broker channel" before "broker". Targets are filtered at runtime to keys
# that actually exist in the loaded registry.

EXPLICIT_DIMENSION_TERMS = {
    "broker channel": "broker_channel",
    "brokers": "broker_channel",
    "broker": "broker_channel",
    "product type": "erm_product_type",
    "sub product type": "erm_sub_product_type",
    "products": "erm_product_type",
    "product": "erm_product_type",
    # Region family. Generic region terms below are resolved data-aware via
    # _preferred_region() (readable collateral_geography first, then NUTS code
    # fields) — NEVER geographic_region_classification (a year). Specific
    # obligor/collateral terms keep their exact field.
    "obligor region": "geographic_region_obligor",
    "borrower region": "geographic_region_obligor",
    "collateral region": "geographic_region_collateral",
    "property region": "collateral_geography",
    "geographic region": "geographic_region_obligor",
    "geography": "geographic_region_obligor",
    "geographic": "geographic_region_obligor",
    "regions": "geographic_region_obligor",
    "region": "geographic_region_obligor",
    "account status": "account_status",
    "status": "account_status",
    "borrower age bucket": "age_bucket",
    "age bucket": "age_bucket",
    "age band": "age_bucket",
    "ltv buckets": "ltv_bucket",
    "ltv bucket": "ltv_bucket",
    "ltv bands": "ltv_bucket",
    "ltv band": "ltv_bucket",
    "age buckets": "age_bucket",
    "age bands": "age_bucket",
    "interest rate buckets": "interest_rate_bucket",
    "ticket size": "ticket_bucket",
    "ticket buckets": "ticket_bucket",
    "ticket bucket": "ticket_bucket",
    "vintage year": "vintage_year",
    "vintage": "vintage_year",
    "origination year": "vintage_year",
    "maturity year": "maturity_year",
    "tenure": "tenure",
    "occupancy": "occupancy_type",
    "interest rate type": "interest_rate_type",
    "rate type": "interest_rate_type",
    "borrower jurisdiction": "borrower_jurisdiction",
    "jurisdiction": "borrower_jurisdiction",
}

# Generic region terms resolved by data-aware preference (see _preferred_region).
_REGION_GENERIC_TERMS = {"region", "regions", "geography", "geographic",
                         "geographic region"}
# Preference for the MI "Region" dimension: readable display field first, then
# NUTS3 code fields. geographic_region_classification (a YEAR) is never a region.
_REGION_PREFERENCE = ("collateral_geography", "geographic_region_collateral",
                      "geographic_region_obligor")


def _preferred_region(semantics: dict, available_columns=None) -> Optional[str]:
    """Pick the MI 'Region' field: readable collateral_geography first, then a
    NUTS3 code field. When available_columns is given, prefer a field whose
    canonical column is actually present in the dataset."""
    fields = _fields(semantics)
    cols = set(available_columns) if available_columns is not None else None
    if cols is not None:
        # Data-aware: only a region field whose column is actually present.
        # If none is present, return None so validation fails clearly rather
        # than substituting an absent field.
        for key in _REGION_PREFERENCE:
            entry = fields.get(key)
            if entry and entry.get("canonical_field", key) in cols:
                return key
        return None
    # No column context: fall back to registry presence (parse-time default).
    for key in _REGION_PREFERENCE:
        if key in fields:
            return key
    return None

# Metric NL terms -> resolver. Order matters (longer/more-specific first).
_METRIC_TERMS = (
    ("weighted average ltv", "ltv"),
    ("loan to value", "ltv"),
    ("ltv", "ltv"),
    ("outstanding balance", "balance"),
    ("balance", "balance"),
    ("exposure", "balance"),
    ("redemptions", "redemptions"),
    ("redemption", "redemptions"),
    ("recoveries", "recoveries"),
    ("recovery", "recoveries"),
    ("default amount", "default_amount"),
    ("losses", "losses"),
    ("arrears", "arrears"),
    ("interest rate", "rate"),
    ("borrower age", "age"),
    ("age", "age"),
    ("count", "count"),
)


def _explicit_dimensions(q: str, semantics: dict, grouping: bool = False,
                         available_columns=None
                         ) -> Tuple[List[str], List[str], str]:
    """Find explicitly-requested dimensions in order of appearance.

    Returns (dimension_keys, matched_terms, remaining_text). Only terms whose
    target key exists in the registry are honoured; matched spans are removed
    from ``remaining_text`` so metric detection does not re-trip on them.

    Generic region terms ("region", "geography", ...) resolve data-aware via
    _preferred_region (readable display field first, then NUTS code fields).

    ``grouping=True`` enables a small set of context-only bucketing terms (a
    bare "age" axis -> age_bucket) used by heatmap/treemap.
    """
    fields = _fields(semantics)
    terms_map = dict(EXPLICIT_DIMENSION_TERMS)
    if grouping:
        # In a grouping chart (heatmap/treemap) a bare "age" axis means the
        # age band, not the numeric age metric. Same idea for the other
        # bucketable measures, but only when their bucket dimension exists.
        for bare, bucket in (("age", "age_bucket"),):
            if bucket in fields and bare not in terms_map:
                terms_map[bare] = bucket
    remaining = q
    found: List[Tuple[int, str, str]] = []  # (position, key, term)
    for term in sorted(terms_map, key=len, reverse=True):
        if term in _REGION_GENERIC_TERMS:
            key = _preferred_region(semantics, available_columns)
        else:
            key = terms_map[term]
        if not key or key not in fields:
            continue
        pat = r"\b" + re.escape(term) + r"\b"
        m = re.search(pat, remaining)
        if m:
            found.append((m.start(), key, term))
            remaining = remaining[:m.start()] + " " * (m.end() - m.start()) + remaining[m.end():]
    found.sort(key=lambda t: t[0])
    keys: List[str] = []
    terms: List[str] = []
    for _, key, term in found:
        if key not in keys:
            keys.append(key)
            terms.append(term)
    return keys, terms, remaining


def _resolve_metric(token: str, semantics: dict) -> Tuple[Optional[str], str]:
    """Map a metric token to a semantic key + a default aggregation."""
    fields = _fields(semantics)
    if token == "balance":
        return _balance_metric(semantics), "sum"
    if token == "ltv":
        return _ltv_metric(semantics), "weighted_avg"
    if token == "age":
        return _age_metric(semantics), "avg"
    if token == "count":
        return None, "count"
    direct = {
        "redemptions": "redemptions_received_in_period",
        "recoveries": "recoveries_in_period",
        "default_amount": "default_amount",
        "losses": "allocated_losses",
        "arrears": "arrears_balance",
        "rate": "current_interest_rate",
    }
    key = direct.get(token)
    if key and key in fields:
        agg = "weighted_avg" if fields[key].get("format") == "percent" else "sum"
        return key, agg
    return None, "sum"


# Aggregation-intent qualifiers in a metric phrase. Distinguishes:
#   total / sum / aggregate            -> sum
#   weighted average / weighted avg    -> weighted_avg
#   simple / unweighted average        -> avg (forced unweighted)
#   average / avg / mean               -> avg_generic (resolved by metric format)
_WEIGHTED_AVG_RE = re.compile(r"\bweighted\s+(?:average|avg|mean)\b")
_SIMPLE_AVG_RE = re.compile(r"\b(?:simple|unweighted|plain|straight)\s+(?:average|avg|mean)\b")
_AVG_RE = re.compile(r"\b(?:average|avg|mean)\b")
_TOTAL_RE = re.compile(r"\b(?:total|sum of|aggregate|overall)\b")


def _aggregation_intent(text: str) -> Optional[str]:
    """Explicit aggregation qualifier in ``text``:
    'weighted_avg' | 'avg' | 'avg_generic' | 'sum' | None."""
    if _WEIGHTED_AVG_RE.search(text):
        return "weighted_avg"
    if _SIMPLE_AVG_RE.search(text):
        return "avg"
    if _AVG_RE.search(text):
        return "avg_generic"
    if _TOTAL_RE.search(text):
        return "sum"
    return None


def _apply_agg_intent(metric_key: Optional[str], default_agg: str,
                      intent: Optional[str], semantics: dict) -> str:
    """Resolve the aggregation given an explicit qualifier and the metric format.

    'average loan balance' -> avg (mean = sum/count); 'weighted average ltv' ->
    weighted_avg; a bare 'average' on a percent metric defaults to the balance-
    weighted average (the MI convention) while currency/integer use a plain mean.
    """
    if not intent:
        return default_agg
    fmt = _fields(semantics).get(metric_key, {}).get("format") if metric_key else None
    if intent == "weighted_avg":
        return "weighted_avg"
    if intent == "avg":
        return "avg"
    if intent == "avg_generic":
        return "weighted_avg" if fmt == "percent" else "avg"
    if intent == "sum":
        # Never coerce a percent metric to a (meaningless) raw sum.
        return default_agg if fmt == "percent" else "sum"
    return default_agg


def _detect_metric(text: str, semantics: dict) -> Tuple[Optional[str], str, List[str]]:
    """Return (metric_key, aggregation, matched_terms) from free text.

    An explicit aggregation qualifier ("average"/"weighted average"/"total") in the
    same phrase overrides the metric's default aggregation, so "average loan
    balance" means the mean balance, not the total.
    """
    matched: List[str] = []
    intent = _aggregation_intent(text)
    for term, token in _METRIC_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", text):
            key, agg = _resolve_metric(token, semantics)
            if token != "count":
                agg = _apply_agg_intent(key, agg, intent, semantics)
            matched.append(term)
            return key, agg, matched
    return None, "sum", matched


def _detect_top_n(q: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)\b", q)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# ERE securitisation sprint — analytical-intent recognition.
# These run FIRST in the deterministic parser so a cross-period comparison,
# a securitisation scale-up / run-rate forecast, or a risk-limit question is
# never silently collapsed to a point-in-time KPI. Each emits a governed spec
# (no hallucinated fields) that the runtime / API layer resolves against the
# governed evolution / forecast / risk-monitor data.
# --------------------------------------------------------------------------- #

# A "count of things" intent that the legacy metric grammar does not surface as a
# metric token (e.g. "number of loans"). Used to keep loan/case COUNT evolutions
# as a count metric instead of defaulting to balance/sum.
_COUNT_INTENT_RE = re.compile(
    r"\b(loan count|case count|number of (?:loans|cases|mortgages|accounts|deals|"
    r"pipeline cases)|how many (?:loans|cases|borrowers|mortgages|accounts)|"
    r"count of (?:loans|cases)|loan numbers|case numbers|deal count)\b")


def _wants_count(q: str) -> bool:
    return bool(_COUNT_INTENT_RE.search(q)) or bool(re.search(r"\bcount\b", q))


# Period tokens for cross-period comparison. Only FULL month names and a small
# set of unambiguous abbreviations are matched (never bare "may"/"mar"/"jun"
# which are common words), plus explicit relative-period phrases.
_MONTH_NAMES = ("january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december")
_SAFE_MONTH_ABBR = {"oct": "October", "nov": "November", "dec": "December",
                    "jan": "January", "feb": "February", "sept": "September"}
_RELATIVE_PERIOD_TERMS = ("last week", "prior week", "previous week", "last month",
                          "prior month", "previous month", "prior pipeline",
                          "prior run", "prior period")


def _detect_periods(q: str) -> List[str]:
    """Ordered, de-duplicated period tokens mentioned in ``q`` (months only —
    the relative-period fallback is handled by the compare recogniser)."""
    found: List[Tuple[int, str]] = []
    for name in _MONTH_NAMES:
        for mt in re.finditer(r"\b" + name + r"\b", q):
            found.append((mt.start(), name.capitalize()))
    for ab, full in _SAFE_MONTH_ABBR.items():
        for mt in re.finditer(r"\b" + ab + r"\b", q):
            found.append((mt.start(), full))
    found.sort(key=lambda t: t[0])
    out: List[str] = []
    seen = set()
    for _, p in found:
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


_COMPARE_TRIGGER_RE = re.compile(
    r"\bcompare[ds]?\b|change (?:from|between)|how did .+ change|"
    r"compared (?:to|with)|versus")


def _compare_recognizer(q: str, title: str, semantics: dict
                        ) -> Optional[Tuple[MIQuerySpec, dict]]:
    """Cross-period comparison → governed ``temporal_mode='compare'`` plan.

    Resolves the comparison metric and exactly two period tokens (A vs B). The
    runtime / API layer fills value A, value B, absolute + % delta, source
    periods and a controlled insufficient-data response from evolution data.
    """
    if not _COMPARE_TRIGGER_RE.search(q):
        return None
    periods = _detect_periods(q)
    if len(periods) < 2:
        rel = next((rp for rp in _RELATIVE_PERIOD_TERMS if rp in q), None)
        if rel:
            periods = ([periods[0], rel] if periods else ["latest", rel])
        else:
            return None
    metric, agg, matched = _detect_metric(q, semantics)
    if _wants_count(q):
        metric, agg = None, "count"
    elif metric is None:
        agg = "sum"  # money compare (funded / pipeline amount); no field referenced
    spec = MIQuerySpec(
        intent="summary", chart_type="none", metric=metric, aggregation=agg,
        execution_mode="temporal", temporal_mode="compare",
        compare_periods=periods[:2], output_format="table", title=title,
        explanation=("Governed cross-period comparison (period A vs period B) over "
                     "governed evolution data: value A, value B, absolute and % "
                     "delta, source periods and a controlled insufficient-data "
                     "response when a period is unavailable."))
    return spec, _det_meta("high", False, [metric] if metric else ["temporal_compare"],
                           note="temporal_compare")


_FORECAST_SCALE_RE = re.compile(
    r"run[\s-]?rate|extrapolat|scale[\s-]?up|"
    r"when (?:do|does|will|can) (?:we|the book|it|the portfolio) reach|"
    r"time to (?:reach|securitisation|scale)|reach £?\s?\d|"
    r"(?:downside|upside|base) forecast|securitisation scale|"
    r"how much pipeline is needed|completion rate is assumed|what conversion rate|"
    r"funded balance extrapolation|annualised completion|"
    r"what happens if .*run.?rate|milestone|"
    # A "forecast curve" / "projection curve" / "balance curve" is a request for
    # the forward funded-balance line, not a point-in-time KPI.
    r"forecast curve|projection curve|balance curve|"
    r"(?:forecast|projected|project).{0,20}curve|curve.{0,20}(?:forecast|funded balance)")


def _forecast_target_value(q: str) -> Optional[float]:
    m = re.search(r"£?\s*(\d+(?:\.\d+)?)\s*m(?:illion)?\b", q)
    if m:
        return float(m.group(1)) * 1_000_000
    m2 = re.search(r"£\s*([\d,]{4,})", q)
    if m2:
        try:
            return float(m2.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def _forecast_question_kind(q: str) -> str:
    if "compare" in q and ("forecast" in q or "extrapolat" in q or "run" in q):
        return "compare_models"
    if "how much pipeline" in q and "reach" in q:
        return "pipeline_needed"
    if "reach" in q and ("when" in q or re.search(r"£?\s?\d+\s*m", q)):
        return "reach_threshold"
    if "what happens if" in q:
        return "scenario"
    if "downside" in q:
        return "scenario_downside"
    if "upside" in q:
        return "scenario_upside"
    if "completion rate is assumed" in q or "what conversion rate" in q:
        return "conversion"
    if "annualised" in q:
        return "run_rate_annualised"
    if re.search(r"run[\s-]?rate", q):
        return "run_rate"
    if "compare" in q and ("forecast" in q or "extrapolat" in q):
        return "compare_models"
    if "extrapolat" in q:
        return "extrapolation_curve"
    return "extrapolation_curve"


def _forecast_scale_recognizer(q: str, title: str
                               ) -> Optional[Tuple[MIQuerySpec, dict]]:
    """Securitisation scale-up / run-rate question → governed
    ``forecast_mode='extrapolation'`` plan (resolved by /mi/forecast/extrapolation)."""
    if not _FORECAST_SCALE_RE.search(q):
        return None
    kind = _forecast_question_kind(q)
    spec = MIQuerySpec(
        intent="summary", chart_type="none", metric=None, aggregation="sum",
        execution_mode="state", forecast_mode="extrapolation",
        forecast_question=kind, forecast_target_value=_forecast_target_value(q),
        output_format="table", title=title,
        explanation=("Securitisation scale-up forecast (completion run-rate / KFI "
                     "conversion extrapolation) with downside/base/upside scenario "
                     "bands and milestone dates to funding thresholds. Distinct from "
                     "the point-in-time weighted-pipeline forecast."))
    return spec, _det_meta("high", False, ["forecast_extrapolation"],
                           note="forecast_scale:" + kind)


_RISK_LIMIT_RE = re.compile(
    r"\brisk limits?\b|concentration limit|\blimit breach|\bbreach(?:ed|es)?\b|"
    r"\bheadroom\b|within (?:the |our )?limits?|over (?:the )?limits?|"
    r"exceed(?:s|ed)? (?:the )?limits?|against (?:the )?limits?|schedule 8|"
    r"limit status|limit utilis|which limits|are we within")

# Natural-language risk-limit category -> the category key used by the risk
# monitor (``risk_limits.testsByCategory``). Order matters (most specific first).
_RISK_LIMIT_CATEGORY_TERMS: List[Tuple[str, str]] = [
    (r"top\s*\d*\s*broker|broker|intermediary|introducer", "broker_concentration"),
    (r"geograph|region|location|area|nuts", "geographic_concentration"),
    (r"large loan|loan size|single loan|big loan", "large_loan_concentration"),
    (r"\bltv\b|loan to value|valuation", "ltv_limit"),
    (r"variable rate|interest rate|\bwac\b|coupon", "interest_rate_limit"),
    (r"joint borrower|joint lives", "joint_borrower_limit"),
    (r"single borrower|per borrower|borrower concentration", "borrower_concentration"),
    (r"aged|age limit|over 85", "age_limit"),
]


def _risk_limit_category(q: str) -> Optional[str]:
    """The specific risk-limit category a question scopes to, or None for all."""
    for pattern, cat in _RISK_LIMIT_CATEGORY_TERMS:
        if re.search(pattern, q):
            return cat
    return None


def _risk_limit_recognizer(q: str, title: str
                           ) -> Optional[Tuple[MIQuerySpec, dict]]:
    """Risk-limit / concentration question → governed
    ``risk_monitor_mode='concentration'`` plan (resolved by /mi/risk-limits)."""
    if not _RISK_LIMIT_RE.search(q):
        return None
    category = _risk_limit_category(q)
    spec = MIQuerySpec(
        intent="summary", chart_type="none", metric=None, aggregation="count",
        execution_mode="risk", risk_monitor_mode="concentration",
        risk_limit_query=True, risk_limit_category=category,
        output_format="table", title=title,
        explanation=("Governed risk-limit / concentration monitor: actual exposure "
                     "vs Schedule 8 limit, headroom, pass/warn/fail status, source and "
                     "movement; controlled needs-review / unavailable when a limit or "
                     "field is missing."))
    return spec, _det_meta("high", False, ["risk_limits"],
                           note=f"risk_limit:{category or 'all'}")


def _det_meta(confidence: str, explicit: bool, terms: List[str],
              substituted: bool = False, note: str = "") -> dict:
    return {
        "explicit_dimension_requested": explicit,
        "requested_dimension_terms": terms,
        "dimension_substituted": substituted,
        "parser_confidence": confidence,
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Deterministic (offline) parser
# --------------------------------------------------------------------------- #


# Numeric comparison phrases -> canonical operator (longest match first).
# A finance value: optional currency prefix (£/$/€), digits with optional
# thousands commas, optional decimal, an optional k/m/bn multiplier and an
# optional trailing %.  Captures (number, suffix).  Examples it accepts:
#   "40", "40%", "200000", "100,000", "£100k", "£0.2m", "$1.5bn", "£200K"
_VALUE = r"(?:£|\$|€)?\s*(-?\d[\d,]*(?:\.\d+)?)\s*(k|m|bn|b|K|M|BN|B)?\s*%?"
_MULTIPLIER = {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9}


def _amount(num: str, suffix: Optional[str]) -> float:
    """Coerce a captured (number, suffix) into a float, applying k/m/bn and
    stripping thousands separators. ``"100,000" -> 100000``, ``"£0.2m" -> 200000``."""
    value = float(str(num).replace(",", ""))
    if suffix:
        value *= _MULTIPLIER.get(suffix.lower(), 1.0)
    return value


# (regex, op).  Each non-``between`` pattern captures (number, suffix); ``between``
# captures (n1, s1, n2, s2).  Order matters: the two-word operators come first so
# "greater than or equal to" is not shadowed by "greater than".
_FILTER_COMPARATORS: List[Tuple[str, str]] = [
    (rf"between\s+{_VALUE}\s+and\s+{_VALUE}", "between"),
    (rf"(?:greater than or equal to|at least|no less than|>=)\s*{_VALUE}", "ge"),
    (rf"(?:less than or equal to|at most|no more than|<=)\s*{_VALUE}", "le"),
    (rf"(?:more than|greater than|over|above|>)\s*{_VALUE}", "gt"),
    (rf"(?:less than|under|below|fewer than|<)\s*{_VALUE}", "lt"),
    (rf"(?:equal to|equals|exactly|=)\s*{_VALUE}", "eq"),
]


def _amount_from_match(m: "re.Match", op: str):
    """Value(s) for a comparator match, applying currency/k-m/comma coercion."""
    if op == "between":
        return [_amount(m.group(1), m.group(2)), _amount(m.group(3), m.group(4))]
    return _amount(m.group(1), m.group(2))


# Age stated WITHOUT an explicit comparator: "60 year old", "aged 60", "age 60",
# "60-year-old", "60 yo", "60 years of age". Resolved to an equality on the
# borrower-age field (only when no comparator/postfix clause already matched).
_AGE_EQUALITY_RE = re.compile(
    r"\b(\d{2,3})\s*[- ]?\s*(?:year[- ]?old|years?\s*old|yo|years?\s+of\s+age)\b"
    r"|\b(?:aged|age)\s+(\d{2,3})\b")


def _age_equality_value(clause: str) -> Optional[float]:
    m = _AGE_EQUALITY_RE.search(clause)
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    return float(raw) if raw else None


def _filter_field_of(q: str, semantics: dict, available_columns=None) -> Optional[str]:
    """Resolve the field a numeric threshold applies to from the question text."""
    if "ltv" in q or "loan to value" in q:
        return _ltv_metric(semantics, available_columns)
    # Age threshold: "age", "youngest", "aged", "borrower(s)", "years"/"yrs",
    # "year old" — all imply the borrower-age field in a numeric-threshold clause.
    if re.search(r"\b(age|aged|youngest|borrowers?|years?|yrs?|yo|year[- ]?old|older)\b", q):
        return _age_metric(semantics, available_columns)
    if "rate" in q or "interest" in q or "coupon" in q:
        return _rate_metric(semantics, available_columns)
    if "balance" in q or "outstanding" in q or "exposure" in q:
        return _balance_metric(semantics, available_columns)
    if "valuation" in q or "value" in q:
        return find_field(semantics, role="metric", fmt="currency",
                          keywords=("valuation", "value"))
    return None


def _parse_numeric_filter(q: str, semantics: dict) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Detect a single numeric comparison filter, e.g. "youngest age more than 70".

    Returns ``(field_key, {"op": ..., "value": ...})`` or ``None``.
    """
    for pattern, op in _FILTER_COMPARATORS:
        m = re.search(pattern, q)
        if not m:
            continue
        field = _filter_field_of(q, semantics)
        if not field:
            return None
        return field, {"op": op, "value": _amount_from_match(m, op)}
    return None


# --------------------------------------------------------------------------- #
# Ranking ("largest ... / top ...") + two-dimensional grouping helpers
# --------------------------------------------------------------------------- #

# Ranking trigger words and the implied sort direction.
# NB: "most" is deliberately excluded — "most concentrated" is a concentration
# question, not a top-N ranking; it is handled by the generic concentration path.
_RANK_DESC = ("largest", "biggest", "highest", "greatest", "top ")
_RANK_ASC = ("smallest", "lowest", "bottom")


def _detect_ranking(q: str) -> Tuple[bool, str, Optional[int]]:
    """Return ``(is_ranking, direction, limit)`` for a 'largest/top N' phrase."""
    direction = "desc"
    is_ranking = False
    if any(t in q for t in _RANK_DESC):
        is_ranking, direction = True, "desc"
    if any(t in q for t in _RANK_ASC):
        is_ranking, direction = True, "asc"
    limit = _detect_top_n(q)
    if limit is None:
        m = re.search(r"\b(\d+)\s+(?:largest|biggest|highest|smallest|lowest)\b", q)
        if m:
            limit = int(m.group(1))
    return is_ranking, direction, limit


# Bare numeric-axis terms (NOT explicitly bucketed) -> the bucket dimension they
# group into when used as a categorical grouping segment, plus a resolver for the
# underlying numeric field (used for bubble axes).
_NUMERIC_AXIS_BUCKET = {
    "ltv": "ltv_bucket",
    "loan to value": "ltv_bucket",
    "age": "age_bucket",
    "borrower age": "age_bucket",
    "rate": "interest_rate_bucket",
    "interest rate": "interest_rate_bucket",
    "balance": "ticket_bucket",
    "outstanding balance": "ticket_bucket",
    "exposure": "ticket_bucket",
}


def _resolve_numeric_axis(term: str, semantics: dict, available_columns=None) -> Optional[str]:
    """The numeric (measure) field for a bare axis term (bubble axis)."""
    if "ltv" in term or "loan to value" in term:
        return _ltv_metric(semantics, available_columns)
    if "age" in term:
        return _age_metric(semantics, available_columns)
    if "rate" in term or "interest" in term:
        return _rate_metric(semantics, available_columns)
    if "balance" in term or "outstanding" in term or "exposure" in term:
        return _balance_metric(semantics, available_columns)
    return None


def _grouping_segments(q: str) -> Tuple[str, List[str]]:
    """Split ``<metric part> by <dim> [by/and <dim> ...]`` into the metric part
    (before the first ``by``) and the ordered list of grouping segments after it.

    Handles both ``by X by Y`` and ``by X and Y`` / ``X, Y`` separators.
    """
    parts = re.split(r"\bby\b", q)
    metric_part = parts[0].strip()
    segments: List[str] = []
    for chunk in parts[1:]:
        for seg in re.split(r"\band\b|,", chunk):
            seg = seg.strip()
            # strip trailing presentation words ("as a heatmap" etc.)
            seg = re.sub(r"\b(as a|as an|chart|heatmap|treemap|bar|table)\b.*$", "", seg).strip()
            if seg:
                segments.append(seg)
    return metric_part, segments


def _classify_segment(seg: str, semantics: dict, available_columns=None
                      ) -> Optional[Tuple[str, str, Optional[str]]]:
    """Classify one grouping segment.

    Returns ``("categorical", dim_key, None)`` for an inherently categorical
    dimension (region, broker, product, *bucket*/*band*, vintage, status, …),
    ``("numeric", numeric_field, bucket_dim)`` for a bare numeric axis term
    (ltv / age / rate / balance without an explicit bucket word), or ``None``.
    """
    # Explicit categorical dimension (NOT grouping=True, so a bare "age"/"ltv"
    # is NOT forced to a bucket here — that is what distinguishes bubble axes).
    keys, _terms, _rem = _explicit_dimensions(seg, semantics, grouping=False,
                                               available_columns=available_columns)
    if keys:
        return ("categorical", keys[0], None)
    # Bare numeric axis term -> numeric (bubble axis) + its bucket dimension.
    for term, bucket in sorted(_NUMERIC_AXIS_BUCKET.items(), key=lambda kv: len(kv[0]),
                               reverse=True):
        if re.search(r"\b" + re.escape(term) + r"\b", seg):
            return ("numeric", _resolve_numeric_axis(term, semantics, available_columns), bucket)
    return None


def _classify_segments(q: str, semantics: dict, available_columns=None
                       ) -> Tuple[str, List[Tuple[str, str, Optional[str]]]]:
    """``(metric_part, [classified_segment, ...])`` for the grouping part of q."""
    metric_part, segments = _grouping_segments(q)
    classes: List[Tuple[str, str, Optional[str]]] = []
    for seg in segments:
        c = _classify_segment(seg, semantics, available_columns)
        if c is not None and c[1]:
            classes.append(c)
    return metric_part, classes


def _is_bucket_dim(key: Optional[str]) -> bool:
    return bool(key) and (key.endswith("_bucket") or key.endswith("_band")
                          or "bucket" in key)


def _grouped_metric(metric_part: str, q: str, semantics: dict) -> Tuple[Optional[str], str]:
    """Resolve the metric for a grouped query, preferring the phrase BEFORE the
    first ``by`` (the metric side) and falling back to the whole question."""
    metric, agg, matched = _detect_metric(metric_part, semantics)
    if metric is None and not matched:
        metric, agg, _ = _detect_metric(q, semantics)
    return metric, agg


# --- multi-filter parsing --------------------------------------------------- #
# Region/geography categorical filter, e.g. "geographic region south west",
# "region south west", "in south west". The value is normalised to Title Case;
# the executor matches case-insensitively against the prepared dimension values.
_CATEGORICAL_FILTER_RE = re.compile(
    r"(?:geographic\s+region|geographic|geography|region|in)\s+"
    r"([a-z][a-z]*(?:\s+[a-z]+){0,2})\s*$")
_CATEGORICAL_STOPWORDS = {"the", "loans", "loan", "with", "and", "by", "more",
                          "less", "than", "over", "under", "above", "below"}


def _parse_categorical_filter(clause: str, semantics: dict, available_columns=None
                              ) -> Optional[Tuple[str, str]]:
    """Detect a categorical region filter in a clause -> (field_key, value)."""
    m = _CATEGORICAL_FILTER_RE.search(clause.strip())
    if not m:
        return None
    value = m.group(1).strip()
    if not value or value in _CATEGORICAL_STOPWORDS:
        return None
    field = _preferred_region(semantics, available_columns) or "geographic_region_obligor"
    if field not in _fields(semantics):
        return None
    return field, value.title()


# Borrower-structure intent ("joint" / "sole" borrowers). Resolved to a
# borrower_structure value filter when that field is present, else to a
# number_of_borrowers numeric filter as a documented fallback.
_BORROWER_STRUCTURE_TERMS = (
    ("joint borrowers", "joint"), ("joint borrower", "joint"), ("joint", "joint"),
    ("sole borrower", "sole"), ("single borrower", "sole"), ("sole", "sole"),
)
_BORROWER_STRUCTURE_VALUE = {"joint": "Joint", "sole": "Sole"}


def _borrower_structure_filter(q: str, semantics: dict, available_columns=None
                               ) -> Optional[Tuple[Dict[str, Any], str]]:
    """Detect a 'joint'/'sole' borrower intent and resolve it to a filter.

    Returns ``(filters, note)`` or None. Prefers a ``borrower_structure`` value
    filter; falls back to a ``number_of_borrowers`` threshold (>=2 joint / ==1
    sole) and notes the substitution; if neither field exists, returns an empty
    filter set with a note suggesting number_of_borrowers.
    """
    fields = _fields(semantics)
    kind = None
    for term, k in _BORROWER_STRUCTURE_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", q):
            kind = k
            break
    if kind is None:
        return None
    cols = set(available_columns) if available_columns is not None else None
    has = lambda key: key in fields and (cols is None or
                                         fields[key].get("canonical_field", key) in cols)
    if has("borrower_structure"):
        return {"borrower_structure": _BORROWER_STRUCTURE_VALUE[kind]}, \
            f"borrower_structure = {_BORROWER_STRUCTURE_VALUE[kind]}"
    if has("number_of_borrowers"):
        cond = {"op": "ge", "value": 2} if kind == "joint" else {"op": "eq", "value": 1}
        return {"number_of_borrowers": cond}, \
            (f"borrower_structure not available; used number_of_borrowers "
             f"{'>= 2' if kind == 'joint' else '== 1'} as a proxy for {kind}")
    return {}, ("borrower_structure is not in this dataset; consider mapping "
                "number_of_borrowers to identify joint vs sole borrowers")


# Postfix comparators where the NUMBER precedes the operator, e.g. "70+",
# "aged 70 or above", "75 or older", "60 or below". (Prefix comparators in
# _FILTER_COMPARATORS cover "above 70", "between 20 and 40", etc.)
_POSTFIX_COMPARATORS: List[Tuple[str, str]] = [
    (r"(-?\d+(?:\.\d+)?)\s*(?:years?|yrs?)?\s*(?:\+|\bor (?:above|over|older|more|greater)\b|\band (?:above|over|older)\b)", "ge"),
    (r"(-?\d+(?:\.\d+)?)\s*(?:years?|yrs?)?\s*(?:\bor (?:below|under|younger|less|fewer)\b|\band (?:below|under|younger)\b)", "le"),
]


def _parse_filters(q: str, semantics: dict, available_columns=None) -> Dict[str, Any]:
    """Parse one or more filters joined by ``and`` / ``with`` (numeric thresholds —
    prefix OR postfix — and a categorical region value). ``{field_key: condition}``."""
    filters: Dict[str, Any] = {}
    work_q = q
    # Parse a 'between A and B' first so its 'and' is not used as a clause split.
    bm = re.search(_FILTER_COMPARATORS[0][0], work_q)
    if bm:
        field = _filter_field_of(work_q[max(0, bm.start() - 40):bm.end()], semantics)
        if field:
            filters[field] = {"op": "between", "value": _amount_from_match(bm, "between")}
        work_q = work_q[:bm.start()] + " " + work_q[bm.end():]

    # Split into clauses on 'and' / 'with' so "<age> 70+ with LTV above 50" yields
    # two independent thresholds.
    for clause in re.split(r"\band\b|\bwith\b", work_q):
        clause = clause.strip()
        if not clause:
            continue
        field = _filter_field_of(clause, semantics)
        # Postfix first ("70+", "70 or above") — a number-before-operator phrase.
        matched = False
        for pattern, op in _POSTFIX_COMPARATORS:
            m = re.search(pattern, clause)
            if m and field:
                filters[field] = {"op": op, "value": float(m.group(1))}
                matched = True
                break
        if matched:
            continue
        for pattern, op in _FILTER_COMPARATORS[1:]:  # skip 'between' (done above)
            m = re.search(pattern, clause)
            if not m:
                continue
            if field:
                filters[field] = {"op": op, "value": _amount_from_match(m, op)}
                matched = True
            break
        if matched:
            continue
        # Age stated without a comparator ("60 year old", "aged 60") -> equality.
        age_field = _age_metric(semantics, available_columns)
        if field == age_field and age_field:
            age_val = _age_equality_value(clause)
            if age_val is not None:
                filters[age_field] = {"op": "eq", "value": age_val}
                continue
        cat = _parse_categorical_filter(clause, semantics, available_columns)
        if cat:
            filters[cat[0]] = cat[1]
    return filters


def _build_two_dim_spec(metric: Optional[str], dims: List[str], semantics: dict,
                        title: str, explicit: bool, terms: List[str],
                        has_count: bool = False) -> Tuple[MIQuerySpec, dict]:
    """Build a two-dimensional grouped (heatmap / matrix) spec."""
    fields = _fields(semantics)
    if has_count or metric is None:
        metric, agg, weight = (None, "count", None)
    else:
        agg = "weighted_avg" if fields.get(metric, {}).get("format") == "percent" else "sum"
        weight = _default_weight(semantics, metric) if agg == "weighted_avg" else None
    conf = "high" if len([d for d in dims if d]) >= 2 else "low"
    spec = MIQuerySpec(
        intent="chart", chart_type="heatmap", metric=metric,
        dimensions=[d for d in dims if d][:2], aggregation=agg, weight_field=weight,
        title=title, explanation="Matrix of a metric across two dimensions.",
        output_format="chart")
    return spec, _det_meta(conf, explicit, terms)


def _build_ranking_spec(q: str, title: str, rank_dir: str, rank_limit: Optional[int],
                        top_n: Optional[int], semantics: dict, available_columns=None
                        ) -> Optional[Tuple[MIQuerySpec, dict]]:
    """Build a ranked spec: grouped ranking bar (a categorical dimension is
    present) or a loan-level 'top loans' ranking table."""
    fields = _fields(semantics)
    rmetric, ragg, _ = _detect_metric(q, semantics)
    # The ranked measure: prefer balance whenever it is explicitly named so that
    # "largest balance by ltv" ranks balance, not the LTV grouping term. Respect an
    # explicit aggregation qualifier so "highest AVERAGE loan balance by broker"
    # ranks the mean balance, not the total.
    if re.search(r"\b(balance|outstanding|exposure)\b", q):
        rmetric = _balance_metric(semantics)
        intent = _aggregation_intent(q)
        ragg = _apply_agg_intent(rmetric, "sum", intent, semantics) if intent else "sum"
    if rmetric is None:
        rmetric, ragg = _balance_metric(semantics), "sum"

    # Grouping dimension: an explicit categorical dimension anywhere, otherwise a
    # bare post-"by" numeric term's bucket dimension — never the ranked metric.
    dim: Optional[str] = None
    gkeys, gterms, _ = _explicit_dimensions(q, semantics, grouping=True,
                                            available_columns=available_columns)
    for k in gkeys:
        if k != rmetric:
            dim, gterms = k, gterms
            break
    if dim is None:
        _mp, segs = _grouping_segments(q)
        for seg in segs:
            c = _classify_segment(seg, semantics, available_columns)
            if not c:
                continue
            if c[0] == "categorical" and c[1] != rmetric:
                dim = c[1]
                break
            if c[0] == "numeric" and c[2] in fields:
                seg_is_metric = (bool(re.search(r"\b(balance|outstanding|exposure)\b", seg))
                                 and rmetric == _balance_metric(semantics))
                if not seg_is_metric:
                    dim = c[2]
                    break

    if dim is not None:
        weight = _default_weight(semantics, rmetric) if ragg == "weighted_avg" else None
        spec = MIQuerySpec(
            intent="chart", chart_type="bar", metric=rmetric, dimension=dim,
            aggregation=ragg, weight_field=weight, top_n=(rank_limit or top_n),
            sort_by=rmetric, sort_direction=rank_dir, ranking_mode="grouped",
            title=title, explanation="Ranked bar of a metric by dimension.",
            output_format="chart")
        return spec, _det_meta("high", True, gterms or [dim])

    # No dimension -> a loan-level "top loans" ranking table.
    spec = MIQuerySpec(
        intent="table", chart_type="none", metric=rmetric,
        aggregation="loan_level", ranking_mode="loan_level", sort_by=rmetric,
        sort_direction=rank_dir, limit=(rank_limit or top_n or 10),
        output_format="table", title=title,
        explanation="Top loans ranked by a measure.")
    return spec, _det_meta("high", False, [rmetric])


def _deterministic_parse(question: str, semantics: dict,
                         available_columns=None) -> Tuple[MIQuerySpec, dict]:
    """Parse a question into (MIQuerySpec, deterministic-parser metadata).

    Honours explicitly-requested dimensions EXACTLY and never substitutes an
    unrelated dimension. An explicit term whose canonical column is missing is
    still returned (validation then fails cleanly) — it is never swapped out.
    """
    q = question.lower().strip()
    title = question.strip()
    top_n = _detect_top_n(q)

    # ---- ERE analytical intents (checked first; emit governed plans) --------
    # A scale-up / run-rate forecast, a cross-period comparison, or a risk-limit
    # question must never fall through to a point-in-time KPI. Forecast is checked
    # before compare so "compare ... run-rate extrapolation" routes to forecast.
    fc = _forecast_scale_recognizer(q, title)
    if fc is not None:
        return fc
    cmp_spec = _compare_recognizer(q, title, semantics)
    if cmp_spec is not None:
        return cmp_spec
    rl = _risk_limit_recognizer(q, title)
    if rl is not None:
        return rl

    # ---- filtered count / balance ("how many loans with <field> <op> N") ---
    # A counting/aggregating question with a numeric threshold routes to a
    # filtered summary (count or balance), NOT a bar chart, so "how many loans
    # with youngest age more than 70" answers a number.
    is_count_q = bool(re.search(r"\bhow many\b|\bnumber of\b|\bcount of\b", q))
    is_balance_q = bool(re.search(r"\bhow much\b|\btotal balance\b", q))
    wants_balance_too = bool(re.search(r"\b(balance|exposure|outstanding)\b", q))
    if is_count_q or is_balance_q:
        # Support one OR MORE filters joined by "and" (numeric thresholds and a
        # categorical region value), e.g. "youngest age more than 70 and
        # geographic region south west".
        filters = _parse_filters(q, semantics, available_columns)
        # Borrower-structure intent ("how many joint borrowers"): resolve joint/sole
        # to a filter. When the field is unavailable, record the predicate as
        # UNAVAILABLE (never silently dropped).
        unavailable: List[str] = []
        bnote = ""
        bstruct = _borrower_structure_filter(q, semantics, available_columns)
        if bstruct is not None:
            bfilters, bnote = bstruct
            if bfilters:
                filters.update(bfilters)
            else:
                unavailable.append(bnote)
        if filters or unavailable:
            if is_balance_q or (is_count_q and wants_balance_too):
                metric = _balance_metric(semantics, available_columns)
                spec = MIQuerySpec(
                    intent="summary", chart_type="none", metric=metric,
                    aggregation="sum", filters=filters, title=title,
                    unavailable_filters=unavailable,
                    explanation=("Filtered balance (and loan count / share of the funded "
                                 "book) over loans matching the criteria. " + bnote).strip(),
                    output_format="table")
                base_note = "filtered_count_and_balance" if is_count_q else "filtered_balance"
            else:
                spec = MIQuerySpec(
                    intent="summary", chart_type="none", aggregation="count",
                    filters=filters, title=title, unavailable_filters=unavailable,
                    explanation="Filtered loan count over one or more criteria.",
                    output_format="table")
                base_note = "filtered_count"
            note = f"{base_note}: {bnote}" if bnote else base_note
            return spec, _det_meta("high", True, sorted(filters) or ["filtered"],
                                   note=note)

    # ---- "show/list loans where <filter>" drill-through -------------------
    # A filtered loan-level drill (NOT a grouped breakdown): "show loans with LTV
    # above 50%", "show loans where balance is below 50000". Routed to a filtered
    # loan-level table so the operator sees the matching records.
    is_show_loans = (bool(re.search(r"\b(show|list|display|drill)\b", q))
                     and bool(re.search(r"\bloans?\b", q)) and " by " not in q)
    if is_show_loans:
        d_filters = _parse_filters(q, semantics, available_columns)
        bstruct = _borrower_structure_filter(q, semantics, available_columns)
        if bstruct is not None:
            d_filters.update(bstruct[0])
        if d_filters:
            rmetric = _balance_metric(semantics, available_columns)
            spec = MIQuerySpec(
                intent="table", chart_type="none", metric=rmetric,
                aggregation="loan_level", ranking_mode="loan_level", sort_by=rmetric,
                sort_direction="desc", filters=d_filters, limit=(top_n or 50),
                output_format="table", title=title,
                explanation="Filtered loan-level drill-through.")
            return spec, _det_meta("high", True, sorted(d_filters), note="drill_filtered")

    dim_keys, dim_terms, remaining = _explicit_dimensions(q, semantics, available_columns=available_columns)
    explicit = bool(dim_keys)

    # ---- heatmap (two dimensions + metric) --------------------------------
    if "heatmap" in q:
        g_keys, g_terms, g_remaining = _explicit_dimensions(q, semantics, grouping=True, available_columns=available_columns)
        metric, _agg, matched = _detect_metric(g_remaining, semantics)
        return _build_two_dim_spec(metric, g_keys[:2], semantics, title,
                                   bool(g_keys), g_terms, has_count=("count" in matched))

    # ---- two-dimensional grouped query -> heatmap / matrix ----------------
    # "<metric> by <dim> by/and <dim>". A categorical dimension (region, broker,
    # *bucket*, …) makes this a grouped matrix (heatmap), NOT a loan-level bubble.
    # Two NUMERIC axes (e.g. ltv & age) remain a bubble (handled below).
    metric_part, seg_classes = _classify_segments(q, semantics, available_columns)
    explicit_plot = ("bubble" in q or "scatter" in q or "sized by" in q
                     or " vs " in q or "plot" in q or "against" in q)
    numeric_bubble = False
    if len(seg_classes) >= 2 and not explicit_plot and "treemap" not in q:
        n_categorical = sum(1 for c in seg_classes if c[0] == "categorical")
        if n_categorical >= 1:
            # Convert each of the two segments to a categorical dimension key
            # (numeric segments are bucketed via their bucket dimension), applying
            # the row/column convention later in the adapter.
            dims: List[str] = []
            for c in seg_classes[:2]:
                key = c[1] if c[0] == "categorical" else c[2]
                if key and key not in dims:
                    dims.append(key)
            metric, _agg, matched = _detect_metric(metric_part, semantics)
            return _build_two_dim_spec(metric, dims, semantics, title, True,
                                       [c[1] for c in seg_classes[:2]],
                                       has_count=("count" in matched))
        # All-numeric two-segment grouping -> bubble (two numeric axes + size).
        numeric_bubble = True

    # ---- ranked / "largest" queries ---------------------------------------
    is_ranking, rank_dir, rank_limit = _detect_ranking(q)
    if is_ranking and "treemap" not in q and "heatmap" not in q:
        ranked = _build_ranking_spec(q, title, rank_dir, rank_limit, top_n,
                                     semantics, available_columns)
        if ranked is not None:
            return ranked

    # ---- treemap (hierarchy + metric) -------------------------------------
    if "treemap" in q:
        g_keys, g_terms, g_remaining = _explicit_dimensions(q, semantics, grouping=True, available_columns=available_columns)
        metric, agg, _ = _detect_metric(g_remaining, semantics)
        if metric is None:
            metric, agg = _balance_metric(semantics), "sum"
        conf = "high" if len(g_keys) >= 1 else "low"
        return (MIQuerySpec(
            intent="chart", chart_type="treemap", metric=metric,
            hierarchy=g_keys[:3], aggregation=agg, top_n=top_n, title=title,
            explanation="Treemap sized by metric across a dimension hierarchy.",
            output_format="chart"),
            _det_meta(conf, bool(g_keys), g_terms))

    # ---- bubble (two NUMERIC axes + a size measure) -----------------------
    # Triggered by explicit "bubble"/"sized by", or by two numeric grouping
    # segments (e.g. "balance by ltv by age") — NEVER by a categorical pair
    # (that is a heatmap, handled above).
    by_parts = [p.strip() for p in re.split(r"\bby\b", q) if p.strip()]
    if "bubble" in q or "sized by" in q or numeric_bubble:
        x = (_age_metric(semantics, available_columns) if "age" in q
             else _balance_metric(semantics, available_columns))
        y = (_ltv_metric(semantics, available_columns) if "ltv" in q
             else _balance_metric(semantics, available_columns))
        size = _balance_metric(semantics, available_columns)
        # If the heuristic collapsed the two axes onto one field, recover the two
        # distinct numeric axes from the classified grouping segments.
        if numeric_bubble and x == y:
            nums = [c[1] for c in seg_classes if c[0] == "numeric" and c[1]]
            if len(nums) >= 2:
                x, y = nums[0], nums[1]
        # Never let two roles select the same column (would trip the loan-level
        # duplicate-column guard). Fall back to a distinct balance field.
        if size in (x, y):
            size = next((b for b in _PREFERRED_BALANCE
                         if b in _fields(semantics) and b not in (x, y)), size)
        return (MIQuerySpec(
            intent="chart", chart_type="bubble", x=x, y=y, size=size,
            aggregation="loan_level", title=title,
            explanation="Bubble chart: two numeric axes sized by a measure.",
            output_format="chart"),
            _det_meta("medium", explicit, dim_terms))

    # ---- scatter ----------------------------------------------------------
    if "scatter" in q or " vs " in q:
        if "ltv" in q and "rate" in q:
            x, y = _ltv_metric(semantics, available_columns), _rate_metric(
                semantics, available_columns)
        else:
            x = _ltv_metric(semantics, available_columns)
            y = _age_metric(semantics, available_columns)
        return (MIQuerySpec(
            intent="chart", chart_type="scatter", x=x, y=y,
            aggregation="loan_level", title=title,
            explanation="Scatter of two numeric measures.",
            output_format="chart"),
            _det_meta("medium", explicit, dim_terms))

    # ---- line (trend over time) -------------------------------------------
    is_line = ("over time" in q or "trend" in q or "monthly" in q
               or "by month" in q or "evolution" in q or "by reporting date" in q
               or "over the months" in q or "by reporting month" in q
               or "reporting month" in q or "by week" in q or "per week" in q
               or "weekly" in q or "by reporting period" in q
               or "vintage_year" in dim_keys)
    # Resolve the metric from the phrase BEFORE the first "by" (the metric side),
    # so "<metric> by <dimension>" never picks the grouping term as the metric
    # (e.g. "balance by ltv" -> metric=balance, not LTV). Fall back to the dim-
    # blanked remaining text when the metric side names nothing.
    metric, agg, _matched = _detect_metric(metric_part, semantics)
    if metric is None and not _matched:
        metric, agg, _ = _detect_metric(remaining, semantics)
    if is_line:
        x = ("origination_date" if "origination_date" in _fields(semantics)
             else None)
        if "vintage_year" in dim_keys:
            x = "vintage_year"
        # Loan/case COUNT evolutions stay a COUNT metric (not balance/sum): "loan
        # count evolution", "number of loans by reporting month", "case count by
        # week" all resolve to a governed count time-series.
        if _wants_count(q) or agg == "count":
            metric, agg = None, "count"
        elif metric is None:
            metric, agg = _balance_metric(semantics), "sum"
        return (MIQuerySpec(
            intent="chart", chart_type="line", x=x, metric=metric,
            aggregation=agg, title=title,
            explanation="Line chart of a metric over time.",
            output_format="chart"),
            _det_meta("medium" if x else "low", explicit, dim_terms))

    # ---- bar (one dimension + metric, optional top_n) ---------------------
    # Determine the dimension WITHOUT substitution.
    dimension = dim_keys[0] if dim_keys else None
    if dimension is None and len(by_parts) >= 2:
        right = by_parts[-1]
        if any(t in _REGION_GENERIC_TERMS for t in right.split()):
            # Generic region request: resolve data-aware (display field first,
            # then NUTS code fields). When no region column is available this is
            # None -> no substitution, validation then fails clearly.
            dimension = _preferred_region(semantics, available_columns)
        else:
            # Strict keyword match against the post-"by" text only (no arbitrary
            # fallback). If nothing matches, leave dimension None.
            right_tokens = tuple(t for t in right.split() if len(t) > 2)
            if right_tokens:
                dimension = _dimension(semantics, keywords=right_tokens)

    if metric is None and dimension is not None:
        # "<dimension> by <metric>" or count-by-dimension
        metric, agg, _ = _detect_metric(q, semantics)

    # Generic concentration questions may pick a sensible default dimension.
    generic = False
    if dimension is None and not explicit and any(
            w in q for w in ("concentrat", "most ", "where are", "split", "breakdown")):
        for cand in (_preferred_region(semantics, available_columns) or "geographic_region_obligor",
                     "broker_channel", "erm_product_type", "account_status"):
            if cand in _fields(semantics):
                dimension = cand
                generic = True
                break
        if metric is None:
            metric, agg = _balance_metric(semantics), "sum"

    if dimension is None and metric is None:
        # Nothing usable -> summary (no chart), low confidence.
        return (MIQuerySpec(
            intent="summary", chart_type="none", aggregation="count", title=title,
            explanation="Could not map question to a chart deterministically.",
            output_format="text"),
            _det_meta("low", explicit, dim_terms))

    # ---- single-metric KPI (a metric with NO grouping dimension) ----------
    # A bare metric ("interest rate", "total balance") is a single number — a
    # KPI/card, NEVER a one-bar chart (which would fail "bar requires a dimension"
    # and mislead the operator). Render it as a summary card + supporting table.
    #
    # Only when NO grouping was requested. If the user DID ask to group ("... by
    # region") but the dimension could not be resolved (e.g. region absent from the
    # data), fall through to the bar path so validation fails cleanly instead of
    # silently collapsing the request to a single KPI.
    grouping_requested = bool(re.search(r"\bby\b", q)) or bool(dim_terms)
    if dimension is None and metric is not None and not grouping_requested:
        weight = _default_weight(semantics, metric) if agg == "weighted_avg" else None
        return (MIQuerySpec(
            intent="summary", chart_type="none", metric=metric, aggregation=agg,
            weight_field=weight, title=title,
            explanation=f"{agg} of {metric} (single KPI; no grouping dimension "
                        "requested).",
            output_format="table"),
            _det_meta("medium" if explicit else "low", explicit, dim_terms))

    if metric is None:
        metric, agg = _balance_metric(semantics, available_columns), "sum"
    weight = _default_weight(semantics, metric) if agg == "weighted_avg" else None
    conf = "high" if explicit else ("medium" if not generic else "low")
    return (MIQuerySpec(
        intent="chart", chart_type="bar", metric=metric, dimension=dimension,
        aggregation=agg, weight_field=weight, top_n=top_n, title=title,
        explanation=f"Bar chart of {agg} metric by dimension.",
        output_format="chart"),
        _det_meta(conf, explicit, dim_terms))


def _deterministic_spec(question: str, semantics: dict,
                        available_columns=None) -> MIQuerySpec:
    """Backward-compatible wrapper returning just the spec."""
    spec, _ = _deterministic_parse(question, semantics,
                                   available_columns=available_columns)
    return spec


# --------------------------------------------------------------------------- #
# Compact catalogue (cost control)
# --------------------------------------------------------------------------- #


def _catalogue(semantics: dict) -> List[Dict[str, Any]]:
    """Full, data-free catalogue (kept for back-compat / full mode)."""
    out = []
    for key, entry in _fields(semantics).items():
        out.append({
            "field": key,
            "mi_tier": entry.get("mi_tier"),
            "business_name": entry.get("business_name", ""),
            "display_name": entry.get("display_name", ""),
            "business_description": entry.get("business_description", ""),
            "synonyms": entry.get("synonyms", []),
            "role": entry.get("role"),
            "format": entry.get("format"),
            "chartable": entry.get("chartable"),
            "allowed_aggregations": entry.get("allowed_aggregations", []),
            "allowed_chart_roles": entry.get("allowed_chart_roles", []),
        })
    return out


def _extra_keys_for_question(question: str, semantics: dict) -> List[str]:
    """Extended-tier field keys the user explicitly references (so they are
    included even in compact/core mode)."""
    q = question.lower()
    extra: List[str] = []
    for key, entry in _fields(semantics).items():
        if entry.get("mi_tier") == "core":
            continue
        terms = [entry.get("business_name", ""), key.replace("_", " ")]
        terms += list(entry.get("synonyms", []) or [])
        if any(t and t.lower() in q for t in terms):
            extra.append(key)
    return extra


def compact_catalogue(semantics: dict, mode: str = "core",
                      extra_keys: Iterable[str] = ()) -> str:
    """Compact, line-per-field catalogue. Materially smaller than the full
    JSON catalogue. Columns: key | business_name | role | format |
    allowed_aggs | chart_roles | synonyms(<=3)."""
    extra = set(extra_keys)
    lines = ["field|business_name|role|format|aggs|chart_roles|synonyms"]
    for key, entry in _fields(semantics).items():
        if mode != "full" and entry.get("mi_tier") != "core" and key not in extra:
            continue
        syn = ",".join((entry.get("synonyms", []) or [])[:3])
        lines.append("|".join([
            key,
            str(entry.get("business_name", "")),
            str(entry.get("role", "")),
            str(entry.get("format", "")),
            ",".join(entry.get("allowed_aggregations", []) or []),
            ",".join(entry.get("allowed_chart_roles", []) or []),
            syn,
        ]))
    return "\n".join(lines)


_SYSTEM_INSTRUCTIONS = (
    "You translate a natural-language Management Information (MI) question into "
    "a single JSON object describing a chart/table request (an MIQuerySpec).\n"
    "RULES:\n"
    "1. Use ONLY field keys from the catalogue for metric, dimension, x, y, "
    "size, color, dimensions, hierarchy and filter keys.\n"
    "2. Prefer mi_tier: core fields unless the user specifically asks for a "
    "more specialised field.\n"
    "3. If the user explicitly asks for a specific dimension (broker, product, "
    "region, vintage, age bucket, account status, etc.), use THAT exact field. "
    "Do NOT replace an explicitly requested field with a different available "
    "field unless the user asks for a proxy/alternative.\n"
    "4. Prefer fields whose canonical column is in the provided 'Available "
    "dataset columns'. BUT if the user explicitly asks for a field whose column "
    "is absent, still return the requested field and let validation fail — "
    "NEVER silently substitute another field.\n"
    "5. chart_type must be one of: bar, line, scatter, bubble, heatmap, "
    "treemap, none. intent must be one of: chart, table, summary.\n"
    "6. aggregation must be one of: sum, avg, weighted_avg, count, "
    "count_distinct, median, distribution, loan_level, balance_sum, and must be "
    "allowed for the chosen metric. Respect each field's allowed_chart_roles.\n"
    "7. Output STRICT JSON ONLY — no prose, no markdown fences. Always include "
    "a short 'explanation' string.\n"
)


def build_prompt(user_question: str, mi_semantics: dict,
                 available_columns: Optional[Iterable[str]] = None,
                 catalog_mode: str = "core") -> Dict[str, str]:
    """Return {"system": ..., "user": ...} prompt parts for the LLM.

    The *system* block is the stable, cacheable prefix (instructions + compact
    catalogue). The *user* block holds only dynamic, data-free content: the
    available dataset COLUMN NAMES (never values) and the question.
    """
    extra = _extra_keys_for_question(user_question, mi_semantics)
    catalogue = compact_catalogue(mi_semantics, mode=catalog_mode, extra_keys=extra)
    system = _SYSTEM_INSTRUCTIONS + "\nSemantic field catalogue:\n" + catalogue

    cols_section = ""
    if available_columns is not None:
        col_lines = "\n".join(f"- {c}" for c in sorted(available_columns))
        cols_section = ("Available dataset columns (names only):\n"
                        + col_lines + "\n\n")
    user = (cols_section
            + "User question:\n" + user_question.strip()
            + "\n\nReturn the MIQuerySpec JSON now.")
    return {"system": system, "user": user}


def parse_llm_response_to_spec(response_json: Any) -> MIQuerySpec:
    """Parse a raw LLM response (str or dict) into an MIQuerySpec."""
    if isinstance(response_json, MIQuerySpec):
        return response_json
    if isinstance(response_json, str):
        text = response_json.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        if not text.startswith("{"):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
        data = json.loads(text)
    elif isinstance(response_json, dict):
        data = response_json
    else:
        raise TypeError("response_json must be str, dict, or MIQuerySpec")
    return MIQuerySpec.from_dict(data)


# --------------------------------------------------------------------------- #
# Token / cost observability
# --------------------------------------------------------------------------- #

# Conservative USD pricing per 1,000,000 tokens (input, output).
_PRICING = {
    "haiku": (1.00, 5.00),
    "sonnet": (3.00, 15.00),
    "opus": (15.00, 75.00),
}


def _price_for_model(model: str):
    m = (model or "").lower()
    for key, price in _PRICING.items():
        if key in m:
            return price
    return None


def estimate_cost(model: str, usage: Optional[dict]) -> dict:
    """Estimate USD cost from token usage. Marks status 'unknown' when the
    model's pricing is not in the internal map."""
    out = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
        "estimated_input_cost": 0.0, "estimated_output_cost": 0.0,
        "estimated_total_cost": 0.0, "cost_estimate_status": "unknown",
    }
    if not usage:
        return out
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
    out["input_tokens"] = in_tok
    out["output_tokens"] = out_tok
    out["cache_read_tokens"] = cache_read
    out["cache_write_tokens"] = cache_write
    out["total_tokens"] = in_tok + out_tok + cache_read + cache_write
    price = _price_for_model(model)
    if price is None:
        return out  # status stays 'unknown'
    pin, pout = price
    in_cost = (in_tok / 1e6) * pin + (cache_read / 1e6) * pin * 0.1 \
        + (cache_write / 1e6) * pin * 1.25
    out_cost = (out_tok / 1e6) * pout
    out["estimated_input_cost"] = round(in_cost, 6)
    out["estimated_output_cost"] = round(out_cost, 6)
    out["estimated_total_cost"] = round(in_cost + out_cost, 6)
    out["cost_estimate_status"] = "estimated"
    return out


def _message_text(message) -> str:
    """The concatenated text of an Anthropic message's TEXT blocks.

    Robust to a leading non-text block: when extended thinking is enabled the
    first content block is a ``ThinkingBlock`` (which exposes ``.thinking``, not
    ``.text``), and tool-use blocks carry no text either. Reading
    ``message.content[0].text`` blindly then raises
    ``'ThinkingBlock' object has no attribute 'text'``. We instead walk every
    block and keep only real text, so the parser works whether or not the
    account/model returns thinking blocks.
    """
    parts = []
    for block in getattr(message, "content", None) or []:
        # Thinking blocks have no ``.text``; a genuine text block does and its
        # ``.type`` is "text". ``getattr`` keeps us safe across SDK versions.
        if getattr(block, "type", "text") == "thinking":
            continue
        txt = getattr(block, "text", None)
        if isinstance(txt, str):
            parts.append(txt)
    return "".join(parts)


def _call_llm(prompt: Dict[str, str], model: str, use_cache: bool = True):
    """Live Claude call. Returns (text, usage_dict, prompt_cache_supported)."""
    import os

    try:
        import anthropic  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when used
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic>=0.40.0"
        ) from exc

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    cache_supported = False
    message = None
    if use_cache:
        try:
            message = client.messages.create(
                model=model, max_tokens=1024, temperature=0.0,
                system=[{"type": "text", "text": prompt["system"],
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt["user"]}],
            )
            cache_supported = True
        except Exception:  # pragma: no cover - SDK without cache support
            message = None
    if message is None:
        message = client.messages.create(
            model=model, max_tokens=1024, temperature=0.0,
            system=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
        )
    text = _message_text(message)
    u = getattr(message, "usage", None)
    usage = {}
    if u is not None:
        for k in ("input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens"):
            usage[k] = getattr(u, k, 0) or 0
    return text, usage, cache_supported


def _invoke(prompt, model, llm_callable, use_cache=True):
    """Normalise a live or mocked LLM call to (text, usage, cache_supported).

    A mock ``llm_callable`` may return: str, (str, usage_dict), or
    {"text": str, "usage": dict}.
    """
    if llm_callable is not None:
        res = llm_callable(prompt)
        if isinstance(res, tuple):
            return res[0], (res[1] if len(res) > 1 else None), None
        if isinstance(res, dict) and ("text" in res or "content" in res):
            return res.get("text") or res.get("content"), res.get("usage"), None
        return res, None, None
    return _call_llm(prompt, model, use_cache=use_cache)


# --------------------------------------------------------------------------- #
# Public entry point (back-compat)
# --------------------------------------------------------------------------- #


def parse_user_question(
    user_question: str,
    semantics_path,
    model: Optional[str] = None,
    llm_enabled: bool = False,
    llm_callable=None,
) -> MIQuerySpec:
    """Translate a natural-language question into an MIQuerySpec (no repair)."""
    semantics = semantics_path if isinstance(semantics_path, dict) \
        else load_mi_semantics(semantics_path)

    if not llm_enabled and llm_callable is None:
        return _deterministic_spec(user_question, semantics)

    prompt = build_prompt(user_question, semantics)
    text, _usage, _c = _invoke(prompt, model or DEFAULT_MODEL, llm_callable)
    return parse_llm_response_to_spec(text)


# --------------------------------------------------------------------------- #
# Validate-and-repair loop (cost-hardened)
# --------------------------------------------------------------------------- #

_MISSING_COL_MARK = "not present in dataset columns"


def _missing_column_only(errors: List[str]) -> bool:
    """True if every validation error is a missing-dataset-column error."""
    return bool(errors) and all(_MISSING_COL_MARK in e for e in errors)


def _repair_prompt(base_prompt: Dict[str, str], previous_json: str,
                   errors: List[str]) -> Dict[str, str]:
    """Append the previous (invalid) JSON + validation errors to the dynamic
    user block so the model can correct itself. Still data-free; the cached
    system prefix is unchanged."""
    error_lines = "\n".join(f"- {e}" for e in errors) or "- (unparseable JSON)"
    user = (
        base_prompt["user"]
        + "\n\nYour previous answer was:\n" + previous_json
        + "\n\nThat answer FAILED validation with these errors:\n" + error_lines
        + "\n\nReturn corrected STRICT JSON only (no prose). Use only catalogue "
        "field keys and respect allowed aggregations / chart roles. Do not "
        "substitute an explicitly requested field."
    )
    return {"system": base_prompt["system"], "user": user}


def _empty_llm_meta(provider: str, model: Optional[str]) -> dict:
    return {
        "provider": provider, "model": model, "calls": 0,
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
        "estimated_input_cost": 0.0, "estimated_output_cost": 0.0,
        "estimated_total_cost": 0.0, "cost_estimate_status": "n/a",
        "prompt_cache_supported": None, "prompt_cache_used": False,
    }


def parse_with_repair(
    user_question: str,
    semantics,
    available_columns=None,
    *,
    llm_enabled: bool = False,
    model: Optional[str] = None,
    max_attempts: int = 1,
    llm_callable=None,
    provider: str = "anthropic",
    catalog_mode: str = "core",
    zero_cost_first: bool = True,
) -> Tuple[MIQuerySpec, dict]:
    """Parse a question into a validated MIQuerySpec with cost-hardened repair.

    Cost controls:
      * zero_cost_first: try the deterministic parser first; if it produces a
        confident valid spec, or a controlled missing-column failure for an
        explicit request, do NOT call the LLM.
      * never run LLM repair when validation fails only because required
        dataset columns are missing (the LLM cannot fix that without an
        unapproved substitution).
      * compact ``catalog_mode`` + cached system prefix + token/cost metadata.
    """
    if isinstance(semantics, (str, Path)):
        semantics = load_mi_semantics(semantics)
    cols = set(available_columns) if available_columns is not None else None

    use_llm = bool(llm_enabled) or (llm_callable is not None)
    llm_meta = _empty_llm_meta(provider, model if use_llm else None)

    def _accumulate(usage: Optional[dict], cache_supported, model_id):
        est = estimate_cost(model_id or "", usage)
        llm_meta["calls"] += 1
        for k in ("input_tokens", "output_tokens", "total_tokens",
                  "cache_read_tokens", "cache_write_tokens"):
            llm_meta[k] += est[k]
        for k in ("estimated_input_cost", "estimated_output_cost",
                  "estimated_total_cost"):
            llm_meta[k] = round(llm_meta[k] + est[k], 6)
        llm_meta["cost_estimate_status"] = est["cost_estimate_status"]
        if cache_supported is not None:
            llm_meta["prompt_cache_supported"] = cache_supported
        if est.get("cache_read_tokens"):
            llm_meta["prompt_cache_used"] = True

    # ---- deterministic parse (always computed; free) ----------------------
    det_spec, det_meta = _deterministic_parse(user_question, semantics,
                                              available_columns=cols)
    det_vr = validate_mi_query(det_spec, semantics, available_columns=cols)

    def _det_result(parser_detail: str, repair_skipped_reason=None) -> Tuple[MIQuerySpec, dict]:
        meta = {
            "parser_mode": "deterministic",
            "parser_mode_detail": parser_detail,
            "ok": det_vr.ok,
            "validation_errors": list(det_vr.errors),
            "validation_warnings": list(det_vr.warnings),
            "repair_attempts": 0,
            "attempts": [],
            "model": None,
            "repair_skipped_reason": repair_skipped_reason,
            "llm": _empty_llm_meta(provider, None),
            "status": ("parsed deterministically" if det_vr.ok
                       else "deterministic parse failed validation"),
        }
        meta.update({k: det_meta[k] for k in (
            "explicit_dimension_requested", "requested_dimension_terms",
            "dimension_substituted", "parser_confidence")})
        return det_spec, meta

    # No LLM at all -> deterministic only.
    if not use_llm:
        return _det_result("deterministic")

    # Zero-cost-first: avoid the LLM where the deterministic parser is decisive.
    if zero_cost_first:
        if det_vr.ok and det_meta["parser_confidence"] in ("high", "medium"):
            return _det_result("deterministic_zero_cost")
        # Explicit request that fails ONLY because the column is missing:
        # the LLM cannot fix this without substituting — fail clearly, no call.
        if (not det_vr.ok and det_meta["explicit_dimension_requested"]
                and _missing_column_only(det_vr.errors)):
            spec, meta = _det_result("validation_failed",
                                     repair_skipped_reason="missing_dataset_columns")
            meta["status"] = ("explicit request references a column missing from "
                              "the dataset; LLM repair skipped")
            return spec, meta

    # ---- LLM path (with repair loop) --------------------------------------
    base_prompt = build_prompt(user_question, semantics,
                               available_columns=cols, catalog_mode=catalog_mode)
    prompt = base_prompt
    attempts: List[dict] = []
    last_spec: Optional[MIQuerySpec] = None
    last_errors: List[str] = []
    original_error_count: Optional[int] = None
    repair_skipped_reason = None
    model_id = model or DEFAULT_MODEL

    total_tries = max(1, int(max_attempts) + 1)  # initial try + repairs
    for i in range(total_tries):
        text, usage, cache_supported = _invoke(prompt, model_id, llm_callable)
        _accumulate(usage, cache_supported, model_id)
        raw_text = text if isinstance(text, str) else json.dumps(text)
        try:
            spec = parse_llm_response_to_spec(text)
            parse_error = None
        except Exception as exc:
            spec = None
            parse_error = str(exc)

        if spec is None:
            errors = [f"could not parse model output as JSON: {parse_error}"]
            vr_ok = False
        else:
            last_spec = spec
            vr = validate_mi_query(spec, semantics, available_columns=cols)
            errors = list(vr.errors)
            vr_ok = vr.ok

        if original_error_count is None:
            original_error_count = len(errors)
        last_errors = errors
        attempts.append({"attempt": i, "ok": vr_ok,
                         "error_count": len(errors), "errors": errors})

        if vr_ok and spec is not None:
            detail = "llm" if i == 0 else "llm_repaired"
            return spec, {
                "parser_mode": "llm",
                "parser_mode_detail": detail,
                "ok": True,
                "validation_errors": [],
                "repair_attempts": i,
                "original_error_count": original_error_count,
                "attempts": attempts,
                "model": model_id,
                "repair_skipped_reason": None,
                "llm": llm_meta,
                "status": ("parsed by LLM" if i == 0
                           else f"parsed by LLM after {i} repair attempt(s)"),
            }

        # Do NOT spend repair calls on missing-column-only failures.
        if spec is not None and _missing_column_only(errors):
            repair_skipped_reason = "missing_dataset_columns"
            break

        prompt = _repair_prompt(base_prompt, raw_text, errors)

    if last_spec is None:
        last_spec = MIQuerySpec(
            intent="summary", chart_type="none", aggregation="count",
            title=user_question.strip(),
            explanation="LLM did not return a usable MIQuerySpec.",
            output_format="text")
    return last_spec, {
        "parser_mode": "llm",
        "parser_mode_detail": "validation_failed",
        "ok": False,
        "validation_errors": last_errors,
        "repair_attempts": max(0, len(attempts) - 1),
        "original_error_count": original_error_count,
        "attempts": attempts,
        "model": model_id,
        "repair_skipped_reason": repair_skipped_reason,
        "llm": llm_meta,
        "status": ("LLM output references a missing dataset column; repair skipped"
                   if repair_skipped_reason
                   else "LLM output failed validation after repair attempts"),
    }
