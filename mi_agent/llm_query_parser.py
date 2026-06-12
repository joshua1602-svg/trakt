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

    def keyword_hit(key: str, entry: dict) -> bool:
        if not keywords:
            return False
        hay = " ".join([
            key,
            str(entry.get("display_name", "")),
            str(entry.get("business_name", "")),
        ]).lower()
        return any(kw in hay for kw in keywords)

    preferred_kw: Optional[str] = None
    fallback_kw: Optional[str] = None
    preferred_any: Optional[str] = None
    fallback_any: Optional[str] = None

    for key, entry in items.items():
        if not ok(key, entry):
            continue
        if keyword_hit(key, entry):
            if is_preferred(entry):
                if preferred_kw is None:
                    preferred_kw = key
            elif fallback_kw is None:
                fallback_kw = key
        if is_preferred(entry):
            if preferred_any is None:
                preferred_any = key
        elif fallback_any is None:
            fallback_any = key

    if strict and keywords:
        return preferred_kw or fallback_kw
    return preferred_kw or fallback_kw or preferred_any or fallback_any


# Preferred balance/exposure fields (mirrors the executor's balance hierarchy).
_PREFERRED_BALANCE = ("current_outstanding_balance", "current_principal_balance",
                      "original_principal_balance")


def _balance_metric(semantics) -> Optional[str]:
    # Prefer the canonical balance hierarchy so "balance" resolves to the
    # primary exposure field rather than an alphabetically-earlier keyword hit
    # such as ``arrears_balance``.
    fields = _fields(semantics)
    for key in _PREFERRED_BALANCE:
        if key in fields:
            return key
    return find_field(semantics, role="metric", fmt="currency",
                      keywords=("balance", "outstanding", "principal"))


def _ltv_metric(semantics) -> Optional[str]:
    return find_field(semantics, role="metric", fmt="percent",
                      keywords=("ltv", "loan_to_value"))


def _age_metric(semantics) -> Optional[str]:
    return find_field(semantics, role="metric", fmt="integer", keywords=("age",))


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
    "ltv bucket": "ltv_bucket",
    "ltv band": "ltv_bucket",
    "ticket size": "ticket_bucket",
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


def _detect_metric(text: str, semantics: dict) -> Tuple[Optional[str], str, List[str]]:
    """Return (metric_key, aggregation, matched_terms) from free text."""
    matched: List[str] = []
    for term, token in _METRIC_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", text):
            key, agg = _resolve_metric(token, semantics)
            matched.append(term)
            return key, agg, matched
    return None, "sum", matched


def _detect_top_n(q: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)\b", q)
    return int(m.group(1)) if m else None


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

    dim_keys, dim_terms, remaining = _explicit_dimensions(q, semantics, available_columns=available_columns)
    explicit = bool(dim_keys)

    # ---- heatmap (two dimensions + metric) --------------------------------
    if "heatmap" in q:
        g_keys, g_terms, g_remaining = _explicit_dimensions(q, semantics, grouping=True, available_columns=available_columns)
        metric, agg, _ = _detect_metric(g_remaining, semantics)
        if metric is None:
            metric = _balance_metric(semantics)
        agg = "weighted_avg" if (metric and _fields(semantics).get(metric, {})
                                 .get("format") == "percent") else "avg"
        dims = g_keys[:2]
        conf = "high" if len(dims) >= 2 else "low"
        return (MIQuerySpec(
            intent="chart", chart_type="heatmap", metric=metric,
            dimensions=dims, aggregation=agg, title=title,
            explanation="Heatmap of metric intensity across two dimensions.",
            output_format="chart"),
            _det_meta(conf, bool(g_keys), g_terms))

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

    # ---- bubble ("<a> by <b> by <c>" or "... sized by ...") ---------------
    by_parts = [p.strip() for p in re.split(r"\bby\b", q) if p.strip()]
    if "bubble" in q or "sized by" in q or len(by_parts) >= 3:
        x = _age_metric(semantics) if "age" in q else _balance_metric(semantics)
        y = _ltv_metric(semantics) if "ltv" in q else _balance_metric(semantics)
        size = _balance_metric(semantics)
        return (MIQuerySpec(
            intent="chart", chart_type="bubble", x=x, y=y, size=size,
            aggregation="loan_level", title=title,
            explanation="Bubble chart: two numeric axes sized by a measure.",
            output_format="chart"),
            _det_meta("medium", explicit, dim_terms))

    # ---- scatter ----------------------------------------------------------
    if "scatter" in q or " vs " in q:
        if "ltv" in q and "rate" in q:
            x, y = _ltv_metric(semantics), find_field(
                semantics, role="metric", fmt="percent", keywords=("rate",))
        else:
            x = _ltv_metric(semantics)
            y = _age_metric(semantics)
        return (MIQuerySpec(
            intent="chart", chart_type="scatter", x=x, y=y,
            aggregation="loan_level", title=title,
            explanation="Scatter of two numeric measures.",
            output_format="chart"),
            _det_meta("medium", explicit, dim_terms))

    # ---- line (trend over time) -------------------------------------------
    is_line = ("over time" in q or "trend" in q or "monthly" in q
               or "by month" in q or "vintage_year" in dim_keys)
    metric, agg, _ = _detect_metric(remaining, semantics)
    if is_line:
        x = ("origination_date" if "origination_date" in _fields(semantics)
             else None)
        if "vintage_year" in dim_keys:
            x = "vintage_year"
        if metric is None:
            metric, agg = _balance_metric(semantics), "sum"
        return (MIQuerySpec(
            intent="chart", chart_type="line", x=x, metric=metric,
            aggregation=agg if agg != "count" else "sum", title=title,
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

    if metric is None:
        metric, agg = _balance_metric(semantics), "sum"
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
    text = message.content[0].text
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
