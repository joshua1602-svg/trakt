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
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .mi_query_spec import MAX_MEASURES, MIQuerySpec
from .mi_query_validator import load_mi_semantics, validate_mi_query
from . import statistic as _statistic
from . import population as _population_mod

logger = logging.getLogger(__name__)

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
    # The adjectival form. "largest regional concentration" named the governed
    # region dimension and matched nothing, so it fell through to a loan-level
    # ranking table — a list of individual loans in answer to a question about
    # regions.
    "regional": "geographic_region_obligor",
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
    # Borrower-type family (single vs joint). Resolved data-aware via
    # _preferred_borrower_dim (materialised borrower_type first, then the
    # legacy borrower_structure band) — see _BORROWER_GENERIC_TERMS.
    "borrower types": "borrower_type",
    "borrower type": "borrower_type",
    "borrower structure": "borrower_type",
    "applicant type": "borrower_type",
    "single vs joint": "borrower_type",
    "joint vs single": "borrower_type",
    "single or joint": "borrower_type",
    "sole or joint": "borrower_type",
    "joint or single": "borrower_type",
    "joint or sole": "borrower_type",
}

# Single-word tokens that must NOT be auto-mapped from registry synonyms: they
# are either too generic (would hijack unrelated questions) or collide with
# other grammar (ranking / count / summary intents, metric buckets). Curated
# EXPLICIT_DIMENSION_TERMS still map them where a specific meaning is intended.
_GENERIC_DIM_TOKENS = frozenset({
    "type", "types", "status", "band", "bands", "bucket", "buckets", "date",
    "year", "name", "code", "id", "value", "amount", "rate", "balance", "age",
    "ltv", "region", "regions", "geography", "geographic", "group", "total",
    "class", "level", "score", "stage", "grade", "term",
    # collide with other grammar / are ambiguous on their own:
    "ranking",     # top-N ranking grammar
    "portfolio",   # "portfolio summary" / "the portfolio"
    "borrowers",   # count intent ("how many borrowers")
    "charge",      # "early repayment charge" etc.
})


def _registry_dimension_terms(semantics: dict) -> Dict[str, str]:
    """Business synonyms / names for every dimension-role field, so the parser
    recognises a term the MOMENT it is added to the registry (no code change).

    Curated ``EXPLICIT_DIMENSION_TERMS`` override these; an ambiguous synonym
    (mapping to more than one dimension) and over-generic single tokens
    (``_GENERIC_DIM_TOKENS``) are dropped so registry vocabulary can never
    hijack an unrelated question. Multi-word phrases are always safe to add."""
    out: Dict[str, str] = {}
    ambiguous: set = set()
    for key, entry in _fields(semantics).items():
        if entry.get("role") != "dimension":
            continue
        phrases = list(_synonyms(entry))
        for name in (entry.get("business_name"), entry.get("display_name")):
            if name:
                phrases.append(str(name))
        phrases.append(key.replace("_", " "))
        for phrase in phrases:
            p = str(phrase).strip().lower()
            if len(p) < 3:
                continue
            if " " not in p and p in _GENERIC_DIM_TOKENS:
                continue
            existing = out.get(p)
            if existing is not None and existing != key:
                ambiguous.add(p)
            else:
                out[p] = key
    for p in ambiguous:
        out.pop(p, None)
    return out


# Generic region terms resolved by data-aware preference (see _preferred_region).
_REGION_GENERIC_TERMS = {"region", "regions", "regional", "geography",
                         "geographic", "geographic region"}
# Borrower-type terms resolved by data-aware preference (see
# _preferred_borrower_dim). borrower_type is the dimension the funded prep
# actually materialises; borrower_structure is a legacy band kept for datasets
# that carry it.
_BORROWER_GENERIC_TERMS = {"borrower type", "borrower types", "borrower structure",
                           "applicant type", "single vs joint", "joint vs single",
                           "single or joint", "sole or joint", "joint or single",
                           "joint or sole"}
_BORROWER_DIM_PREFERENCE = ("borrower_type", "borrower_structure")


def _preferred_borrower_dim(semantics: dict, available_columns=None) -> Optional[str]:
    """Pick the single-vs-joint dimension: the materialised ``borrower_type``
    first, then ``borrower_structure``, preferring one this book actually has.

    When the book has NEITHER, this returns the first registry-known choice
    rather than ``None``. That distinction is the whole defect it was written
    with:

    ``None`` meant the caller's ``continue`` dropped the term WITHOUT masking
    the words it matched, so "balance by borrower type" left "type" lying in the
    question for a shorter, weaker term to claim — and it was claimed, by
    ``amortisation_type``. The question was silently rewritten to fit the book.
    Thirteen calibration cases were that one line.

    Returning the first choice makes a generic term behave exactly like an
    ordinary one: "balance by broker" resolves to ``broker_channel``, the column
    is absent, and the executor refuses NAMING the field the user asked for.
    Preferring an available synonym stays correct — the alternatives all mean the
    same thing — but preferring a DIFFERENT concept never was."""
    fields = _fields(semantics)
    cols = set(available_columns) if available_columns is not None else None
    known = [k for k in _BORROWER_DIM_PREFERENCE if fields.get(k)]
    for key in known:
        entry = fields[key]
        if cols is None or entry.get("canonical_field", key) in cols:
            return key
    return known[0] if known else None
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
    known = [k for k in _REGION_PREFERENCE if k in fields]
    if cols is not None:
        # Data-aware: prefer a region field whose column is actually present.
        for key in known:
            entry = fields.get(key) or {}
            if entry.get("canonical_field", key) in cols:
                return key
        # None present. Return the FIRST KNOWN choice, not None.
        #
        # The intent recorded here was always "fail clearly rather than
        # substitute an absent field" — but returning None achieved the
        # opposite. The caller's ``continue`` dropped the term without masking
        # the words it matched, leaving them for a shorter, weaker term to
        # claim. Keeping the concept is what makes the failure clear: the
        # executor then refuses NAMING the field the user asked for, exactly as
        # it does for any ordinary absent dimension.
        return known[0] if known else None
    # No column context: fall back to registry presence (parse-time default).
    return known[0] if known else None

# Metric NL terms -> resolver. Order matters (longer/more-specific first).
_METRIC_TERMS = (
    ("weighted average ltv", "ltv"),
    ("loan to value", "ltv"),
    ("ltv", "ltv"),
    ("outstanding balance", "balance"),
    ("balance", "balance"),
    ("exposure", "balance"),
    # "size" is a MEASURE noun when it is what is being averaged and a bucket
    # DIMENSION when it is what the answer is grouped by. The dimension reading
    # is masked upstream when an aggregator precedes it (see
    # ``_explicit_dimensions``); these entries supply the measure reading, so
    # "average loan size" is the mean balance rather than an unmapped question.
    ("loan size", "balance"),
    ("ticket size", "balance"),
    ("deal size", "balance"),
    ("average size", "balance"),
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

    Governed SCOPE phrases are masked first: "for the acquired portfolio" names
    the population being reported on, and was being grouped BY
    ``acquired_portfolio_id`` — a real registry field whose synonym the phrase
    happens to match. The reference field keeps its meaning everywhere else;
    it simply cannot be reached through a scope clause.

    Returns (dimension_keys, matched_terms, remaining_text). Only terms whose
    target key exists in the registry are honoured; matched spans are removed
    from ``remaining_text`` so metric detection does not re-trip on them.

    Generic region terms ("region", "geography", ...) resolve data-aware via
    _preferred_region (readable display field first, then NUTS code fields).

    NOTE: ``remaining_text`` is derived from the ORIGINAL question, not the
    masked copy, so metric detection downstream still sees every word the user
    wrote.

    ``grouping=True`` enables a small set of context-only bucketing terms (a
    bare "age" axis -> age_bucket) used by heatmap/treemap.
    """
    fields = _fields(semantics)
    # Registry-derived dimension synonyms first, then the curated map on top so
    # curated disambiguation always wins. This makes a synonym added to the
    # semantic registry immediately understood by the chat, without a code edit.
    terms_map = _registry_dimension_terms(semantics)
    terms_map.update(EXPLICIT_DIMENSION_TERMS)
    if grouping:
        # In a grouping chart (heatmap/treemap) a bare "age" axis means the
        # age band, not the numeric age metric. Same idea for the other
        # bucketable measures, but only when their bucket dimension exists.
        for bare, bucket in (("age", "age_bucket"),):
            if bucket in fields and bare not in terms_map:
                terms_map[bare] = bucket
    from .portfolio_lens import mask_scope_phrases  # local: avoids a cycle

    remaining = q
    # Dimension terms are searched in a copy with governed scope phrases blanked.
    # ``mask_scope_phrases`` preserves offsets, so every index below is equally
    # valid against ``remaining``, which stays faithful to what the user wrote.
    search = mask_scope_phrases(q)
    # An aggregation qualifier standing in front of a BUCKET synonym names a
    # measure, not a grouping — you cannot take the average of a band. Before
    # this, "average loan size" matched the ticket_bucket synonym "loan size",
    # attached the dimension, found no metric and defaulted to balance, so a
    # question asking for ONE number answered with a breakdown across ten. The
    # span is blanked in the SEARCH copy only, so metric detection downstream
    # still sees every word the user wrote. A grouped phrasing keeps its
    # dimension because "by" precedes the term rather than an aggregator.
    for _term in sorted((t for t, k in terms_map.items() if _is_bucket_dim(k)),
                        key=len, reverse=True):
        search = re.sub(
            r"\b(average|avg|mean|median|typical|total|sum of)\s+(?:the\s+)?"
            + re.escape(_term) + r"\b",
            lambda m: " " * len(m.group(0)), search)
    found: List[Tuple[int, str, str]] = []  # (position, key, term)
    for term in sorted(terms_map, key=len, reverse=True):
        if term in _REGION_GENERIC_TERMS:
            key = _preferred_region(semantics, available_columns)
        elif term in _BORROWER_GENERIC_TERMS:
            key = _preferred_borrower_dim(semantics, available_columns)
        else:
            key = terms_map[term]
        if not key or key not in fields:
            continue
        pat = r"\b" + re.escape(term) + r"\b"
        m = re.search(pat, search)
        if m:
            found.append((m.start(), key, term))
            blank = " " * (m.end() - m.start())
            remaining = remaining[:m.start()] + blank + remaining[m.end():]
            search = search[:m.start()] + blank + search[m.end():]
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


# Over-generic single tokens the registry-driven metric pass must NOT bind on
# its own (they are handled by the curated grammar / default resolution and
# would otherwise let a measure synonym hijack an unrelated question).
_GENERIC_METRIC_TOKENS = {
    "balance", "value", "amount", "rate", "count", "age", "ltv", "exposure",
    "total", "sum", "principal", "interest", "loan", "loans", "mortgage",
    "income", "margin", "ratio", "period", "number", "term",
}


def _registry_metric_terms(semantics: dict) -> Dict[str, str]:
    """Business synonyms / names for every measure-role field, so a governed
    metric the parser would otherwise not recognise (e.g. 'valuation', 'original
    balance') resolves to its OWN field instead of silently falling back to the
    default balance metric. Mirrors :func:`_registry_dimension_terms`:
    multi-word phrases are always safe; over-generic single tokens
    (``_GENERIC_METRIC_TOKENS``) and ambiguous phrases are dropped."""
    out: Dict[str, str] = {}
    ambiguous: set = set()
    for key, entry in _fields(semantics).items():
        if entry.get("role") not in ("metric", "measure"):
            continue
        phrases = list(_synonyms(entry))
        for name in (entry.get("business_name"), entry.get("display_name")):
            if name:
                phrases.append(str(name))
        phrases.append(key.replace("_", " "))
        for phrase in phrases:
            p = str(phrase).strip().lower()
            if len(p) < 3:
                continue
            if " " not in p and p in _GENERIC_METRIC_TOKENS:
                continue
            existing = out.get(p)
            if existing is not None and existing != key:
                ambiguous.add(p)
            else:
                out[p] = key
    for p in ambiguous:
        out.pop(p, None)
    return out


# Words that may legitimately sit on the metric side of "<metric> by <dim>"
# without naming a measure: request verbs, determiners, aggregation qualifiers,
# chart words and connectives. Anything left over after removing these AND every
# registry term is a measure the user named that this dataset does not have.
_METRIC_SIDE_STOPWORDS = frozenset({
    # request framing
    "show", "shows", "display", "give", "gimme", "get", "list", "plot", "chart",
    "draw", "render", "provide", "produce", "see", "view", "want", "need",
    "please", "can", "you", "could", "would", "tell", "what", "whats", "which",
    "how", "much", "many", "who", "where", "when", "why", "is", "are", "was",
    "were", "do", "does", "did", "has", "have", "had", "the", "a", "an", "my",
    "our", "me", "us", "it", "its", "this", "that", "these", "those", "there",
    "of", "for", "in", "on", "at", "to", "from", "with", "and", "or", "as",
    "per", "each", "every", "all", "any", "some", "across", "over", "under",
    "between", "within", "into", "out", "up", "down", "look", "looking",
    # aggregation / shape qualifiers (resolved separately by _aggregation_intent)
    "sum", "total", "totals", "aggregate", "average", "avg", "mean", "median",
    "weighted", "simple", "unweighted", "plain", "straight", "count", "counts",
    "number", "numbers", "distribution", "breakdown", "break", "broken", "split",
    "grouped", "group", "summary", "summarise", "summarize", "overview",
    # chart / output words
    "bar", "line", "pie", "table", "graph", "heatmap", "treemap", "bubble",
    "scatter", "map", "matrix", "grid", "trend", "series", "chartd",
})

# Words that FRAME an analysis rather than name a measure. A question built only
# from these legitimately has no metric of its own and defaults to the balance
# measure — that is long-standing, governed behaviour ("concentration by region",
# "coverage by borrower type", "regions with the most loans"), not a silent
# substitution, because the user never named a different measure to substitute
# for. Kept separate from the framing/stopword list above so the distinction
# stays explicit: these are *analytical* words, not grammatical filler.
_ANALYTICAL_FRAMING_WORDS = frozenset({
    "concentration", "concentrations", "concentrated", "concentrate",
    "coverage", "covered", "data", "quality", "completeness", "missing",
    "most", "least", "largest", "biggest", "smallest", "highest", "lowest",
    "top", "bottom", "rank", "ranked", "ranking", "leading",
    # "best" and "worst" are NOT in this set. "largest"/"highest"/"top" name a
    # DIRECTION on a measure the question still has to supply; "best" names a
    # value judgement whose basis the question does not give, and absorbing it
    # as framing is what let "show best brokers" answer with balance by broker.
    # Left out of the set they survive as residue, which triggers the existing
    # unresolved-metric clarify — but only when nothing else resolved, so
    # "which broker has the worst arrears" still answers on arrears.
    "mix", "composition", "profile", "position", "snapshot", "spread",
    "exposure", "exposures", "book", "portfolio", "portfolios",
})

#: Minimum token length for a residue word to count as a named measure. Filters
#: stray short tokens without needing an exhaustive stopword list.
_METRIC_RESIDUE_MIN_LEN = 3


def _metric_side_residue(metric_part: str, semantics: dict,
                         available_columns=None) -> Optional[str]:
    """The measure the user named that this dataset does not carry, if any.

    ``_deterministic_parse`` used to default an unresolved metric to the balance
    field, so "show me the unicorn ratio by region" answered as *balance by
    region* with ``ok:true`` — a confident answer to a question nobody asked.
    Refusing requires distinguishing two cases:

    * **no measure named** ("breakdown by region") — defaulting to balance is the
      documented, expected behaviour and must not change;
    * **a measure named that does not resolve** ("unicorn ratio by region") — the
      request must be refused, naming the term.

    This returns the residual phrase for the second case and ``None`` for the
    first. It removes request framing, aggregation qualifiers and chart words,
    then every governed metric and dimension term the registry knows. Whatever
    survives is a noun phrase the user supplied and the registry cannot map.
    """
    text = (metric_part or "").strip().lower()
    if not text:
        return None
    # Blank out every registry term (longest first, so multi-word phrases win)
    # plus the hard-coded explicit dimension vocabulary.
    terms = set(_registry_metric_terms(semantics)) | set(_registry_dimension_terms(semantics))
    terms |= set(EXPLICIT_DIMENSION_TERMS)
    terms |= set(_NUMERIC_AXIS_BUCKET)
    terms |= _REGION_GENERIC_TERMS | _BORROWER_GENERIC_TERMS
    for term in sorted(terms, key=len, reverse=True):
        if len(term) < 2:
            continue
        text = re.sub(r"\b" + re.escape(term) + r"\b", " ", text)
    # Also blank the generic measure tokens: they are recognised metric words
    # even when too ambiguous to map to one field on their own.
    for token in _GENERIC_METRIC_TOKENS:
        text = re.sub(r"\b" + re.escape(token) + r"\b", " ", text)
    residue = [w for w in re.findall(r"[a-z][a-z\-']*", text)
               if w not in _METRIC_SIDE_STOPWORDS
               and w not in _ANALYTICAL_FRAMING_WORDS
               and len(w) >= _METRIC_RESIDUE_MIN_LEN]
    if not residue:
        return None
    return " ".join(residue)


def _detect_metric(text: str, semantics: dict) -> Tuple[Optional[str], str, List[str]]:
    """Return (metric_key, aggregation, matched_terms) from free text.

    An explicit aggregation qualifier ("average"/"weighted average"/"total") in the
    same phrase overrides the metric's default aggregation, so "average loan
    balance" means the mean balance, not the total.

    Resolution order: the curated ``_METRIC_TERMS`` grammar governs the core
    measures (balance / ltv / rate / age / count) and always wins for its tokens.
    A registry-driven pass then recognises any OTHER governed measure by its
    business synonym (longest phrase first) so a requested metric is never
    silently substituted with the default balance.
    """
    matched: List[str] = []
    intent = _aggregation_intent(text)
    reg_terms = _registry_metric_terms(semantics)

    def _resolve_registry(term: str) -> Tuple[Optional[str], str]:
        key = reg_terms[term]
        entry = _fields(semantics).get(key, {})
        default_agg = entry.get("default_aggregation") or (
            "weighted_avg" if entry.get("format") == "percent" else "sum")
        return key, _apply_agg_intent(key, default_agg, intent, semantics)

    # 1) Registry MULTI-WORD phrases first (longest, most specific): a governed
    #    measure named by a phrase — "original balance", "property value" — must
    #    beat a curated single token it happens to contain (e.g. "balance").
    multi = sorted((t for t in reg_terms if " " in t), key=len, reverse=True)
    for term in multi:
        if re.search(r"\b" + re.escape(term) + r"\b", text):
            key, agg = _resolve_registry(term)
            matched.append(term)
            return key, agg, matched
    # 2) Curated grammar — the core measures and their disambiguation.
    for term, token in _METRIC_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", text):
            key, agg = _resolve_metric(token, semantics)
            if token != "count":
                agg = _apply_agg_intent(key, agg, intent, semantics)
            matched.append(term)
            return key, agg, matched
    # 3) Registry SINGLE-WORD synonyms for any remaining governed measure.
    single = sorted((t for t in reg_terms if " " not in t), key=len, reverse=True)
    for term in single:
        if re.search(r"\b" + re.escape(term) + r"\b", text):
            key, agg = _resolve_registry(term)
            matched.append(term)
            return key, agg, matched
    return None, "sum", matched


# --------------------------------------------------------------------------- #
# P1E — the measure SET a question names
# --------------------------------------------------------------------------- #
# "Give me the balance, loan count, weighted-average LTV and rate for London" is
# ONE analytical request carrying four governed measures over one population —
# not four questions, and not an excuse to answer one of them.
#
# This reuses ``_detect_metric``'s vocabularies rather than adding a parallel
# one: the same curated grammar, the same registry synonyms, the same
# aggregation qualifiers. The only difference is that it collects EVERY match in
# order of appearance instead of returning the first.


def _local_aggregation_intent(text: str, start: int) -> Optional[str]:
    """The aggregation qualifier attached to THIS measure, not to the sentence.

    "weighted-average LTV and average borrower age" carries two different
    qualifiers; reading the whole sentence would give both measures whichever
    one matched first.
    """
    window = text[max(0, start - _AGG_QUALIFIER_WINDOW):start]
    # Stop at a conjunction or comma: the qualifier before "and" belongs to the
    # PREVIOUS measure ("total balance and LTV" does not make LTV a sum).
    for separator in (",", " and ", " plus ", ";"):
        cut = window.rfind(separator)
        if cut != -1:
            window = window[cut + len(separator):]
    return _aggregation_intent(window)


#: How far back to look for a measure's own aggregation qualifier.
_AGG_QUALIFIER_WINDOW = 40

#: How a reader names the loan-count measure inside a multi-measure request.
_COUNT_MEASURE_RE = re.compile(
    r"\b(?:loan\s+count|number\s+of\s+(?:loans|cases|accounts|mortgages)|"
    r"no\.?\s+of\s+(?:loans|cases|accounts)|count\s+of\s+(?:loans|cases|accounts)|"
    r"how\s+many\s+(?:loans|cases|accounts)|loan\s+numbers)\b", re.I)


#: Introduces a GROUPING clause. Everything from here to the end of the clause
#: names axes, not measures — "balance by region and age bucket" measures one
#: thing across two, and reading "age" there as a second measure would turn a
#: governed two-dimensional breakdown into a spurious multi-measure request.
_GROUPING_CLAUSE_RE = re.compile(
    r"\b(?:split\s+by|grouped\s+by|broken\s+down\s+by|by|per|across)\b", re.I)

#: Where a grouping clause ends and ordinary sentence resumes.
_GROUPING_CLAUSE_END_RE = re.compile(
    r"[,?.;:]|\b(?:where|for|with|in|over|under|above|below|between|show|give)\b",
    re.I)


#: A measure word carrying one of these suffixes names a BUCKETED DIMENSION —
#: "age bucket", "LTV band" — so "which age bucket has the largest balance" is a
#: ranking over one measure, not a request for two.
_DIMENSION_SUFFIX_RE = re.compile(
    r"^\s*(?:bucket|band|range|group|grouping|segment|category|categories|"
    r"cohort|tier|type)s?\b", re.I)


def _grouping_regions(text: str) -> List[Tuple[int, int]]:
    """Spans of ``text`` that name grouping AXES rather than measures."""
    regions: List[Tuple[int, int]] = []
    for match in _GROUPING_CLAUSE_RE.finditer(text):
        end = _GROUPING_CLAUSE_END_RE.search(text, match.end())
        regions.append((match.end(), end.start() if end else len(text)))
    return regions


def _measure_hits(text: str, semantics: dict, available_columns=None
                  ) -> List[Tuple[int, int, str, str]]:
    """Non-overlapping ``(start, end, semantic_key, default_aggregation)`` hits.

    The single place the measure vocabulary is applied to a question. Both the
    measure SET and the unresolved-slot guard read it, so the guard can never
    disagree with the parser about which words were understood as measures.
    """
    reg_terms = _registry_metric_terms(semantics)
    fields = _fields(semantics)
    columns = set(available_columns) if available_columns is not None else None

    # P0 already decides which measure words in a question are MEASURES and
    # which are grammar. Reusing its two exclusions — rather than re-deriving
    # them — is what keeps the parser and the P0 ledger from disagreeing about
    # the same sentence.
    from .execution_receipt import _is_filter_subject  # local: avoids a cycle

    grouping = _grouping_regions(text)

    #: (start, end, semantic_key, default_aggregation)
    hits: List[Tuple[int, int, str, str]] = []

    def _record(match, key: Optional[str], default_agg: str) -> None:
        if not key:
            return
        entry = fields.get(key, {}) or {}
        canonical = entry.get("canonical_field", key)
        if columns is not None and key != "loan_count" and canonical not in columns:
            return
        # "balance below 75% LTV" measures balance and FILTERS on LTV; "balance
        # BY region" measures balance and GROUPS by region. Neither second word
        # is a measure, and reading it as one turns a good filtered answer into
        # a spurious multi-measure request.
        if _is_filter_subject(text, match.start(), match.end()):
            return
        if any(start <= match.start() < end for start, end in grouping):
            return
        if _DIMENSION_SUFFIX_RE.match(text[match.end():match.end() + 16]):
            return
        hits.append((match.start(), match.end(), key, default_agg))

    # 1) Registry multi-word phrases (longest first) — a phrase measure must beat
    #    a curated token it contains.
    for term in sorted((t for t in reg_terms if " " in t), key=len, reverse=True):
        for match in re.finditer(r"\b" + re.escape(term) + r"\b", text):
            key = reg_terms[term]
            entry = fields.get(key, {})
            _record(match, key, entry.get("default_aggregation")
                    or ("weighted_avg" if entry.get("format") == "percent" else "sum"))
    # 2) The curated grammar — the core measures.
    for term, token in _METRIC_TERMS:
        for match in re.finditer(r"\b" + re.escape(term) + r"\b", text):
            key, default_agg = _resolve_metric(token, semantics)
            _record(match, key or ("loan_count" if token == "count" else None),
                    "count" if token == "count" else default_agg)
    # 2b) Count synonyms. "number of loans" and "loan count" are the same governed
    #     measure; the curated grammar carries only the bare token "count",
    #     which a CFO rarely writes. Scoped to the measure set so
    #     ``_detect_metric``'s single-measure behaviour is untouched.
    for match in _COUNT_MEASURE_RE.finditer(text):
        _record(match, "loan_count", "count")
    # 3) Registry single-word synonyms for anything still unnamed.
    for term in sorted((t for t in reg_terms if " " not in t), key=len, reverse=True):
        for match in re.finditer(r"\b" + re.escape(term) + r"\b", text):
            key = reg_terms[term]
            entry = fields.get(key, {})
            _record(match, key, entry.get("default_aggregation")
                    or ("weighted_avg" if entry.get("format") == "percent" else "sum"))

    # Overlapping spans: keep the longest match at each position, so "loan to
    # value" is one measure rather than also matching "value".
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    chosen: List[Tuple[int, int, str, str]] = []
    for hit in hits:
        if any(hit[0] < c[1] and c[0] < hit[1] for c in chosen):
            continue
        chosen.append(hit)
    return chosen


def detect_measure_set(text: str, semantics: dict, available_columns=None, *,
                       with_spans: bool = False):
    """Every governed measure the question names, in order, with its aggregation.

    Returns ``[]`` when fewer than two distinct measures are named, so a
    single-measure question keeps exactly the parse it has today and this
    function can never change it.

    ``with_spans`` additionally returns the text spans the measures consumed.
    A caller masks them before resolving dimensions and filters, so "average
    borrower AGE" as a measure cannot also be read as a grouping by age band —
    the same consume-the-span discipline the single-measure parser follows.
    """
    # P1N: "exposure-weighted borrower age" names one measure and a weighting,
    # not two measures. Masked before the hits are taken, offsets preserved.
    text = _statistic.mask_statistic_phrases(text)
    chosen = _measure_hits(text, semantics, available_columns)

    measures: List[Dict[str, str]] = []
    spans: List[Tuple[int, int]] = []
    seen: set = set()
    for start, end, key, default_agg in sorted(chosen):
        if key in seen:
            continue
        seen.add(key)
        spans.append((start, end))
        if key == "loan_count":
            measures.append({"field": "loan_count", "aggregation": "count"})
            continue
        aggregation = _apply_agg_intent(
            key, default_agg, _local_aggregation_intent(text, start), semantics)
        measures.append({"field": key, "aggregation": aggregation})

    if len(measures) < 2:
        measures, spans = [], []
    return (measures, tuple(spans)) if with_spans else measures


#: Where a coordinated measure list stops. Punctuation ends the clause; a
#: preposition introduces the SCOPE or the GROUPING rather than another measure
#: ("... and weighted-average LTV **by** region", "... **for** the London book").
_MEASURE_LIST_STOP_RE = re.compile(
    r"[?.;:!]|\b(?:by|for|in|across|between|over|under|where|with|per|split|"
    r"grouped|broken|during|since|versus|vs)\b", re.I)

#: Words that can occupy a slot without naming a measure. A trailing courtesy
#: ("..., please") is not a measure the agent failed to understand.
_MEASURE_SLOT_FILLER = frozenset({
    "please", "thanks", "thank you", "ta", "cheers", "as well", "too", "also",
    "and", "or", "etc", "etc.", "the", "a", "an", "me", "us", "it", "that",
})

_SLOT_SPLIT_RE = re.compile(r",|\band\b|&", re.I)

#: A measure slot is a NOUN PHRASE — "weighted average rate", "loan count". A
#: slot carrying a question word or a finite verb is a second CLAUSE, not a
#: measure the parser failed to understand: "the largest exposure and what share
#: of the book is it" asks two things, and the second is a governed share
#: question rather than an unrecognised measure name.
_SLOT_IS_A_CLAUSE_RE = re.compile(
    r"\b(?:what|which|who|whose|how|why|when|where|is|are|was|were|be|been|"
    r"do|does|did|has|have|had|can|could|will|would|should|make|makes|made|"
    r"give|gives|show|shows|tell|tells)\b", re.I)


def unresolved_measure_slots(text: str, semantics: dict,
                             available_columns=None) -> Tuple[str, ...]:
    """Slots in the question's measure list that named no governed measure.

    A CFO writes measures as a coordinated list — "balance, loan count,
    weighted-average LTV and weighted-average rate". If the vocabulary
    understands three of those four, answering the three and saying nothing
    about the fourth is a silent omission, and a silent omission is the one
    outcome this product must never produce.

    The guard is deliberately STRUCTURAL rather than lexical: it does not need
    to know what the unrecognised words mean, only that a slot of the same list
    resolved to nothing. That is why it also covers vocabulary this parser has
    not learnt yet, instead of needing a new pattern per missing synonym.

    Bounded to the list itself. The region runs from the first recognised
    measure to the end of that clause, so words before the list (a leading scope
    clause) and after it (a grouping or a filter) are never mistaken for
    measures.
    """
    hits = _measure_hits(text, semantics, available_columns)
    if not hits:
        return ()          # no measure list was recognised at all — not ours
    spans = [(h[0], h[1]) for h in hits]
    start = min(s for s, _ in spans)
    last = max(e for _, e in spans)

    stop = len(text)
    tail = _MEASURE_LIST_STOP_RE.search(text, last)
    if tail is not None:
        stop = tail.start()

    # Slot boundaries come from the separators' own offsets, so a slot's text and
    # its position can never disagree — the check below is "did a measure span
    # fall inside THIS slot", and an off-by-one there would invent a finding.
    bounds: List[Tuple[int, int]] = []
    cursor = start
    for sep in _SLOT_SPLIT_RE.finditer(text, start, stop):
        bounds.append((cursor, sep.start()))
        cursor = sep.end()
    bounds.append((cursor, stop))

    unresolved: List[str] = []
    for slot_start, slot_end in bounds:
        if any(slot_start < e and b < slot_end for b, e in spans):
            continue        # this slot named a measure
        piece = text[slot_start:slot_end]
        if _SLOT_IS_A_CLAUSE_RE.search(piece):
            continue        # a second question, not a measure name
        residue = [w for w in re.findall(r"[a-z][a-z'-]*", piece.lower())
                   if w not in _MEASURE_SLOT_FILLER]
        if not residue or len(residue) > 6:
            continue        # filler, or too long a clause to be a measure slot
        unresolved.append(" ".join(piece.split()).strip(" ,"))
    return tuple(dict.fromkeys(u for u in unresolved if u))


def _mask_spans(text: str, spans) -> str:
    """``text`` with the measure spans blanked, preserving offsets.

    Blanking rather than deleting keeps every other offset valid, so a filter or
    dimension detector reading the remainder sees the sentence it expects.
    """
    if not spans:
        return text
    chars = list(text)
    for start, end in spans:
        for i in range(start, min(end, len(chars))):
            chars[i] = " "
    return "".join(chars)


#: "top 5", "bottom 5", "largest 10". The DIRECTION is resolved separately by
#: ``_detect_ranking``; this only extracts the N. "bottom"/"smallest"/"lowest"
#: were missing, so a Bottom-N question kept its ascending sort but silently lost
#: its limit and returned every group.
_TOP_N_RE = re.compile(
    r"\b(?:top|bottom|first|last|largest|biggest|highest|smallest|lowest)\s+(\d+)\b")


def _detect_top_n(q: str) -> Optional[int]:
    m = _TOP_N_RE.search(q)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# ERE securitisation sprint — analytical-intent recognition.
# These run FIRST in the deterministic parser so a cross-period comparison,
# a securitisation scale-up / run-rate forecast, or a risk-limit question is
# never silently collapsed to a point-in-time KPI. Each emits a governed spec
# (no hallucinated fields) that the runtime / API layer resolves against the
# governed evolution / forecast / risk-monitor data.
# --------------------------------------------------------------------------- #

# An explicit whole-book summary intent. Only these questions may fall back to
# the whole-book count+balance summary; anything else that resolves no metric
# and no dimension is an UNMAPPED question and must be refused, not answered.
_SUMMARY_INTENT_RE = re.compile(
    r"\b(summary|summarise|summarize|overview|snapshot|at a glance|"
    r"key metrics|kpis?|headlines?|portfolio (?:summary|overview|position)|"
    r"the (?:whole )?book|total (?:balance|exposure))\b")

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
    # A pipeline/funding "bridge" to a target amount, and "securitisation
    # size/target", are scale-up questions (gap to target + time at run-rate).
    r"(?:pipeline|funding|completion) bridge|bridge to £?\s?\d|"
    r"securitisation (?:size|target|threshold)|"
    r"how much pipeline is needed|completion rate is assumed|what conversion rate|"
    # KFI→completion conversion-rate questions route to the governed
    # conversion assumption (not a point-in-time KPI).
    r"conversion rates?\b|completion conversion|"
    r"funded balance extrapolation|annualised completion|"
    r"what happens if .*run.?rate|milestone|"
    # A "forecast curve" / "projection curve" / "balance curve" is a request for
    # the forward funded-balance line, not a point-in-time KPI.
    r"forecast curve|projection curve|balance curve|"
    r"(?:forecast|projected|project).{0,20}curve|curve.{0,20}(?:forecast|funded balance)")

# Magnitude suffixes for a forecast target. "mm" (securitisation notation for
# millions) must sort before "m", and "bn"/"billion" before "b".
_TARGET_MULTIPLIER = {"k": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6,
                      "b": 1e9, "bn": 1e9, "billion": 1e9}
_TARGET_VALUE_RE = re.compile(
    r"£?\s*(\d+(?:\.\d+)?)\s*(mm|million|bn|billion|b|m|k)\b")


def _forecast_target_value(q: str) -> Optional[float]:
    m = _TARGET_VALUE_RE.search(q)
    if m:
        return float(m.group(1)) * _TARGET_MULTIPLIER[m.group(2)]
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
    if "conversion rate" in q or "completion conversion" in q \
            or "what conversion" in q:
        return "conversion"
    if "how much pipeline" in q and "reach" in q:
        return "pipeline_needed"
    # A "bridge to £X" asks for the gap to the target (additional completions /
    # pipeline needed), not just the milestone date.
    if "bridge" in q and _forecast_target_value(q):
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


# A funded-balance ATTRIBUTION bridge (waterfall): opening balance → per-category
# change → latest balance. Triggered by explicit "waterfall"/"bridge" or an
# attribution phrasing ("what drove / contributed to the growth/movement"). NB the
# forecast recogniser runs FIRST and owns "…bridge to £<target>" (a scale-up), so a
# £-target bridge never reaches here.
_BRIDGE_TRIGGER_RE = re.compile(
    r"\bwaterfall\b|\bbridge\b|"
    r"what (?:drove|is driving|contributed)|"
    r"contribut(?:ion|ions|ed|ors?)\b|"
    r"(?:growth|movement|change|increase|decrease|swing)\s+(?:by|across|driven|attribut)")


def _bridge_recognizer(q: str, title: str, semantics: dict, available_columns=None
                       ) -> Optional[Tuple[MIQuerySpec, dict]]:
    """Funded balance attribution bridge → governed ``bridge_query`` plan
    (resolved by the API's funded-bridge service into a waterfall)."""
    if not _BRIDGE_TRIGGER_RE.search(q):
        return None
    dim_keys, terms, _rem = _explicit_dimensions(q, semantics,
                                                 available_columns=available_columns)
    dim = dim_keys[0] if dim_keys else None
    # A bare numeric axis after "by" ("… by LTV", "… by age") attributes by that
    # measure's BAND. Scoped to the post-"by" text so the word "balance" in
    # "balance bridge" never selects a ticket-band attribution by accident.
    if dim is None and " by " in q:
        after_by = q.split(" by ", 1)[1]
        for term, bucket in sorted(_NUMERIC_AXIS_BUCKET.items(),
                                   key=lambda kv: len(kv[0]), reverse=True):
            if bucket in _fields(semantics) and re.search(r"\b" + re.escape(term) + r"\b", after_by):
                dim = bucket
                if not terms:
                    terms = [term]
                break
    periods = _detect_periods(q)
    start = periods[0] if periods else None
    spec = MIQuerySpec(
        intent="chart", chart_type="none", metric=None, aggregation="sum",
        execution_mode="temporal", bridge_query=True, bridge_dimension=dim,
        compare_periods=([start] if start else []),
        output_format="chart", title=title,
        explanation=("Funded balance attribution bridge: opening balance → per-"
                     "category change over the chosen dimension → the latest "
                     "balance. Deltas reconcile to the net change; a source-"
                     "portfolio lens (total / direct / acquired / cohort) scopes it."))
    return spec, _det_meta("high", bool(dim_keys),
                           terms or ([dim] if dim else ["funded_bridge"]),
                           note="funded_bridge")


# Static-pool cohort progression: how a cohort's funded metrics EVOLVE across
# reporting periods. Distinguished from a plain whole-book evolution by a cohort
# SCOPE — a source portfolio (acquired_001 / the acquired book / direct) and/or
# an origination vintage.
_PROGRESSION_MARKER_RE = re.compile(
    r"\bevolv|\bprogress|\bseason|static[\s-]?pool|over time|\btrend|"
    r"how (?:has|have|did).*(?:evolv|change|move|progress|grow|season|track)|"
    r"across (?:periods|reports|reporting)|by reporting")
_VINTAGE_PHRASE_RE = re.compile(
    r"originated in\s+(20\d{2})(?:[-\s]?q([1-4]))?|"
    r"vintage\s+(20\d{2})(?:[-\s]?q([1-4]))?|"
    r"(20\d{2})(?:[-\s]?q([1-4]))?\s+vintage|"
    r"\bcohort\b.*?(20\d{2})")


def _cohort_vintage(q: str) -> Tuple[Optional[str], Optional[str]]:
    """(vintage_label, grain) from an origination-vintage phrase, e.g.
    'originated in 2023' → ('2023', 'Y'); '2023 q2 vintage' → ('2023-Q2', 'Q')."""
    m = _VINTAGE_PHRASE_RE.search(q)
    if not m:
        return None, None
    groups = [g for g in m.groups() if g]
    year = next((g for g in groups if re.fullmatch(r"20\d{2}", g)), None)
    quarter = next((g for g in groups if re.fullmatch(r"[1-4]", g)), None)
    if not year:
        return None, None
    if quarter:
        return f"{year}-Q{quarter}", "Q"
    return year, "Y"


def _cohort_progression_recognizer(q: str, title: str, semantics: dict
                                   ) -> Optional[Tuple[MIQuerySpec, dict]]:
    """Cohort static-pool progression → governed ``cohort_progression`` plan.

    Fires only when the question has BOTH a progression marker and a cohort
    scope — a source portfolio (``mentions_portfolio``) or an origination
    vintage — so a plain whole-book 'balance evolution' stays with the ordinary
    evolution route."""
    if not _PROGRESSION_MARKER_RE.search(q):
        return None
    from .portfolio_lens import mentions_portfolio  # local: avoid import cycle at load
    vintage, grain = _cohort_vintage(q)
    if not (vintage or mentions_portfolio(q)):
        return None
    metric, _agg, _matched = _detect_metric(q, semantics)
    spec = MIQuerySpec(
        intent="chart", chart_type="line", metric=metric, aggregation="sum",
        execution_mode="temporal", cohort_progression=True,
        cohort_vintage=vintage, cohort_grain=grain,
        output_format="chart", title=title,
        explanation=("Static-pool cohort progression: the chosen funded metric "
                     "(balance / LTV / rate / NNEG) for a cohort — a source "
                     "portfolio ± origination vintage — tracked across reporting "
                     "periods."))
    return spec, _det_meta("high", False, [vintage or "cohort_progression"],
                           note="cohort_progression")


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
from question_interpretation import lexical as _lexical  # noqa: E402

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
#: Item 1 — THE PHRASES COME FROM `question_interpretation.lexical`, which owns
#: the comparator vocabulary. This list used to name its own words, and
#: `execution_receipt._THRESHOLD_PATTERNS` named a different set; they agreed on
#: 16 of 30. Where both were blind — "bigger than", "larger than", "higher
#: than", "smaller than", "lower than" — the narrowing vanished, no facet was
#: raised, and the whole book came back as fact.
#:
#: The symbol operators stay here: `>=`, `<=`, `>`, `<`, `=` are notation, not
#: English, and the receipt has its own reason to treat them separately.
def _comparator_pattern(op: str, symbols: str = "") -> str:
    alternation = _lexical.comparator_alternation((op,))
    if symbols:
        alternation = alternation + "|" + symbols
    # A NEGATED comparator is a different operator, and its phrase CONTAINS the
    # un-negated one: "no more than 150000" holds "more than 150000". Without
    # this guard the `gt` pattern matches inside the `le` phrase and the filter
    # can invert — the narrowing runs the wrong way and the answer is confidently
    # backwards. The negated forms are carried explicitly by the `ge`/`le`
    # entries in the owning vocabulary, so refusing to match after "no"/"not"
    # loses nothing.
    # `or ` for the same reason one step along: "greater than or equal to"
    # contains "equal to", and the compound is already carried as a `ge` phrase.
    return (rf"(?<!\bno )(?<!\bnot )(?<!\bor )"
            rf"(?:{alternation})\s*{_VALUE}")


_FILTER_COMPARATORS: List[Tuple[str, str]] = [
    # `between` keeps its own shape: it is the one operator taking two values.
    (rf"between\s+{_VALUE}\s+and\s+{_VALUE}", "between"),
    # Order is load-bearing and now guaranteed by the owner: the vocabulary is
    # sorted longest-first, so "greater than or equal to" is not shadowed by
    # "greater than" and "no more than" is not read as "more than" with the
    # negation discarded — which would invert the filter.
    (_comparator_pattern("ge", ">="), "ge"),
    (_comparator_pattern("le", "<="), "le"),
    (_comparator_pattern("gt", ">"), "gt"),
    (_comparator_pattern("lt", "<"), "lt"),
    (_comparator_pattern("eq", "="), "eq"),
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


#: Subject keyword -> resolver, for proximity-based threshold binding.
_FILTER_SUBJECT_PATTERNS = (
    (r"\bltv\b|\bloan to value\b", "ltv"),
    (r"\b(?:age|aged|youngest|borrowers?|years?|yrs?|yo|year[- ]?old|older|younger)\b", "age"),
    (r"\brate\b|\binterest\b|\bcoupon\b", "rate"),
    (r"\bbalance\b|\boutstanding\b|\bexposure\b|\bloan size\b|\bticket\b", "balance"),
    (r"\bvaluation\b|\bproperty value\b", "valuation"),
)


def _resolve_subject(kind: str, semantics: dict, available_columns=None):
    if kind == "ltv":
        return _ltv_metric(semantics, available_columns)
    if kind == "age":
        return _age_metric(semantics, available_columns)
    if kind == "rate":
        return _rate_metric(semantics, available_columns)
    if kind == "balance":
        return _balance_metric(semantics, available_columns)
    return find_field(semantics, role="metric", fmt="currency",
                      keywords=("valuation", "value"))


def _filter_field_of(q: str, semantics: dict, available_columns=None,
                     anchor: Optional[int] = None,
                     value_end: Optional[int] = None) -> Optional[str]:
    """Resolve the field a numeric threshold applies to from the question text.

    When ``anchor`` (the comparator's position) is supplied, the subject NEAREST
    BEFORE the comparator wins. Fixed precedence alone bound "what is the average
    LTV for borrowers over 75?" to LTV — the metric named earlier in the sentence
    — and silently applied the age threshold to the wrong column. Proximity is
    what a reader uses, and it is what the predicate actually means.
    """
    if anchor is not None:
        head = q[:anchor]
        # A currency amount is a balance threshold regardless of earlier nouns.
        #
        # THE WINDOW IS THE MATCH, NOT A GUESS. This probed a fixed twelve
        # characters after the comparator, which was wide enough for every
        # phrase the old vocabulary held — "over " is five, "more than " is ten.
        # Item 1 added the phrases both owners were missing, and "bigger than "
        # is exactly twelve: the £ fell one character outside the window, the
        # currency test failed, and the threshold bound to `current_loan_to_value`
        # instead of the balance. No loan has an LTV over 150,000, so the answer
        # became a refusal — safer than the whole-book figure it used to return,
        # and still not the answer.
        #
        # The caller knows where the value ends, so it says so. A fixed span
        # around a variable-length vocabulary is the same hard-coding this
        # programme keeps finding, one layer down from the lists themselves.
        probe_end = value_end if value_end is not None else anchor + 12
        if re.search(r"£\s*$|£\s*\d", q[max(0, anchor - 2):probe_end]):
            balance = _balance_metric(semantics, available_columns)
            if balance:
                return balance
        # A subject stated immediately AFTER the number is a postfix predicate
        # ("above 60% LTV") and binds tightest. Otherwise the subject nearest
        # BEFORE the comparator wins ("for borrowers over 75").
        tail = q[anchor:anchor + 28]
        for pattern, kind in _FILTER_SUBJECT_PATTERNS:
            if re.search(pattern, tail):
                resolved = _resolve_subject(kind, semantics, available_columns)
                if resolved:
                    return resolved
        best: Optional[Tuple[int, str]] = None
        for pattern, kind in _FILTER_SUBJECT_PATTERNS:
            for match in re.finditer(pattern, head):
                if best is None or match.start() > best[0]:
                    best = (match.start(), kind)
        if best is not None:
            resolved = _resolve_subject(best[1], semantics, available_columns)
            if resolved:
                return resolved
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


def _scatter_axes(q: str, semantics: dict, available_columns=None
                  ) -> Optional[Tuple[str, str]]:
    """The two numeric axes a scatter question actually names, in order of
    appearance ("ltv vs age" -> (ltv, age)), or None when the question does not
    name two distinct numeric measures. A scatter is only ever emitted from
    axes resolved here — axes are NEVER defaulted/invented, so categorical
    "X vs Y" phrasing ("single vs joint") is not hijacked into a scatter."""
    found: List[Tuple[int, str]] = []
    for term in sorted(_NUMERIC_AXIS_BUCKET, key=len, reverse=True):
        m = re.search(r"\b" + re.escape(term) + r"\b", q)
        if not m:
            continue
        fld = _resolve_numeric_axis(term, semantics, available_columns)
        if fld and all(fld != f for _, f in found):
            found.append((m.start(), fld))
    found.sort(key=lambda t: t[0])
    if len(found) >= 2:
        return found[0][1], found[1][1]
    return None


#: Clause openers that introduce a CONDITION rather than a measure. A measure
#: word appearing after one of these names the field being filtered ON, never
#: the field being reported: in "balance where LTV above 50%" the subject is
#: balance and LTV is the condition.


def _metric_slot(text: str) -> str:
    """The span of a question that may legitimately NAME the metric.

    ``_detect_metric`` walks a fixed-priority vocabulary and returns the highest
    -priority entry appearing ANYWHERE in the text it is given. It has no notion
    of sentence position or of grammatical role, so a field named inside a
    condition captures the metric slot: "balance where LTV above 50%" resolved to
    weighted-average LTV because ``ltv`` precedes ``balance`` in ``_METRIC_TERMS``.

    The fix is precedence, not vocabulary: give the detector only the subject
    side of the sentence. This truncates at the first FILTER clause, which is
    recognised conservatively — an opener counts only when a numeric bound
    follows it. Without that guard "regions with the highest LTV" would be cut at
    "with" and lose its measure, while "loans with LTV above 50%" must be cut.

    Grouping clauses are already handled upstream by ``_grouping_segments``;
    this composes with that split rather than repeating it.

    Stage 3, conversion 3: the thirteen openers used to be declared here AND,
    byte for byte, in ``answer_type``. They are now declared once in
    ``question_interpretation.lexical``, which owns this decision. The rule is
    unchanged — proved identical on all 690 real-surface corpus questions and on
    12 edge cases before the delegation was made.
    """
    from question_interpretation.lexical import metric_slot as _owner
    return _owner(text)


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
#: P1A — a categorical scope may be introduced by any of several prepositions,
#: may carry a leading article, and may be followed by trailing punctuation or a
#: descriptive noun. Previously anchored bare at ``$`` after ``in`` only, so
#: "in London?" (a question mark), "for London" and "in the South East" all
#: failed while "in london" succeeded — the shapes a person actually types were
#: exactly the ones that missed.
_CATEGORICAL_FILTER_RE = re.compile(
    # "to" is NOT a general scope preposition. Admitting it outright made
    # "when is it expected to complete?" bind a geography called "Complete",
    # which selected nothing and turned a pipeline question into a refusal. It
    # is admitted only in the idiom that actually names a place —
    # "exposure to the South East", "lending to Scotland".
    r"(?:geographic\s+region|geographic|geography|region|in|for|across|within|from"
    r"|(?:exposure|exposures|lending|concentration|allocation)\s+to)\s+"
    r"(?:the\s+)?"
    r"([a-z][a-z]*(?:\s+[a-z]+){0,3}?)"
    r"(?:\s+(?:region|regions|area|areas|loans|loan|book|portfolio))?"
    r"[\s?.!,]*$")
_CATEGORICAL_STOPWORDS = {"the", "loans", "loan", "with", "and", "by", "more",
                          "less", "than", "over", "under", "above", "below",
                          "book", "portfolio", "region", "regions", "area",
                          "areas", "total", "all", "each", "this", "that",
                          "it", "them", "me", "my", "our", "us", "there"}
#: Terms another governed resolver already owns. Widening the preposition list
#: to accept "for <place>" also made "for joint borrowers" look like a place, so
#: a borrower-structure predicate was bound to the GEOGRAPHY field and selected
#: nothing. A categorical geography value may not contain any of these.
_NON_PLACE_TERMS = {
    "joint", "sole", "single", "borrower", "borrowers", "applicant",
    "applicants", "fixed", "variable", "floating", "rate", "product",
    "products", "broker", "brokers", "channel", "active", "redeemed",
    "arrears", "default", "performing", "vintage", "cohort", "type",
    "status", "band", "bucket", "ltv", "age", "balance", "exposure",
    "value", "valuation", "interest", "securitisation", "them", "these",
    # Sourcing cohorts. "the acquired book" names HOW the loans were sourced,
    # which the portfolio lens already resolves; reading it as a place invented
    # a geography value the column does not contain and emptied the population.
    "acquired", "direct", "originated", "origination", "purchased",
    "sponsored", "warehouse", "book", "books", "portfolio", "portfolios",
    # Funding state, not a place. "how many loans are in the FUNDED book"
    # resolved to a geography called "Funded", which matches nothing, and the
    # question was refused for an empty population.
    "funded", "unfunded", "pipeline", "drawn", "undrawn",
    # Time, not a place. The scope prepositions include "to" so that "exposure
    # to the South East" is read as a geography (it used to return the whole
    # book at low confidence, with no filter at all). "to" also appears in
    # comparison and horizon phrasings — "compared to last month", "relative to
    # the prior quarter" — where the trailing noun phrase is a period. Binding
    # one of those to the geography field would select nothing and turn a
    # comparison into a refusal.
    "month", "months", "quarter", "quarters", "year", "years", "week", "weeks",
    "day", "days", "date", "dates", "period", "periods", "today", "now",
    "yesterday", "prior", "previous", "last", "next", "ago", "date-range",
}


#: "What PROPORTION of the book ...". A share question needs two populations —
#: the filtered numerator and the whole-book denominator — so it is a distinct
#: governed aggregation, not a KPI of the filtered rows.
_SHARE_RE = re.compile(
    r"\bwhat\s+(?:proportion|percentage|share|fraction|%)\b|"
    r"\b(?:proportion|percentage|share|fraction)\s+of\s+(?:the\s+)?"
    r"(?:book|portfolio|loans|balance|exposure)\b|"
    r"\bhow much of the (?:book|portfolio)\b|"
    r"\bwhat\s+(?:%|percent)\s+of\b", re.I)
#: A share is measured on a balance basis unless the question counts loans.
_SHARE_COUNT_RE = re.compile(r"\b(?:loans|cases|accounts|borrowers)\b", re.I)


def _share_request(q: str, semantics: dict, available_columns=None
                   ) -> Optional[Tuple[str, Optional[str]]]:
    """``(basis, metric)`` for a share question, or None.

    ``basis`` is ``"balance"`` or ``"count"``; ``metric`` is the balance field
    for a value share and ``None`` for a count share (the executor counts rows
    when no metric is set).
    """
    if not _SHARE_RE.search(q):
        return None
    counts_loans = bool(re.search(
        r"\b(?:proportion|percentage|share|fraction|%|percent)\s+of\s+(?:the\s+)?"
        r"(?:loans|cases|accounts|borrowers)\b", q, re.I))
    if counts_loans:
        return ("count", None)
    return ("balance", _balance_metric(semantics, available_columns))


#: P1D — an AGGREGATE-CONTRIBUTION question.
#:
#: "Which region CONTRIBUTES MOST TO the weighted average LTV?" is not "which
#: region has the highest LTV". A portfolio weighted average is
#: sum(w_i * v_i) / sum(w_i), so group g contributes
#:
#:     (sum over g of w * v) / (sum over the book of w)  ==  weight_share_g * value_g
#:
#: and those contributions sum to the portfolio figure. A small group with a
#: high value contributes almost nothing. On the demonstration book the two
#: rankings are near-inverted: West Midlands has the highest regional LTV
#: (43.90%) and contributes 2.72pp, while South East contributes 11.34pp of the
#: portfolio's 43.15%. Answering one with the other is a silent semantic error,
#: which is why this is a governed aggregation of its own rather than a sort
#: order on the existing one.
_CONTRIBUTION_RE = re.compile(
    r"\b(?:contributes?|contributing|contributed)\s+(?:the\s+)?most\b|"
    r"\b(?:biggest|largest|greatest|main|primary|top)\s+contributors?\b|"
    r"\bcontributions?\s+to\b|"
    r"\bcontributes?\s+(?:the\s+)?most\s+to\b|"
    r"\bdriv(?:es|ing|en)\s+(?:the\s+)?most\s+of\b|"
    r"\b(?:accounts?|accounting)\s+for\s+(?:the\s+)?most\s+of\b|"
    r"\bwhat\s+is\s+driving\s+(?:the\s+)?(?:most\s+of\s+)?(?:the\s+)?"
    r"(?:portfolio\s+)?(?:weighted[\s-]?average|wa)\b", re.I)

#: The object must be a WEIGHTED aggregate. Two ways it qualifies: the question
#: says so, or the governed registry says the metric's default aggregation is a
#: weighted average. The registry is the authority; the phrase list only lets a
#: reader be explicit.
_WEIGHTED_AGGREGATE_RE = re.compile(
    r"\bweighted[\s-]?(?:average|avg|mean)\b|\bwa\s+(?:ltv|rate|yield)\b|"
    r"\bwavg\b", re.I)


def _contribution_request(q: str, semantics: dict, available_columns=None
                          ) -> Optional[Tuple[str, str]]:
    """``(metric_key, weight_field)`` for an aggregate-contribution question.

    Returns None unless BOTH hold, so an ordinary ranking is never converted:

    * the question uses contribution language ("contributes most to", "drives
      most of", "accounts for most of", "largest contributor to"); and
    * the object is a weighted aggregate — the governed registry gives the
      metric a ``weighted_avg`` default aggregation and a weight field, or the
      question names a weighted average explicitly.

    "Which region has the highest LTV?" carries no contribution language and is
    untouched. That distinctness is the point of the function.
    """
    if not _CONTRIBUTION_RE.search(q):
        return None
    metric, _agg, _terms = _detect_metric(q, semantics)
    if not metric:
        return None
    if available_columns is not None and metric not in set(available_columns):
        return None
    entry = (_fields(semantics) or {}).get(metric) or {}
    weight = entry.get("weight_field") or (
        semantics.get("metadata", {}) or {}).get("default_weight_field")
    weighted = (str(entry.get("default_aggregation") or "").lower()
                in ("weighted_avg", "weighted_average"))
    if not weighted and not _WEIGHTED_AGGREGATE_RE.search(q):
        return None
    if not weight:
        return None
    if available_columns is not None and weight not in set(available_columns):
        return None
    return metric, weight


def _parse_categorical_filter(clause: str, semantics: dict, available_columns=None
                              ) -> Optional[Tuple[str, str]]:
    """Detect a categorical region filter in a clause -> (field_key, value).

    Governed SCOPE phrases are claimed before the search runs, so "the entire
    portfolio" and "the current portfolio" cannot be read as places called
    "Entire" and "Current". Claiming the span is what prevents the invalid
    filter existing at all; removing one afterwards would silently widen the
    population it had narrowed.
    """
    from .portfolio_lens import mask_scope_phrases  # local: avoids a cycle
    from .seasoning import mask_segment_phrases

    # Governed SCOPE phrases (P1I-A) and governed SEASONING phrases (P1J-1) are
    # both claimed before the place-resolver runs. Either one read as a place
    # invents a region that does not exist — "Entire", "Current", "Front".
    m = _CATEGORICAL_FILTER_RE.search(
        mask_segment_phrases(mask_scope_phrases(clause)).strip())
    if not m:
        return None
    value = m.group(1).strip()
    # "for loans in Wales" captures "loans in wales" — peel leading filler and
    # any nested preposition so the VALUE is the place, not the phrase.
    while True:
        head = value.split(" ", 1)[0]
        if head in _CATEGORICAL_STOPWORDS or head in ("in", "for", "within",
                                                      "across", "from"):
            if " " not in value:
                return None
            value = value.split(" ", 1)[1].strip()
            continue
        break
    if not value or value in _CATEGORICAL_STOPWORDS:
        return None
    if any(word in _NON_PLACE_TERMS for word in value.split()):
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
# The materialised borrower_type dimension uses "joint"/"single" values
# (matched case-insensitively by the executor).
_BORROWER_TYPE_VALUE = {"joint": "Joint", "sole": "Single"}


def _borrower_structure_filter(q: str, semantics: dict, available_columns=None
                               ) -> Optional[Tuple[Dict[str, Any], str]]:
    """Detect a 'joint'/'sole' borrower intent and resolve it to a filter.

    Returns ``(filters, note)`` or None. Prefers the materialised
    ``borrower_type`` value filter, then a ``borrower_structure`` value filter;
    falls back to a ``number_of_borrowers`` threshold (>=2 joint / ==1 sole) and
    notes the substitution; if none of those fields exists, returns an empty
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
    if has("borrower_type"):
        return {"borrower_type": _BORROWER_TYPE_VALUE[kind]}, \
            f"borrower_type = {_BORROWER_TYPE_VALUE[kind]}"
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


#: Clause boundaries for predicate parsing now live in
#: ``question_interpretation.lexical.clause_spans``, which owns the connectives
#: and the "85 and over" exception, and returns SPANS rather than strings.
#:
#: Why they had to move: a threshold must be resolved against its own clause or
#: not at all — "balance by region WHERE flurb is above 3" was once one clause
#: containing the word "balance", and the threshold bound to the balance column,
#: silently filtering by a predicate nobody asked for. That reasoning, and the
#: exception that keeps "85 and over" whole, are recorded with the code.


def _parse_filters(q: str, semantics: dict, available_columns=None,
                   unresolved: Optional[List[str]] = None,
                   spans: Optional[Dict[str, Tuple[int, int]]] = None
                   ) -> Dict[str, Any]:
    """Parse one or more filters joined by ``and`` / ``with`` / ``where`` (numeric
    thresholds — prefix OR postfix — and a categorical value).
    ``{field_key: condition}``.

    ``unresolved`` — when supplied — collects a note for every clause that stated
    a numeric threshold whose FIELD could not be resolved. Such a predicate is
    never guessed onto another column and never silently dropped: the caller
    surfaces it so the operator learns the filter was not applied.

    ``spans`` — when supplied — collects ``{field_key: (start, end)}``, the
    offsets of the CLAUSE each filter was resolved from. This is the parser half
    of the filter join: the facet layer supplies a clause's wording and its
    offsets but no field, and this supplies the field and the bound. Neither
    could be linked to the other while this function rewrote the question as it
    consumed clauses.
    """
    from question_interpretation.lexical import blank_consumed, clause_spans

    filters: Dict[str, Any] = {}
    # Parse a 'between A and B' first so its 'and' is not used as a clause split.
    #
    # It used to be excised from the string — ``work_q = work_q[:start] + " " +
    # work_q[end:]`` — and everything after parsed the rewritten text. That threw
    # away every offset, which is why the parser could supply a filter's FIELD
    # and BOUND but never say WHICH WORDS it came from, and why the two halves of
    # a filter clause could not be joined.
    #
    # Now the span is MARKED CONSUMED instead. The string is never mutated, the
    # clause splitter skips connectives inside a consumed span, and each clause
    # keeps its offsets into the original question.
    consumed: List[Tuple[int, int]] = []
    bm = re.search(_FILTER_COMPARATORS[0][0], q)
    if bm:
        field = _filter_field_of(q[max(0, bm.start() - 40):bm.end()], semantics)
        if field:
            filters[field] = {"op": "between", "value": _amount_from_match(bm, "between")}
            if spans is not None:
                spans[field] = (bm.start(), bm.end())
        consumed.append((bm.start(), bm.end()))

    # Split into clauses so "<age> 70+ with LTV above 50" yields two independent
    # thresholds, and a threshold is only ever resolved against its own clause.
    for clause_start, clause_end in clause_spans(q, tuple(consumed)):
        clause = blank_consumed(q, clause_start, clause_end, tuple(consumed)).strip()
        if not clause:
            continue
        field = _filter_field_of(clause, semantics)
        # Postfix first ("70+", "70 or above") — a number-before-operator phrase.
        matched = False
        for pattern, op in _POSTFIX_COMPARATORS:
            m = re.search(pattern, clause)
            if m and field:
                filters[field] = {"op": op, "value": float(m.group(1))}
                if spans is not None:
                    spans[field] = (clause_start, clause_end)
                matched = True
                break
        if matched:
            continue
        for pattern, op in _FILTER_COMPARATORS[1:]:  # skip 'between' (done above)
            m = re.search(pattern, clause)
            if not m:
                continue
            # Re-resolve against THIS comparator's position: the subject nearest
            # before the operator is the one the predicate is about.
            field = _filter_field_of(clause, semantics, available_columns,
                                     anchor=m.start(),
                                     value_end=m.end()) or field
            if field:
                filters[field] = {"op": op, "value": _amount_from_match(m, op)}
                if spans is not None:
                    spans[field] = (clause_start, clause_end)
                matched = True
            elif unresolved is not None:
                # A threshold was stated but its field is not a governed field in
                # this dataset. Refuse it visibly rather than binding it to some
                # other column that happens to be named elsewhere in the question.
                unresolved.append(
                    f"'{clause.strip()}' — no governed field matches this "
                    "condition, so the filter was not applied")
            break
        if matched:
            continue
        # Age stated without a comparator ("60 year old", "aged 60") -> equality.
        age_field = _age_metric(semantics, available_columns)
        if field == age_field and age_field:
            age_val = _age_equality_value(clause)
            if age_val is not None:
                filters[age_field] = {"op": "eq", "value": age_val}
                if spans is not None:
                    spans[age_field] = (clause_start, clause_end)
                continue
        cat = _parse_categorical_filter(clause, semantics, available_columns)
        if cat:
            filters[cat[0]] = cat[1]
            if spans is not None:
                spans[cat[0]] = (clause_start, clause_end)
    return filters


def _grouped_value_filters(q: str, semantics: dict, available_columns,
                           exclude_dims: Iterable[str] = ()) -> Tuple[Dict[str, Any], List[str]]:
    """Value filters expressed ALONGSIDE a grouping, e.g. 'balance by region where
    LTV above 50%' or 'balance by broker in the north'. Execution applies filters
    to the mask BEFORE grouping, so a grouped spec may legitimately carry them.

    A filter whose field is itself a grouping dimension is dropped (that is the
    grouping, not a filter). Returns ``(filters, unavailable_notes)`` — mirrors
    the filtered-KPI branch so a grouped filter is never silently discarded."""
    exclude = set(exclude_dims or ())
    unavailable: List[str] = []
    filters = _parse_filters(q, semantics, available_columns, unresolved=unavailable)
    # A borrower-structure value filter ("... for joint borrowers") resolves to a
    # categorical filter (or an unavailable note). Skip it when the grouping IS
    # the borrower dimension (that is the breakdown, not a filter).
    bstruct = _borrower_structure_filter(q, semantics, available_columns)
    if bstruct is not None:
        bfilters, bnote = bstruct
        bfilters = {k: v for k, v in (bfilters or {}).items() if k not in exclude}
        if bfilters:
            filters.update(bfilters)
        elif bnote and not (bfilters and set(bfilters) & exclude):
            unavailable.append(bnote)
    for d in exclude:
        filters.pop(d, None)
    return filters, unavailable


def _build_two_dim_spec(metric: Optional[str], dims: List[str], semantics: dict,
                        title: str, explicit: bool, terms: List[str],
                        has_count: bool = False,
                        filters: Optional[Dict[str, Any]] = None,
                        unavailable_filters: Optional[List[str]] = None
                        ) -> Tuple[MIQuerySpec, dict]:
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
        filters=filters or {}, unavailable_filters=unavailable_filters or [],
        title=title, explanation="Matrix of a metric across two dimensions.",
        output_format="chart")
    return spec, _det_meta(conf, explicit, terms)


def _build_multi_dim_table_spec(metric: Optional[str], dims: List[str], semantics: dict,
                                title: str, explicit: bool, terms: List[str],
                                has_count: bool = False,
                                filters: Optional[Dict[str, Any]] = None,
                                unavailable_filters: Optional[List[str]] = None
                                ) -> Tuple[MIQuerySpec, dict]:
    """Build a TABLE/pivot spec across 3+ requested dimensions.

    A chart shows at most two dimensions, so 3+ dimensions are grouped into a
    table over EVERY requested dimension — never silently truncated at parse.
    The executor groups by ``_all_group_dims(spec)`` and the dimension invariant
    then sees all of them applied."""
    fields = _fields(semantics)
    dims = [d for d in dims if d]
    if has_count or metric is None:
        metric, agg, weight = (None, "count", None)
    else:
        agg = "weighted_avg" if fields.get(metric, {}).get("format") == "percent" else "sum"
        weight = _default_weight(semantics, metric) if agg == "weighted_avg" else None
    spec = MIQuerySpec(
        intent="table", chart_type="none", metric=metric, dimensions=dims,
        aggregation=agg, weight_field=weight,
        filters=filters or {}, unavailable_filters=unavailable_filters or [],
        title=title,
        explanation=(f"Table across {len(dims)} dimensions "
                     "(a chart shows at most two, so the full breakdown is a table)."),
        output_format="table")
    return spec, _det_meta("high", explicit, terms)


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
        # A value filter on the ranked population ("top 5 regions by balance where
        # LTV above 50%") is applied before ranking — attach it (excluding the
        # ranking dimension itself) so it is never silently dropped.
        g_filters, g_unavail = _grouped_value_filters(
            q, semantics, available_columns, exclude_dims=[dim])
        spec = MIQuerySpec(
            intent="chart", chart_type="bar", metric=rmetric, dimension=dim,
            aggregation=ragg, weight_field=weight, top_n=(rank_limit or top_n),
            sort_by=rmetric, sort_direction=rank_dir, ranking_mode="grouped",
            filters=g_filters, unavailable_filters=g_unavail,
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


#: Adjectival forms of governed dimensions, used ONLY by the contribution
#: recogniser below.
_ADJECTIVAL_DIMENSIONS = {"regional": "region", "geographic": "geography",
                         "geographical": "geography"}
_ADJECTIVAL_DIMENSION_RE = re.compile(
    r"\b(?:" + "|".join(_ADJECTIVAL_DIMENSIONS) + r")\b", re.I)


def _contribution_recognizer(q: str, title: str, semantics: dict,
                             available_columns=None):
    """A governed aggregate-contribution plan, or None.

    Needs a contribution question AND a dimension to attribute the contribution
    across — "what drives the weighted average LTV?" with no grouping names no
    groups to rank, so it is left alone and the P0 contribution facet refuses it
    rather than this parser inventing a dimension.
    """
    request = _contribution_request(q, semantics, available_columns)
    if request is None:
        return None
    metric, weight = request
    dims, _terms, _rest = _explicit_dimensions(
        q, semantics, grouping=True, available_columns=available_columns)
    if not dims:
        # "the REGIONAL contribution to ..." names the same dimension in its
        # adjectival form. Normalised HERE rather than in the shared dimension
        # vocabulary: widening that vocabulary changes grouping behaviour for
        # every question in the product, and this phase is scoped to
        # aggregate-contribution semantics.
        normalised = _ADJECTIVAL_DIMENSION_RE.sub(
            lambda m: _ADJECTIVAL_DIMENSIONS[m.group(0).lower()], q)
        if normalised != q:
            dims, _terms, _rest = _explicit_dimensions(
                normalised, semantics, grouping=True,
                available_columns=available_columns)
    if not dims:
        return None
    spec = MIQuerySpec(
        intent="chart", chart_type="bar", metric=metric, dimension=dims[0],
        x=dims[0], aggregation="contribution", weight_field=weight,
        sort_direction="desc", output_format="chart_and_table", title=title,
        explanation=("Each group's contribution to the portfolio weighted "
                     "average: its share of the weight multiplied by its own "
                     "value, so the contributions sum to the portfolio "
                     "figure."))
    return spec, _det_meta("high", True, [dims[0]],
                           note="aggregate_contribution")


def _measure_set_recognizer(q: str, title: str, semantics: dict,
                            available_columns=None):
    """A governed MULTI-MEASURE plan, or None.

    One population, one optional filter set, one optional governed grouping,
    one reporting period — and every governed measure the question named. Filters
    and dimensions are resolved by the SAME helpers the single-measure paths use,
    so the population a multi-measure answer describes is the population a
    single-measure answer would have described.
    """
    # A RELATIONSHIP is not a measure set. "ltv vs interest rate" names two
    # governed measures but asks how they RELATE, which only a loan-level plot
    # can express — so it is left to the scatter path that already owns it, and
    # P0 keeps its relationship facet. Deferring here rather than competing.
    if any(token in q for token in (" vs ", " versus ", "scatter", "bubble",
                                    "plot", "against", "sized by",
                                    "relative to")):
        return None

    measures, spans = detect_measure_set(q, semantics, available_columns,
                                         with_spans=True)
    if len(measures) < 2:
        return None
    if len(measures) > MAX_MEASURES:
        # Reported, never truncated: answering four of five without saying so is
        # the silent omission P0 exists to prevent.
        return None

    # Dimensions and filters are read from the text the measures did NOT claim.
    remainder = _mask_spans(q, spans)
    dims, _terms, _rest = _explicit_dimensions(
        remainder, semantics, grouping=True, available_columns=available_columns)
    filters = _parse_filters(remainder, semantics, available_columns)
    region = _parse_categorical_filter(remainder, semantics, available_columns)
    if region is None:
        # A CFO usually states the scope FIRST — "For the London book, give me
        # …". The existing categorical resolver reads a trailing scope clause,
        # so the leading clause is handed to that same resolver rather than a
        # second pattern being invented for it.
        lead = remainder.split(",", 1)[0].strip()
        if lead and lead != remainder.strip():
            region = _parse_categorical_filter(lead, semantics, available_columns)
    if region and region[0] not in filters:
        filters[region[0]] = region[1]

    grouped = bool(dims)
    spec = MIQuerySpec(
        intent="chart" if grouped else "summary",
        chart_type="bar" if grouped else "none",
        measures=measures,
        metric=measures[0]["field"] if measures[0]["field"] != "loan_count" else None,
        aggregation=measures[0].get("aggregation") or "sum",
        dimension=dims[0] if grouped else None,
        x=dims[0] if grouped else None,
        filters=filters,
        output_format="chart_and_table" if grouped else "table",
        title=title,
        explanation=("Governed multi-measure request: "
                     + ", ".join(m["field"] for m in measures)
                     + " over one population."))
    return spec, _det_meta("high", bool(dims), [m["field"] for m in measures],
                           note="multi_measure")


#: Bare qualitative magnitudes. The COMPARATIVE and SUPERLATIVE forms are
#: deliberately absent: "highest LTV" is a ranking direction and is answerable,
#: while "high LTV" states a threshold the question never gives.
_QUALITATIVE_MAGNITUDES = ("high", "low", "large", "small", "big", "tiny",
                           "heavy", "light", "elevated", "significant",
                           "material", "modest", "poor", "strong", "weak")
#: The numeric subjects such a word can qualify, as the user says them.
_QUALITATIVE_SUBJECTS = ("age", "ltv", "loan to value", "balance", "balances",
                         "loan", "loans", "exposure", "rate", "rates", "value",
                         "size", "amount", "amounts", "arrears")
_QUALITATIVE_RE = re.compile(
    r"\b(" + "|".join(_QUALITATIVE_MAGNITUDES) + r")\s+"
    r"(?:" + "|".join(_QUALITATIVE_SUBJECTS) + r")\b")
#: Any parseable bound anywhere in the question disqualifies the guard.
_NUMERIC_BOUND_RE = re.compile(r"\d")


#: A question about how much data is MISSING, rather than about the data.
#: "missing region count" asks for the size of the excluded population; the
#: estate reports coverage on the reconciliation block of a normal answer and
#: has no intent that answers it directly. It used to answer the adjacent
#: question instead — "missing region count" returned balance by region — which
#: is a confident answer to something nobody asked.
_MISSING_COUNT_RE = re.compile(
    r"\b(?:missing|excluded|blank|null|unknown|incomplete)\s+[a-z_ ]{0,24}?\b(?:count|counts)\b|"
    r"\b(?:count|number)\s+of\s+(?:the\s+)?(?:missing|excluded|blank|null|incomplete)\b|"
    r"\b(?:loans|cases|accounts|records|rows)\s+(?:excluded|missing|omitted)\s+from\b|"
    r"\bhow\s+many\s+(?:loans|cases|accounts|records|rows)?\s*(?:are\s+)?"
    r"(?:missing|excluded|blank|null|incomplete)\b")


def _missing_data_request(q: str) -> bool:
    """True when the question asks for the SIZE of the excluded population."""
    return bool(_MISSING_COUNT_RE.search(q))


def _qualitative_threshold(q: str) -> Optional[str]:
    """The unparseable qualitative bound a question states, if any.

    "balance for borrowers over 80" states a threshold. "high age borrower
    exposure" states one too — it just never says what "high" is. The second
    used to return the WHOLE-BOOK balance with no filter at all, which answers a
    different question with confidence. There is no defensible default for
    "high", so the governed response is to ask.
    """
    if _NUMERIC_BOUND_RE.search(q):
        return None
    m = _QUALITATIVE_RE.search(q)
    return m.group(0) if m else None


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

    # "How much is missing?" is a data-quality question, and the estate has no
    # governed intent that answers it. Answering the adjacent question instead
    # is the failure mode this guards.
    if _missing_data_request(q):
        return (MIQuerySpec(
            intent="summary", chart_type="none", aggregation="count", title=title,
            explanation="This asks how much data is missing or excluded, which "
                        "is not a governed analytic here — coverage is reported "
                        "on the reconciliation of a normal answer, not as a "
                        "standalone count. Ask for the measure you want and the "
                        "coverage will be stated with it.",
            output_format="text"),
            _det_meta("low", False, [], note="unresolved_metric"))

    # A qualitative bound with no number is a declared element that cannot be
    # honoured. Asking is the only safe response; answering unfiltered is not.
    qual = _qualitative_threshold(q)
    if qual is not None:
        return (MIQuerySpec(
            intent="summary", chart_type="none", aggregation="count", title=title,
            explanation=f"'{qual}' does not state a threshold I can apply. Give a "
                        "bound (for example 'over 80') and I will filter on it; "
                        "no unfiltered figure has been substituted.",
            output_format="text"),
            _det_meta("low", False, [], note="unresolved_metric"))

    # ---- ERE analytical intents (checked first; emit governed plans) --------
    # A scale-up / run-rate forecast, a cross-period comparison, or a risk-limit
    # question must never fall through to a point-in-time KPI. Forecast is checked
    # before compare so "compare ... run-rate extrapolation" routes to forecast.
    fc = _forecast_scale_recognizer(q, title)
    if fc is not None:
        return fc
    # An aggregate-contribution question is a DIFFERENT calculation from both
    # the per-group ranking and the balance bridge it can resemble in wording
    # ("largest contributor to", "drives most of"). A bridge decomposes a
    # BALANCE MOVEMENT between two dates; this decomposes a WEIGHTED AVERAGE at
    # one date. Recognised first, and narrow enough that it cannot claim a
    # genuine bridge question: the object must be a weighted aggregate, and a
    # balance is a sum.
    contrib = _contribution_recognizer(q, title, semantics,
                                       available_columns=available_columns)
    if contrib is not None:
        return contrib
    br = _bridge_recognizer(q, title, semantics, available_columns=available_columns)
    if br is not None:
        return br
    cp = _cohort_progression_recognizer(q, title, semantics)
    if cp is not None:
        return cp
    cmp_spec = _compare_recognizer(q, title, semantics)
    if cmp_spec is not None:
        return cmp_spec
    rl = _risk_limit_recognizer(q, title)
    if rl is not None:
        return rl
    # A question naming several governed measures is ONE request over one
    # population. Recognised before the generic single-metric paths, which would
    # otherwise keep whichever measure they matched first and drop the rest.
    ms = _measure_set_recognizer(q, title, semantics,
                                 available_columns=available_columns)
    if ms is not None:
        return ms

    # ---- filtered count / balance ("how many loans with <field> <op> N") ---
    # A counting/aggregating question with a numeric threshold routes to a
    # filtered summary (count or balance), NOT a bar chart, so "how many loans
    # with youngest age more than 70" answers a number.
    is_count_q = bool(re.search(r"\bhow many\b|\bnumber of\b|\bcount of\b", q))
    is_balance_q = bool(re.search(r"\bhow much\b|\btotal balance\b", q))
    # A COUNT question also wants the balance only when the balance word sits
    # BEFORE the counting phrase — "total balance and how many loans over 80".
    # In "how many loans have a balance above £250k" the word names the field
    # being filtered ON, and reading it as a second measure turned a count into
    # a filtered BALANCE with a currency headline. Everything after "how many"
    # is the population being counted, never the measure.
    _balance_word = r"\b(balance|exposure|outstanding)\b"
    if is_count_q:
        _subject = re.split(r"\bhow many\b|\bnumber of\b|\bcount of\b", q)[0]
        wants_balance_too = bool(re.search(_balance_word, _subject))
    else:
        wants_balance_too = bool(re.search(_balance_word, _metric_slot(q)))
    if is_count_q or is_balance_q:
        # Support one OR MORE filters joined by "and" (numeric thresholds and a
        # categorical region value), e.g. "youngest age more than 70 and
        # geographic region south west".
        unresolved_notes: List[str] = []
        filters = _parse_filters(q, semantics, available_columns,
                                 unresolved=unresolved_notes)
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
        # When the ONLY predicate is one whose field this dataset does not carry
        # ("how many loans have Risk Score above 700"), the filter IS the
        # question. Answering the unfiltered count would answer a different
        # question with a confident number, so refuse instead — the workflow's
        # controlled-unmapped guard turns "unmapped" into an honest explanation.
        # A predicate that merely NARROWS an otherwise-valid grouped question
        # ("balance by region where flurb is above 3") is different: that answer
        # still stands, with the unapplied filter disclosed (see
        # ``_grouped_value_filters``).
        if unresolved_notes and not filters and not unavailable:
            return (MIQuerySpec(
                intent="summary", chart_type="none", aggregation="count",
                title=title, unavailable_filters=unresolved_notes,
                explanation="Could not map the requested condition to a governed "
                            "field.",
                output_format="text"),
                _det_meta("low", False, [], note="unmapped"))
        unavailable = unavailable + unresolved_notes
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
    # Two resolvable numeric axes joined by "vs"/"scatter" make this a plot.
    # A bare " vs " between categorical values ("single vs joint") does NOT —
    # that phrasing stays with the categorical grouping/filter grammar.
    scatter_axes = (_scatter_axes(q, semantics, available_columns)
                    if ("scatter" in q or " vs " in q or " versus " in q) else None)
    explicit_plot = ("bubble" in q or "scatter" in q or "sized by" in q
                     or scatter_axes is not None or "plot" in q or "against" in q)
    numeric_bubble = False
    if len(seg_classes) >= 2 and not explicit_plot and "treemap" not in q:
        n_categorical = sum(1 for c in seg_classes if c[0] == "categorical")
        if n_categorical >= 1:
            # The two visual dimensions (row/column), in question order.
            dims: List[str] = []
            for c in seg_classes[:2]:
                key = c[1] if c[0] == "categorical" else c[2]
                if key and key not in dims:
                    dims.append(key)
            # The first dimension may sit in the metric position ("ticket size
            # by borrower type"): recover it from the explicitly-named
            # dimensions, in question order.
            if len(dims) < 2 and dim_keys:
                merged: List[str] = []
                for k in list(dim_keys) + dims:
                    if k and k not in merged:
                        merged.append(k)
                dims = merged[:2]
            # The FULL ordered set of requested dimensions (explicit first, in
            # question order, then any bucketed grouping segments) — used to detect
            # 3+ dimensions so none is ever silently truncated at parse.
            full_dims: List[str] = []
            for k in list(dim_keys):
                if k and k not in full_dims:
                    full_dims.append(k)
            for c in seg_classes:
                key = c[1] if c[0] == "categorical" else c[2]
                if key and key not in full_dims:
                    full_dims.append(key)
            metric, _agg, matched = _detect_metric(metric_part, semantics)
            if len(full_dims) >= 3:
                # 3+ dimensions cannot be charted -> a table/pivot over all of them.
                g_filters, g_unavail = _grouped_value_filters(
                    q, semantics, available_columns, exclude_dims=full_dims)
                return _build_multi_dim_table_spec(
                    metric, full_dims, semantics, title, True,
                    [c[1] for c in seg_classes], has_count=("count" in matched),
                    filters=g_filters, unavailable_filters=g_unavail)
            g_filters, g_unavail = _grouped_value_filters(
                q, semantics, available_columns, exclude_dims=dims)
            return _build_two_dim_spec(metric, dims, semantics, title, True,
                                       [c[1] for c in seg_classes[:2]],
                                       has_count=("count" in matched),
                                       filters=g_filters, unavailable_filters=g_unavail)
        # All-numeric two-segment grouping -> bubble (two numeric axes + size).
        numeric_bubble = True

    is_ranking, rank_dir, rank_limit = _detect_ranking(q)

    # ---- two explicit dimensions ("<dim> by <dim>") -> matrix --------------
    # "ticket size by borrower type": the first dimension sits in the metric
    # position, so the segment classifier above sees only one grouping segment.
    # Two explicitly-named dimensions with no ranking/plot intent are a
    # cross-tab of the two (count, or the named metric).
    if (len(dim_keys) >= 2 and not explicit_plot and not is_ranking
            and "treemap" not in q):
        metric, _agg, matched = _detect_metric(remaining, semantics)
        g_filters, g_unavail = _grouped_value_filters(
            q, semantics, available_columns, exclude_dims=dim_keys)
        if len(dim_keys) >= 3:
            # 3+ dimensions -> a table/pivot over ALL of them (never truncate).
            return _build_multi_dim_table_spec(
                metric, list(dim_keys), semantics, title, True, dim_terms,
                has_count=("count" in matched or _wants_count(q)),
                filters=g_filters, unavailable_filters=g_unavail)
        return _build_two_dim_spec(metric, dim_keys[:2], semantics, title, True,
                                   dim_terms,
                                   has_count=("count" in matched or _wants_count(q)),
                                   filters=g_filters, unavailable_filters=g_unavail)

    # ---- ranked / "largest" queries ---------------------------------------
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
        g_filters, g_unavail = _grouped_value_filters(
            q, semantics, available_columns, exclude_dims=g_keys[:3])
        return (MIQuerySpec(
            intent="chart", chart_type="treemap", metric=metric,
            hierarchy=g_keys[:3], aggregation=agg, top_n=top_n, title=title,
            filters=g_filters, unavailable_filters=g_unavail,
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
    # Only when the question actually names two numeric measures ("ltv vs age",
    # "scatter of rate vs balance"). Axes are never invented: an explicit
    # "scatter" with no resolvable axes, or a categorical "X vs Y" ("single vs
    # joint"), falls through to the grouping / refusal grammar instead.
    if scatter_axes is not None:
        x, y = scatter_axes
        return (MIQuerySpec(
            intent="chart", chart_type="scatter", x=x, y=y,
            aggregation="loan_level", title=title,
            explanation="Scatter of two numeric measures.",
            output_format="chart"),
            _det_meta("high" if "scatter" in q else "medium", explicit, dim_terms))

    # ---- line (trend over time) -------------------------------------------
    is_line = ("over time" in q or "trend" in q or "monthly" in q
               or "by month" in q or "evolution" in q or "by reporting date" in q
               or "over the months" in q or "by reporting month" in q
               or "reporting month" in q or "by week" in q or "per week" in q
               or "weekly" in q or "by reporting period" in q)
    # NOTE: a vintage request used to force a line here. It must not: a VINTAGE
    # is a cohort label (2014, 2015, …), not a point on a time axis, and the
    # line path coerces its x to a date — turning every integer year into epoch
    # month "1970-01" and collapsing all thirteen vintages into ONE row that
    # still reported itself as "by origination vintage". A grouped bar over
    # vintage_year is the honest shape, and is what the ranking path already
    # produced for "which vintages have the highest LTV".
    # Resolve the metric from the phrase BEFORE the first "by" (the metric side),
    # so "<metric> by <dimension>" never picks the grouping term as the metric
    # (e.g. "balance by ltv" -> metric=balance, not LTV). Fall back to the dim-
    # blanked remaining text when the metric side names nothing.
    # ``_metric_slot`` truncates a FILTER clause off the subject side, so a field
    # named as a condition cannot capture the metric: "balance where LTV above
    # 50%" is a balance question with an LTV condition, and used to resolve to
    # weighted-average LTV purely because ``ltv`` precedes ``balance`` in
    # ``_METRIC_TERMS``. Grouping clauses are already excluded — ``metric_part``
    # is the pre-"by" text and ``remaining`` is dimension-blanked.
    metric, agg, _matched = _detect_metric(_metric_slot(metric_part), semantics)
    if metric is None and not _matched:
        metric, agg, _ = _detect_metric(_metric_slot(remaining), semantics)
    if is_line:
        x = ("origination_date" if "origination_date" in _fields(semantics)
             else None)
        # A value filter alongside a trend ("balance by month where LTV above
        # 50%") is applied to the mask BEFORE the time-series grouping (the
        # executor filters `work` before _execute_line), so attach it — a
        # filtered trend is never silently returned unfiltered.
        line_filters, line_unavail = _grouped_value_filters(
            q, semantics, available_columns, exclude_dims=[])
        # If a FILTER-field keyword hijacked the metric (e.g. "balance trend where
        # LTV above 50%" -> metric=LTV, because the LTV filter term is also read as
        # a metric) but a balance measure is explicitly named, prefer balance so
        # the trend is a balance trend — NOT the filter field's trend. Never
        # override a legitimately-resolved measure (e.g. forecast_funded_balance).
        if (re.search(r"\b(balance|outstanding|exposure)\b", q) and not _wants_count(q)
                and (metric is None or metric in line_filters)):
            metric, agg = _balance_metric(semantics, available_columns), "sum"
        # Loan/case COUNT evolutions stay a COUNT metric (not balance/sum): "loan
        # count evolution", "number of loans by reporting month", "case count by
        # week" all resolve to a governed count time-series.
        if _wants_count(q) or agg == "count":
            metric, agg = None, "count"
        elif metric is None:
            metric, agg = _balance_metric(semantics), "sum"
        return (MIQuerySpec(
            intent="chart", chart_type="line", x=x, metric=metric,
            aggregation=agg, filters=line_filters, unavailable_filters=line_unavail,
            title=title, explanation="Line chart of a metric over time.",
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
        # "<dimension> by <metric>" or count-by-dimension.
        # Read from the DIMENSION-BLANKED text, not the raw question: scanning
        # ``q`` here let the grouping term supply the measure, so "concentration
        # by LTV bucket" answered weighted-average LTV per bucket instead of the
        # balance concentration it asks for. ``remaining`` still carries a
        # measure named on the far side of "by" ("region by balance"), which is
        # the phrasing this branch exists to serve.
        metric, agg, _ = _detect_metric(_metric_slot(remaining), semantics)

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
        # An explicit portfolio-summary / count intent keeps the whole-book
        # summary. ANYTHING ELSE is marked "unmapped" so the workflow returns a
        # controlled "I couldn't interpret this" response instead of silently
        # answering a different question with a whole-book KPI.
        wants_summary = (bool(_SUMMARY_INTENT_RE.search(q)) or is_count_q
                         or is_balance_q or _wants_count(q))
        # P1A — "what percentage of the book is <predicate>" names no measure,
        # because "the book" IS the basis. It reached here and was answered as a
        # WHOLE-BOOK summary, which is the opposite of the question: a share is
        # the filtered population OVER the book, not the book itself. Resolve it
        # to a governed share before the summary default claims it.
        share = _share_request(q, semantics, available_columns)
        if share and not (re.search(r"\bby\b", q) or dim_terms):
            share_filters, share_unavail = _grouped_value_filters(
                q, semantics, available_columns)
            if share_filters:
                return (MIQuerySpec(
                    intent="summary", chart_type="none",
                    metric=share[1] or _balance_metric(semantics, available_columns),
                    aggregation="share", title=title,
                    filters=share_filters, unavailable_filters=share_unavail,
                    explanation="Share of the whole book meeting the stated "
                                "condition (filtered population over total).",
                    output_format="table"),
                    _det_meta("medium", explicit, dim_terms))
        if wants_summary:
            return (MIQuerySpec(
                intent="summary", chart_type="none", aggregation="count", title=title,
                explanation="Whole-book portfolio summary (count + balance).",
                output_format="text"),
                _det_meta("medium", explicit, dim_terms, note="portfolio_summary"))
        return (MIQuerySpec(
            intent="summary", chart_type="none", aggregation="count", title=title,
            explanation="Could not map question to a governed analytic.",
            output_format="text"),
            _det_meta("low", explicit, dim_terms, note="unmapped"))

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
        # P1A — a single-metric KPI carries its predicate. Until now this branch
        # built the spec with NO filters, so "average LTV in London" and
        # "exposure to borrowers over 85" were parsed as whole-book KPIs: the
        # predicate was extracted upstream and then discarded here, before the
        # spec existed. The filtered-KPI branch above only runs for counting and
        # total-balance phrasings, which is why "how many loans over 85" worked
        # while "exposure to borrowers over 85" did not. Same helper the bar
        # branch uses, so an unavailable predicate is surfaced, never dropped.
        kpi_filters, kpi_unavail = _grouped_value_filters(
            q, semantics, available_columns)
        share = _share_request(q, semantics, available_columns)
        if share and kpi_filters:
            # "What proportion of the book is below 75% LTV" is a ratio of two
            # populations, not a KPI of one. The filter defines the numerator;
            # the denominator is the same measure over the whole book.
            _, metric = share
            agg, weight = "share", None
        return (MIQuerySpec(
            intent="summary", chart_type="none", metric=metric, aggregation=agg,
            weight_field=weight, title=title,
            filters=kpi_filters, unavailable_filters=kpi_unavail,
            explanation=f"{agg} of {metric} (single KPI; no grouping dimension "
                        "requested).",
            output_format="table"),
            _det_meta("medium" if (explicit or kpi_filters) else "low",
                      explicit, dim_terms))

    if metric is None:
        # NEVER substitute a different measure for one the user named. A grouped
        # question whose metric side carries an unresolvable noun phrase ("the
        # unicorn ratio by region") used to default to balance and answer with
        # ok:true — a confident answer to a question nobody asked. Refuse it,
        # naming the term, so the governed response is traceable.
        residue = _metric_side_residue(metric_part, semantics, available_columns)
        if residue:
            return (MIQuerySpec(
                intent="summary", chart_type="none", aggregation="count",
                title=title, dimension=None,
                explanation=f"'{residue}' is not a governed measure in this "
                            "dataset; no substitute was used.",
                output_format="text"),
                _det_meta("low", explicit, dim_terms, note="unresolved_metric"))
        # "What percentage of the book is ..." names no measure because "the
        # book" IS the measure — the governed default basis for a share is the
        # balance. Resolve it here rather than falling through to a bar chart,
        # which would answer a two-population question with a breakdown.
        share = _share_request(q, semantics, available_columns)
        if share and not grouping_requested:
            share_filters, share_unavail = _grouped_value_filters(
                q, semantics, available_columns)
            if share_filters:
                return (MIQuerySpec(
                    intent="summary", chart_type="none",
                    metric=share[1] or _balance_metric(semantics, available_columns),
                    aggregation="share", title=title,
                    filters=share_filters, unavailable_filters=share_unavail,
                    explanation="Share of the whole book meeting the stated "
                                "condition (filtered population over total).",
                    output_format="table"),
                    _det_meta("medium", explicit, dim_terms))
        # An explicit COUNT is a measure the user NAMED, not the absence of one.
        # Falling straight through to the balance default answered "show loan
        # count by age band" with total balance per bucket — and before the
        # metric-slot fix it answered with average borrower age, the grouping
        # field. Neither is a count. The line path already makes this
        # distinction; the bar path did not.
        if _wants_count(q):
            metric, agg = None, "count"
        else:
            metric, agg = _balance_metric(semantics, available_columns), "sum"
    weight = _default_weight(semantics, metric) if agg == "weighted_avg" else None
    conf = "high" if explicit else ("medium" if not generic else "low")
    # A value filter expressed alongside the grouping ("balance by region where
    # LTV above 50%") is applied to the mask before grouping — attach it so it is
    # never silently dropped. The grouping dimension itself is excluded.
    g_filters, g_unavail = _grouped_value_filters(
        q, semantics, available_columns, exclude_dims=[dimension] if dimension else [])
    return (MIQuerySpec(
        intent="chart", chart_type="bar", metric=metric, dimension=dimension,
        aggregation=agg, weight_field=weight, top_n=top_n, title=title,
        filters=g_filters, unavailable_filters=g_unavail,
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
    allowed_aggs | chart_roles | synonyms(<=3).

    The three-synonym cut is arbitrary and it does hide governed vocabulary:
    ``exposure`` is the fifth synonym of ``current_outstanding_balance`` and
    never reaches the model, which is why "total exposure" was parsed as
    exposure at default — ``exposure at default`` sits third on the EAD line
    and did reach it. The instruction in rule 11 settles that pair directly
    instead.

    Lifting the cut entirely was tried and REVERTED. It restores 99 synonyms
    across 51 fields, and one of them — ``origination type`` on
    ``source_portfolio_type`` — let "the credit quality of new ORIGINATION
    versus the BACK BOOK" be answered grouped by sourcing channel. Direct
    versus acquired is not new lending versus seasoned lending; the correct
    concept is vintage, which this book does not carry, so the correct outcome
    was the refusal it used to give. The numbers were right for a question
    nobody asked.

    That is a real defect in this truncation AND a real gap in the guard that
    let the substitute pass, and both want a phase of their own. Widening what
    the model can see is only safe once a wrong cohort cannot be presented as
    the right one.
    """
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
    "8. FILTERS. The key is 'filters' (plural) and it is a JSON OBJECT keyed by "
    "field key — never a list, and never a singular 'filter'. Each value is "
    "either a bare value for equality/membership, or an object "
    '{\"op\": ..., \"value\": ...}. Allowed ops: gt, ge, lt, le, eq, ne, '
    "between (value is a two-element list), in, not_in (value is a list). "
    "Examples:\n"
    '   {\"filters\": {\"geographic_region_obligor\": \"London\"}}\n'
    '   {\"filters\": {\"youngest_borrower_age\": {\"op\": \"ge\", \"value\": 85}}}\n'
    '   {\"filters\": {\"current_loan_to_value\": {\"op\": \"le\", \"value\": 75}}}\n'
    "9. PERCENT SCALE. Percent-format fields (LTV, interest rate) are expressed "
    "in PERCENTAGE POINTS, not fractions: 75% is 75, not 0.75. Write filter "
    "thresholds on those fields in points.\n"
    "10. SEVERAL MEASURES IN ONE QUESTION. When the question names more than one "
    "measure (\"balance, loan count, weighted-average LTV and rate\"), that is ONE "
    "request over ONE population — not several questions. Return them in "
    "'measures': a JSON ARRAY of objects, each {\"field\": <catalogue key>, "
    "\"aggregation\": <allowed aggregation>}. Use the key 'loan_count' for a "
    "count of loans. Keep the shared filters in 'filters' and the shared "
    "grouping in 'dimension' — they apply to every measure. Do NOT pick one "
    "measure and drop the rest, and do NOT invent a measure the question did "
    "not ask for. Example:\n"
    '   {\"intent\": \"summary\", \"chart_type\": \"none\", \"measures\": ['
    '{\"field\": \"current_outstanding_balance\", \"aggregation\": \"sum\"}, '
    '{\"field\": \"loan_count\", \"aggregation\": \"count\"}, '
    '{\"field\": \"current_loan_to_value\", \"aggregation\": \"weighted_avg\"}], '
    '\"filters\": {\"collateral_geography\": \"London\"}}\n'
    "    For a SINGLE measure keep using 'metric' + 'aggregation' as before.\n"
    "11. EXPOSURE. Bare 'exposure' language — exposure, total exposure, current "
    "exposure, portfolio exposure, book exposure, outstanding exposure — means "
    "the catalogue's CURRENT OUTSTANDING BALANCE measure. It is the same "
    "concept as balance in an MI book, and it is the field to use whenever the "
    "question says exposure without naming another exposure measure. Use "
    "'exposure_at_default' ONLY when the question explicitly says EAD or "
    "exposure at default. The two are different measures and one may be absent "
    "from the dataset, so never substitute either for the other: if the one "
    "asked for is missing, return it and let validation refuse.\n"
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

# USD pricing per 1,000,000 tokens (input, output), keyed by model-family.
# Kept current with published Anthropic list prices. Cache reads bill at 0.1x
# input and cache writes at 1.25x input (applied in ``estimate_cost``).
_PRICING = {
    "haiku": (1.00, 5.00),
    "sonnet": (3.00, 15.00),
    "opus": (5.00, 25.00),
    "fable": (10.00, 50.00),
    "mythos": (10.00, 50.00),
}

# Longest family tokens first so "opus"/"sonnet" win before any generic key,
# and so a future family whose name embeds another ("fable" vs "able") can't
# be shadowed by a shorter substring.
_PRICING_KEYS = sorted(_PRICING, key=len, reverse=True)


def _price_for_model(model: str):
    """Look up (input, output) $/1M for a model id by family token.

    Returns ``None`` for an unrecognised model and logs a warning once so an
    overridden ``MI_AGENT_LLM_MODEL`` that we have no price for surfaces as a
    'cost unknown' status rather than a silent $0 estimate.
    """
    m = (model or "").lower()
    for key in _PRICING_KEYS:
        if key in m:
            return _PRICING[key]
    if m and m not in _UNPRICED_WARNED:
        _UNPRICED_WARNED.add(m)
        logger.warning(
            "No pricing entry for model %r; cost estimate will report status "
            "'unknown'. Add its family to _PRICING to enable cost tracking.",
            model,
        )
    return None


# Models we've already warned about, so the log line fires once per process.
_UNPRICED_WARNED: set = set()


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


# Model families that REJECT sampling params (`temperature`/`top_p`/`top_k`)
# with an HTTP 400. Newer reasoning models fix their own sampling; sending
# `temperature=0.0` to them fails the request outright. When overriding
# ``MI_AGENT_LLM_MODEL`` to one of these, we must omit the sampling kwargs.
_NO_SAMPLING_MODELS = (
    "opus-4-7", "opus-4-8", "opus-4.7", "opus-4.8",
    "sonnet-5", "fable", "mythos",
)


def _supports_temperature(model: str) -> bool:
    m = (model or "").lower()
    return not any(tok in m for tok in _NO_SAMPLING_MODELS)


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
    # Determinism where the model allows it; newer models reject `temperature`
    # and are deterministic enough for strict-JSON parsing without it.
    sampling = {"temperature": 0.0} if _supports_temperature(model) else {}
    cache_supported = False
    message = None
    # NOTE: ``temperature`` is intentionally NOT sent. Newer Claude models
    # (Sonnet 5 / Opus 4.x …) reject it with a 400 "temperature is deprecated"
    # error; the model's default sampling is used. The task is a constrained
    # NL->JSON parse validated downstream, so a fixed temperature isn't needed.
    if use_cache:
        try:
            message = client.messages.create(
                model=model, max_tokens=1024,
                system=[{"type": "text", "text": prompt["system"],
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt["user"]}],
                **sampling,
            )
            cache_supported = True
        except Exception:  # pragma: no cover - SDK without cache support
            message = None
    if message is None:
        message = client.messages.create(
            model=model, max_tokens=1024,
            system=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
            **sampling,
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

#: P1M. The validator's phrasing for "this field does not permit that statistic".
#: Like a missing column, this is NOT a repairable parse error: the only way for
#: the model to satisfy it is to substitute a DIFFERENT statistic, which is
#: exactly the defect P1M exists to remove. Spending a repair call on it is how
#: "median LTV" became a weighted average and was returned as a success.
_AGG_NOT_ALLOWED_MARK = "not allowed for metric"

#: A governed population concept the spec executes that the question never asked
#: for. Not repairable by re-prompting: the model has already demonstrated it
#: will invent a population for this wording, and a second attempt may invent a
#: different one. Recovery is the deterministic interpretation, which resolves
#: the governed scope phrase correctly, or a refusal.
_FABRICATED_POP_MARK = "population not requested"


def _missing_column_only(errors: List[str]) -> bool:
    """True if every validation error is a missing-dataset-column error."""
    return bool(errors) and all(_MISSING_COL_MARK in e for e in errors)


_AGG_REJECTED_RE = re.compile(r"Aggregation '([^']+)' not allowed for metric")


def _statistic_not_permitted(errors: List[str],
                             requested: Optional[str] = None) -> bool:
    """True if a validation error refuses the statistic THE USER ASKED FOR.

    The distinction is the whole point. Two different specs fail with the same
    validation error and deserve opposite treatment:

      * "what is the median LTV?" -> the model asked for a median, the registry
        does not govern one for LTV, and the only repair available is to ask for
        a DIFFERENT statistic. That is the defect. Refuse.
      * "weighted ltv by region" -> the model returned a sum on a percent metric.
        The user asked for a weighted average; the model simply got it wrong, and
        a repair can only move the spec back TOWARDS what was asked for. Repair.

    So a permission error blocks repair only when the rejected statistic is the
    one the question named. ``requested`` of None means the question named no
    statistic, in which case no repair can move away from one.

    ``any``, not ``all``: a spec can fail for several reasons at once, and fixing
    the others would still leave a spec that can only become valid by
    substituting the statistic.
    """
    if not requested:
        return False
    for err in (errors or []):
        if _AGG_NOT_ALLOWED_MARK not in err:
            continue
        match = _AGG_REJECTED_RE.search(err)
        rejected = match.group(1) if match else None
        if rejected is None or _statistic.satisfies(requested, rejected):
            return True
    return False


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


# Layering signals — a question with any of these reads better via the LLM than
# the narrow deterministic matcher, so we do NOT short-circuit to deterministic
# for it even when the deterministic parse looks confident.
_LAYERED_COMPARISON = (
    " vs ", " vs.", "versus", "compare", "compared", "relative to", " against ",
    "difference between", "year on year", "year-on-year", "over time", "trend",
)
_LAYERED_CONDITIONAL = (
    "where", "among", "sitting on", "that have", "who have", "with high",
    "with older", "with low", "combined with", "as well as", "both ", "exposed to",
    "concentrat", "breakdown of", "split by",
)


def _is_layered_question(question: str) -> bool:
    """True when a question is multi-faceted / layered rather than a single
    deterministic lookup. Deliberately errs toward the LLM: any comparison or
    conditional phrasing, or two+ ``by`` dimension clauses, counts as layered."""
    q = f" {(question or '').lower().strip()} "
    if any(tok in q for tok in _LAYERED_COMPARISON):
        return True
    if any(tok in q for tok in _LAYERED_CONDITIONAL):
        return True
    if q.count(" by ") >= 2:  # two+ dimensions ("balance by region by vintage")
        return True
    # "and" joining two substantive clauses (not a trailing filler) — e.g.
    # "older borrowers and high LTV". Require some length on each side.
    if " and " in q:
        left, _, right = q.partition(" and ")
        if len(left.strip()) >= 8 and len(right.strip()) >= 8:
            return True
    return False


#: Specialist analytical intent the DETERMINISTIC parser detects and the LLM
#: cannot: nothing in the field catalogue it is prompted with expresses "this is
#: a covenant-headroom question" or "this is a run-rate question". The chat
#: recognisers dispatch on exactly these fields
#: (``mi_agent_api.chat_routing._register_default_recognisers``), so an LLM spec
#: that omits them silently demotes a purpose-built governed capability to a
#: generic chart — which is how "am I close to breaching any concentration
#: limits?" became a field-unavailable refusal and "what is the run rate of new
#: lending?" became a 150-point line chart.
#:
#: These are INTENT markers, never data. Carrying them forward cannot change
#: which rows or measures the LLM asked for; it only preserves the routing
#: decision the deterministic parser already made.
_SPECIALIST_INTENT_FIELDS: Tuple[str, ...] = (
    "risk_limit_query", "risk_monitor", "risk_monitor_mode", "risk_dimension",
    "risk_limit_category",
    "forecast_mode", "forecast_question", "forecast_target_value",
    "bridge_query", "cohort_progression",
    "temporal_mode", "compare_periods", "baseline_date", "current_date",
    "execution_mode", "state",
)


def reject_scope_role_filters(spec: MIQuerySpec, question: str,
                              columns=None) -> List[str]:
    """Refuse a filter that is really a SCOPE reference. Returns what it refused.

    The model reads "the funded portfolio" as a predicate and emits
    ``funded_status = "Funded"``. There is no such column: the whole governed
    tape IS the funded book, so the phrase names the population being reported
    on, not a subset of it. Left alone the query fails validation and the
    question refuses for a field that was never asked about.

    This is a ROLE rejection at normalisation, not a filter deletion. Both
    conditions must hold, and each matters:

      * the field is ABSENT from the dataset — so a genuine governed status
        field is never touched, and "funded versus unfunded applications"
        keeps its real predicate;
      * the question contains a governed scope phrase naming that concept — so
        an absent field the user genuinely asked to filter on still refuses by
        name rather than being quietly dropped.

    A filter that narrows the population can therefore never be removed here:
    an absent column narrows nothing, because the query would not have run.

    One further role rejection is handled separately below: a predicate on the
    GOVERNED SCOPE FIELD itself. ``source_portfolio_type`` IS a real column, so
    the absent-field test can never reach it, yet it is not a row predicate
    either — the portfolio scope is resolved by the governed lens, which filters
    on the explicit ids the registry holds and never on the type string. Two
    scopes for one question is one too many, and the redundant one is the model's.
    """
    from .portfolio_lens import (  # local: avoids a cycle
        LENS_ACQUIRED, LENS_DIRECT, LENS_TOTAL, SOURCE_TYPE_FIELD,
        names_total_scope, resolve_lens, scope_phrase_spans,
    )

    filters = getattr(spec, "filters", None)
    if not isinstance(filters, dict) or not filters:
        return []
    available = {str(c) for c in (columns or ())}
    if not available:
        return []
    fields = _fields(_SEMANTICS_FOR_SCOPE or {})
    text = (question or "").lower()
    scoped = " ".join(text[a:b] for a, b in scope_phrase_spans(text))
    if not scoped.strip():
        return []
    rejected: List[str] = []

    # -- the governed scope field itself ---------------------------------- #
    # "the direct book" names the scope. The governed lens resolves it to the
    # registry's explicit portfolio ids; a model-emitted
    # ``source_portfolio_type = "direct"`` alongside that is a second, coarser
    # scope expressed as a row predicate, and it defeats the mechanism whose
    # whole point is that a group answers as the sum of its members. It is
    # refused only when it is EXACTLY REDUNDANT — the same question resolves to
    # that same type lens — so the governed id filter that replaces it comes
    # from the identical phrase and the rejection cannot change the intended
    # population. The id field is never rejected here: it is the finer grain,
    # so dropping it could widen, and nothing above may widen.
    scope_lens = resolve_lens(question)
    if (SOURCE_TYPE_FIELD in filters
            and getattr(scope_lens, "name", None) in (LENS_DIRECT, LENS_ACQUIRED)
            and str(filters[SOURCE_TYPE_FIELD]).strip().lower() == scope_lens.name):
        filters.pop(SOURCE_TYPE_FIELD, None)
        rejected.append(
            f"{SOURCE_TYPE_FIELD} (scope resolved by governed lens "
            f"'{scope_lens.name}', not a predicate)")

    # -- an explicit FULL-AuM scope phrase overrides a narrower type ------ #
    # "the sponsored book" / "the whole book" / "the entire portfolio" name the
    # client's full AuM. When the question resolves to Total AND carries an
    # explicit full-AuM phrase, a model-emitted ``source_portfolio_type``
    # predicate is the same scope misread as above — here it narrows the full
    # book to one type. The governed intent is the widening, so the narrower
    # predicate is refused, matching the deterministic path, which produces no
    # type filter for these phrases at all. This is the ONE place a widening is
    # correct: it is not an accident to be prevented but the stated scope, so
    # the "nothing may widen" rule of the block above does not apply.
    if (SOURCE_TYPE_FIELD in filters
            and getattr(scope_lens, "name", None) == LENS_TOTAL
            and names_total_scope(question)):
        filters.pop(SOURCE_TYPE_FIELD, None)
        rejected.append(
            f"{SOURCE_TYPE_FIELD} (explicit full-AuM scope, not a predicate)")

    for key in list(filters):
        canonical = (fields.get(key, {}) or {}).get("canonical_field", key)
        if key in available or canonical in available:
            continue          # a real column: whatever it means, it is not ours
        stem = str(key).split("_")[0].lower()
        value = str(filters[key]).strip().lower()
        if stem and (stem in scoped or (value and value in scoped)):
            filters.pop(key, None)
            rejected.append(f"{key} (scope phrase, not a predicate)")
    return rejected


def resolve_statistic_role(spec: MIQuerySpec, question: str,
                           semantics=None) -> Optional[str]:
    """Make the statistic the question NAMED the statistic the spec carries.

    P1M. Two parsers lost the requested statistic in two different ways, and both
    ended in the same wrong number.

    The deterministic parser has no vocabulary for "median", so the field's
    default aggregation was applied and nothing recorded that a different
    statistic had been asked for — the request never reached the governance layer
    that would have refused it. This function puts it there.

    The rule is deliberately one-directional: it only overwrites the spec's
    aggregation when the aggregation ALREADY THERE does not satisfy what the
    question asked for. So "what is the average LTV?" keeps ``weighted_avg`` —
    the house convention for a ratio measure is what a plain "average" means, and
    rewriting it to a simple mean would be its own silent substitution.

    When the named statistic is not permitted for the field, the spec is left
    carrying the statistic the USER asked for rather than the one the registry
    would default to. Validation then refuses it. That is the point: a request
    the product cannot honour must fail loudly at the boundary instead of being
    quietly rounded to the nearest thing that works.

    Returns the statistic named, for the receipt and the facet ledger.
    """
    # A grouped spec makes a superlative a RANKING over groups, not a statistic
    # on the measure — "which region has the highest LTV" wants the winning
    # region, not one extreme loan. The ranking facet already owns that.
    grouped = bool(getattr(spec, "dimension", None)
                   or (getattr(spec, "dimensions", None) or [])
                   or (getattr(spec, "hierarchy", None) or []))
    named = _statistic.statistic_named(question, grouped=grouped)
    if not named or spec is None:
        return None
    metric = getattr(spec, "metric", None)
    if not metric:
        return named
    # A MEASURE SET carries a statistic per measure, and the one the question
    # names belongs to the measure it modifies — "balance, loan count and
    # weighted average LTV" asks for a weighted average of the LTV, not of the
    # balance. Forcing the question-level statistic onto whichever measure
    # happens to be primary refused a question that answers correctly. P1E
    # already guards that no measure is dropped, and the statistic facet still
    # checks identity across the whole executed set.
    if len([m for m in (getattr(spec, "measures", None) or []) if m.get("field")]) > 1:
        return named
    # The same reasoning read from the QUESTION rather than the spec. A request
    # for more measures than the contract carries is refused by P1E, and the spec
    # that reaches here is already degraded — attributing the question's
    # statistic to whichever measure survived produced a refusal that named the
    # wrong problem ("weighted average Valuation is not governed") for a question
    # whose actual fault was asking for six measures.
    try:
        from .execution_receipt import named_measure_concepts  # local: cycle

        if len(named_measure_concepts(question)) > 1:
            return named
    except Exception:  # noqa: BLE001 - the guard must never break a parse
        pass
    entry = ((semantics or {}).get("fields") or {}).get(metric) or {}
    current = getattr(spec, "aggregation", None)
    if _statistic.satisfies(named, current):
        # A plain "average" is satisfied by EITHER governed averaging statistic,
        # but WHICH one it means is the field's house convention, not whichever
        # the parser picked on the day. "What is the average LTV in London?" came
        # back as a simple mean of 39.6193 while the same question over the whole
        # book gave the governed weighted average of 43.1562 — one phrasing, two
        # statistics, 7% apart. Normalising a bare mean to the field's default
        # makes the convention deterministic across both parser paths, and leaves
        # measures whose default IS the simple mean (borrower age) untouched.
        if named == _statistic.MEAN:
            governed = _statistic.concrete_for(named, entry)
            if governed and current != governed:
                spec.aggregation = governed
        return named
    # NB: ``loan_level`` is deliberately NOT rewritten to a min/max, even though
    # a superlative question looks like it wants one. "Largest loan balance" is
    # an established governed LOAN-LEVEL RANKING — a table of the biggest loans,
    # with intent=table and sort_direction=desc — and converting it to a scalar
    # deleted that capability (caught by test_mi_ranking_matrix and the
    # calibration bank). The extreme-value STATISTIC is reached through
    # "maximum"/"minimum"; the ranking TABLE is reached through "largest"/
    # "highest". Both are governed, and neither is allowed to eat the other.
    # "Which region contributes most to the weighted average LTV" names a
    # weighted average, but the spec's aggregation is a CONTRIBUTION — the
    # decomposition of exactly that weighted average. Rewriting it to a plain
    # weighted average would destroy the analytic the question asked for.
    if str(current or "") in _statistic.ANALYTIC_MODES:
        return named
    concrete = _statistic.concrete_for(named, entry)
    # ``concrete`` is None exactly when the registry does not permit the
    # statistic. Carrying the raw request forward is what makes validation
    # refuse instead of the default silently standing in for it.
    spec.aggregation = concrete or (named if named != _statistic.MEAN else "avg")
    return named


def resolve_seasoning_role(spec: MIQuerySpec, question: str,
                           columns=None) -> Optional[Dict[str, Any]]:
    """Make a named seasoning POPULATION a filter, not a grouping. Returns it.

    "What is the average LTV of the back book?" names the population being
    reported on. Resolved as a grouping it answered for the front book AND the
    back book — both segments, 11,035 loans — while presenting itself as an
    answer about the back book. That is a silent semantic error of exactly the
    kind P0/P1G exist to stop, so the ROLE is decided here, at normalisation,
    before the spec is validated or executed.

    A question naming BOTH sides is a comparison and is left alone: the grouping
    is what makes it answerable. Nothing is deleted that could narrow a
    population — the segment moves from the grouping to the filter that
    expresses it, and the receipt then states which population ran.

    What changed, and why
    ---------------------
    This used to read `_SEGMENT_PHRASES` through `resolve_segment_population`,
    which knows front book and back book and nothing finer. "New lending" named
    no population, so it fell through to the dimension reader and "what is the
    balance of new lending?" answered over 11,035 loans (B13). The vocabulary
    that does know it — the governed lending windows — was read only by the
    analytical intent layer, which runs only for composite plans, so for a
    simple question nobody took the decision at all.

    The role is now `seasoning.resolve_population_predicate`'s to make, for
    every reader. This applies its answer to the spec and nothing else.
    """
    from . import seasoning as _seasoning

    if spec is None:
        return None
    predicate = _seasoning.resolve_population_predicate(question, columns)
    if not predicate:
        return None

    filters = getattr(spec, "filters", None)
    if not isinstance(filters, dict):
        filters = {}
    for field, value in predicate.items():
        if field in filters:
            # Already expressed as a population — but a model spells the value
            # its own way ("back book", "backbook", "BACK"). The governed values
            # are "Front Book" / "Back Book", and a value that matches NO ROWS
            # would answer a seasoning question with an empty book. So it is
            # canonicalised to what the question named; anything naming neither
            # is left alone to fail visibly rather than be guessed at.
            if field == _seasoning.SEASONING_SEGMENT_FIELD:
                current = str(filters[field]).strip().lower().replace(" ", "")
                for governed in (_seasoning.FRONT_BOOK, _seasoning.BACK_BOOK):
                    if current == governed.lower().replace(" ", ""):
                        filters[field] = governed
                        break
            continue
        filters[field] = value
    spec.filters = filters

    # The population named the rows, so it is not also the axis. Any OTHER
    # requested grouping (region, vintage) is untouched.
    #
    # The seasoning AXIS is stripped whatever field the predicate is on. "New
    # lending" narrows on `months_on_book`, but the phrase the reader used is
    # still seasoning wording, and the parser had already put
    # `seasoning_segment` on the axis from it. Stripping only the predicate's
    # own fields left the narrowing correct and the breakdown standing, so
    # "balance of new lending" came back as £X for months_on_book <= 1 GROUPED
    # BY seasoning segment — the right rows cut by an axis nobody asked for.
    strip = set(predicate) | {_seasoning.SEASONING_SEGMENT_FIELD}
    for attr in ("dimensions", "hierarchy"):
        values = getattr(spec, attr, None)
        if isinstance(values, list):
            setattr(spec, attr, [d for d in values if d not in strip])
    if getattr(spec, "dimension", None) in strip:
        spec.dimension = None
    return predicate


#: Set by ``parse_with_repair`` so the scope-role check can resolve canonical
#: field names without changing its signature.
_SEMANTICS_FOR_SCOPE: Optional[dict] = None


def carry_specialist_intent(llm_spec: MIQuerySpec, det_spec: MIQuerySpec) -> List[str]:
    """Copy specialist ROUTING intent from the deterministic spec onto the LLM's.

    Only fields the LLM left unset are filled, so a spec that genuinely
    expresses one of these keeps its own value. Returns the names carried, for
    parser metadata and tests.

    This is the precedence rule in one place: a specialist capability the
    deterministic parser positively recognised may not be shadowed by a generic
    LLM spec. It runs below every channel (React, Copilot, workflow, harness)
    because they all parse here.
    """
    carried: List[str] = []
    for field_name in _SPECIALIST_INTENT_FIELDS:
        det_value = getattr(det_spec, field_name, None)
        if not det_value:
            continue
        if getattr(llm_spec, field_name, None):
            continue          # the LLM expressed it itself — leave it alone
        try:
            setattr(llm_spec, field_name, det_value)
        except Exception:  # noqa: BLE001 - never let this break a parse
            continue
        carried.append(field_name)
    carried.extend(reconcile_threshold_operators(llm_spec, det_spec))
    carried.extend(carry_measure_set(llm_spec, det_spec))
    carried.extend(reconcile_measure_aggregations(llm_spec, det_spec))
    return carried


def carry_measure_set(llm_spec: MIQuerySpec, det_spec: MIQuerySpec) -> List[str]:
    """Carry a governed measure SET the model returned as a single metric.

    The same precedence rule as ``carry_specialist_intent``: something the
    deterministic parser positively recognised may not be shadowed by a more
    generic LLM spec. This is the original P1E defect on the LLM path — the
    model understood the question, said so in its explanation, and returned one
    metric because the contract had one slot. It now has the array, but if it
    still answers with a single metric the two parsers would disagree about the
    same sentence, and the LLM path would refuse a question the deterministic
    path answers.

    Deliberately narrow. The set is carried ONLY when the model expressed no
    measure of its own, or expressed one the deterministic set already contains
    — so this can never re-point the question at a measure the model did not
    name.
    """
    det_measures = [m for m in (getattr(det_spec, "measures", None) or [])
                    if isinstance(m, dict) and m.get("field")]
    if len(det_measures) < 2:
        return []
    llm_measures = [m for m in (getattr(llm_spec, "measures", None) or [])
                    if isinstance(m, dict) and m.get("field")]
    if len(llm_measures) > 1:
        return []                     # the model expressed a set — leave it
    det_fields = {str(m["field"]) for m in det_measures}
    if llm_measures and str(llm_measures[0]["field"]) not in det_fields:
        return []                     # the model saw a different measure
    single = getattr(llm_spec, "metric", None)
    if single and str(single) not in det_fields:
        return []

    llm_spec.measures = [dict(m) for m in det_measures]
    first = llm_spec.measures[0]
    llm_spec.metric = None if first["field"] == "loan_count" else first["field"]
    if first.get("aggregation"):
        llm_spec.aggregation = first["aggregation"]
    return ["measures"]


def reconcile_measure_aggregations(llm_spec: MIQuerySpec,
                                   det_spec: MIQuerySpec) -> List[str]:
    """Apply the house aggregation convention to the LLM's own measure set.

    The convention is a reading of the QUESTION's language, not a preference: a
    bare "average" on a PERCENT measure means the balance-weighted average in MI,
    which is why ``_apply_agg_intent`` resolves it that way. The model does not
    know that convention — on "average interest rate" it returned a simple mean,
    a different number from the governed one.

    Only the AGGREGATION moves, and only where both parsers named the SAME
    measure field, so this can never add, drop or re-point a measure. Where the
    model expressed something the deterministic parser did not recognise at all,
    the model's choice stands.
    """
    det_measures = {str((m or {}).get("field")): (m or {}).get("aggregation")
                    for m in (getattr(det_spec, "measures", None) or [])
                    if isinstance(m, dict) and m.get("field")}
    llm_measures = getattr(llm_spec, "measures", None) or []
    if not det_measures or not llm_measures:
        return []

    adjusted: List[str] = []
    for measure in llm_measures:
        if not isinstance(measure, dict):
            continue
        field_name = str(measure.get("field") or "")
        governed = det_measures.get(field_name)
        if not governed or measure.get("aggregation") == governed:
            continue
        measure["aggregation"] = governed
        adjusted.append(f"measure_aggregation:{field_name}")
    if adjusted and llm_measures:
        # Keep the singular slots pointing at measures[0], as normalise_measures
        # guarantees everywhere else.
        first = llm_measures[0]
        if isinstance(first, dict) and first.get("aggregation"):
            llm_spec.aggregation = first["aggregation"]
    return adjusted


def reconcile_threshold_operators(llm_spec: MIQuerySpec,
                                  det_spec: MIQuerySpec) -> List[str]:
    """Apply the house threshold convention to an LLM spec's own predicate.

    The convention is a reading of the QUESTION's language, not a preference:

        over 85 / older than 85          -> age >  85
        85 or older / at least 85 / 85+  -> age >= 85

    The deterministic parser resolves it from the wording. The LLM does not: on
    "what is my exposure to borrowers over 85?" it returned ``>= 85``, which is
    136 loans and £31.1m instead of 86 loans and £19.4m — a materially different
    answer to the same question depending on which parser happened to run.

    Only the OPERATOR moves, and only when both parsers picked the same field
    and the same number — so this can never change which field is filtered, or
    the value filtered on, or introduce a predicate the LLM did not state. The
    receipt continues to print whichever predicate actually executed.
    """
    det_filters = getattr(det_spec, "filters", None) or {}
    llm_filters = getattr(llm_spec, "filters", None) or {}
    if not det_filters or not llm_filters:
        return []

    adjusted: List[str] = []
    for field_name, det_condition in det_filters.items():
        llm_condition = llm_filters.get(field_name)
        if not isinstance(det_condition, dict) or not isinstance(llm_condition, dict):
            continue
        det_op, llm_op = det_condition.get("op"), llm_condition.get("op")
        if det_op == llm_op or not det_op or not llm_op:
            continue
        # Same direction, different strictness — that is the convention's
        # territory. ">" vs "<" is a genuine disagreement about the question and
        # is left to the P0 guard rather than silently reconciled here.
        if {det_op, llm_op} not in ({"gt", "ge"}, {"lt", "le"}):
            continue
        try:
            if float(det_condition.get("value")) != float(llm_condition.get("value")):
                continue
        except (TypeError, ValueError):
            continue
        llm_condition["op"] = det_op
        adjusted.append(f"threshold_operator:{field_name}")
    return adjusted


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
    det_seasoning = resolve_seasoning_role(det_spec, user_question, cols)
    # P1M: before validation, so an ungoverned statistic is REFUSED here rather
    # than silently replaced by the field default further down.
    det_statistic = resolve_statistic_role(det_spec, user_question, semantics)
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
            "seasoning_population": det_seasoning,
            "requested_statistic": det_statistic,
        }
        meta.update({k: det_meta[k] for k in (
            "explicit_dimension_requested", "requested_dimension_terms",
            "dimension_substituted", "parser_confidence", "note")})
        return det_spec, meta

    # No LLM at all -> deterministic only.
    if not use_llm:
        return _det_result("deterministic")

    # Zero-cost-first: skip the LLM only for genuinely SIMPLE, high-confidence
    # questions (a single-variable metric/dimension the deterministic parser
    # matches cleanly — "portfolio summary", "balance by region"). Layered or
    # multi-faceted questions ("older borrowers sitting on high LTVs", "X vs Y",
    # multiple dimensions) go to the LLM even when the deterministic parser is
    # confident, because deterministic NLQ coverage is narrow and the LLM reads
    # the intent better. Only applies when the LLM is actually available (above,
    # ``not use_llm`` already returned a deterministic result).
    if zero_cost_first:
        layered = _is_layered_question(user_question)
        if (det_vr.ok and not layered
                and det_meta["parser_confidence"] == "high"):
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

    # P1M. The question names a statistic the registry does not permit for the
    # measure. No amount of re-prompting can fix that without changing the
    # statistic, so the LLM is not asked: this refuses on the deterministic spec.
    if _statistic_not_permitted(det_vr.errors, det_statistic):
        spec, meta = _det_result("validation_failed",
                                 repair_skipped_reason="statistic_not_permitted")
        meta["status"] = ("the requested statistic is not governed for this "
                          "measure; LLM repair skipped")
        return spec, meta

    # ---- LLM path (with repair loop) --------------------------------------
    global _SEMANTICS_FOR_SCOPE
    _SEMANTICS_FOR_SCOPE = semantics
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
    llm_call_error: Optional[str] = None
    for i in range(total_tries):
        try:
            text, usage, cache_supported = _invoke(prompt, model_id, llm_callable)
        except Exception as exc:  # noqa: BLE001 - LLM call failed; deterministic is the safety net
            llm_call_error = str(exc)
            break
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
            # A scope phrase the model turned into a predicate is refused the
            # ROLE before validation sees it, so the query is never built with
            # a filter on a column the book does not have.
            scope_rejected = reject_scope_role_filters(spec, user_question, cols)
            llm_seasoning = resolve_seasoning_role(spec, user_question, cols)
            # P1M: the model can also omit a statistic the question named. Same
            # one-directional rule as the deterministic path — an aggregation
            # that already satisfies the request is left exactly as it is.
            llm_statistic = resolve_statistic_role(spec, user_question, semantics)
            vr = validate_mi_query(spec, semantics, available_columns=cols)
            errors = list(vr.errors)
            # A population the QUESTION never asked for. P1L guards the losing
            # direction; this is the mirror. The spec validates perfectly well —
            # ``seasoning_segment = back book`` is a real column and a real value
            # — so nothing downstream can tell that the user asked about the
            # whole book. Caught here, at the same normalisation seam that owns
            # the other population roles.
            fabricated = _population_mod.fabricated_concepts(
                getattr(spec, "filters", None), user_question)
            # The same rule applied to BOUNDS rather than concepts. "How does
            # recent lending compare with what we were originating earlier in
            # the year?" names no date, and the model sometimes answered it with
            # ``origination_date ge 2024-01-01`` — a population invented from
            # nothing, applied silently, and emitted only on some runs, so the
            # same question answered once and refused the next time. The
            # deterministic parse resolves "recent lending" through the governed
            # window, and the safety net below is what reaches it.
            invented = _population_mod.fabricated_bounds(
                getattr(spec, "filters", None), user_question, semantics)
            vr_ok = vr.ok and not fabricated and not invented
            if fabricated:
                errors = errors + [
                    f"{_FABRICATED_POP_MARK}: {', '.join(sorted(fabricated))} "
                    f"(the question does not request this population)"]
            if invented:
                errors = errors + [
                    f"{_FABRICATED_POP_MARK}: {', '.join(invented)} "
                    f"(the question does not support this filter)"]

        if original_error_count is None:
            original_error_count = len(errors)
        last_errors = errors
        attempts.append({"attempt": i, "ok": vr_ok,
                         "error_count": len(errors), "errors": errors})

        if vr_ok and spec is not None:
            detail = "llm" if i == 0 else "llm_repaired"
            carried = carry_specialist_intent(spec, det_spec)
            return spec, {
                "parser_mode": "llm",
                "parser_mode_detail": detail,
                "specialist_intent_carried": carried,
                "scope_role_rejected": list(scope_rejected),
                "seasoning_population": llm_seasoning,
                "requested_statistic": llm_statistic,
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

        # P1M. The model asked for a statistic the registry does not permit. A
        # repair can only satisfy that error by asking for a DIFFERENT statistic,
        # and the repaired spec then validates cleanly and is returned as a
        # success — which is precisely how "median LTV" shipped a weighted
        # average. Stop here and let the refusal stand.
        if spec is not None and _statistic_not_permitted(errors, llm_statistic):
            repair_skipped_reason = "statistic_not_permitted"
            break

        # Do not re-prompt a model that invented a population: it may invent a
        # different one. The deterministic parse already resolves the governed
        # scope phrase — that is the sanctioned recovery, and it is what the
        # safety net below applies.
        if spec is not None and any(_FABRICATED_POP_MARK in e for e in errors):
            repair_skipped_reason = "fabricated_population"
            break

        prompt = _repair_prompt(base_prompt, raw_text, errors)

    # ---- deterministic safety net ----------------------------------------
    # The LLM is primary for hard questions, but the deterministic parser is the
    # fallback for the MI Agent: when the LLM call failed outright, or produced a
    # spec that does not validate, prefer a VALID deterministic parse over a
    # broken LLM one rather than erroring the whole query.
    # The statistic block belongs to the USER'S request, not to the model's. If
    # the LLM invented an impermissible aggregation ("sum" on a percent metric)
    # for a question that asked for something else entirely, the deterministic
    # parse still honours what was actually asked and must still answer — that is
    # what the fallback is for. It is withheld only when the deterministic spec
    # would NOT honour the statistic the question named, which is the case the
    # blocker is about; a question whose own statistic is ungoverned has already
    # refused above, before any model call.
    _fallback_honours_request = (
        repair_skipped_reason != "statistic_not_permitted"
        or _statistic.satisfies(det_statistic,
                                getattr(det_spec, "aggregation", None)))
    if (det_vr.ok and repair_skipped_reason != "missing_dataset_columns"
            and _fallback_honours_request):
        spec, meta = _det_result("deterministic_fallback")
        meta["llm"] = llm_meta
        meta["repair_skipped_reason"] = repair_skipped_reason
        meta["status"] = (
            f"LLM parse unavailable ({llm_call_error}); used the deterministic parse"
            if llm_call_error
            else "LLM parse failed validation; fell back to the deterministic parse")
        return spec, meta

    if last_spec is None:
        last_spec = MIQuerySpec(
            intent="summary", chart_type="none", aggregation="count",
            title=user_question.strip(),
            explanation="LLM did not return a usable MIQuerySpec.",
            output_format="text")
    status = ("LLM output references a missing dataset column; repair skipped"
              if repair_skipped_reason
              else f"LLM call failed ({llm_call_error}); no valid deterministic parse either"
              if llm_call_error
              else "LLM output failed validation after repair attempts")
    # Even an LLM spec that failed validation must not swallow a specialist
    # intent the deterministic parser recognised: routing runs on the parsed
    # spec BEFORE workflow validation, so a covenant-headroom question whose
    # deterministic spec also fails validation still reaches the governed risk
    # route — unless the flag is dropped here.
    carried_on_failure = carry_specialist_intent(last_spec, det_spec)
    return last_spec, {
        "parser_mode": "llm",
        "parser_mode_detail": "validation_failed",
        "specialist_intent_carried": carried_on_failure,
        "ok": False,
        "validation_errors": last_errors,
        "repair_attempts": max(0, len(attempts) - 1),
        "original_error_count": original_error_count,
        "attempts": attempts,
        "model": model_id,
        "repair_skipped_reason": repair_skipped_reason,
        "llm": llm_meta,
        "status": status,
    }
