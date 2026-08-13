"""trakt_tools.handlers.readiness — the securitisation readiness surface.

Five tools, and between them they answer the four questions a first-pass
readiness review asks:

    readiness_framework     what should be assessed, and what can Trakt measure?
    readiness_metrics       measure one framework fact from the governed library
    valuation_age_profile   how old and how good is the collateral evidence?
    regulatory_readiness    would a submission be possible from this tape?
    evaluate_rule_packs     apply every rulebook to those facts, at once

The one thing this module must never do
---------------------------------------
Blur a Trakt screening threshold into an external requirement. A screening flag
is Trakt suggesting somebody look; a warehouse breach is a contractual failure.
They read alike in a summary and mean entirely different things, so every result
carries ``authority`` and ``authority_label``, internal packs FLAG while external
packs BREACH, and :func:`trakt_core.readiness.summarise` counts them separately
and refuses to total them.

No new calculation lives here
-----------------------------
``readiness_metrics`` calls the 39 registered evaluators in
``mi_agent.concentration_tests.metrics`` — the same ones the Risk Limits
workspace uses. ``valuation_age_profile`` calls ``analytics_lib.valuation_age``.
``regulatory_readiness`` reads the regime's own field universe. There is no
securitisation-specific arithmetic anywhere in Trakt, and this file is where it
would most easily have appeared.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from trakt_core.errors import ErrorCode, TraktError
from trakt_core.readiness import (
    Fact,
    evaluate_packs,
    load_framework,
    load_rule_packs,
)

from ..schema import object_schema
from ..spec import ToolInvocation
from .analysis import _apply_filters, _balance_column, _population_block
from .loans import _jsonable, _scope_block, resolve_governed_frame

_RESOURCE_PROPERTY = {
    "type": "string",
    "description": ("The portfolio resource to assess, as "
                    "'{tenant}/{kind}/{resource_id}'."),
    "pattern": r"^[^/]+/[^/]+/[^/]+$",
}

#: Ceiling on one rule-pack evaluation. Far above any realistic rulebook; a
#: request beyond it is a configuration error rather than a query.
MAX_RULES = 500


# =========================================================================== #
# readiness_framework — what should be assessed
# =========================================================================== #
FRAMEWORK_INPUT = object_schema(
    description=("The Securitisation Readiness Framework: what a first-pass "
                 "review assesses, which of it Trakt can measure "
                 "deterministically, and which tool supplies each fact."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "category": {"type": ["string", "null"],
                     "description": "Optional category filter.",
                     "default": None},
        "asset_class": {"type": ["string", "null"],
                        "description": ("Narrow to metrics applicable to this "
                                        "asset class."),
                        "default": None},
    },
    required=["resource"],
)

FRAMEWORK_OUTPUT = object_schema(
    description="The framework registry and Trakt's coverage of it.",
    properties={
        "resource": {"type": "string"},
        "framework": {"type": "string"},
        "framework_name": {"type": "string"},
        "authority": {"type": "string"},
        "authority_note": {"type": "string"},
        "categories": {"type": "object"},
        "metrics": {"type": "array", "items": {"type": "object"}},
        "coverage": {"type": "object"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "framework", "metrics", "coverage", "scope"],
)


def readiness_framework(args: Dict[str, Any],
                        inv: ToolInvocation) -> Dict[str, Any]:
    """Publish the framework so an agent can plan rather than guess.

    An agent that does not know what a readiness review covers will invent a
    checklist, and the parts it invents will be the parts nobody can reproduce.
    """
    framework = load_framework()
    if not framework.configured:
        raise TraktError(
            ErrorCode.CONFIGURATION_INVALID,
            "No securitisation readiness framework is configured on this "
            "deployment.", request_id=inv.request_id)

    asset_class = str(args.get("asset_class") or "")
    category = args.get("category")
    metrics = [m for m in framework.metrics if m.applies_to(asset_class)]
    warnings: List[str] = []
    if category:
        wanted = str(category).strip().lower()
        matched = [m for m in metrics if m.category.lower() == wanted]
        if not matched:
            warnings.append(
                f"No framework metric is in category {category!r}. Available: "
                f"{sorted({m.category for m in metrics})}.")
        metrics = matched

    inv.telemetry.returned(len(metrics))
    return {
        "resource": inv.authorised.resource.ref.key,
        "framework": framework.reference,
        "framework_name": framework.framework_name,
        "authority": framework.authority,
        "authority_note": framework.authority_note,
        "categories": dict(framework.categories),
        "metrics": [m.to_dict() for m in metrics],
        "coverage": framework.coverage(asset_class),
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# =========================================================================== #
# readiness_metrics — measure framework facts from the governed library
# =========================================================================== #
METRICS_INPUT = object_schema(
    description=("Measure one or more framework facts using the governed "
                 "concentration-test metric library. Batch-first: ask for "
                 "several in one call."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "requests": {
            "type": "array", "minItems": 1, "maxItems": 100,
            "items": {"type": "object"},
            "description": ("Objects of {metric_id, parameters?, label?}. "
                            "metric_id names a metric in the governed library."),
        },
        "filters": {"type": ["object", "null"],
                    "description": "Optional equality filters applied first.",
                    "default": None},
    },
    required=["resource", "requests"],
)

METRICS_OUTPUT = object_schema(
    description="One measured fact per request.",
    properties={
        "resource": {"type": "string"},
        "measurements": {"type": "array", "items": {"type": "object"}},
        "unresolved": {"type": "array", "items": {"type": "object"}},
        "population": {"type": "object"},
        "library_version": {"type": ["string", "null"]},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "measurements", "unresolved", "population", "scope"],
)


def _measure(df, library, metric_id: str,
             parameters: Dict[str, Any]) -> Dict[str, Any]:
    """One governed metric. The SAME evaluator the Risk Limits workspace uses."""
    from mi_agent.concentration_tests.metrics import evaluate_metric

    metric = library.get(metric_id)
    if metric is None:
        return {"available": False,
                "reason": f"{metric_id!r} is not in the governed metric library"}
    if not metric.implemented:
        return {"available": False,
                "reason": (f"{metric_id!r} is declared in the library but has no "
                           "registered evaluator, so it cannot be measured. "
                           "Report it as unavailable rather than estimating it.")}
    computation = evaluate_metric(df, library, metric, dict(parameters or {}))
    return {
        "available": computation.value is not None,
        "value": computation.value,
        "unit": computation.unit,
        "data_status": computation.data_status,
        "numerator": getattr(computation, "numerator", None),
        "denominator": getattr(computation, "denominator", None),
        "denominator_basis": getattr(computation, "denominator_basis", None),
        "loans_in_numerator": getattr(computation, "loans_in_numerator", None),
        "total_loans": getattr(computation, "total_loans", None),
        "resolved_columns": list(getattr(computation, "resolved_columns", ()) or ()),
        "notes": getattr(computation, "notes", "") or "",
        "metric_id": metric_id,
        "metric_name": metric.display_name,
        "metric_category": metric.category,
        "calculation_source": f"concentration_tests.metrics.{metric.evaluator}",
        "reason": (getattr(computation, "notes", "") or "")
                  if computation.value is None else "",
    }


def readiness_metrics(args: Dict[str, Any],
                      inv: ToolInvocation) -> Dict[str, Any]:
    """Measure framework facts. One frame access for the whole batch."""
    from mi_agent.concentration_tests.library import load_library

    raw = args.get("requests") or []
    if not raw:
        raise TraktError(ErrorCode.TOOL_INPUT_INVALID,
                         "At least one {metric_id} request is required.",
                         request_id=inv.request_id)

    df = resolve_governed_frame(inv)
    inv.telemetry.scanned(len(df))
    df = _apply_filters(df, args.get("filters"), inv)
    library = load_library()

    measurements: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            unresolved.append({"request": str(item),
                               "reason": "not an object of {metric_id, ...}"})
            continue
        metric_id = str(item.get("metric_id") or "").strip()
        if not metric_id:
            unresolved.append({"reason": "metric_id is required"})
            continue
        outcome = _measure(df, library, metric_id, item.get("parameters") or {})
        outcome["label"] = item.get("label") or metric_id
        outcome["parameters"] = dict(item.get("parameters") or {})
        if not outcome.get("available") and outcome.get("value") is None:
            unresolved.append({"metric_id": metric_id,
                               "label": outcome["label"],
                               "reason": outcome.get("reason")
                                         or "could not be measured"})
        measurements.append(outcome)

    inv.telemetry.returned(len(measurements))
    return {
        "resource": inv.authorised.resource.ref.key,
        "measurements": measurements,
        "unresolved": unresolved,
        "population": _population_block(df, _balance_column(df)),
        "library_version": getattr(library, "version", None),
        "scope": _scope_block(inv),
        "warnings": [],
    }


# =========================================================================== #
# valuation_age_profile
# =========================================================================== #
VALUATION_AGE_INPUT = object_schema(
    description=("How old and how good the collateral evidence is: valuation "
                 "age distribution, method mix and missing-valuation share, all "
                 "balance-weighted."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "stale_after_months": {
            "type": "integer", "minimum": 1, "maximum": 600,
            "description": ("Months at or beyond which a valuation counts as "
                            "stale. Defaults to the governed selection "
                            "policy's own limit."),
            "default": 24,
        },
        "filters": {"type": ["object", "null"], "default": None},
    },
    required=["resource"],
)

VALUATION_AGE_OUTPUT = object_schema(
    description="The valuation-evidence profile.",
    properties={
        "resource": {"type": "string"},
        "available": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
        "as_of": {"type": ["string", "null"]},
        "stale_balance_pct": {"type": ["number", "null"]},
        "indexed_balance_pct": {"type": ["number", "null"]},
        "missing_balance_pct": {"type": ["number", "null"]},
        "weighted_average_age_months": {"type": ["number", "null"]},
        "age_distribution": {"type": "array", "items": {"type": "object"}},
        "method_mix": {"type": "object"},
        "population": {"type": "object"},
        "notes": {"type": "array", "items": {"type": "string"}},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "available", "scope"],
)


def valuation_age_profile(args: Dict[str, Any],
                          inv: ToolInvocation) -> Dict[str, Any]:
    """Wraps ``analytics_lib.valuation_age`` — shared, not agent-specific."""
    from analytics_lib.valuation_age import valuation_age_profile as profile

    df = resolve_governed_frame(inv)
    inv.telemetry.scanned(len(df))
    df = _apply_filters(df, args.get("filters"), inv)

    outcome = profile(df, stale_after_months=int(args.get("stale_after_months")
                                                 or 24))
    warnings: List[str] = []
    if not outcome.get("available"):
        warnings.append(
            f"The valuation profile could not be built: {outcome.get('reason')}. "
            "Treat every LTV metric on this book as unverified rather than "
            "assuming the valuations are sound.")

    inv.telemetry.returned(len(outcome.get("age_distribution") or ()))
    return {
        "resource": inv.authorised.resource.ref.key,
        **{k: v for k, v in outcome.items() if k != "notes"},
        "reason": outcome.get("reason"),
        "notes": outcome.get("notes", []),
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# =========================================================================== #
# regulatory_readiness
# =========================================================================== #
REGULATORY_INPUT = object_schema(
    description=("Whether a regulatory submission could be produced from this "
                 "tape: coverage of the regime's required loan-level fields, "
                 "separating hard blockers from fields that may report a "
                 "No-Data value."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "regime": {"type": "string",
                   "description": "The disclosure regime. ESMA Annex 2 is the "
                                  "loan-level template.",
                   "default": "ESMA_Annex2"},
        "include_field_detail": {
            "type": "boolean",
            "description": "Return per-field coverage as well as the summary.",
            "default": False,
        },
    },
    required=["resource"],
)

REGULATORY_OUTPUT = object_schema(
    description="Regulatory field readiness for this snapshot.",
    properties={
        "resource": {"type": "string"},
        "regime": {"type": "string"},
        "available": {"type": "boolean"},
        "no_nd_fallback_coverage_pct": {
            "type": ["number", "null"],
            "description": ("Coverage of fields that may NOT report a No-Data "
                            "value. Anything below 100% is a hard submission "
                            "blocker."),
        },
        "overall_coverage_pct": {"type": ["number", "null"]},
        "blocking_gaps": {"type": "array", "items": {"type": "object"}},
        "degraded_fields": {"type": "array", "items": {"type": "object"}},
        "summary": {"type": "object"},
        "fields": {"type": "array", "items": {"type": "object"}},
        "population": {"type": "object"},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "regime", "available", "summary", "scope"],
)


def regulatory_readiness(args: Dict[str, Any], inv: ToolInvocation, *,
                         frame=None) -> Dict[str, Any]:
    """Would a submission be possible from this tape, and what would stop it?

    Three tiers, because the regime itself has three and conflating them turns a
    blocker into a footnote: a field with no ND fallback stops the submission
    outright; one permitting ND1-ND4 weakens the disclosure; one permitting ND5
    is often correctly inapplicable.

    ``frame`` lets an in-process caller that has already resolved the governed
    frame hand it over. ``evaluate_rule_packs`` does exactly that: resolving it a
    second time would double the cost of the readiness sweep at exactly the scale
    where it matters.
    """
    from trakt_core.regulatory_regime import BLOCKING, DEGRADED, load_regime_model

    regime = str(args.get("regime") or "ESMA_Annex2")
    model = load_regime_model(regime)
    if not model.configured:
        raise TraktError(
            ErrorCode.CONFIGURATION_INVALID,
            f"No field universe is configured for regime {regime!r}.",
            request_id=inv.request_id)

    if frame is None:
        df = resolve_governed_frame(inv)
        inv.telemetry.scanned(len(df))
    else:
        df = frame
    total = int(len(df))

    rows: List[Dict[str, Any]] = []
    for regime_field in model.fields:
        canonical = regime_field.canonical_field
        present = bool(canonical and canonical in df.columns)
        populated = int(df[canonical].notna().sum()) if present else 0
        coverage = (round(populated / total * 100, 4) if total and present
                    else (0.0 if total else None))
        rows.append({
            **regime_field.to_dict(),
            "column_present": present,
            "populated": populated,
            "missing": (total - populated) if present else total,
            "coverage_pct": coverage,
            "satisfied": bool(present and populated == total),
        })

    blocking = [r for r in rows if r["fallback_severity"] == BLOCKING]
    degraded = [r for r in rows if r["fallback_severity"] == DEGRADED]
    blocking_gaps = [r for r in blocking if not r["satisfied"]]
    degraded_gaps = [r for r in degraded if not r["satisfied"]]

    def _mean_coverage(subset: List[Dict[str, Any]]) -> Optional[float]:
        values = [r["coverage_pct"] for r in subset
                  if r["coverage_pct"] is not None]
        return round(sum(values) / len(values), 4) if values else None

    warnings: List[str] = []
    if blocking_gaps:
        warnings.append(
            f"{len(blocking_gaps)} field(s) that may NOT report a No-Data value "
            "are incomplete. No valid submission is possible until they are "
            "resolved — this is a blocker, not a quality issue.")

    inv.telemetry.returned(len(rows))
    return {
        "resource": inv.authorised.resource.ref.key,
        "regime": regime,
        "available": True,
        # The screening pack reads this key. 100% is the regime's requirement,
        # not a Trakt judgement.
        "no_nd_fallback_coverage_pct": _mean_coverage(blocking),
        "overall_coverage_pct": _mean_coverage(rows),
        "blocking_gaps": sorted(blocking_gaps,
                                key=lambda r: (r["coverage_pct"] or 0, r["code"])),
        "degraded_fields": sorted(degraded_gaps,
                                  key=lambda r: (r["coverage_pct"] or 0,
                                                 r["code"]))[:50],
        "summary": {
            "regime_fields": len(rows),
            "blocking_fields": len(blocking),
            "blocking_satisfied": len(blocking) - len(blocking_gaps),
            "blocking_gaps": len(blocking_gaps),
            "nd1_4_permitted_fields": len(degraded),
            "nd1_4_gaps": len(degraded_gaps),
            "nd5_permitted_fields": len(rows) - len(blocking) - len(degraded),
            "submission_blocked": bool(blocking_gaps),
        },
        "fields": rows if args.get("include_field_detail") else [],
        "population": _population_block(df, _balance_column(df)),
        "limitations": [
            "Field coverage is necessary for a submission and is not evidence "
            "that the portfolio is sound. Regulatory readiness and credit "
            "quality are different questions.",
            "This measures whether values are PRESENT. It does not validate "
            "them against the regime's format or permitted-value rules — see "
            "list_validation_exceptions for validation outcomes.",
            "Submission acceptance is external evidence Trakt does not hold.",
        ],
        "scope": _scope_block(inv),
        "warnings": warnings,
    }


# =========================================================================== #
# evaluate_rule_packs — one fact, many rulebooks
# =========================================================================== #
RULE_PACKS_INPUT = object_schema(
    description=("Apply every configured rulebook to the same measured facts: "
                 "warehouse criteria, Trakt's internal screening thresholds, "
                 "and any supplied securitisation criteria."),
    properties={
        "resource": _RESOURCE_PROPERTY,
        "packs": {
            "type": ["array", "null"], "items": {"type": "string"},
            "description": ("Pack ids to evaluate. Omit for every configured "
                            "pack."),
            "default": None,
        },
        "filters": {"type": ["object", "null"], "default": None},
    },
    required=["resource"],
)

RULE_PACKS_OUTPUT = object_schema(
    description="Every rulebook's verdict on the same facts.",
    properties={
        "resource": {"type": "string"},
        "framework": {"type": ["string", "null"]},
        "packs": {"type": "array", "items": {"type": "object"}},
        "results": {"type": "array", "items": {"type": "object"}},
        "summary": {"type": "object"},
        "facts_computed": {"type": "integer"},
        "rules_evaluated": {"type": "integer"},
        "population": {"type": "object"},
        "scope": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    required=["resource", "packs", "results", "summary", "scope"],
)


def evaluate_rule_packs(args: Dict[str, Any],
                        inv: ToolInvocation) -> Dict[str, Any]:
    """Evaluate every rulebook against facts computed ONCE.

    The shared computation is not only an optimisation. If each pack measured
    its own London share, a parameter drift could let the warehouse pass and the
    securitisation criteria fail on the same portfolio for a reason no reviewer
    could explain. Sharing the fact makes that impossible rather than unlikely.
    """
    from mi_agent.concentration_tests.library import load_library

    framework = load_framework()
    packs = load_rule_packs()
    wanted = {str(p) for p in (args.get("packs") or ())}
    if wanted:
        packs = [p for p in packs if p.pack_id in wanted or p.reference in wanted]

    warnings: List[str] = []
    if not packs:
        warnings.append(
            "No rule packs are configured, so no criteria were evaluated. This "
            "is not a pass: an absent rulebook has not been satisfied.")

    total_rules = sum(len(p.rules) for p in packs)
    if total_rules > MAX_RULES:
        raise TraktError(
            ErrorCode.TOOL_INPUT_INVALID,
            f"{total_rules} rules exceed the {MAX_RULES} ceiling for one call.",
            request_id=inv.request_id)

    df = resolve_governed_frame(inv)
    inv.telemetry.scanned(len(df))
    df = _apply_filters(df, args.get("filters"), inv)
    library = load_library()

    # Facts that come from a whole-portfolio profile rather than a library
    # metric are computed lazily and once, keyed by the tool that produces them.
    _profiles: Dict[str, Dict[str, Any]] = {}

    def _profile(name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        key = f"{name}:{sorted((parameters or {}).items())}"
        if key not in _profiles:
            if name == "valuation":
                from analytics_lib.valuation_age import valuation_age_profile as p
                _profiles[key] = p(df, stale_after_months=int(
                    parameters.get("max_age_months") or 24))
            elif name == "completeness":
                critical = [str(f) for f in
                            (parameters.get("critical_fields") or ())]
                present = [f for f in critical if f in df.columns]
                total = len(df)
                if not critical or not total:
                    _profiles[key] = {"available": False,
                                      "reason": "no critical fields declared"}
                else:
                    rates = [float(df[f].notna().sum()) / total * 100
                             for f in present]
                    # A field absent from the tape counts as 0% rather than
                    # being skipped: skipping it would raise the average by
                    # removing the worst case.
                    rates += [0.0] * (len(critical) - len(present))
                    _profiles[key] = {
                        "available": True,
                        "critical_field_completeness_pct":
                            round(sum(rates) / len(rates), 6)}
            elif name == "validation":
                from .provenance import _index_for
                index = _index_for(inv)
                if index is None:
                    _profiles[key] = {
                        "available": False,
                        "reason": ("this snapshot carries no lineage index, so "
                                   "validation outcomes are unknown — which is "
                                   "not the same as clean")}
                else:
                    failed = sum(1 for e in index.entries.values()
                                 if getattr(e, "validation_status", "") == "failed")
                    _profiles[key] = {"available": True,
                                      "failed_field_count": float(failed)}
            elif name == "regulatory":
                # The frame is handed over rather than re-resolved: a second
                # resolution would double the cost of the sweep at scale.
                outcome = regulatory_readiness(
                    {"resource": args.get("resource"), "regime": "ESMA_Annex2"},
                    inv, frame=df)
                _profiles[key] = {
                    "available": outcome.get("no_nd_fallback_coverage_pct")
                                 is not None,
                    "no_nd_fallback_coverage_pct":
                        outcome.get("no_nd_fallback_coverage_pct")}
            else:
                _profiles[key] = {"available": False,
                                  "reason": f"unknown fact source {name!r}"}
        return _profiles[key]

    #: Which profile supplies each named fact key.
    _FACT_SOURCE = {
        "stale_balance_pct": "valuation",
        "indexed_balance_pct": "valuation",
        "missing_balance_pct": "valuation",
        "critical_field_completeness_pct": "completeness",
        "failed_field_count": "validation",
        "no_nd_fallback_coverage_pct": "regulatory",
    }

    def compute_fact(rule) -> Fact:
        if rule.metric_id:
            outcome = _measure(df, library, rule.metric_id, rule.parameters)
            return Fact(signature=rule.fact_signature,
                        value=outcome.get("value"),
                        unit=outcome.get("unit") or "",
                        available=bool(outcome.get("available")),
                        reason=outcome.get("reason") or "",
                        detail={k: outcome.get(k) for k in
                                ("metric_id", "metric_name", "numerator",
                                 "denominator", "denominator_basis",
                                 "loans_in_numerator", "total_loans",
                                 "calculation_source")
                                if outcome.get(k) is not None})
        if rule.fact_key:
            source = _FACT_SOURCE.get(rule.fact_key)
            if source is None:
                return Fact(signature=rule.fact_signature, value=None,
                            available=False,
                            reason=f"no source for fact {rule.fact_key!r}")
            block = _profile(source, rule.parameters)
            if not block.get("available"):
                return Fact(signature=rule.fact_signature, value=None,
                            available=False,
                            reason=block.get("reason") or "unavailable")
            value = block.get(rule.fact_key)
            return Fact(signature=rule.fact_signature,
                        value=None if value is None else float(value),
                        unit=rule.unit, available=value is not None,
                        reason="" if value is not None else "not measured",
                        detail={"source": source})
        return Fact(signature=rule.fact_signature, value=None, available=False,
                    reason="the rule names neither a metric_id nor a fact_key")

    outcome = evaluate_packs(packs, compute_fact, framework=framework)
    inv.telemetry.returned(outcome["rules_evaluated"])
    inv.telemetry.count("facts_computed", outcome["facts_computed"])

    return {
        "resource": inv.authorised.resource.ref.key,
        **outcome,
        "population": _population_block(df, _balance_column(df)),
        "scope": _scope_block(inv),
        "warnings": warnings + [
            "Screening flags come from Trakt's internal thresholds and are NOT "
            "breaches of any external requirement. Check 'authority' on every "
            "result before reporting it."],
    }
