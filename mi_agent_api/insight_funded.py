#!/usr/bin/env python3
"""Sprint B — funded material change, composed by the EXISTING insight engine.

WHAT THIS MODULE IS, AND WHAT IT DELIBERATELY IS NOT.

It is a set of generators, in the same shape as ``insight_generators``: each
takes an already-computed governed output and returns either an ``Insight`` or
an ``Omission`` saying why not. It is the funded counterpart of that file and
nothing more.

It is NOT a second insight engine. There is no new ``Insight`` type, no second
selector, no second ordering rule and no second threshold hierarchy. Ordering
and capping are ``insight_engine.rank_key`` and ``insight_engine.select``; the
envelope is ``insight_contract.build_brief``; every threshold is read through
``insight_config``. A funded finding and a pipeline finding differ in what they
are about, never in what they are or in how they are chosen.

It contains NO ARITHMETIC THAT PRODUCES A PUBLISHED NUMBER. Every figure it
reports was computed by one of exactly two governed owners:

    mi_agent.period_change.workflow.run_period_change_analysis
        the measure movements, the composition shifts and the balance bridge,
        reached through ``mi_agent_api.period_change_route.analyse_period_change``;

    mi_agent.concentration_tests.evaluation.evaluate_active_tests
        the per-test current value, prior value, movement, utilisation,
        headroom and STATUS TRANSITION — that evaluator takes the snapshot PAIR
        and establishes the transition itself, so nothing here diffs anything.

The only arithmetic in this file is (a) ``abs()`` and a comparison against a
configured threshold, and (b) the ×100 that turns a governed fraction into the
points a threshold is expressed in. Neither produces a number that is published:
the threshold comparison produces a boolean, and the points conversion happens
inside a formatter, so ``metrics`` always carries the owner's value verbatim.

WHY THE THRESHOLDS EXIST HERE AT ALL. ``period_change`` states in its own output
that it has no materiality rule — "No governed materiality threshold is
configured for this portfolio, so movements are ranked by observed size only ...
No movement is described as material, significant, a breach or high risk." That
sentence is correct about the workflow and is the exact gap this sprint closes:
the funded sections of ``config/mi/insights.yaml`` ARE the configured rule, read
through the loader every other threshold in the product already uses. Each
insight records which section and which key admitted it, so "why did this
appear" is answerable from the insight alone.

THE PAIR IS NOT RESOLVED HERE. ``period_change.periods.resolve_periods`` owns
which two snapshots a comparison is between, and the concentration evaluator is
handed the pair its caller already resolved. This module chooses no date, reads
no catalogue and holds no notion of "the previous period".

NO QUESTION IS READ. Nothing here takes a question, and the governed owners it
calls do not need one: ``analyse_period_change`` defaults ``question=""`` and the
only read of it inside ``period_change`` echoes it into provenance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import insight_config as cfg
from . import insight_generators as gen
from .insight_contract import (
    FUNDED_BALANCE_ATTRIBUTION, FUNDED_BALANCE_MOVEMENT,
    FUNDED_COMPOSITION_SHIFT, FUNDED_METRIC_MOVEMENT, FUNDED_QUIET_PERIOD,
    LIMIT_STATUS_TRANSITION, OMITTED_IMMATERIAL, OMITTED_UNAVAILABLE,
    SEVERITY_ATTENTION, SEVERITY_CONCERN, SEVERITY_INFO, Insight, Omission,
)

logger = logging.getLogger("mi_agent_api.insights")

Result = Tuple[List[Insight], List[Omission]]

#: The two governed owners, named so every insight can carry the one it came
#: from and an auditor never has to guess which layer produced a figure.
OWNER_PERIOD_CHANGE = "mi_agent.period_change.workflow.run_period_change_analysis"
OWNER_LIMIT_TESTS = "mi_agent.concentration_tests.evaluation.evaluate_active_tests"

#: The governed statuses a concentration/limit test can hold. Mirrors
#: ``mi_agent.concentration_tests.models``; imported lazily at use so this module
#: stays importable without that package present.
_RISK_STATUSES = ("pass", "warning", "breach")


# --------------------------------------------------------------------------- #
# Formatting — presentation only. Nothing below is stored in ``metrics``.
# --------------------------------------------------------------------------- #
def fraction_as_pct(v: Optional[float], dp: int = 1) -> str:
    """A governed FRACTION rendered as a percentage.

    ``period_change`` expresses a relative change as ``movement / |start|`` and a
    category share as a proportion summing to one. The ×100 lives here, in a
    formatter, so the published ``metrics`` block always carries the owner's own
    value and a reader can never be shown a number this module invented.
    """
    return "—" if v is None else f"{v * 100:.{dp}f}%"


def fraction_as_points(v: Optional[float], dp: int = 1) -> str:
    """A governed share fraction rendered as percentage POINTS of movement."""
    if v is None:
        return "—"
    return f"{'+' if v >= 0 else '−'}{abs(v) * 100:.{dp}f}pp"


def signed_points(v: Optional[float], dp: int = 1) -> str:
    """A movement ALREADY expressed in points (a ``percentage_point`` unit)."""
    if v is None:
        return "—"
    return f"{'+' if v >= 0 else '−'}{abs(v):.{dp}f}pp"


def _materiality(section: str, gate: str, value: Any) -> Dict[str, Any]:
    """The audit record of WHY an insight was admitted.

    Carried on every funded insight. ``period_change`` correctly says it has no
    materiality rule of its own; this states the rule that was applied instead,
    where it came from and what it was set to.
    """
    return {"threshold_section": section, "threshold_key": gate,
            "threshold_value": value, "threshold_source": cfg.load().get("source"),
            "applied_by": "mi_agent_api.insight_funded"}


#: Keys of ``MetricChange.to_dict()`` that describe HOW a figure was produced
#: rather than the figure itself, and the keys that describe how far it can be
#: trusted. Split out of ``metrics`` so that block carries movements and nothing
#: else — the same separation ``Insight`` already draws for the pipeline types.
_METRIC_CONTEXT = frozenset({
    "display_name", "analytical_concept", "analytical_role", "temporality",
    "aggregation", "weight_field", "share_basis", "valid_population",
    "excluded_population", "start_detail", "end_detail", "confidence",
    "rationale", "significance", "flow_basis"})
_METRIC_QUALITY = frozenset({"evidence", "notes"})

_DISTRIBUTION_CONTEXT = frozenset({
    "display_name", "analytical_concept", "analytical_role", "aggregation",
    "balance_field", "categories", "largest_increases", "largest_decreases",
    "confidence", "rationale", "evidence", "notes", "status"})

_BRIDGE_CONTEXT = frozenset({
    "status", "balance_field", "identifier_field", "identifier_fields"})
_BRIDGE_QUALITY = frozenset({"evidence", "limitation"})


_EMPTY: "frozenset[str]" = frozenset()


def _blocks(owner: Any, *, context: "frozenset[str]" = _EMPTY,
            quality: "frozenset[str]" = _EMPTY
            ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """``(figures, context, quality)`` — ONE governed dict, partitioned ONCE.

    Every block of a funded insight is a projection of its owner's own
    ``to_dict()``. Nothing is renamed, rescaled or reconstructed on the way
    through, and a field the owner adds later arrives here without a change.

    The owner is read once and partitioned once, so the three blocks provably
    come from a single snapshot of a single contract. Calling ``to_dict()`` per
    block, as this did, made "one projection of one owner" a claim in a comment
    rather than a property of the code.

    Every key lands in exactly one block: whatever is not context and not
    quality is a figure.
    """
    published = owner.to_dict()
    return ({k: v for k, v in published.items()
             if k not in context and k not in quality},
            {k: v for k, v in published.items()
             if k in context and k not in quality},
            {k: v for k, v in published.items() if k in quality})


def _dates(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    return {"funded_as_of": ctx.get("as_of_date"),
            "funded_comparison": ctx.get("comparison_date")}


# --------------------------------------------------------------------------- #
# B4. Limit and concentration status transitions
# --------------------------------------------------------------------------- #
def limit_status_transitions(ctx: Dict[str, Any],
                             envelope: Optional[Mapping[str, Any]]) -> Result:
    """One insight per test whose governed status CHANGED between the dates.

    No size gate is applied and none would be correct: the threshold was already
    applied by the configured test, and the status changing is itself the
    finding. ``report_improvements`` decides only whether a recovery is shown
    alongside a deterioration.

    Proximity to a limit that did NOT change status is a different finding and
    belongs to the existing ``CONCENTRATION_PROXIMITY`` generator, which is why
    this one reports transitions only and never duplicates it.
    """
    if not envelope:
        return [], [Omission(
            LIMIT_STATUS_TRANSITION,
            "No activated concentration or limit configuration is available for "
            "this portfolio, so no limit test could be evaluated and no status "
            "transition can be reported.", OMITTED_UNAVAILABLE)]

    tests = list(envelope.get("tests") or ())
    if not tests:
        return [], [Omission(
            LIMIT_STATUS_TRANSITION,
            "The activated configuration contains no test, so there is no limit "
            "status to report.", OMITTED_UNAVAILABLE)]
    if not envelope.get("priorAvailable"):
        return [], [Omission(
            LIMIT_STATUS_TRANSITION,
            "Only one reporting date has an evaluated limit configuration, so no "
            "status transition can be established.", OMITTED_UNAVAILABLE)]

    t = cfg.thresholds("funded_limits")
    report_improvements = bool(t.get("report_improvements", True))

    out: List[Insight] = []
    unchanged = 0
    suppressed = 0
    for row in tests:
        transition = row.get("statusTransition")
        if not transition:
            unchanged += 1
            continue

        status = str(row.get("status") or "")
        prior = str(row.get("priorStatus") or "")
        deteriorated = bool(row.get("deteriorated"))
        # A transition into or out of a non-risk status (unavailable,
        # insufficient_data, expired) is a change in whether the test could be
        # evaluated, NOT a change in risk. Saying "improved" about a test that
        # simply stopped being measurable would be the worst reading available.
        evaluability = not (status in _RISK_STATUSES and prior in _RISK_STATUSES)

        if not deteriorated and not evaluability and not report_improvements:
            suppressed += 1
            continue

        if evaluability:
            severity = SEVERITY_INFO
            verb = "changed evaluation status"
        elif deteriorated:
            severity = (SEVERITY_CONCERN if status == "breach"
                        else SEVERITY_ATTENTION)
            verb = "deteriorated"
        else:
            severity = SEVERITY_INFO
            verb = "recovered"

        name = str(row.get("displayName") or row.get("testId") or "A limit test")
        summary = f"{name} moved from {prior or 'no prior status'} to {status}."
        if row.get("currentValue") is not None:
            summary += (f" Now {row.get('currentValue')}"
                        f"{'%' if row.get('unit') == 'percent' else ''}"
                        f" against a threshold of {row.get('threshold')}.")
        if row.get("headroom") is not None:
            summary += f" Headroom {row.get('headroom')}."
        if row.get("breachAmount") is not None:
            summary += f" Exceeded by {row.get('breachAmount')}."
        if evaluability:
            summary += (" This is a change in whether the test could be "
                        "evaluated, not a measured change in exposure.")

        out.append(Insight(
            insight_type=LIMIT_STATUS_TRANSITION,
            headline=f"{name} {verb}: {transition}",
            summary=summary,
            severity=severity,
            discriminator=str(row.get("testId") or name),
            metrics={
                # Verbatim from the evaluator. Nothing recomputed, nothing
                # rounded, nothing renamed.
                "current_value": row.get("currentValue"),
                "prior_value": row.get("priorValue"),
                "absolute_change": row.get("absoluteChange"),
                "percentage_point_change": row.get("percentagePointChange"),
                "relative_change_pct": row.get("relativeChange"),
                "threshold": row.get("threshold"),
                "operator": row.get("operator"),
                "unit": row.get("unit"),
                "utilization": row.get("utilization"),
                "headroom": row.get("headroom"),
                "breach_amount": row.get("breachAmount"),
                "status": status,
                "prior_status": prior,
                "status_transition": transition,
                "deteriorated": deteriorated,
            },
            components={"data_status": row.get("dataStatus"),
                        "population": row.get("population"),
                        "population_label": row.get("populationLabel"),
                        "population_basis": row.get("populationBasis")},
            methodology={
                "owner": OWNER_LIMIT_TESTS,
                "definition": row.get("definition"),
                "configuration_version": envelope.get("configurationVersion"),
                "configuration_hash": envelope.get("configurationHash"),
                "library_version": envelope.get("libraryVersion"),
                "provenance": row.get("provenance"),
                "transition_established_by": (
                    "the pair evaluator, which is handed both reporting dates "
                    "and computes the transition itself"),
                "materiality": _materiality(
                    "funded_limits", "report_improvements", report_improvements),
            },
            source_dates={"funded_as_of": envelope.get("reportingDate"),
                          "funded_comparison": envelope.get("priorReportingDate")},
            **gen._base(ctx)))

    # A SUPPRESSED IMPROVEMENT IS STILL AN OMISSION. This reported nothing when
    # findings had also been produced, so a configuration that hides recoveries
    # hid the fact that it was hiding them — the one failure mode this contract
    # exists to prevent. It is reported whether or not anything else qualified.
    omissions: List[Omission] = []
    if suppressed:
        omissions.append(Omission(
            LIMIT_STATUS_TRANSITION,
            f"{suppressed} configured limit test(s) improved and are not shown: "
            f"improvements are switched off in configuration.",
            OMITTED_IMMATERIAL))
    if out:
        return out, omissions

    omissions.append(Omission(
        LIMIT_STATUS_TRANSITION,
        (f"All {unchanged} configured limit test(s) held the same governed "
         f"status at both reporting dates.") if unchanged else
        ("No configured limit test changed governed status between the two "
         "reporting dates."),
        OMITTED_IMMATERIAL))
    return [], omissions


# --------------------------------------------------------------------------- #
# B5. Governed measure movements
# --------------------------------------------------------------------------- #
def _gate(change: Any, t: Mapping[str, Any]) -> Tuple[bool, str, Any, Any]:
    """``(qualifies, gate_key, gate_value, observed)`` for one measure.

    The gate is chosen by the measure's own UNIT, not by its name. A percentage
    -point measure is gated on the points it moved, because 2% of a 62% weighted
    -average LTV is 1.2 points and reads as a far smaller change than it is;
    everything else is gated on its relative move.
    """
    if change.movement_unit == "percentage_point":
        key = "min_change_pp"
        floor = float(t.get(key, 1.0))
        observed = change.movement_value
        return (observed is not None and abs(observed) >= floor,
                key, floor, observed)
    key = "min_relative_change_pct"
    floor = float(t.get(key, 2.0))
    observed = change.relative_change
    # ``relative_change`` is a FRACTION; the threshold is a percentage. The
    # comparison is made in the threshold's units by dividing the threshold,
    # never by multiplying the governed value.
    return (observed is not None and abs(observed) >= floor / 100.0,
            key, floor, observed)


def _severity_for(change: Any, t: Mapping[str, Any]) -> str:
    """Attention by default, concern for a large governed DETERIORATION.

    An improvement is never escalated. A book improving quickly is not a
    concern, and the governed ``interpretation`` is the only thing consulted —
    this module never decides for itself whether a direction is good.
    """
    if change.interpretation != "deterioration":
        return SEVERITY_INFO
    if change.movement_unit == "percentage_point":
        big = float(t.get("concern_change_pp", 5.0))
        observed = change.movement_value
        severe = observed is not None and abs(observed) >= big
    else:
        big = float(t.get("concern_relative_change_pct", 10.0))
        observed = change.relative_change
        severe = observed is not None and abs(observed) >= big / 100.0
    return SEVERITY_CONCERN if severe else SEVERITY_ATTENTION


def _movement_text(change: Any) -> str:
    """How much it moved, in the expression its unit makes meaningful."""
    if change.movement_unit == "currency":
        return gen.signed_money(change.movement_value)
    if change.movement_unit == "percentage_point":
        return signed_points(change.movement_value)
    if change.movement_unit == "count":
        v = change.movement_value
        return "—" if v is None else f"{'+' if v >= 0 else '−'}{abs(v):,.0f}"
    v = change.movement_value
    return "—" if v is None else f"{'+' if v >= 0 else '−'}{abs(v):,.4g}"


def _metric_insight(ctx: Dict[str, Any], change: Any, result: Any, *,
                    insight_type: str, severity: str,
                    gate_key: str, gate_value: Any) -> Insight:
    figures, context, quality = _blocks(
        change, context=_METRIC_CONTEXT, quality=_METRIC_QUALITY)
    moved = _movement_text(change)
    relative = (f" ({fraction_as_pct(change.relative_change)})"
                if change.relative_change is not None else "")
    summary = (f"{change.display_name} moved {moved}{relative}, from "
               f"{change.start_value} to {change.end_value}.")
    if change.notes:
        summary += " " + " ".join(str(n) for n in change.notes)

    return Insight(
        insight_type=insight_type,
        headline=f"{change.display_name} {gen.direction(change.movement_value)} {moved}",
        summary=summary,
        severity=severity,
        discriminator=change.field,
        # THE OWNER'S OWN PUBLISHED CONTRACT, verbatim. Projecting
        # ``MetricChange.to_dict()`` rather than re-listing its fields is the
        # point: there is no key this module could rename, rescale or drop, and
        # a field the workflow adds later arrives without a change here.
        # ``relative_change`` therefore stays a fraction and ``movement_value``
        # stays in its own unit — a consumer is told which unit rather than
        # being handed a silently converted number.
        metrics=dict(
            figures,
            # ONE KEY ADDED TO THE VERBATIM PROJECTION, and it is a label rather
            # than a number. `relative_change` is a FRACTION of the starting
            # value; a consumer reading 0.05 as five percentage points would be
            # wrong by a factor of twenty. The alternative — publishing a
            # rescaled number — is the thing this module must not do.
            relative_change_is="a fraction of the starting value"),
        components=context,
        methodology={
            "owner": OWNER_PERIOD_CHANGE,
            "calculation_version": (result.audit or {}).get("calculation_version"),
            "result_schema_version": result.result_schema_version,
            "materiality": _materiality("funded_metric_movement",
                                        gate_key, gate_value),
        },
        data_quality=quality,
        source_dates=_dates(ctx),
        **gen._base(ctx))


def metric_movements(ctx: Dict[str, Any], result: Optional[Any], *,
                     balance_field: Optional[str] = None) -> Result:
    """Every governed measure whose movement crosses its configured gate.

    The book's BALANCE is separated into its own type, because it is the one
    movement a bridge can decompose and the one a reader looks for first. Which
    field that is comes from ``period_change.calculations.BALANCE_FIELD`` — the
    governed owner of the question — never from a name match here.
    """
    if result is None:
        return [], [Omission(
            FUNDED_METRIC_MOVEMENT,
            "No governed period-change analysis is available for this portfolio.",
            OMITTED_UNAVAILABLE)]

    changes = list(result.metric_changes or ())
    if not changes:
        return [], [Omission(
            FUNDED_METRIC_MOVEMENT,
            "The governed field-selection policy admitted no measure for this "
            "portfolio at these two reporting dates.", OMITTED_UNAVAILABLE)]

    t = cfg.thresholds("funded_metric_movement")
    out: List[Insight] = []
    immaterial: List[str] = []
    incomparable: List[str] = []

    for change in changes:
        if not change.comparable:
            incomparable.append(f"{change.display_name} ({change.status})")
            continue
        qualifies, gate_key, gate_value, _observed = _gate(change, t)
        if not qualifies:
            immaterial.append(change.display_name)
            continue
        is_balance = balance_field is not None and change.field == balance_field
        out.append(_metric_insight(
            ctx, change, result,
            insight_type=(FUNDED_BALANCE_MOVEMENT if is_balance
                          else FUNDED_METRIC_MOVEMENT),
            severity=_severity_for(change, t),
            gate_key=gate_key, gate_value=gate_value))

    omissions: List[Omission] = []
    if immaterial:
        omissions.append(Omission(
            FUNDED_METRIC_MOVEMENT,
            f"{len(immaterial)} governed measure(s) moved by less than the "
            f"configured threshold and are not reported: "
            f"{', '.join(sorted(immaterial))}.", OMITTED_IMMATERIAL))
    if incomparable:
        omissions.append(Omission(
            FUNDED_METRIC_MOVEMENT,
            f"{len(incomparable)} governed measure(s) are not comparable across "
            f"these two snapshots: {', '.join(sorted(incomparable))}.",
            OMITTED_UNAVAILABLE))
    return out, omissions


# --------------------------------------------------------------------------- #
# B6. Composition shifts
# --------------------------------------------------------------------------- #
def composition_shifts(ctx: Dict[str, Any], result: Optional[Any]) -> Result:
    """The categories that gained and lost the most share, per dimension.

    WHICH categories those are is the workflow's decision, read from its own
    ``largest_increases`` / ``largest_decreases``. This generator only decides
    whether the move it already ranked first is large enough to report, so the
    brief and the distribution output can never name different leaders.

    Balance share is preferred over count share for the same reason the pipeline
    mix insight prefers it: composition matters because EXPOSURE moved, and two
    large loans replacing twenty small ones is the same count and a very
    different book. Count share is used only where the governed balance share is
    unavailable, and the basis is always stated.
    """
    if result is None:
        return [], [Omission(
            FUNDED_COMPOSITION_SHIFT,
            "No governed period-change analysis is available for this portfolio.",
            OMITTED_UNAVAILABLE)]

    distributions = list(result.distribution_changes or ())
    if not distributions:
        return [], [Omission(
            FUNDED_COMPOSITION_SHIFT,
            "The governed field-selection policy admitted no dimension for this "
            "portfolio at these two reporting dates.", OMITTED_UNAVAILABLE)]

    t = cfg.thresholds("funded_composition")
    floor_pp = float(t.get("min_share_change_pp", 5.0))
    max_cats = int(t.get("max_categories_reported", 2))

    out: List[Insight] = []
    immaterial: List[str] = []

    for dist in distributions:
        by_name = {c.category: c for c in (dist.categories or ())}
        # The workflow's own ranking, in its own order. Nothing re-sorted here.
        ranked = [by_name[n] for n in
                  (list(dist.largest_increases) + list(dist.largest_decreases))
                  if n in by_name]

        qualifying: List[Tuple[Any, float, str]] = []
        for shift in ranked:
            movement, basis = ((shift.balance_share_movement, "balance share")
                               if shift.balance_share_movement is not None
                               else (shift.count_share_movement, "count share"))
            # A share is a FRACTION summing to one; the threshold is in points.
            # Divided rather than multiplied, so no governed value is rescaled.
            if movement is not None and abs(movement) >= floor_pp / 100.0:
                qualifying.append((shift, movement, basis))
            if len(qualifying) >= max_cats:
                break

        if not qualifying:
            immaterial.append(dist.display_name)
            continue

        lead, lead_move, lead_basis = qualifying[0]
        dist_figures, dist_context, _ = _blocks(
            dist, context=_DISTRIBUTION_CONTEXT)
        parts = [f"{s.category} {fraction_as_points(m)} of {b}"
                 for s, m, b in qualifying]
        out.append(Insight(
            insight_type=FUNDED_COMPOSITION_SHIFT,
            headline=(f"{dist.display_name}: {lead.category} "
                      f"{fraction_as_points(lead_move)}"),
            summary=(f"Composition by {dist.display_name} shifted — "
                     f"{'; '.join(parts)}. Shares are measured against each "
                     f"snapshot's own total."),
            severity=SEVERITY_INFO,
            discriminator=dist.field,
            # As above: the workflow's own published shape, minus the full
            # category list, which is replaced by the qualifying categories so
            # the insight carries what it actually reported. ``basis`` is the
            # one key added, and it names which of the workflow's two share
            # measures the gate was applied to.
            metrics=dict(dist_figures,
                         basis=lead_basis,
                         shares_are="fractions of each snapshot's own total"),
            contributors={"categories": [c.to_dict() for c, _m, _b in qualifying]},
            components=dist_context,
            methodology={
                "owner": OWNER_PERIOD_CHANGE,
                "ranked_by": ("the period-change workflow's own "
                              "largest_increases / largest_decreases"),
                "materiality": _materiality("funded_composition",
                                            "min_share_change_pp", floor_pp),
            },
            source_dates=_dates(ctx),
            **gen._base(ctx)))

    omissions: List[Omission] = []
    if immaterial:
        omissions.append(Omission(
            FUNDED_COMPOSITION_SHIFT,
            f"{len(immaterial)} dimension(s) shifted by less than the configured "
            f"{floor_pp}pp threshold and are not reported: "
            f"{', '.join(sorted(immaterial))}.", OMITTED_IMMATERIAL))
    return out, omissions


# --------------------------------------------------------------------------- #
# B7. Balance attribution
# --------------------------------------------------------------------------- #
def balance_attribution(ctx: Dict[str, Any], result: Optional[Any], *,
                        balance_is_material: bool) -> Result:
    """What the balance movement decomposes into, when that movement qualified.

    There is no size gate of its own, deliberately. A bridge EXPLAINS a
    movement, so reporting one whose movement was itself immaterial would tell a
    reader why something that did not matter happened. ``balance_is_material``
    is simply whether the balance movement insight was produced.
    """
    if result is None:
        return [], [Omission(
            FUNDED_BALANCE_ATTRIBUTION,
            "No governed period-change analysis is available for this portfolio.",
            OMITTED_UNAVAILABLE)]

    bridge = result.balance_bridge
    if bridge is None:
        return [], [Omission(
            FUNDED_BALANCE_ATTRIBUTION,
            "No balance bridge was requested for this analysis.",
            OMITTED_UNAVAILABLE)]

    t = cfg.thresholds("funded_attribution")
    require_reconciliation = bool(t.get("require_reconciliation", True))

    if bridge.status != "available":
        # ``does_not_reconcile`` is the case this rule exists for: the
        # decomposition exists but does not add up to its own closing balance,
        # and presenting it as the explanation of the movement would be wrong.
        return [], [Omission(
            FUNDED_BALANCE_ATTRIBUTION,
            (bridge.limitation
             or f"The balance bridge is unavailable ({bridge.status}).")
            + (" It is reported as a limitation rather than as an explanation "
               "of the movement." if require_reconciliation else ""),
            OMITTED_UNAVAILABLE)]

    if not balance_is_material:
        return [], [Omission(
            FUNDED_BALANCE_ATTRIBUTION,
            "The balance movement did not cross its configured threshold, so its "
            "decomposition is not reported: an immaterial movement's attribution "
            "is immaterial by construction.", OMITTED_IMMATERIAL)]

    bridge_figures, bridge_context, bridge_quality = _blocks(
        bridge, context=_BRIDGE_CONTEXT, quality=_BRIDGE_QUALITY)
    summary = (
        f"New lending contributed {gen.money(bridge.new_loan_balance)} across "
        f"{bridge.new_loan_count} loan(s); exits removed "
        f"{gen.money(bridge.exited_loan_balance)} across "
        f"{bridge.exited_loan_count} loan(s); loans present at both dates moved "
        f"{gen.signed_money(bridge.continuing_movement)} across "
        f"{bridge.continuing_loan_count} loan(s).")
    if bridge.limitation:
        summary += f" {bridge.limitation}"

    return [Insight(
        insight_type=FUNDED_BALANCE_ATTRIBUTION,
        headline=(f"Balance movement decomposes to "
                  f"{gen.money(bridge.opening_balance)} → "
                  f"{gen.money(bridge.closing_balance)}"),
        summary=summary,
        severity=SEVERITY_INFO,
        discriminator=str(bridge.balance_field or "balance"),
        # The bridge's own published shape, split into the figures and the
        # identity of the key they were computed over.
        metrics=bridge_figures,
        components=bridge_context,
        methodology={
            "owner": f"{OWNER_PERIOD_CHANGE} (mi_agent.period_change.bridge)",
            "basis": ("opening → closing reconciliation over a stable loan "
                      "identifier; no loan-level value is carried"),
            "materiality": _materiality(
                "funded_attribution", "reported with the balance movement it "
                                      "explains", balance_is_material),
        },
        data_quality=bridge_quality,
        source_dates=_dates(ctx),
        **gen._base(ctx))], []


# --------------------------------------------------------------------------- #
# B8. The quiet period
# --------------------------------------------------------------------------- #
def quiet_period(ctx: Dict[str, Any], result: Optional[Any], *,
                 produced: Sequence[Insight],
                 limits_evaluated: bool) -> Result:
    """One explicit statement when nothing crossed a threshold.

    Silence and "nothing happened" must not look the same. An empty brief is
    indistinguishable from a brief that failed, so a period in which every
    governed measure held is SAID, with what was examined and against which
    thresholds, rather than left to be inferred from a blank page.

    Only ever emitted when the analysis genuinely ran: a period with no governed
    result produces an unavailability omission, not a reassuring statement.
    """
    t = cfg.thresholds("funded_brief")
    if not bool(t.get("emit_quiet_period", True)):
        return [], []
    if produced:
        return [], []
    if result is None or not result.metric_changes:
        return [], [Omission(
            FUNDED_QUIET_PERIOD,
            "No governed period-change analysis ran, so the absence of findings "
            "is not evidence that nothing changed.", OMITTED_UNAVAILABLE)]

    comparable = [m for m in result.metric_changes if m.comparable]
    if not comparable:
        return [], [Omission(
            FUNDED_QUIET_PERIOD,
            "No governed measure is comparable across these two snapshots, so "
            "the absence of findings is not evidence that nothing changed.",
            OMITTED_UNAVAILABLE)]

    m = cfg.thresholds("funded_metric_movement")
    c = cfg.thresholds("funded_composition")
    examined = (f"{len(comparable)} governed measure(s) and "
                f"{len(result.distribution_changes or ())} dimension(s)")
    limits_note = (" No configured limit test changed status."
                   if limits_evaluated else
                   " No limit configuration is activated for this portfolio, so "
                   "no limit test was evaluated.")

    return [Insight(
        insight_type=FUNDED_QUIET_PERIOD,
        headline="No material change between the two reporting dates",
        summary=(f"{examined} were compared and none crossed its configured "
                 f"materiality threshold "
                 f"({m.get('min_relative_change_pct')}% relative, "
                 f"{m.get('min_change_pp')}pp for rate measures, "
                 f"{c.get('min_share_change_pp')}pp of share for composition)."
                 f"{limits_note}"),
        severity=SEVERITY_INFO,
        discriminator="quiet",
        metrics={"measures_examined": len(comparable),
                 "dimensions_examined": len(result.distribution_changes or ()),
                 "findings": 0},
        methodology={
            "owner": OWNER_PERIOD_CHANGE,
            "basis": ("an explicit statement that the period was examined, so "
                      "that silence is never mistaken for a failed brief"),
            "materiality": _materiality("funded_brief", "emit_quiet_period",
                                        True),
        },
        data_quality={"warnings": list(result.warnings or ()),
                      "limitations": list(result.limitations or ())},
        source_dates=_dates(ctx),
        **gen._base(ctx))], []


# --------------------------------------------------------------------------- #
# Composition — the EXISTING selector, the EXISTING envelope
# --------------------------------------------------------------------------- #
def _balance_field() -> Optional[str]:
    """The governed owner of "which field is the book's balance".

    Imported lazily and at use, so this module stays importable (and its
    generators stay testable) without ``period_change`` and its pandas
    dependency present.
    """
    try:
        from mi_agent.period_change.calculations import BALANCE_FIELD
        return BALANCE_FIELD
    except Exception as exc:  # noqa: BLE001 - a missing owner is not a crash
        logger.warning("funded brief: balance field owner unavailable (%s)", exc)
        return None


def compose(result: Optional[Any], *, tenant_id: str, portfolio_id: str,
            portfolio_context: str = "total",
            limit_envelope: Optional[Mapping[str, Any]] = None,
            run_id: Optional[str] = None,
            limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The funded material-change brief, from two already-computed outputs.

    PURE. It performs no I/O, resolves no snapshot and calls no owner: both
    governed outputs are handed in. That is what makes the whole composition
    reproducible from a fixture and what keeps the resolution of the pair
    entirely outside this file.

    Ordering, per-type caps and the total limit are ``insight_engine.select``
    unchanged — the same function that orders the Weekly Portfolio Brief — and
    the envelope is ``insight_contract.build_brief`` unchanged. There is no
    second selector and no second envelope anywhere in this sprint.
    """
    from .insight_contract import build_brief
    from .insight_engine import select

    conf = cfg.load()
    ctx = {
        "tenant_id": tenant_id, "portfolio_id": portfolio_id,
        "portfolio_context": portfolio_context, "run_id": run_id,
        "as_of_date": None, "comparison_date": None,
    }
    if result is not None:
        resolution = result.period_resolution
        ctx["as_of_date"] = resolution.end_snapshot.reporting_date
        ctx["comparison_date"] = resolution.start_snapshot.reporting_date

    if result is None:
        return build_brief(
            [], [], tenant_id=tenant_id, portfolio_id=portfolio_id,
            portfolio_context=portfolio_context, as_of_date=None,
            comparison_date=None, config_source=conf.get("source"),
            status="unavailable",
            reason=("No governed period-change analysis is available for this "
                    "portfolio, so no funded change can be reported."))

    produced: List[Insight] = []
    omissions: List[Omission] = []
    failures = 0

    balance_field = _balance_field()
    # (label, callable) — each isolated, exactly as the weekly brief isolates
    # its own: one generator raising must not cost the reader the other four.
    steps: List[Tuple[str, Any]] = [
        (LIMIT_STATUS_TRANSITION,
         lambda: limit_status_transitions(ctx, limit_envelope)),
        (FUNDED_METRIC_MOVEMENT,
         lambda: metric_movements(ctx, result, balance_field=balance_field)),
        (FUNDED_COMPOSITION_SHIFT,
         lambda: composition_shifts(ctx, result)),
    ]
    for label, step in steps:
        try:
            ins, omit = step()
            produced.extend(ins)
            omissions.extend(omit)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logger.warning("funded brief: %s generator failed: %s", label, exc)
            omissions.append(Omission(
                label, "This finding could not be produced for this period.",
                "error"))

    # Attribution runs after the measures, because whether it runs at all is
    # decided by whether the movement it explains qualified.
    balance_material = any(i.insight_type == FUNDED_BALANCE_MOVEMENT
                           for i in produced)
    try:
        ins, omit = balance_attribution(ctx, result,
                                        balance_is_material=balance_material)
        produced.extend(ins)
        omissions.extend(omit)
    except Exception as exc:  # noqa: BLE001
        failures += 1
        logger.warning("funded brief: attribution generator failed: %s", exc)
        omissions.append(Omission(FUNDED_BALANCE_ATTRIBUTION,
                                  "This finding could not be produced for this "
                                  "period.", "error"))

    # The quiet statement runs last and only when nothing else qualified, which
    # is why it is passed what was produced rather than deciding for itself.
    try:
        ins, omit = quiet_period(
            ctx, result, produced=produced,
            limits_evaluated=bool(limit_envelope
                                  and (limit_envelope.get("tests") or ())))
        produced.extend(ins)
        omissions.extend(omit)
    except Exception as exc:  # noqa: BLE001
        failures += 1
        logger.warning("funded brief: quiet-period generator failed: %s", exc)

    kept, capped = select(produced, limits=limits or conf.get("brief"))
    omissions.extend(capped)

    return build_brief(
        kept, omissions, tenant_id=tenant_id, portfolio_id=portfolio_id,
        portfolio_context=portfolio_context, as_of_date=ctx["as_of_date"],
        comparison_date=ctx["comparison_date"], run_id=run_id,
        config_source=conf.get("source"),
        status="partial" if failures else "success",
        reason=(f"{failures} finding type(s) could not be produced."
                if failures else None),
        source_dates={
            "funded_as_of": ctx["as_of_date"],
            "funded_comparison": ctx["comparison_date"],
            "period_resolution": result.period_resolution.to_dict(),
            "dataset_provenance": [dict(d) for d in result.dataset_provenance],
            "limit_configuration_version": (limit_envelope or {}).get(
                "configurationVersion"),
        })
