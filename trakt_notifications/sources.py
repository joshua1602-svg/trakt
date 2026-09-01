#!/usr/bin/env python3
"""The only module in this package that talks to MI — and it only reads.

Every governed input a notification needs is resolved here, once, and handed to
the message builders as plain data. That containment is the point: if this
module is the sole importer of ``mi_agent_api``, then no message builder, no
card and no delivery worker can reach a calculation even by accident.

Two further rules hold throughout:

* **Nothing is computed.** Each resolver calls an existing governed service and
  returns its output unchanged. Where a service is unavailable, the failure is
  recorded as a named unavailable capability rather than substituted — a Risk
  Review that cannot tell "clear" from "not checked" is the specific failure
  this design exists to prevent.

* **One resolution per update, not per recipient.** The batch is built once and
  rendered for however many authorised recipients there are, so adding a
  recipient costs a card render, not an MI run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("trakt_notifications.sources")

# Named capabilities, so an unavailable one can be reported precisely.
CAP_WEEKLY_BRIEF = "weekly portfolio brief"
CAP_CONCENTRATION = "concentration tests"
CAP_FUNNEL = "origination funnel and conversion"
CAP_PIPELINE_EVOLUTION = "pipeline five-week comparison"
CAP_FUNDED_MOVEMENT = "funded period movement"
CAP_CONTRIBUTORS = "concentration contributor attribution"
CAP_MONTHLY_BRIEF = "monthly funded brief"

#: Risk domains a Risk Review MUST have evaluated before it may state a clear
#: position. Concentration is the only one today: it is the domain that carries
#: the contractual limits, so "no material risks" said without it is a claim the
#: evidence does not support.
#:
#: This exists because the failure it prevents was REACHABLE, not hypothetical. A
#: funded-only approval resolved no pipeline side, concentration was therefore
#: never computed AND never recorded as unavailable, and the Risk Review emitted
#: the unqualified clear statement on the strength of a check that did not run.
#: Listing the domain here makes the omission self-reporting, so a future change
#: that stops resolving one degrades the message instead of silently widening
#: what it claims.
REQUIRED_RISK_CAPABILITIES = (CAP_CONCENTRATION,)


@dataclass
class GovernedInputs:
    """Everything the two messages are built from, and nothing else.

    ``unavailable`` is as load-bearing as the data: it is what allows the Risk
    Review to say "no material risks were identified from the checks that were
    available" instead of the stronger claim it would not be entitled to make.
    """

    tenant_id: str
    portfolio_id: str
    portfolio_context: str = "total"

    brief: Optional[Dict[str, Any]] = None
    concentration: Optional[Dict[str, Any]] = None
    funnel: Optional[Dict[str, Any]] = None
    #: Carries the governed weekly series AND the ``fiveWeekAverage`` block.
    #: Current levels are read from that block rather than from a separately
    #: resolved snapshot, so the level and the average it is compared against
    #: can never come from two different resolutions of the same week.
    pipeline_evolution: Optional[Dict[str, Any]] = None
    funded_movement: Optional[Dict[str, Any]] = None
    #: The governed MONTHLY insight set, the funded analogue of ``brief``. Kept
    #: separate rather than merged into it: a weekly pipeline insight and a
    #: monthly funded insight describe different periods over different
    #: populations, and one list would let a consumer read a pipeline date onto
    #: a funded figure.
    funded_brief: Optional[Dict[str, Any]] = None
    #: test_id -> governed contributor aggregates (never per-case drivers).
    contributors: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    #: capability name -> why it could not be used.
    unavailable: Dict[str, str] = field(default_factory=dict)
    #: Approved run ids that produced these outputs.
    run_ids: List[str] = field(default_factory=list)
    source_dates: Dict[str, Any] = field(default_factory=dict)
    methodology_versions: Dict[str, Any] = field(default_factory=dict)

    def missing(self, capability: str) -> bool:
        return capability in self.unavailable

    def insights(self) -> List[Dict[str, Any]]:
        return list((self.brief or {}).get("insights") or [])

    def insight(self, insight_type: str) -> Optional[Dict[str, Any]]:
        return next((i for i in self.insights()
                     if i.get("insight_type") == insight_type), None)

    def omission(self, insight_type: str) -> Optional[Dict[str, Any]]:
        return next((o for o in ((self.brief or {}).get("omitted") or [])
                     if o.get("insight_type") == insight_type), None)

    def funded_insights(self) -> List[Dict[str, Any]]:
        return list((self.funded_brief or {}).get("insights") or [])

    def funded_insight(self, insight_type: str) -> Optional[Dict[str, Any]]:
        return next((i for i in self.funded_insights()
                     if i.get("insight_type") == insight_type), None)


def _safe(inputs: GovernedInputs, capability: str,
          fn: Callable[[], Any]) -> Any:
    """Resolve one governed input; a failure disables that capability only.

    Deliberately records the capability as unavailable rather than letting the
    exception escape. One MI service being down must cost the reader that one
    section, clearly labelled — not the whole notification, and never a
    silently shorter one.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - one input, not the batch
        logger.warning("notifications: %s unavailable (%s)", capability, exc)
        inputs.unavailable[capability] = (
            f"{capability} could not be resolved for this update")
        return None


def resolve(*, tenant_id: str, portfolio_id: str,
            portfolio_context: str = "total",
            pipeline_root: Optional[str] = None,
            output_root: Optional[str] = None,
            funded_run_id: Optional[str] = None,
            want_pipeline: bool = True,
            want_funded: bool = True,
            scope: Any = None) -> GovernedInputs:
    """Resolve every governed input for one approved update.

    ``want_pipeline`` / ``want_funded`` reflect what was actually approved. A
    funded-only approval does not resolve the weekly pipeline services, both
    because it would be wasted work and because a funded card must not quietly
    acquire a pipeline date it was not approved against.
    """
    inputs = GovernedInputs(tenant_id=tenant_id, portfolio_id=portfolio_id,
                            portfolio_context=portfolio_context)

    # The SAME root resolution the dashboard uses, so a notification and the
    # workspace read the identical governed dataset for the identical week.
    from mi_agent_api import datasets as datasets_mod
    if pipeline_root is None:
        pipeline_root = _safe(inputs, "pipeline root",
                              datasets_mod._pipeline_discovery_root)
    if output_root is None:
        output_root = _safe(inputs, "funded output root",
                            datasets_mod._onboarding_output_root)

    # Concentration is resolved for EVERY update type, before either side, and
    # exactly once. It is a funded-book measure (it reads ``output_root``), and
    # the Risk Review needs it whatever was approved — a monthly funded update is
    # the one most likely to move a concentration test, and was previously the
    # one that never evaluated any. Resolving it here rather than inside the
    # pipeline side also means a combined update pays for it once.
    _resolve_concentration(inputs, output_root, funded_run_id, scope)

    if want_pipeline:
        _resolve_pipeline_side(inputs, pipeline_root, output_root,
                               funded_run_id, scope)
    if want_funded:
        _resolve_funded_side(inputs, output_root, funded_run_id, scope)

    return inputs


def _resolve_concentration(inputs: GovernedInputs, output_root: Optional[str],
                           funded_run_id: Optional[str], scope: Any) -> None:
    """Resolve the concentration snapshot, recording every way it can be absent.

    ``_safe`` only records an unavailable capability when the call RAISES. A
    governed service that returns ``{"available": False, "reason": ...}`` — no
    approved configuration, no funded run, a tape without the tested fields — is
    a controlled outcome, not an exception, so it left ``unavailable`` empty and
    the Risk Review read the absence as a clear position. Both shapes are
    recorded here, with the service's own reason where it gave one.
    """
    if not output_root:
        inputs.unavailable[CAP_CONCENTRATION] = (
            "no governed funded output root is configured for this deployment, "
            "so no concentration test was evaluated")
        return

    snapshot = _safe(inputs, CAP_CONCENTRATION, lambda: _concentration(
        output_root, inputs.portfolio_id, funded_run_id, scope))
    inputs.concentration = snapshot

    if snapshot is None:
        # _safe has already recorded the capability; nothing to add.
        return
    if not snapshot.get("available"):
        inputs.unavailable[CAP_CONCENTRATION] = (
            snapshot.get("reason")
            or "no concentration test could be evaluated for this update")
        return
    # A snapshot that resolved is evidence the domain WAS evaluated; clear any
    # earlier record so a retry does not leave a stale unavailable entry.
    inputs.unavailable.pop(CAP_CONCENTRATION, None)
    _resolve_contributors(inputs, output_root, funded_run_id, scope)


def _resolve_pipeline_side(inputs: GovernedInputs, pipeline_root: Optional[str],
                           output_root: Optional[str],
                           funded_run_id: Optional[str], scope: Any) -> None:
    from mi_agent_api import datasets as datasets_mod
    from mi_agent_api import evolution as evolution_mod
    from mi_agent_api import insight_engine as engine
    from mi_agent_api import insight_metrics as metrics

    if not pipeline_root:
        reason = ("no governed weekly pipeline root is configured for this "
                  "deployment")
        inputs.unavailable[CAP_WEEKLY_BRIEF] = reason
        inputs.unavailable[CAP_PIPELINE_EVOLUTION] = reason
        inputs.unavailable[CAP_FUNNEL] = reason
        return

    # Passed through every resolver below purely so they SHARE the frames the
    # dashboard already prepared rather than preparing every weekly extract
    # again under a different cache key.
    history = _safe(inputs, "pipeline history",
                    lambda: datasets_mod._pipeline_history(inputs.portfolio_id))

    # Resolved once for the whole update by ``_resolve_concentration``, before
    # either side ran. Read here rather than recomputed: the brief and the Risk
    # Review must be looking at one evaluation of one funded position.
    concentration = inputs.concentration

    # ``lag_weeks`` is passed exactly as the dashboard passes it. Omitting it
    # would silently compute the conversion rate UNLAGGED — a different, larger
    # number than the workspace and the Weekly Brief publish for the same week,
    # which is precisely the two-channels-disagree failure this design exists
    # to prevent.
    funnel = _safe(inputs, CAP_FUNNEL, lambda: evolution_mod.pipeline_funnel_evolution(
        pipeline_root, inputs.portfolio_id, None,
        lag_weeks=datasets_mod._kfi_lag_weeks_from_model(history),
        historical_model=history))
    inputs.funnel = funnel

    evo = _safe(inputs, CAP_PIPELINE_EVOLUTION,
                lambda: evolution_mod.pipeline_evolution(
                    pipeline_root, inputs.portfolio_id,
                    historical_model=history))
    inputs.pipeline_evolution = evo

    # THE governed insight set. Built by the same engine, from the same inputs,
    # that serves the Weekly Portfolio Brief — so a card and the brief cannot
    # report a different version of the same week.
    brief = _safe(inputs, CAP_WEEKLY_BRIEF, lambda: engine.build(
        pipeline_root, inputs.portfolio_id, tenant_id=inputs.tenant_id,
        historical_model=history, scope=inputs.portfolio_context,
        concentration_snapshot=concentration, funnel=funnel))
    inputs.brief = brief

    if brief:
        if brief.get("status") == "unavailable":
            inputs.unavailable[CAP_WEEKLY_BRIEF] = (
                brief.get("reason") or "the weekly brief was unavailable")
        elif brief.get("status") == "partial":
            # A partial brief is usable, but the reader must be told which
            # sections are missing before any clear-state claim is made.
            for omitted in brief.get("omitted") or []:
                if omitted.get("category") == "error":
                    inputs.unavailable[str(omitted.get("insight_type"))] = (
                        omitted.get("reason") or "this check could not be produced")
        if brief.get("run_id"):
            inputs.run_ids.append(str(brief["run_id"]))
        inputs.source_dates.update(brief.get("source_dates") or {})
        inputs.methodology_versions["insight_contract"] = brief.get("brief_version")
        inputs.methodology_versions["insight_metrics"] = getattr(
            metrics, "METHODOLOGY_VERSION", None)


def _concentration(output_root: Optional[str], client_id: str,
                   funded_run_id: Optional[str], scope: Any) -> Dict[str, Any]:
    from mi_agent_api import concentration_tests_api as conc_mod
    return conc_mod.compute_concentration_tests(
        output_root, client_id, funded_run_id, scope=scope)


def _resolve_contributors(inputs: GovernedInputs, output_root: Optional[str],
                          funded_run_id: Optional[str], scope: Any) -> None:
    """Contributor attribution for the tests a Risk Review might actually cite.

    Resolved only for tests already at or beyond their warning state, and only
    for the few a card could carry. Attributing every test would run the
    expected-state evaluator across the whole library to produce evidence for
    findings that will never be shown.
    """
    from .config import load as load_config

    snapshot = inputs.concentration or {}
    if not snapshot.get("available"):
        return
    limit = max(1, int(load_config().get("maximum_risk_items") or 3))
    candidates = [
        t for t in (snapshot.get("tests") or [])
        if t.get("expectedBreach") or t.get("fullPipelineBreach")
        or str(t.get("status")) in ("breach", "warning")
        or (t.get("expected") or {}).get("status") == "warning"
    ][:limit]
    if not candidates:
        return

    from mi_agent_api import concentration_tests_api as conc_mod
    for test in candidates:
        test_id = test.get("testId")
        if not test_id:
            continue
        result = _safe(inputs, CAP_CONTRIBUTORS, lambda tid=test_id:
                       conc_mod.compute_pipeline_drivers(
                           output_root, inputs.portfolio_id, funded_run_id, tid,
                           scope=scope))
        contributors = (result or {}).get("contributors")
        if contributors and contributors.get("available"):
            inputs.contributors[str(test_id)] = contributors
            # A capability that succeeded for at least one test is available.
            inputs.unavailable.pop(CAP_CONTRIBUTORS, None)


def _resolve_funded_side(inputs: GovernedInputs, output_root: Optional[str],
                         funded_run_id: Optional[str], scope: Any) -> None:
    from mi_agent_api import movement_summary as movement_mod

    if not output_root:
        inputs.unavailable[CAP_FUNDED_MOVEMENT] = (
            "no governed funded output root is configured for this deployment")
        inputs.unavailable[CAP_MONTHLY_BRIEF] = (
            "no governed funded output root is configured for this deployment")
        return

    movement = _safe(inputs, CAP_FUNDED_MOVEMENT,
                     lambda: movement_mod.period_movement(
                         output_root, inputs.portfolio_id,
                         to_run_id=funded_run_id))
    inputs.funded_movement = movement
    if movement and not movement.get("available"):
        inputs.unavailable[CAP_FUNDED_MOVEMENT] = (
            movement.get("reason") or "funded period movement was unavailable")
    if movement and movement.get("available"):
        inputs.source_dates["funded_as_of"] = movement.get("currentReportingDate")
        inputs.source_dates["funded_comparison"] = movement.get("priorReportingDate")

    # THE governed monthly insight set — the funded analogue of the weekly
    # brief. Resolved with the concentration snapshot already in hand, so the
    # risk-limit transitions the month reports are read from the SAME evaluation
    # the Risk Review is graded against.
    _resolve_monthly_brief(inputs, output_root, funded_run_id, scope)


def _resolve_monthly_brief(inputs: GovernedInputs, output_root: Optional[str],
                           funded_run_id: Optional[str], scope: Any) -> None:
    """The Monthly Funded Brief, resolved once for this approved update."""
    from mi_agent_api import insight_engine as engine

    brief = _safe(inputs, CAP_MONTHLY_BRIEF, lambda: engine.build_funded(
        output_root, inputs.portfolio_id, tenant_id=inputs.tenant_id,
        to_run_id=funded_run_id, scope=inputs.portfolio_context,
        concentration_snapshot=inputs.concentration))
    inputs.funded_brief = brief
    if not brief:
        return

    if brief.get("status") == "unavailable":
        inputs.unavailable[CAP_MONTHLY_BRIEF] = (
            brief.get("reason") or "the monthly funded brief was unavailable")
    elif brief.get("status") == "partial":
        # Same discipline as the weekly side: a partial brief is usable, but the
        # reader is told which sections are missing before any clear-state claim.
        for omitted in brief.get("omitted") or []:
            if omitted.get("category") == "error":
                inputs.unavailable[str(omitted.get("insight_type"))] = (
                    omitted.get("reason") or "this check could not be produced")
    inputs.source_dates.update(
        {k: v for k, v in (brief.get("source_dates") or {}).items() if v})
    inputs.methodology_versions["funded_insight_contract"] = \
        brief.get("brief_version")


#: How a required domain is shown to have been evaluated. One entry per member
#: of ``REQUIRED_RISK_CAPABILITIES``: the evidence, not the attempt.
_RISK_EVIDENCE = {
    CAP_CONCENTRATION: lambda i: bool((i.concentration or {}).get("available")),
}


def unevaluated_risk_domains(inputs: GovernedInputs) -> Dict[str, str]:
    """Required risk domains with no positive evidence that they ran.

    Deliberately asks for EVIDENCE rather than trusting the absence of an error.
    A domain that was skipped, that returned a controlled unavailable state, or
    that a future caller forgets to resolve all look identical here — which is
    the point. The alternative, inferring "it ran and found nothing" from an
    empty findings list, is exactly how a check that never executed came to be
    reported as a clear position.
    """
    out: Dict[str, str] = {}
    for capability in REQUIRED_RISK_CAPABILITIES:
        evidence = _RISK_EVIDENCE.get(capability)
        if evidence is not None and evidence(inputs):
            continue
        out[capability] = inputs.unavailable.get(capability) or (
            f"{capability} was not evaluated for this update")
    return out


def unavailable_summary(inputs: GovernedInputs) -> List[str]:
    """Plain-language names of the checks that did not run.

    Includes any required risk domain that cannot show it was evaluated, so the
    list a clear-state message is gated on can never be shorter than the set of
    domains actually checked. Sorted, so the same set of failures always reads
    the same way; listed on the card so a clear-state message is never mistaken
    for a complete one.
    """
    return sorted(set(inputs.unavailable) | set(unevaluated_risk_domains(inputs)))
