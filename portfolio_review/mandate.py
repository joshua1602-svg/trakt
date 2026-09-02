"""The Portfolio Review Agent's mandate: what it is for, and what it may touch.

WHY THIS FILE EXISTS
--------------------
The first real-model red-team put the agent on real canonical and it did two
things it was told not to. One was arithmetic (``numeric_gate`` answers that).
The other was scope: asked what changed in the portfolio this month, it
investigated ESMA Annex 2 field coverage, reported ``RREC1`` as a blocker, and
announced that the book breached "Example Warehouse Facility Criteria
(SYNTHETIC)" — a rulebook whose own payload says it is *not a real facility
agreement and not approved by anyone*.

None of that was a hallucination. Every one of those findings was true, governed
and correctly measured. They belong to a **different agent**. A portfolio
manager asking "what changed this month" is not asking whether a securitisation
could be issued, and an answer that drifts there is wrong even when every
sentence in it is right.

The prompt already said so. It did not hold. So the boundary is moved out of the
prompt and into the tool surface: the readiness tools are **not registered** for
this agent, and a call naming one is refused by
:class:`~portfolio_review.session.MIScopedSession` before it reaches execution.
A model cannot investigate what it cannot call.

WHAT IS AND IS NOT DECIDED HERE
-------------------------------
This file decides the **mandate** — the objective, the domains, and the tool
allow-list. It decides nothing about method: which tool to call first, in what
order, or how many. That remains the model's, and is the whole reason this is an
agent rather than a report.

THE CLASSIFICATION IS TOTAL, AND THAT IS THE POINT
--------------------------------------------------
``ALLOWED`` and ``EXCLUDED`` must together account for **every** tool in the
registry. :func:`audit_registry` checks it and
``test_portfolio_review_mandate`` fails the build otherwise. So a new tool
registered tomorrow does not quietly become available to this agent because
nobody thought about it — the suite breaks until somebody classifies it. An
allow-list that silently grows is not an allow-list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Tuple

# --------------------------------------------------------------------------- #
# The mandate
# --------------------------------------------------------------------------- #
#: The agent's production objective. One question, narrow on purpose.
MANDATE = (
    "Review the current accepted portfolio reporting period against the "
    "immediately prior accepted period. Identify the most material operating "
    "changes in pipeline and/or funded assets, explain those changes using only "
    "governed MI evidence, investigate their effect on approved client "
    "risk/concentration limits where relevant, distinguish acquisition-driven "
    "from underlying portfolio movement, and return a concise ranked set of "
    "management findings. Do not perform securitisation, regulatory, Annex "
    "2/12, rating-agency, warehouse-readiness or transaction-readiness analysis."
)

#: What the agent is. Stated positively because "not a readiness agent" does not
#: tell a reader what it *is*.
ROLE = "management information / portfolio monitoring"

#: What the agent is NOT. Each of these is a real agent's job or nobody's.
NOT_ROLES: Tuple[str, ...] = (
    "securitisation readiness",
    "regulatory reporting",
    "covenant underwriting",
    "transaction readiness",
)

#: The MI domains in scope, as a reader would name them.
IN_SCOPE: Dict[str, Tuple[str, ...]] = {
    "pipeline": (
        "pipeline balance", "case count", "period movement", "product mix",
        "geography", "LTV", "borrower characteristics where supported",
        "stages", "conversion", "fallout",
        "movement into funded where governed linkage exists",
        "expected funding only where a governed forecast already exists",
    ),
    "funded": (
        "funded balance", "loan count", "period movement",
        "organic new lending", "acquired portfolio additions",
        "repayments, redemptions and exits", "existing-book balance movement",
        "product mix", "geography", "LTV", "borrower age",
        "joint or single borrower status",
        "interest-rate characteristics where supported", "vintage and cohort",
        "source portfolio", "combined vs underlying portfolio movement",
    ),
    "risk": (
        "existing governed portfolio concentrations",
        "existing APPROVED client risk limits", "limit utilisation",
        "green / amber / red transitions", "new breach", "resolved breach",
        "movement toward or away from an actual configured limit",
    ),
}

#: Domains the agent must not enter. Not advice — every one of these is enforced
#: by the absence of a tool, and the words are here so a reader can check that
#: the two lists agree.
PROHIBITED: Tuple[str, ...] = (
    "ESMA Annex 2", "ESMA Annex 12", "regulatory field coverage",
    "regulatory schema completeness", "regulatory blockers",
    "securitisation readiness", "rating-agency readiness",
    "warehouse diligence readiness", "transaction eligibility",
    "transaction perimeter", "proposed securitisation criteria",
    "synthetic warehouse criteria", "example or illustrative rule packs",
    "investor-reporting readiness",
    "covenant evaluation other than an actual approved client operating limit",
)


# --------------------------------------------------------------------------- #
# The tool allow-list
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Exclusion:
    """One tool this agent may not call, and why not."""

    tool: str
    reason: str
    belongs_to: str = "Securitisation Readiness Agent"


#: Every tool the Portfolio Review Agent may call.
#:
#: ``evaluate_covenants`` is on this list deliberately and is the one entry that
#: needs justifying. It resolves through ``concentration_tests_api`` — the same
#: evaluator behind the Risk Limits workspace — so what it returns is the
#: client's own APPROVED operating limits, which §2 puts squarely in scope. When
#: no configuration has been approved it says so ("This is an absence of
#: evidence, not a clean result") rather than substituting a rulebook, which is
#: exactly the behaviour that makes it safe to expose and ``evaluate_rule_packs``
#: unsafe.
ALLOWED: FrozenSet[str] = frozenset({
    # discovery
    "portfolio_capabilities",
    # funded position and movement
    "portfolio_summary", "period_change", "funded_composition",
    "portfolio_history", "stratify", "cohort_comparison",
    # pipeline
    "pipeline_position", "pipeline_movement", "pipeline_conversion",
    # risk against APPROVED client limits
    "concentration", "forward_concentration", "evaluate_covenants",
    "covenant_drillthrough",
    # portfolio performance — what the book DID this period
    "transition_analysis", "default_analysis", "cure_analysis",
    "loss_analysis", "prepayment_analysis",
    # drill-down onto the cases behind an aggregate
    "rank_loans", "get_loan", "get_loans",
})

#: Every tool it may not, with the reason. The reason is not decoration: it is
#: what a reviewer reads when deciding whether the line was drawn correctly, and
#: what the report in ``docs/`` quotes.
EXCLUDED: Tuple[Exclusion, ...] = (
    Exclusion("evaluate_rule_packs",
              "Applies example/synthetic warehouse and proposed securitisation "
              "rulebooks. Its own payload labels them 'not a real facility "
              "agreement and not approved by anyone'; the agent nevertheless "
              "reported a warehouse breach in its headline. Scope exclusion is "
              "the fix, not better wording."),
    Exclusion("readiness_framework",
              "Enumerates what a securitisation readiness review should assess. "
              "A different question from what changed this period."),
    Exclusion("readiness_metrics",
              "The readiness metric library. LTV and balance distributions "
              "remain reachable through `stratify` without the readiness "
              "framing that comes with them here."),
    Exclusion("regulatory_readiness",
              "ESMA Annex 2/12 submission feasibility and field coverage. "
              "Produced the 'RREC1 is a blocker' finding verbatim."),
    Exclusion("valuation_age_profile",
              "Collateral evidence diligence — how old and how good the "
              "valuations are. A diligence question, and the source of the "
              "'every LTV in the book is unverified' framing."),
    Exclusion("contractual_analytics",
              "Contractual WAL and YTM: transaction analytics used to size and "
              "price a deal, not to monitor a period."),
    Exclusion("data_completeness",
              "Field population against a regulatory field universe. Produced "
              "'borrower_identifier is 0% populated' as an ESMA finding."),
    Exclusion("list_validation_exceptions",
              "Canonical validation and lineage outcomes. A data-assurance "
              "question owned by Operations Control, not period MI.",
              belongs_to="Operations Control Centre"),
    Exclusion("explain_value",
              "Single-value provenance drill-through. Lineage evidence rather "
              "than a portfolio measure.",
              belongs_to="Operations Control Centre"),
    Exclusion("explain_values",
              "As `explain_value`, in bulk.",
              belongs_to="Operations Control Centre"),
)

EXCLUDED_NAMES: FrozenSet[str] = frozenset(e.tool for e in EXCLUDED)


# --------------------------------------------------------------------------- #
# Enforcement
# --------------------------------------------------------------------------- #
def audit_registry() -> Dict[str, List[str]]:
    """Reconcile the mandate against the live registry.

    Returns the three ways the two can disagree. All three must be empty:

    ``unclassified``
        registered, and on neither list — the case that matters. A tool added
        to Trakt is not available to this agent until somebody decides it
        should be, and this is what makes that true rather than aspirational.
    ``missing``
        allow-listed but not registered — a typo, or a tool that was removed.
    ``excluded_but_absent``
        excluded but not registered — a stale exclusion worth deleting.
    """
    from trakt_tools.registry import tool_names

    registered = set(tool_names())
    return {
        "unclassified": sorted(registered - ALLOWED - EXCLUDED_NAMES),
        "missing": sorted(ALLOWED - registered),
        "excluded_but_absent": sorted(EXCLUDED_NAMES - registered),
    }


def tool_schemas() -> List[Dict[str, Any]]:
    """The agent's tool surface: allow-listed tools only, registry schemas.

    Generated from ``trakt_tools.registry`` for the same reason
    ``readiness_agent.agent.governed_tool_schemas`` is — a hand-maintained copy
    drifts, and the first symptom is an agent calling something that no longer
    exists. The only difference is the filter.
    """
    from trakt_tools.registry import all_tools

    out: List[Dict[str, Any]] = []
    for spec in sorted(all_tools(), key=lambda s: s.name):
        if spec.name not in ALLOWED:
            continue
        description = spec.description
        if spec.agent_guidance:
            description = f"{description}\n\nGuidance: {spec.agent_guidance}"
        out.append({"name": spec.name, "description": description[:1800],
                    "input_schema": spec.input_schema})
    return out


def is_allowed(tool: str) -> bool:
    return str(tool) in ALLOWED


def exclusion_for(tool: str) -> Exclusion | None:
    return next((e for e in EXCLUDED if e.tool == tool), None)
