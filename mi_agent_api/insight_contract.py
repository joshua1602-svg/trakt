#!/usr/bin/env python3
"""Phase 3A — the governed insight object and the Weekly Portfolio Brief.

ONE versioned contract, designed so the same object can later be rendered by
React, pushed to Teams, quoted by Copilot or lifted into commentary WITHOUT any
of them recalculating the metric. That is the point: an insight carries its own
numbers, its own dates, its own methodology and its own provenance, so a second
channel can never produce a different version of the same fact.

Three properties make that safe:

* **Deterministic identity.** The id is a hash of what the insight IS — tenant,
  portfolio, scope, type, the two dates, and the discriminator that separates
  two insights of the same type (a test id, a band label). The same week
  regenerated gives the same id, so a downstream channel can de-duplicate and
  an audit trail can be followed. It is NOT a hash of the values, so a
  corrected upload updates an insight rather than creating a second one.

* **No loan-level data.** Contributors are dimension aggregates. Nothing here
  carries a case id, and a test asserts it.

* **Severity and priority are separate.** Severity is what the reader is told;
  priority is how the brief is ordered. Keeping them apart means the ordering
  can be tuned without changing what an insight claims about itself.

``notification_eligible`` is set but not acted on — Teams is not built in this
phase. It exists so the eligibility rule lives with the insight rather than
being re-derived by whichever channel eventually sends it.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

#: Bumped when the SHAPE of an insight changes. Individual metric methodologies
#: carry their own versions inside ``methodology``.
CONTRACT_VERSION = "1"

# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
PIPELINE_MOVEMENT = "PIPELINE_MOVEMENT"
TICKET_SIZE = "TICKET_SIZE"
WEIGHTED_LTV = "WEIGHTED_LTV"
TICKET_MIX_SHIFT = "TICKET_MIX_SHIFT"
LTV_MIX_SHIFT = "LTV_MIX_SHIFT"
COMPLETIONS_MOVEMENT = "COMPLETIONS_MOVEMENT"
CONVERSION_CONTEXT = "CONVERSION_CONTEXT"
CONCENTRATION_PROXIMITY = "CONCENTRATION_PROXIMITY"
DATA_QUALITY = "DATA_QUALITY"

INSIGHT_TYPES = (
    PIPELINE_MOVEMENT, TICKET_SIZE, WEIGHTED_LTV, TICKET_MIX_SHIFT,
    LTV_MIX_SHIFT, COMPLETIONS_MOVEMENT, CONVERSION_CONTEXT,
    CONCENTRATION_PROXIMITY, DATA_QUALITY,
)

# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #
SEVERITY_INFO = "info"
SEVERITY_ATTENTION = "attention"
SEVERITY_CONCERN = "concern"

#: Ordering weight, high to low. Used by the selector, never by a generator.
SEVERITY_RANK = {SEVERITY_CONCERN: 3, SEVERITY_ATTENTION: 2, SEVERITY_INFO: 1}

#: Base priority per type, before severity. Concentration and data quality sit
#: above everything because one is a contractual exposure and the other governs
#: whether the rest of the brief can be believed.
TYPE_PRIORITY = {
    CONCENTRATION_PROXIMITY: 100,
    DATA_QUALITY: 90,
    PIPELINE_MOVEMENT: 70,
    COMPLETIONS_MOVEMENT: 65,
    CONVERSION_CONTEXT: 60,
    TICKET_SIZE: 40,
    WEIGHTED_LTV: 38,
    TICKET_MIX_SHIFT: 30,
    LTV_MIX_SHIFT: 28,
}


@dataclass
class Insight:
    """One governed observation about one week, for one portfolio scope."""

    insight_type: str
    tenant_id: str
    portfolio_id: str
    portfolio_context: str
    as_of_date: Optional[str]
    comparison_date: Optional[str]
    headline: str
    summary: str
    severity: str = SEVERITY_INFO
    #: Separates two insights of the same type in the same week (a test id, a
    #: band label). Part of the identity hash.
    discriminator: str = ""
    run_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    contributors: Dict[str, Any] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)
    methodology: Dict[str, Any] = field(default_factory=dict)
    source_dates: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    deep_link: Optional[str] = None
    #: Ordering weight. Set by the selector, not by the generator.
    priority: int = 0
    version: str = CONTRACT_VERSION

    @property
    def insight_id(self) -> str:
        """Stable across regenerations of the same week; distinct across
        tenants, portfolios, scopes, types, dates and discriminators.

        Deliberately excludes the VALUES: a corrected upload for the same week
        must update this insight, not sit alongside a stale twin.
        """
        parts = [self.version, self.insight_type, self.tenant_id or "-",
                 self.portfolio_id or "-", self.portfolio_context or "total",
                 self.as_of_date or "-", self.comparison_date or "-",
                 self.discriminator or "-"]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]

    @property
    def notification_eligible(self) -> bool:
        """Whether a future channel SHOULD push this unprompted.

        Set here so the rule lives with the insight rather than being
        re-derived by whichever channel eventually sends it. Nothing acts on it
        in this phase.
        """
        return self.severity in (SEVERITY_ATTENTION, SEVERITY_CONCERN)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("discriminator", None)
        d["insight_id"] = self.insight_id
        d["notification_eligible"] = self.notification_eligible
        return d


@dataclass
class Omission:
    """Why an insight the brief would normally carry is absent.

    Every generator that declines returns one of these. A brief that silently
    dropped a section would be indistinguishable from a book with nothing to
    report, which is the failure mode this phase must not have.
    """

    insight_type: str
    reason: str
    #: "immaterial" (below threshold — the system working) vs "unavailable"
    #: (data or methodology missing) vs "error" (generation failed).
    category: str = "immaterial"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


OMITTED_IMMATERIAL = "immaterial"
OMITTED_UNAVAILABLE = "unavailable"
OMITTED_ERROR = "error"


def build_brief(insights: List[Insight], omissions: List[Omission], *,
                tenant_id: str, portfolio_id: str, portfolio_context: str,
                as_of_date: Optional[str], comparison_date: Optional[str],
                run_id: Optional[str] = None,
                source_dates: Optional[Dict[str, Any]] = None,
                config_source: Optional[str] = None,
                status: str = "success",
                reason: Optional[str] = None) -> Dict[str, Any]:
    """The Weekly Portfolio Brief envelope.

    ``status`` distinguishes a complete brief from one where a generator failed
    (``partial``) or where the governed source itself could not be resolved
    (``unavailable``). A partial brief still returns everything that DID work —
    one failed insight type must not cost the reader the other seven.
    """
    return {
        "brief_version": CONTRACT_VERSION,
        "status": status,
        "reason": reason,
        "tenant_id": tenant_id,
        "portfolio_id": portfolio_id,
        "portfolio_context": portfolio_context,
        "run_id": run_id,
        "as_of_date": as_of_date,
        "comparison_date": comparison_date,
        "generated_from": "governed weekly pipeline extracts and existing MI outputs",
        "config_source": config_source,
        "insight_count": len(insights),
        "insights": [i.to_dict() for i in insights],
        "omitted": [o.to_dict() for o in omissions],
        "source_dates": source_dates or {},
    }
