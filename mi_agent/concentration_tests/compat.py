"""mi_agent.concentration_tests.compat — controlled migration of legacy limits.

Two legacy stores of client thresholds exist:

* ``config/client/risk_limits_config.py`` — the Streamlit-era hard-coded
  Python dictionaries (still consumed by ``analytics/risk_monitor.py``);
* the Schedule 8 extracted contract
  (``config/clients/<client>/risk_limits_extracted.yaml`` /
  ``output/risk/risk_limits_config.yaml``) the current ``/mi/risk-limits``
  route reads.

Neither is an approved configuration. This module converts either shape into
:class:`TestProposal` records so they enter the SAME review → approval →
activation lifecycle as client-supplied tests. Nothing here activates
anything: migration produces proposals with provenance naming the legacy
source, and a human approves them (or doesn't). Hard-coded client thresholds
never move into shared source; they move into versioned, tenant-scoped,
operator-approved configuration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .library import ConcentrationLibrary
from .models import (
    MATCH_AMBIGUOUS,
    MATCH_MATCHED,
    MATCH_UNSUPPORTED,
    PROPOSAL_PENDING_APPROVAL,
    PROPOSAL_PENDING_CONFIRMATION,
    PROPOSAL_UNSUPPORTED,
    ConcernCode,
    SourceEvidence,
    TestProposal,
)

#: Legacy Streamlit-era region wrappers → readable region labels.
_LEGACY_REGION_LABELS = {
    "uki": "London", "ukj": "South East", "ukc": "North East",
    "ukd": "North West", "uke": "Yorkshire and The Humber",
    "ukf": "East Midlands", "ukg": "West Midlands", "ukh": "East Anglia",
    "ukk": "South West", "ukl": "Wales", "ukm": "Scotland",
    "ukn": "Northern Ireland",
}


def _proposal(*, client_id: str, source: str, name: str, metric_id: str,
              outcome: str, operator: str, threshold: Optional[float],
              unit: str, parameters: Dict[str, Any],
              concerns: Optional[List[str]] = None,
              source_text: str = "") -> TestProposal:
    concerns = concerns or []
    if outcome == MATCH_UNSUPPORTED:
        status = PROPOSAL_UNSUPPORTED
    elif concerns or outcome == MATCH_AMBIGUOUS:
        status = PROPOSAL_PENDING_CONFIRMATION
    else:
        status = PROPOSAL_PENDING_APPROVAL
    return TestProposal(
        client_id=client_id,
        evidence=SourceEvidence(source_reference=source,
                                source_text=source_text or name),
        extracted_test_name=name,
        proposed_metric_id=metric_id,
        match_outcome=outcome,
        extracted_operator=operator,
        extracted_threshold=threshold,
        extracted_unit=unit,
        parameters=parameters,
        extraction_confidence=1.0,  # the legacy config is explicit, not parsed
        mapping_confidence=0.9 if outcome == MATCH_MATCHED else 0.5,
        concern_codes=list(concerns),
        concern_messages=[c.replace("_", " ") for c in concerns],
        status=status,
    )


def proposals_from_legacy_limits(limits: Dict[str, Dict[str, Any]],
                                 client_id: str,
                                 *, source: str = "config/client/risk_limits_config.py"
                                 ) -> List[TestProposal]:
    """Convert the legacy hard-coded limit dictionaries into proposals."""
    out: List[TestProposal] = []
    for limit_id, spec in (limits or {}).items():
        threshold = spec.get("limit_value")
        operator = spec.get("direction", "max")
        description = str(spec.get("description") or limit_id)
        lowered = limit_id.lower()
        if lowered.startswith("max_region_"):
            token = lowered.replace("max_region_", "").replace("_pct", "")
            if token.endswith("_combined") or "_" in token:
                codes = [t for t in token.split("_") if t in _LEGACY_REGION_LABELS]
            else:
                codes = [token] if token in _LEGACY_REGION_LABELS else []
            regions = [_LEGACY_REGION_LABELS[c] for c in codes]
            if regions:
                out.append(_proposal(
                    client_id=client_id, source=source, name=description,
                    metric_id="geo_region_share", outcome=MATCH_MATCHED,
                    operator=operator, threshold=threshold, unit="percent",
                    parameters={"regions": regions}))
                continue
        mapped = {
            "max_low_value_property_pct": (
                "property_value_below_share",
                {"amount": 150000.0, "value_basis": "original",
                 "comparison": "below"}),
            "max_high_value_property_pct": (
                "property_value_above_share",
                {"amount": 1000000.0, "value_basis": "original",
                 "comparison": "above"}),
            "max_single_borrower_balance_pct": ("borrower_largest_share", {}),
            "max_loans_per_borrower": ("borrower_max_loans", {}),
            "max_balance_to_multi_loan_borrowers": (
                "borrower_multi_loan_share", {"min_loans": 3}),
            "max_age_over_85_pct": (
                "borrower_age_share",
                {"age": 85.0, "comparison": "above", "age_basis": "youngest"}),
            "max_variable_rate_pct": (
                "rate_type_share", {"values": ["variable", "floating"]}),
        }.get(lowered)
        if mapped:
            metric_id, params = mapped
            unit = "count" if metric_id == "borrower_max_loans" else "percent"
            out.append(_proposal(
                client_id=client_id, source=source, name=description,
                metric_id=metric_id, outcome=MATCH_MATCHED, operator=operator,
                threshold=threshold, unit=unit, parameters=params))
        else:
            out.append(_proposal(
                client_id=client_id, source=source, name=description,
                metric_id="", outcome=MATCH_UNSUPPORTED, operator=operator,
                threshold=threshold, unit=str(spec.get("unit") or ""),
                parameters={}, concerns=[ConcernCode.UNSUPPORTED_METRIC]))
    return out


def proposals_from_extracted_config(config: Dict[str, Any], client_id: str
                                    ) -> List[TestProposal]:
    """Convert the Schedule 8 extracted contract into proposals.

    Every proposal keeps the extractor's source snippet as evidence, so the
    operator reviews the same wording the extractor read.
    """
    source = str(config.get("source_file") or config.get("source_document")
                 or "risk_limits_extracted.yaml")
    out: List[TestProposal] = []
    for lim in config.get("limits") or []:
        category = str(lim.get("category") or "")
        threshold = lim.get("limit_value", lim.get("threshold"))
        operator = str(lim.get("direction") or lim.get("operator") or "max")
        snippet = str(lim.get("source_snippet") or lim.get("source_text") or "")
        name = str(lim.get("limit_id") or category)
        needs_review = bool(lim.get("needs_review"))
        region = lim.get("region")
        lowered = snippet.lower()

        metric_id = ""
        params: Dict[str, Any] = {}
        outcome = MATCH_UNSUPPORTED
        concerns: List[str] = []
        if category == "geographic_concentration":
            if region and "single region" not in str(region).lower():
                metric_id, outcome = "geo_region_share", MATCH_MATCHED
                params = {"regions": [str(region)]}
            else:
                metric_id, outcome = "geo_largest_region_share", MATCH_MATCHED
        elif category == "broker_concentration":
            if "top" in lowered:
                import re
                m = re.search(r"top\s+(\d+)", lowered)
                # Top-N brokers is a grouped top-N — representable only as the
                # largest-group primitive when N == 1; otherwise unsupported
                # until a grouped top-N metric ships.
                if m and int(m.group(1)) > 1:
                    metric_id, outcome = "", MATCH_UNSUPPORTED
                    concerns.append(ConcernCode.UNSUPPORTED_METRIC)
                else:
                    metric_id, outcome = "composition_largest_group_share", MATCH_MATCHED
                    params = {"dimension": "broker"}
            else:
                metric_id, outcome = "composition_largest_group_share", MATCH_MATCHED
                params = {"dimension": "broker"}
        elif category == "large_loan_concentration":
            from .matching import _parse_money
            amount = _parse_money(snippet)
            if amount:
                metric_id, outcome = "balance_above_share", MATCH_MATCHED
                params = {"amount": amount, "balance_basis": "current"}
            else:
                metric_id, outcome = "top_n_loans_share", MATCH_MATCHED
                params = {"n": 1}
        elif category == "ltv_limit":
            if "weighted average" in lowered:
                metric_id, outcome = "ltv_weighted_average", MATCH_MATCHED
                params = {"ltv_basis": "current"}
            else:
                metric_id, outcome = "ltv_maximum", MATCH_MATCHED
                params = {"ltv_basis": "current"}
        elif category == "interest_rate_limit":
            metric_id, outcome = "rate_type_share", MATCH_MATCHED
            params = {"values": ["variable", "floating"]}
        elif category == "borrower_concentration":
            if str(lim.get("metric")) == "count" or "more than" in lowered:
                metric_id, outcome = "borrower_max_loans", MATCH_MATCHED
            else:
                metric_id, outcome = "borrower_largest_share", MATCH_MATCHED
        elif category == "joint_borrower_limit":
            metric_id, outcome = "borrower_joint_share", MATCH_AMBIGUOUS
            concerns.append(ConcernCode.JOINT_DEFINITION_UNCERTAIN)
        elif category == "age_limit":
            metric_id, outcome = "borrower_age_share", MATCH_MATCHED
            params = {"age": 85.0, "comparison": "above",
                      "age_basis": "youngest"}
        else:
            concerns.append(ConcernCode.UNSUPPORTED_METRIC)

        if threshold is None and outcome != MATCH_UNSUPPORTED:
            concerns.append(ConcernCode.THRESHOLD_UNIT_UNCERTAIN)
            outcome = MATCH_AMBIGUOUS
        if needs_review and outcome == MATCH_MATCHED:
            outcome = MATCH_AMBIGUOUS

        prop = _proposal(
            client_id=client_id, source=source, name=name,
            metric_id=metric_id, outcome=outcome, operator=operator,
            threshold=threshold,
            unit=str(lim.get("unit") or "percent"),
            parameters=params, concerns=concerns, source_text=snippet)
        out.append(prop)
    return out


def migrate_legacy_defaults(client_id: str, store, lib: ConcentrationLibrary,
                            *, limits: Optional[Dict[str, Dict[str, Any]]] = None,
                            extracted_config: Optional[Dict[str, Any]] = None
                            ) -> List[TestProposal]:
    """Persist legacy limits as proposals awaiting approval. Idempotent by
    evidence + metric: an already-proposed legacy limit is not re-proposed."""
    existing = {(p.evidence.source_reference, p.proposed_metric_id,
                 p.extracted_test_name)
                for p in store.list_proposals(client_id)}
    proposals: List[TestProposal] = []
    if limits:
        proposals.extend(proposals_from_legacy_limits(limits, client_id))
    if extracted_config:
        proposals.extend(proposals_from_extracted_config(extracted_config,
                                                         client_id))
    saved: List[TestProposal] = []
    for p in proposals:
        key = (p.evidence.source_reference, p.proposed_metric_id,
               p.extracted_test_name)
        if key in existing:
            continue
        if p.proposed_metric_id:
            problems = lib.validate_parameters(p.proposed_metric_id, p.parameters)
            required_missing = [x for x in problems if "is missing" in x]
            if required_missing and p.status == PROPOSAL_PENDING_APPROVAL:
                p.status = PROPOSAL_PENDING_CONFIRMATION
                p.concern_codes.append(ConcernCode.AGGREGATION_BASIS_UNDEFINED)
                p.concern_messages.extend(required_missing)
        saved.append(store.save_proposal(client_id, p))
    return saved
