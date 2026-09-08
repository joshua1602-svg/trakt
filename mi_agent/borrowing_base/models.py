"""mi_agent.borrowing_base.models — typed contracts, no pandas, no I/O.

The vocabularies here are closed on purpose, for the same reason the
concentration-test vocabularies are: an unknown eligibility status is a bug in
the caller, not a fourth state the platform silently learns.

Two conventions matter downstream and are stated once, here:

* **NOT_CALCULABLE is a value, not a zero.** Every measure the facility cannot
  produce — headroom with no drawn balance, Net WAC with no Series Fixed Rate —
  comes back as :data:`NOT_CALCULABLE` with the missing inputs named. Nothing
  in this package ever substitutes 0.0 for "we do not know".
* **A negative headroom is kept negative.** The governed calculation retains
  the real figure; flooring at zero is a PRESENTATION choice the UI makes for
  the headroom tile, alongside an explicit deficiency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Vocabularies (closed)
# --------------------------------------------------------------------------- #

#: Governed loan-level eligibility statuses.
ELIGIBLE = "ELIGIBLE"
INELIGIBLE = "INELIGIBLE"
UNDETERMINED = "UNDETERMINED"
ELIGIBILITY_STATUSES = (ELIGIBLE, INELIGIBLE, UNDETERMINED)

#: The sentinel a measure carries when its inputs do not exist. Deliberately a
#: string: a caller that forgets to handle it renders "NOT_CALCULABLE", which
#: is loud, rather than a plausible-looking number, which is not.
NOT_CALCULABLE = "NOT_CALCULABLE"

#: Canonical field names produced by the governed eligibility derivation.
FIELD_ELIGIBLE = "borrowing_base_eligible"
FIELD_ELIGIBILITY_STATUS = "borrowing_base_eligibility_status"
FIELD_ELIGIBILITY_REASON = "borrowing_base_eligibility_reason"
FIELD_FACILITY_ID = "borrowing_base_facility_id"
CANONICAL_FIELDS = (FIELD_ELIGIBLE, FIELD_ELIGIBILITY_STATUS,
                    FIELD_ELIGIBILITY_REASON, FIELD_FACILITY_ID)

#: How a concentration breach affects the borrowing base. v1 supports exactly
#: one behaviour — deduct nothing — because Schedule 8 establishes the tests
#: and NOT the contractual consequence of failing one. ``exclude_excess`` is
#: recognised by the configuration loader and refused by the calculator until
#: an approved facility rule supplies the wording; it is not invented here.
TREATMENT_MONITOR_ONLY = "monitor_only"
TREATMENT_EXCLUDE_EXCESS = "exclude_excess"
BORROWING_BASE_TREATMENTS = (TREATMENT_MONITOR_ONLY, TREATMENT_EXCLUDE_EXCESS)

#: Facility environments. Only ``prototype`` may honour a prototype assumption.
ENV_PROTOTYPE = "prototype"
ENV_PRODUCTION = "production"
ENVIRONMENTS = (ENV_PROTOTYPE, ENV_PRODUCTION)

#: The population a concentration test is measured over.
POPULATION_ELIGIBLE = "eligible_mortgage_loans"
POPULATION_ALL_FUNDED = "all_funded_loans"
POPULATIONS = (POPULATION_ELIGIBLE, POPULATION_ALL_FUNDED)

#: Reason codes the derivation itself raises (a configured rule supplies its
#: own). Closed so a reader can enumerate why a book is undetermined.
REASON_NO_APPROVED_RULES = "no_approved_eligibility_rules"
REASON_PROTOTYPE_ASSUMPTION = "prototype_financing_portfolio_assumption"
REASON_RULES_SATISFIED = "all_approved_eligibility_rules_satisfied"
REASON_RULE_INPUT_MISSING = "eligibility_rule_input_missing"
REASON_OUTSIDE_FINANCING_PORTFOLIO = "outside_financing_portfolio"

#: The words every surface uses for the demonstration assumption. Written once
#: so the derivation receipt, the calculation, the borrowing-base envelope and
#: the dashboard banner cannot drift into describing it differently — or, worse,
#: into one of them not describing it at all.
PROTOTYPE_ASSUMPTION_NOTE = (
    "prototype_assume_financing_portfolio_eligible: every loan in the "
    "configured Financing Portfolio is treated as an Eligible Mortgage Loan. "
    "This is a PROTOTYPE ASSUMPTION, not a contractual eligibility "
    "determination.")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _from_known(cls, d: Dict[str, Any]):
    known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
    return cls(**known)


class FacilityConfigError(ValueError):
    """The facility configuration is not usable. Operator-facing message."""


# --------------------------------------------------------------------------- #
# Eligibility rules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EligibilityRule:
    """One approved contractual criterion an Eligible Mortgage Loan must meet.

    A rule reads ONE governed input — either a ``field_role`` from the shared
    concentration-test field-role vocabulary (preferred: it survives a column
    rename) or an explicit canonical ``field`` — and compares it. Rules are
    conjunctive: a loan is eligible only when every rule passes.

    A rule whose input is absent or blank on a loan does not fail that loan and
    does not pass it: the loan becomes UNDETERMINED, naming the missing input.
    That is the whole point of a tri-state — "we cannot tell" is not "no".
    """

    rule_id: str
    description: str = ""
    field_role: str = ""
    field: str = ""
    #: max | min | equals | not_equals | in | not_in | present
    operator: str = "max"
    value: Any = None
    reason_code: str = ""
    reason: str = ""
    source_reference: str = ""

    OPERATORS = ("max", "min", "equals", "not_equals", "in", "not_in", "present")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EligibilityRule":
        return _from_known(cls, d)

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.rule_id:
            problems.append("An eligibility rule is missing rule_id.")
        if not self.field_role and not self.field:
            problems.append(
                f"{self.rule_id or 'rule'}: declare field_role or field.")
        if self.operator not in self.OPERATORS:
            problems.append(
                f"{self.rule_id or 'rule'}: unknown operator "
                f"{self.operator!r} (expected one of {list(self.OPERATORS)}).")
        if self.operator in ("max", "min") and not isinstance(
                self.value, (int, float)):
            problems.append(
                f"{self.rule_id or 'rule'}: '{self.operator}' needs a numeric "
                "value.")
        if self.operator in ("in", "not_in") and not isinstance(
                self.value, (list, tuple)):
            problems.append(
                f"{self.rule_id or 'rule'}: '{self.operator}' needs a list of "
                "values.")
        return problems


# --------------------------------------------------------------------------- #
# Facility configuration
# --------------------------------------------------------------------------- #
@dataclass
class FacilityConfiguration:
    """One approved funding facility's governed terms.

    Loaded from configuration — never constructed from literals in production
    code. ``config_version`` and ``content_hash`` are what a receipt cites so a
    later reviewer can reproduce the exact terms a figure was calculated on.
    """

    client_id: str = ""
    facility_id: str = ""
    facility_label: str = ""
    facility_type: str = "warehouse"
    currency: str = "GBP"

    commitment: Optional[float] = None
    #: Ratio, not percent. 1.03 == 103%.
    advance_rate: Optional[float] = None
    concentration_denominator_floor: Optional[float] = None
    current_drawn_amount: Optional[float] = None
    current_drawn_amount_as_of: str = ""

    effective_date: str = ""
    maturity_date: str = ""
    environment: str = ENV_PRODUCTION

    governance: Dict[str, Any] = field(default_factory=dict)
    financing_portfolio: Dict[str, Any] = field(default_factory=dict)

    eligibility_rule_version: str = ""
    eligibility_rules: List[EligibilityRule] = field(default_factory=list)
    prototype_assume_financing_portfolio_eligible: bool = False

    concentration_population: str = POPULATION_ELIGIBLE
    borrowing_base_treatment: str = TREATMENT_MONITOR_ONLY

    #: Where this configuration was read from, for disclosure.
    config_source: str = ""
    config_version: str = ""
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    @property
    def advance_rate_pct(self) -> Optional[float]:
        return None if self.advance_rate is None else round(
            self.advance_rate * 100.0, 6)

    @property
    def prototype_assumption_active(self) -> bool:
        """Whether the demonstration assumption is BOTH configured AND allowed.

        Configured alone is not enough: the assumption is honoured only under
        ``environment: prototype``, so it can never become the production
        default by a copy-paste of a config block.
        """
        return bool(self.prototype_assume_financing_portfolio_eligible
                    and self.environment == ENV_PROTOTYPE)

    @property
    def eligibility_governed(self) -> bool:
        """True when approved contractual criteria exist for this facility."""
        return bool(self.eligibility_rules)

    @property
    def drawn_available(self) -> bool:
        return isinstance(self.current_drawn_amount, (int, float))

    # ------------------------------------------------------------------ #
    def content_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("config_source", None)
        return stable_hash(canonical_json(payload))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["eligibility_rules"] = [r.to_dict() for r in self.eligibility_rules]
        d["advance_rate_pct"] = self.advance_rate_pct
        return d

    def summary(self) -> Dict[str, Any]:
        """The operator-facing terms, without the rule bodies."""
        return {
            "clientId": self.client_id,
            "facilityId": self.facility_id,
            "facilityLabel": self.facility_label or self.facility_id,
            "facilityType": self.facility_type,
            "currency": self.currency,
            "commitment": self.commitment,
            "advanceRate": self.advance_rate,
            "advanceRatePct": self.advance_rate_pct,
            "concentrationDenominatorFloor": self.concentration_denominator_floor,
            "currentDrawnAmount": self.current_drawn_amount,
            "currentDrawnAmountAsOf": self.current_drawn_amount_as_of or None,
            "effectiveDate": self.effective_date or None,
            "maturityDate": self.maturity_date or None,
            "environment": self.environment,
            "eligibilityRuleVersion": self.eligibility_rule_version or None,
            "eligibilityRuleCount": len(self.eligibility_rules),
            "eligibilityGoverned": self.eligibility_governed,
            "prototypeAssumptionActive": self.prototype_assumption_active,
            "concentrationPopulation": self.concentration_population,
            "borrowingBaseTreatment": self.borrowing_base_treatment,
            "configSource": self.config_source,
            "configVersion": self.config_version,
            "configHash": self.content_hash(),
            "governance": dict(self.governance or {}),
        }

    # ------------------------------------------------------------------ #
    def validate(self) -> List[str]:
        """Structural problems, in plain English. Empty means usable."""
        problems: List[str] = []
        if not self.facility_id:
            problems.append("The facility is missing facility_id.")
        if self.environment not in ENVIRONMENTS:
            problems.append(
                f"{self.facility_id}: unknown environment "
                f"{self.environment!r} (expected one of {list(ENVIRONMENTS)}).")
        if self.commitment is not None and self.commitment < 0:
            problems.append(f"{self.facility_id}: commitment cannot be negative.")
        if self.advance_rate is not None and self.advance_rate < 0:
            problems.append(f"{self.facility_id}: advance_rate cannot be negative.")
        if (self.concentration_denominator_floor is not None
                and self.concentration_denominator_floor < 0):
            problems.append(
                f"{self.facility_id}: concentration_denominator_floor cannot "
                "be negative.")
        if self.concentration_population not in POPULATIONS:
            problems.append(
                f"{self.facility_id}: unknown concentration population "
                f"{self.concentration_population!r}.")
        if self.borrowing_base_treatment not in BORROWING_BASE_TREATMENTS:
            problems.append(
                f"{self.facility_id}: unknown borrowing_base_treatment "
                f"{self.borrowing_base_treatment!r}.")
        for rule in self.eligibility_rules:
            problems.extend(rule.validate())
        if (self.prototype_assume_financing_portfolio_eligible
                and self.environment != ENV_PROTOTYPE):
            problems.append(
                f"{self.facility_id}: the prototype eligibility assumption is "
                "configured but the facility is not marked "
                "`environment: prototype`. It will NOT be honoured.")
        return problems

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FacilityConfiguration":
        d = dict(d or {})
        d["eligibility_rules"] = [EligibilityRule.from_dict(r)
                                  for r in (d.get("eligibility_rules") or [])]
        return _from_known(cls, d)
