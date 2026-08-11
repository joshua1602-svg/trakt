"""trakt_core.valuation — valuation observations, and the policy that selects one.

Two things that must not be conflated, and this module keeps them apart:

    OBSERVATION   a fact.  "full valuation, £300,000, 30 June 2026, Countrywide"
    POLICY        a rule.  "for current LTV prefer a full valuation <= 24 months
                            old; otherwise the approved indexed valuation"

A collateral asset has 0..n observations. Which one backs a reported LTV is a
credit decision, so it lives in versioned configuration
(``config/system/valuation_selection_policy.yaml``) and is applied
deterministically here — never inferred, and never decided inside an agent
prompt.

The point of the separation is that ``explain_values`` can then answer *why*:

    current_loan_to_value = 61.2%
      balance          183,450  (current_principal_balance)
      valuation        300,000  (val_… full, 2026-06-30, Countrywide)
      selected under   CURRENT_LTV_SELECTION@v1
      rejected         val_… indexed  — a full valuation within 24 months existed
                       val_… full     — 2019-01-04 is older than 24 months
      calculation      CURRENT_LTV@v1

Not a second source of truth. The canonical tape still carries
``current_valuation_amount`` and ``current_loan_to_value``, and the LTV
derivation in ``engine.gate_2_transform.canonical_transform`` is untouched. This
module explains and reconciles; it does not recompute (``recomputes: false`` in
the configuration says so explicitly).

Standard library plus an optional pandas import, so the pipeline, the API and a
CLI can all use it.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import config_cache
from .errors import ErrorCode, TraktError

DEFAULT_POLICY_CONFIG = "config/system/valuation_selection_policy.yaml"
POLICY_CONFIG_ENV = "TRAKT_VALUATION_POLICY_CONFIG"

#: Written beside the canonical output by the ingestion pipeline.
VALUATION_SIDECAR_NAME = "valuations.parquet"
VALUATION_SIDECAR_CSV_NAME = "valuations.csv"

# --------------------------------------------------------------------------- #
# Observation types. A closed vocabulary: a policy expresses a preference over
# these, so an unrecognised type cannot be silently preferred.
# --------------------------------------------------------------------------- #
TYPE_FULL = "full"
TYPE_DRIVE_BY = "drive_by"
TYPE_DESKTOP = "desktop"
TYPE_AVM = "avm"
TYPE_INDEXED = "indexed"
TYPE_PURCHASE_PRICE = "purchase_price"

KNOWN_VALUATION_TYPES = (TYPE_FULL, TYPE_DRIVE_BY, TYPE_DESKTOP, TYPE_AVM,
                         TYPE_INDEXED, TYPE_PURCHASE_PRICE)

STATUS_APPROVED = "approved"
STATUS_SUPERSEDED = "superseded"
STATUS_REJECTED = "rejected"


def _parse_date(value: Any) -> Optional[_dt.date]:
    if value is None or value == "":
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    text = str(value).strip()[:10]
    try:
        return _dt.date.fromisoformat(text)
    except ValueError:
        return None


def _months_between(earlier: _dt.date, later: _dt.date) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


# --------------------------------------------------------------------------- #
# The observation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValuationObservation:
    """One dated valuation of one collateral asset.

    Nine fields, and no more. Each earns its place:

    ``valuation_id``    addressable, so an explanation can name exactly which
                        observation was used
    ``collateral_id``   the parent — what makes this a repeatable child
    ``valuation_type``  what a policy expresses a preference over
    ``amount``/``currency``  the fact
    ``valuation_date``  when it was performed — what an age limit measures
    ``effective_date``  what it speaks to. Kept SEPARATE because an indexed
                        valuation is produced later than the date it describes,
                        and collapsing the two would make every index look stale
    ``source``          who says so — a provider or an index name
    ``status``          whether it may be selected at all
    ``provenance_ref``  back to the snapshot and field it came from

    Deliberately absent, each with the trigger that would add it:
    ``supersedes`` (derivable from type + date ordering — add when an
    observation is genuinely retracted rather than superseded) and a separate
    ``methodology`` (``valuation_type`` carries today's distinction — add when a
    client supplies two AVMs with different methodologies).
    """

    valuation_id: str
    collateral_id: str
    valuation_type: str
    amount: Optional[float] = None
    currency: Optional[str] = None
    valuation_date: Optional[str] = None
    effective_date: Optional[str] = None
    source: Optional[str] = None
    status: str = STATUS_APPROVED
    provenance_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "ValuationObservation":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (doc or {}).items() if k in known})

    @property
    def observed_on(self) -> Optional[_dt.date]:
        return _parse_date(self.valuation_date)

    @property
    def speaks_to(self) -> Optional[_dt.date]:
        return _parse_date(self.effective_date) or self.observed_on


def make_valuation_id(collateral_id: str, valuation_type: str,
                      valuation_date: Optional[str],
                      source: Optional[str] = None) -> str:
    """A deterministic id, so re-ingesting the same snapshot produces the same
    observation identity and an explanation stays citable across runs."""
    blob = "|".join([str(collateral_id), str(valuation_type),
                     str(valuation_date or ""), str(source or "")])
    return "val_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SelectionPolicy:
    """A versioned preference order over observation types."""

    policy_id: str
    version: int = 1
    description: str = ""
    prefer: Tuple[str, ...] = ()
    max_age_months: Dict[str, Optional[int]] = field(default_factory=dict)
    require_status: Tuple[str, ...] = (STATUS_APPROVED,)
    on_no_qualifying_observation: str = "refuse"

    @property
    def reference(self) -> str:
        """``CURRENT_LTV_SELECTION@v1`` — what an explanation cites."""
        return f"{self.policy_id}@v{self.version}"


@dataclass(frozen=True)
class CalculationRule:
    """A named, versioned calculation that consumes a selection."""

    calculation_id: str
    version: int = 1
    description: str = ""
    numerator: Optional[str] = None
    numerator_fallback: Optional[str] = None
    selection_policy: Optional[str] = None
    unit: Optional[str] = None
    canonical_field: Optional[str] = None
    #: False means Trakt does NOT recompute this here — the canonical tape's
    #: value stands and this rule only explains how it was arrived at. There is
    #: one LTV in Trakt; this module never becomes a second one.
    recomputes: bool = False

    @property
    def reference(self) -> str:
        return f"{self.calculation_id}@v{self.version}"


@dataclass(frozen=True)
class SelectionResult:
    """Which observation a policy chose, and why the others were not chosen."""

    policy: SelectionPolicy
    selected: Optional[ValuationObservation] = None
    #: ``[{valuation_id, valuation_type, reason}]`` — every observation the
    #: policy declined, with the reason. This is the half that makes an
    #: explanation defensible rather than merely stated.
    rejected: Tuple[Dict[str, Any], ...] = ()
    reason: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self.selected is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.reference,
            "policy_description": self.policy.description,
            "selected": self.selected.to_dict() if self.selected else None,
            "rejected": [dict(r) for r in self.rejected],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ValuationPolicyBook:
    """Every declared selection policy and calculation rule."""

    policies: Dict[str, SelectionPolicy] = field(default_factory=dict)
    calculations: Dict[str, CalculationRule] = field(default_factory=dict)
    configured: bool = False

    def policy(self, policy_id: str) -> Optional[SelectionPolicy]:
        return self.policies.get(str(policy_id))

    def calculation(self, calculation_id: str) -> Optional[CalculationRule]:
        return self.calculations.get(str(calculation_id))

    def calculation_for_field(self, canonical_field: str) -> Optional[CalculationRule]:
        for rule in self.calculations.values():
            if rule.canonical_field == canonical_field:
                return rule
        return None


def load_valuation_policies(config_path: Optional[str | Path] = None
                            ) -> ValuationPolicyBook:
    """Load the policy book, cached on content identity.

    An absent or unreadable file yields an EMPTY book rather than a default
    policy. A selection rule that nobody wrote down must not be invented: with no
    book, ``select_valuation`` refuses and an explanation says the policy is
    unconfigured — which is a true statement an operator can act on.
    """
    import os

    path = Path(config_path
                or os.environ.get(POLICY_CONFIG_ENV)
                or DEFAULT_POLICY_CONFIG)

    def _load() -> ValuationPolicyBook:
        if not path.exists():
            return ValuationPolicyBook(configured=False)
        try:
            import yaml
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 - a bad policy file fails closed
            return ValuationPolicyBook(configured=False)

        policies: Dict[str, SelectionPolicy] = {}
        for policy_id, spec in (raw.get("policies") or {}).items():
            spec = spec or {}
            prefer = tuple(str(t) for t in (spec.get("prefer") or ()))
            unknown = [t for t in prefer if t not in KNOWN_VALUATION_TYPES]
            if unknown:
                # A preference for a type that cannot exist would silently never
                # match. Refuse the whole book rather than apply a policy that
                # does not mean what it says.
                return ValuationPolicyBook(configured=False)
            policies[str(policy_id)] = SelectionPolicy(
                policy_id=str(policy_id),
                version=int(spec.get("version") or 1),
                description=str(spec.get("description") or "").strip(),
                prefer=prefer,
                max_age_months={str(k): (int(v) if v is not None else None)
                                for k, v in (spec.get("max_age_months") or {}).items()},
                require_status=tuple(str(s) for s in
                                     (spec.get("require_status") or (STATUS_APPROVED,))),
                on_no_qualifying_observation=str(
                    spec.get("on_no_qualifying_observation") or "refuse"))

        calculations: Dict[str, CalculationRule] = {}
        for calc_id, spec in (raw.get("calculations") or {}).items():
            spec = spec or {}
            calculations[str(calc_id)] = CalculationRule(
                calculation_id=str(calc_id),
                version=int(spec.get("version") or 1),
                description=str(spec.get("description") or "").strip(),
                numerator=spec.get("numerator"),
                numerator_fallback=spec.get("numerator_fallback"),
                selection_policy=spec.get("selection_policy"),
                unit=spec.get("unit"),
                canonical_field=spec.get("canonical_field"),
                recomputes=bool(spec.get("recomputes", False)))

        return ValuationPolicyBook(policies=policies, calculations=calculations,
                                   configured=True)

    return config_cache.get_or_load("valuation_policies", path, _load)


# --------------------------------------------------------------------------- #
# The selection — deterministic, and it explains itself
# --------------------------------------------------------------------------- #
def select_valuation(observations: Sequence[ValuationObservation],
                     policy: SelectionPolicy,
                     *,
                     as_of: Optional[str] = None) -> SelectionResult:
    """Apply ``policy`` to ``observations``. Deterministic and total.

    Every observation is either selected or rejected **with a stated reason** —
    there is no silent discard, because "why not that one?" is the question a
    credit committee actually asks.

    Ties are broken by the most recent ``valuation_date``, then by
    ``valuation_id`` so the result is stable across runs.
    """
    anchor = _parse_date(as_of) or _dt.date.today()
    rejected: List[Dict[str, Any]] = []
    eligible: Dict[str, List[ValuationObservation]] = {}

    for obs in observations:
        if obs.status not in policy.require_status:
            rejected.append({"valuation_id": obs.valuation_id,
                             "valuation_type": obs.valuation_type,
                             "reason": f"status is {obs.status!r}, and the policy "
                                       f"requires one of {list(policy.require_status)}"})
            continue
        if obs.amount is None:
            rejected.append({"valuation_id": obs.valuation_id,
                             "valuation_type": obs.valuation_type,
                             "reason": "the observation carries no amount"})
            continue
        if obs.valuation_type not in policy.prefer:
            rejected.append({"valuation_id": obs.valuation_id,
                             "valuation_type": obs.valuation_type,
                             "reason": "this policy expresses no preference for "
                                       f"{obs.valuation_type!r}"})
            continue
        limit = policy.max_age_months.get(obs.valuation_type)
        observed = obs.observed_on
        if limit is not None:
            if observed is None:
                rejected.append({"valuation_id": obs.valuation_id,
                                 "valuation_type": obs.valuation_type,
                                 "reason": "the observation carries no valuation "
                                           "date, so its age cannot be checked"})
                continue
            age = _months_between(observed, anchor)
            if age > limit:
                rejected.append({"valuation_id": obs.valuation_id,
                                 "valuation_type": obs.valuation_type,
                                 "reason": f"{age} months old at {anchor.isoformat()}, "
                                           f"and the policy allows {limit}"})
                continue
        eligible.setdefault(obs.valuation_type, []).append(obs)

    for valuation_type in policy.prefer:
        candidates = eligible.get(valuation_type)
        if not candidates:
            continue
        candidates.sort(key=lambda o: (o.observed_on or _dt.date.min, o.valuation_id),
                        reverse=True)
        chosen, *runners_up = candidates
        for other in runners_up:
            rejected.append({"valuation_id": other.valuation_id,
                             "valuation_type": other.valuation_type,
                             "reason": "a more recent observation of the same type "
                                       "was available"})
        # Everything the policy ranked lower is rejected for that reason, so the
        # explanation records the preference order actually applied.
        for lower in policy.prefer[policy.prefer.index(valuation_type) + 1:]:
            for other in eligible.get(lower, ()):
                rejected.append({"valuation_id": other.valuation_id,
                                 "valuation_type": other.valuation_type,
                                 "reason": f"a qualifying {valuation_type!r} "
                                           "valuation was available, which this "
                                           "policy prefers"})
        return SelectionResult(policy=policy, selected=chosen,
                               rejected=tuple(rejected),
                               reason=f"selected the most recent qualifying "
                                      f"{valuation_type!r} observation")

    return SelectionResult(
        policy=policy, selected=None, rejected=tuple(rejected),
        reason=("no observation qualified under this policy" if observations
                else "no valuation observations are available for this collateral"))
