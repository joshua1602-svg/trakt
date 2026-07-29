"""
engine.annex_delivery_agent.evidence
====================================

Governed evidence of what an artefact builder *did to the data*.

A schema-valid XML file is a weak claim: the builder may have reached validity
by inserting no-data sentinels, coercing a value into the shape a branch
demanded, or dropping an optional field it could not place. Those are legitimate
behaviours of the proven Annex 2 builder and this package does not change them —
but an operator who is about to submit a regulatory report is entitled to see
them itemised before approving.

Every record here answers the same six questions:

    what was affected, how many records, what was there before, what is there
    now, which rule caused it, and does a human need to look at it.

Annex-neutral by construction: the shared agent aggregates and reports these
records without knowing what an ``RREL12`` is.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_BLOCKING = "blocking"

#: How many distinct example values to keep per evidence record. Enough to
#: recognise a pattern; bounded so an 11k-record run does not produce an
#: unreadable audit file.
MAX_SAMPLES = 5


def _sample(values: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for v in values:
        s = str(v)
        if s not in seen:
            seen.append(s)
        if len(seen) >= MAX_SAMPLES:
            break
    return seen


# --------------------------------------------------------------------------- #
# Evidence records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AutomaticFill:
    """A value the builder inserted that was not present in the input data."""

    field_code: str
    xml_path: str
    records_affected: int
    original_state: str
    generated_value: str
    reason: str
    severity: str = SEVERITY_WARNING
    review_required: bool = True
    #: How this was established — ``"structural"`` (derived from the finished
    #: artefact), ``"intercepted"`` (observed at the call site), or
    #: ``"reconciled"`` (input/output count difference).
    detection: str = "structural"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_code": self.field_code,
            "xml_path": self.xml_path,
            "records_affected": self.records_affected,
            "original_state": self.original_state,
            "generated_value": self.generated_value,
            "reason": self.reason,
            "severity": self.severity,
            "review_required": self.review_required,
            "detection": self.detection,
        }


@dataclass(frozen=True)
class Coercion:
    """An input value the builder changed to satisfy a mapping branch."""

    field_code: str
    records_affected: int
    original_value_sample: Tuple[str, ...]
    generated_value: str
    reason: str
    severity: str = SEVERITY_WARNING
    review_required: bool = True
    detection: str = "intercepted"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_code": self.field_code,
            "records_affected": self.records_affected,
            "original_value_sample": list(self.original_value_sample),
            "generated_value": self.generated_value,
            "reason": self.reason,
            "severity": self.severity,
            "review_required": self.review_required,
            "detection": self.detection,
        }


@dataclass(frozen=True)
class Omission:
    """A field that carried data but did not reach the artefact."""

    field_code: str
    records_affected: int
    original_value_sample: Tuple[str, ...]
    reason: str
    optional: bool = True
    severity: str = SEVERITY_WARNING
    review_required: bool = True
    detection: str = "analytic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_code": self.field_code,
            "records_affected": self.records_affected,
            "original_value_sample": list(self.original_value_sample),
            "reason": self.reason,
            "optional": self.optional,
            "severity": self.severity,
            "review_required": self.review_required,
            "detection": self.detection,
        }


@dataclass(frozen=True)
class UnsupportedField:
    """A field the schema/mapping universe knows about that this route cannot emit."""

    field_code: str
    field_name: str
    reason: str
    mandatory: bool = False
    severity: str = SEVERITY_INFO
    review_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_code": self.field_code,
            "field_name": self.field_name,
            "reason": self.reason,
            "mandatory": self.mandatory,
            "severity": self.severity,
            "review_required": self.review_required,
        }


@dataclass(frozen=True)
class StructuralAssumption:
    """A shape the builder assumed rather than derived from the data.

    Annex 2's builder emits one flat ``UndrlygXpsrRcrd`` per CSV row and one
    collateral block per record. Where the source could legitimately carry more
    than one collateral item per exposure, that assumption is a real disclosure
    obligation, not an implementation detail.
    """

    name: str
    description: str
    records_affected: int = 0
    severity: str = SEVERITY_WARNING
    review_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "records_affected": self.records_affected,
            "severity": self.severity,
            "review_required": self.review_required,
        }


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #


@dataclass
class TransformationEvidence:
    """Everything the builder did to the data, in one governed object."""

    automatic_fills: List[AutomaticFill] = field(default_factory=list)
    coercions: List[Coercion] = field(default_factory=list)
    omissions: List[Omission] = field(default_factory=list)
    unsupported_fields: List[UnsupportedField] = field(default_factory=list)
    structural_assumptions: List[StructuralAssumption] = field(default_factory=list)
    #: Input-vs-output no-data reconciliation, the independent cross-check on
    #: the structural fill counts.
    nd_reconciliation: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # -- counts ------------------------------------------------------------- #
    @property
    def automatic_fill_count(self) -> int:
        return sum(f.records_affected for f in self.automatic_fills)

    @property
    def coercion_count(self) -> int:
        return sum(c.records_affected for c in self.coercions)

    @property
    def omission_count(self) -> int:
        return sum(o.records_affected for o in self.omissions)

    @property
    def review_required(self) -> bool:
        return any(item.review_required for item in (
            *self.automatic_fills, *self.coercions, *self.omissions,
            *self.unsupported_fields, *self.structural_assumptions))

    def operator_warnings(self) -> List[str]:
        """Plain-English lines for the operator console.

        Deliberately says *what was invented*, never "validation passed".
        """
        out: List[str] = []
        for f in self.automatic_fills:
            if not f.records_affected:
                continue
            out.append(
                f"{f.records_affected:,} value(s) for {f.field_code} were filled "
                f"automatically with '{f.generated_value}' because {f.reason}. "
                f"These were not supplied by your data.")
        for c in self.coercions:
            if not c.records_affected:
                continue
            sample = ", ".join(c.original_value_sample[:3]) or "(blank)"
            out.append(
                f"{c.records_affected:,} value(s) for {c.field_code} were changed to "
                f"'{c.generated_value}' to fit the required format (examples: {sample}). "
                f"{c.reason}")
        for o in self.omissions:
            if not o.records_affected:
                continue
            out.append(
                f"{o.records_affected:,} value(s) for {o.field_code} were present in "
                f"the prepared data but do not appear in the report: {o.reason}")
        for u in self.unsupported_fields:
            out.append(
                f"{u.field_code} ({u.field_name}) cannot be produced by this route: "
                f"{u.reason}")
        for a in self.structural_assumptions:
            out.append(f"{a.description}")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "automatic_fill_count": self.automatic_fill_count,
            "automatic_fills": [f.to_dict() for f in self.automatic_fills],
            "coercion_count": self.coercion_count,
            "coercions": [c.to_dict() for c in self.coercions],
            "omission_count": self.omission_count,
            "omissions": [o.to_dict() for o in self.omissions],
            "unsupported_fields": [u.to_dict() for u in self.unsupported_fields],
            "structural_assumptions": [a.to_dict() for a in self.structural_assumptions],
            "nd_reconciliation": dict(self.nd_reconciliation),
            "review_required": self.review_required,
            "operator_warnings": self.operator_warnings(),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "TransformationEvidence":
        d = dict(data or {})
        ev = cls()
        for f in d.get("automatic_fills") or ():
            ev.automatic_fills.append(AutomaticFill(
                field_code=str(f.get("field_code", "")),
                xml_path=str(f.get("xml_path", "")),
                records_affected=int(f.get("records_affected") or 0),
                original_state=str(f.get("original_state", "")),
                generated_value=str(f.get("generated_value", "")),
                reason=str(f.get("reason", "")),
                severity=str(f.get("severity", SEVERITY_WARNING)),
                review_required=bool(f.get("review_required", True)),
                detection=str(f.get("detection", "structural"))))
        for c in d.get("coercions") or ():
            ev.coercions.append(Coercion(
                field_code=str(c.get("field_code", "")),
                records_affected=int(c.get("records_affected") or 0),
                original_value_sample=tuple(c.get("original_value_sample") or ()),
                generated_value=str(c.get("generated_value", "")),
                reason=str(c.get("reason", "")),
                severity=str(c.get("severity", SEVERITY_WARNING)),
                review_required=bool(c.get("review_required", True)),
                detection=str(c.get("detection", "intercepted"))))
        for o in d.get("omissions") or ():
            ev.omissions.append(Omission(
                field_code=str(o.get("field_code", "")),
                records_affected=int(o.get("records_affected") or 0),
                original_value_sample=tuple(o.get("original_value_sample") or ()),
                reason=str(o.get("reason", "")),
                optional=bool(o.get("optional", True)),
                severity=str(o.get("severity", SEVERITY_WARNING)),
                review_required=bool(o.get("review_required", True)),
                detection=str(o.get("detection", "analytic"))))
        for u in d.get("unsupported_fields") or ():
            ev.unsupported_fields.append(UnsupportedField(
                field_code=str(u.get("field_code", "")),
                field_name=str(u.get("field_name", "")),
                reason=str(u.get("reason", "")),
                mandatory=bool(u.get("mandatory")),
                severity=str(u.get("severity", SEVERITY_INFO)),
                review_required=bool(u.get("review_required", False))))
        for a in d.get("structural_assumptions") or ():
            ev.structural_assumptions.append(StructuralAssumption(
                name=str(a.get("name", "")),
                description=str(a.get("description", "")),
                records_affected=int(a.get("records_affected") or 0),
                severity=str(a.get("severity", SEVERITY_WARNING)),
                review_required=bool(a.get("review_required", True))))
        ev.nd_reconciliation = dict(d.get("nd_reconciliation") or {})
        ev.notes = list(d.get("notes") or [])
        return ev


# --------------------------------------------------------------------------- #
# Collector used by providers while a build is running
# --------------------------------------------------------------------------- #


class CoercionCollector:
    """Accumulates intercepted coercions without holding 11k rows in memory.

    Keyed by ``(field_code, produced_value)``; keeps a bounded sample of the
    inputs that produced it plus an exact affected-record count.
    """

    def __init__(self) -> None:
        self._counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._samples: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    def record(self, field_code: str, original: str, produced: str) -> None:
        if original == produced:
            return
        key = (str(field_code), str(produced))
        self._counts[key] += 1
        samples = self._samples[key]
        if len(samples) < MAX_SAMPLES and original not in samples:
            samples.append(original if original != "" else "(blank)")

    def as_evidence(self, *, reason_for: Mapping[str, str]) -> List[Coercion]:
        out: List[Coercion] = []
        for (code, produced), count in sorted(self._counts.items()):
            out.append(Coercion(
                field_code=code,
                records_affected=count,
                original_value_sample=tuple(self._samples[(code, produced)]),
                generated_value=produced,
                reason=reason_for.get(code, "A builder branch rule required this shape."),
            ))
        return out

    def __len__(self) -> int:  # pragma: no cover - trivial
        return sum(self._counts.values())


__all__ = [
    "SEVERITY_INFO", "SEVERITY_WARNING", "SEVERITY_BLOCKING", "MAX_SAMPLES",
    "AutomaticFill", "Coercion", "Omission", "UnsupportedField",
    "StructuralAssumption", "TransformationEvidence", "CoercionCollector",
]
