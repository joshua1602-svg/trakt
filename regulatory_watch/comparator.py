"""regulatory_watch.comparator — deterministic Annex 2 spec-vs-spec diff.

Pure functions over two :class:`NormalizedSpec` objects. No I/O, no network,
no model. The same pair of specs always produces byte-identical output, in a
stable order.

Three things this module is careful about:

* **A hash change is not a regulatory change.** Source-byte comparison lives in
  :func:`regulatory_watch.contracts.source_status_for`; this module only ever
  compares *parsed requirements*. A workbook can be re-issued, re-saved or
  re-ordered without a single Annex 2 requirement moving, and that case must
  report zero regulatory deltas.
* **Unknown is not "unchanged".** Any attribute the normalizer could not derive
  on either side downgrades the affected delta to ``review-required``, and an
  otherwise-clean comparison over a spec containing unresolved attributes is
  reported as ``SPEC_UNDETERMINED`` rather than ``SPEC_UNCHANGED``.
* **Ordering is compared as a sequence, not as an index.** Adding a field must
  not report every following field as re-ordered.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contracts import (
    DETERMINISTIC,
    ENUM_CHANGED,
    FIELD_ADDED,
    FIELD_DESCRIPTION_CHANGED,
    FIELD_REMOVED,
    FORMAT_CHANGED,
    MANDATORY_STATUS_CHANGED,
    MULTIPLICITY_CHANGED,
    ND_PERMISSION_CHANGED,
    NormalizedSpec,
    ORDER_CHANGED,
    Provenance,
    REVIEW_REQUIRED,
    SPEC_CHANGED,
    SPEC_UNCHANGED,
    SPEC_UNDETERMINED,
    SpecDelta,
    SpecField,
    VALIDATION_RULE_CHANGED,
    XML_PATH_CHANGED,
    severity_for,
)


class ComparatorError(ValueError):
    """The two specs cannot be meaningfully compared."""


#: change type -> the SpecField attributes it is derived from. Declaring this
#: once keeps confidence scoring honest: a delta is deterministic only when
#: every attribute behind it was actually parsed on both sides.
ATTRIBUTES_BY_CHANGE_TYPE: Dict[str, Tuple[str, ...]] = {
    FIELD_DESCRIPTION_CHANGED: ("label", "description"),
    FORMAT_CHANGED: ("data_type", "format_pattern"),
    MANDATORY_STATUS_CHANGED: ("mandatory",),
    ND_PERMISSION_CHANGED: ("nd_allowed",),
    ENUM_CHANGED: ("enum_type", "enum_values"),
    XML_PATH_CHANGED: ("xml_path", "xml_tag", "value_paths"),
    MULTIPLICITY_CHANGED: ("multiplicity",),
    VALIDATION_RULE_CHANGED: ("validation_rules",),
}

#: Evidence rows carried per delta per side. Capped for report size; the cap is
#: deterministic (first N in parse order), never sampled.
_EVIDENCE_PER_SIDE = 3


def _evidence(*fields: Optional[SpecField]) -> List[Provenance]:
    out: List[Provenance] = []
    for f in fields:
        if f is None:
            continue
        out.extend(f.provenance[:_EVIDENCE_PER_SIDE])
    return out


def _confidence(change_type: str, old: Optional[SpecField],
                new: Optional[SpecField]) -> str:
    attrs = ATTRIBUTES_BY_CHANGE_TYPE.get(change_type, ())
    for side in (old, new):
        if side is None:
            continue
        if any(a in side.unresolved for a in attrs):
            return REVIEW_REQUIRED
    return DETERMINISTIC


def _values(field_obj: SpecField, attrs: Sequence[str]) -> Dict[str, Any]:
    return {a: getattr(field_obj, a) for a in attrs}


def _longest_common_subsequence(a: Sequence[str],
                                b: Sequence[str]) -> List[str]:
    """Classic LCS. Used so one moved field does not flag every neighbour."""
    n, m = len(a), len(b)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            table[i][j] = (table[i + 1][j + 1] + 1 if a[i] == b[j]
                           else max(table[i + 1][j], table[i][j + 1]))
    out: List[str] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def _ordered_common(spec: NormalizedSpec, common: set) -> List[str]:
    """The spec's own field sequence, restricted to codes present in both."""
    seq = [c for c in spec.order if c in common]
    if len(seq) == len(common):
        return seq
    # Fall back to order_index for any code missing from `order` (a spec loaded
    # from a partial fixture); ties broken by code for determinism.
    missing = sorted(common - set(seq),
                     key=lambda c: (spec.fields[c].order_index, c))
    return seq + missing


def compare(baseline: NormalizedSpec, candidate: NormalizedSpec,
            *, strict_parser_version: bool = True) -> List[SpecDelta]:
    """Return every detected regulatory delta, in a stable order.

    Raises :class:`ComparatorError` when the two specs were produced by
    different normalizer generations (``strict_parser_version``) or belong to
    different regimes — comparing across those is meaningless, and silently
    returning "no change" would be the worst possible answer.
    """
    if baseline.regime != candidate.regime:
        raise ComparatorError(
            f"regime mismatch: {baseline.regime!r} vs {candidate.regime!r}")
    if strict_parser_version and baseline.parser_version != candidate.parser_version:
        raise ComparatorError(
            f"parser version mismatch: {baseline.parser_version!r} vs "
            f"{candidate.parser_version!r}; re-normalize both snapshots with "
            f"one parser before comparing")

    va, vb = baseline.spec_version, candidate.spec_version
    deltas: List[SpecDelta] = []

    base_codes = set(baseline.fields)
    cand_codes = set(candidate.fields)

    # -- added / removed ---------------------------------------------------- #
    for code in sorted(cand_codes - base_codes):
        f = candidate.fields[code]
        deltas.append(SpecDelta(
            code=code, change_type=FIELD_ADDED, old_value=None,
            new_value={"label": f.label, "xml_path": f.xml_path,
                       "mandatory": f.mandatory, "nd_allowed": f.nd_allowed},
            severity=severity_for(FIELD_ADDED),
            confidence=DETERMINISTIC if not f.unresolved else REVIEW_REQUIRED,
            source_version_a=va, source_version_b=vb,
            evidence=_evidence(f),
            detail=f"{code} is published in {vb} and absent from {va}"))

    for code in sorted(base_codes - cand_codes):
        f = baseline.fields[code]
        deltas.append(SpecDelta(
            code=code, change_type=FIELD_REMOVED,
            old_value={"label": f.label, "xml_path": f.xml_path,
                       "mandatory": f.mandatory, "nd_allowed": f.nd_allowed},
            new_value=None,
            severity=severity_for(FIELD_REMOVED),
            confidence=DETERMINISTIC if not f.unresolved else REVIEW_REQUIRED,
            source_version_a=va, source_version_b=vb,
            evidence=_evidence(f),
            detail=f"{code} is published in {va} and absent from {vb}"))

    # -- attribute changes on common codes ---------------------------------- #
    common = base_codes & cand_codes
    for code in sorted(common):
        old, new = baseline.fields[code], candidate.fields[code]
        for change_type, attrs in ATTRIBUTES_BY_CHANGE_TYPE.items():
            old_values = _values(old, attrs)
            new_values = _values(new, attrs)
            if old_values == new_values:
                continue
            changed = sorted(a for a in attrs
                             if old_values[a] != new_values[a])
            deltas.append(SpecDelta(
                code=code, change_type=change_type,
                old_value={a: old_values[a] for a in changed},
                new_value={a: new_values[a] for a in changed},
                severity=severity_for(change_type),
                confidence=_confidence(change_type, old, new),
                source_version_a=va, source_version_b=vb,
                evidence=_evidence(old, new),
                detail=f"{code}: {', '.join(changed)} differ"))

    # -- ordering ------------------------------------------------------------ #
    seq_a = _ordered_common(baseline, common)
    seq_b = _ordered_common(candidate, common)
    if seq_a != seq_b:
        kept = set(_longest_common_subsequence(seq_a, seq_b))
        moved = [c for c in seq_b if c not in kept]
        index_a = {c: i for i, c in enumerate(seq_a)}
        index_b = {c: i for i, c in enumerate(seq_b)}
        for code in sorted(moved):
            old, new = baseline.fields[code], candidate.fields[code]
            deltas.append(SpecDelta(
                code=code, change_type=ORDER_CHANGED,
                old_value={"position": index_a[code],
                           "preceded_by": (seq_a[index_a[code] - 1]
                                           if index_a[code] else None)},
                new_value={"position": index_b[code],
                           "preceded_by": (seq_b[index_b[code] - 1]
                                           if index_b[code] else None)},
                severity=severity_for(ORDER_CHANGED),
                confidence=DETERMINISTIC,
                source_version_a=va, source_version_b=vb,
                evidence=_evidence(old, new),
                detail=(f"{code} moved from position {index_a[code]} to "
                        f"{index_b[code]} in the authoritative field sequence "
                        f"(positions are over codes common to both versions)")))

    deltas.sort(key=lambda d: (d.code, d.change_type))
    return deltas


def unresolved_items(baseline: NormalizedSpec,
                     candidate: NormalizedSpec) -> List[Dict[str, Any]]:
    """Everything the parser could not determine, on either side.

    These are the items a human must resolve before the comparison can be
    called complete. They are reported, never silently treated as "unchanged".
    """
    out: List[Dict[str, Any]] = []
    for label, spec in (("baseline", baseline), ("candidate", candidate)):
        for code in sorted(spec.fields):
            f = spec.fields[code]
            if f.unresolved:
                out.append({
                    "side": label,
                    "spec_version": spec.spec_version,
                    "code": code,
                    "unresolved_attributes": list(f.unresolved),
                    "reason": "attribute not deterministically derivable from "
                              "the authoritative artefacts",
                })
        for warning in spec.parse_warnings:
            out.append({
                "side": label,
                "spec_version": spec.spec_version,
                "code": warning.get("code", ""),
                "unresolved_attributes": [],
                "reason": f"{warning.get('warning')}: {warning.get('detail')}",
            })
    return out


def spec_status(deltas: Sequence[SpecDelta], baseline: NormalizedSpec,
                candidate: NormalizedSpec) -> str:
    """Collapse the diff into SPEC_CHANGED / UNCHANGED / UNDETERMINED.

    Fail-closed: an unresolved attribute anywhere in either spec means the
    comparison cannot assert "unchanged", even when no delta was produced.
    """
    if any(d.confidence == DETERMINISTIC for d in deltas):
        return SPEC_CHANGED
    if deltas:                       # only review-required deltas remain
        return SPEC_UNDETERMINED
    if any(f.unresolved for f in baseline.fields.values()):
        return SPEC_UNDETERMINED
    if any(f.unresolved for f in candidate.fields.values()):
        return SPEC_UNDETERMINED
    return SPEC_UNCHANGED
