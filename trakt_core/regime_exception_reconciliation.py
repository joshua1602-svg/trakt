#!/usr/bin/env python3
"""
regime_exception_reconciliation.py

Why a canonical validation exception does — or does not — block a regulatory
submission.

Deliberately OUTSIDE `engine/gate_*` and `config/regime/`. Those paths are under
a change freeze that six test modules enforce
(`test_no_regulatory_or_annex2_files_modified`), and rightly so: they are the
surface a regulator's output is built from. This module reads their committed
artefacts and writes its own report. It imports nothing from them, modifies
nothing in them, and cannot alter a preflight verdict or a delivered value.

The problem this solves
-----------------------
Two gates validate, against two different vocabularies, and until now neither
said so in its output:

* **Gate 3 (canonical validation)** checks values against the *canonical model's*
  enumerations (``config/system/enum_mapping.yaml``). Its verdict is about the
  governed dataset.
* **Gate 4b (delivery preflight)** checks the *projected regulatory output*
  against ``config/regime/annex2_delivery_rules.yaml``, after Gate 4 projection
  has applied the documented source-value transforms. Its verdict is about the
  submission.

A field can legitimately be BLOCKING at Gate 3 and clean at Gate 4b — because it
is not an Annex 2 source field at all, or because a documented projection
transform resolves the offending value. Both verdicts are right. But a reader
seeing "2 blocking exceptions" beside "preflight PASS, 0 issues" cannot tell a
resolved mapping from an unnoticed gap, and that ambiguity is the dangerous part:
it is exactly the shape a real regulatory miss would take.

This module removes the ambiguity by classifying every canonical exception into
one of three dispositions, with the evidence for each:

``OUT_OF_REGIME_SCOPE``
    No Annex 2 field rule sources this canonical field, so the submission is
    unaffected. The exception is still real, and still a data-quality finding.

``RESOLVED_AT_PROJECTION``
    An Annex 2 field does source it, and the delivery rules carry an explicit,
    documented transform mapping the offending value to a valid regime code.
    Cites the code, the mapping and where the mapping is defined.

``DEFAULTED_AT_DELIVERY``
    The source supplied nothing, and the delivery rules substitute an ESMA
    no-data code so the submission still passes. Nothing is broken — but the
    submission now *asserts* something the source never said, which is the
    quietest and most dangerous of the four states. It is reported precisely
    because a clean preflight would otherwise hide it.

``BLOCKS_DELIVERY``
    An Annex 2 field sources it and nothing resolves the value. This is the case
    that must reach a human before submission.

Because the delivery preflight cannot be changed to state its own scope, the
scope of BOTH gates is recorded here instead — see ``scope`` in the output.

Usage::

    python trakt_core/regime_exception_reconciliation.py \\
      --field-summary  out_validation/<portfolio>_field_summary.csv \\
      --violations     out_validation/<portfolio>_canonical_violations.csv \\
      --canonical      out/<portfolio>_canonical_typed.csv \\
      --rules          config/regime/annex2_delivery_rules.yaml \\
      --universe       config/regime/annex2_field_universe.yaml \\
      --enum-mapping   config/system/enum_mapping.yaml \\
      --output         out/<portfolio>_ESMA_Annex2_exception_reconciliation.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUT_OF_SCOPE = "OUT_OF_REGIME_SCOPE"
RESOLVED = "RESOLVED_AT_PROJECTION"
DEFAULTED = "DEFAULTED_AT_DELIVERY"
BLOCKING = "BLOCKS_DELIVERY"

#: Canonical provenance column carrying book identity, when the tape is combined.
BOOK_ID_COL = "source_portfolio_id"
BOOK_LABEL_COL = "source_portfolio_label"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _source_field_index(rules: Dict[str, Any]) -> Dict[str, List[str]]:
    """canonical source field -> [Annex 2 codes that project from it]."""
    index: Dict[str, List[str]] = {}
    for code, rule in (rules.get("field_rules") or {}).items():
        if not isinstance(rule, dict):
            continue
        source = rule.get("projected_source_field")
        if source:
            index.setdefault(str(source), []).append(str(code))
    return index


def _observed_values(violations: pd.DataFrame, field: str) -> List[str]:
    """The distinct offending values a field's violations actually reported.

    The violation message is the only place the raw value survives, so it is
    parsed rather than re-derived — re-reading the canonical would risk
    reporting a value the validator did not object to.
    """
    rows = violations[violations["field"].astype(str) == field]
    values: List[str] = []
    for message in rows["message"].astype(str):
        if "'" in message:
            candidate = message.split("'")[1]
            if candidate and candidate not in values:
                values.append(candidate)
    return values


def _books_for_rows(canonical: Optional[pd.DataFrame],
                    row_indices: List[int]) -> List[Dict[str, Any]]:
    """Per-book exposure counts for a set of canonical row indices.

    "Which book, and how many exposures" is the question a lender actually asks
    about an exception, and on a combined tape it is only answerable because
    Gate 2 stamped book identity onto every row.

    Returns an empty list when the tape carries no book identity, so a caller
    can tell "single book" from "book attribution unavailable".
    """
    if canonical is None or BOOK_ID_COL not in canonical.columns:
        return []
    indices = [i for i in row_indices if 0 <= i < len(canonical)]
    if not indices:
        return []

    subset = canonical.iloc[indices]
    labels = {}
    if BOOK_LABEL_COL in canonical.columns:
        labels = (canonical[[BOOK_ID_COL, BOOK_LABEL_COL]]
                  .dropna().drop_duplicates()
                  .set_index(BOOK_ID_COL)[BOOK_LABEL_COL].to_dict())

    counts = Counter(subset[BOOK_ID_COL].astype(str))
    return [
        {"bookId": book, "bookLabel": labels.get(book, book), "exposures": int(count)}
        for book, count in sorted(counts.items())
    ]


def _row_indices(frame: Optional[pd.DataFrame], column: str,
                 match_column: str, value: str) -> List[int]:
    """Row indices from a violations/issues frame for one field or rule."""
    if frame is None or frame.empty or column not in frame.columns:
        return []
    if match_column not in frame.columns:
        return []
    rows = frame[frame[match_column].astype(str) == value]
    if rows.empty:
        return []
    return pd.to_numeric(rows[column], errors="coerce").dropna().astype(int).tolist()


def _allowed_canonical_values(enum_mapping: Dict[str, Any], field: str) -> List[str]:
    """The values the CANONICAL model accepts for a field.

    The mapping file is nested by regime (``{ESMA_Annex2: {field: {...}}}``), so
    a flat lookup silently returns nothing — which reads in the artefact as "no
    constraint" rather than "not looked up". Both shapes are handled.
    """
    tables = []
    for value in enum_mapping.values():
        if isinstance(value, dict) and field in value:
            tables.append(value[field])
    if field in enum_mapping:
        tables.append(enum_mapping[field])

    allowed: set = set()
    for table in tables:
        if isinstance(table, dict):
            allowed |= {str(v) for v in table.values()} | {str(k) for k in table}
        elif isinstance(table, list):
            allowed |= {str(v) for v in table}
    return sorted(allowed)


def _delivery_only_exceptions(delivery_issues: Optional[pd.DataFrame],
                              seen_codes: set, rules: Dict[str, Any],
                              universe: Dict[str, Any],
                              canonical: Optional[pd.DataFrame] = None,
                              ) -> List[Dict[str, Any]]:
    """Gate 4b errors that no Gate 3 exception accounts for.

    The reconciliation has to run in both directions. A field can be clean
    against the canonical enumerations and still fail the submission — a
    mandatory Annex value that is simply absent is the common case, because
    Gate 3 validates the values that ARE there. Reporting only the Gate 3 side
    would reproduce the exact failure this module exists to prevent, in mirror
    image: a surface showing no exceptions while the submission is blocked.
    """
    if delivery_issues is None or delivery_issues.empty:
        return []

    field_rules = rules.get("field_rules") or {}
    universe_fields = universe.get("fields") or {}
    errors = delivery_issues[delivery_issues.get("severity").astype(str) == "error"] \
        if "severity" in delivery_issues.columns else delivery_issues

    out: List[Dict[str, Any]] = []
    for code, group in errors.groupby(errors["field"].astype(str)):
        if code in seen_codes:
            continue
        rule = field_rules.get(code) or {}
        meta = universe_fields.get(code) or {}
        issue_types = sorted({str(t) for t in group.get("issue_type", pd.Series(dtype=str))})
        messages = sorted({str(m) for m in group.get("message", pd.Series(dtype=str))})[:3]
        source = rule.get("projected_source_field")
        observed = sorted({str(v) for v in group.get("input_value", pd.Series(dtype=str))
                           if str(v).strip()})[:6]
        indices = pd.to_numeric(group.get("row_index", pd.Series(dtype=str)),
                                errors="coerce").dropna().astype(int).tolist()
        out.append({
            "canonicalField": source,
            "issueType": ", ".join(issue_types) or "delivery",
            "canonicalSeverity": "",
            "canonicalMateriality": "",
            "affectedExposures": int(len(group)),
            "observedValues": observed,
            "allowedCanonicalValues": [],
            "allowedRegimeValues": sorted(
                {str(v) for v in (rule.get("transform") or {}).get("enum_map", {}).values()}),
            "byBook": _books_for_rows(canonical, indices),
            "annexCodes": [code],
            "annex": [{
                "code": code,
                "fieldName": meta.get("field_name"),
                "section": meta.get("section"),
                "requirement": (meta.get("content") or "").strip() or None,
                "format": meta.get("format"),
                "mandatory": bool(rule.get("mandatory")),
                "enforcePresence": bool(rule.get("enforce_presence")),
                "valueMapping": {},
            }],
            "disposition": BLOCKING,
            "dispositionReason": (
                f"{code} failed the Annex 2 delivery preflight ({'; '.join(messages)}). "
                "No canonical exception accounts for it: the value is absent, or valid "
                "against the canonical enumeration but outside the regime's own code "
                "list. Either way it is caught only at the regime layer."
            ),
            "resolution": (
                f"Supply a valid {code} value at source"
                + (f" (canonical field '{source}')" if source else "")
                + ". Do not submit until the field is populated and conforms."
            ),
        })
    return out


def _nd_defaulted_exceptions(projected: Optional[pd.DataFrame],
                             canonical: Optional[pd.DataFrame],
                             rules: Dict[str, Any],
                             universe: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fields the submission will fill with a no-data code because nothing was supplied.

    These never appear as an exception anywhere else: the canonical validator
    sees an empty cell and moves on, and the preflight passes because the rule
    told it to substitute. The result is a submission that states "not
    applicable" about data the client simply did not send. Surfacing it is the
    difference between a clean preflight and a true one.
    """
    if projected is None or projected.empty:
        return []

    field_rules = rules.get("field_rules") or {}
    universe_fields = universe.get("fields") or {}
    out: List[Dict[str, Any]] = []

    for code, rule in field_rules.items():
        if not isinstance(rule, dict) or not rule.get("default_allowed"):
            continue
        default_value = str(rule.get("default_value") or "")
        if not default_value.upper().startswith("ND"):
            continue
        if code not in projected.columns:
            continue

        blank = projected[code].astype(str).str.strip() == ""
        indices = [int(i) for i in projected.index[blank].tolist()]

        # Blank on EVERY row means the product genuinely does not have the
        # field, which is a documented ND decision in the client config, not a
        # data gap. Blank on SOME rows means the source does carry it and did
        # not supply it here — that is the finding.
        if not indices or len(indices) == len(projected):
            continue

        meta = universe_fields.get(code) or {}
        source = rule.get("projected_source_field")
        out.append({
            "canonicalField": source,
            "issueType": "nd_defaulted",
            "canonicalSeverity": "",
            "canonicalMateriality": "",
            "affectedExposures": len(indices),
            "observedValues": [],
            "allowedCanonicalValues": [],
            "allowedRegimeValues": [],
            "byBook": _books_for_rows(canonical, indices),
            "annexCodes": [code],
            "annex": [{
                "code": code,
                "fieldName": meta.get("field_name"),
                "section": meta.get("section"),
                "requirement": (meta.get("content") or "").strip() or None,
                "format": meta.get("format"),
                "mandatory": bool(rule.get("mandatory")),
                "enforcePresence": bool(rule.get("enforce_presence")),
                "valueMapping": {},
            }],
            "disposition": DEFAULTED,
            "dispositionReason": (
                f"{code} ({meta.get('field_name') or source}) is mandatory and the "
                f"source supplied no value, so the delivery rules substitute "
                f"{default_value}. The preflight therefore passes — but the "
                f"submission asserts {default_value} about these exposures, which "
                "is a statement the source data never made."
            ),
            "resolution": (
                f"Supply '{source}' at source for the affected exposures, or "
                f"confirm that {default_value} is the correct assertion for them "
                "and record that decision. Do not let a default stand in for a "
                "value nobody checked."
            ),
        })
    return out


def reconcile(field_summary: pd.DataFrame, violations: pd.DataFrame,
              canonical: Optional[pd.DataFrame], rules: Dict[str, Any],
              universe: Dict[str, Any], enum_mapping: Dict[str, Any],
              delivery_issues: Optional[pd.DataFrame] = None,
              business_rules: Optional[pd.DataFrame] = None,
              projected: Optional[pd.DataFrame] = None,
              preflight: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classify every exception, from either gate, by its effect on the submission."""
    source_index = _source_field_index(rules)
    field_rules = rules.get("field_rules") or {}
    universe_fields = universe.get("fields") or {}

    exceptions: List[Dict[str, Any]] = []

    for _, row in field_summary.iterrows():
        field = str(row.get("canonical_field") or row.get("field_name") or "").strip()
        if not field:
            continue

        issue_type = str(row.get("issue_type") or "")

        # A business-rule finding is a cross-field contradiction, so the
        # aggregator files it against the portfolio rather than a column. Treat
        # it as the rule it is: no Annex source field, and the rule's own
        # description is the requirement.
        if field.upper() == "PORTFOLIO":
            rule_rows = business_rules[
                business_rules["rule_id"].astype(str) == issue_type
            ] if business_rules is not None and "rule_id" in business_rules.columns \
                else None
            description = ""
            messages: List[str] = []
            if rule_rows is not None and not rule_rows.empty:
                description = str(rule_rows.iloc[0].get("description") or "")
                messages = [str(m) for m in rule_rows.get(
                    "message", pd.Series(dtype=str)).head(3)]
            books = _books_for_rows(canonical, _row_indices(
                business_rules, "row_index", "rule_id", issue_type))
            exceptions.append({
                "canonicalField": None,
                "businessRule": issue_type,
                "issueType": issue_type,
                "canonicalSeverity": str(row.get("severity") or ""),
                "canonicalMateriality": str(row.get("materiality") or ""),
                "affectedExposures": int(row.get("affected_rows") or 0),
                "observedValues": messages,
                "allowedCanonicalValues": [],
                "allowedRegimeValues": [],
                "byBook": books,
                "annexCodes": [],
                "disposition": OUT_OF_SCOPE,
                "dispositionReason": (
                    f"Business rule {issue_type} — {description} This is a "
                    "contradiction between fields, not an invalid value, so no "
                    "single Annex 2 field rule owns it and the submission is not "
                    "blocked by it. It is still a governed finding on the dataset."
                ),
                "resolution": (
                    "Correct the inconsistent dates at source. No Annex 2 action "
                    "is required, but the canonical exception stays open until "
                    "the source is fixed."
                ),
            })
            continue

        observed = _observed_values(violations, field)
        codes = source_index.get(field, [])

        entry: Dict[str, Any] = {
            "canonicalField": field,
            "issueType": issue_type,
            "canonicalSeverity": str(row.get("severity") or ""),
            "canonicalMateriality": str(row.get("materiality") or ""),
            "affectedExposures": int(row.get("affected_rows") or 0),
            "observedValues": observed,
            "allowedCanonicalValues": _allowed_canonical_values(enum_mapping, field),
            "allowedRegimeValues": [],
            "byBook": _books_for_rows(canonical, _row_indices(
                violations, "row", "field", field)),
            "annexCodes": codes,
        }

        if not codes:
            entry["disposition"] = OUT_OF_SCOPE
            entry["dispositionReason"] = (
                f"No ESMA Annex 2 field rule projects from '{field}', so the exception "
                "does not reach the submission. It remains a governed data-quality "
                "finding on the canonical model."
            )
            entry["resolution"] = (
                f"Correct '{field}' at source, or extend the canonical enumeration if "
                "the value is legitimate. No Annex 2 action is required."
            )
            exceptions.append(entry)
            continue

        # An Annex 2 field does source it. Does a documented transform resolve
        # every offending value?
        annex_detail = []
        unresolved: List[str] = []
        for code in codes:
            rule = field_rules.get(code) or {}
            enum_map = (rule.get("transform") or {}).get("enum_map") or {}
            meta = universe_fields.get(code) or {}
            mapped = {v: enum_map.get(v) or enum_map.get(str(v).upper()) for v in observed}
            annex_detail.append({
                "code": code,
                "fieldName": meta.get("field_name"),
                "section": meta.get("section"),
                "requirement": (meta.get("content") or "").strip() or None,
                "format": meta.get("format"),
                "mandatory": bool(rule.get("mandatory")),
                "enforcePresence": bool(rule.get("enforce_presence")),
                "valueMapping": mapped,
            })
            unresolved.extend(v for v, m in mapped.items() if not m)

        entry["annex"] = annex_detail

        if observed and not unresolved:
            entry["disposition"] = RESOLVED
            pairs = ", ".join(
                f"'{v}' → {d['valueMapping'][v]}"
                for d in annex_detail for v in observed if d["valueMapping"].get(v)
            )
            entry["dispositionReason"] = (
                f"{', '.join(codes)} projects from '{field}', and the delivery rules "
                f"map the reported value at projection ({pairs}). The submitted value "
                "is a valid regime code, which is why the Annex 2 preflight passes "
                "while the canonical exception stands."
            )
            entry["resolution"] = (
                f"No submission action required. To clear the canonical exception, "
                f"correct '{field}' at source so the governed model carries a "
                "categorised value rather than relying on the projection synonym."
            )
        else:
            entry["disposition"] = BLOCKING
            entry["dispositionReason"] = (
                f"{', '.join(codes)} projects from '{field}' and no delivery-rule "
                f"transform maps {unresolved or observed}. The submission would carry "
                "an invalid or absent regime value."
            )
            entry["resolution"] = (
                f"Correct '{field}' at source, or add an explicit mapping for "
                f"{unresolved or observed} to the {', '.join(codes)} enum_map in "
                "config/regime/annex2_delivery_rules.yaml. Do not submit until one "
                "of the two is done."
            )

        exceptions.append(entry)

    # Second direction: preflight errors no canonical exception accounts for.
    seen_codes = {c for e in exceptions for c in e.get("annexCodes", [])}
    exceptions.extend(
        _delivery_only_exceptions(delivery_issues, seen_codes, rules, universe,
                                  canonical))
    seen_codes |= {c for e in exceptions for c in e.get("annexCodes", [])}
    exceptions.extend(
        e for e in _nd_defaulted_exceptions(projected, canonical, rules, universe)
        if not set(e["annexCodes"]) & seen_codes)

    # Blocking first — the only ones that stop a submission.
    order = {BLOCKING: 0, DEFAULTED: 1, RESOLVED: 2, OUT_OF_SCOPE: 3}
    exceptions.sort(key=lambda e: (order.get(e["disposition"], 3),
                                   -int(e.get("affectedExposures") or 0)))

    counts = Counter(e["disposition"] for e in exceptions)
    return {
        "preflight": preflight or {},
        "scope": {
            "canonicalValidation": (
                "Gate 3. Checks the governed canonical model against "
                "config/system/enum_mapping.yaml and the field registry. Its verdict "
                "is about the dataset, not the submission."
            ),
            "annexPreflight": (
                "Gate 4b. Checks the projected Annex 2 output against the "
                f"{len(field_rules)} field rules in annex2_delivery_rules.yaml, AFTER "
                "Gate 4 has applied the documented projection transforms. Its verdict "
                "is about the submission, not the dataset. A PASS there is NOT 'the "
                "dataset has no exceptions'."
            ),
            "note": (
                "The two verdicts can legitimately differ. Every canonical exception "
                "below is classified by its actual effect on the Annex 2 submission, "
                "so a clean preflight beside an open exception is explained rather "
                "than merely asserted."
            ),
        },
        "totals": {
            "canonicalExceptions": len(exceptions),
            "outOfRegimeScope": counts.get(OUT_OF_SCOPE, 0),
            "resolvedAtProjection": counts.get(RESOLVED, 0),
            "defaultedAtDelivery": counts.get(DEFAULTED, 0),
            "blocksDelivery": counts.get(BLOCKING, 0),
        },
        "submissionBlocked": counts.get(BLOCKING, 0) > 0,
        "exceptions": exceptions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Annex 2 exception reconciliation (Gate 4b)")
    ap.add_argument("--field-summary", required=True)
    ap.add_argument("--violations", required=True)
    ap.add_argument("--canonical", required=False)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--universe", required=True)
    ap.add_argument("--enum-mapping", required=True)
    ap.add_argument("--business-rules", required=False,
                    help="Gate 3b *_business_rules_violations.csv, so cross-field "
                         "rule findings carry their rule text and book attribution")
    ap.add_argument("--delivery-issues", required=False,
                    help="Gate 4b *_ESMA_Annex2_delivery_issues.csv, so preflight "
                         "failures with no canonical exception are reported too")
    ap.add_argument("--delivery-report", required=False,
                    help="Gate 4b *_delivery_report.json, so the preflight verdict "
                         "travels with the reconciliation and no downstream reader "
                         "needs the Annex-named artefact")
    ap.add_argument("--projected", required=False,
                    help="Gate 4 *_ESMA_Annex2_projected.csv, so mandatory values "
                         "the delivery rules fill with an ND code are surfaced")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    def _optional(path: Optional[str], require_column: Optional[str] = None):
        if not path or not Path(path).exists():
            return None
        frame = pd.read_csv(path).fillna("")
        if frame.empty:
            return None
        if require_column and require_column not in frame.columns:
            return None
        return frame

    field_summary = pd.read_csv(args.field_summary).fillna("")
    violations = pd.read_csv(args.violations).fillna("")
    canonical = pd.read_csv(args.canonical, dtype=str) if args.canonical else None

    result = reconcile(
        field_summary=field_summary,
        violations=violations,
        canonical=canonical,
        rules=_load_yaml(Path(args.rules)),
        universe=_load_yaml(Path(args.universe)),
        enum_mapping=_load_yaml(Path(args.enum_mapping)),
        delivery_issues=_optional(args.delivery_issues, "field"),
        business_rules=_optional(args.business_rules, "rule_id"),
        projected=(pd.read_csv(args.projected, dtype=str).fillna("")
                   if args.projected and Path(args.projected).exists() else None),
        preflight=(json.loads(Path(args.delivery_report).read_text(encoding="utf-8"))
                   .get("preflight")
                   if args.delivery_report and Path(args.delivery_report).exists()
                   else None),
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    totals = result["totals"]
    logging.info(
        "[Gate 4b] Exception reconciliation......... %d exception(s): "
        "%d out of scope, %d resolved at projection, %d ND-defaulted, %d blocking",
        totals["canonicalExceptions"], totals["outOfRegimeScope"],
        totals["resolvedAtProjection"], totals["defaultedAtDelivery"],
        totals["blocksDelivery"],
    )
    logging.info("[Gate 4b] Reconciliation report............ %s", output.name)


if __name__ == "__main__":
    main()
