"""regulatory_watch.discovery.triage — Annex 2 relevance, in two stages.

**Stage A — deterministic filter.** Explicit keyword/metadata rules, applied
first. Deliberately broad: a false positive costs a reviewer thirty seconds, a
false negative means Trakt never hears about a change to its only live regime.
So the filter passes anything touching securitisation reporting, and a
rejection always records which rule rejected it and why.

**Stage B — bounded semantic classification.** For items that survive Stage A,
a classifier assigns a publication type and development status. The default
implementation is deterministic (signal-based, no model). An LLM classifier is
supported through the same interface and is *bounded by construction*:

* it sees only the title, source metadata and extracted official text;
* it must answer a strict schema, validated against the controlled vocabulary;
* anything it returns outside the vocabulary, or with low confidence, becomes
  ``REVIEW_REQUIRED`` rather than being trusted;
* it cannot reach Stage 1A data, and there is no field in its schema that
  could express a field-level change.

Neither stage may conclude that a regime changed. The strongest thing Stage 1B
can say about a technical package is ``TECHNICAL_SPEC_CHANGE``, which means
"Stage 1A should now run", not "these fields changed".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .contracts import (
    CLASSIFIER_DETERMINISTIC,
    CLASSIFIER_SEMANTIC,
    CLASSIFIER_SEMANTIC_FAILED,
    CONSULTATION,
    Classification,
    DEVELOPMENT_STATUSES,
    FINAL_REPORT,
    FUTURE_CHANGE_EXPECTED,
    HIGH,
    INFORMATION_ONLY,
    INTERPRETATION_REVIEW_REQUIRED,
    LOW,
    MEDIUM,
    NOT_RELEVANT,
    OTHER_REGULATORY_PUBLICATION,
    POTENTIAL_FUTURE_CHANGE,
    PUBLICATION_TYPES,
    Q_AND_A,
    RELEVANCE_NOT_RELEVANT,
    RELEVANCE_POTENTIAL,
    RELEVANCE_RELEVANT,
    RELEVANCE_UNKNOWN,
    RELEVANCE_VALUES,
    REVIEW_REQUIRED,
    RTS_ITS,
    RawPublication,
    TECHNICAL_INSTRUCTION,
    TECHNICAL_SPEC_CHANGE,
    TEMPLATE_SCHEMA_UPDATE,
    TriageResult,
    VALIDATION_RULE_UPDATE,
    requires_review,
)

# --------------------------------------------------------------------------- #
# Stage A — the deterministic relevance filter
# --------------------------------------------------------------------------- #
#
# Grouped so a match can say WHICH concept fired, not just "matched". A
# publication needs only ONE group to survive: broad by design.

RELEVANCE_TERMS: Dict[str, Tuple[str, ...]] = {
    "securitisation": (
        "securitisation", "securitization", "securitised", "securitized",
        "abcp", "non-abcp",
    ),
    "disclosure": (
        "disclosure requirement", "disclosure template", "disclosure rts",
        "disclosure", "article 7", "art. 7", "transparency requirement",
    ),
    "underlying_exposures": (
        "underlying exposure", "underlying exposures", "loan-level",
        "loan level data",
    ),
    "annex2": (
        "annex 2", "annex ii", "annex2",
    ),
    "residential_real_estate": (
        "residential real estate", "residential mortgage", " rre ", "rre ",
    ),
    # NOTE: the bare acronym "its" is deliberately absent — it matches the
    # English possessive ("ESMA publishes its newsletter") and floods triage.
    # The acronym is only recognised in an unambiguous pairing.
    "technical_standards": (
        "technical standard", "regulatory technical standard",
        "implementing technical standard", "rts/its", "rts and its",
        "draft rts", " rts ",
    ),
    "reporting_instructions": (
        "reporting instruction", "technical reporting instruction",
        "reporting manual",
    ),
    "templates_schema": (
        "reporting template", "disclosure template", "xml schema", "xsd",
        "schema and instructions", "taxonomy",
    ),
    "validation_rules": (
        "validation rule", "validation rules", "data quality check",
    ),
    "q_and_a": (
        "q&a", "q & a", "questions and answers", "qa document",
    ),
    "no_data": (
        "no-data", "no data option", "nd1", "nd2", "nd3", "nd4", "nd5",
        "data completeness",
    ),
    "esma_securitisation_reporting": (
        "securitisation reporting", "securitisation repository",
        "securitisation register",
    ),
}

#: Groups that on their own only weakly suggest Annex 2. A publication matching
#: ONLY these is passed through (never rejected) but classified with lower
#: confidence — "securitisation" alone covers a great deal of ESMA output.
_WEAK_GROUPS = frozenset({"securitisation", "esma_securitisation_reporting"})


def _haystack(publication: RawPublication) -> str:
    parts = [publication.title, publication.summary,
             " ".join(publication.categories), publication.url]
    # Padded so space-delimited terms (" rre ", " rts ") can match at the edges.
    return " " + " ".join(p for p in parts if p).lower() + " "


def deterministic_triage(publication: RawPublication) -> TriageResult:
    """Stage A. Broad by design; a rejection always records why."""
    text = _haystack(publication)
    matched_terms: List[str] = []
    matched_groups: List[str] = []
    for group, terms in RELEVANCE_TERMS.items():
        hits = [t for t in terms if t in text]
        if hits:
            matched_groups.append(group)
            matched_terms.extend(hits)

    if not matched_groups:
        return TriageResult(
            passed=False, matched_terms=[], matched_groups=[],
            reason=("no ESMA Annex 2 relevance term matched the title, "
                    "summary, categories or URL"))
    return TriageResult(
        passed=True,
        matched_terms=sorted(set(matched_terms)),
        matched_groups=sorted(matched_groups),
        reason=f"matched relevance group(s): {', '.join(sorted(matched_groups))}")


# --------------------------------------------------------------------------- #
# Stage B — classification signals
# --------------------------------------------------------------------------- #

#: DOCUMENT FORM signals, checked FIRST and in this order. What a publication
#: *is* beats what it is *about*: "Consultation Paper on amendments to the
#: disclosure templates" is a CONSULTATION about templates, not a template
#: update, and "Final Report on draft RTS" is a FINAL_REPORT, not an RTS_ITS.
#: Getting this the wrong way round marks every discussion of a template as a
#: technical change and destroys the distinction Stage 1B exists to preserve.
_FORM_SIGNALS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (Q_AND_A, ("q&a", "q & a", "questions and answers")),
    (CONSULTATION, ("consultation paper", "consultation on", "call for evidence",
                    "discussion paper", "public consultation")),
    (FINAL_REPORT, ("final report",)),
    (TECHNICAL_INSTRUCTION, ("reporting instruction", "technical instruction",
                             "reporting manual")),
    (RTS_ITS, ("regulatory technical standard",
               "implementing technical standard", "draft rts", "rts/its",
               "rts and its")),
)

#: SUBJECT-MATTER signals, checked only when no document form matched. These
#: are the artefact packages themselves.
_ARTEFACT_SIGNALS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (VALIDATION_RULE_UPDATE, ("validation rule", "validation rules",
                              "data quality check")),
    (TEMPLATE_SCHEMA_UPDATE, ("xml schema", "xsd", "schema and instructions",
                              "reporting template", "disclosure template",
                              "taxonomy package")),
)

_TYPE_SIGNALS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    _FORM_SIGNALS + _ARTEFACT_SIGNALS)

#: A publication CARRIES an artefact when it ships the package itself, as
#: opposed to discussing one. The distinction decides whether the document form
#: or the subject matter wins, so it has to be tight:
#:
#:   * "Q&A ... on disclosure templates"        — about templates, carries none
#:   * "Consultation on amendments to the
#:      disclosure templates"                   — about templates, carries none
#:   * "... XML schema and instructions v1.3.1" — IS the package
#:
#: So the test is on artefact-FORMAT terms in the TITLE (a schema, a package),
#: never on subject terms like "disclosure template" which any commentary uses,
#: plus any linked artefact file.
_ARTEFACT_CARRIER_TITLE_TERMS = (
    "xml schema", "xsd", "schema and instructions", "taxonomy package",
    "validation rules package", "reporting templates package",
)
_ARTEFACT_FILE_SUFFIXES = (".xsd", ".xlsx", ".xls", ".zip")

_FUTURE_TERMS = ("will apply", "will be amended", "will replace", "shall apply",
                 "enters into force", "entry into force", "from 1 ",
                 "applicable from", "as of", "transition period",
                 "amendments to")

_DATE_LABELS = (
    ("effective_date", (r"effective (?:date|from)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})",
                        r"enters? into force on\s*(\d{4}-\d{2}-\d{2})")),
    ("implementation_date", (r"applicable from\s*(\d{4}-\d{2}-\d{2})",
                             r"implementation date\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})")),
    ("consultation_deadline", (r"(?:responses?|comments?|consultation)\s+"
                               r"(?:close|closes|deadline|by)\s*[:\-]?\s*"
                               r"(\d{4}-\d{2}-\d{2})",)),
)


def extract_dates(publication: RawPublication) -> Dict[str, Optional[str]]:
    """Dates the publication *states*, in an unambiguous ISO form.

    Nothing is inferred. A date written in prose ("early next year", "Q3") is
    left unknown rather than resolved to a guess, because a wrong effective
    date is worse than an absent one.
    """
    text = f"{publication.title} {publication.summary}".lower()
    out: Dict[str, Optional[str]] = {"effective_date": None,
                                     "implementation_date": None,
                                     "consultation_deadline": None}
    for label, patterns in _DATE_LABELS:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                out[label] = match.group(1)
                break
    return out


def carries_artefact(publication: RawPublication) -> bool:
    """Whether the publication ships an Annex 2 technical artefact itself."""
    title = f" {publication.title.lower()} "
    if any(term in title for term in _ARTEFACT_CARRIER_TITLE_TERMS):
        return True
    return any(url.lower().endswith(_ARTEFACT_FILE_SUFFIXES)
               for url in publication.linked_urls)


def _classify_type(text: str, carrying: bool) -> Tuple[str, List[str]]:
    """Publication type. Artefact-carrying packages are typed by the artefact.

    A package that ships the schema is a technical update whatever prose it
    also contains — the auth.099 package bundles the XSD, the templates AND the
    reporting instructions, and typing it as an instructions document would
    hide the one publication that most needs Stage 1A to run.
    """
    ordered = ((_ARTEFACT_SIGNALS + _FORM_SIGNALS) if carrying
               else _TYPE_SIGNALS)
    for publication_type, terms in ordered:
        hits = [t for t in terms if t in text]
        if hits:
            return publication_type, hits
    return OTHER_REGULATORY_PUBLICATION, []


def _classify_status(publication_type: str, text: str, triage: TriageResult,
                     carries_artefact: bool) -> Tuple[str, str]:
    """Map (type, signals) onto a development status plus a reason."""
    if publication_type == TEMPLATE_SCHEMA_UPDATE and carries_artefact:
        return (TECHNICAL_SPEC_CHANGE,
                "publishes an Annex 2 technical artefact (template/schema); "
                "Stage 1A must run to determine the exact field deltas")
    if publication_type == VALIDATION_RULE_UPDATE and carries_artefact:
        return (TECHNICAL_SPEC_CHANGE,
                "publishes updated validation rules; Stage 1A must run to "
                "determine the exact rule deltas")
    if publication_type == Q_AND_A:
        return (INTERPRETATION_REVIEW_REQUIRED,
                "a Q&A can change how a field must be populated without "
                "changing the machine-readable specification")
    if publication_type == CONSULTATION:
        return (POTENTIAL_FUTURE_CHANGE,
                "a consultation proposes change; nothing is in force yet")
    if publication_type == FINAL_REPORT:
        return (FUTURE_CHANGE_EXPECTED,
                "a final report settles ESMA's position; technical artefacts "
                "typically follow")
    if publication_type == RTS_ITS:
        if any(term in text for term in _FUTURE_TERMS):
            return (FUTURE_CHANGE_EXPECTED,
                    "technical standards with a stated application date")
        return (POTENTIAL_FUTURE_CHANGE,
                "technical standards published; application timing not stated "
                "in the publication metadata")
    if publication_type == TECHNICAL_INSTRUCTION:
        return (INTERPRETATION_REVIEW_REQUIRED,
                "reporting instructions govern how fields must be populated")
    if publication_type == TEMPLATE_SCHEMA_UPDATE:
        return (FUTURE_CHANGE_EXPECTED,
                "discusses reporting templates or schema without publishing "
                "the artefact itself")
    # Nothing specific matched. Only weak relevance groups fired.
    if set(triage.matched_groups) <= _WEAK_GROUPS:
        return (INFORMATION_ONLY,
                "matched only broad securitisation terms with no reporting, "
                "disclosure or technical-standards signal")
    return (REVIEW_REQUIRED,
            "matched Annex 2 relevance terms but no publication-type signal "
            "was recognised")


def _relevance(triage: TriageResult, status: str) -> Tuple[str, str]:
    strong = [g for g in triage.matched_groups if g not in _WEAK_GROUPS]
    if status == TECHNICAL_SPEC_CHANGE:
        return RELEVANCE_RELEVANT, HIGH
    if status == INFORMATION_ONLY:
        return RELEVANCE_NOT_RELEVANT, MEDIUM
    if status == REVIEW_REQUIRED:
        return RELEVANCE_UNKNOWN, LOW
    if len(strong) >= 2:
        return RELEVANCE_RELEVANT, HIGH
    if strong:
        return RELEVANCE_POTENTIAL, MEDIUM
    return RELEVANCE_POTENTIAL, LOW


def classify_deterministic(publication: RawPublication,
                           triage: TriageResult) -> Classification:
    """The default Stage B classifier. No model, fully reproducible."""
    text = _haystack(publication)
    carrying = carries_artefact(publication)
    publication_type, type_signals = _classify_type(text, carrying)
    status, reason = _classify_status(publication_type, text, triage, carrying)
    relevance, confidence = _relevance(triage, status)
    return Classification(
        publication_type=publication_type,
        development_status=status,
        annex2_relevance=relevance,
        relevance_reason=reason,
        confidence=confidence,
        classifier=CLASSIFIER_DETERMINISTIC,
        review_required=requires_review(status, confidence, relevance),
        matched_signals=sorted(set(type_signals) | set(triage.matched_groups)),
    )


# --------------------------------------------------------------------------- #
# The semantic classifier interface
# --------------------------------------------------------------------------- #

#: A semantic classifier takes the bounded prompt payload and returns raw text
#: (or a dict). It never receives Stage 1A data and never receives the full
#: publication body beyond the extracted official text handed to it.
SemanticBackend = Callable[[Dict[str, Any]], Any]


@dataclass
class SemanticClassifier:
    """Optional Stage B refinement over the deterministic classification.

    Disabled by default. When a backend is supplied, the deterministic result
    is computed FIRST and is what survives if the backend fails, returns
    unparseable output, or answers outside the controlled vocabulary. The
    backend can therefore only ever sharpen a classification or escalate it to
    review — it cannot silently downgrade one.
    """

    backend: Optional[SemanticBackend] = None
    #: The classifier may never mark something less urgent than the
    #: deterministic pass did. Escalation only.
    escalate_only: bool = True
    max_text_chars: int = 4000

    @property
    def enabled(self) -> bool:
        return self.backend is not None

    def payload(self, publication: RawPublication, triage: TriageResult,
                official_text: str = "") -> Dict[str, Any]:
        """Exactly what the backend is allowed to see. Nothing more."""
        return {
            "task": ("Classify an ESMA publication's relevance to the ESMA "
                     "Annex 2 (non-ABCP underlying exposures) disclosure "
                     "regime. Do not infer field-level or configuration "
                     "changes; that is out of scope."),
            "title": publication.title,
            "source_id": publication.source_id,
            "url": publication.url,
            "publication_date": publication.publication_date,
            "categories": list(publication.categories),
            "official_text": (official_text or publication.summary
                              )[:self.max_text_chars],
            "matched_relevance_groups": list(triage.matched_groups),
            "allowed_publication_type": list(PUBLICATION_TYPES),
            "allowed_development_status": list(DEVELOPMENT_STATUSES),
            "allowed_annex2_relevance": list(RELEVANCE_VALUES),
            "response_schema": {
                "publication_type": "one of allowed_publication_type",
                "development_status": "one of allowed_development_status",
                "annex2_relevance": "one of allowed_annex2_relevance",
                "relevance_reason": "one sentence",
                "confidence": "HIGH | MEDIUM | LOW",
            },
        }

    def classify(self, publication: RawPublication, triage: TriageResult,
                 official_text: str = "") -> Classification:
        baseline = classify_deterministic(publication, triage)
        if not self.enabled:
            return baseline

        try:
            raw = self.backend(self.payload(publication, triage, official_text))
        except Exception as exc:          # noqa: BLE001 - fail to review
            return self._failed(baseline, f"{type(exc).__name__}: {exc}")

        parsed, status, error = _extract_json(raw)
        if status not in ("ok", "ok_extracted") or not isinstance(parsed, dict):
            return self._failed(baseline, f"unparseable response: {error}")

        publication_type = str(parsed.get("publication_type") or "")
        development_status = str(parsed.get("development_status") or "")
        relevance = str(parsed.get("annex2_relevance") or "")
        confidence = str(parsed.get("confidence") or "").upper()
        reason = str(parsed.get("relevance_reason") or "").strip()

        if (publication_type not in PUBLICATION_TYPES
                or development_status not in DEVELOPMENT_STATUSES
                or relevance not in RELEVANCE_VALUES
                or confidence not in (HIGH, MEDIUM, LOW)):
            return self._failed(
                baseline, "response fell outside the controlled vocabulary")
        if confidence == LOW:
            return self._failed(baseline, "classifier reported low confidence")

        if self.escalate_only and not _more_urgent(development_status,
                                                   baseline.development_status):
            development_status = baseline.development_status
            reason = (reason or baseline.relevance_reason) + \
                " (deterministic status retained: the semantic classifier " \
                "may escalate but never downgrade)"

        return Classification(
            publication_type=publication_type,
            development_status=development_status,
            annex2_relevance=relevance,
            relevance_reason=reason or baseline.relevance_reason,
            confidence=confidence,
            classifier=CLASSIFIER_SEMANTIC,
            review_required=requires_review(development_status, confidence,
                                            relevance),
            matched_signals=baseline.matched_signals,
        )

    @staticmethod
    def _failed(baseline: Classification, detail: str) -> Classification:
        """A failed semantic pass escalates to review; it never discards."""
        return Classification(
            publication_type=baseline.publication_type,
            development_status=(baseline.development_status
                                if baseline.development_status
                                in (TECHNICAL_SPEC_CHANGE, NOT_RELEVANT)
                                else REVIEW_REQUIRED),
            annex2_relevance=baseline.annex2_relevance,
            relevance_reason=(f"{baseline.relevance_reason}; semantic "
                              f"classification unavailable ({detail})"),
            confidence=LOW,
            classifier=CLASSIFIER_SEMANTIC_FAILED,
            review_required=True,
            matched_signals=baseline.matched_signals,
        )


#: Urgency order used by ``escalate_only``. Higher means "needs a human sooner".
_URGENCY = {
    NOT_RELEVANT: 0,
    INFORMATION_ONLY: 1,
    POTENTIAL_FUTURE_CHANGE: 2,
    FUTURE_CHANGE_EXPECTED: 3,
    INTERPRETATION_REVIEW_REQUIRED: 4,
    REVIEW_REQUIRED: 5,
    TECHNICAL_SPEC_CHANGE: 6,
}


def _more_urgent(candidate: str, baseline: str) -> bool:
    return _URGENCY.get(candidate, 5) > _URGENCY.get(baseline, 5)


def _extract_json(raw: Any):
    """Reuse the repo's hardened LLM JSON extractor when it is importable."""
    try:
        from engine.onboarding_agent.llm_json import extract_json
        return extract_json(raw)
    except Exception:                     # noqa: BLE001 - keep Stage 1B standalone
        import json as _json
        if isinstance(raw, dict):
            return raw, "ok", ""
        try:
            return _json.loads(raw), "ok", ""
        except Exception as exc:          # noqa: BLE001
            return None, "parse_failed", str(exc)


def triage_and_classify(publication: RawPublication,
                        classifier: Optional[SemanticClassifier] = None,
                        official_text: str = ""
                        ) -> Tuple[TriageResult, Optional[Classification]]:
    """Run Stage A then Stage B. Returns ``(triage, classification|None)``."""
    triage = deterministic_triage(publication)
    if not triage.passed:
        return triage, None
    if classifier is not None and classifier.enabled:
        return triage, classifier.classify(publication, triage, official_text)
    return triage, classify_deterministic(publication, triage)
