"""tests/test_regulatory_watch_discovery_triage.py — relevance and classification.

Two properties are load-bearing and both are pinned here:

1. **Stage A must not produce false negatives.** A publication Trakt never
   hears about is invisible forever; a false positive costs a reviewer thirty
   seconds. So the filter is deliberately broad, and the tests assert that
   breadth on the borderline cases rather than assuming it.
2. **Document form beats subject matter.** "Consultation on amendments to the
   disclosure templates" is a CONSULTATION, not a template update. Getting this
   backwards would mark every discussion of a template as a technical change
   and destroy the distinction Stage 1B exists to preserve.

Run: python -m pytest tests/test_regulatory_watch_discovery_triage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch.discovery.contracts import (            # noqa: E402
    CLASSIFIER_DETERMINISTIC,
    CLASSIFIER_SEMANTIC,
    CLASSIFIER_SEMANTIC_FAILED,
    CONSULTATION,
    DEVELOPMENT_STATUSES,
    FINAL_REPORT,
    FUTURE_CHANGE_EXPECTED,
    HIGH,
    INFORMATION_ONLY,
    INTERPRETATION_REVIEW_REQUIRED,
    LOW,
    POTENTIAL_FUTURE_CHANGE,
    PUBLICATION_TYPES,
    Q_AND_A,
    RELEVANCE_NOT_RELEVANT,
    RELEVANCE_VALUES,
    REVIEW_REQUIRED,
    RTS_ITS,
    TECHNICAL_INSTRUCTION,
    TECHNICAL_SPEC_CHANGE,
    TEMPLATE_SCHEMA_UPDATE,
    VALIDATION_RULE_UPDATE,
)
from regulatory_watch.discovery.triage import (               # noqa: E402
    SemanticClassifier, carries_artefact, classify_deterministic,
    deterministic_triage, extract_dates, triage_and_classify,
)
from tests.helpers import regulatory_watch_discovery as fx     # noqa: E402


# --------------------------------------------------------------------------- #
# Stage A — deterministic relevance
# --------------------------------------------------------------------------- #

def test_a_clearly_irrelevant_market_publication_is_rejected():
    result = deterministic_triage(fx.publication(
        "ESMA publishes latest edition of its Spotlight on Markets newsletter",
        "https://www.esma.europa.eu/press-news/esma-news/spotlight",
        summary="Recent supervisory convergence work, speeches and staff "
                "appointments."))
    assert result.passed is False
    assert result.reason
    assert result.matched_groups == []


def test_the_possessive_its_does_not_masquerade_as_the_its_acronym():
    """Regression: " its" as a keyword matched "ESMA publishes ITS newsletter".

    A substring match on a two-letter acronym that is also an English word
    passed unrelated press releases into triage as technical standards.
    """
    result = deterministic_triage(fx.publication(
        "ESMA renews its supervisory priorities",
        summary="The authority sets out its plans for the coming year."))
    assert result.passed is False


def test_a_broad_securitisation_publication_passes_but_only_weakly():
    result = deterministic_triage(fx.publication(
        "ESMA report on the securitisation market in the European Union",
        summary="Statistical overview of issuance volumes."))
    assert result.passed is True
    assert "securitisation" in result.matched_groups
    classification = classify_deterministic(fx.publication(
        "ESMA report on the securitisation market in the European Union",
        summary="Statistical overview of issuance volumes."), result)
    assert classification.development_status == INFORMATION_ONLY
    assert classification.annex2_relevance == RELEVANCE_NOT_RELEVANT


def test_an_explicit_annex_2_publication_passes():
    result = deterministic_triage(fx.publication(
        "Amendments to Annex 2 of the securitisation disclosure RTS"))
    assert result.passed is True
    assert "annex2" in result.matched_groups


def test_an_article_7_disclosure_publication_passes():
    result = deterministic_triage(fx.publication(
        "Guidance on Article 7 disclosure requirements",
        summary="Transparency requirements for originators."))
    assert result.passed is True
    assert "disclosure" in result.matched_groups


def test_a_q_and_a_on_disclosure_fields_passes_with_several_groups():
    result = deterministic_triage(fx.publication(
        "Questions and Answers on the Securitisation Regulation — disclosure "
        "templates",
        summary="Clarifies completion of residential real estate underlying "
                "exposure fields, including no-data options ND1 to ND5."))
    assert result.passed is True
    assert {"q_and_a", "disclosure", "underlying_exposures",
            "residential_real_estate", "no_data"} <= set(result.matched_groups)


def test_a_rejection_always_records_a_traceable_reason():
    result = deterministic_triage(fx.publication("Speech by the ESMA Chair"))
    assert result.passed is False
    assert "no ESMA Annex 2 relevance term matched" in result.reason


def test_relevance_terms_match_the_url_as_well_as_the_text():
    result = deterministic_triage(fx.publication(
        "Untitled document",
        url="https://www.esma.europa.eu/document/securitisation-disclosure-x"))
    assert result.passed is True


# --------------------------------------------------------------------------- #
# Stage B — classification
# --------------------------------------------------------------------------- #

def _classify(title: str, **kwargs):
    pub = fx.publication(title, **kwargs)
    triage = deterministic_triage(pub)
    assert triage.passed, f"Stage A rejected: {title}"
    return classify_deterministic(pub, triage)


def test_consultation_is_a_potential_future_change():
    result = _classify(
        "Consultation Paper on amendments to the securitisation disclosure "
        "templates",
        summary="ESMA seeks views on proposed changes to the Article 7 "
                "disclosure templates for underlying exposures.")
    assert result.publication_type == CONSULTATION
    assert result.development_status == POTENTIAL_FUTURE_CHANGE
    assert result.review_required is True


def test_final_report_is_a_future_change_expected():
    result = _classify(
        "Final Report — Draft RTS and ITS on securitisation reporting",
        summary="ESMA final report on the draft technical standards on "
                "disclosure requirements under Article 7.")
    assert result.publication_type == FINAL_REPORT
    assert result.development_status == FUTURE_CHANGE_EXPECTED


def test_q_and_a_is_an_interpretation_review():
    result = _classify(
        "Questions and Answers on the Securitisation Regulation — disclosure "
        "templates",
        summary="Clarifies how to complete residential real estate underlying "
                "exposure fields.")
    assert result.publication_type == Q_AND_A
    assert result.development_status == INTERPRETATION_REVIEW_REQUIRED
    assert "populated" in result.relevance_reason


def test_rts_its_without_a_stated_date_is_potential_not_expected():
    result = _classify(
        "Regulatory technical standards on securitisation disclosure",
        summary="Technical standards covering underlying exposure reporting.")
    assert result.publication_type == RTS_ITS
    assert result.development_status == POTENTIAL_FUTURE_CHANGE


def test_rts_its_with_a_stated_application_date_is_expected():
    result = _classify(
        "Regulatory technical standards on securitisation disclosure",
        summary="These technical standards will apply from 1 January 2027 to "
                "underlying exposure reporting.")
    assert result.development_status == FUTURE_CHANGE_EXPECTED


def test_technical_instructions_are_an_interpretation_review():
    result = _classify(
        "Technical reporting instructions for securitisation disclosure",
        summary="How to populate the underlying exposure reporting fields.")
    assert result.publication_type == TECHNICAL_INSTRUCTION
    assert result.development_status == INTERPRETATION_REVIEW_REQUIRED


def test_a_schema_package_is_a_technical_spec_change():
    result = _classify(
        "Securitisation reporting technical standards — XML schema and "
        "instructions (version 1.3.1)",
        summary="Updated reporting templates, XML schema and technical "
                "reporting instructions for securitisation disclosure.")
    assert result.publication_type == TEMPLATE_SCHEMA_UPDATE
    assert result.development_status == TECHNICAL_SPEC_CHANGE
    assert result.confidence == HIGH
    assert "Stage 1A must run" in result.relevance_reason


def test_a_validation_rules_package_is_a_technical_spec_change():
    result = _classify(
        "Securitisation reporting validation rules package",
        summary="Validation rules applied by securitisation repositories.")
    assert result.publication_type == VALIDATION_RULE_UPDATE
    assert result.development_status == TECHNICAL_SPEC_CHANGE


def test_document_form_beats_subject_matter():
    """A Q&A *about* templates is a Q&A, not a template update.

    Regression: subject-matter signals were checked first, so every Q&A and
    consultation mentioning "disclosure templates" was classified as a
    TECHNICAL_SPEC_CHANGE — which is precisely the confusion between a
    regulatory development and a technical delta that Stage 1B must not make.
    """
    for title in ("Questions and Answers on the disclosure templates",
                  "Consultation Paper on the disclosure templates",
                  "Final Report on the reporting templates"):
        result = _classify(title, summary="Securitisation disclosure.")
        assert result.publication_type != TEMPLATE_SCHEMA_UPDATE, title
        assert result.development_status != TECHNICAL_SPEC_CHANGE, title


def test_a_package_that_carries_the_artefact_is_typed_by_the_artefact():
    """The auth.099 package bundles the schema AND the instructions.

    Typing it as an instructions document would hide the one publication that
    most needs Stage 1A to run.
    """
    pub = fx.publication(
        "Securitisation reporting technical standards — XML schema and "
        "instructions",
        summary="Includes the technical reporting instructions.")
    assert carries_artefact(pub) is True
    triage = deterministic_triage(pub)
    assert classify_deterministic(pub, triage).publication_type == \
        TEMPLATE_SCHEMA_UPDATE


def test_a_linked_artefact_file_also_counts_as_carrying():
    pub = fx.publication(
        "Securitisation disclosure package",
        linked_urls=["https://www.esma.europa.eu/file/auth099.xsd"])
    assert carries_artefact(pub) is True


def test_discussing_templates_is_not_carrying_them():
    pub = fx.publication("Report on the use of disclosure templates")
    assert carries_artefact(pub) is False


def test_an_ambiguous_publication_becomes_review_required():
    """Relevant terms matched, but no publication-type signal was recognised."""
    result = _classify(
        "ESMA statement on underlying exposure data",
        summary="Annex 2 residential real estate matters.")
    assert result.development_status == REVIEW_REQUIRED
    assert result.confidence == LOW
    assert result.review_required is True


def test_every_classification_uses_the_published_vocabulary():
    for title in ("Consultation Paper on securitisation disclosure",
                  "Final Report on securitisation reporting",
                  "Q&A on the securitisation disclosure templates",
                  "XML schema for securitisation reporting",
                  "Validation rules for securitisation disclosure",
                  "ESMA report on the securitisation market"):
        result = _classify(title, summary="securitisation disclosure")
        assert result.publication_type in PUBLICATION_TYPES
        assert result.development_status in DEVELOPMENT_STATUSES
        assert result.annex2_relevance in RELEVANCE_VALUES
        assert result.relevance_reason
        assert result.classifier == CLASSIFIER_DETERMINISTIC


def test_classification_is_deterministic():
    pub = fx.publication("Q&A on securitisation disclosure templates",
                         summary="residential real estate fields")
    triage = deterministic_triage(pub)
    assert classify_deterministic(pub, triage).to_dict() == \
        classify_deterministic(pub, triage).to_dict()


# --------------------------------------------------------------------------- #
# Dates are never guessed
# --------------------------------------------------------------------------- #

def test_a_stated_consultation_deadline_is_captured():
    dates = extract_dates(fx.publication(
        "Consultation Paper", summary="Responses close 2026-05-15."))
    assert dates["consultation_deadline"] == "2026-05-15"


def test_a_stated_application_date_is_captured():
    dates = extract_dates(fx.publication(
        "RTS", summary="Applicable from 2027-01-01 for all originators."))
    assert dates["implementation_date"] == "2027-01-01"


def test_a_vague_date_stays_unknown():
    dates = extract_dates(fx.publication(
        "Consultation Paper", summary="Responses close in the spring."))
    assert dates == {"effective_date": None, "implementation_date": None,
                     "consultation_deadline": None}


# --------------------------------------------------------------------------- #
# The bounded semantic classifier
# --------------------------------------------------------------------------- #

def _pub():
    return fx.publication("ESMA statement on underlying exposure data",
                          summary="Annex 2 residential real estate matters.")


def test_the_semantic_classifier_is_off_by_default():
    pub = _pub()
    _, result = triage_and_classify(pub)
    assert result.classifier == CLASSIFIER_DETERMINISTIC


def test_the_semantic_backend_sees_only_bounded_publication_context():
    """It must never be handed Stage 1A data or free rein over the record."""
    captured = {}

    def backend(payload):
        captured.update(payload)
        return '{"publication_type": "Q_AND_A", "development_status": ' \
               '"INTERPRETATION_REVIEW_REQUIRED", "annex2_relevance": ' \
               '"RELEVANT", "relevance_reason": "clarifies fields", ' \
               '"confidence": "HIGH"}'

    pub = _pub()
    triage = deterministic_triage(pub)
    SemanticClassifier(backend=backend).classify(pub, triage)
    assert set(captured) == {
        "task", "title", "source_id", "url", "publication_date", "categories",
        "official_text", "matched_relevance_groups",
        "allowed_publication_type", "allowed_development_status",
        "allowed_annex2_relevance", "response_schema"}
    # No Annex 2 field codes, no config, no Stage 1A anything.
    blob = repr(captured).lower()
    for forbidden in ("rrel", "rrec", "fields_registry", "delivery_rules",
                      "stage1a", "xsd_path_map"):
        assert forbidden not in blob


def test_a_valid_semantic_escalation_is_accepted():
    def backend(payload):
        return {"publication_type": "TEMPLATE_SCHEMA_UPDATE",
                "development_status": "TECHNICAL_SPEC_CHANGE",
                "annex2_relevance": "RELEVANT",
                "relevance_reason": "ships the schema",
                "confidence": "HIGH"}

    pub = _pub()
    triage = deterministic_triage(pub)
    result = SemanticClassifier(backend=backend).classify(pub, triage)
    assert result.development_status == TECHNICAL_SPEC_CHANGE
    assert result.classifier == CLASSIFIER_SEMANTIC


def test_the_semantic_classifier_may_escalate_but_never_downgrade():
    """A model must not be able to quietly make a finding less urgent."""
    def backend(payload):
        return {"publication_type": "OTHER_REGULATORY_PUBLICATION",
                "development_status": "INFORMATION_ONLY",
                "annex2_relevance": "NOT_RELEVANT",
                "relevance_reason": "looks routine to me",
                "confidence": "HIGH"}

    pub = fx.publication(
        "Securitisation reporting technical standards — XML schema and "
        "instructions", summary="securitisation disclosure")
    triage = deterministic_triage(pub)
    baseline = classify_deterministic(pub, triage)
    assert baseline.development_status == TECHNICAL_SPEC_CHANGE
    result = SemanticClassifier(backend=backend).classify(pub, triage)
    assert result.development_status == TECHNICAL_SPEC_CHANGE
    assert "never downgrade" in result.relevance_reason


def test_an_out_of_vocabulary_response_becomes_review_required():
    def backend(payload):
        return {"publication_type": "PROBABLY_IMPORTANT",
                "development_status": "SEEMS_FINE",
                "annex2_relevance": "MAYBE", "relevance_reason": "dunno",
                "confidence": "HIGH"}

    pub = _pub()
    result = SemanticClassifier(backend=backend).classify(
        pub, deterministic_triage(pub))
    assert result.development_status == REVIEW_REQUIRED
    assert result.classifier == CLASSIFIER_SEMANTIC_FAILED
    assert result.review_required is True


def test_an_unparseable_response_becomes_review_required():
    def backend(payload):
        return "I think this one is probably about securitisation?"

    pub = _pub()
    result = SemanticClassifier(backend=backend).classify(
        pub, deterministic_triage(pub))
    assert result.classifier == CLASSIFIER_SEMANTIC_FAILED
    assert result.review_required is True


def test_a_raising_backend_becomes_review_required():
    def backend(payload):
        raise TimeoutError("model unavailable")

    pub = _pub()
    result = SemanticClassifier(backend=backend).classify(
        pub, deterministic_triage(pub))
    assert result.classifier == CLASSIFIER_SEMANTIC_FAILED
    assert "TimeoutError" in result.relevance_reason


def test_a_low_confidence_response_becomes_review_required():
    def backend(payload):
        return {"publication_type": "Q_AND_A",
                "development_status": "INTERPRETATION_REVIEW_REQUIRED",
                "annex2_relevance": "POTENTIAL", "relevance_reason": "maybe",
                "confidence": "LOW"}

    pub = _pub()
    result = SemanticClassifier(backend=backend).classify(
        pub, deterministic_triage(pub))
    assert result.classifier == CLASSIFIER_SEMANTIC_FAILED
    assert result.review_required is True


def test_a_failing_backend_never_loses_a_technical_spec_change():
    def backend(payload):
        raise RuntimeError("boom")

    pub = fx.publication(
        "Securitisation reporting technical standards — XML schema and "
        "instructions", summary="securitisation disclosure")
    result = SemanticClassifier(backend=backend).classify(
        pub, deterministic_triage(pub))
    assert result.development_status == TECHNICAL_SPEC_CHANGE


def test_stage_a_rejection_short_circuits_stage_b():
    """A model is never asked about something the filter already rejected."""
    def backend(payload):                 # pragma: no cover - must not run
        raise AssertionError("Stage B ran on a Stage A rejection")

    triage, classification = triage_and_classify(
        fx.publication("Speech by the ESMA Chair"),
        SemanticClassifier(backend=backend))
    assert triage.passed is False
    assert classification is None
