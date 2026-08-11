"""tests/test_regulatory_watch_discovery_linkage.py — the Stage 1A boundary.

The four required cases:

* Case A — a consultation: a regulatory development, no technical change;
* Case B — a final report: a future change expected, artefacts unchanged;
* Case C — a technical artefact publication: Stage 1A runs and produces exact
  field deltas, which Stage 1B *quotes* rather than computes;
* an unknown technical URL: ``NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED``, never an
  automatic addition to the Stage 1A allowlist.

The boundary itself is asserted structurally: Stage 1B must not import Stage
1A's comparison engine, because a module that cannot reach the comparator
cannot accidentally reimplement it.

Run: python -m pytest tests/test_regulatory_watch_discovery_linkage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch.discovery.contracts import (            # noqa: E402
    FUTURE_CHANGE_EXPECTED,
    MATCHES_STAGE1A_SOURCE,
    NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED,
    NO_TECHNICAL_ARTEFACT,
    POTENTIAL_FUTURE_CHANGE,
    TECHNICAL_SPEC_CHANGE,
)
from regulatory_watch.discovery.linkage import (              # noqa: E402
    assess_links, attach_stage1a_report, looks_technical, stage1a_source_urls,
)
from regulatory_watch.discovery.triage import (               # noqa: E402
    classify_deterministic, deterministic_triage,
)
from tests.helpers import regulatory_watch_discovery as fx     # noqa: E402


def _classify(pub):
    triage = deterministic_triage(pub)
    assert triage.passed
    return classify_deterministic(pub, triage)


# --------------------------------------------------------------------------- #
# Case A — narrative publication, no technical change
# --------------------------------------------------------------------------- #

def test_case_a_a_consultation_has_no_technical_linkage():
    pub = fx.publication(
        "Consultation Paper on amendments to the securitisation disclosure "
        "templates",
        "https://www.esma.europa.eu/document/consultation-disclosure-review",
        summary="ESMA seeks views on proposed changes to the Article 7 "
                "disclosure templates.")
    assert _classify(pub).development_status == POTENTIAL_FUTURE_CHANGE
    linkage = assess_links(pub, known_urls=fx.KNOWN_TECHNICAL_URLS)
    assert linkage.status == NO_TECHNICAL_ARTEFACT
    assert linkage.stage1a_verification_required is False
    assert linkage.regulatory_delta_count is None


# --------------------------------------------------------------------------- #
# Case B — final report, future change expected, artefacts unchanged
# --------------------------------------------------------------------------- #

def test_case_b_a_final_report_expects_change_without_a_technical_artefact():
    pub = fx.publication(
        "Final Report — Draft RTS and ITS on securitisation reporting",
        fx.REAL_FINAL_REPORT_URL,
        summary="ESMA final report on the draft technical standards on "
                "disclosure requirements under Article 7, covering the "
                "underlying exposures reporting templates.")
    assert _classify(pub).development_status == FUTURE_CHANGE_EXPECTED
    linkage = assess_links(pub, known_urls=fx.KNOWN_TECHNICAL_URLS)
    assert linkage.status == NO_TECHNICAL_ARTEFACT
    assert linkage.stage1a_verification_required is False


# --------------------------------------------------------------------------- #
# Case C — technical artefact publication
# --------------------------------------------------------------------------- #

def test_case_c_a_technical_package_matches_an_allowlisted_stage1a_source():
    pub = fx.publication(
        "Securitisation reporting technical standards — XML schema and "
        "instructions (version 1.3.1)", fx.STAGE1A_PACKAGE_URL,
        summary="Updated reporting templates and XML schema.")
    assert _classify(pub).development_status == TECHNICAL_SPEC_CHANGE
    linkage = assess_links(pub, known_urls=fx.KNOWN_TECHNICAL_URLS)
    assert linkage.status == MATCHES_STAGE1A_SOURCE
    assert linkage.stage1a_verification_required is True
    assert linkage.artefact_links[0].stage1a_artefact_id
    # Still no delta: Stage 1B has not run and cannot run the comparison.
    assert linkage.regulatory_delta_count is None


def test_a_match_names_every_stage1a_artefact_on_that_url():
    """Stage 1A declares three artefacts against the same ESMA package URL.

    Naming only one would leave a reviewer re-verifying a subset.
    """
    pub = fx.publication("Technical standards", fx.STAGE1A_PACKAGE_URL)
    linkage = assess_links(pub, known_urls=fx.KNOWN_TECHNICAL_URLS)
    named = linkage.artefact_links[0].stage1a_artefact_id
    assert "esma_annex2_message_workbook" in named
    assert "esma_annex2_xml_schema" in named


def test_case_c_stage1a_results_are_quoted_not_computed(tmp_path):
    """Stage 1B reads Stage 1A's published report contract. That is all."""
    report = tmp_path / "stage1a.json"
    report.write_text(json.dumps({
        "outcome": "REGULATORY_SPEC_CHANGED",
        "regulatory_delta_count": 3,
        "implementation_impact_count": 4,
        "baseline": {"sources": [
            {"artefact_id": "esma_annex2_xml_schema",
             "artefact_sha256": "a" * 64}]},
        "candidate": {"sources": [
            {"artefact_id": "esma_annex2_xml_schema",
             "artefact_sha256": "b" * 64}]},
    }), encoding="utf-8")

    pub = fx.publication("Technical standards — XML schema and instructions",
                         fx.STAGE1A_PACKAGE_URL)
    linkage = attach_stage1a_report(
        assess_links(pub, known_urls=fx.KNOWN_TECHNICAL_URLS), report)

    assert linkage.stage1a_outcome == "REGULATORY_SPEC_CHANGED"
    assert linkage.regulatory_delta_count == 3
    assert linkage.implementation_impact_count == 4
    assert linkage.stage1a_baseline_snapshot_ids == [
        "esma_annex2_xml_schema@aaaaaaaaaaaaaaaa"]
    assert linkage.stage1a_candidate_snapshot_ids == [
        "esma_annex2_xml_schema@bbbbbbbbbbbbbbbb"]
    # A Stage 1A run has happened, so verification is no longer outstanding.
    assert linkage.stage1a_verification_required is False


def test_an_unreadable_stage1a_report_is_reported_not_swallowed(tmp_path):
    missing = tmp_path / "absent.json"
    pub = fx.publication("Technical standards", fx.STAGE1A_PACKAGE_URL)
    linkage = attach_stage1a_report(
        assess_links(pub, known_urls=fx.KNOWN_TECHNICAL_URLS), missing)
    assert linkage.stage1a_outcome.startswith("STAGE1A_REPORT_UNREADABLE")
    assert linkage.stage1a_verification_required is True
    assert linkage.regulatory_delta_count is None


# --------------------------------------------------------------------------- #
# Unknown technical sources
# --------------------------------------------------------------------------- #

def test_an_unknown_technical_url_requires_review_and_is_never_auto_added():
    pub = fx.publication(
        "Securitisation reporting validation rules package",
        "https://www.esma.europa.eu/document/securitisation-validation-rules")
    linkage = assess_links(pub, known_urls=fx.KNOWN_TECHNICAL_URLS)
    assert linkage.status == NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED
    assert linkage.artefact_links[0].stage1a_artefact_id is None
    assert "never edits it" in linkage.artefact_links[0].detail


def test_stage1b_does_not_modify_the_stage1a_allowlist():
    import hashlib
    path = _REPO / "config" / "regulatory_watch" / "esma_annex2_sources.yaml"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    pub = fx.publication("Brand new package",
                         "https://www.esma.europa.eu/document/new-thing.xsd")
    assess_links(pub)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_a_linked_artefact_file_is_detected_even_from_a_narrative_page():
    pub = fx.publication(
        "ESMA publishes updated securitisation disclosure package",
        "https://www.esma.europa.eu/press-news/esma-news/update",
        summary="See the attached schema.",
        linked_urls=["https://www.esma.europa.eu/sites/default/files/auth099.xsd"])
    linkage = assess_links(pub, known_urls=fx.KNOWN_TECHNICAL_URLS)
    assert linkage.status == NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED


@pytest.mark.parametrize("url,expected", [
    ("https://www.esma.europa.eu/file/a.xsd", True),
    ("https://www.esma.europa.eu/file/templates.xlsx", True),
    ("https://www.esma.europa.eu/document/xml-schema-and-instructions", True),
    ("https://www.esma.europa.eu/document/validation-rules-package", True),
    ("https://www.esma.europa.eu/press-news/speech", False),
    ("https://www.esma.europa.eu/document/market-report", False),
])
def test_technical_url_detection(url, expected):
    assert looks_technical(url) is expected


# --------------------------------------------------------------------------- #
# The live Stage 1A allowlist
# --------------------------------------------------------------------------- #

def test_the_live_stage1a_allowlist_is_readable_and_matches_the_real_url():
    """Proves the linkage works against the ACTUAL Stage 1A config, not a stub."""
    known = stage1a_source_urls()
    assert known, "Stage 1A artefact allowlist should be readable"
    normalised = fx.STAGE1A_PACKAGE_URL.rstrip("/")
    assert normalised in known, sorted(known)
    assert "esma_annex2_message_workbook" in known[normalised]


def test_stage1b_never_imports_the_stage1a_comparison_engine():
    """Structural boundary: what cannot be reached cannot be reimplemented."""
    import re
    package = _REPO / "regulatory_watch" / "discovery"
    forbidden = re.compile(
        r"(from|import)\s+.*\b(comparator|annex2_spec|impact|changelog)\b")
    offenders = []
    for module in sorted(package.glob("*.py")):
        for line_no, line in enumerate(
                module.read_text(encoding="utf-8").splitlines(), start=1):
            if forbidden.search(line):
                offenders.append(f"{module.name}:{line_no}: {line.strip()}")
    assert offenders == [], offenders


def test_the_development_record_cannot_express_a_field_level_change():
    """There is no field on the record that could hold a field delta."""
    import dataclasses
    from regulatory_watch.discovery.contracts import (
        RegulatoryDevelopment, TechnicalSpecLinkage,
    )
    names = {f.name for f in dataclasses.fields(RegulatoryDevelopment)}
    names |= {f.name for f in dataclasses.fields(TechnicalSpecLinkage)}
    for forbidden in ("esma_code", "field_code", "deltas", "nd_allowed",
                      "enum_values", "xml_path", "old_value", "new_value"):
        assert forbidden not in names, forbidden
