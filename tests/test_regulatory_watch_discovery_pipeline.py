"""tests/test_regulatory_watch_discovery_pipeline.py — the weekly pass, end to end.

Covers the full control — discover -> dedupe -> classify -> report — plus the
CLI, the report contract, and the historical replay against the real ESMA
publications this repository already relies on.

The two properties that make the weekly run trustworthy:

* running the same week twice creates **zero** duplicate developments;
* a discovery failure is **never** reported as "nothing new".

Run: python -m pytest tests/test_regulatory_watch_discovery_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch.discovery import DISCOVERY_REPORT_CONTRACT_VERSION  # noqa: E402
from regulatory_watch.discovery.contracts import (            # noqa: E402
    CORROBORATING,
    DEVELOPMENT_STATUSES,
    DISCOVERY_FAILED,
    DISCOVERY_OK,
    DISCOVERY_PARTIAL,
    FINAL_REPORT,
    FUTURE_CHANGE_EXPECTED,
    INFORMATION_ONLY,
    INTERPRETATION_REVIEW_REQUIRED,
    MATCHES_STAGE1A_SOURCE,
    NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED,
    POTENTIAL_FUTURE_CHANGE,
    PUBLICATION_TYPES,
    Q_AND_A,
    SEEN_REVISED,
    SOURCE_FETCH_FAILED,
    SOURCE_UNEXPECTED_SCHEMA,
    TECHNICAL_SPEC_CHANGE,
    TEMPLATE_SCHEMA_UPDATE,
)
from regulatory_watch.discovery.pipeline import run_discovery  # noqa: E402
from regulatory_watch.discovery.report import (               # noqa: E402
    to_json, to_markdown, write_reports,
)
from regulatory_watch.discovery.state import SeenStore        # noqa: E402
from tests.helpers import regulatory_watch_discovery as fx     # noqa: E402

FIXTURE_DIR = fx.FIXTURE_DIR
WHEN = fx.DISCOVERY_DATE
LATER = "2026-02-23T00:00:00Z"


def _sources():
    return fx.source_set([
        fx.source("esma_publications_feed", parser="rss"),
        fx.source("esma_securitisation_technical_standards",
                  parser="html_listing",
                  url="https://www.esma.europa.eu/document/technical-standards"),
        fx.source("esma_securitisation_landing", parser="html_listing",
                  criticality=CORROBORATING,
                  url="https://www.esma.europa.eu/topic/securitisation"),
    ])


def _payloads(sources=None):
    sources = sources or _sources()
    return [
        fx.payload(sources.sources[0], fx.feed_bytes()),
        fx.payload(sources.sources[1],
                   fx.TECHNICAL_PAGE_FIXTURE.read_bytes()),
        fx.payload(sources.sources[2], fx.LANDING_FIXTURE.read_bytes()),
    ]


def _run(tmp_path, payloads=None, sources=None, when=WHEN, store=None,
         **kwargs):
    sources = sources or _sources()
    store = store if store is not None else SeenStore.load(tmp_path / "seen.json")
    return run_discovery(sources, payloads if payloads is not None
                         else _payloads(sources), store, when,
                         known_technical_urls=fx.KNOWN_TECHNICAL_URLS,
                         **kwargs)


# --------------------------------------------------------------------------- #
# The weekly pass
# --------------------------------------------------------------------------- #

def test_a_weekly_pass_classifies_the_whole_fixture_week(tmp_path):
    result = _run(tmp_path)
    report = result.report
    assert report.discovery_status == DISCOVERY_OK
    assert report.sources_checked == 3
    assert report.sources_failed == 0

    by_title = {d.publication["title"]: d for d in result.developments}
    statuses = {t: d.development_status for t, d in by_title.items()}

    assert any(s == TECHNICAL_SPEC_CHANGE for s in statuses.values())
    assert any(s == FUTURE_CHANGE_EXPECTED for s in statuses.values())
    assert any(s == POTENTIAL_FUTURE_CHANGE for s in statuses.values())
    assert any(s == INTERPRETATION_REVIEW_REQUIRED for s in statuses.values())
    assert any(s == INFORMATION_ONLY for s in statuses.values())


def test_an_irrelevant_publication_is_rejected_with_a_reason(tmp_path):
    result = _run(tmp_path)
    rejected = result.report.rejected
    assert len(rejected) == 1
    assert "Spotlight" in rejected[0]["title"]
    assert rejected[0]["rejected_by"] == "deterministic_relevance_filter"
    assert rejected[0]["reason"]


def test_the_same_publication_from_two_sources_is_one_development(tmp_path):
    """The auth.099 package is in the feed AND on the technical-standards page."""
    result = _run(tmp_path)
    ids = [d.development_id for d in result.developments]
    assert len(ids) == len(set(ids)), "duplicate development records"
    assert result.report.publications_seen > len(result.developments)
    merged = [d for d in result.developments
              if d.publication.get("also_seen_in_sources")]
    assert merged, "a cross-source sighting should record the extra source"
    assert any("merged into a single development record" in n
               for n in result.report.notes)


def test_a_run_is_deterministic(tmp_path):
    a = _run(tmp_path / "a").report.to_dict()
    b = _run(tmp_path / "b").report.to_dict()
    assert a == b


# --------------------------------------------------------------------------- #
# Deduplication across weeks
# --------------------------------------------------------------------------- #

def test_running_the_same_week_twice_creates_zero_duplicates(tmp_path):
    path = tmp_path / "seen.json"

    first = _run(tmp_path, store=SeenStore.load(path))
    first.store.save()
    assert first.report.new_publications > 0
    assert len(first.developments) > 0

    second = _run(tmp_path, store=SeenStore.load(path), when=LATER)
    second.store.save()
    assert second.report.new_publications == 0
    assert second.developments == []
    assert second.report.previously_seen_publications > 0
    assert SeenStore.load(path).records.keys() == \
        first.store.records.keys()


def test_a_revised_publication_is_raised_again_as_revised(tmp_path):
    path = tmp_path / "seen.json"
    sources = _sources()
    _run(tmp_path, store=SeenStore.load(path)).store.save()

    corrected = fx.rss(fx.rss_item(
        "Questions and Answers on the Securitisation Regulation — disclosure "
        "templates (corrected)",
        "https://www.esma.europa.eu/document/"
        "qa-securitisation-regulation-disclosure-templates",
        description="Corrected guidance on residential real estate underlying "
                    "exposure fields."))
    payloads = [fx.payload(sources.sources[0], corrected),
                fx.payload(sources.sources[1],
                           fx.TECHNICAL_PAGE_FIXTURE.read_bytes()),
                fx.payload(sources.sources[2], fx.LANDING_FIXTURE.read_bytes())]

    second = _run(tmp_path, payloads=payloads, sources=sources,
                  store=SeenStore.load(path), when=LATER)
    revised = [d for d in second.developments if d.seen_state == SEEN_REVISED]
    assert len(revised) == 1
    assert revised[0].publication_type == Q_AND_A
    assert second.report.revised_publications == 1


def test_a_dry_run_does_not_teach_the_store(tmp_path):
    path = tmp_path / "seen.json"
    _run(tmp_path, store=SeenStore.load(path), commit_state=False)
    assert len(SeenStore.load(path)) == 0


# --------------------------------------------------------------------------- #
# Failure behaviour
# --------------------------------------------------------------------------- #

def test_a_gating_source_failure_fails_the_whole_run(tmp_path):
    sources = _sources()
    payloads = [
        fx.failed_payload(sources.sources[0], SOURCE_FETCH_FAILED,
                          "connection refused"),
        fx.payload(sources.sources[1], fx.TECHNICAL_PAGE_FIXTURE.read_bytes()),
        fx.payload(sources.sources[2], fx.LANDING_FIXTURE.read_bytes()),
    ]
    report = _run(tmp_path, payloads=payloads, sources=sources).report
    assert report.discovery_status == DISCOVERY_FAILED
    assert report.sources_failed == 1
    assert any("cannot be read as 'no new relevant ESMA publication'" in n
               for n in report.notes)
    assert "not** evidence" in to_markdown(report)


def test_a_corroborating_source_failure_only_narrows_the_run(tmp_path):
    sources = _sources()
    payloads = [
        fx.payload(sources.sources[0], fx.feed_bytes()),
        fx.payload(sources.sources[1], fx.TECHNICAL_PAGE_FIXTURE.read_bytes()),
        fx.failed_payload(sources.sources[2], SOURCE_FETCH_FAILED, "timeout"),
    ]
    report = _run(tmp_path, payloads=payloads, sources=sources).report
    assert report.discovery_status == DISCOVERY_PARTIAL
    assert report.developments


def test_a_malformed_feed_is_an_unexpected_schema_not_an_empty_sweep(tmp_path):
    sources = _sources()
    payloads = [
        fx.payload(sources.sources[0], b"<html><body>maintenance</body></html>"),
        fx.payload(sources.sources[1], fx.TECHNICAL_PAGE_FIXTURE.read_bytes()),
        fx.payload(sources.sources[2], fx.LANDING_FIXTURE.read_bytes()),
    ]
    report = _run(tmp_path, payloads=payloads, sources=sources).report
    assert report.discovery_status == DISCOVERY_FAILED
    outcome = next(o for o in report.source_outcomes
                   if o["source_id"] == "esma_publications_feed")
    assert outcome["outcome"] == SOURCE_UNEXPECTED_SCHEMA
    assert outcome["detail"]


def test_every_source_is_accounted_for_even_when_it_fails(tmp_path):
    sources = _sources()
    payloads = [fx.failed_payload(s, SOURCE_FETCH_FAILED, "down")
                for s in sources.sources]
    report = _run(tmp_path, payloads=payloads, sources=sources).report
    assert len(report.source_outcomes) == 3
    assert report.discovery_status == DISCOVERY_FAILED
    assert report.developments == []
    # Crucially: zero developments here does NOT read as "nothing new".
    assert report.discovery_status != DISCOVERY_OK


# --------------------------------------------------------------------------- #
# Stage 1A linkage through the pipeline
# --------------------------------------------------------------------------- #

def test_a_technical_package_asks_for_a_stage1a_run(tmp_path):
    result = _run(tmp_path)
    technical = [d for d in result.developments
                 if d.development_status == TECHNICAL_SPEC_CHANGE]
    assert technical
    matched = [d for d in technical
               if (d.technical_spec_linkage or {})["status"]
               == MATCHES_STAGE1A_SOURCE]
    assert matched
    assert matched[0].technical_spec_linkage[
        "stage1a_verification_required"] is True
    assert matched[0].technical_spec_linkage["regulatory_delta_count"] is None
    assert any("Stage 1B does not determine field deltas" in n
               for n in result.report.notes)


def test_an_attached_stage1a_report_is_quoted_in_the_record(tmp_path):
    stage1a = tmp_path / "stage1a.json"
    stage1a.write_text(json.dumps({
        "outcome": "REGULATORY_SPEC_CHANGED",
        "regulatory_delta_count": 3, "implementation_impact_count": 4,
        "baseline": {"sources": [{"artefact_id": "esma_annex2_xml_schema",
                                  "artefact_sha256": "a" * 64}]},
        "candidate": {"sources": [{"artefact_id": "esma_annex2_xml_schema",
                                   "artefact_sha256": "b" * 64}]},
    }), encoding="utf-8")

    preview = _run(tmp_path / "preview")
    target = next(d for d in preview.developments
                  if (d.technical_spec_linkage or {})["status"]
                  == MATCHES_STAGE1A_SOURCE)

    result = _run(tmp_path / "run",
                  stage1a_reports={target.development_id: stage1a})
    linked = next(d for d in result.developments
                  if d.development_id == target.development_id)
    assert linked.technical_spec_linkage["regulatory_delta_count"] == 3
    assert linked.technical_spec_linkage["implementation_impact_count"] == 4
    markdown = to_markdown(result.report)
    assert "3 regulatory delta(s)" in markdown
    assert "Trakt implementation impacts: 4" in markdown


def test_an_unknown_technical_url_is_flagged_in_the_report(tmp_path):
    result = _run(tmp_path)
    flagged = [d for d in result.developments
               if (d.technical_spec_linkage or {})["status"]
               == NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED]
    assert flagged, "the validation-rules package is not a Stage 1A source"
    assert NEW_TECHNICAL_SOURCE_REVIEW_REQUIRED in to_markdown(result.report)


# --------------------------------------------------------------------------- #
# Historical replay — real ESMA publications
# --------------------------------------------------------------------------- #

def test_historical_replay_of_two_real_esma_publications(tmp_path):
    """Both items are REAL ESMA securitisation publications.

    Their titles and URLs are the ones this repository already depends on:
    ``config/system/esma_code_order.yaml`` cites the Final Report as the source
    of the ESMA field ordering, and the technical-standards package is the
    Stage 1A artefact source. The v1.3.1 publication date is the date ESMA
    records in the workbook's own change log.

    Limitation, stated rather than papered over: the repository holds no
    archived ESMA *feed*, so the surrounding feed document is a structural
    fixture. No regulatory history has been invented — the two real items carry
    their real titles, URLs and dates.
    """
    result = _run(tmp_path)
    by_url = {d.publication["url"]: d for d in result.developments}

    final_report = by_url[fx.REAL_FINAL_REPORT_URL]
    assert final_report.publication_type == FINAL_REPORT
    assert final_report.development_status == FUTURE_CHANGE_EXPECTED
    assert final_report.publication["publication_date"] == "2019-11-27"
    assert final_report.final_status == "final"
    assert final_report.review_required is True

    package = by_url[fx.STAGE1A_PACKAGE_URL]
    assert package.publication_type == TEMPLATE_SCHEMA_UPDATE
    assert package.development_status == TECHNICAL_SPEC_CHANGE
    assert package.publication["publication_date"] == "2021-06-11"
    assert package.technical_spec_linkage["status"] == MATCHES_STAGE1A_SOURCE


# --------------------------------------------------------------------------- #
# The report contract
# --------------------------------------------------------------------------- #

def test_the_report_contract_shape(tmp_path):
    payload = json.loads(to_json(_run(tmp_path).report))
    for key in ("discovery_status", "sources_checked", "sources_failed",
                "new_publications", "previously_seen_publications",
                "potentially_relevant", "not_relevant",
                "review_required_count", "technical_spec_change_candidates",
                "developments", "rejected", "stage1a_linkages"):
        assert key in payload, key
    assert payload["authority"] == "ESMA"
    assert payload["regime"] == "ESMA_Annex2"
    assert payload["contract_version"] == DISCOVERY_REPORT_CONTRACT_VERSION


def test_every_development_record_matches_the_published_contract(tmp_path):
    for development in _run(tmp_path).developments:
        record = development.to_dict()
        assert record["authority"] == "ESMA"
        assert record["regime"] == "ESMA_Annex2"
        assert record["publication_type"] in PUBLICATION_TYPES
        assert record["development_status"] in DEVELOPMENT_STATUSES
        assert record["relevance_reason"]
        assert isinstance(record["review_required"], bool)
        assert record["publication"]["content_sha256"]
        assert record["development_id"].startswith("esma-a2-")
        assert record["discovery_version"]


def test_unknown_dates_stay_null_rather_than_being_guessed(tmp_path):
    for development in _run(tmp_path).developments:
        for key in ("effective_date", "implementation_date"):
            value = development.to_dict()[key]
            assert value is None or len(value) == 10


def test_the_report_is_ui_agnostic(tmp_path):
    text = to_json(_run(tmp_path).report).lower()
    for forbidden in ("<div", "<span", "class=", "colour", "css"):
        assert forbidden not in text


def test_the_report_is_byte_reproducible(tmp_path):
    assert to_json(_run(tmp_path / "a").report) == \
        to_json(_run(tmp_path / "b").report)


def test_the_markdown_is_concise_and_states_the_stage_boundary(tmp_path):
    markdown = to_markdown(_run(tmp_path).report)
    assert markdown.startswith("# ESMA Annex 2 Regulatory Watch")
    assert "Discovery status" in markdown
    assert "Sources checked" in markdown
    assert "Potentially relevant to Annex 2" in markdown
    assert "does not determine field-level Annex 2 changes" in markdown
    assert len(markdown.splitlines()) < 120, "the weekly summary must stay short"


def test_write_reports_emits_both_files(tmp_path):
    paths = write_reports(_run(tmp_path).report, tmp_path / "out")
    assert paths["json"].exists() and paths["markdown"].exists()
    assert json.loads(paths["json"].read_text("utf-8"))["regime"] == \
        "ESMA_Annex2"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_the_cli_runs_the_whole_pass_from_fixtures(tmp_path):
    from regulatory_watch.discovery.cli import main
    code = main(["discover", "--out-dir", str(tmp_path / "out"),
                 "--fixture-dir", str(FIXTURE_DIR),
                 "--state", str(tmp_path / "seen.json"),
                 "--discovery-date", WHEN])
    assert code == 0
    payload = json.loads(
        (tmp_path / "out" / "esma_annex2_publication_discovery.json")
        .read_text("utf-8"))
    assert payload["discovery_status"] == DISCOVERY_OK
    assert payload["developments"]
    assert (tmp_path / "seen.json").exists()


def test_the_cli_is_idempotent_across_two_weekly_runs(tmp_path):
    from regulatory_watch.discovery.cli import main
    args = ["discover", "--out-dir", str(tmp_path / "out"),
            "--fixture-dir", str(FIXTURE_DIR),
            "--state", str(tmp_path / "seen.json")]
    main(args + ["--discovery-date", WHEN])
    first = json.loads((tmp_path / "out" /
                        "esma_annex2_publication_discovery.json").read_text())
    main(args + ["--discovery-date", LATER])
    second = json.loads((tmp_path / "out" /
                         "esma_annex2_publication_discovery.json").read_text())
    assert first["new_publications"] > 0
    assert second["new_publications"] == 0
    assert second["developments"] == []


def test_the_cli_exits_non_zero_when_a_gating_source_fails(tmp_path):
    from regulatory_watch.discovery.cli import main
    empty = tmp_path / "no-fixtures"
    empty.mkdir()
    code = main(["discover", "--out-dir", str(tmp_path / "out"),
                 "--fixture-dir", str(empty),
                 "--state", str(tmp_path / "seen.json"),
                 "--discovery-date", WHEN])
    assert code == 2
    payload = json.loads((tmp_path / "out" /
                          "esma_annex2_publication_discovery.json").read_text())
    assert payload["discovery_status"] == DISCOVERY_FAILED


def test_the_cli_without_fixtures_or_live_reports_every_source_unchecked(
        tmp_path):
    """Never an empty sweep by default."""
    from regulatory_watch.discovery.cli import main
    code = main(["discover", "--out-dir", str(tmp_path / "out"),
                 "--state", str(tmp_path / "seen.json"),
                 "--discovery-date", WHEN])
    assert code == 2
    payload = json.loads((tmp_path / "out" /
                          "esma_annex2_publication_discovery.json").read_text())
    assert payload["discovery_status"] == DISCOVERY_FAILED
    assert all(o["outcome"] == SOURCE_FETCH_FAILED
               for o in payload["source_outcomes"])


def test_the_cli_refuses_an_unusable_source_config(tmp_path):
    from regulatory_watch.discovery.cli import main
    bad = tmp_path / "bad.yaml"
    bad.write_text("manifest: {}\n", encoding="utf-8")
    assert main(["--sources", str(bad), "discover",
                 "--out-dir", str(tmp_path / "out")]) == 3


# --------------------------------------------------------------------------- #
# Stage 1B boundary
# --------------------------------------------------------------------------- #

WATCHED = (
    "config/regulatory_watch/esma_annex2_sources.yaml",
    "config/regulatory_watch/esma_annex2_publication_sources.yaml",
    "config/system/fields_registry.yaml",
    "config/system/esma_code_order.yaml",
)


def test_a_discovery_run_modifies_no_configuration(tmp_path):
    import hashlib
    from regulatory_watch.discovery.cli import main
    before = {p: hashlib.sha256((_REPO / p).read_bytes()).hexdigest()
              for p in WATCHED}
    main(["discover", "--out-dir", str(tmp_path / "out"),
          "--fixture-dir", str(FIXTURE_DIR),
          "--state", str(tmp_path / "seen.json"), "--discovery-date", WHEN])
    after = {p: hashlib.sha256((_REPO / p).read_bytes()).hexdigest()
             for p in WATCHED}
    assert before == after
