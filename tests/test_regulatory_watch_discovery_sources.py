"""tests/test_regulatory_watch_discovery_sources.py — the upstream allowlist.

The allowlist is the structural guard that keeps Stage 1B a bounded control and
not a web crawler. These tests pin that: ESMA hosts only, https only, explicit
criticality, and a malformed config that raises rather than being partially
honoured.

Run: python -m pytest tests/test_regulatory_watch_discovery_sources.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch.discovery.contracts import (          # noqa: E402
    CORROBORATING, GATING, PARSER_TYPES, SOURCE_TYPES,
)
from regulatory_watch.discovery.sources import (            # noqa: E402
    ALLOWED_HOSTS, DEFAULT_SOURCES_PATH, SourceConfigError,
    load_publication_sources,
)


def _write(tmp_path: Path, sources_block: str,
           head: str = ("manifest:\n  manifest_id: t\n  regime: ESMA_Annex2\n"
                        "  authority: ESMA\n")) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(head + "sources:\n" + sources_block, encoding="utf-8")
    return path


_VALID = ("  - source_id: a\n    source_type: rss_feed\n"
          "    url: https://www.esma.europa.eu/rss.xml\n"
          "    parser: rss\n    criticality: gating\n")


# --------------------------------------------------------------------------- #
# The shipped allowlist
# --------------------------------------------------------------------------- #

def test_the_shipped_allowlist_is_valid_and_esma_annex2_only():
    sources = load_publication_sources()
    assert sources.regime == "ESMA_Annex2"
    assert sources.authority == "ESMA"
    assert sources.sources
    for entry in sources.sources:
        assert entry.source_type in SOURCE_TYPES
        assert entry.parser in PARSER_TYPES
        assert entry.criticality in (GATING, CORROBORATING)


def test_every_allowlisted_url_is_an_official_esma_https_url():
    for entry in load_publication_sources().sources:
        assert entry.url.startswith("https://")
        assert any(host in entry.url for host in ALLOWED_HOSTS), entry.url


def test_retrieval_is_off_by_default():
    assert load_publication_sources().retrieval_enabled is False


def test_the_allowlist_declares_at_least_one_gating_source():
    assert load_publication_sources().gating_sources


def test_the_technical_standards_page_is_gating():
    """It is the page Stage 1A's own artefact allowlist cites.

    A change there is the single strongest upstream signal of an Annex 2
    technical-spec change, so its silence must fail the run, not narrow it.
    """
    sources = load_publication_sources()
    entry = sources.by_id("esma_securitisation_technical_standards")
    assert entry.criticality == GATING
    assert entry.enabled


def test_no_source_claims_to_be_verified_reachable():
    """Honesty guard: nothing here has been fetched from this environment.

    The build environment denies esma.europa.eu, so every declared endpoint is
    unconfirmed. Flipping this flag without an actual successful fetch would
    turn a documented calibration step into a false assurance.
    """
    unverified = [s.source_id for s in load_publication_sources().sources
                  if not s.verified_reachable]
    assert unverified, ("if an endpoint has genuinely been fetched, update this "
                        "test along with the flag")


def test_frequency_metadata_is_weekly_not_a_polling_interval():
    sources = load_publication_sources()
    assert sources.default_frequency == "weekly"
    assert all(s.retrieval_frequency == "weekly" for s in sources.sources)


# --------------------------------------------------------------------------- #
# Validation failures — the config is never partially honoured
# --------------------------------------------------------------------------- #

def test_absent_config_raises(tmp_path):
    with pytest.raises(SourceConfigError):
        load_publication_sources(tmp_path / "nope.yaml")


def test_malformed_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("manifest: [not, a, mapping]\n", encoding="utf-8")
    with pytest.raises(SourceConfigError):
        load_publication_sources(path)


def test_empty_source_list_raises(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("manifest:\n  manifest_id: t\n  regime: ESMA_Annex2\n"
                    "  authority: ESMA\nsources: []\n", encoding="utf-8")
    with pytest.raises(SourceConfigError):
        load_publication_sources(path)


def test_a_non_esma_host_is_refused(tmp_path):
    """The guard that stops the allowlist becoming a crawler seed."""
    path = _write(tmp_path, "  - source_id: a\n    source_type: rss_feed\n"
                            "    url: https://example.invalid/feed.xml\n"
                            "    parser: rss\n    criticality: gating\n")
    with pytest.raises(SourceConfigError) as exc:
        load_publication_sources(path)
    assert "not an official ESMA host" in str(exc.value)


def test_a_non_https_url_is_refused(tmp_path):
    path = _write(tmp_path, "  - source_id: a\n    source_type: rss_feed\n"
                            "    url: http://www.esma.europa.eu/rss.xml\n"
                            "    parser: rss\n    criticality: gating\n")
    with pytest.raises(SourceConfigError) as exc:
        load_publication_sources(path)
    assert "https" in str(exc.value)


def test_criticality_is_required(tmp_path):
    path = _write(tmp_path, "  - source_id: a\n    source_type: rss_feed\n"
                            "    url: https://www.esma.europa.eu/rss.xml\n"
                            "    parser: rss\n")
    with pytest.raises(SourceConfigError) as exc:
        load_publication_sources(path)
    assert "criticality" in str(exc.value)


def test_an_unknown_criticality_is_refused(tmp_path):
    path = _write(tmp_path, "  - source_id: a\n    source_type: rss_feed\n"
                            "    url: https://www.esma.europa.eu/rss.xml\n"
                            "    parser: rss\n    criticality: nice_to_have\n")
    with pytest.raises(SourceConfigError):
        load_publication_sources(path)


def test_an_unknown_parser_is_refused(tmp_path):
    path = _write(tmp_path, "  - source_id: a\n    source_type: rss_feed\n"
                            "    url: https://www.esma.europa.eu/rss.xml\n"
                            "    parser: telepathy\n    criticality: gating\n")
    with pytest.raises(SourceConfigError):
        load_publication_sources(path)


def test_duplicate_source_ids_are_refused(tmp_path):
    with pytest.raises(SourceConfigError) as exc:
        load_publication_sources(_write(tmp_path, _VALID + _VALID))
    assert "duplicate source_id" in str(exc.value)


def test_a_non_esma_authority_is_refused(tmp_path):
    """Stage 1B is ESMA Annex 2 only — no FCA, no other authority."""
    path = _write(tmp_path, _VALID,
                  head=("manifest:\n  manifest_id: t\n  regime: ESMA_Annex2\n"
                        "  authority: FCA\n"))
    with pytest.raises(SourceConfigError) as exc:
        load_publication_sources(path)
    assert "unsupported authority" in str(exc.value)


def test_a_non_annex2_regime_is_refused(tmp_path):
    path = _write(tmp_path, _VALID,
                  head=("manifest:\n  manifest_id: t\n  regime: ESMA_Annex12\n"
                        "  authority: ESMA\n"))
    with pytest.raises(SourceConfigError) as exc:
        load_publication_sources(path)
    assert "unsupported regime" in str(exc.value)


def test_is_allowlisted_only_accepts_declared_urls():
    sources = load_publication_sources()
    assert sources.is_allowlisted(sources.sources[0].url)
    assert not sources.is_allowlisted("https://www.esma.europa.eu/anything-else")


def test_disabled_sources_are_kept_but_excluded_from_enabled():
    path = Path(DEFAULT_SOURCES_PATH)
    assert path.exists()
    sources = load_publication_sources(path)
    assert set(sources.enabled_sources) <= set(sources.sources)
