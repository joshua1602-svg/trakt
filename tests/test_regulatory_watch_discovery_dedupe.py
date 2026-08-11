"""tests/test_regulatory_watch_discovery_dedupe.py — identity and deduplication.

The weekly guarantee: running the same feed twice must not create a second
regulatory-development record. A watch that re-raises everything every week is
a watch nobody reads.

Three cases must stay distinguishable, and confusing any two of them is a
control failure:

* **NEW** — never seen;
* **UNCHANGED** — seen, and identical;
* **REVISED** — seen, but ESMA has corrected or replaced it.

Run: python -m pytest tests/test_regulatory_watch_discovery_dedupe.py
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
    SEEN_NEW, SEEN_REVISED, SEEN_UNCHANGED,
)
from regulatory_watch.discovery.state import (                # noqa: E402
    SeenStore, StateStoreError, canonical_url, content_fingerprint,
    derive_identity,
)
from tests.helpers import regulatory_watch_discovery as fx     # noqa: E402

WHEN = fx.DISCOVERY_DATE
LATER = "2026-02-23T00:00:00Z"


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #

def test_identity_is_deterministic():
    pub = fx.publication("Q&A", "https://www.esma.europa.eu/document/qa")
    assert derive_identity(pub) == derive_identity(pub)


def test_identity_is_source_independent():
    """The same publication reached from two sources is ONE development.

    ESMA publishes the auth.099 package in the publications feed AND on the
    technical-standards page, with a different identifier in each. Keying on
    the source, or on whichever identifier a source supplies, would raise the
    same publication once per source every week.
    """
    url = fx.STAGE1A_PACKAGE_URL
    from_feed = fx.publication("Technical standards", url,
                               source_id="esma_publications_feed",
                               official_identifier=url)
    from_page = fx.publication("Technical standards package", url,
                               source_id="esma_securitisation_technical_standards",
                               official_identifier="ESMA-TS-1.3.1")
    assert derive_identity(from_feed)[0] == derive_identity(from_page)[0]


def test_identity_ignores_tracking_parameters():
    plain = fx.publication("Q&A", "https://www.esma.europa.eu/document/qa")
    tracked = fx.publication(
        "Q&A", "https://www.esma.europa.eu/document/qa?utm_source=newsletter")
    assert derive_identity(plain)[0] == derive_identity(tracked)[0]


def test_identity_preserves_a_genuinely_different_document():
    a = fx.publication("A", "https://www.esma.europa.eu/document/a")
    b = fx.publication("A", "https://www.esma.europa.eu/document/b")
    assert derive_identity(a)[0] != derive_identity(b)[0]


def test_identity_falls_back_when_no_url_is_published():
    pub = fx.publication("Orphan", url="", official_identifier="ESMA-123")
    development_id, basis = derive_identity(pub)
    assert basis == "official_identifier"
    assert development_id.startswith("esma-a2-")


def test_identity_falls_back_to_title_and_date_as_a_last_resort():
    pub = fx.publication("Orphan", url="", publication_date="2026-02-11")
    assert derive_identity(pub)[1] == "derived"


def test_canonical_url_lowercases_host_but_preserves_path_case():
    assert canonical_url("HTTPS://WWW.ESMA.EUROPA.EU/Document/Qa") == \
        "https://www.esma.europa.eu/Document/Qa"


# --------------------------------------------------------------------------- #
# Fingerprint / revision
# --------------------------------------------------------------------------- #

def test_the_fingerprint_changes_when_the_title_changes():
    base = fx.publication("Q&A", "https://www.esma.europa.eu/document/qa")
    edited = fx.publication("Q&A (corrected)",
                            "https://www.esma.europa.eu/document/qa")
    assert content_fingerprint(base) != content_fingerprint(edited)


def test_the_fingerprint_changes_when_a_linked_artefact_changes():
    base = fx.publication("Package", "https://www.esma.europa.eu/document/p",
                          linked_urls=["https://www.esma.europa.eu/a.xsd"])
    updated = fx.publication("Package", "https://www.esma.europa.eu/document/p",
                             linked_urls=["https://www.esma.europa.eu/b.xsd"])
    assert content_fingerprint(base) != content_fingerprint(updated)


def test_the_fingerprint_is_insensitive_to_whitespace_reflow():
    a = fx.publication("Q&A  on   disclosure",
                       "https://www.esma.europa.eu/document/qa")
    b = fx.publication("Q&A on disclosure",
                       "https://www.esma.europa.eu/document/qa")
    assert content_fingerprint(a) == content_fingerprint(b)


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #

def test_a_first_sighting_is_new(tmp_path):
    store = SeenStore.load(tmp_path / "seen.json")
    identity = store.classify(fx.publication("Q&A"), WHEN)
    assert identity.seen_state == SEEN_NEW
    assert identity.first_seen == WHEN


def test_a_second_identical_sighting_is_unchanged(tmp_path):
    store = SeenStore.load(tmp_path / "seen.json")
    pub = fx.publication("Q&A")
    first = store.classify(pub, WHEN)
    store.record(pub, first, WHEN)
    second = store.classify(pub, LATER)
    assert second.seen_state == SEEN_UNCHANGED
    assert second.first_seen == WHEN


def test_a_corrected_publication_is_revised_not_new(tmp_path):
    store = SeenStore.load(tmp_path / "seen.json")
    original = fx.publication("Q&A on disclosure",
                              "https://www.esma.europa.eu/document/qa")
    identity = store.classify(original, WHEN)
    store.record(original, identity, WHEN)

    corrected = fx.publication("Q&A on disclosure (corrected)",
                               "https://www.esma.europa.eu/document/qa")
    revised = store.classify(corrected, LATER)
    assert revised.seen_state == SEEN_REVISED
    assert revised.development_id == identity.development_id
    assert revised.previous_fingerprint == identity.fingerprint


def test_classify_does_not_mutate_the_store(tmp_path):
    """Read-only: a run that dies midway must not half-remember a publication."""
    store = SeenStore.load(tmp_path / "seen.json")
    store.classify(fx.publication("Q&A"), WHEN)
    assert len(store) == 0


def test_recording_is_idempotent_for_an_unchanged_publication(tmp_path):
    store = SeenStore.load(tmp_path / "seen.json")
    pub = fx.publication("Q&A")
    for when in (WHEN, LATER, LATER):
        store.record(pub, store.classify(pub, when), when)
    assert len(store) == 1
    record = next(iter(store.records.values()))
    assert record.revision_count == 0
    assert record.first_seen == WHEN
    assert record.last_seen == LATER


def test_a_revision_increments_the_revision_count(tmp_path):
    store = SeenStore.load(tmp_path / "seen.json")
    url = "https://www.esma.europa.eu/document/qa"
    first = fx.publication("Q&A", url)
    store.record(first, store.classify(first, WHEN), WHEN)
    second = fx.publication("Q&A v2", url)
    store.record(second, store.classify(second, LATER), LATER)
    record = next(iter(store.records.values()))
    assert record.revision_count == 1
    assert record.title == "Q&A v2"


def test_the_store_round_trips_through_disk(tmp_path):
    path = tmp_path / "seen.json"
    store = SeenStore.load(path)
    pub = fx.publication("Q&A")
    store.record(pub, store.classify(pub, WHEN), WHEN)
    store.save()

    reloaded = SeenStore.load(path)
    assert len(reloaded) == 1
    assert reloaded.classify(pub, LATER).seen_state == SEEN_UNCHANGED


def test_the_saved_store_is_stable_and_sorted(tmp_path):
    path = tmp_path / "seen.json"
    store = SeenStore.load(path)
    for title in ("C", "A", "B"):
        pub = fx.publication(title, f"https://www.esma.europa.eu/document/{title}")
        store.record(pub, store.classify(pub, WHEN), WHEN)
    first = store.save().read_text(encoding="utf-8")
    second = SeenStore.load(path)
    second.save()
    assert path.read_text(encoding="utf-8") == first
    payload = json.loads(first)
    ids = [r["development_id"] for r in payload["records"]]
    assert ids == sorted(ids)


def test_an_unreadable_store_refuses_rather_than_looking_empty(tmp_path):
    """Treating a corrupt store as empty would re-raise every publication."""
    path = tmp_path / "seen.json"
    path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(StateStoreError):
        SeenStore.load(path)


def test_an_absent_store_is_simply_empty(tmp_path):
    assert len(SeenStore.load(tmp_path / "never-written.json")) == 0


def test_a_new_source_carrying_a_known_publication_is_not_a_revision(tmp_path):
    """Two sources describe the same publication differently.

    The feed carries an editorial summary; the listing page carries a link
    label. Comparing fingerprints across sources would flip a publication
    between REVISED and UNCHANGED depending on which source was scanned first.
    Fingerprints are therefore kept per source, and a new vantage point on a
    known publication is UNCHANGED — not new, not revised.
    """
    store = SeenStore.load(tmp_path / "seen.json")
    url = fx.STAGE1A_PACKAGE_URL
    from_feed = fx.publication("Technical standards package", url,
                               source_id="esma_publications_feed",
                               summary="Editorial summary from the feed.")
    store.record(from_feed, store.classify(from_feed, WHEN), WHEN)

    from_page = fx.publication("Securitisation reporting technical standards",
                               url,
                               source_id="esma_securitisation_technical_standards",
                               summary="")
    identity = store.classify(from_page, LATER)
    assert identity.seen_state == SEEN_UNCHANGED
    assert identity.previous_fingerprint is None


def test_a_revision_is_detected_per_source(tmp_path):
    store = SeenStore.load(tmp_path / "seen.json")
    url = "https://www.esma.europa.eu/document/qa"
    first = fx.publication("Q&A", url, source_id="feed", summary="v1")
    store.record(first, store.classify(first, WHEN), WHEN)

    same_source_changed = fx.publication("Q&A (corrected)", url,
                                         source_id="feed", summary="v2")
    assert store.classify(same_source_changed, LATER).seen_state == SEEN_REVISED

    other_source = fx.publication("Q&A", url, source_id="page", summary="x")
    assert store.classify(other_source, LATER).seen_state == SEEN_UNCHANGED


def test_per_source_fingerprints_survive_a_round_trip(tmp_path):
    path = tmp_path / "seen.json"
    store = SeenStore.load(path)
    pub = fx.publication("Q&A", "https://www.esma.europa.eu/document/qa",
                         source_id="feed")
    store.record(pub, store.classify(pub, WHEN), WHEN)
    store.save()
    record = next(iter(SeenStore.load(path).records.values()))
    assert set(record.fingerprints) == {"feed"}
