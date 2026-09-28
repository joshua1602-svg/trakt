"""The configuration in force, and the configuration that was deployed.

WHAT AN OPERATOR SAW

They asked why the Platform configuration screens showed no change to the
asset layer since July, when the geography work — which geography a book
reports on, the obligor's or the collateral's — had landed in September.

TWO SEPARATE CAUSES, BOTH REAL.

1. THE FILE WAS NOT IN THE LAYER.

   ``config/asset/mi_geography.yaml`` carries the per-asset-class decision and
   was not in ``LAYER_FILES[LAYER_ASSET]``. So it could not appear on the
   screens, be compared between versions, be validated or be rolled back, and
   changing it left no trace in the config audit. It is a governed decision
   living outside the thing that governs decisions.

   ``product_defaults_ERM.yaml`` HAS always been in the layer and carries
   equity release's own basis, which is why this surfaced as "nothing has
   changed" rather than as a failure.

2. A LAYER IS SEEDED FROM THE REPOSITORY ONCE AND NEVER AGAIN.

   ``ensure_seeded`` snapshots the files, activates version 1, and returns
   early ever after. That is the right governance — a configuration change
   becomes a version through draft, validate and activate, not by someone
   landing a file — but nothing said that a later deployment's edits were not
   in force. Every layer of the story was truthful and the whole was
   misleading.

   And it is not cosmetic. ``resolver`` reads pack CONTENT from the package,
   so a stale package is what a delivery is prepared against, while
   ``mi_agent.mi_geography`` loads its repo file directly. Two readers, two
   versions of one decision, no way to notice.

WHAT IS NOT DONE ABOUT IT

Automatic re-seeding. Whatever was last deployed silently becoming the
configuration in force is precisely what the version model exists to prevent.
Drift is REPORTED, and an operator can draft a new version from the deployed
files — which still has to be validated and activated.
"""

from __future__ import annotations

import pytest

from operations_control.configuration import admin_views
from operations_control.configuration.packages import (
    LAYER_ASSET,
    LAYER_FILES,
    ConfigPackageStore,
    STATUS_DRAFT,
)


@pytest.fixture()
def packages(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    from apps.blob_trigger_app.storage import Storage
    from operations_control.stores import OpsStore
    repo = tmp_path / "repo"
    for rel in LAYER_FILES[LAYER_ASSET]:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel}\nversion: 1\n", encoding="utf-8")
    return ConfigPackageStore(OpsStore(Storage(tmp_path / "blob")),
                              repo_root=repo)


def edit(packages, rel, text):
    (packages.repo / rel).write_text(text, encoding="utf-8")


GEOGRAPHY = "config/asset/mi_geography.yaml"


class TestTheGeographyDecisionIsGoverned:
    def test_it_is_part_of_the_asset_layer(self):
        """A governed decision has to live in the thing that governs."""
        assert GEOGRAPHY in LAYER_FILES[LAYER_ASSET]

    def test_the_real_file_carries_a_basis_per_asset_class(self):
        """Guarding what the layer now versions, not just that it is listed."""
        import yaml

        from operations_control.configuration.packages import REPO
        doc = yaml.safe_load((REPO / GEOGRAPHY).read_text(encoding="utf-8"))
        table = doc["primary_basis_by_asset_class"]
        assert table, "the table is what makes this a governed decision"
        assert set(table.values()) <= {"borrower", "collateral"}

    def test_it_is_versioned_with_the_rest_of_the_pack(self, packages):
        active = packages.ensure_seeded(LAYER_ASSET)
        assert GEOGRAPHY in active["files"]


class TestADeploymentThatIsNotInForceSaysSo:
    def test_a_fresh_seed_has_no_drift(self, packages):
        assert packages.drift(LAYER_ASSET)["differs"] is False

    def test_an_edited_file_is_reported(self, packages):
        """The whole defect: the edit simply was not in force, silently."""
        packages.ensure_seeded(LAYER_ASSET)
        edit(packages, GEOGRAPHY, "# changed\nversion: 2\n")
        drift = packages.drift(LAYER_ASSET)
        assert drift["differs"] is True
        assert [c["path"] for c in drift["changed"]] == [GEOGRAPHY]

    def test_it_says_which_version_is_actually_in_force(self, packages):
        packages.ensure_seeded(LAYER_ASSET)
        edit(packages, GEOGRAPHY, "# changed\n")
        assert packages.drift(LAYER_ASSET)["active_version"] == 1

    def test_a_file_the_deployment_has_lost_is_drift_too(self, packages):
        packages.ensure_seeded(LAYER_ASSET)
        (packages.repo / GEOGRAPHY).unlink()
        assert packages.drift(LAYER_ASSET)["removed"] == [GEOGRAPHY]

    def test_seeding_does_not_quietly_adopt_the_new_file(self, packages):
        """`ensure_seeded` returns early once a version is active, which is
        the behaviour that made this invisible. It stays."""
        packages.ensure_seeded(LAYER_ASSET)
        edit(packages, GEOGRAPHY, "# changed\nversion: 2\n")
        active = packages.ensure_seeded(LAYER_ASSET)
        assert active["version"] == 1
        assert "changed" not in active["files"][GEOGRAPHY]["content"]

    def test_the_sentence_says_what_is_true_not_that_a_screen_is_stale(
            self, packages):
        """"Two files differ" invites "the screen is out of date". What is
        true is that the deployment carries something Trakt is not using."""
        packages.ensure_seeded(LAYER_ASSET)
        edit(packages, GEOGRAPHY, "# changed\n")
        sentence = admin_views.describe_drift(packages, LAYER_ASSET)["sentence"]
        assert "not in force" in sentence
        assert "version 1" in sentence

    def test_the_layer_view_carries_it(self, packages):
        packages.ensure_seeded(LAYER_ASSET)
        edit(packages, GEOGRAPHY, "# changed\n")
        view = admin_views.describe_layer(packages, LAYER_ASSET)
        assert view["drift"]["differs"] is True

    def test_a_clean_deployment_says_so_plainly(self, packages):
        view = admin_views.describe_layer(packages, LAYER_ASSET)
        assert view["drift"]["differs"] is False
        assert "in force" in view["drift"]["sentence"]


class TestAdoptingADeploymentIsADecision:
    def test_it_produces_a_draft_rather_than_activating(self, packages):
        """Automatic adoption is the one thing the version model prevents."""
        packages.ensure_seeded(LAYER_ASSET)
        edit(packages, GEOGRAPHY, "# changed\nversion: 2\n")
        doc = packages.create_draft_from_deployment(LAYER_ASSET, by="Josh")
        assert doc["status"] == STATUS_DRAFT
        assert packages.active_version(LAYER_ASSET)["version"] == 1

    def test_the_draft_holds_what_was_deployed(self, packages):
        packages.ensure_seeded(LAYER_ASSET)
        edit(packages, GEOGRAPHY, "# changed\nversion: 2\n")
        doc = packages.create_draft_from_deployment(LAYER_ASSET, by="Josh")
        assert "changed" in doc["files"][GEOGRAPHY]["content"]

    def test_activating_it_clears_the_drift(self, packages):
        packages.ensure_seeded(LAYER_ASSET)
        edit(packages, GEOGRAPHY, "# changed\nversion: 2\n")
        doc = packages.create_draft_from_deployment(LAYER_ASSET, by="Josh")
        packages.validate_version(LAYER_ASSET, doc["version"])
        packages.activate_version(LAYER_ASSET, doc["version"], by="Josh")
        assert packages.drift(LAYER_ASSET)["differs"] is False

    def test_it_says_who_asked_for_it(self, packages):
        packages.ensure_seeded(LAYER_ASSET)
        edit(packages, GEOGRAPHY, "# changed\n")
        doc = packages.create_draft_from_deployment(LAYER_ASSET, by="Josh")
        assert doc["created_by"] == "Josh"
        assert doc["based_on_version"] == 1
