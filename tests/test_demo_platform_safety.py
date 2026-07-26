"""Tests for the demonstration's data-safety gate.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

The safety scan is the thing standing between this repository and a published
video that leaks a live client. These tests check that it actually catches what it
claims to, that its prohibited list is derived from the production configs rather
than hand-maintained, and that the built demonstration passes it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_platform import config as cfg      # noqa: E402
from demo_platform import safety             # noqa: E402


# --------------------------------------------------------------------------- #
# The prohibited list
# --------------------------------------------------------------------------- #
def test_prohibited_strings_include_the_configured_list():
    banned = safety.prohibited_strings()
    for needle in cfg.PROHIBITED_STRINGS:
        assert needle in banned


def test_prohibited_strings_are_derived_from_the_production_config():
    """The gate must not depend on someone remembering to add a client name.

    Reading identifying values live out of `config/client/*` means whatever the
    live config calls the client is automatically forbidden here.
    """
    derived = safety.production_identifiers()
    assert derived, (
        "no identifying values were read from the production client configs — "
        "the gate has lost its automatic protection"
    )
    # The live client's display name and originator name must both be in there.
    assert any("ERE" in value for value in derived), derived


def test_the_derived_list_excludes_generic_values():
    derived = safety.production_identifiers()
    for value in derived:
        assert value.lower() not in safety._GENERIC_VALUES
        assert len(value) >= 4


# --------------------------------------------------------------------------- #
# Detection — the scan must actually catch things
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "content,expected_rule",
    [
        ("client: ERE Funding Limited", "prohibited_client_string"),
        ("contact: operations@example-lender.co.uk", "email_address"),
        ("tenantId: 7f3d1a2b-4c5e-4f6a-8b9c-0d1e2f3a4b5c", "guid_or_tenant_id"),
        ("uri: https://traktdata.blob.core.windows.net/processed", "azure_storage_endpoint"),
        ("api: https://trakt-mi-api.azurewebsites.net/mi/query", "azure_web_host"),
        ("conn: DefaultEndpointsProtocol=https;AccountName=x", "storage_connection_string"),
        ("url: https://x/file.csv?sv=2024-01-01&sig=abcDEF123456", "sas_signed_url"),
        ("header: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9abcdef", "bearer_token"),
        ("key: -----BEGIN PRIVATE KEY-----", "private_key"),
        (
            "link: https://api/v1/copilot/artifacts/download?token=AbC123dEf456GhI",
            "download_token",
        ),
    ],
)
def test_the_scan_detects_prohibited_content(tmp_path, content, expected_rule):
    probe = tmp_path / "probe.json"
    probe.write_text(content, encoding="utf-8")
    report = safety.scan(extra_paths=[probe])
    rules = {f["rule"] for f in report["findings"] if f["path"].endswith("probe.json")}
    assert expected_rule in rules, (
        f"the scan did not flag {expected_rule} in {content!r}; found {rules}"
    )


def test_the_scan_does_not_flag_clean_synthetic_content(tmp_path):
    probe = tmp_path / "clean.json"
    probe.write_text(
        json.dumps({
            "synthetic_notice": cfg.SYNTHETIC_BANNER,
            "client": cfg.CLIENT_DISPLAY_NAME,
            "portfolio": cfg.PORTFOLIO_A.display_id,
            "region": cfg.PRIMARY_MOVEMENT_REGION,
            "balance": 1_964_886_258.21,
            "downloadUrl": "https://<trakt-api>/v1/copilot/artifacts/download?token=<redacted>",
        }),
        encoding="utf-8",
    )
    report = safety.scan(extra_paths=[probe])
    findings = [f for f in report["findings"] if f["path"].endswith("clean.json")]
    assert findings == [], findings


def test_a_word_boundary_prevents_a_false_positive(tmp_path):
    """"ERE" must not fire inside an ordinary word such as "WHERE" or "here"."""
    probe = tmp_path / "words.txt"
    probe.write_text("WHERE the balance is here, therefore severe.", encoding="utf-8")
    report = safety.scan(extra_paths=[probe])
    findings = [f for f in report["findings"] if f["path"].endswith("words.txt")]
    assert findings == [], findings


def test_the_demo_client_name_is_not_a_real_market_participant():
    """A sanity check on the fictional identity itself."""
    name = cfg.CLIENT_DISPLAY_NAME.lower()
    for needle in cfg.PROHIBITED_STRINGS:
        assert needle.lower() not in name
    assert "alderbridge" in name


# --------------------------------------------------------------------------- #
# The built demonstration must pass
# --------------------------------------------------------------------------- #
def test_the_built_demonstration_passes_the_scan():
    if not (cfg.export_dir() / "demo_metrics.json").exists():
        pytest.skip(
            "the demonstration has not been built — run "
            "`python -m demo_platform.run_demo --all`")
    report = safety.scan()
    assert report["ok"], json.dumps(report["findings"][:10], indent=2)
    assert report["filesScanned"] > 10


def test_every_generated_source_file_is_marked_synthetic():
    if not cfg.source_dir().exists():
        pytest.skip("the demonstration has not been generated")
    files = sorted(cfg.source_dir().rglob("*.csv"))
    assert files, "no generated source files found"
    for path in files:
        assert "SYNTHETIC" in path.name.upper(), path.name
        head = path.read_text(encoding="utf-8").splitlines()[:2]
        assert any(cfg.SYNTHETIC_SHORT in line for line in head), path.name


def test_the_manifest_carries_the_synthetic_notice():
    manifest = cfg.export_dir() / "demo_manifest.json"
    if not manifest.exists():
        pytest.skip("the demonstration has not been built")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["synthetic_notice"] == cfg.SYNTHETIC_BANNER
