#!/usr/bin/env python3
"""mi_agent_api/tests/test_copilot_actions.py

The Microsoft 365 Copilot actions (/v1/copilot/*):

  askTraktMi   - valid question via the deterministic MI path; live data source
                 required; synthetic fallback refused; malformed request;
                 unauthenticated request; classification + package telemetry.
  getArtifact  - ONE generic action over the artifact registry: investor_deck
                 and canonical_tape (behaviour-preserving migration) plus a
                 registry-added type; unknown type 404; absent; storage
                 unavailable; unauthenticated request; controlled download.

Functional tests run with TRAKT_COPILOT_AUTH_MODE=disabled (the documented
local-dev mode); the auth tests exercise the fail-closed entra default.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
from fastapi.testclient import TestClient

from mi_agent_api.app import app
from mi_agent_api import data_source

client = TestClient(app)

DECK_NAME = "investor_pack.pptx"
POINTER_NAME = "latest_investor_pack.json"


def _demo_csv() -> Path:
    hits = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))
    assert hits, "expected the bundled synthetic demo canonical CSV"
    return hits[0]


@pytest.fixture()
def copilot_auth_off(monkeypatch):
    monkeypatch.setenv("TRAKT_COPILOT_AUTH_MODE", "disabled")


@pytest.fixture()
def live_dataset(monkeypatch, copilot_auth_off):
    """A governed (explicitly configured) live dataset: the demo CSV supplied
    via MI_AGENT_DATA_CSV resolves as kind=explicit_csv, not synthetic_demo."""
    monkeypatch.setenv("MI_AGENT_DATA_CSV", str(_demo_csv()))
    monkeypatch.setenv("MI_AGENT_DATA_CACHE_TTL", "0")
    data_source.reset_cache()
    yield
    data_source.reset_cache()


@pytest.fixture()
def synthetic_dataset(monkeypatch, copilot_auth_off):
    """Default resolution (no env) → the synthetic demo fallback kind."""
    for var in ("MI_AGENT_DATA_CSV", "MI_AGENT_ANALYTICS_DATASET",
                "MI_AGENT_CENTRAL_TAPE", "MI_AGENT_ONBOARDING_OUTPUT_ROOT",
                "MI_AGENT_PLATFORM_URI", "MI_AGENT_PLATFORM_CANONICAL",
                "MI_AGENT_PLATFORM_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MI_AGENT_DATA_CACHE_TTL", "0")
    data_source.reset_cache()
    yield
    data_source.reset_cache()


@pytest.fixture()
def deck_root(tmp_path, monkeypatch, copilot_auth_off):
    cid = "client_001"
    root = tmp_path / "decks"
    deck = root / cid / "latest" / DECK_NAME
    deck.parent.mkdir(parents=True, exist_ok=True)
    deck.write_bytes(b"PK\x03\x04 fake pptx bytes")
    (root / cid / "latest" / POINTER_NAME).write_text(json.dumps({
        "client_id": cid, "reporting_period": "2026-06",
        "generated_at": "2026-06-15T09:00:00+00:00"}))
    monkeypatch.setenv("MI_AGENT_DECK_ROOT", str(root))
    monkeypatch.delenv("MI_AGENT_CLIENT_ID", raising=False)
    return root


@pytest.fixture()
def tape_dir(tmp_path, monkeypatch, copilot_auth_off):
    pdir = tmp_path / "platform"
    pdir.mkdir(parents=True)
    tape = pdir / "platform_canonical_typed.csv"
    tape.write_text("loan_identifier,current_outstanding_balance\nL1,100000\n")
    monkeypatch.setenv("MI_AGENT_PLATFORM_DIR", str(pdir))
    monkeypatch.delenv("MI_AGENT_PLATFORM_URI", raising=False)
    monkeypatch.delenv("MI_AGENT_PLATFORM_CANONICAL", raising=False)
    return tape


# --------------------------------------------------------------------------- #
# askTraktMi
# --------------------------------------------------------------------------- #
def test_mi_valid_question_uses_deterministic_path(live_dataset):
    r = client.post("/v1/copilot/mi/query",
                    json={"question": "What is the total current balance?"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True, body
    assert body["dataSourceKind"] == "explicit_csv"      # governed, not synthetic
    assert body["answer"]
    assert isinstance(body["supportingValues"], list) and body["supportingValues"]
    # No raw path leaks in the payload.
    assert str(_demo_csv().parent) not in json.dumps(body)


def test_mi_supporting_values_carry_kpis(live_dataset):
    r = client.post("/v1/copilot/mi/query",
                    json={"question": "What is the total current balance?"})
    kinds = {a["kind"] for a in r.json()["supportingValues"]}
    assert "kpi" in kinds or "table" in kinds


def test_mi_synthetic_fallback_is_refused(synthetic_dataset, monkeypatch):
    # The refusal now lives in the governed capability rather than in this
    # adapter, and applies in PRODUCTION mode. The suite default is ``test``
    # mode, which legitimately permits fixtures, so the production rule is
    # asserted explicitly here.
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "production")
    data_source.reset_cache()
    assert data_source.data_source_kind() == data_source.KIND_SYNTHETIC_DEMO
    r = client.post("/v1/copilot/mi/query", json={"question": "total balance?"})
    assert r.status_code == 503
    body = r.json()
    assert body["ok"] is False
    assert body["errorCode"] == "DATA_SOURCE_NOT_APPROVED"
    assert "synthetic" in body["error"].lower()


def test_mi_missing_data_is_a_clear_error(monkeypatch, copilot_auth_off):
    # No dataset resolves at all. Patched at the resolution seam because the
    # approval policy now decides on the resolution BASE, not the display kind —
    # that is what stopped MI_AGENT_DATA_CSV=<demo file> slipping through as
    # "explicit_csv".
    monkeypatch.setattr(data_source, "resolve_data_source", lambda: (None, "unavailable"))
    from mi_agent_api import datasets as datasets_mod
    monkeypatch.setattr(datasets_mod, "resolve_data_source", lambda: (None, "unavailable"))
    data_source.reset_cache()
    r = client.post("/v1/copilot/mi/query", json={"question": "total balance?"})
    assert r.status_code == 503
    body = r.json()
    assert body["ok"] is False
    assert body["errorCode"] == "DATA_SOURCE_UNAVAILABLE"
    # An infrastructure gap is retryable; an unapproved source is not.
    assert body["retryable"] is True


def test_mi_malformed_request_is_422(copilot_auth_off):
    r = client.post("/v1/copilot/mi/query", json={"portfolioId": "client_001"})
    assert r.status_code == 422
    r = client.post("/v1/copilot/mi/query", json={"question": ""})
    assert r.status_code == 422


def test_mi_unauthenticated_request_refused(monkeypatch):
    # Default mode (entra) unconfigured → fail closed 503; configured but no
    # bearer → 401; garbage bearer → 401. Never a data answer.
    for var in ("TRAKT_COPILOT_AUTH_MODE", "TRAKT_COPILOT_ENTRA_TENANT_ID",
                "TRAKT_COPILOT_ENTRA_AUDIENCE"):
        monkeypatch.delenv(var, raising=False)
    r = client.post("/v1/copilot/mi/query", json={"question": "balance?"})
    assert r.status_code == 503

    monkeypatch.setenv("TRAKT_COPILOT_ENTRA_TENANT_ID",
                       "00000000-0000-0000-0000-000000000000")
    monkeypatch.setenv("TRAKT_COPILOT_ENTRA_AUDIENCE", "api://trakt-test")
    r = client.post("/v1/copilot/mi/query", json={"question": "balance?"})
    assert r.status_code == 401

    r = client.post("/v1/copilot/mi/query", json={"question": "balance?"},
                    headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401


# --------------------------------------------------------------------------- #
# getArtifact — investor_deck (behaviour-preserving migration) + unknown type
# --------------------------------------------------------------------------- #
def _get_artifact(artifact_type: str):
    return client.get("/v1/copilot/artifacts/latest",
                      params={"artifactType": artifact_type})


def test_deck_latest_found_with_metadata_and_download(deck_root):
    r = _get_artifact("investor_deck")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["artifactType"] == "investor_deck"
    assert body["fileName"].endswith(".pptx")
    assert body["reportingPeriod"] == "2026-06"
    assert body["sizeBytes"] and body["sizeBytes"] > 0
    assert "token=" in body["downloadUrl"]
    # No storage path or account detail in the payload.
    assert str(deck_root) not in json.dumps(body)

    # Controlled download: the signed link serves the actual bytes.
    token = body["downloadUrl"].split("token=", 1)[1]
    d = client.get("/v1/copilot/artifacts/download", params={"token": token})
    assert d.status_code == 200
    assert d.content.startswith(b"PK\x03\x04")
    assert "presentationml" in d.headers["content-type"]


def test_deck_alias_resolves_to_same_artifact(deck_root):
    """The model may pass a natural phrase; the registry normalises it."""
    r = _get_artifact("Investor Deck")
    assert r.status_code == 200
    assert r.json()["artifactType"] == "investor_deck"


def test_deck_absent_is_404(tmp_path, monkeypatch, copilot_auth_off):
    monkeypatch.setenv("MI_AGENT_DECK_ROOT", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    r = _get_artifact("investor_deck")
    assert r.status_code == 404
    assert r.json()["ok"] is False


def test_deck_storage_unavailable_is_503(monkeypatch, copilot_auth_off):
    import mi_agent_api.decks as decks

    def _boom(*_a, **_k):
        raise RuntimeError("blob store down")

    monkeypatch.setattr(decks, "resolve_deck_local", _boom)
    r = _get_artifact("investor_deck")
    assert r.status_code == 503
    assert r.json()["ok"] is False


def test_artifact_unauthenticated_refused(monkeypatch):
    for var in ("TRAKT_COPILOT_AUTH_MODE",):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TRAKT_COPILOT_ENTRA_TENANT_ID",
                       "00000000-0000-0000-0000-000000000000")
    monkeypatch.setenv("TRAKT_COPILOT_ENTRA_AUDIENCE", "api://trakt-test")
    r = _get_artifact("investor_deck")
    assert r.status_code == 401


def test_unknown_artifact_type_is_404_listing_available_types(copilot_auth_off):
    r = _get_artifact("mystery_report")
    assert r.status_code == 404
    body = r.json()
    assert body["ok"] is False
    # The error names the real registered types so the model can relay them.
    assert "investor_deck" in body["error"] and "esma_xml" in body["error"]


# --------------------------------------------------------------------------- #
# getArtifact — canonical_tape (behaviour-preserving migration)
# --------------------------------------------------------------------------- #
def test_tape_latest_found_with_metadata_and_download(tape_dir):
    r = _get_artifact("canonical_tape")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["artifactType"] == "canonical_tape"
    assert body["fileName"].endswith(".csv")
    assert body["contentType"] == "text/csv"
    assert body["sizeBytes"] and body["sizeBytes"] > 0
    assert str(tape_dir.parent) not in json.dumps(body)

    token = body["downloadUrl"].split("token=", 1)[1]
    d = client.get("/v1/copilot/artifacts/download", params={"token": token})
    assert d.status_code == 200
    assert b"loan_identifier" in d.content


def test_tape_absent_is_404(monkeypatch, tmp_path, copilot_auth_off):
    monkeypatch.setenv("MI_AGENT_PLATFORM_DIR", str(tmp_path / "nothing"))
    monkeypatch.delenv("MI_AGENT_PLATFORM_URI", raising=False)
    monkeypatch.delenv("MI_AGENT_PLATFORM_CANONICAL", raising=False)
    monkeypatch.chdir(tmp_path)   # keep the out_platform/ convention from matching
    import mi_agent_api.copilot_artifacts as ca_art
    monkeypatch.setattr(ca_art, "_resolve_latest_tape", lambda *_a, **_k: None)
    r = _get_artifact("canonical_tape")
    assert r.status_code == 404
    assert r.json()["ok"] is False
    assert "canonical loan tape" in r.json()["error"]


def test_tape_storage_unavailable_is_503(monkeypatch, copilot_auth_off):
    import mi_agent_api.copilot_artifacts as ca_art

    def _boom(*_a, **_k):
        raise RuntimeError("blob store down")

    monkeypatch.setattr(ca_art, "_resolve_latest_tape", _boom)
    r = _get_artifact("canonical_tape")
    assert r.status_code == 503


# --------------------------------------------------------------------------- #
# getArtifact — a registry-added type ships WITHOUT any package change
# --------------------------------------------------------------------------- #
def test_new_registry_type_resolves_server_side(tmp_path, monkeypatch,
                                                 copilot_auth_off):
    """mapping_report is served purely from the server registry + a resolver;
    no manifest function or path exists for it."""
    for var in ("MI_AGENT_PLATFORM_URI", "MI_AGENT_PLATFORM_DIR",
                "MI_AGENT_CLIENT_ID"):
        monkeypatch.delenv(var, raising=False)
    report = tmp_path / "mapping.csv"
    report.write_text("source_field,canonical_field\nfoo,bar\n")
    monkeypatch.setenv("MI_AGENT_MAPPING_REPORT", str(report))
    r = _get_artifact("mapping_report")
    assert r.status_code == 200
    body = r.json()
    assert body["artifactType"] == "mapping_report"
    assert body["contentType"] == "text/csv"
    token = body["downloadUrl"].split("token=", 1)[1]
    d = client.get("/v1/copilot/artifacts/download", params={"token": token})
    assert d.status_code == 200
    assert b"canonical_field" in d.content


def test_new_registry_type_absent_is_404(monkeypatch, copilot_auth_off):
    for var in ("MI_AGENT_ESMA_XML", "MI_AGENT_PLATFORM_URI",
                "MI_AGENT_PLATFORM_DIR"):
        monkeypatch.delenv(var, raising=False)
    r = _get_artifact("esma_xml")
    assert r.status_code == 404
    assert "ESMA Annex 2 XML" in r.json()["error"]


# --------------------------------------------------------------------------- #
# Signed download tokens
# --------------------------------------------------------------------------- #
def test_download_token_tamper_and_expiry(monkeypatch, copilot_auth_off):
    from mi_agent_api.copilot_actions import make_download_token, verify_download_token

    token, _exp = make_download_token("deck", "client_001")
    assert verify_download_token(token) is not None

    tampered = token[:-4] + ("0000" if not token.endswith("0000") else "1111")
    assert verify_download_token(tampered) is None
    assert verify_download_token("garbage") is None
    assert verify_download_token("a.b") is None

    # Expired token → invalid → the download route returns 403.
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 7200)
    assert verify_download_token(token) is None
    r = client.get("/v1/copilot/artifacts/download", params={"token": token})
    assert r.status_code == 403
    assert r.json()["ok"] is False


# --------------------------------------------------------------------------- #
# Governed artifact retrieval — client isolation + credential hygiene
# --------------------------------------------------------------------------- #
def test_download_token_carries_no_credentials_or_paths(deck_root):
    """The token is an opaque, signed reference — never a path or a secret."""
    import base64

    r = _get_artifact("investor_deck")
    token = r.json()["downloadUrl"].split("token=", 1)[1]
    payload = token.split(".", 1)[0]
    decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode()

    assert set(json.loads(decoded)) == {"k", "c", "e"}   # kind, client, expiry
    for secret in ("AccountKey", "DefaultEndpointsProtocol", "SharedAccessSignature",
                   "blob.core.windows.net", str(deck_root)):
        assert secret.lower() not in decoded.lower()
        assert secret.lower() not in r.text.lower()


def test_download_is_scoped_to_the_signed_client(deck_root, monkeypatch):
    """A token signed for one client must not serve another client's deck: the
    client travels inside the signed payload, so it cannot be swapped."""
    from mi_agent_api.copilot_actions import make_download_token

    token, _exp = make_download_token("deck", "other_client")
    r = client.get("/v1/copilot/artifacts/download", params={"token": token})
    # deck_root only contains client_001, so the other client resolves to nothing.
    assert r.status_code == 404
    assert r.json()["ok"] is False


def test_download_token_kind_cannot_be_repurposed(deck_root, tape_dir):
    from mi_agent_api.copilot_actions import make_download_token

    token, _exp = make_download_token("something_else", "client_001")
    r = client.get("/v1/copilot/artifacts/download", params={"token": token})
    assert r.status_code == 403


def test_tape_refuses_a_client_mismatch(monkeypatch, tmp_path, copilot_auth_off):
    """Client isolation: a platform URI naming a different client than the
    deployment's is a misconfiguration and must fail closed, not serve a tape."""
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", "ERE")
    monkeypatch.setenv("MI_AGENT_PLATFORM_URI",
                       "blob://processed-v2/platform/OTHER/latest/"
                       "platform_canonical_typed.csv")
    r = _get_artifact("canonical_tape")
    assert r.status_code == 404
    assert r.json()["ok"] is False


# --------------------------------------------------------------------------- #
# askTraktMi — governed envelope parity fields + explicit truncation
# --------------------------------------------------------------------------- #
def test_mi_answer_carries_the_full_governed_envelope(live_dataset):
    r = client.post("/v1/copilot/mi/query",
                    json={"question": "Show balance by LTV bucket."})
    body = r.json()
    assert body["ok"] is True, body
    # The analytical fields the React channel gets, not a reduced view.
    assert body["querySpec"]["metric"]
    assert body["querySpec"]["dimension"] == "ltv_bucket"
    assert body["validation"] is not None
    assert body["selectedClient"]
    # A point-in-time question is never reported as run-scoped.
    assert body["selectedRun"] is None
    assert body["truncated"] is False and body["truncationNote"] is None


def test_mi_row_truncation_is_explicit_and_deterministic(live_dataset, monkeypatch):
    import mi_agent_api.copilot_actions as ca

    monkeypatch.setattr(ca, "_MAX_SUPPORTING_ROWS", 3)
    r = client.post("/v1/copilot/mi/query",
                    json={"question": "Show balance by collateral region."})
    body = r.json()
    table = next(a for a in body["supportingValues"] if a["kind"] in ("table", "chart"))
    assert table["truncated"] is True
    assert len(table["rows"]) == 3
    assert table["totalRows"] > 3
    assert body["truncated"] is True
    assert "first 3 of" in body["truncationNote"]
    # Truncation is a display cap only — it never re-ranks or re-aggregates.
    react = client.post("/mi/query",
                        json={"question": "Show balance by collateral region."}).json()
    react_rows = next(a for a in react["artifacts"]
                      if a["type"] in ("table", "chart"))["rows"]
    assert table["rows"] == react_rows[:3]
