"""Tests for the additive Annex-delivery routes on the Operator console.

Two things are being defended here:

1. the existing source-approval console is untouched — same routes, same gate;
2. the browser-supplied ``client_id`` is authorised server-side before it can
   become a tenant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from engine.annex_delivery_agent.contracts import DeliveryStatus  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
CI_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "annex2_projected_ci.csv"

TOKEN = "test-operator-token"
HEADERS = {"X-Operator-Token": TOKEN}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAKT_OPERATOR_TOKEN", TOKEN)
    monkeypatch.setenv("TRAKT_ANNEX_DELIVERY_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("TRAKT_ANNEX_DELIVERY_PUBLISH_ROOT", str(tmp_path / "published"))
    monkeypatch.delenv("TRAKT_OPERATOR_TENANTS", raising=False)

    import mi_agent_operator.operator_app as operator_app

    return TestClient(operator_app.app)


def _prepare_body(**overrides):
    body = {"client_id": "ERE", "annex_id": "ESMA_Annex2",
            "reporting_date": "2026-01-31", "projected_input": str(CI_FIXTURE)}
    body.update(overrides)
    return body


class TestExistingConsoleIsUnchanged:
    def test_original_routes_are_still_registered(self, client):
        paths = set(client.app.openapi()["paths"])
        for path in ("/api/queue", "/api/audit", "/api/item/{approval_id}",
                     "/api/item/{approval_id}/approve", "/api/item/{approval_id}/reject",
                     "/api/item/{approval_id}/edit", "/healthz"):
            assert path in paths

    def test_annex_delivery_routes_are_additive(self, client):
        paths = set(client.app.openapi()["paths"])
        assert "/api/annex-delivery/prepare" in paths
        assert "/api/annex-delivery/runs/{run_id}/approve" in paths


class TestOperatorGate:
    def test_routes_require_the_operator_token(self, client):
        assert client.get("/api/annex-delivery/reports").status_code == 401
        assert client.post("/api/annex-delivery/prepare",
                           json=_prepare_body()).status_code == 401

    def test_console_is_disabled_when_no_token_is_configured(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TRAKT_OPERATOR_TOKEN", raising=False)
        import mi_agent_operator.operator_app as operator_app

        disabled = TestClient(operator_app.app)
        response = disabled.get("/api/annex-delivery/reports", headers=HEADERS)
        assert response.status_code == 503

    def test_reports_lists_annex2_and_no_placeholders(self, client):
        response = client.get("/api/annex-delivery/reports", headers=HEADERS)
        assert response.status_code == 200
        ids = {r["annex_id"] for r in response.json()["reports"]}
        assert "ESMA_Annex2" in ids
        assert "ESMA_Annex3" not in ids
        assert "ESMA_Annex9" not in ids


class TestClientIdIsAuthorisedNotTrusted:
    def test_missing_client_id_is_rejected(self, client):
        response = client.post("/api/annex-delivery/prepare", headers=HEADERS,
                               json=_prepare_body(client_id=""))
        assert response.status_code == 400

    def test_allowlist_refuses_a_client_outside_it(self, client, monkeypatch):
        monkeypatch.setenv("TRAKT_OPERATOR_TENANTS", "ACME,GLOBEX")
        response = client.post("/api/annex-delivery/prepare", headers=HEADERS,
                               json=_prepare_body(client_id="ERE"))
        assert response.status_code == 403
        assert "not authorised" in response.json()["detail"]

    def test_allowlist_permits_a_listed_client(self, client, monkeypatch):
        monkeypatch.setenv("TRAKT_OPERATOR_TENANTS", "ERE")
        response = client.post("/api/annex-delivery/prepare", headers=HEADERS,
                               json=_prepare_body())
        assert response.status_code == 200

    def test_unknown_client_is_refused_when_tenancy_is_configured(
            self, client, monkeypatch, tmp_path):
        config = tmp_path / "tenancy.yaml"
        config.write_text("tenants:\n  ACME:\n    portfolios: [ACME]\n", encoding="utf-8")
        monkeypatch.setenv("TRAKT_TENANCY_CONFIG", str(config))
        response = client.post("/api/annex-delivery/prepare", headers=HEADERS,
                               json=_prepare_body(client_id="ERE"))
        assert response.status_code == 403
        assert response.json()["detail"] == "unknown client"

    def test_context_records_the_operator_as_the_actor(self, monkeypatch):
        from mi_agent_operator.annex_delivery_routes import operator_context

        monkeypatch.delenv("TRAKT_OPERATOR_TENANTS", raising=False)
        context = operator_context("ERE", "alice")
        assert context.tenant_id == "ERE"
        assert context.actor_id == "alice"
        assert context.actor_type == "service"


@pytest.mark.skipif(not CI_FIXTURE.is_file(), reason="Annex 2 CI fixture missing")
class TestDeliveryWorkflow:
    def test_prepare_returns_the_occ_workflow_view(self, client):
        response = client.post("/api/annex-delivery/prepare", headers=HEADERS,
                               json=_prepare_body())
        assert response.status_code == 200
        payload = response.json()
        assert payload["governance"]["capability"] == "annex_delivery.prepare"

        view = payload["view"]
        assert view["status"] in (DeliveryStatus.PREPARED,
                                  DeliveryStatus.PREPARED_WITH_WARNINGS)
        labels = [s["label"] for s in view["stages"]]
        assert "Annex 2 data prepared" in labels
        assert "Annex 2 XML created" in labels
        assert "XML passed schema validation" in labels
        assert "Ready for publication" in labels
        assert "does not confirm" in view["disclaimer"]

    def test_publish_is_refused_until_approved(self, client):
        prepared = client.post("/api/annex-delivery/prepare", headers=HEADERS,
                               json=_prepare_body()).json()["view"]
        run_id = prepared["run_id"]

        refused = client.post(f"/api/annex-delivery/runs/{run_id}/publish",
                              headers=HEADERS, json=_prepare_body())
        assert refused.status_code == 403
        assert "not been approved" in refused.json()["detail"]

        approved = client.post(f"/api/annex-delivery/runs/{run_id}/approve",
                               headers=HEADERS,
                               json={**_prepare_body(), "note": "reviewed"})
        assert approved.status_code == 200
        assert approved.json()["status"] == DeliveryStatus.APPROVED

        published = client.post(f"/api/annex-delivery/runs/{run_id}/publish",
                                headers=HEADERS, json=_prepare_body())
        assert published.status_code == 200
        assert published.json()["status"] == DeliveryStatus.PUBLISHED

    def test_rejection_requires_a_reason(self, client):
        prepared = client.post("/api/annex-delivery/prepare", headers=HEADERS,
                               json=_prepare_body()).json()["view"]
        response = client.post(f"/api/annex-delivery/runs/{prepared['run_id']}/reject",
                               headers=HEADERS, json={**_prepare_body(), "reason": "  "})
        assert response.status_code == 400

    def test_another_client_cannot_reach_the_run(self, client, monkeypatch):
        prepared = client.post("/api/annex-delivery/prepare", headers=HEADERS,
                               json=_prepare_body()).json()["view"]
        response = client.post(
            f"/api/annex-delivery/runs/{prepared['run_id']}/approve",
            headers=HEADERS, json=_prepare_body(client_id="OTHER"))
        assert response.status_code in (403, 404)

    def test_a_blocked_run_returns_the_operator_message(self, client, tmp_path):
        response = client.post("/api/annex-delivery/prepare", headers=HEADERS,
                               json=_prepare_body(projected_input=str(tmp_path / "no.csv")))
        assert response.status_code == 200
        view = response.json()["view"]
        assert view["status"] == DeliveryStatus.PREFLIGHT_BLOCKED
        assert "was not found" in view["blocking_errors"][0]["message"]
