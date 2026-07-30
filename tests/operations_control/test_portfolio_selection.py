"""Portfolio is a governed choice, not free text.

``portfolio_id`` is the source book — ``direct_001`` — and the source registry is
the authority on which books a client has, which datasets each delivers, and
whether the funded book must report under a regime. What is proven here:

  * the catalogue is read from that registry, not from a second one;
  * a client only ever sees its own books;
  * the API validates the submitted identifier server-side, so the browser's
    filtering is a convenience and never the control;
  * a book Trakt has never seen is onboarding, and has to be declared as such;
  * asset class is reported as what it actually is today — a platform-level
    setting, not something looked up per portfolio.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.blob_trigger_app.source_registry import SourceRecord
from operations_control import catalogue
from operations_control.api import app as app_module

from .conftest import OP_A, OP_ALL, OP_B, make_client_config, make_engine


def _h(token: str):
    return {"X-Operator-Token": token}


def _register(registry, client_id, portfolio_id, dataset="funded",
              frequency="monthly", regime=False, book_type=None):
    registry.upsert(SourceRecord(
        client_id=client_id, source_portfolio_id=portfolio_id, dataset=dataset,
        frequency=frequency, regime_required=regime, status="active",
        source_portfolio_type=book_type))


@pytest.fixture()
def api(store, source_registry, tmp_path):
    engine = make_engine(store, source_registry,
                         client_config=make_client_config(tmp_path))
    _register(source_registry, "client_a", "direct_001", regime=True,
              book_type="direct")
    _register(source_registry, "client_a", "direct_001", dataset="pipeline",
              frequency="weekly")
    _register(source_registry, "client_a", "acquired_002", book_type="acquired")
    _register(source_registry, "client_b", "direct_900", regime=True)
    app_module.set_engine(engine)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    yield {"client": client, "engine": engine, "registry": source_registry}
    app_module.set_engine(None)


def _body(**kw):
    body = {"client_id": "client_a", "portfolio_id": "direct_001",
            "reporting_date": "2026-06-30", "workflow_type": "mi_annex2"}
    body.update(kw)
    return body


class TestTheCatalogue:
    def test_it_comes_from_the_source_registry(self, api):
        r = api["client"].get("/ops/portfolios?client=client_a",
                              headers=_h(OP_A))
        assert r.status_code == 200, r.text
        ids = [p["portfolio_id"] for p in r.json()["portfolios"]]
        assert ids == ["acquired_002", "direct_001"]

    def test_a_book_reports_the_datasets_it_delivers(self, api):
        portfolios = {p["portfolio_id"]: p for p in
                      api["client"].get("/ops/portfolios?client=client_a",
                                        headers=_h(OP_A)).json()["portfolios"]}
        assert portfolios["direct_001"]["datasets"] == ["funded", "pipeline"]
        assert portfolios["acquired_002"]["datasets"] == ["funded"]

    def test_the_reporting_requirement_follows_the_funded_book(self, api):
        portfolios = {p["portfolio_id"]: p for p in
                      api["client"].get("/ops/portfolios?client=client_a",
                                        headers=_h(OP_A)).json()["portfolios"]}
        assert portfolios["direct_001"]["regime_required"] is True
        assert portfolios["direct_001"]["outcome"] == "mi_annex2"
        assert portfolios["acquired_002"]["regime_required"] is False
        assert portfolios["acquired_002"]["outcome"] == "mi"

    def test_the_book_type_is_reported_separately_from_the_identifier(self, api):
        portfolios = {p["portfolio_id"]: p for p in
                      api["client"].get("/ops/portfolios?client=client_a",
                                        headers=_h(OP_A)).json()["portfolios"]}
        assert portfolios["acquired_002"]["book_type"] == "acquired"
        assert portfolios["acquired_002"]["book_type_label"] == "Acquired book"

    def test_asset_class_is_its_own_fact(self, api):
        """It is a platform-level setting today, not a per-portfolio lookup —
        so it is reported alongside the portfolio, never as its identity."""
        portfolio = api["client"].get("/ops/portfolios?client=client_a",
                                      headers=_h(OP_A)).json()["portfolios"][0]
        assert portfolio["asset_id"] == "equity_release"
        assert portfolio["asset_label"] == "Equity Release"
        assert portfolio["portfolio_id"] != portfolio["asset_id"]

    def test_the_display_name_falls_back_to_the_identifier(self, api):
        """With no label overlay configured — the current state — the governed
        identifier IS the display name. Nothing is invented."""
        portfolio = api["client"].get("/ops/portfolios?client=client_a",
                                      headers=_h(OP_A)).json()["portfolios"][0]
        assert portfolio["display_name"] == portfolio["portfolio_id"]

    def test_a_label_overlay_is_used_when_one_exists(self, source_registry):
        described = catalogue.describe_portfolio(
            "direct_001",
            [SourceRecord(client_id="ERE", source_portfolio_id="direct_001",
                          dataset="funded", frequency="monthly")],
            {"source_portfolio_label": "ERE Direct Originations"})
        assert described["display_name"] == "ERE Direct Originations"
        assert described["portfolio_id"] == "direct_001"


class TestTenancy:
    def test_a_client_only_sees_its_own_books(self, api):
        ids = [p["portfolio_id"] for p in
               api["client"].get("/ops/portfolios?client=client_a",
                                 headers=_h(OP_A)).json()["portfolios"]]
        assert "direct_900" not in ids

    def test_another_clients_catalogue_is_not_readable(self, api):
        r = api["client"].get("/ops/portfolios?client=client_b",
                              headers=_h(OP_A))
        assert r.status_code == 404

    def test_a_cross_client_portfolio_is_refused_on_submission(self, api):
        """The browser's filtering is a convenience. This is the control."""
        r = api["client"].post("/ops/batches",
                               json=_body(portfolio_id="direct_900",
                                          workflow_type="mi"),
                               headers=_h(OP_A))
        assert r.status_code == 400
        assert r.json()["detail"]["errorCode"] == "OPS_PORTFOLIO_UNKNOWN"

    def test_an_operator_for_all_clients_still_cannot_mix_them(self, api):
        r = api["client"].post("/ops/batches",
                               json=_body(portfolio_id="direct_900",
                                          workflow_type="mi"),
                               headers=_h(OP_ALL))
        assert r.status_code == 400
        assert r.json()["detail"]["errorCode"] == "OPS_PORTFOLIO_UNKNOWN"
        # …but the same book under its OWN client is fine.
        ok = api["client"].post("/ops/batches",
                                json=_body(client_id="client_b",
                                           portfolio_id="direct_900",
                                           workflow_type="mi_annex2"),
                                headers=_h(OP_ALL))
        assert ok.status_code == 201, ok.text


class TestSubmission:
    def test_a_registered_portfolio_is_accepted(self, api):
        r = api["client"].post("/ops/batches", json=_body(), headers=_h(OP_A))
        assert r.status_code == 201, r.text
        assert r.json()["batch"]["portfolio_id"] == "direct_001"

    def test_a_portfolio_that_does_not_exist_is_refused(self, api):
        r = api["client"].post("/ops/batches",
                               json=_body(portfolio_id="direct_999",
                                          workflow_type="mi"),
                               headers=_h(OP_A))
        assert r.status_code == 400
        assert r.json()["detail"]["errorCode"] == "OPS_PORTFOLIO_UNKNOWN"

    def test_an_empty_portfolio_is_refused(self, api):
        r = api["client"].post("/ops/batches",
                               json=_body(portfolio_id="  ",
                                          workflow_type="mi"),
                               headers=_h(OP_A))
        assert r.status_code == 400
        assert r.json()["detail"]["errorCode"] == "OPS_PORTFOLIO_REQUIRED"

    def test_a_new_book_must_be_declared_as_new(self, api):
        """Onboarding a book Trakt has never seen stays possible — but only
        deliberately, so a typo cannot open a delivery under a book that does
        not exist."""
        refused = api["client"].post("/ops/batches",
                                     json=_body(portfolio_id="direct_777",
                                                workflow_type="mi"),
                                     headers=_h(OP_A))
        assert refused.status_code == 400

        allowed = api["client"].post("/ops/batches",
                                     json=_body(portfolio_id="direct_777",
                                                workflow_type="mi",
                                                new_portfolio=True),
                                     headers=_h(OP_A))
        assert allowed.status_code == 201, allowed.text

    def test_an_unregistered_book_prepares_management_information_only(self):
        described = catalogue.unregistered_portfolio("direct_777")
        assert described["registered"] is False
        assert described["outcome"] == "mi"
        assert described["regime_required"] is False


class TestTheAutomatedRouteIsUnaffected:
    def test_the_trigger_still_onboards_an_unknown_book(self, api):
        """The portfolio rule belongs to the browser's door. An arrival for a
        book nobody has registered yet is exactly how onboarding starts."""
        batch = api["engine"].create_batch(
            client_id="client_a", portfolio_id="never_seen_001",
            reporting_date="2026-06-30", workflow_type="mi",
            created_by="blob-trigger", auto_start_when_ready=True)
        assert batch["portfolio_id"] == "never_seen_001"
