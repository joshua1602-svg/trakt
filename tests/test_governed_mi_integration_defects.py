#!/usr/bin/env python3
"""tests/test_governed_mi_integration_defects.py

Integration defects found by manual validation of the local MI Agent against the
combined ERE platform, and the invariants that keep them fixed.

Each class pins one defect at the layer that actually caused it:

  1. CLIENT vs PORTFOLIO   the client index emitted one entry per
                           ``source_portfolio_id``, so ``direct_001`` appeared in
                           the tenant selector and "Client = direct_001,
                           Portfolio = Total" was a reachable, contradictory state
  2. borrower age          a column-level "already populated?" test on the
                           COMBINED tape let the direct book's explicit age
                           suppress the derivation for every other book
  3. region                the same column-level test on the group-dimension
                           alias, so an acquired tape naming region under a
                           different canonical field reported "not supplied"
  4. region taxonomy       "South West" and "south-west" consolidating as two
                           separate categories
  5. pipeline              eligibility, and honest attribution when the extract
                           carries no source-portfolio provenance
  6. chat                  the governed query route answering 200 with scope and
                           coverage attached

Run: python -m pytest tests/test_governed_mi_integration_defects.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine import region_taxonomy as rt  # noqa: E402
from mi_agent_api.funded_prep import augment_platform_canonical_dimensions  # noqa: E402

#: The ERE shape: one direct book that supplies an explicit age and its region
#: under the obligor field, one acquired book that supplies borrower DOBs and its
#: region under the collateral field in a different casing.
DIRECT_ROWS, ACQUIRED_ROWS = 73, 885
REPORTING_DATE = "2025-11-30"


def _platform_frame(direct: int = DIRECT_ROWS, acquired: int = ACQUIRED_ROWS,
                    *, acquired_region="south-west", direct_region="South West"):
    rows = []
    for i in range(direct):
        rows.append({
            "loan_identifier": f"direct_001-{i}", "source_portfolio_id": "direct_001",
            "source_portfolio_type": "direct", "source_portfolio_label": "direct_001",
            "portfolio_cohort": "direct_001", "client_id": "ERE",
            "reporting_date": REPORTING_DATE,
            "current_outstanding_balance": 100000.0 + i,
            "geographic_region_obligor": direct_region,
            "youngest_borrower_age": 70 + (i % 10),
        })
    for i in range(acquired):
        rows.append({
            "loan_identifier": f"acquired_001-{i}", "source_portfolio_id": "acquired_001",
            "source_portfolio_type": "acquired", "source_portfolio_label": "acquired_001",
            "portfolio_cohort": "acquired_001", "client_id": "ERE",
            "reporting_date": REPORTING_DATE,
            "current_outstanding_balance": 200000.0 + i,
            "geographic_region_collateral": acquired_region,
            # Borrower 1 born 1950, borrower 2 (younger) born 1955 on every third row.
            "borrower_1_DOB": "1950-01-01",
            "borrower_2_DOB": "1955-06-01" if i % 3 == 0 else "",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Borrower age — one derivation, every funded portfolio
# --------------------------------------------------------------------------- #
class TestBorrowerAgeDerivation(unittest.TestCase):
    """The derivation is generic: a book gets an age because it supplied a DOB and
    a reporting date, never because of what kind of book it is."""

    def test_acquired_derives_age_even_when_direct_supplies_one(self):
        """The defect: one book's explicit age suppressed every other book's."""
        out, derived = augment_platform_canonical_dimensions(_platform_frame())
        self.assertIn("youngest_borrower_age", derived)
        by_book = out.groupby("source_portfolio_id")["youngest_borrower_age"]
        self.assertEqual(int(by_book.apply(lambda s: s.notna().sum())["acquired_001"]),
                         ACQUIRED_ROWS)

    def test_an_explicit_age_is_never_overwritten(self):
        frame = _platform_frame()
        expected = frame.loc[frame.source_portfolio_id == "direct_001",
                             "youngest_borrower_age"].tolist()
        out, _ = augment_platform_canonical_dimensions(frame)
        self.assertEqual(
            out.loc[out.source_portfolio_id == "direct_001",
                    "youngest_borrower_age"].tolist(), expected)

    def test_age_is_measured_at_the_reporting_date_not_today(self):
        """A 1950-01-01 borrower is 75 at 2025-11-30 — and stays 75 next year."""
        out, _ = augment_platform_canonical_dimensions(
            _platform_frame(direct=0, acquired=3))
        ages = out["youngest_borrower_age"].tolist()
        # Row 0 has a 1955 second borrower (younger); rows 1-2 have only 1950.
        self.assertEqual(ages[0], 70)   # 1955-06-01 -> 70 at 2025-11-30
        self.assertEqual(ages[1], 75)   # 1950-01-01 -> 75

    def test_youngest_borrower_is_the_minimum_across_dob_fields(self):
        frame = pd.DataFrame([{
            "loan_identifier": "L1", "source_portfolio_id": "acquired_001",
            "reporting_date": REPORTING_DATE,
            "borrower_1_DOB": "1940-01-01", "borrower_2_DOB": "1960-01-01",
        }])
        out, _ = augment_platform_canonical_dimensions(frame)
        self.assertEqual(int(out["youngest_borrower_age"].iloc[0]), 65)

    def test_a_single_borrower_uses_borrower_one(self):
        frame = pd.DataFrame([{
            "loan_identifier": "L1", "source_portfolio_id": "acquired_001",
            "reporting_date": REPORTING_DATE,
            "borrower_1_DOB": "1948-03-01", "borrower_2_DOB": "",
        }])
        out, _ = augment_platform_canonical_dimensions(frame)
        self.assertEqual(int(out["youngest_borrower_age"].iloc[0]), 77)

    def test_missing_or_invalid_dob_stays_null_never_zero(self):
        frame = pd.DataFrame([
            {"loan_identifier": "L1", "source_portfolio_id": "acquired_001",
             "reporting_date": REPORTING_DATE, "borrower_1_DOB": "", "borrower_2_DOB": ""},
            {"loan_identifier": "L2", "source_portfolio_id": "acquired_001",
             "reporting_date": REPORTING_DATE, "borrower_1_DOB": "not-a-date",
             "borrower_2_DOB": ""},
            {"loan_identifier": "L3", "source_portfolio_id": "acquired_001",
             "reporting_date": REPORTING_DATE, "borrower_1_DOB": "1950-01-01",
             "borrower_2_DOB": ""},
        ])
        out, _ = augment_platform_canonical_dimensions(frame)
        ages = out["youngest_borrower_age"].tolist()
        self.assertTrue(pd.isna(ages[0]) and pd.isna(ages[1]))
        self.assertEqual(int(ages[2]), 75)

    def test_a_dob_after_the_reporting_date_is_never_a_negative_age(self):
        """A DOB later than the reporting date is a bad value, not a negative age.
        The row yields no age at all — the column is absent or null, never < 0."""
        frame = pd.DataFrame([{
            "loan_identifier": "L1", "source_portfolio_id": "acquired_001",
            "reporting_date": REPORTING_DATE, "borrower_1_DOB": "2030-01-01",
        }])
        out, derived = augment_platform_canonical_dimensions(frame)
        self.assertNotIn("youngest_borrower_age", derived)
        if "youngest_borrower_age" in out.columns:
            ages = out["youngest_borrower_age"].dropna()
            self.assertTrue(ages.empty or (ages >= 0).all())

    def test_a_bad_dob_does_not_suppress_a_good_one_on_another_row(self):
        frame = pd.DataFrame([
            {"loan_identifier": "L1", "source_portfolio_id": "acquired_001",
             "reporting_date": REPORTING_DATE, "borrower_1_DOB": "2030-01-01"},
            {"loan_identifier": "L2", "source_portfolio_id": "acquired_001",
             "reporting_date": REPORTING_DATE, "borrower_1_DOB": "1950-01-01"},
        ])
        out, _ = augment_platform_canonical_dimensions(frame)
        ages = out["youngest_borrower_age"].tolist()
        self.assertTrue(pd.isna(ages[0]))
        self.assertEqual(int(ages[1]), 75)

    def test_the_derivation_never_reads_portfolio_type(self):
        """Identical DOB inputs give identical ages whatever the book is called."""
        base = {"loan_identifier": "L1", "reporting_date": REPORTING_DATE,
                "borrower_1_DOB": "1950-01-01"}
        ages = []
        for pid, ptype in (("direct_002", "direct"), ("acquired_009", "acquired")):
            out, _ = augment_platform_canonical_dimensions(pd.DataFrame([
                {**base, "source_portfolio_id": pid, "source_portfolio_type": ptype}]))
            ages.append(int(out["youngest_borrower_age"].iloc[0]))
        self.assertEqual(ages[0], ages[1])


# --------------------------------------------------------------------------- #
# 3 + 4. Region: reaching canonical, then harmonising across books
# --------------------------------------------------------------------------- #
class TestRegionMappingAndTaxonomy(unittest.TestCase):

    def test_acquired_region_reaches_the_catalogue_primary(self):
        """The defect: the primary existed (direct only), so acquired was skipped."""
        out, derived = augment_platform_canonical_dimensions(_platform_frame())
        self.assertIn("geographic_region_obligor", derived)
        populated = out.groupby("source_portfolio_id")["geographic_region_obligor"].apply(
            lambda s: s.notna().sum())
        self.assertEqual(int(populated["acquired_001"]), ACQUIRED_ROWS)

    def test_direct_region_is_unchanged(self):
        out, _ = augment_platform_canonical_dimensions(_platform_frame())
        direct = out.loc[out.source_portfolio_id == "direct_001",
                         "geographic_region_obligor"].unique().tolist()
        self.assertEqual(direct, ["South West"])

    def test_case_and_punctuation_variants_normalise_deterministically(self):
        taxonomy = rt.resolve_taxonomy("ERE")
        for raw in ("South West", "south-west", "SOUTH WEST", "south_west", " South  West "):
            self.assertEqual(taxonomy.resolve_detail(raw)[0], "South West", raw)

    def test_an_approved_synonym_resolves_without_an_llm(self):
        taxonomy = rt.resolve_taxonomy("ERE")
        value, method = taxonomy.resolve_detail("Greater London")
        self.assertEqual((value, method), ("London", rt.METHOD_SYNONYM))

    def test_an_unknown_value_is_disclosed_never_forced(self):
        taxonomy = rt.resolve_taxonomy("ERE")
        value, method = taxonomy.resolve_detail("Atlantis")
        self.assertIsNone(value)
        self.assertEqual(method, rt.METHOD_UNRESOLVED)

    def test_consolidation_harmonises_the_two_books(self):
        """south-west and South West become ONE category, not two."""
        out, _ = augment_platform_canonical_dimensions(_platform_frame())
        self.assertEqual(sorted(out[rt.FIELD_REPORTING].dropna().unique()), ["South West"])
        self.assertEqual(int((out[rt.FIELD_REPORTING] == "South West").sum()),
                         DIRECT_ROWS + ACQUIRED_ROWS)

    def test_the_raw_source_value_stays_traceable(self):
        out, _ = augment_platform_canonical_dimensions(_platform_frame())
        acquired = out[out.source_portfolio_id == "acquired_001"]
        self.assertEqual(sorted(acquired[rt.FIELD_SOURCE_VALUE].unique()), ["south-west"])
        self.assertEqual(sorted(acquired[rt.FIELD_METHOD].unique()), [rt.METHOD_EXACT])

    def test_london_stays_london_at_the_detail_level(self):
        out, _ = augment_platform_canonical_dimensions(
            _platform_frame(direct=1, acquired=1, direct_region="London",
                            acquired_region="Greater London"))
        self.assertEqual(sorted(out[rt.FIELD_DETAIL].dropna().unique()), ["London"])

    def test_london_is_not_folded_into_south_east_by_default(self):
        """There is no unconditional rule anywhere that consolidates London."""
        taxonomy = rt.resolve_taxonomy("ERE")
        self.assertFalse(taxonomy.consolidates)
        self.assertEqual(taxonomy.to_reporting("London"), ("London", rt.RULE_IDENTITY))
        out, _ = augment_platform_canonical_dimensions(
            _platform_frame(direct=1, acquired=0, direct_region="London"))
        self.assertEqual(out[rt.FIELD_REPORTING].iloc[0], "London")

    def test_london_folds_into_south_east_only_under_an_approved_taxonomy(self):
        config = dict(rt.load_config())
        config["clients"] = {"CLIENT_X": {
            "taxonomy": "uk_itl1",
            "reporting_taxonomy": "uk_itl1_london_in_south_east"}}
        taxonomy = rt.resolve_taxonomy("CLIENT_X", config=config)
        self.assertTrue(taxonomy.consolidates)
        self.assertEqual(taxonomy.to_reporting("London"),
                         ("South East", rt.RULE_CONSOLIDATED))
        # Reversible: the detail level still says London.
        self.assertEqual(taxonomy.resolve_detail("Greater London")[0], "London")
        # And disclosable.
        self.assertEqual(taxonomy.describe()["consolidations"], {"London": "South East"})

    def test_unresolved_rows_are_counted_and_reported(self):
        frame = _platform_frame(direct=1, acquired=2, acquired_region="Atlantis")
        report = rt.apply(frame, rt.resolve_taxonomy("ERE"))
        self.assertEqual(report["rows_unresolved"], 2)
        self.assertEqual(report["unresolved_values"], {"Atlantis": 2})

    def test_no_taxonomy_configured_is_a_no_op(self):
        frame = _platform_frame(direct=1, acquired=1)
        self.assertEqual(rt.apply(frame, None), {})
        self.assertNotIn(rt.FIELD_DETAIL, frame.columns)


class TestRegionProposalsAreOnboardingOnly(unittest.TestCase):
    """The LLM proposes a mapping ONCE, at onboarding, for a value deterministic
    mapping could not resolve — and never while answering a question."""

    def test_a_proposal_is_structured_scored_and_attributed(self):
        taxonomy = rt.resolve_taxonomy("ERE")
        calls = []

        def fake_llm(prompt):
            calls.append(prompt)
            return ('[{"raw_value": "Greater Manchester", '
                    '"proposed_canonical_value": "North West", '
                    '"confidence": 0.97, "reason": "Metropolitan county of the '
                    'North West region"}]')

        proposals = rt.propose_unresolved(
            {"Greater Manchester": 12}, taxonomy,
            source_portfolio_id="acquired_001", llm_callable=fake_llm)
        self.assertEqual(len(calls), 1)
        item = proposals[0].to_dict()
        self.assertEqual(item["raw_value"], "Greater Manchester")
        self.assertEqual(item["proposed_canonical_value"], "North West")
        self.assertEqual(item["source_portfolio_id"], "acquired_001")
        self.assertEqual(item["row_count"], 12)
        self.assertEqual(item["status"], "pending")
        self.assertGreater(item["confidence"], 0.9)

    def test_a_proposal_outside_the_taxonomy_is_discarded(self):
        proposals = rt.propose_unresolved(
            {"Atlantis": 1}, rt.resolve_taxonomy("ERE"),
            llm_callable=lambda _p: '[{"raw_value": "Atlantis", '
                                    '"proposed_canonical_value": "Atlantis Region", '
                                    '"confidence": 0.99, "reason": "invented"}]')
        self.assertIsNone(proposals[0].proposed_canonical_value)

    def test_a_pending_proposal_changes_nothing(self):
        proposals = rt.propose_unresolved(
            {"Greater Manchester": 1}, rt.resolve_taxonomy("ERE"),
            llm_callable=lambda _p: '[{"raw_value": "Greater Manchester", '
                                    '"proposed_canonical_value": "North West", '
                                    '"confidence": 0.9, "reason": "x"}]')
        self.assertEqual(rt.approved_synonyms(proposals), {})

    def test_an_approved_proposal_becomes_a_deterministic_synonym(self):
        from dataclasses import replace
        proposals = rt.propose_unresolved(
            {"Greater Manchester": 1}, rt.resolve_taxonomy("ERE"),
            llm_callable=lambda _p: '[{"raw_value": "Greater Manchester", '
                                    '"proposed_canonical_value": "North West", '
                                    '"confidence": 0.9, "reason": "x"}]')
        approved = [replace(p, status="approved") for p in proposals]
        synonyms = rt.approved_synonyms(approved)
        self.assertEqual(synonyms, {"greater manchester": "North West"})

        # Persisted into the taxonomy, the value resolves deterministically —
        # the LLM is never consulted for it again.
        config = dict(rt.load_config())
        config["taxonomies"] = {**config["taxonomies"]}
        spec = dict(config["taxonomies"]["uk_itl1"])
        spec["synonyms"] = {**spec["synonyms"], "greater manchester": "North West"}
        config["taxonomies"]["uk_itl1"] = spec
        taxonomy = rt.resolve_taxonomy("ERE", config=config)
        self.assertEqual(taxonomy.resolve_detail("Greater Manchester"),
                         ("North West", rt.METHOD_SYNONYM))

    def test_without_an_llm_nothing_is_guessed(self):
        proposals = rt.propose_unresolved({"Atlantis": 3}, rt.resolve_taxonomy("ERE"))
        self.assertIsNone(proposals[0].proposed_canonical_value)
        self.assertEqual(proposals[0].method, "none")

    def test_the_runtime_region_path_never_calls_an_llm(self):
        """apply() is the whole runtime path, and it takes no llm_callable."""
        import inspect
        self.assertNotIn("llm", inspect.signature(rt.apply).parameters)
        source = inspect.getsource(rt.apply)
        self.assertNotIn("llm_callable", source)


# --------------------------------------------------------------------------- #
# 1 + 5 + 6. The live API surface
# --------------------------------------------------------------------------- #
class _ApiCase(unittest.TestCase):
    """Serves the ERE-shaped platform canonical (and optionally a pipeline root)."""

    PIPELINE = False

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient  # noqa: F401
        except Exception:  # pragma: no cover
            raise unittest.SkipTest("fastapi not installed")

    def setUp(self):
        from engine import platform_assembler as pa
        from fastapi.testclient import TestClient

        keys = ("MI_AGENT_AUTH_ENABLED", "MI_AGENT_PLATFORM_CANONICAL",
                "MI_AGENT_CLIENT_ID", "MI_AGENT_PIPELINE_ROOT")
        self._saved = {k: os.environ.get(k) for k in keys}
        os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
        os.environ["MI_AGENT_CLIENT_ID"] = "ERE"
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / pa.PLATFORM_CANONICAL_NAME
        _platform_frame().to_csv(path, index=False)
        os.environ["MI_AGENT_PLATFORM_CANONICAL"] = str(path)

        if self.PIPELINE:
            fixture = (_REPO / "tests" / "fixtures" / "client_001_mi_pack" / "pipeline"
                       / "2025-11-01" / "M2L_KFI_and_Pipeline_2025_11_01.csv")
            root = Path(self._tmp.name) / "pipeline_root" / "ERE" / "pipeline" / "2025-11-03"
            root.mkdir(parents=True, exist_ok=True)
            (root / "M2L_KFI_and_Pipeline_2025_11_03.csv").write_text(
                fixture.read_text(encoding="utf-8"), encoding="utf-8")
            os.environ["MI_AGENT_PIPELINE_ROOT"] = str(Path(self._tmp.name) / "pipeline_root")
        else:
            os.environ.pop("MI_AGENT_PIPELINE_ROOT", None)

        from mi_agent_api import data_source as ds
        self.ds = ds
        ds.reset_cache()
        from mi_agent_api.app import app
        self.client = TestClient(app)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.ds.reset_cache()
        self._tmp.cleanup()

    def snapshot(self, context=None):
        params = {"portfolioId": f"ERE/{REPORTING_DATE}"}
        if context:
            params["portfolioContext"] = context
        return self.client.get("/mi/snapshot", params=params).json()


class TestClientVersusPortfolioSelectors(_ApiCase):
    """The tenant axis and the portfolio axis are separate, and cannot conflict."""

    def test_the_client_index_lists_the_tenant_not_source_portfolios(self):
        body = self.client.get("/mi/snapshots").json()
        ids = [pf["client_id"] for pf in body["portfolios"]]
        self.assertEqual(ids, ["ERE"])

    def test_no_source_portfolio_id_can_appear_as_a_client(self):
        body = self.client.get("/mi/snapshots").json()
        clients = {pf["client_id"] for pf in body["portfolios"]}
        context = self.client.get("/mi/portfolio-context").json()
        portfolios = {p["portfolio_id"] for p in context["portfolios"]}
        self.assertTrue(portfolios)                       # there ARE portfolios
        self.assertEqual(clients & portfolios, set())     # and none is a client

    def test_the_portfolio_axis_carries_the_source_portfolios(self):
        body = self.client.get("/mi/portfolio-context").json()
        self.assertEqual([c["context_id"] for c in body["contexts"]],
                         ["total", "direct", "direct_001", "acquired", "acquired_001"])

    def test_the_reporting_date_is_a_run_not_a_portfolio(self):
        runs = self.client.get("/mi/snapshots").json()["portfolios"][0]["runs"]
        self.assertEqual([r["reporting_date"] for r in runs], [REPORTING_DATE])
        self.assertEqual(runs[0]["loan_count"], DIRECT_ROWS + ACQUIRED_ROWS)

    def test_client_ere_with_portfolio_total_is_the_whole_platform(self):
        self.assertEqual(self.snapshot("total")["loan_count"],
                         DIRECT_ROWS + ACQUIRED_ROWS)

    def test_client_ere_with_portfolio_acquired_001_is_only_that_book(self):
        self.assertEqual(self.snapshot("acquired_001")["loan_count"], ACQUIRED_ROWS)
        scope = self.snapshot("acquired_001")["portfolioScope"]["scope"]
        self.assertEqual(scope["portfolio_ids"], ["acquired_001"])

    def test_every_portfolio_context_reconciles(self):
        counts = {c: self.snapshot(c)["loan_count"]
                  for c in ("total", "direct", "acquired", "direct_001", "acquired_001")}
        self.assertEqual(counts["direct"], counts["direct_001"])
        self.assertEqual(counts["acquired"], counts["acquired_001"])
        self.assertEqual(counts["total"], counts["direct"] + counts["acquired"])

    def test_the_client_selection_cannot_pre_narrow_the_portfolio(self):
        """A run resolved for the client returns the whole client, and portfolio
        scope is applied only from the governed context."""
        self.assertEqual(self.snapshot()["loan_count"], DIRECT_ROWS + ACQUIRED_ROWS)


class TestAcquiredDimensionsOverTheApi(_ApiCase):

    def test_acquired_renders_borrower_age(self):
        strat = {s["key"]: s for s in self.snapshot("acquired_001")["stratifications"]}
        self.assertEqual(strat["age"]["availability"], "available")
        self.assertTrue(strat["age"]["bars"])

    def test_acquired_renders_region(self):
        strat = {s["key"]: s for s in self.snapshot("acquired_001")["stratifications"]}
        self.assertEqual(strat["region"]["availability"], "available")
        self.assertEqual([b["label"] for b in strat["region"]["bars"]], ["South West"])

    def test_total_geography_uses_one_harmonised_category(self):
        strat = {s["key"]: s for s in self.snapshot("total")["stratifications"]}
        labels = [b["label"] for b in strat["region"]["bars"]]
        self.assertEqual(labels, ["South West"])          # not "south-west" as well
        self.assertEqual(strat["region"]["bars"][0]["count"], DIRECT_ROWS + ACQUIRED_ROWS)

    def test_the_mi_agent_answers_borrower_age_for_every_scope(self):
        for context in ("total", "direct", "acquired", "direct_001", "acquired_001"):
            body = self.client.post("/mi/query", json={
                "question": "Show average borrower age.",
                "sourcePortfolioLens": context}).json()
            self.assertTrue(body["ok"], f"{context}: {body.get('error')}")
            self.assertTrue(body["portfolioCoverage"]["portfolios_used"], context)


class TestChatRouteAnswers(_ApiCase):
    """The governed query route the React client calls, end to end."""

    def test_the_query_route_exists_and_answers_200(self):
        response = self.client.post("/mi/query", json={
            "question": "What is the total current balance?"})
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.status_code, 404)

    def test_the_frontend_calls_the_route_the_api_serves(self):
        """The path in HttpAgentClient must be one FastAPI actually registers."""
        source = (_REPO / "frontend" / "mi-agent-ui" / "src" / "api"
                  / "HttpAgentClient.ts").read_text(encoding="utf-8")
        self.assertIn("/mi/query", source)
        from mi_agent_api.app import app
        paths = {getattr(r, "path", "") for r in app.routes}
        self.assertIn("/mi/query", paths)

    def test_every_client_path_is_proxied_by_the_dev_server(self):
        """A path the client calls but the dev proxy does not forward reaches the
        SPA instead of the API — the class of misconfiguration behind the 404."""
        import re
        source = (_REPO / "frontend" / "mi-agent-ui" / "src" / "api"
                  / "HttpAgentClient.ts").read_text(encoding="utf-8")
        called = {"/" + m.split("/")[1]
                  for m in re.findall(r"[`\"'](/(?:mi|me|health|v1)[^`\"'?]*)", source)}
        proxy = (_REPO / "frontend" / "mi-agent-ui" / "vite.config.ts").read_text(
            encoding="utf-8")
        forwarded = set(re.findall(r'"(/[a-z0-9]+)"', proxy.split("proxy:")[1]))
        self.assertTrue(called)
        self.assertEqual(called - forwarded, set(),
                         f"client calls {called - forwarded} but the dev proxy "
                         "does not forward them")

    def test_the_answer_carries_scope_and_field_coverage(self):
        body = self.client.post("/mi/query", json={
            "question": "What is the total current balance?",
            "sourcePortfolioLens": "total"}).json()
        self.assertTrue(body["ok"])
        coverage = body["portfolioCoverage"]
        self.assertEqual(sorted(coverage["portfolios_in_scope"]),
                         ["acquired_001", "direct_001"])
        self.assertIn("field_coverage", coverage)
        self.assertIn("scope", body["governance"])

    def test_queries_work_for_every_scope(self):
        for context in ("total", "direct", "acquired", "direct_001", "acquired_001"):
            body = self.client.post("/mi/query", json={
                "question": "What is the total current balance?",
                "sourcePortfolioLens": context}).json()
            self.assertTrue(body["ok"], context)
            self.assertEqual(body["portfolioCoverage"]["scope_requested"]["context_id"],
                             context)


class TestPipelineRegistration(_ApiCase):
    PIPELINE = True

    def test_the_pipeline_dataset_is_registered_and_loads(self):
        body = self.client.get("/mi/pipeline/snapshot",
                               params={"portfolioId": f"ERE/{REPORTING_DATE}"}).json()
        self.assertTrue(body.get("ok"), body.get("error"))
        self.assertGreater(body.get("pipelineRowCount", 0), 0)

    def test_pipeline_is_available_for_total_and_direct(self):
        contexts = {c["context_id"]: c
                    for c in self.client.get("/mi/portfolio-context").json()["contexts"]}
        for context in ("total", "direct", "direct_001"):
            self.assertTrue(contexts[context]["capabilities"]["pipeline"]["enabled"],
                            context)

    def test_pipeline_is_unavailable_for_a_non_originating_acquired_book(self):
        contexts = {c["context_id"]: c
                    for c in self.client.get("/mi/portfolio-context").json()["contexts"]}
        for context in ("acquired", "acquired_001"):
            state = contexts[context]["capabilities"]["pipeline"]
            self.assertFalse(state["enabled"], context)
            self.assertEqual(state["reason_code"], "NON_ORIGINATING")

    def test_total_keeps_its_context_and_discloses_participation(self):
        contexts = {c["context_id"]: c
                    for c in self.client.get("/mi/portfolio-context").json()["contexts"]}
        state = contexts["total"]["capabilities"]["pipeline"]
        self.assertEqual(state["contributing_portfolios"], ["direct_001"])
        self.assertEqual(state["excluded_portfolios"], ["acquired_001"])
        body = self.client.get("/mi/pipeline/snapshot",
                               params={"portfolioId": f"ERE/{REPORTING_DATE}",
                                       "portfolioContext": "total"}).json()
        self.assertEqual(body["portfolioScope"]["context_id"], "total")

    def test_an_unattributed_extract_says_so_and_is_not_allocated(self):
        """The fixture carries no source_portfolio_id, so the pipeline belongs to
        the originating GROUP — never split across books by an invented rule."""
        index = self.client.get("/mi/portfolio-context").json()
        self.assertFalse(index["pipeline_attributed"])
        state = {c["context_id"]: c for c in index["contexts"]}["direct"]["capabilities"]["pipeline"]
        self.assertFalse(state["attributed"])
        self.assertIn("no source-portfolio provenance", state["detail"])


if __name__ == "__main__":
    unittest.main()
