"""An ITL code is a governed region, and the canonical derivation must know it.

MEASURED. On the consolidated platform book `canonical_region_reporting` is null
for all 11,035 rows and `region_mapping_method` is `unresolved` for all 11,035 —
the estate's own disclosure that the harmonisation ran and mapped nothing. The
consequence reaches the reader: "Total balance by region" refuses with
`canonical_region_reporting: dimension_no_values`, on a book whose regions are
sitting in the tape.

WHY. `SOURCE_FIELDS` takes the first POPULATED source column, which is
`geographic_region_obligor`, and that column holds ITL3 CODES — `TLC31`,
`TLH27`, `TLI43`. The taxonomy's vocabulary is NAMES, so every code fell to
`METHOD_UNRESOLVED`. The readable names sat in `collateral_geography`, third in
preference and never consulted.

THE FIX IS NOT A REORDERING. Preferring the name column would make this book
work and leave a code-only book exactly as broken, with the analytical meaning
of "region" still decided by which column a tape happened to fill.

THE FIX IS NOT A SECOND TABLE either. `TLC31 -> North East` written into the
taxonomy would be a parallel geography vocabulary maintained beside the ITL
ladder and free to drift from it. The ladder in `mi_agent.region_resolution` —
postcode prefix, ITL3, ITL2, ITL1, over `uk_itl_master_lookup_v2.csv` — is the
estate's authority on what an ITL code means, and it already resolves every one
of these. It was simply never asked.

So the canonicalisation layer DELEGATES: when its own exact and synonym lookups
fail, it asks the ladder owner which of the taxonomy's OWN governed values the
raw value denotes. No vocabulary is added anywhere. `codes_for` already applies
the alias table that makes "East of England" and the ITL spelling "East
(England)" the same region, which is why the one name that differs between the
two vocabularies needs no bridge written for it.

NO CYCLE. `region_resolution._canonical_matches` already delegates the NAMING
question to the taxonomy. The new call goes the other way and uses only the
ladder — `codes_for`, over the master lookup — so it never re-enters the
taxonomy, and the two owners keep one direction each.
"""

from __future__ import annotations

import pytest

#: ``(raw value, the governed taxonomy value it denotes)``. Every ITL1 region is
#: probed through a code drawn from the master lookup itself, so this cannot
#: pass by agreeing with a list someone typed here.
LADDER_CASES = (
    ("TLC31", "North East"),
    ("TLD13", "North West"),
    ("TLE11", "Yorkshire and The Humber"),
    ("TLF11", "East Midlands"),
    ("TLG11", "West Midlands"),
    ("TLH21", "East of England"),
    ("TLI33", "London"),
    ("TLJ12", "South East"),
    ("TLK30", "South West"),
    ("TLL31", "Wales"),
    ("TLM01", "Scotland"),
    ("TLN06", "Northern Ireland"),
)

#: Values the ladder must NOT resolve. `ND1` is in the book and is not an ITL
#: code; force-filling it would invent geography.
NOT_GOVERNED = ("ND1", "GBZZZ", "ZZ999", "unknown")

TAXONOMY_VALUES = [v for _code, v in LADDER_CASES]


class TestTheLadderOwnerAnswersEquivalence:
    @pytest.mark.parametrize("code,expected", LADDER_CASES)
    def test_a_code_denotes_exactly_one_governed_value(self, code, expected) -> None:
        from mi_agent import region_resolution as ladder

        hits = [v for v in TAXONOMY_VALUES
                if ladder.same_governed_region(code, v)]
        assert hits == [expected], f"{code!r} -> {hits}, expected [{expected!r}]"

    @pytest.mark.parametrize("value", NOT_GOVERNED)
    def test_an_ungoverned_value_denotes_nothing(self, value) -> None:
        from mi_agent import region_resolution as ladder

        assert not any(ladder.same_governed_region(value, v)
                       for v in TAXONOMY_VALUES), (
            f"{value!r} is not a governed region and must resolve to nothing")

    def test_a_name_still_matches_itself(self) -> None:
        """The ladder must not have become code-only."""
        from mi_agent import region_resolution as ladder

        for name in TAXONOMY_VALUES:
            assert ladder.same_governed_region(name, name), name


class TestTheCanonicalDerivationDelegates:
    @pytest.mark.parametrize("code,expected", LADDER_CASES)
    def test_a_code_resolves_to_its_governed_detail(self, code, expected) -> None:
        from engine.region_taxonomy import METHOD_UNRESOLVED, resolve_taxonomy

        taxonomy = resolve_taxonomy(None)
        assert taxonomy is not None, "no taxonomy configured"
        detail, method = taxonomy.resolve_detail(code)
        assert detail == expected, f"{code!r} -> {detail!r}"
        assert method != METHOD_UNRESOLVED

    @pytest.mark.parametrize("value", ("ND1", "GBZZZ", "ZZ999"))
    def test_an_ungoverned_value_stays_unresolved(self, value) -> None:
        from engine.region_taxonomy import METHOD_UNRESOLVED, resolve_taxonomy

        taxonomy = resolve_taxonomy(None)
        detail, method = taxonomy.resolve_detail(value)
        assert detail is None and method == METHOD_UNRESOLVED, (
            f"{value!r} was assigned {detail!r} — geography is never force-filled")

    def test_an_exact_name_keeps_its_existing_method(self) -> None:
        """Delegation is a LAST resort; it must not displace what already worked."""
        from engine.region_taxonomy import METHOD_EXACT, resolve_taxonomy

        taxonomy = resolve_taxonomy(None)
        assert taxonomy.resolve_detail("Scotland") == ("Scotland", METHOD_EXACT)

    def test_an_approved_synonym_keeps_its_existing_method(self) -> None:
        from engine.region_taxonomy import METHOD_SYNONYM, resolve_taxonomy

        taxonomy = resolve_taxonomy(None)
        assert taxonomy.resolve_detail("Greater London") == ("London", METHOD_SYNONYM)


class TestTheDerivationPopulatesTheCanonicalColumn:
    def test_a_frame_of_codes_is_harmonised(self) -> None:
        import pandas as pd

        from engine import region_taxonomy as RT

        frame = pd.DataFrame({"geographic_region_obligor":
                              ["TLC31", "TLI33", "TLM01", "ND1", None]})
        RT.apply(frame, RT.resolve_taxonomy(None))
        reporting = list(frame[RT.FIELD_REPORTING])
        assert reporting[:3] == ["North East", "London", "Scotland"]
        assert reporting[3] is None, "ND1 is not governed geography"
        assert reporting[4] is None


# --------------------------------------------------------------------------- #
# One region, three consumers, the same cells
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def platform():
    """The generated multi-period platform, or a skip.

    The demo platform is BUILT, not checked in (`python -m demo_platform.run_demo
    --generate --orchestrate`). Skipping when it is absent keeps this suite
    runnable everywhere while still pinning the invariant wherever the data
    exists — which is the only place the invariant can be measured at all.
    """
    import os

    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg

    if not (cfg.local_blob_root()).exists():
        pytest.skip("demo platform not generated in this checkout")
    env = cfg.mi_env(period_role="current")
    os.environ.update(env)
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    return cfg, env


CANONICAL = "canonical_region_reporting"
BALANCE = "current_outstanding_balance"


class TestOneRegionAcrossEveryConsumer:
    """THE INVARIANT this sprint exists for.

        ordinary "balance by region" at snapshot T
            == historical "balance by region" at snapshot T
            == dashboard region evolution cells at snapshot T

    Grand totals agreeing proves nothing here — two paths can agree on the book
    and disagree about every region inside it, which is exactly what they did:
    MI answered in ITL1 names while the dashboard published ITL3 CODES. So the
    comparison is cell for cell, on balance AND count.
    """

    def _cells(self, platform):
        import pandas as pd
        from fastapi.testclient import TestClient

        from mi_agent_api import evolution as evo
        from mi_agent_api.app import app

        cfg, env = platform
        client = TestClient(app)
        latest = max(f["run_id"] for f in evo.funded_frames(
            env["MI_AGENT_ONBOARDING_OUTPUT_ROOT"], env["MI_AGENT_CLIENT_ID"], None))

        envelope = client.post("/mi/query", json={
            "question": "Total balance by region",
            "portfolioId": cfg.CLIENT_ID}).json()
        assert envelope.get("ok"), envelope.get("answer")
        table = next(a for a in envelope["artifacts"] if a.get("type") == "table")
        serving = {str(r[CANONICAL]): (round(float(r[BALANCE + "_sum"]), 2),
                                       int(r["loan_count"])) for r in table["rows"]}

        frames = evo.funded_frames(env["MI_AGENT_ONBOARDING_OUTPUT_ROOT"],
                                   env["MI_AGENT_CLIENT_ID"], None)
        frame = next(f for f in frames if f["run_id"] == latest)["df"]
        key = frame[CANONICAL].fillna(evo.MISSING_BUCKET).astype(str)
        historical = {k: (round(float(g[BALANCE].sum()), 2), int(len(g)))
                      for k, g in frame.groupby(key)}

        series = client.get("/mi/evolution/funded",
                            params={"portfolioId": cfg.CLIENT_ID}).json()
        dashboard = {str(r["key"]): round(float(r["value"]), 2)
                     for r in (series.get("breakdowns") or {}).get("region") or []
                     if r.get("period") == latest[:7]}
        return serving, historical, dashboard

    def test_the_three_paths_publish_the_same_region_keys(self, platform) -> None:
        serving, historical, dashboard = self._cells(platform)
        assert set(serving) == set(historical) == set(dashboard), (
            "the three consumers do not even agree on which regions exist: "
            f"serving={sorted(serving)} historical={sorted(historical)} "
            f"dashboard={sorted(dashboard)}")

    def test_every_region_cell_agrees(self, platform) -> None:
        serving, historical, dashboard = self._cells(platform)
        mismatched = {
            region: {"serving": serving.get(region),
                     "historical": historical.get(region),
                     "dashboard": dashboard.get(region)}
            for region in sorted(set(serving) | set(historical) | set(dashboard))
            if not (serving.get(region, (None,))[0]
                    == historical.get(region, (None,))[0]
                    == dashboard.get(region))
            or serving.get(region, (None, None))[1] != historical.get(
                region, (None, None))[1]}
        assert not mismatched, mismatched

    def test_scotland_identifies_the_same_rows_everywhere(self, platform) -> None:
        """Scotland was the visible failure mode; it gets its own assertion."""
        from fastapi.testclient import TestClient

        from mi_agent_api.app import app

        cfg, _env = platform
        serving_cells, historical, dashboard = self._cells(platform)
        envelope = TestClient(app).post("/mi/query", json={
            "question": "What is the total balance in Scotland?",
            "portfolioId": cfg.CLIENT_ID}).json()
        assert envelope.get("ok"), envelope.get("answer")
        assert (envelope["spec"]["filters"] or {}).get(CANONICAL) == "Scotland", (
            "Scotland must bind through the canonical region field")
        reconciliation = envelope["reconciliation"]
        assert (round(float(reconciliation["balance_included"]), 2),
                int(reconciliation["records_included"])) == historical["Scotland"]
        assert round(float(reconciliation["balance_included"]), 2) == dashboard["Scotland"]

    def test_the_grouped_parts_sum_to_the_book(self, platform) -> None:
        serving, _historical, _dashboard = self._cells(platform)
        from fastapi.testclient import TestClient

        from mi_agent_api.app import app

        cfg, _env = platform
        whole = TestClient(app).post("/mi/query", json={
            "question": "What is the total balance?",
            "portfolioId": cfg.CLIENT_ID}).json()
        assert round(sum(v[0] for v in serving.values()), 2) == round(
            float(whole["reconciliation"]["balance_included"]), 2)


class TestAnEmptyCanonicalColumnIsNeverStamped:
    """The reverse failure this repair could have caused, and nearly did.

    `canonical_region_reporting` is the sole analytical region field and the axis
    owner takes it whenever it is PRESENT. Stamping it on a book whose geography
    the taxonomy cannot read — a tape carrying `GBZZZ`, or one never harmonised —
    would give the axis an all-null column to choose over a populated raw field,
    and "balance by region" would refuse `dimension_no_values` on a book that can
    answer it perfectly well. That is the defect this sprint removed, arriving by
    the other door.

    Decided at DERIVATION, once, so the analytical choice never moves back into
    the query path as a runtime populated-aware fallback.
    """

    def test_a_book_whose_geography_resolves_carries_the_column(self) -> None:
        import pandas as pd

        from mi_agent_api.funded_prep import prepare_funded_mi_dataset

        raw = pd.DataFrame({
            "loan_identifier": [1, 2, 3, 4],
            "current_outstanding_balance": [100000.0, 200000.0, 300000.0, 400000.0],
            "reporting_date": ["2025-12-31"] * 4,
            "geographic_region_obligor": ["TLC31", "TLI33", "TLM01", "TLL31"],
        })
        out, report = prepare_funded_mi_dataset(raw)
        assert "canonical_region_reporting" in out.columns
        assert out["canonical_region_reporting"].notna().sum() == 4
        assert report["region_harmonisation"]["applied"] is True

    def test_a_book_whose_geography_resolves_to_nothing_does_not(self) -> None:
        import pandas as pd

        from mi_agent_api.funded_prep import prepare_funded_mi_dataset

        raw = pd.DataFrame({
            "loan_identifier": [1, 2, 3, 4],
            "current_outstanding_balance": [100000.0, 200000.0, 300000.0, 400000.0],
            "reporting_date": ["2025-12-31"] * 4,
            "geographic_region_obligor": ["GBZZZ"] * 4,
        })
        out, report = prepare_funded_mi_dataset(raw)
        assert "canonical_region_reporting" not in out.columns, (
            "an all-null canonical column would be chosen over the populated raw "
            "field and refuse a question this book can answer")
        assert report["region_harmonisation"]["applied"] is False
        assert report["region_harmonisation"]["withheld"]

    def test_an_unharmonisable_book_still_reaches_its_raw_geography(self) -> None:
        """What actually happens today on a book the taxonomy cannot read.

        No canonical column is stamped (above), so the axis owner's preference
        order falls through to the populated raw field and the book answers
        "balance by region" out of its own ungoverned vocabulary.

        THAT FALLBACK IS KNOWN DEBT, NOT THE INTENDED END STATE. Retiring it —
        making `canonical_region_reporting` the only analytical answer and
        refusing where it cannot be derived — was implemented and MEASURED:
        127 new failures across 31 files and 85 movements in the 882-question
        census, every one of them a book or fixture that never went through the
        canonical derivation at all. Deleting the fallback without first running
        the derivation over those books does not remove the ungoverned reading;
        it relocates the failure. The patch is kept rather than shipped, and
        this assertion records the current contract honestly so the change is
        visible when it lands.
        """
        import pandas as pd

        from mi_agent import llm_query_parser as parser
        from mi_agent_api.funded_prep import prepare_funded_mi_dataset

        raw = pd.DataFrame({
            "loan_identifier": [1, 2],
            "current_outstanding_balance": [100000.0, 200000.0],
            "reporting_date": ["2025-12-31"] * 2,
            "geographic_region_obligor": ["GBZZZ", "GBZZZ"],
        })
        out, _report = prepare_funded_mi_dataset(raw)
        assert "canonical_region_reporting" not in out.columns
        chosen = next((f for f in parser._REGION_PREFERENCE if f in out.columns), None)
        assert chosen == "geographic_region_obligor"
        assert out[chosen].notna().all()
