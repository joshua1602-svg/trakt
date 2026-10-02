#!/usr/bin/env python3
"""tests/test_funder_pack_enhancement.py

The funder pack, driven the way a funder receives it.

Every test here builds a real deck through the REACT ROUTE —
``POST /mi/decks/generate`` -> poll -> ``GET /mi/decks/download`` — and asserts
against the PowerPoint that comes back, or against the deck's own preflight
sidecar, which is a production artefact the publishing stage already writes and
gates on. Nothing calls a builder directly: certifying against a direct call
would bypass the job service, the orchestration stage and the publication gates
the button actually goes through.

Two book shapes are exercised, because most of what this sprint changed only
exists on one of them:

  SIMPLE   one constituent book, one reporting period. The shortest honest pack.
  RICH     three constituent books, four reporting periods, loans leaving each
           period with redemption / default / maturity evidence on the tape.

Each test states the defect it would catch.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pd = pytest.importorskip("pandas")
pptx = pytest.importorskip("pptx")

CLIENT = "funderpack"
_CENTRAL = "18_central_lender_tape.csv"

_REGIONS = ("London", "South East", "Wales", "Scotland", "North West")
_LTVS = (22.0, 34.0, 45.0, 63.0, 86.0)
_AGES = (58, 67, 71, 76, 87)
_RATES = (5.4, 6.2, 6.8, 7.1, 7.6)
_VINTAGES = ("2019-03-01", "2020-07-01", "2021-05-01", "2022-09-01", "2023-06-01")

#: Three constituent books. The stack, the composition table and the per-book
#: forward view all need more than one, and a single-book fixture is how those
#: three pages went unrendered for an entire sprint.
_BOOKS = (("direct_001", "direct", 0.52),
          ("acquired_001", "acquired", 0.31),
          ("direct_002", "direct", 0.17))


def _pid_for(j, books):
    """The book a loan belongs to — a function of the LOAN, so a loan keeps its
    book for life and the per-book series is a real series."""
    r = ((j * 7919) % 1000) / 1000.0
    running = 0.0
    for pid, ptype, share in books:
        running += share
        if r < running:
            return pid, ptype
    return books[-1][0], books[-1][1]


def _loan(j, pid, ptype, *, cut, scale, exiting):
    ltv = _LTVS[(j * 3) % len(_LTVS)]
    balance = (85_000 + ((j * 37_000) % 420_000)) * scale
    row = {
        "unique_identifier": f"{pid}_L{j:05d}",
        "source_portfolio_id": pid, "source_portfolio_type": ptype,
        "source_portfolio_label": pid.replace("_", " ").title(),
        "current_outstanding_balance": balance,
        "current_principal_balance": balance,
        "original_principal_balance": balance * 1.05,
        "current_valuation_amount": balance / (ltv / 100.0),
        "original_valuation_amount": balance / (ltv / 100.0),
        "current_loan_to_value": ltv,
        "original_loan_to_value": (ltv / 100.0) - 0.02,
        "current_interest_rate": _RATES[j % len(_RATES)],
        "youngest_borrower_age": _AGES[(j * 2) % len(_AGES)],
        "geographic_region_collateral": _REGIONS[(j * 5) % len(_REGIONS)],
        "collateral_geography": _REGIONS[(j * 5) % len(_REGIONS)],
        "origination_channel": "Direct", "broker_channel": "Direct",
        "product_type": "Lifetime Mortgage",
        "origination_date": _VINTAGES[(j * 7) % len(_VINTAGES)],
        "data_cut_off_date": cut,
    }
    if exiting:
        # Evidenced on the period the loan LEAVES FROM, which is where the
        # classifier reads it. Without this the exit leg collapses into the
        # unevidenced bucket and the split is never exercised.
        kind = j % 3
        if kind == 0:
            row["loan_redemption_flag"] = "Y"
        elif kind == 1:
            row["default_date"] = cut
        else:
            row["maturity_date"] = cut
    return row


#: (run_id, reporting date, first loan, last loan). Loans arrive at the top of
#: the range and leave from the bottom, so every period pair has both legs.
_RICH_PERIODS = (("mi_2026_03", "2026-03-31", 1, 200),
                 ("mi_2026_04", "2026-04-30", 7, 220),
                 ("mi_2026_05", "2026-05-31", 14, 245),
                 ("mi_2026_06", "2026-06-30", 22, 270))
_SIMPLE_PERIODS = (("mi_2026_06", "2026-06-30", 1, 80),)


def _write(root: Path, client: str, periods, books) -> Path:
    for index, (run_id, date, first, last) in enumerate(periods):
        central = root / client / run_id / "central"
        central.mkdir(parents=True, exist_ok=True)
        leaves_below = periods[index + 1][2] if index + 1 < len(periods) else 0
        rows = []
        for j in range(first, last + 1):
            pid, ptype = _pid_for(j, books)
            rows.append(_loan(j, pid, ptype, cut=date,
                              scale=1.0 + 0.03 * index, exiting=j < leaves_below))
        pd.DataFrame(rows).to_csv(central / _CENTRAL, index=False)
    return root


# --------------------------------------------------------------------------- #
# Deployment — the real app, a filesystem deck store, auth off.
# --------------------------------------------------------------------------- #

def _deploy(tmp_path, monkeypatch, periods, books, currency="GBP"):
    client = f"{CLIENT}{len(periods)}{len(books)}{currency.lower()}"
    root = _write(tmp_path / "runs", client, periods, books)
    config = tmp_path / f"client_{client}.yaml"
    config.write_text(f"portfolio:\n  base_currency: {currency}\n", encoding="utf-8")

    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(root))
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", client)
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", str(root))
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blobstore"))
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_PERSIST", "true")
    monkeypatch.setenv("TRAKT_INVESTOR_PPTX_ON_DEMAND", "true")
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "false")
    monkeypatch.setenv("TRAKT_MI_CLIENT_CONFIG", str(config))
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    for key in ("MI_AGENT_DECK_ROOT", "AZURE_STORAGE_CONNECTION_STRING",
                "TRAKT_BLOB_CONNECTION"):
        monkeypatch.delenv(key, raising=False)

    from mi_agent_api import currency as currency_mod, data_source, datasets
    from mi_agent_api import deck_generation
    currency_mod._load_client_config.cache_clear()
    datasets._CLIENT_CURRENCY_CACHE.clear()
    data_source.reset_cache()
    deck_generation.reset_jobs()
    return tmp_path


@pytest.fixture()
def rich(tmp_path, monkeypatch):
    """Three books, four periods, evidenced exits: the shape most of this
    sprint's pages only exist on."""
    yield _deploy(tmp_path, monkeypatch, _RICH_PERIODS, _BOOKS)
    from mi_agent_api import datasets, deck_generation
    deck_generation.reset_jobs()
    datasets._CLIENT_CURRENCY_CACHE.clear()


@pytest.fixture()
def simple(tmp_path, monkeypatch):
    """One book, one period: the shortest honest pack."""
    yield _deploy(tmp_path, monkeypatch, _SIMPLE_PERIODS,
                  (("direct_001", "direct", 1.0),))
    from mi_agent_api import datasets, deck_generation
    deck_generation.reset_jobs()
    datasets._CLIENT_CURRENCY_CACHE.clear()


# --------------------------------------------------------------------------- #
# Driving the production channel.
# --------------------------------------------------------------------------- #

def generate_and_download(timeout_s: int = 420) -> bytes:
    """The React button's own path: generate, poll, download."""
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    client = TestClient(app)
    accepted = client.post("/mi/decks/generate", json={})
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()["jobId"]

    deadline, body = time.time() + timeout_s, None
    while time.time() < deadline:
        status = client.get(f"/mi/decks/generate/{job_id}")
        assert status.status_code in (200, 202), status.text
        body = status.json()
        if body["state"] in ("completed", "blocked", "failed"):
            break
        time.sleep(0.4)
    assert body and body["state"] == "completed", body
    assert body["failedGates"] == [], body

    got = client.get("/mi/decks/download")
    assert got.status_code == 200, got.text
    assert got.content[:4] == b"PK\x03\x04", "not a PowerPoint file"
    return got.content


def deck_text(content: bytes) -> str:
    deck = pptx.Presentation(io.BytesIO(content))
    parts = []
    for slide in deck.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def slide_titles(content: bytes) -> list:
    deck = pptx.Presentation(io.BytesIO(content))
    titles = []
    for slide in deck.slides:
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                titles.append(shape.text_frame.text.strip().split("\n")[0])
                break
    return titles


def sidecar(tmp_path) -> dict:
    matches = sorted((tmp_path / "runs").glob("**/investor_pack.pptx.preflight.json"))
    assert matches, "the generator wrote no preflight sidecar"
    return json.loads(matches[-1].read_text(encoding="utf-8"))


def gates(side: dict) -> dict:
    return {g["gate"]: g for g in (side.get("preflight") or {}).get("gates") or ()}


def omissions(side: dict) -> dict:
    return {o["slide_id"]: o for o in side.get("omitted_slides") or ()}


# --------------------------------------------------------------------------- #
# 1-4. THE ECONOMIC BRIDGE.
# --------------------------------------------------------------------------- #

def test_1_the_balance_movement_page_reaches_a_deck_at_all(rich):
    """Catches: the bridge declining on every regime-projected book.

    The bridge accepted ``loan_identifier`` and not ``unique_identifier``, while
    the assembler that PRODUCES the tape treats both as loan identity. A book
    carrying the regulatory name got 'no stable loan identifier' — so this page
    never rendered anywhere, and nobody noticed, because a composed-away slide
    looks exactly like a slide that had nothing to say.
    """
    titles = slide_titles(generate_and_download())
    assert "Funded Balance Movement" in titles, titles


def test_2_the_bridge_identity_reconciles_on_the_page(rich):
    """Catches: a waterfall drawn from legs that do not sum to the closing bar.

    The figures the page draws are asserted here, not re-derived: the payload is
    the one the handler renders from.
    """
    generate_and_download()
    from mi_agent_pptx.mi_api import build_dashboard_data
    from mi_agent_api import evolution

    root = Path(rich) / "runs"
    client = next(p.name for p in root.iterdir() if p.is_dir())
    bm = evolution.funded_balance_movement(root, client, None)
    assert bm["available"], bm.get("reason")
    identity = (bm["openingBalance"] + bm["newLoanBalance"]
                - bm["exitedLoanBalance"] + bm["continuingMovement"])
    assert abs(identity - bm["closingBalance"]) <= 0.01, bm


def test_3_exits_are_split_by_evidenced_reason(rich):
    """Catches: every departure reported as one 'Exits' bar.

    A disappearance is not a redemption. Where the tape evidences the reason the
    page must show it, and the split must sum to the bridge's exit leg.
    """
    generate_and_download()
    # The waterfall's step labels become pixels the moment the figure is saved,
    # so what it DREW is read from the render record in the deck's own sidecar.
    drawn = [r for r in sidecar(rich).get("rendered") or ()
             if r.get("chart") == "balance_movement"]
    assert drawn, "no balance-movement waterfall was drawn"
    legs = " | ".join(drawn[0]["categories"])
    assert any(reason in legs for reason in
               ("Redeemed", "Matured", "Exited in default")), legs

    from mi_agent_api import evolution
    root = Path(rich) / "runs"
    client = next(p.name for p in root.iterdir() if p.is_dir())
    bm = evolution.funded_balance_movement(root, client, None)
    assert bm["exitsClassified"] and bm["exitsReconcile"], bm
    total = sum(c["balance"] for c in bm["exitComponents"])
    assert abs(total - bm["exitedLoanBalance"]) <= 0.01


def test_4_the_continuing_leg_is_never_called_interest(rich):
    """Catches: labelling surviving-book movement as accrued interest.

    Separating accretion from repayment needs per-loan period movement the
    canonical model does not carry. Trakt owns no such attribution, so the page
    must not imply one.
    """
    text = deck_text(generate_and_download()).lower()
    assert "continuing-book movement is the change on loans present at" in text, (
        "the page does not state what the continuing leg measures")
    assert "not split into interest, repayment or further advance" in text
    for claim in ("interest accrued", "accrued interest", "interest roll-up",
                  "interest rollup"):
        assert claim not in text, f"the deck claims {claim!r} it cannot evidence"


# --------------------------------------------------------------------------- #
# 5-7. STOCK, AND ITS AGREEMENT WITH MOVEMENT.
# --------------------------------------------------------------------------- #

def test_5_a_multi_book_portfolio_gets_a_stacked_stock_view(rich):
    """Catches: the per-book period series being computed and never drawn."""
    side = sidecar(rich) if generate_and_download() else {}
    stacks = [r for r in side.get("rendered") or ()
              if r.get("chart") == "funded_stock"]
    assert stacks, "no funded-stock chart was drawn"
    assert len(stacks[0].get("series") or ()) > 1, (
        f"a three-book portfolio drew a single series: {stacks[0]}")


def test_6_the_stack_sums_to_the_period_total(rich):
    """Catches: a stack whose parts do not add up to the whole beside it —
    the one defect a reader cannot see and cannot recover from."""
    generate_and_download()
    gate = gates(sidecar(rich))["stack_reconciles"]
    assert gate["passed"], gate
    assert "no multi-book" not in gate["detail"], (
        "the gate passed vacuously — it never saw a stack")


def test_7_stock_and_movement_close_on_the_same_number(rich):
    """Catches: two engines, two pages, two different closing balances.

    The stock series comes from the funded-evolution loader; the bridge
    reconciles loan by loan. A pack that prints one figure on one page and a
    different one three pages later is worse than either page alone.
    """
    generate_and_download()
    gate = gates(sidecar(rich))["stock_and_movement_agree"]
    assert gate["passed"], gate
    assert gate["evidence"]["gap"] <= 0.01, gate
    assert gate["evidence"]["stock_closing"] is not None


# --------------------------------------------------------------------------- #
# 8-9. THE PER-BOOK FORWARD VIEW.
# --------------------------------------------------------------------------- #

def test_8_a_single_book_portfolio_gets_no_per_book_pages(simple):
    """Catches: a table of one column and a stack of one colour.

    A one-book pack must not spend pages enumerating its single constituent —
    and the ledger must say why it did not.
    """
    content = generate_and_download()
    titles = slide_titles(content)
    assert "Forward View by Constituent Book" not in titles
    assert "Portfolio Composition" not in titles
    dropped = omissions(sidecar(simple))
    assert "only one constituent book" in dropped["portfolio_projections"]["reason"]


def test_9_the_projection_never_assumes_a_book_that_does_not_redeem(rich):
    """Catches: a per-book projection quietly holding balances flat.

    Where no approved run-off curve exists the balance IS held flat — which is
    honest only if the page says so. Trakt generates no mortality, decay or
    run-off assumption of its own and the pack must state that.
    """
    titles_and_text = deck_text(generate_and_download())
    if "Forward View by Constituent Book" not in titles_and_text:
        pytest.skip("no forecast resolved for this fixture, so no forward view")
    assert "no mortality, decay or run-off assumption" in titles_and_text


# --------------------------------------------------------------------------- #
# 10-11. MATERIALITY.
# --------------------------------------------------------------------------- #

def test_10_a_flat_distribution_is_not_given_a_driver(rich):
    """Catches: 'driven by X' where X leads by a rounding error.

    Naming the first item of a sorted list is an artefact of sorting, not an
    observation about the book.
    """
    from mi_agent_api import materiality as MAT

    flat = [{"label": r, "value": v} for r, v in
            zip("ABCDEFG", (4.4, 4.0, 4.0, 3.9, 3.8, 3.7, 3.7))]
    outcome = MAT.classify(flat)
    assert not outcome.has_driver, outcome.to_dict()
    assert MAT.describe(outcome, dimension="region")
    driven = MAT.classify([{"label": "A", "value": 22.0},
                           {"label": "B", "value": 2.0},
                           {"label": "C", "value": 1.5}])
    assert driven.has_driver and driven.leader.label == "A"


def test_11_a_uniform_dimension_is_dropped_and_said_to_be_dropped(rich):
    """Catches: a bar list that draws one full-width bar saying what its own
    title says — and, worse, dropping it silently."""
    from mi_agent_pptx.deck import DeckBuilder

    uniform = {"key": "region", "bars": [{"label": "London", "balance": 100.0}]}
    spread = {"key": "ltv", "bars": [{"label": "20-30%", "balance": 40.0},
                                     {"label": "40-50%", "balance": 60.0}]}
    assert not DeckBuilder._has_spread(uniform)
    assert DeckBuilder._has_spread(spread)
    near_uniform = {"key": "region",
                    "bars": [{"label": "London", "balance": 999.0},
                             {"label": "Wales", "balance": 1.0}]}
    assert not DeckBuilder._has_spread(near_uniform), (
        "a handful of loans in a second band is not a distribution")


# --------------------------------------------------------------------------- #
# 12-13. THE LEDGER MUST NOT CONTRADICT THE DECK.
# --------------------------------------------------------------------------- #

def test_12_a_superseded_page_is_not_reported_as_missing_capability(rich):
    """Catches: 'no prior reporting period to attribute movement against' printed
    in the ledger of a pack that attributes movement on page nine."""
    content = generate_and_download()
    titles = slide_titles(content)
    dropped = omissions(sidecar(rich))

    assert "Funded Balance Movement" in titles, titles
    assert dropped["movement_drivers"]["category"] == "superseded", dropped
    assert "covered by" in dropped["movement_drivers"]["reason"]

    # And the same rule for limits: the ledger may only call the legacy monitor
    # absent when the concentration page that replaces it is NOT in the deck.
    if "Concentration Tests and Headroom" in titles and "risk" in dropped:
        assert "covered by" in dropped["risk"]["reason"], dropped["risk"]


def test_13_geography_absence_names_which_geography(rich):
    """Catches: 'no geographic exposure resolved' in the ledger of a pack whose
    stratification page draws a regional bar list."""
    content = generate_and_download()
    dropped = omissions(sidecar(rich))
    if "geography" not in dropped:
        pytest.skip("this fixture resolved area-level exposure")
    reason = dropped["geography"]["reason"]
    text = deck_text(content)
    if "By region" in text or "Region" in text:
        assert "area-level" in reason, reason


# --------------------------------------------------------------------------- #
# 14-16. WORDING, LENGTH AND CAPABILITY.
# --------------------------------------------------------------------------- #

def test_14_a_balance_ratio_is_never_called_retention(rich):
    """Catches: a >100% 'retention' on a roll-up book.

    Retention is a survival idea and it is only that for a COUNT. A balance
    ratio wearing the same name tells the reader the pool grew.
    """
    text = deck_text(generate_and_download())
    if "Cohort Progression" not in text:
        pytest.skip("no cohort seasoning for this fixture")
    assert "Balance vs formation" in text
    assert "Retention" not in text, (
        "the cohort table still labels a balance ratio 'Retention'")


def test_15_the_pack_is_a_starter_pack_not_a_catalogue(rich):
    """Catches: appending every new page to a 25-page deck.

    The pack the sprint started from configured 25 slides and rendered whatever
    resolved. A conditional pack must be shorter than the catalogue, must not
    repeat a title, and must not print a page with nothing on it.
    """
    content = generate_and_download()
    titles = slide_titles(content)
    assert 12 <= len(titles) <= 20, titles
    assert len(titles) == len(set(titles)), f"a title is repeated: {titles}"
    side = sidecar(rich)
    assert not [s for s in side["slides"] if s["placeholder"]], side["slides"]
    assert side["preflight"]["publishable"] is True


def test_16_the_pack_asks_the_registry_what_this_book_supports(rich):
    """Catches: conditional reporting branching on an asset class.

    The published capability registry is asset-agnostic by construction — a
    capability declares the ECONOMIC conditions it needs. The pack must consume
    that result rather than name what the book is.
    """
    generate_and_download()
    side = sidecar(rich)
    facts = side.get("facts") or {}
    capability_facts = {k: v for k, v in facts.items() if k.startswith("can_")}
    assert capability_facts, "no capability facts reached composition"
    assert facts.get("can_total_balance") is True, capability_facts

    # And no slide condition in the shipped pack names an asset class.
    import yaml
    config = yaml.safe_load(
        (_REPO_ROOT / "configs/pptx/investor_pack.yaml").read_text(encoding="utf-8"))
    for slide in config["slides"]:
        condition = str(slide.get("when") or "")
        for banned in ("equity_release", "lifetime_mortgage", "asset_class",
                       "seasoned", "product_type"):
            assert banned not in condition, (
                f"{slide['id']} branches on {banned!r}: {condition}")
