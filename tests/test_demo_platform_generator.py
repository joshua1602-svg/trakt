"""Tests for the synthetic demonstration generator.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

These cover the properties the demonstration depends on and that a change to the
generator could silently break: reproducibility, loan-identifier uniqueness, the
difference between the two source schemas, the reporting periods, and the internal
economic coherence of every generated row.

The generator's calibration is the slow part (it rebuilds both books ~24 times),
so it is run ONCE per session and shared.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_platform import config as cfg          # noqa: E402
from demo_platform import generator, schemas     # noqa: E402


@pytest.fixture(scope="module")
def generated():
    """Both calibrated books. Slow (~90s), so module-scoped."""
    return generator.generate(verbose=False)


@pytest.fixture(scope="module")
def current_frames(generated):
    return {pid: fr[cfg.CURRENT_PERIOD.period] for pid, fr in generated.frames.items()}


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_generation_is_reproducible():
    """The same seed must produce byte-identical source files.

    Run in a subprocess with an isolated workspace so it cannot disturb the
    session's generated environment, and compare content hashes.
    """
    import hashlib
    import json
    import tempfile

    script = (
        "import json, hashlib, os, sys, warnings; warnings.simplefilter('ignore');"
        "sys.path.insert(0, %r);"
        "from demo_platform import generator, config as cfg;"
        "g = generator.generate(verbose=False);"
        "w = generator.write_source_files(g);"
        "print(json.dumps({f['path'].split('/')[-1]: "
        "hashlib.sha256(open(f['path'],'rb').read()).hexdigest() for f in w}))"
        % str(REPO_ROOT)
    )
    hashes = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as td:
            env = {**dict(**__import__("os").environ), "TRAKT_DEMO_ROOT": td}
            proc = subprocess.run(
                [sys.executable, "-c", script], cwd=str(REPO_ROOT),
                capture_output=True, text=True, env=env, timeout=1800,
            )
            assert proc.returncode == 0, proc.stderr[-2000:]
            hashes.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert hashes[0] == hashes[1], (
        "Two runs of the generator produced different source files — the "
        "demonstration is no longer reproducible."
    )
    assert len(hashes[0]) == len(cfg.PORTFOLIOS) * len(cfg.PERIODS)


def test_calibration_converges_on_every_target(generated):
    """The calibration must land every narrative target within tolerance."""
    m = generated.model_metrics
    assert cfg.TARGET_WA_LTV_CURRENT.ok(m["wa_current_ltv_points"]), m["wa_current_ltv_points"]
    assert cfg.TARGET_BORROWER_AGE_CURRENT.ok(m["avg_borrower_age"]), m["avg_borrower_age"]
    assert cfg.TARGET_MOVEMENT_TOTAL.ok(m["consolidated_movement"]), m["consolidated_movement"]
    moves = m["movement_by_portfolio"]
    assert cfg.TARGET_MOVEMENT_A.ok(moves[cfg.PORTFOLIO_A.source_portfolio_id])
    assert cfg.TARGET_MOVEMENT_B.ok(moves[cfg.PORTFOLIO_B.source_portfolio_id])


# --------------------------------------------------------------------------- #
# Scale and identity
# --------------------------------------------------------------------------- #
def test_portfolio_scale_is_in_the_intended_band(generated):
    """Each book, and the consolidated view, must be the intended size."""
    a = generated.books[cfg.PORTFOLIO_A.source_portfolio_id]
    b = generated.books[cfg.PORTFOLIO_B.source_portfolio_id]
    assert 6_000 <= a.total_loans <= 8_000, a.total_loans
    assert 3_000 <= 5_000 >= b.total_loans and b.total_loans >= 3_000, b.total_loans
    lo, hi = cfg.LOAN_COUNT_BAND
    assert lo <= generated.model_metrics["loan_count"] <= hi


def test_loan_identifiers_are_unique_within_each_portfolio(current_frames):
    for pid, frame in current_frames.items():
        dupes = frame["loan_id"].duplicated().sum()
        assert dupes == 0, f"{pid}: {dupes} duplicate loan identifiers"


def test_loan_identifiers_are_unique_across_the_platform(current_frames):
    combined = pd.concat(current_frames.values(), ignore_index=True)
    assert combined["loan_id"].duplicated().sum() == 0


def test_loan_identifiers_carry_the_portfolio_prefix(current_frames):
    for pid, frame in current_frames.items():
        prefix = cfg.portfolio(pid).display_id
        assert frame["loan_id"].str.startswith(f"{prefix}-").all()


# --------------------------------------------------------------------------- #
# Source schemas
# --------------------------------------------------------------------------- #
def test_the_two_source_schemas_share_no_business_column_names():
    shared = [h for h in schemas.shared_headers() if h != "Synthetic Data Notice"]
    assert shared == [], f"schemas share business columns: {shared}"


def test_every_declared_mapping_names_a_canonical_field_or_is_deliberate():
    for schema in schemas.SCHEMAS.values():
        for column in schema.columns:
            if column.canonical_field:
                assert column.resolution in ("global", "contract")
            else:
                # An unmapped header must be documented, not accidental.
                assert column.note, f"{column.header} is unmapped with no note"


def test_each_schema_covers_the_fields_the_mi_layer_needs():
    """Both schemas must supply everything the MI and reporting layers use."""
    required = {
        "loan_identifier", "data_cut_off_date", "origination_date",
        "original_principal_balance", "current_principal_balance",
        "original_valuation_amount", "current_valuation_amount",
        "current_interest_rate", "youngest_borrower_age", "collateral_geography",
        "account_status", "amortisation_type",
    }
    for name, schema in schemas.SCHEMAS.items():
        provided = {c.canonical_field for c in schema.columns if c.canonical_field}
        missing = required - provided
        assert not missing, f"{name} does not supply {sorted(missing)}"


def test_each_schema_covers_the_annex2_exposure_level_fields():
    """Both schemas must supply what the ESMA Annex 2 submission needs per loan.

    Every field below is mandatory at exposure level in auth.099.001.04 and has to
    arrive from the source extract — it cannot be derived, defaulted or ND-ed. A
    schema change that drops one of them breaks the regulatory artefact, and the
    demonstration would then be showing an output it cannot produce.
    """
    required = {
        "original_underlying_exposure_identifier",   # RREL2 -> RREL3, RREC2
        "original_obligor_identifier",               # RREL4 -> RREL5
        "purpose",                                   # RREL30
        "maturity_date",                             # RREL24
        "pool_addition_date",                        # RREL7
        "total_credit_limit",                        # RREL33
        "new_collateral_identifier",                 # RREC4
        "original_collateral_identifier",            # RREC3
        "collateral_type",                           # RREC5
        "property_type",                             # RREC9
        "current_valuation_method",                  # RREC14
        "origination_channel",                       # RREL26
        "resident",                                  # RREL10
        "credit_impaired_obligor",                   # RREL14
        "litigation",                                # RREL75
        "payment_due",                               # RREL39
        "arrears_balance",                           # RREL67
        "number_of_days_in_arrears",                 # RREL68
        "default_amount",                            # RREL71
        "allocated_losses",                          # RREL73
        "cumulative_recoveries",                     # RREL74
        "prepayment_fee",                            # RREL61
        "occupancy_type",                            # RREC7
    }
    for name, schema in schemas.SCHEMAS.items():
        provided = {c.canonical_field for c in schema.columns if c.canonical_field}
        missing = required - provided
        assert not missing, f"{name} does not supply {sorted(missing)}"


def test_declared_resolutions_match_what_gate_1_actually_does():
    """The film states which mappings needed the client contract; prove it.

    Runs the production header mapper both with and without each portfolio's
    contract overlay. A header the global vocabulary already resolves must be
    declared ``global``; one that only resolves with the overlay must be declared
    ``contract``. Otherwise the onboarding scene overstates (or understates) what
    the client contract did.
    """
    from demo_platform.onboarding import _GLOBAL_ALIASES, _REGISTRY, _mapper_for
    from engine.gate_1_alignment.semantic_alignment import (
        HeaderMapper, load_aliases_from_dir, load_field_registry,
        select_registry_fields,
    )

    registry = load_field_registry(_REGISTRY)
    canonical = select_registry_fields(registry, "equity_release")
    bare = HeaderMapper(canonical, load_aliases_from_dir(_GLOBAL_ALIASES))

    wrong = []
    for portfolio in cfg.PORTFOLIOS:
        overlaid, _ = _mapper_for(portfolio)
        for column in schemas.schema_for(portfolio.schema_name).columns:
            want = column.canonical_field or None
            achieved, _tier, _conf = overlaid.map_one(column.header)
            if achieved != want:
                wrong.append((portfolio.display_id, column.header,
                              f"resolves to {achieved!r}, declared {want!r}"))
                continue
            if want is None:
                continue
            without, _t, _c = bare.map_one(column.header)
            expected = "global" if without == want else "contract"
            if expected != column.resolution:
                wrong.append((portfolio.display_id, column.header,
                              f"declared {column.resolution!r}, measured {expected!r}"))
    assert not wrong, wrong


# --------------------------------------------------------------------------- #
# Reporting periods
# --------------------------------------------------------------------------- #
def test_three_reporting_periods_with_current_and_prior():
    roles = [p.role for p in cfg.PERIODS]
    assert roles == ["opening", "prior", "current"]
    assert cfg.CURRENT_PERIOD.reporting_date > cfg.PRIOR_PERIOD.reporting_date
    assert cfg.PRIOR_PERIOD.reporting_date > cfg.OPENING_PERIOD.reporting_date


def test_all_reporting_dates_are_month_ends_and_historical():
    import calendar
    for p in cfg.PERIODS:
        year, month, day = (int(x) for x in p.reporting_date.split("-"))
        assert day == calendar.monthrange(year, month)[1], p.reporting_date
        assert date(year, month, day) < date(2026, 7, 25), (
            f"{p.reporting_date} is not historical relative to the build date"
        )


def test_each_period_frame_carries_exactly_one_reporting_date(generated):
    for pid, per_period in generated.frames.items():
        for period, frame in per_period.items():
            unique = set(frame["reporting_date"].unique())
            assert unique == {cfg.period(period).reporting_date}, (pid, period, unique)


# --------------------------------------------------------------------------- #
# Economic coherence, row by row
# --------------------------------------------------------------------------- #
def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def test_ltv_reconciles_to_balance_and_valuation(current_frames):
    for pid, frame in current_frames.items():
        bal = _num(frame["current_balance"])
        val = _num(frame["current_valuation"])
        mask = bal.notna() & val.notna() & (val > 0) & (bal > 0)
        ltv = (bal[mask] / val[mask])
        # A lifetime mortgage's rolled-up LTV must stay inside a plausible band.
        assert ltv.min() > 0.02, (pid, float(ltv.min()))
        assert ltv.max() < 1.20, (pid, float(ltv.max()))


def test_redeemed_accounts_carry_nil_balance(current_frames):
    for pid, frame in current_frames.items():
        redeemed = frame["status"].isin(("Redeemed",))
        if redeemed.any():
            assert float(_num(frame.loc[redeemed, "current_balance"]).abs().sum()) == 0.0
            assert (frame.loc[redeemed, "redemption_date"] != "").all()


def test_active_accounts_carry_no_redemption_date(current_frames):
    for pid, frame in current_frames.items():
        active = frame["status"] == "Active"
        assert (frame.loc[active, "redemption_date"] == "").all(), pid


def test_balances_grow_by_roll_up_between_periods(generated):
    """A loan on book in both periods must have grown at its own roll-up rate."""
    pid = cfg.PORTFOLIO_B.source_portfolio_id   # closed book: no completions
    cur = generated.frames[pid][cfg.CURRENT_PERIOD.period].set_index("loan_id")
    pri = generated.frames[pid][cfg.PRIOR_PERIOD.period].set_index("loan_id")
    common = cur.index.intersection(pri.index)
    # Exclude this month's redemptions (nil balance) and this month's drawdowns.
    live = cur.loc[common]
    live = live[(live["status"] == "Open") | (live["status"] == "Active")]
    live = live[_num(live["drawdown_amount"]).fillna(0.0) == 0.0]
    assert len(live) > 1_000, len(live)
    ratio = _num(live["current_balance"]) / _num(pri.loc[live.index, "current_balance"])
    monthly = _num(live["rate"]) / 1200.0
    expected = 1.0 + monthly
    assert (ratio - expected).abs().max() < 1e-6


def test_borrower_age_moves_directionally_across_periods(generated):
    pid = cfg.PORTFOLIO_A.source_portfolio_id
    cur = generated.frames[pid][cfg.CURRENT_PERIOD.period].set_index("loan_id")
    pri = generated.frames[pid][cfg.PRIOR_PERIOD.period].set_index("loan_id")
    common = cur.index.intersection(pri.index)
    # One row per period carries a deliberately injected below-minimum age, and
    # the injection lands on a different loan each period. Those rows are the
    # validation gate's subject matter, not the age model's, so exclude them.
    injected = {e["loan_id"] for e in generated.injected_exceptions
                if e["rule"] == "borrower_age_below_product_minimum"}
    common = common.difference(pd.Index(sorted(injected)))
    delta = _num(cur.loc[common, "borrower_age"]) - _num(pri.loc[common, "borrower_age"])
    assert delta.dropna().min() >= 0, "a borrower got younger"
    assert delta.dropna().max() <= 1, "a borrower aged more than a year in a month"


def test_regions_are_only_the_allowed_uk_regions(current_frames):
    allowed = set(cfg.ALLOWED_REGIONS) | {""}
    for pid, frame in current_frames.items():
        seen = set(frame["region"].astype(str).unique())
        assert seen <= allowed, (pid, sorted(seen - allowed))


def test_postcodes_sit_inside_the_region_they_claim(current_frames):
    """A loan's postcode district must belong to the ITL1 region it reports.

    This is what makes the readable region label and the transform's derived ITL3
    code agree — the demonstration's geography reconciles by construction.
    """
    pool = generator.postcode_pool()
    for pid, frame in current_frames.items():
        sample = frame[(frame["region"] != "") & (frame["postcode"] != "")].head(400)
        for _, row in sample.iterrows():
            district = str(row["postcode"]).split(" ")[0]
            assert district in pool[str(row["region"])], (
                f"{pid}: {district} is not in {row['region']}"
            )


def test_injected_exceptions_are_bounded_and_recorded(generated):
    expected = (cfg.INJECTED_EXCEPTIONS_PER_PERIOD
                * len(cfg.PORTFOLIOS) * len(cfg.PERIODS))
    assert len(generated.injected_exceptions) == expected
    rules = {e["rule"] for e in generated.injected_exceptions}
    assert len(rules) >= 4, rules
    for entry in generated.injected_exceptions:
        assert entry["loan_id"] and entry["detail"]


def test_every_source_file_carries_the_synthetic_marker(generated):
    written = generator.write_source_files(generated)
    assert len(written) == len(cfg.PORTFOLIOS) * len(cfg.PERIODS)
    for descriptor in written:
        path = Path(descriptor["path"])
        assert path.exists()
        head = pd.read_csv(path, nrows=3)
        assert "Synthetic Data Notice" in head.columns
        assert (head["Synthetic Data Notice"] == cfg.SYNTHETIC_BANNER).all()
