"""Golden regression for the Annex 2 delivery route.

Two tiers, and the difference between them is stated rather than blurred:

* **CI tier** (default) — a small committed fixture, run on every test
  invocation. It proves the route is intact end to end: normalise → build →
  XSD PASS, with the evidence reconciling exactly. It proves **nothing** about
  performance at scale, and no assertion here claims otherwise.

* **Golden tier** (opt-in) — the full-scale reproduction of the historically
  recorded run: 11,035 exposure records through the same components, with the
  metrics recorded for comparison. Enabled with::

      TRAKT_ANNEX2_GOLDEN=1 python -m pytest tests/test_annex_delivery_agent_golden.py -s

  or by running ``scripts/run_annex2_golden.py`` directly.

The scale input is *generated deterministically* from the committed synthetic
projected data (see :mod:`scripts.generate_annex2_scale_fixture`). The original
11,035-record client tape is not in this repository, so this is a like-for-like
reproduction of the route and its shape at the recorded scale, not a re-run of
the original data — the numbers below are labelled accordingly.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from trakt_core.context import ExecutionContext

from engine.annex_delivery_agent.agent import AnnexDeliveryAgent
from engine.annex_delivery_agent.contracts import DeliveryRequest, DeliveryResult, DeliveryStatus
from engine.annex_delivery_agent.persistence import DeliveryRunStore

_REPO_ROOT = Path(__file__).resolve().parents[1]
CI_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "annex2_projected_ci.csv"
#: The scale tier is generated from the CI fixture. It used to be generated from
#: the synthetic demo projection, which predates the 107-code contract: that tape
#: leaves 24 mandatory codes unanswered because they are portfolio decisions no
#: layer has made for it, and the retired delivery-rules file used to fill them
#: with a regime-level default on the lender's behalf. A governed projection is
#: the right seed for a scale reproduction; an ungoverned one was never the point.
SOURCE_FIXTURE = CI_FIXTURE

#: The historically recorded full-scale run this tier reproduces.
HISTORICAL = {
    "records": 11_035,
    # 105 populated from projected source + 2 populated by declared delivery
    # rule (RREL20/RREL21 as ND5) = 107 represented. Before Phase 2 the two
    # were injected by the builder and so absent from the delivery-ready CSV,
    # which is why this read 105. The XML is byte-identical either way.
    "fields": 107,
    "fields_from_projected_source": 105,
    "fields_from_delivery_rule": 2,
    "artefact_mb": 208.8,
    "seconds": 171.5,
    "peak_rss_mb": 772.7,
    "xsd": "PASSED",
}

GOLDEN_ENABLED = os.environ.get("TRAKT_ANNEX2_GOLDEN", "").strip().lower() in ("1", "true", "yes")


def _ctx() -> ExecutionContext:
    return ExecutionContext.for_internal("ERE", actor_id="golden-regression")


def _prepare(projected: Path, store_root: Path) -> DeliveryResult:
    agent = AnnexDeliveryAgent(store=DeliveryRunStore(store_root))
    governed = agent.prepare(_ctx(), DeliveryRequest(
        annex_id="ESMA_Annex2", reporting_date="2026-01-31",
        projected_input=str(projected), output_root=str(store_root)))
    assert governed.result is not None, (
        governed.error.message if governed.error else "no result")
    return DeliveryResult.from_dict(governed.result)


# --------------------------------------------------------------------------- #
# CI tier
# --------------------------------------------------------------------------- #


def _nd_in_non_emitting_codes(fixture: Path) -> int:
    """No-data values in the prepared data for codes auth.099 has no element for.

    The three are currency qualifiers carried as an XML attribute of the amount
    they describe, so they can hold a currency but not a no-data code.
    """
    import csv

    from engine.regime_contract import build_contract

    codes = [c for c, f in build_contract().fields.items() if not f.emitting]
    with fixture.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return sum(1 for r in rows for c in codes
               if str(r.get(c, "")).strip().upper().startswith("ND"))


@pytest.mark.skipif(not CI_FIXTURE.is_file(), reason="CI fixture missing")
def test_ci_regression_route_is_intact(tmp_path):
    """The route runs end to end and the schema accepts the result.

    Deliberately not a performance test: thirty records say nothing about an
    11,035-record run.
    """
    expected_records = len(CI_FIXTURE.read_text().strip().splitlines()) - 1
    result = _prepare(CI_FIXTURE, tmp_path / "store")

    assert result.status in (DeliveryStatus.PREPARED, DeliveryStatus.PREPARED_WITH_WARNINGS)
    assert result.schema_validation.valid is True, "XSD validation must pass"
    assert result.record_count == expected_records
    assert result.projected_field_count == 107
    assert Path(result.artefact.path).is_file()

    recon = result.transformation_evidence["nd_reconciliation"]
    assert recon["unattributed"] == 0
    # The builder invents no no-data value of its own — the property this line
    # has always been about. It used to be written as an equality against
    # ``difference_added_by_builder``, which held only while that difference was
    # non-negative. It is negative here: the prepared data carries 30 no-data
    # values the report does not, one per record for RREL18 (Primary Income
    # Currency). auth.099 carries the three currency concepts as an ATTRIBUTE of
    # the amount they qualify, and an attribute has nowhere to put ND1 — so the
    # shortfall is the schema's shape, not a lost value, and it is asserted as
    # exactly that rather than absorbed into an equality that no longer means
    # anything.
    assert recon["attributed_as_automatic"] == 0
    shortfall = -recon["difference_added_by_builder"]
    assert shortfall == _nd_in_non_emitting_codes(CI_FIXTURE), (
        "a no-data value went missing from a code the schema does emit")


@pytest.mark.skipif(not SOURCE_FIXTURE.is_file(), reason="synthetic projected data missing")
def test_ci_regression_is_deterministic(tmp_path):
    """The same input and configuration produce the same artefact bytes."""
    first = _prepare(SOURCE_FIXTURE, tmp_path / "a")
    second = _prepare(SOURCE_FIXTURE, tmp_path / "b")
    assert first.run_id == second.run_id
    assert first.artefact.sha256 == second.artefact.sha256
    assert first.record_count == second.record_count == 30


# --------------------------------------------------------------------------- #
# Golden tier — opt-in
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not GOLDEN_ENABLED,
                    reason="set TRAKT_ANNEX2_GOLDEN=1 to run the full-scale reproduction")
def test_golden_full_scale_reproduction(tmp_path):
    """Reproduce the recorded run at 11,035 records and report the metrics."""
    from scripts.generate_annex2_scale_fixture import build_scale_fixture

    projected = build_scale_fixture(SOURCE_FIXTURE, tmp_path / "scale_projected.csv",
                                    target_records=HISTORICAL["records"])
    started = time.perf_counter()
    result = _prepare(projected, tmp_path / "store")
    elapsed = time.perf_counter() - started

    metrics = {
        "records": result.record_count,
        "projected_fields": result.projected_field_count,
        "delivered_fields": result.delivered_field_count,
        "artefact_mb": round(result.artefact.size_bytes / (1024 * 1024), 1),
        "seconds": round(elapsed, 1),
        "peak_rss_mb": round(result.peak_rss_mb, 1) if result.peak_rss_mb else None,
        "xsd": "PASSED" if result.schema_validation.valid else "FAILED",
        "stage_seconds": {t.stage: round(t.seconds, 1) for t in result.timings},
        "automatic_fills": result.transformation_evidence["automatic_fill_count"],
        "nd_reconciliation": {
            k: v for k, v in result.transformation_evidence["nd_reconciliation"].items()
            if k != "per_field_in_prepared_data"},
    }
    print("\nGolden Annex 2 reproduction\n" + json.dumps(metrics, indent=2))
    print("Historical reference\n" + json.dumps(HISTORICAL, indent=2))

    assert result.schema_validation.valid is True, "the golden run must pass XSD validation"
    assert result.record_count == HISTORICAL["records"]
    assert result.transformation_evidence["nd_reconciliation"]["unattributed"] == 0
