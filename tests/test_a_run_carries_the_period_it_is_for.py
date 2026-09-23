"""The run id was the literal string "run", and two things read a period from it.

From the live delivery's artefacts, all evening::

    18f_central_universe_debug.json   run_reporting_period: ''
    04c_source_period_eligibility     run= (empty, on every row)
    28c_human_decision_queue          reporting_date — blocking=True
        "required target field has no source, derivation, default or ND rule"

`reporting_date` is not supposed to need an operator. ``target_coverage``
fills it, and ``data_cut_off_date`` with it, by reading a period token out of
the run id::

    for tok in (run_id,):
        hits = rc.dates_from_period_token(tok) if tok else []

The live adapter called onboarding with ``run_id="run"``. It has no period, so
the field was reported missing and BLOCKED the delivery — while the period sat
in the very same call, two lines below, as ``reporting_date=``.

The second reader is quieter and worse. ``run_context.run_period`` gates funded
sources on the run's period, and an empty period makes that gate permissive: a
September file would have been read into an August delivery without a word.
The gate that exists to stop the wrong month's numbers was switched off by a
placeholder.

A period that cannot be read still yields ``"run"``, so a delivery with no
period behaves exactly as it does today rather than acquiring a wrong one.
"""

from __future__ import annotations

import pytest

from engine.onboarding_agent import run_context as rc
from engine.orchestrator_agent.adapters import run_id_for


class TestTheRunIdCarriesThePeriod:

    @pytest.mark.parametrize("period", ["2026-08", "2026_08", "2026-08-31"])
    def test_a_real_period_produces_a_readable_run_id(self, period):
        assert rc.dates_from_period_token(run_id_for(period)) == ["2026-08-31"]

    def test_the_month_is_the_one_the_delivery_is_for(self):
        assert rc.dates_from_period_token(run_id_for("2026-02")) == ["2026-02-28"]
        assert rc.dates_from_period_token(run_id_for("2024-02")) == ["2024-02-29"]

    def test_it_is_a_name_a_person_can_read(self):
        assert run_id_for("2026-08") == "mi_2026_08"


class TestAnUnusablePeriodChangesNothing:
    """Better the old placeholder than a period nobody meant."""

    @pytest.mark.parametrize("period", ["", None, "   ", "unspecified",
                                        "2026", "not a period", "??"])
    def test_it_falls_back_to_the_placeholder(self, period):
        assert run_id_for(period) == "run"

    def test_and_the_placeholder_still_carries_no_period(self):
        assert rc.dates_from_period_token(run_id_for("")) == []


class TestTheLiveAdapterUsesIt:
    """The defect was not the helper's absence; it was three call sites that
    passed a constant."""

    def test_no_call_site_hardcodes_a_run_id(self):
        import inspect
        from engine.orchestrator_agent import adapters
        src = inspect.getsource(adapters)
        assert 'run_id="run"' not in src, (
            "a hardcoded run id carries no period, and two readers need one")

    def test_every_onboarding_call_derives_it_from_the_reporting_period(self):
        import inspect
        from engine.orchestrator_agent import adapters
        src = inspect.getsource(adapters)
        assert src.count("run_id=run_id_for(") == 3
