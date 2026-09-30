"""The paraphrase-invariance scorer for the held-out variants (P0 design §25)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = (Path(__file__).resolve().parents[1]
         / "due_diligence/evidence/qb_plan_readback/score_variants.py")
_SPEC = importlib.util.spec_from_file_location("score_variants", _PATH)
sv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sv)


def _rec(id_, outcome="ANSWERED", served="NEW", answer=""):
    return {"id": id_, "outcome": outcome, "served": served, "answer": answer}


_BANK = _rec("forecast_003", answer="Forecast funded balance: £94.1m — the "
             "funded balance of £87.1m plus £6.9m of expected completions.")


def test_the_headline_figure_is_the_first_the_answer_states():
    assert sv.figures(_BANK["answer"])[:3] == ["£94.1m", "£87.1m", "£6.9m"]
    assert sv.figures("Expected completion date: 2026-06-08 — over the 5 live "
                      "cases")[:2] == ["2026-06-08", "5"]
    assert sv.figures("1,330 closed cases, 12.5% of them")[:2] == ["1330", "12.5%"]


def test_the_verdicts():
    same = _rec("v", answer="Forecast funded balance: £94.1m, of which ...")
    assert sv.verdict(same, _BANK) == "SAME"
    assert sv.verdict(_rec("v", answer="Weighted amount: £6.9m"), _BANK) == "DIFFERENT"
    assert sv.verdict(_rec("v", served="LEGACY_FALLBACK",
                           answer="£94.1m"), _BANK) == "PATH"
    assert sv.verdict(_rec("v", outcome="REFUSED"), _BANK) == "LOST"
    assert sv.verdict(same, _rec("b", outcome="REFUSED")) == "GAINED"
    assert sv.verdict(_rec("v", outcome="REFUSED"),
                      _rec("b", outcome="REFUSED")) == "DECLINED"
    assert sv.verdict(same, None) == "NO_BANK_RECORD"


def test_scoring_reads_the_variant_map_from_the_holdout_file():
    out = sv.score({"hv_forecast_003_1": _rec("hv_forecast_003_1",
                                              answer="Weighted: £6.9m")},
                   {"forecast_003": _BANK})
    assert out["tally"] == {"DIFFERENT": 1}
    finding = out["findings"][0]
    assert finding["variant_of"] == "forecast_003"
    assert finding["bucket"] == "must_answer"
    assert finding["bank"]["headline"] == "£94.1m"


def test_the_headline_is_the_amount_not_the_month_that_names_it():
    bank = _rec("pipeline_012", answer="The live pipeline weighted expected "
                "funded amount expected to complete next month (2026-10) is "
                "£4.8m, as at the weekly extract of 2026-09-24.")
    variant = _rec("v", answer="The live pipeline amount expected to complete "
                   "next month (2026-10) is £7.8m, as at 2026-09-24.")
    assert sv.headline(bank["answer"]) == "£4.8m"
    assert sv.verdict(variant, bank) == "DIFFERENT"
    assert sv.headline("Expected completion date: 2026-04-01 — over 4835 "
                       "cases") == "2026-04-01"
