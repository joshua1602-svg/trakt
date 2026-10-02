"""The bank runner asks the held-out variants beside their bank questions
(P0 design §25)."""
from __future__ import annotations

from pathlib import Path

import yaml

from mi_agent_api import question_bank as qb

_SCRIPT = Path(__file__).resolve().parents[1] / "mi_agent_api/run_production_bank.sh"


def test_recent_puts_each_bank_question_before_its_variants():
    rows = qb.holdout_rows("recent")
    variants = [r for r in yaml.safe_load(qb.HOLDOUT_BANK.read_text())["questions"]
                if r["category"] == "holdout_recent"]
    assert [r["id"] for r in rows if "variant_of" in r] == \
        [v["id"] for v in sorted(variants, key=lambda v: [
            x["variant_of"] for x in variants].index(v["variant_of"]))]
    seen = set()
    for row in rows:
        if "variant_of" in row:
            assert row["variant_of"] in seen, row["id"]
        else:
            seen.add(row["id"])
    assert len(rows) == len(variants) + len({v["variant_of"] for v in variants})


def test_all_is_every_variant():
    assert len(qb.holdout_rows("all")) == len(
        yaml.safe_load(qb.HOLDOUT_BANK.read_text())["questions"])


def test_the_runner_asks_the_holdout_rows(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(qb, "run_one", lambda row, **_: asked.append(row["id"]) or {
        "id": row["id"], "category": row["category"], "question": row["question"],
        "outcome": "ANSWERED", "route": None, "view": None, "seconds": 0.0,
        "served": "NEW", "serving_reason": "", "timing": {}, "answer": ""})
    monkeypatch.setattr(qb, "_capture_serving", lambda: None)
    assert qb.main(["--holdout", "recent", "--out", str(tmp_path / "o.jsonl")]) == 0
    assert asked == [r["id"] for r in qb.holdout_rows("recent")]


def test_the_production_script_offers_both_selections():
    text = _SCRIPT.read_text()
    assert 'variants-recent) HOLDOUT="recent"' in text
    assert 'variants) HOLDOUT="all"' in text
    assert '--holdout ${HOLDOUT}' in text
    assert 'KIND="qb_variants_${HOLDOUT}"' in text


def test_twins_are_the_conversation_banks_new_twins_each_once():
    """§34: only twins that are not production-bank questions are asked."""
    rows = qb.twin_rows()
    data = yaml.safe_load(qb.CONVERSATION_BANK.read_text())
    new = {t["twin"]["question"] for c in data["conversations"] for t in c["turns"]
           if t.get("twin") and not t["twin"].get("bank_id")}
    assert [r["question"] for r in rows] == list(dict.fromkeys(
        t["twin"]["question"] for c in data["conversations"] for t in c["turns"]
        if t.get("twin") and not t["twin"].get("bank_id")))
    assert {r["question"] for r in rows} == new
    assert len(rows) == 27
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["category"] for r in rows} == {"conversation_twin"}


def test_the_runner_asks_the_twin_rows(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(qb, "run_one", lambda row, **_: asked.append(row["id"]) or {
        "id": row["id"], "category": row["category"], "question": row["question"],
        "outcome": "ANSWERED", "route": None, "view": None, "seconds": 0.0,
        "served": "NEW", "serving_reason": "", "timing": {}, "answer": ""})
    monkeypatch.setattr(qb, "_capture_serving", lambda: None)
    assert qb.main(["--twins", "--out", str(tmp_path / "o.jsonl")]) == 0
    assert asked == [r["id"] for r in qb.twin_rows()]


def test_the_production_script_offers_the_twins():
    text = _SCRIPT.read_text()
    assert 'twins) TWINS="1"' in text
    assert 'echo "--twins"' in text
    assert 'KIND="qb_twins"' in text


def test_unspent_is_every_variant_no_fix_was_made_against():
    rows = qb.holdout_rows("unspent")
    variants = yaml.safe_load(qb.HOLDOUT_BANK.read_text())["questions"]
    assert [r["id"] for r in rows] == [
        v["id"] for v in variants if v["category"] != "holdout_recent"]
    assert len(rows) == 81


def test_the_signoff_asks_the_bank_then_the_unspent_variants(monkeypatch, tmp_path):
    """D13: one run, the 135 bank questions first, then the 81 variants."""
    asked = []
    monkeypatch.setattr(qb, "run_one", lambda row, **_: asked.append(row["id"]) or {
        "id": row["id"], "category": row["category"], "question": row["question"],
        "outcome": "ANSWERED", "route": None, "view": None, "seconds": 0.0,
        "served": "NEW", "serving_reason": "", "timing": {}, "answer": ""})
    monkeypatch.setattr(qb, "_capture_serving", lambda: None)
    categories = ("funded_kpi,funded_breakdown_1d,pipeline,pipeline_evolution,"
                  "forecast,forecast_scale")
    assert qb.main(["--signoff", "--categories", categories,
                    "--out", str(tmp_path / "o.jsonl")]) == 0
    bank = [r["id"] for r in qb.load_bank(qb.DEFAULT_BANKS)
            if r["category"] in categories.split(",")]
    assert len(bank) == 135
    assert asked == bank + [r["id"] for r in qb.holdout_rows("unspent")]


def test_the_production_script_offers_the_signoff():
    text = _SCRIPT.read_text()
    assert 'signoff) SIGNOFF="1"' in text
    assert 'echo "--signoff --categories ${CATEGORIES}"' in text
    assert 'KIND="qb_signoff"' in text


def test_every_turn_the_model_reads_is_played():
    """§39: every live turn of every conversation, in order; the memory's
    mechanics (`run: code`) are enforced in code, not played."""
    data = yaml.safe_load(qb.CONVERSATION_BANK.read_text())
    live = [(c["id"], i) for c in data["conversations"]
            for i, t in enumerate(c["turns"]) if t.get("run") == "live"]
    rows = qb.conversation_rows("all")
    assert [(r["conversation"], r["turn"]) for r in rows] == live
    assert len(rows) == 123
    group_c = qb.conversation_rows("C")
    assert {r["category"] for r in group_c} == {"conversation_C"}
    assert {"ask_back", "fill", "carry"} <= {r["expect"] for r in group_c}


def test_a_reply_is_sent_with_the_continuation_its_ask_back_handed_back(monkeypatch):
    sent = []

    def _run_one(row, *, continuation=None, conversation_id=None, **_):
        sent.append((row["id"], continuation, conversation_id))
        handed = "tok-" + row["id"] if row.get("expect") == "ask_back" else None
        return {"id": row["id"], "category": row["category"],
                "question": row["question"], "outcome": "REFUSED",
                "route": None, "view": None, "seconds": 0.0, "served": "DECLINED",
                "serving_reason": "", "timing": {}, "answer": "",
                "_continuation": handed}
    monkeypatch.setattr(qb, "run_one", _run_one)
    rows = qb.conversation_rows("C")[:2]
    records = list(qb.play_conversations(rows, portfolio=None, lens=None,
                                         principal="p", stamp="S"))
    ask, reply = rows
    assert sent[0] == (ask["id"], None, f"S-{ask['conversation']}")
    assert sent[1] == (reply["id"], f"tok-{ask['id']}", f"S-{ask['conversation']}")
    assert sent[2][0] == f"{reply['id']}_twin" and sent[2][1] is None
    assert [r.get("twin_of") for r in records] == [None, None, reply["id"]]


def test_a_twin_asked_earlier_in_the_run_is_reused(monkeypatch):
    asked = []

    def _run_one(row, *, continuation=None, conversation_id=None, **_):
        asked.append(row["question"])
        return {"id": row["id"], "category": row["category"],
                "question": row["question"], "outcome": "ANSWERED",
                "route": None, "view": None, "seconds": 1.0, "served": "NEW",
                "serving_reason": "", "timing": {}, "answer": "x",
                "_continuation": "tok"}
    monkeypatch.setattr(qb, "run_one", _run_one)
    rows = [{"id": f"c{i}_t1", "conversation": f"c{i}", "turn": 1,
             "category": "conversation_A", "question": f"m{i}", "expect": "carry",
             "twin": "Show WA LTV by region."} for i in (1, 2)]
    records = list(qb.play_conversations(rows, portfolio=None, lens=None,
                                         principal="p", stamp="S"))
    assert asked.count("Show WA LTV by region.") == 1
    assert records[3]["reused_from"] == "c1_t1_twin"
    assert records[3]["twin_of"] == "c2_t1"


def test_the_production_script_offers_the_ask_back_conversations():
    text = _SCRIPT.read_text()
    assert 'askback) CONVERSATIONS="C"' in text
    assert 'conversations) CONVERSATIONS="all"' in text
    assert 'echo "--conversations ${CONVERSATIONS}"' in text
