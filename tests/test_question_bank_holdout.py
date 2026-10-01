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
