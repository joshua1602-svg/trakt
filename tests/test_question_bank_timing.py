"""The bank log says where each question's time went (P0 design §24.2)."""
from __future__ import annotations

from mi_agent_api import question_bank as qb


def test_each_question_prints_its_heaviest_stages_and_a_summary(
        monkeypatch, tmp_path, capsys):
    bank = tmp_path / "bank.yaml"
    bank.write_text("- {id: q1, category: c, question: 'What is the balance?'}\n"
                    "- {id: q2, category: c, question: 'And by region?'}\n")
    timings = iter([{"total_ms": 9000.0,
                     "stages": {"governed.interpret_and_compile": 5000.0,
                                "mi_query.governed_attempt": 8000.0}},
                    {"total_ms": 3000.0,
                     "stages": {"governed.interpret_and_compile": 2000.0}}])

    def fake_run_one(row, **_):
        return {"id": row["id"], "category": row["category"],
                "question": row["question"], "outcome": "ANSWERED",
                "route": None, "view": None, "seconds": 1.0, "served": "NEW",
                "serving_reason": "", "timing": next(timings), "answer": "ok"}

    monkeypatch.setattr(qb, "run_one", fake_run_one)
    monkeypatch.setattr(qb, "_capture_serving", lambda: None)
    assert qb.main(["--bank", str(bank), "--categories", "all",
                    "--out", str(tmp_path / "out.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "time: mi_query.governed_attempt 8.0s, " \
           "governed.interpret_and_compile 5.0s" in out
    assert "TIME BY STAGE" in out
    line = next(l for l in out.splitlines()
                if l.strip().startswith("governed.interpret_and_compile"))
    assert line.split()[1:] == ["3.5", "5.0", "2"]
