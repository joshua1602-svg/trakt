#!/usr/bin/env python3
"""Assemble the 135-question interpretation benchmark from FROZEN sources.

WHY THIS SCRIPT EXISTS, AND WHAT IT DOES NOT DO
------------------------------------------------
The brief asks for "the existing frozen 45 canonical x 3 variant = 135 question
bank". At the base commit no such bank exists. What exists is four frozen banks,
three of which already carry the exact "one canonical, three independent
formulations" structure the brief describes, and one which carries paraphrase
families:

    migration_phase0/MI_FINAL_ACCEPTANCE_75.yaml            25 canonical x 3
    tests/fixtures/mi_query_stage_movement/
        STAGE_MOVEMENT_BANK.yaml                             9 canonical x 3
    due_diligence/evidence/analytical_intent_v1/nl_bank.py   9 canonical x 3
    migration_phase0/BORROWING_BASE_MI_BANK.yaml             2 canonical x 3
                                                            --------------
                                                            45 canonical x 3

So the bank is ASSEMBLED, never authored. Every question string is copied
verbatim from a committed artefact by this script, and the provenance of each
one (source file, source id) is written into the frozen output beside it. Not a
word is edited, reordered inside a sentence, or invented. Re-running this script
against the same commit reproduces the same bank byte for byte, and
``tests/interpretation_v2/test_bank_is_verbatim.py`` re-reads every source and
fails if a single character has drifted.

Where a source offers more than three formulations, the first three in the
source's own order are taken — a rule, not a selection, so nothing was picked
for how well it scores.

    python -m mi_agent.interpretation_v2.banks.build_bank_135
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

BANK75 = _REPO_ROOT / "migration_phase0" / "MI_FINAL_ACCEPTANCE_75.yaml"
STAGE_MOVEMENT = (_REPO_ROOT / "tests" / "fixtures" / "mi_query_stage_movement"
                  / "STAGE_MOVEMENT_BANK.yaml")
BORROWING_BASE = _REPO_ROOT / "migration_phase0" / "BORROWING_BASE_MI_BANK.yaml"
NL_BANK = (_REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
           / "nl_bank.py")

OUTPUT = Path(__file__).resolve().parent / "interpretation_bank_135.yaml"

#: The two borrowing-base canonicals, named by the SOURCE IDS whose questions
#: form each triplet. Chosen because each triplet is three different wordings of
#: one intent in the source bank's own grouping — headroom, and the bridge.
BORROWING_BASE_TRIPLETS = (
    ("BB01", "borrowing-base headroom", ("C03", "C04", "S05")),
    ("BB02", "borrowing-base bridge", ("B01", "B03", "B04")),
)

#: nl_bank's Q8 is parameterised by population pair. The provenance pair is the
#: one every book carries, so it is the one taken — the same rule the source
#: itself applies when it holds the phrasing axis constant.
NL_Q8_PAIR = "provenance"
NL_Q8_BOOK = "alderbridge"


def _canonical(cid: str, family: str, shape: str, source: str,
               rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if len(rows) != 3:
        raise SystemExit(f"{cid}: expected 3 variants, got {len(rows)}")
    return {"id": cid, "family": family, "shape": shape, "source": source,
            "variants": rows}


def _from_bank75() -> List[Dict[str, Any]]:
    data = yaml.safe_load(BANK75.read_text(encoding="utf-8"))
    out = []
    for case in data["cases"]:
        rows = [{"variant": chr(ord("A") + i), "question": f["q"],
                 "source_id": f["id"]}
                for i, f in enumerate(case["formulations"][:3])]
        out.append(_canonical(case["id"], case.get("family", ""),
                              case.get("shape", ""),
                              "migration_phase0/MI_FINAL_ACCEPTANCE_75.yaml", rows))
    return out


def _from_stage_movement() -> List[Dict[str, Any]]:
    data = yaml.safe_load(STAGE_MOVEMENT.read_text(encoding="utf-8"))
    out = []
    for case in data["cases"]:
        rows = [{"variant": chr(ord("A") + i), "question": f["q"],
                 "source_id": f["id"]}
                for i, f in enumerate(case["formulations"][:3])]
        out.append(_canonical(case["id"], "pipeline_stage_movement",
                              case.get("subtype", ""),
                              "tests/fixtures/mi_query_stage_movement/"
                              "STAGE_MOVEMENT_BANK.yaml", rows))
    return out


def _from_borrowing_base() -> List[Dict[str, Any]]:
    data = yaml.safe_load(BORROWING_BASE.read_text(encoding="utf-8"))
    by_id = {q["id"]: q["question"] for q in data["questions"]}
    out = []
    for cid, shape, source_ids in BORROWING_BASE_TRIPLETS:
        rows = [{"variant": chr(ord("A") + i), "question": by_id[sid],
                 "source_id": sid} for i, sid in enumerate(source_ids)]
        out.append(_canonical(cid, "borrowing_base", shape,
                              "migration_phase0/BORROWING_BASE_MI_BANK.yaml", rows))
    return out


def _load_nl_bank():
    """Load ``nl_bank.py`` by path.

    ``due_diligence/`` carries no ``__init__.py`` and this sprint does not add
    files outside its own boundary to make an import tidier, so the module is
    loaded from its file instead.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_nl_bank_frozen", NL_BANK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _from_nl_bank() -> List[Dict[str, Any]]:
    nl_bank = _load_nl_bank()

    out = []
    for intent in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q9"):
        rows = [{"variant": chr(ord("A") + i), "question": q,
                 "source_id": f"{intent}.{i + 1}"}
                for i, q in enumerate(nl_bank.VARIATIONS[intent][:3])]
        out.append(_canonical(f"NL{intent[1:]}", "analytical_intent",
                              nl_bank.CANONICAL[intent],
                              "due_diligence/evidence/analytical_intent_v1/"
                              "nl_bank.py", rows))
    q8 = [row for row in nl_bank.q8_variations(NL_Q8_BOOK)
          if row["pair"] == NL_Q8_PAIR][:3]
    rows = [{"variant": chr(ord("A") + i), "question": row["question"],
             "source_id": f"Q8.{NL_Q8_PAIR}.{row['variation']}"}
            for i, row in enumerate(q8)]
    out.append(_canonical("NL8", "analytical_intent", nl_bank.CANONICAL["Q8"],
                          "due_diligence/evidence/analytical_intent_v1/nl_bank.py",
                          rows))
    return out


def build() -> Dict[str, Any]:
    canonicals = (_from_bank75() + _from_stage_movement()
                  + _from_nl_bank() + _from_borrowing_base())
    questions = sum(len(c["variants"]) for c in canonicals)
    if len(canonicals) != 45 or questions != 135:
        raise SystemExit(f"expected 45 canonical / 135 questions, "
                         f"got {len(canonicals)} / {questions}")
    return {
        "version": 1,
        "note": ("135 questions = 45 canonical x 3 variants, every string copied "
                 "verbatim from a frozen committed bank. Assembled by "
                 "mi_agent/interpretation_v2/banks/build_bank_135.py; not "
                 "authored, not edited. This is an INTERPRETATION benchmark: it "
                 "scores what a question MEANS, never what the answer is."),
        "sources": sorted({c["source"] for c in canonicals}),
        "canonical_count": len(canonicals),
        "question_count": questions,
        "canonicals": canonicals,
    }


def main() -> None:
    bank = build()
    OUTPUT.write_text(
        yaml.safe_dump(bank, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(_REPO_ROOT)}: "
          f"{bank['canonical_count']} canonical / {bank['question_count']} questions")


if __name__ == "__main__":
    main()
