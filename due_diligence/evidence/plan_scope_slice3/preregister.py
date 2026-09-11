#!/usr/bin/env python3
"""Pre-register the slice 3 model-boundary bank, OFFLINE, before any model call.

THE ONE QUESTION. The deterministic scope infrastructure is proved: roles bind,
names resolve, the registry is client-scoped, and the whole chain composes with
slice 1 and slice 2. None of that says whether **Opus** states portfolio scope in
a shape the deterministic layer can act on. That is the only question this bank
exists to answer, and the boundary under test is exactly:

    raw question -> Opus -> CandidateIntent -> compiler (+ client registry)
                 -> GovernedQueryPlan -> scope predicates
    STOP.

Nothing is executed against a book.

WHAT IS PINNED, AND WHAT DELIBERATELY IS NOT. Pinning `operation`, or a
particular `time.form`, or the exact statistic would score Opus against this
file's guess at its wording. Two readings are routinely both correct —
`point_in_time` and `summary` for "what is funded balance", `series` and `range`
for a monthly span. What is pinned is what the governed layer must be able to
ACT on:

  * WHICH SCOPE the intent states (total / role / named), because that decides
    the population the answer is computed over;
  * that the NAMED cases carry the reader's phrase in `source_reference` rather
    than a physical id, because authoring an id is the model deciding which
    dataset to read;
  * that the measure, ordinary filters, dimensions and temporal intent SURVIVE
    alongside the scope, because a scope that costs the rest of the question is
    the silent drop this bank is really for;
  * that the two attribution questions keep an attribution capability, because
    recognising the words "acquired book" must not turn "how much of the
    increase came from it" into a plain scoped balance.

A SAFE REFUSAL IS NOT A FAILURE. A clarify or refuse that loses no semantics
scores better here than a confident answer over a wider population.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "model_bank_manifest.json"
HASH = HERE / "model_bank_manifest.sha256"

#: The governed registry the compile check runs against. Production-shaped: an
#: opaque client id, a label nobody says out loud, and one declared alias.
FIXTURE_REGISTRY = {
    "client_id": "ERE",
    "portfolios": [
        {"source_portfolio_id": "alp_origination", "source_portfolio_type": "direct",
         "source_portfolio_label": "ALP Originations", "aliases": []},
        {"source_portfolio_id": "alp_acquired", "source_portfolio_type": "acquired",
         "source_portfolio_label": "ALP Acquired Back Book",
         "aliases": ["ALP back book"]},
        {"source_portfolio_id": "nbs_acquired", "source_portfolio_type": "acquired",
         "source_portfolio_label": "NBS Acquired", "aliases": []},
    ],
}

TOTAL, ROLE, NAMED, ATTRIBUTION = "total", "role", "named", "attribution"

CASES = [
    {"id": "S3-M01", "question": "What is funded balance?",
     "scope": TOTAL, "role": None, "named": False,
     "why": "no scope stated — the default must stay the whole funded book"},
    {"id": "S3-M02",
     "question": "What is total funded balance across the whole book?",
     "scope": TOTAL, "role": None, "named": False,
     "why": "scope stated EXPLICITLY as total — must not narrow to a role"},
    {"id": "S3-M03", "question": "What is funded balance for the direct book?",
     "scope": ROLE, "role": "direct", "named": False,
     "why": "the direct role; must not be read as a named portfolio"},
    {"id": "S3-M04",
     "question": "What is funded balance for the acquired portfolio?",
     "scope": ROLE, "role": "acquired", "named": False,
     "why": "'portfolio' is the word a named source would use, and this is "
            "still a role"},
    {"id": "S3-M05",
     "question": "Show loan count by LTV bucket for the acquired book.",
     "scope": ROLE, "role": "acquired", "named": False,
     "dimensions": ["ltv_bucket"],
     "why": "a role must not cost the grouping"},
    {"id": "S3-M06",
     "question": "How many acquired drawdown loans have LTV above 50%?",
     "scope": ROLE, "role": "acquired", "named": False,
     "filters": ["erm_product_type", "current_loan_to_value"],
     "why": "a role must not cost either ordinary predicate"},
    {"id": "S3-M07",
     "question": "What is funded balance for ALP Acquired Back Book?",
     "scope": NAMED, "role": None, "named": True, "resolves_to": "alp_acquired",
     "why": "the canonical label; the phrase belongs in source_reference and "
            "the id is the compiler's to produce"},
    {"id": "S3-M08",
     "question": "What is funded balance for the ALP back book?",
     "scope": NAMED, "role": None, "named": True, "resolves_to": "alp_acquired",
     "why": "a declared alias; the model need not know it resolves — only that "
            "a portfolio was named"},
    {"id": "S3-M09",
     "question": "Show ALP Acquired Back Book funded balance each month.",
     "scope": NAMED, "role": None, "named": True, "resolves_to": "alp_acquired",
     "temporal": True,
     "why": "a named source must not cost the series"},
    {"id": "S3-M10",
     "question": "What was ALP Acquired Back Book balance this period versus "
                 "the previous period?",
     "scope": NAMED, "role": None, "named": True, "resolves_to": "alp_acquired",
     "temporal": True,
     "why": "a named source must not cost the comparison"},
    {"id": "S3-M11",
     "question": "How much of this month's increase came from the acquired book?",
     "scope": ATTRIBUTION, "role": None, "named": False,
     "capability_not": ["generic_analysis"],
     "why": "THE control. Recognising 'acquired book' must not turn an "
            "attribution question into a plain scoped balance"},
    {"id": "S3-M12",
     "question": "Why did the acquired portfolio balance change?",
     "scope": ATTRIBUTION, "role": None, "named": False,
     "capability_not": ["generic_analysis"],
     "why": "the same control in the 'why' form; must not become a series"},
]

MANIFEST_BODY = {
    "bank_id": "slice3_model_boundary_v1",
    "what_this_is": "can claude-opus-5 state governed portfolio scope in a "
                    "shape the deterministic layer can act on",
    "model": "claude-opus-5",
    "authorised_live_calls": len(CASES),
    "calls_per_case": 1,
    "retries": 0,
    "boundary": "question -> Opus -> CandidateIntent -> compiler -> plan. "
                "No execution, no MI API, no serving path.",
    "registry": FIXTURE_REGISTRY,
    "not_pinned": ["operation", "capability (except the two attribution "
                   "controls)", "time.form", "statistic", "exact wording of "
                   "source_reference"],
    "compile_checked": [c["id"] for c in CASES if c["scope"] != ATTRIBUTION],
    "cases": CASES,
}


def main() -> int:
    body = json.dumps(MANIFEST_BODY, indent=2, sort_keys=True) + "\n"
    MANIFEST.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    HASH.write_text(f"{digest}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"registered {len(CASES)} cases")
    print(f"  {MANIFEST}")
    print(f"  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
