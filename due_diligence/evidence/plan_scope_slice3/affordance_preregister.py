#!/usr/bin/env python3
"""Pre-register the slice 3 AFFORDANCE bank, OFFLINE, before any model call.

WHAT CHANGED SINCE THE BOUNDARY BANK. The boundary run measured the model
against a contract that did not describe portfolio scope: `source_reference` was
an undescribed string, the prompt named neither the field nor the concept, and
the value-list guidance told the model in terms to raise a blocking ambiguity
for the one thing the named axis exists to express. It also had no way to know
which axis a bare "back book" sits on, because `back_book` is a governed
seasoning token and nothing said otherwise.

That contract has been repaired ONCE. This bank measures the repaired contract
ONCE. It is not the boundary bank re-run: the questions are the brief's ten, the
lifecycle axis is now under test, and the four named questions are asked against
a client registry the model can actually read.

WHAT IS PINNED. Only what the governed layer must be able to act on:

  * WHICH AXIS each phrase landed on — lifecycle (`population.base`), role
    (`population.lens`), identity (`population.source_reference`);
  * that a governed name stayed ATOMIC: the words inside a book's proper name
    did not also become a role, a seasoning or a filter;
  * that no physical identifier was authored;
  * that the rest of the question survived — measure, temporal intent, and the
    attribution capability where one is asked for.

DELIBERATELY NOT PINNED: `operation`, `time.form`, the statistic, and the exact
wording of `source_reference`. Scoring those would score the model against this
file's guess at its phrasing rather than against the governed semantics.

A SAFE CLARIFICATION IS NOT A SILENT FAILURE. It is recorded as CLARIFIED and
counted apart from a silent drop, widening or narrowing, which are the three
outcomes that reach a reader as a wrong answer. It is also not a pass: an axis
the contract now describes should resolve, not ask.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "affordance_bank_manifest.json"
HASH = HERE / "affordance_bank_manifest.sha256"

#: The same governed registry the boundary bank used, so the named cases are
#: comparable across the two runs. Production-shaped: an opaque id nobody says
#: out loud, a label that CONTAINS the words of two other governed axes, and one
#: declared alias that does the same.
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

LIFECYCLE, ROLE, NAMED, ATTRIBUTION = "lifecycle", "role", "named", "attribution"

CASES = [
    {"id": "M1", "question": "What is the back book balance?",
     "axis": LIFECYCLE, "base": "funded", "role": None, "named": False,
     "seasoning_forbidden": True,
     "why": "the generic phrase, alone. It names the funded population; "
            "reading it as a seasoning silently drops the front book"},
    {"id": "M2", "question": "What is in the front book?",
     "axis": LIFECYCLE, "base": "pipeline", "role": None, "named": False,
     "seasoning_forbidden": True,
     "why": "the mirror. New originations are the pipeline population, not a "
            "vintage of the funded one"},
    {"id": "M3", "question": "What is the acquired back book balance?",
     "axis": ROLE, "base": "funded", "role": "acquired", "named": False,
     "seasoning_forbidden": True,
     "why": "two axes in one phrase: lifecycle AND role. Both, and no third"},
    {"id": "M4", "question": "What is the direct back book balance?",
     "axis": ROLE, "base": "funded", "role": "direct", "named": False,
     "seasoning_forbidden": True,
     "why": "the same, the other role. 'back book' must not drag in 'acquired'"},
    {"id": "M5", "question": "What is the balance of purchased loans?",
     "axis": ROLE, "base": None, "role": "acquired", "named": False,
     "why": "role language without the word 'acquired'"},
    {"id": "M6", "question": "What is the balance of originated loans?",
     "axis": ROLE, "base": None, "role": "direct", "named": False,
     "why": "role language without the word 'direct'"},
    {"id": "M7", "question": "What is funded balance for ALP Acquired Back Book?",
     "axis": NAMED, "base": "funded", "role": None, "named": True,
     "resolves_to": "alp_acquired", "atomic": True, "seasoning_forbidden": True,
     "why": "the canonical label. Its words are its NAME — not a role and not "
            "a seasoning. This is the case that would have served wrong"},
    {"id": "M8", "question": "What is funded balance for the ALP back book?",
     "axis": NAMED, "base": "funded", "role": None, "named": True,
     "resolves_to": "alp_acquired", "atomic": True, "seasoning_forbidden": True,
     "why": "a declared alias, which the model can now read. 'back book' here "
            "is part of a name, not the lifecycle phrase"},
    {"id": "M9", "question": "Show ALP Acquired Back Book funded balance each month.",
     "axis": NAMED, "base": "funded", "role": None, "named": True,
     "resolves_to": "alp_acquired", "atomic": True, "seasoning_forbidden": True,
     "temporal": True,
     "why": "a named source must cost neither the series nor its grain"},
    {"id": "M10",
     "question": "How much of this month's increase came from the acquired back book?",
     "axis": ATTRIBUTION, "base": "funded", "role": "acquired", "named": False,
     "seasoning_forbidden": True, "temporal": True,
     "capability_not": ["generic_analysis"],
     "why": "THE control. Now that three axes are readable in one phrase, the "
            "question must still be an attribution and not a scoped balance"},
]

MANIFEST_BODY = {
    "bank_id": "slice3_affordance_v1",
    "what_this_is": "does claude-opus-5 place portfolio language on the right "
                    "governed axis now that the contract describes the axes",
    "model": "claude-opus-5",
    "authorised_live_calls": len(CASES),
    "calls_per_case": 1,
    "retries": 0,
    "boundary": "question -> Opus -> CandidateIntent -> compiler (+ client "
                "registry) -> plan. No execution, no MI API, no serving path.",
    "registry": FIXTURE_REGISTRY,
    "registry_is_visible_to_the_model": True,
    "not_pinned": ["operation", "time.form", "statistic",
                   "exact wording of source_reference",
                   "capability (except the attribution control)"],
    "compile_checked": [c["id"] for c in CASES if c["axis"] != ATTRIBUTION],
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
