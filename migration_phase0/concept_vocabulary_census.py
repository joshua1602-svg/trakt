#!/usr/bin/env python3
"""migration_phase0/concept_vocabulary_census.py — what may the model name?

READ-ONLY, deterministic, NO MODEL CALL. Measures the concept vocabulary the
model is offered and what the registry does with every term in it.

    python -m migration_phase0.concept_vocabulary_census [--json out.json]

The question it answers is the Stage 2 constraint, restated as something
countable: CAN THE MODEL REACH A GOVERNED FIELD THE REGISTRY WOULD NOT CHOOSE?

The measured target is the Opus run, where the model did both halves and bound
`lump sum` to `erm_sub_product_type` and `drawdown` to `account_status`. The
book's own catalogue claims each for `erm_product_type` and for nothing else.

EXITS NON-ZERO if any invariant below stops holding. An instrument that cannot
fail is not assurance.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class CensusError(RuntimeError):
    """The census could not be measured. Never absorbed into a pass."""


#: The Opus run's two mis-bindings, and what the registry must do instead.
OPUS_BREAKS = (
    ("lump sum", "category_value", "erm_product_type",
     "Opus bound this to erm_sub_product_type"),
    ("drawdown", "category_value", "erm_product_type",
     "Opus bound this to account_status"),
)

#: Terms that must NOT be proposable, with the reason.
MUST_NOT_BE_PROPOSABLE = {
    "erm sub product type": "a registered dimension this tape does not carry",
    "platinum": "no governed field claims it",
    "lump summ": "one character from a governed value, and a different thing",
}

#: PRE-REGISTERED, measured at the head this file was written on.
EXPECTED = {
    "category_value": 39, "measure": 53, "dimension": 63,
    "source_book": 3, "dataset": 3,
    "ambiguous_within_kind": 1, "cross_kind_collisions": 5,
    "raw_field_keys_offered": 0, "off_tape_fields_offered": 0,
    "opus_breaks_reachable": 0,
    #: Terms the registry offers that the question-shaped owner will not bind.
    #: Withheld from the model rather than dropped silently, and counted here so
    #: the disagreement between the two stays visible.
    "withheld": 2,
}

PORTFOLIO = "client_001/mi_2026_06"


def _book():
    env = os.environ.get("MI_COMPLETENESS_FIXTURE", "/tmp/cfo_env")
    if not Path(env, "onboarding_output").is_dir():
        raise CensusError(
            "CENSUS INVALID - fixture root %r has no onboarding_output. Set "
            "MI_COMPLETENESS_FIXTURE to the acceptance fixture." % env)
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = "%s/onboarding_output" % env
    os.environ["TRAKT_PORTFOLIO_REGISTRY"] = "%s/portfolio_registry.yaml" % env
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"

    from mi_agent_api import mi_service as S
    from mi_agent_api import workspace as W
    from mi_agent_api.dependencies import build_dependencies
    from migration_phase0.assurance_semantics import load_assurance_semantics

    semantics = load_assurance_semantics()
    frame, err = S._resolve_frame(build_dependencies().datasets, W.DEFAULT_VIEW,
                                  PORTFOLIO)
    if frame is None:
        raise CensusError("CENSUS INVALID - frame did not load: %s" % err)
    values = S._book_values(frame, semantics)
    if not values:
        raise CensusError(
            "CENSUS INVALID - the book's value catalogue loaded EMPTY; every "
            "value term would silently be absent and the census would report a "
            "clean vocabulary containing nothing")
    return frame, semantics, values


def run() -> Dict[str, Any]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    from question_interpretation import concept_proposal as CP

    frame, semantics, values = _book()
    columns = set(frame.columns)
    vocab = CP.vocabulary(semantics, available_values=values,
                          available_columns=columns)

    offered = {t for terms in vocab.terms.values() for t in terms}
    keys = set(semantics.get("fields") or {})
    raw_keys = sorted(offered & keys)
    off_tape = sorted(
        t for t in (vocab.terms.get(CP.KIND_DIMENSION) or ())
        if t.replace(" ", "_") in keys and t.replace(" ", "_") not in columns)

    breaks: List[Dict[str, Any]] = []
    for term, kind, expected_field, note in OPUS_BREAKS:
        bound, rejected = CP.bind([CP.ProposedConcept(kind, term)], vocab)
        got = bound[0].field if bound else None
        breaks.append({"term": term, "kind": kind, "bound_to": got,
                       "expected": expected_field, "note": note,
                       "reachable_wrong_field": bool(got and got != expected_field),
                       "rejected": [r.as_dict() for r in rejected]})

    forbidden: List[Dict[str, Any]] = []
    for term, why in sorted(MUST_NOT_BE_PROPOSABLE.items()):
        rows = []
        for kind in CP.CONCEPT_KINDS:
            bound, rejected = CP.bind([CP.ProposedConcept(kind, term)], vocab)
            rows.append({"kind": kind,
                         "bound_to": bound[0].field if bound else None,
                         "rejected": rejected[0].reason if rejected else None})
        forbidden.append({"term": term, "why": why, "by_kind": rows,
                          "bound_anywhere": any(r["bound_to"] for r in rows)})

    # EVERY OFFERED TERM BINDS. A vocabulary that offers a term the registry
    # then refuses hands the model a trap, and the refusal would look like the
    # model's fault.
    unbindable: List[Dict[str, str]] = []
    for kind, terms in vocab.terms.items():
        for term in terms:
            bound, rejected = CP.bind([CP.ProposedConcept(kind, term)], vocab)
            if not bound:
                unbindable.append({"kind": kind, "term": term,
                                   "reason": rejected[0].reason})

    measured = {k: len(vocab.terms.get(k) or ()) for k in CP.CONCEPT_KINDS}
    measured["ambiguous_within_kind"] = sum(
        len(v) for v in vocab.ambiguous.values())
    measured["cross_kind_collisions"] = len(vocab.cross_kind)
    measured["raw_field_keys_offered"] = len(raw_keys)
    measured["off_tape_fields_offered"] = len(off_tape)
    measured["opus_breaks_reachable"] = sum(
        1 for b in breaks if b["reachable_wrong_field"])
    measured["withheld"] = sum(len(v) for v in vocab.withheld.values())

    return {
        "measured": measured, "pre_registered": EXPECTED,
        "matches_pre_registration": measured == EXPECTED,
        "vocabulary": vocab.as_dict(),
        "raw_field_keys_offered": raw_keys,
        "off_tape_fields_offered": off_tape,
        "opus_breaks": breaks,
        "must_not_be_proposable": forbidden,
        "offered_but_unbindable": unbindable,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="out")
    args = ap.parse_args(argv)

    result = run()
    m = result["measured"]
    print("concept vocabulary census")
    for kind in ("category_value", "measure", "dimension", "source_book", "dataset"):
        print("  %-16s %3d terms" % (kind, m[kind]))
    print("  ambiguous within a kind : %d  %s"
          % (m["ambiguous_within_kind"], result["vocabulary"]["ambiguous"]))
    print("  cross-kind collisions   : %d  %s"
          % (m["cross_kind_collisions"],
             sorted(result["vocabulary"]["cross_kind"])))
    print("  raw field keys offered  : %d" % m["raw_field_keys_offered"])
    print("  off-tape fields offered : %d" % m["off_tape_fields_offered"])
    print("  offered but unbindable  : %d" % len(result["offered_but_unbindable"]))
    print("  withheld (offered by the registry, refused by the owner): %d  %s"
          % (m["withheld"], result["vocabulary"]["withheld"]))
    print("  the Opus mis-bindings:")
    for b in result["opus_breaks"]:
        print("     %-12s -> %-24s (expected %s)  %s"
              % (b["term"], b["bound_to"], b["expected"], b["note"]))
    print("  must not be proposable:")
    for f in result["must_not_be_proposable"]:
        print("     %-22s bound anywhere: %s" % (f["term"], f["bound_anywhere"]))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print("  wrote %s" % args.out)

    ok = True
    if m != result["pre_registered"]:
        ok = False
        print("\nCENSUS MOVED — pre-registered %s, measured %s"
              % (result["pre_registered"], m))
    if result["offered_but_unbindable"]:
        ok = False
        print("\nOFFERED BUT UNBINDABLE — the vocabulary hands the model a "
              "trap: %s" % result["offered_but_unbindable"][:10])
    for f in result["must_not_be_proposable"]:
        if f["bound_anywhere"]:
            ok = False
            print("\nBOUND WHAT MUST NOT BIND — %r (%s): %s"
                  % (f["term"], f["why"], f["by_kind"]))
    print("\n%s" % ("CENSUS HOLDS" if ok else "CENSUS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
