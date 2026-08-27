#!/usr/bin/env python3
"""migration_phase0/data_claim_audit.py — is every claim about the book TRUE?

A refusal can say two very different things:

    "I cannot do that."                     a claim about the SYSTEM
    "This book does not report that."       a claim about the CLIENT'S DATA

The second has to be true. Measured on the 166-question review pack, three
refusals said

    'interest rate bucket' is not available in this dataset. This book does
    not report it, so the question cannot be answered from the current data.

on a tape carrying `interest_rate_bucket` for 640 of 640 rows across five
bands. A refusal that is false about the client's data is worse than a
refusal: it is an assertion about their book that they may act on.

    python -m migration_phase0.data_claim_audit [--json out.json]

WHAT IT DOES. Runs a wide question surface deterministically, finds every
refusal whose stated reason is a claim about what the book CONTAINS, extracts
the thing named, and checks it against the tape. Four outcomes, kept apart
because only one is a lie:

    FALSE_about_the_book      the book holds what the refusal says it does not
    TRUE_about_the_book       the book genuinely lacks it
    TRUE_about_a_NAMED_FILTER a category the reader named that the book lacks.
                              TRUE, and honest — WHERE THE QUOTED WORD IS THE
                              READER'S. Where it is a function word the parser
                              mis-bound as a value it is still true and it
                              misleads: Q21C refuses with "no loans match that
                              filter ('among')" on a question whose reader
                              said nothing about `among`. This audit cannot
                              separate the two without a new reader of the
                              sentence, so it reports the class and names the
                              members; the separation is a judgement about the
                              BINDING, recorded against Q21C already.
    QUOTES_A_MANGLED_PHRASE   a fragment of the sentence quoted back as a name

EXITS NON-ZERO on any FALSE claim, and on any claim it cannot classify.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

FALSE_CLAIM = "FALSE_about_the_book"
TRUE_CLAIM = "TRUE_about_the_book"
UNASKED_FILTER = "TRUE_about_a_NAMED_FILTER"
MANGLED = "QUOTES_A_MANGLED_PHRASE"

#: Phrasings that assert something about the BOOK rather than the system.
DATA_CLAIM_PATTERNS = (
    (r"not available in this dataset", "a field is absent from the dataset"),
    (r"this book does not report", "the book does not report a field"),
    (r"does not carry", "the book does not carry something"),
    (r"no loans in this book match", "no rows match"),
    (r"not present in dataset columns", "a column is absent"),
    (r"this dataset does not", "the dataset lacks something"),
    (r"is not a governed portfolio for this book", "a book is unknown"),
)

#: Words that open a REQUEST, not a name. A refusal quoting one back has
#: mangled the sentence rather than found a thing the book lacks.
_REQUEST_OPENERS = ("break ", "show ", "give ", "plot ", "chart ", "cross-tab ")

_QUOTED = re.compile(r"'([^']{2,60})'")

#: PRE-REGISTERED. A name entering or leaving this set is the finding.
#:
#: EMPTY, and it was not empty. It held `'interest rate bucket'` — the only
#: false claim this audit has ever found across its 226 questions — until the
#: field was declared in `mi_agent/build_mi_semantics_registry.py::CURATION`
#: and the registry regenerated. Nothing about the tape changed: it carried
#: `interest_rate_bucket` on 640 of 640 rows in five bands the whole time, and
#: `config/mi/buckets.yaml` had defined the eight bands, and the chart path had
#: been grouping by the column. One missing declaration was the whole of it.
#:
#: The set is emptied rather than deleted. An empty pre-registration is the
#: strongest form of this instrument: any refusal that starts making a false
#: claim about a client's book, anywhere on the surface, fails the audit on the
#: first run, with the tape evidence printed. Do not repopulate it to make a
#: run pass — a name entering this set IS the finding.
EXPECTED_FALSE: Dict[str, str] = {}

PORTFOLIO = "client_001/mi_2026_06"


class AuditError(RuntimeError):
    """The audit could not run. Never absorbed into a pass."""


def _questions() -> List[Any]:
    import yaml

    out = []
    bank = yaml.safe_load((_REPO / "migration_phase0" /
                           "MI_FINAL_ACCEPTANCE_75.yaml").read_text())
    for case in bank["cases"]:
        for f in case["formulations"]:
            out.append(("BANK75", f["id"], f["q"]))
    ready = json.loads((_REPO / "migration_phase0" /
                        "MI_FINAL_LIVE_DATA_READINESS.json").read_text())
    for i, r in enumerate(ready["cfo_91"]["results"]):
        out.append(("CFO91", "CFO%02d" % (i + 1), r["question"]))
    for path, tag in (("SIMPLE_COMPOSITION_BANK.yaml", "SIMPLE"),
                      ("CFO_GENERALISATION_SUPPLEMENT.yaml", "GEN"),
                      ("ROBUSTNESS_RESIDUALS.yaml", "ROBUST")):
        doc = yaml.safe_load((_REPO / "migration_phase0" / path).read_text())
        items = doc.get("questions") or doc.get("cases") or []
        for i, e in enumerate(items):
            q = e.get("q") if isinstance(e, dict) else None
            if q:
                out.append((tag, e.get("id") or "%s%02d" % (tag, i + 1), q))
    return out


def _tape_carries(frame, term: str) -> Optional[Dict[str, Any]]:
    """Does the book carry the thing a refusal says it does not?"""
    probe = str(term or "").strip().lower()
    flat = re.sub(r"[\s_-]+", "", probe)
    for col in frame.columns:
        name = str(col).lower()
        if name == probe or re.sub(r"[\s_-]+", "", name) == flat:
            non_null = int(frame[col].notna().sum())
            return {"column": str(col), "rows_with_a_value": non_null,
                    "of": int(len(frame)),
                    "distinct": sorted({str(v) for v in
                                        frame[col].dropna().unique()})[:8]}
    return None


def classify(answer: str, error: str, frame) -> Optional[Dict[str, Any]]:
    text = answer if (error or "").strip() in answer else \
        (answer + " " + (error or ""))
    low = text.lower()
    for pattern, what in DATA_CLAIM_PATTERNS:
        if not re.search(pattern, low):
            continue
        terms = _QUOTED.findall(text)
        checks = [{"term": t, "tape": _tape_carries(frame, t)} for t in terms[:3]]
        if any(c["tape"] and c["tape"]["rows_with_a_value"] > 0 for c in checks):
            klass = FALSE_CLAIM
        elif re.search(r"no loans in this book match", low):
            klass = UNASKED_FILTER
        elif any(t.lower().startswith(_REQUEST_OPENERS) for t in terms):
            klass = MANGLED
        else:
            klass = TRUE_CLAIM
        return {"pattern": pattern, "what": what, "klass": klass,
                "quote": text.strip()[:300], "checked": checks,
                "quoted_terms": terms[:3]}
    return None


def run() -> Dict[str, Any]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    env = os.environ.get("MI_COMPLETENESS_FIXTURE", "/tmp/cfo_env")
    if not Path(env, "onboarding_output").is_dir():
        raise AuditError("AUDIT INVALID - fixture root %r has no "
                         "onboarding_output" % env)
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = "%s/onboarding_output" % env
    os.environ["TRAKT_PORTFOLIO_REGISTRY"] = "%s/portfolio_registry.yaml" % env
    os.environ.setdefault("MI_AGENT_PIPELINE_ROOT",
                          str(_REPO / "tests" / "fixtures" / "pipeline_history_5w"))
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"

    from fastapi.testclient import TestClient
    from mi_agent_api import datasets as _D
    from mi_agent_api import mi_service as S
    from mi_agent_api import workspace as W
    from mi_agent_api.app import app
    from mi_agent_api.dependencies import build_dependencies
    from migration_phase0.assurance_semantics import load_assurance_semantics

    cfg = _D._mi_llm_config()
    if cfg.enabled or cfg.available:
        raise AuditError("AUDIT INVALID - the free-form LLM parser arm is live; "
                         "set MI_AGENT_LLM_PARSER=off")
    semantics = load_assurance_semantics()
    frame, err = S._resolve_frame(build_dependencies().datasets, W.DEFAULT_VIEW,
                                  PORTFOLIO)
    if frame is None:
        raise AuditError("AUDIT INVALID - frame did not load: %s" % err)

    client = TestClient(app)
    claims: List[Dict[str, Any]] = []
    asked = 0
    for bank, qid, question in _questions():
        asked += 1
        env_ = client.post("/mi/query", json={"question": question,
                                              "portfolioId": PORTFOLIO,
                                              "asOfDate": "2026-06-30"}).json()
        if env_.get("ok"):
            continue
        found = classify(env_.get("answer") or "", env_.get("error") or "", frame)
        if found:
            claims.append({"bank": bank, "id": qid, "question": question, **found})

    by = Counter(c["klass"] for c in claims)
    false_terms = sorted({
        "'%s'" % chk["term"]
        for c in claims if c["klass"] == FALSE_CLAIM
        for chk in c["checked"]
        if chk["tape"] and chk["tape"]["rows_with_a_value"] > 0})
    return {"questions_asked": asked, "refusals_claiming_about_the_book": len(claims),
            "by_class": dict(by), "false_terms": false_terms,
            "expected_false": sorted(EXPECTED_FALSE),
            "matches_pre_registration": false_terms == sorted(EXPECTED_FALSE),
            "claims": claims}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="out")
    args = ap.parse_args(argv)

    result = run()
    print("data-claim audit over %d questions" % result["questions_asked"])
    print("  refusals whose reason is a claim about the book: %d"
          % result["refusals_claiming_about_the_book"])
    for k in (FALSE_CLAIM, TRUE_CLAIM, UNASKED_FILTER, MANGLED):
        print("    %-52s %d" % (k, result["by_class"].get(k, 0)))
    for c in result["claims"]:
        if c["klass"] != FALSE_CLAIM:
            continue
        print("\n  FALSE — %s  %s" % (c["id"], c["question"][:70]))
        print("     %s" % c["quote"][:200])
        for chk in c["checked"]:
            if chk["tape"]:
                print("     the tape carries `%s`: %d of %d rows, distinct %s"
                      % (chk["tape"]["column"], chk["tape"]["rows_with_a_value"],
                         chk["tape"]["of"], chk["tape"]["distinct"]))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print("\n  wrote %s" % args.out)

    ok = result["matches_pre_registration"]
    if not ok:
        print("\nFALSE-CLAIM SET MOVED — pre-registered %s, measured %s"
              % (result["expected_false"], result["false_terms"]))
    print("\n%s" % ("AUDIT HOLDS — every claim about the book is accounted for"
                    if ok else "AUDIT FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
