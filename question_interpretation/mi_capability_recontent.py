#!/usr/bin/env python3
"""question_interpretation/mi_capability_recontent.py

THE CAPABILITY RECORD, RE-RATED FROM ARTIFACT CONTENTS.

Why this exists
---------------
The eight time-series shapes were rated by `time_series_surface`, which decides
both limbs of a rating by matching COLUMN NAMES:

* a time axis is a column whose NAME contains `period`, `month`, `quarter`, ...
* a requested dimension is a column whose NAME contains `region`, `seasoning`,
  `source_portfolio`, ...

Neither is what the brief's rule says, and neither is what a reader of the
artifact would conclude. Three artifacts carry both limbs under names that match
neither list, and were rated ABSENT with the reason "neither a time axis nor the
requested breakdown" — a statement wrong on both counts.

**The previously published ratings are therefore FLOORS set by the instrument,
not ceilings set by the product.** A shape rated ABSENT means "the name-matching
surface could not see it", which is a strictly weaker claim than "the product
cannot do it". This module re-rates the same runs on CONTENTS:

* **time axis** — a time column with >1 distinct value, OR a column pair naming
  the two ends of a movement (`prior`/`current`, `start`/`end`,
  `opening`/`closing`, `previous`/`latest`). The brief's rule, in full.
* **requested dimension** — a column whose VALUES are the requested dimension's
  values, whatever the column is called. The domain is resolved from the dataset
  itself (`collateral_geography` -> the UK regions; `source_portfolio_type` ->
  direct/acquired), so `category` holding twelve UK regions counts as a region
  breakdown and `population` holding Direct/Acquired counts as a source split.

It also answers the routing question: for every phrasing that does NOT deliver,
does a SIBLING phrasing of the same shape deliver? A shape's phrasings are the
same request in different words, so a phrasing that fails while its sibling
succeeds is a ROUTING gap, not a capability gap.

    python -m question_interpretation.mi_capability_recontent --book alderbridge
    python -m question_interpretation.mi_capability_recontent --book kestrelmoor --llm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from question_interpretation.time_series_surface import (  # noqa: E402
    ABSENT, PARTIAL, PROVEN, SHAPES, _TIME_HINTS, shape_rating)
from question_interpretation.mi_gate_llm_arm import (  # noqa: E402
    _MOVEMENT_ENDS, movement_pair)

#: A phrasing DELIVERS when the artifact carries every limb the shape asks for.
DELIVERS, REFUSED, INCOMPLETE = "DELIVERS", "REFUSED", "INCOMPLETE"

#: Why a phrasing that does not deliver does not deliver.
ROUTING_GAP = "ROUTING GAP — a sibling phrasing of the same shape delivers"
CAPABILITY_GAP = "CAPABILITY GAP — no phrasing of this shape delivers"


# --------------------------------------------------------------------------- #
# Preflight — the two traps that make a whole run read ABSENT
# --------------------------------------------------------------------------- #
def preflight(book: str) -> List[str]:
    """Refuse to measure in an environment that silently rates everything ABSENT.

    THE SAME CLASS OF TRAP AS THE BOOK FALLBACK in
    `shipped_shapes._apply_env`: a misconfiguration that does not fail, but
    answers plausibly from the wrong footing. Both of these produce a clean,
    quotable, entirely wrong table.

    1. `TRAKT_RUNTIME_MODE` defaults to `production`, under which
       `trakt_core.policy` refuses both books as synthetic fixtures. Every one
       of the 29 phrasings then refuses with `route=None` and every shape rates
       ABSENT — which reads as a finding.
    2. `demo_platform/workspace/` is gitignored and regenerable. Without it the
       book carries a single period, so no artifact can show a time axis and
       even T1 rates ABSENT.
    """
    problems: List[str] = []
    from trakt_core.runtime import runtime_mode
    mode = runtime_mode()
    if mode == "production":
        problems.append(
            "TRAKT_RUNTIME_MODE resolves to 'production'; trakt_core.policy will "
            "refuse both books as synthetic fixtures and EVERY shape will rate "
            "ABSENT. Set TRAKT_RUNTIME_MODE=development (or test).")
    if book == "alderbridge":
        from demo_platform import config as cfg
        root = Path(cfg.local_blob_root())
        if not root.exists():
            problems.append(
                "demo_platform workspace is missing (%s). The book will carry a "
                "single period and even T1 will rate ABSENT. Rebuild it with: "
                "python -m demo_platform.run_demo --generate --onboard "
                "--orchestrate" % root)
    return problems


# --------------------------------------------------------------------------- #
# Dimension domains — resolved from the DATA, not from column names
# --------------------------------------------------------------------------- #
def _norm(v: Any) -> str:
    return str(v).strip().lower()


def dimension_domains(fragments: List[str]) -> Dict[str, Set[str]]:
    """fragment -> the set of values that dimension actually takes in the book.

    Resolved by finding the dataset columns whose name carries the fragment and
    taking their distinct values. Name matching is used HERE, against the
    dataset schema where the canonical names live — never against the artifact,
    where the presentation layer is free to call a column whatever it likes.
    """
    from mi_agent_api import data_source
    df = data_source.get_dataframe()
    out: Dict[str, Set[str]] = {}
    for frag in fragments:
        vals: Set[str] = set()
        for col in df.columns:
            if frag not in col.lower():
                continue
            distinct = {_norm(v) for v in df[col].dropna().unique()}
            # A high-cardinality code column (172 NUTS codes) is not the
            # reporting dimension a question means; the reporting label is.
            if 1 < len(distinct) <= 60:
                vals |= distinct
        if vals:
            out[frag] = vals
    return out


def column_carries(values: List[str], domain: Set[str]) -> List[str]:
    """The artifact values that ARE this dimension's values.

    Tolerant of presentation labelling: the book stores `Front Book`, the
    artifact renders `Front Book (0-12 months)`; the book stores `direct`, the
    artifact renders `Direct`. Matched on normalised containment either way, so
    a decorated label still counts and an unrelated value still does not.
    """
    hits = []
    for v in values:
        n = _norm(v)
        if not n:
            continue
        if any(n == d or d in n or n in d for d in domain):
            hits.append(v)
    return hits


# --------------------------------------------------------------------------- #
# The content rating
# --------------------------------------------------------------------------- #
def time_axis(census: Dict[str, Any]) -> Optional[str]:
    """How the artifact carries time, or None. The brief's rule, in full."""
    pair = census.get("movement_pair")
    if pair:
        return "movement pair %s/%s" % tuple(pair)
    for col, meta in census.get("columns", {}).items():
        if any(h in col.lower() for h in _TIME_HINTS) and meta["distinct"] > 1:
            return "%d distinct values in %s" % (meta["distinct"], col)
    return None


def rate_content(ok: bool, want: List[str], census: Dict[str, Any],
                 domains: Dict[str, Set[str]]) -> Tuple[str, str, Dict[str, Any]]:
    """PROVEN / ABSENT on CONTENTS, with the evidence that decided it."""
    evidence: Dict[str, Any] = {"time": None, "dimension": None}
    if not ok:
        return ABSENT, "refused", evidence
    t = time_axis(census)
    evidence["time"] = t
    if not want:
        return ((PROVEN, "time axis: %s" % t, evidence) if t
                else (ABSENT, "answered with no time axis in the artifact", evidence))
    if not t:
        return ABSENT, "no time axis in the artifact", evidence
    # Does ANY column carry the requested dimension, whatever it is called?
    for col, meta in census.get("columns", {}).items():
        for frag in want:
            domain = domains.get(frag)
            if not domain:
                continue
            hits = column_carries(meta.get("values") or meta.get("sample") or [],
                                  domain)
            if len(set(map(_norm, hits))) > 1:
                evidence["dimension"] = {
                    "column": col, "requested": frag,
                    "distinct_matched": len(set(map(_norm, hits))),
                    "values": sorted(set(hits))[:8]}
                return (PROVEN,
                        "time axis: %s | %r carries %s (%d values: %s)"
                        % (t, col, frag, len(set(map(_norm, hits))),
                           ", ".join(sorted(set(hits))[:5])), evidence)
    carrying = {c: m["distinct"] for c, m in census.get("columns", {}).items()
                if m["distinct"] > 1}
    return (ABSENT,
            "time axis present (%s) but NO column carries %s; columns with >1 "
            "value: %s" % (t, want, carrying or "none"), evidence)


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #
def full_census(artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Every column of every artifact, with its DISTINCT VALUES (capped).

    All artifacts are merged rather than one being chosen: a response may carry
    a ranked table and a detail table, and the question is whether the RESPONSE
    delivered, not whether one chosen table did.
    """
    columns: Dict[str, Any] = {}
    pair = None
    total_rows = 0
    for art in artifacts:
        rows = art.get("rows") or []
        if not rows:
            continue
        total_rows += len(rows)
        cols = list(rows[0].keys())
        pair = pair or movement_pair(cols)
        for c in cols:
            vals = sorted({str(r.get(c)) for r in rows if r.get(c) is not None})
            prev = columns.get(c, {"values": [], "distinct": 0})
            merged = sorted(set(prev["values"]) | set(vals))
            columns[c] = {"values": merged[:200], "distinct": len(merged)}
    return {"columns": columns, "movement_pair": list(pair) if pair else None,
            "rows": total_rows, "artifacts": len(artifacts)}


def _book_env(book: str) -> str:
    if book == "alderbridge":
        from demo_platform import config as cfg
        os.environ.update(cfg.mi_env(period_role="current"))
        return cfg.CLIENT_ID
    if book == "kestrelmoor":
        from question_interpretation.run_robustness_deterministic import (
            _KESTRELMOOR_ROOT)
        from tests.analytical import second_book as bk
        if not _KESTRELMOOR_ROOT.exists():
            bk.build(_KESTRELMOOR_ROOT)
        os.environ.update(bk.mi_env(_KESTRELMOOR_ROOT))
        return os.environ["MI_AGENT_CLIENT_ID"]
    raise ValueError("unknown book %r" % (book,))


def _asker(client_id: str, llm: bool):
    os.environ["MI_AGENT_LLM_PARSER"] = "on" if llm else "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "1" if llm else "0"
    import logging
    logging.disable(logging.WARNING)
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext
    ctx = ExecutionContext.for_internal(client_id)
    return lambda q: (execute_governed_mi_query(
        MiQueryRequest(question=q), ctx).result or {})


def run(book: str, llm: bool) -> Dict[str, Any]:
    problems = preflight(book)
    if problems:
        return {"book": book, "blocked": problems}
    client_id = _book_env(book)
    ask = _asker(client_id, llm)
    all_frags = sorted({f for _s, _n, _q, w in
                        [(s, n, q, w) for s, n, qs, w in SHAPES for q in qs]
                        for f in w})
    domains = dimension_domains(all_frags)

    rows: List[Dict[str, Any]] = []
    for sid, name, phrasings, want in SHAPES:
        for q in phrasings:
            res = ask(q)
            ok = bool(res.get("ok"))
            census = full_census(res.get("artifacts") or [])
            rating, why, evidence = rate_content(ok, want, census, domains)
            rows.append({
                "shape": sid, "shape_name": name, "question": q, "want": want,
                "ok": ok, "route": (res.get("metadata") or {}).get("route"),
                "rating_content": rating, "why_content": why,
                "evidence": evidence,
                "status": (DELIVERS if rating == PROVEN
                           else REFUSED if not ok else INCOMPLETE),
                "columns": sorted(census["columns"]),
                "movement_pair": census["movement_pair"],
                "answer": str(res.get("answer") or res.get("error") or "")[:220],
            })
    return {"book": book, "llm": llm, "rows": rows,
            "domains": {k: sorted(v)[:20] for k, v in domains.items()}}


# --------------------------------------------------------------------------- #
# What a phrasing is actually ASKING FOR
# --------------------------------------------------------------------------- #
#: A shape groups phrasings by SHAPE, not by request. "Which region has grown
#: fastest?" and "Which LTV band moved most between periods?" are both T7, but
#: they are not the same question in different words — one asks about regions
#: and the other about LTV bands. Counting the second as reachable because the
#: first works would overstate the routing gap and understate what P1 must
#: build. The target vocabulary below is deliberately explicit and inspectable.
_TARGET_VOCAB: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("source_split", ("direct and acquired", "direct", "acquired")),
    ("seasoning", ("front book", "back book")),
    ("region", ("region", "regions")),
    ("ltv_band", ("ltv band", "ltv bucket")),
    ("ticket", ("ticket size", "ticket")),
    ("threshold", ("over £", "above £", "above 50%", "larger than", "bigger than",
                   "over 150", "above 150")),
)


def request_target(question: str) -> str:
    """WHAT this phrasing asks to be broken down or filtered by.

    Two phrasings of one shape are the same REQUEST only when they name the same
    target. This is what separates "a different wording reaches it" from "a
    different parameter of the same shape happens to work".
    """
    q = question.lower()
    # A BREAKDOWN marker names what the answer must be split by, and outranks a
    # dimension that appears only as a filter. "balance over time by region for
    # the front book" is a REGION request filtered to the front book, not a
    # seasoning request — mislabelling it would group it with the wrong siblings.
    for name, terms in _TARGET_VOCAB:
        for t in terms:
            for marker in ("by %s", "split by %s", "broken down by %s",
                           "by %s and", "and %s"):
                if (marker % t) in q:
                    return name
    found = [name for name, terms in _TARGET_VOCAB if any(t in q for t in terms)]
    # A dimension named alongside a threshold is a dimension request first.
    for name in ("source_split", "seasoning", "region", "ltv_band", "ticket"):
        if name in found:
            return name
    return found[0] if found else "none"


# --------------------------------------------------------------------------- #
# The routing question
# --------------------------------------------------------------------------- #
def sibling_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
    """For every phrasing that does not deliver — does a SIBLING deliver?

    Phrasings of one shape are the same request in different words, so a
    phrasing that fails beside a sibling that succeeds is a ROUTING gap. A shape
    where none delivers is a CAPABILITY gap.
    """
    rows = result["rows"]
    by_shape: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_shape.setdefault(r["shape"], []).append(r)

    # STRICT grouping: same shape AND same target = the same request in
    # different words. This is the number that answers "is it reachable under
    # another phrasing".
    by_request: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        by_request.setdefault((r["shape"], request_target(r["question"])), []).append(r)

    strict = []
    for (sid, target), mine in sorted(by_request.items()):
        delivering = [r for r in mine if r["status"] == DELIVERS]
        for r in mine:
            if r["status"] == DELIVERS:
                verdict = None
            elif delivering:
                verdict = ROUTING_GAP
            else:
                verdict = CAPABILITY_GAP
            strict.append({"shape": sid, "target": target,
                           "question": r["question"], "status": r["status"],
                           "gap": verdict,
                           "reached_by": [d["question"] for d in delivering]})

    out = []
    for sid, mine in by_shape.items():
        delivering = [r for r in mine if r["status"] == DELIVERS]
        for r in mine:
            if r["status"] == DELIVERS:
                verdict = None
            elif delivering:
                verdict = ROUTING_GAP
            else:
                verdict = CAPABILITY_GAP
            out.append({**r, "shape_delivers": [d["question"] for d in delivering],
                        "gap": verdict})
    non_delivering = [r for r in out if r["gap"]]
    routing = [r for r in non_delivering if r["gap"] == ROUTING_GAP]
    capability = [r for r in non_delivering if r["gap"] == CAPABILITY_GAP]
    strict_routing = [r for r in strict if r["gap"] == ROUTING_GAP]
    strict_capability = [r for r in strict if r["gap"] == CAPABILITY_GAP]
    return {"rows": out, "delivering": len(rows) - len(non_delivering),
            "non_delivering": len(non_delivering),
            "routing_gaps": len(routing), "capability_gaps": len(capability),
            "strict": strict,
            "strict_routing_gaps": len(strict_routing),
            "strict_capability_gaps": len(strict_capability),
            "strict_routing_rows": strict_routing,
            "shapes_with_a_working_phrasing":
                sorted({r["shape"] for r in rows if r["status"] == DELIVERS})}


def report(result: Dict[str, Any]) -> int:
    if result.get("blocked"):
        print("REFUSING TO MEASURE — the environment would rate everything ABSENT:")
        for p in result["blocked"]:
            print("  * %s" % p)
        return 4
    rows = result["rows"]
    sib = sibling_analysis(result)
    print("=" * 96)
    print("CAPABILITY RECORD RE-RATED FROM CONTENTS — %s (%s arm)"
          % (result["book"], "LLM" if result["llm"] else "deterministic"))
    print("=" * 96)
    print()
    print("SHAPE RATINGS")
    print("-" * 96)
    for sid, name, qs, _w in SHAPES:
        mine = [r for r in rows if r["shape"] == sid]
        print("  %-3s %-44s %-8s  (%s)"
              % (sid, name, shape_rating([r["rating_content"] for r in mine]),
                 ", ".join(r["rating_content"][:1] for r in mine)))
    print()
    print("PER PHRASING")
    print("-" * 96)
    for r in rows:
        print("  %-4s %-8s %-62s" % (r["shape"], r["status"], r["question"][:62]))
        print("        %s" % r["why_content"][:150])
    print()
    print("THE ROUTING QUESTION")
    print("-" * 96)
    print("  phrasings that DELIVER                       : %d of %d"
          % (sib["delivering"], len(rows)))
    print("  phrasings that do NOT deliver                : %d" % sib["non_delivering"])
    print("    of which ROUTING gaps (a sibling delivers) : %d" % sib["routing_gaps"])
    print("    of which CAPABILITY gaps (none delivers)   : %d" % sib["capability_gaps"])
    print("  shapes with at least one working phrasing    : %s"
          % ", ".join(sib["shapes_with_a_working_phrasing"]))
    print()
    print("  STRICT — same shape AND same target (the same request in other words)")
    print("    ROUTING gaps (another wording reaches THIS request) : %d"
          % sib["strict_routing_gaps"])
    print("    CAPABILITY gaps (no wording reaches this request)   : %d"
          % sib["strict_capability_gaps"])
    for r in sib["strict_routing_rows"]:
        print("      %-4s [%-12s] %s" % (r["shape"], r["target"], r["question"][:58]))
        print("             reached by: %s" % r["reached_by"][0][:66])
    print()
    for r in sib["rows"]:
        if r["gap"] == ROUTING_GAP:
            print("  ROUTING  %-4s %s" % (r["shape"], r["question"][:64]))
            print("           reached instead by: %s" % r["shape_delivers"][0][:70])
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="alderbridge",
                    choices=("alderbridge", "kestrelmoor"))
    ap.add_argument("--llm", action="store_true", help="run the LLM arm instead")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args(argv)
    result = run(args.book, args.llm)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str),
                                   encoding="utf-8")
    return report(result)


if __name__ == "__main__":
    raise SystemExit(main())
