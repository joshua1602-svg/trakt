#!/usr/bin/env python3
"""alignment_probe — do the two books line up, in case and in cut-off date?

WHY THIS EXISTS. The lender reports that acquired loans carry regions in
CAPITALS while direct loans carry them in Sentence Case, and that the two books
have different cut-off dates. Both are facts about the tape that the MI layer
either absorbs or is broken by, and neither can be settled by reading code:
`_apply_filters` casefolds both sides, so a FILTER is case-insensitive by
construction — but nothing casefolds a GROUP KEY, so a breakdown by region can
split one region into two rows, one per book. That is the failure worth finding,
because it does not refuse. It answers, and looks right.

THREE QUESTIONS, and each is answered by evidence rather than by inference:

  A. CASE ON THE WAY IN.  Ask the same question with the region written three
     ways. If the outcomes and the applied filters agree, matching is
     case-insensitive and this is closed.

  B. CASE ON THE WAY OUT.  Ask for a breakdown by region and compare the
     category labels TO EACH OTHER, case-folded. Two labels that differ only in
     case are one region reported twice, and every share, rank and "largest
     region" computed over that grouping is wrong.

  C. CUT-OFF ALIGNMENT.  Ask the same question under each source lens and read
     the reporting date each answer declares. A combined book whose halves
     declare different dates has no single as-of date, so a period comparison
     over it compares two different points in time.

DIAGNOSTICS ONLY, THE STANDING RULE, and tighter here than elsewhere because
this probe must LOOK AT category labels to do its job. It looks at them locally
and emits none of them: the output carries how many labels there were, how many
collided when case-folded, and nothing else. No balance, no row count, no
region name, no answer text. ISO dates are emitted deliberately — a reporting
date is provenance, it is already on every answer this platform gives, and the
whole of question C is about which ones disagree.

USAGE
    export MI_BEARER='<paste your dashboard token>'
    python3 alignment_probe.py --out alignment.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

#: Regions to probe. Governed values from `config/mi/region_taxonomy.yaml`,
#: named here as TEST INPUTS -- they are the platform's own vocabulary, not
#: something discovered from a client's tape.
REGIONS = ("Scotland", "London", "South West")

#: The lenses whose alignment question C exists to ask.
LENSES = (None, "direct", "acquired")

_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_FIGURE = re.compile(
    r"[£$€]\s?[\d,.]+\s*(?:[kKmMbB]{1,2}|MM)?|\b\d[\d,]{3,}(?:\.\d+)?\b"
    r"|\b\d+\.\d+\b")


def _redact(text: Any) -> str:
    """ISO dates survive; money and long numbers do not. Same reader as the
    other probes, deliberately: one rule, several probes, one behaviour."""
    if not text:
        return ""
    keep: Dict[str, str] = {}

    def _stash(m):
        token = "\x00%d\x00" % len(keep)
        keep[token] = m.group(0)
        return token

    out = _ISO_DATE.sub(_stash, str(text))
    out = _FIGURE.sub("[figure]", out)
    for token, original in keep.items():
        out = out.replace(token, original)
    return out.strip()[:240]


# --------------------------------------------------------------------------- #
# The one function that sees client values, and the only one. It is pure, it is
# tested, and it returns counts.
# --------------------------------------------------------------------------- #
def case_collisions(labels: List[Any]) -> Dict[str, Any]:
    """How many of these labels are the same label in different case?

    Returns counts ONLY. The labels are the client's own category values and
    none of them leaves this function -- which is the whole reason the
    comparison happens here rather than in a chat window.
    """
    seen: Dict[str, int] = {}
    for label in labels:
        key = str(label).strip().casefold()
        if not key or key in ("none", "nan"):
            continue
        seen[key] = seen.get(key, 0) + 1
    collided = {k: n for k, n in seen.items() if n > 1}
    return {
        "labels_returned": len([l for l in labels if str(l).strip()]),
        "distinct_case_insensitive": len(seen),
        "colliding_groups": len(collided),
        # The worst case: one region split this many ways.
        "widest_collision": max(collided.values()) if collided else 0,
        "clean": not collided,
    }


def _category_labels(payload: Dict[str, Any]) -> List[List[Any]]:
    """The category keys of each artefact, KEPT APART.

    ONE LIST PER ARTEFACT, and the reason is a false positive this probe
    produced on its first run. A grouped answer returns a chart AND a table
    over the same rows, so pooling their labels made every region appear twice
    and reported the whole book as "one region reported twice" — 20 labels, 10
    distinct, widest exactly 2, in a book with no case problem at all.

    A duplicate only means anything WITHIN one artefact: two rows of the same
    table that differ only in case are one region the grouping split.
    """
    out: List[List[Any]] = []
    for art in (payload.get("artifacts") or []):
        if not isinstance(art, dict):
            continue
        rows = art.get("rows")
        if not isinstance(rows, list):
            continue
        key = art.get("xKey")
        labels: List[Any] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if key and key in row:
                labels.append(row[key])
                continue
            for k, v in row.items():
                if isinstance(v, str) and k not in ("id", "type"):
                    labels.append(v)
                    break
        if labels:
            out.append(labels)
    return out


def worst_collision(per_artifact: List[List[Any]]) -> Dict[str, Any]:
    """The worst split found in ANY single artefact, plus how many were read."""
    reports = [case_collisions(labels) for labels in (per_artifact or [])]
    if not reports:
        return {"artifacts_read": 0, "labels_returned": 0,
                "distinct_case_insensitive": 0, "colliding_groups": 0,
                "widest_collision": 0, "clean": True, "measured": False}
    worst = max(reports, key=lambda r: (r["colliding_groups"],
                                        r["widest_collision"]))
    return {**worst, "artifacts_read": len(reports), "measured": True}


def _reporting_dates(payload: Dict[str, Any]) -> List[str]:
    """Every reporting date the answer DECLARES, from its own provenance.

    Not scraped from the prose: `governance.snapshot`, the metadata and the
    source notes are the route saying which snapshot it read.
    """
    found: List[str] = []
    gov = payload.get("governance") or {}
    snap = gov.get("snapshot") or {}
    for value in (snap.get("reporting_date"), (payload.get("metadata") or {}).get("asOfDate")):
        if value:
            found.append(str(value)[:10])
    for note in (payload.get("sourceNotes") or []):
        if isinstance(note, dict):
            found.extend(_ISO_DATE.findall(str(note.get("detail") or "")))
    for art in (payload.get("artifacts") or []):
        if isinstance(art, dict) and art.get("source"):
            found.extend(_ISO_DATE.findall(str(art.get("source"))))
    seen, ordered = set(), []
    for d in found:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def _ask(base: str, token: str, question: str, *, lens: Optional[str],
         portfolio: Optional[str], timeout: float) -> Dict[str, Any]:
    body: Dict[str, Any] = {"question": question}
    if lens:
        body["sourcePortfolioLens"] = lens
    if portfolio:
        body["portfolioId"] = portfolio
    req = urllib.request.Request(
        base.rstrip("/") + "/mi/query",
        data=json.dumps(body).encode("utf-8"), method="POST")
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    payload: Dict[str, Any] = {}
    status: Any = 0
    transport = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            payload = json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            payload = json.loads(exc.read().decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            payload = {}
    except Exception as exc:  # noqa: BLE001 - the class only; a URL names a book
        transport = type(exc).__name__
    gov = payload.get("governance") if isinstance(
        payload.get("governance"), dict) else {}
    err = gov.get("error") if isinstance(gov.get("error"), dict) else {}
    ok = gov.get("status") == "success" if gov.get("status") else bool(
        payload.get("ok"))
    fi = payload.get("filterInvariant") if isinstance(
        payload.get("filterInvariant"), dict) else {}
    return {
        "ok": bool(ok) and not transport,
        "http": status,
        "transport_error": transport,
        "error_code": err.get("code") or payload.get("errorCode"),
        "reason": _redact(payload.get("error") or err.get("message")),
        "ms": int((time.time() - t0) * 1000),
        # The executor's own ledger: which filters it PARSED and which it
        # APPLIED. Field keys, never values.
        "parsed_filters": sorted(str(f) for f in (fi.get("parsed_filters") or [])),
        "applied_filters": sorted(str(f) for f in (fi.get("applied_filters") or [])),
        "reporting_dates": _reporting_dates(payload),
        "_labels": _category_labels(payload),      # local only; never emitted
    }


def _strip_local(result: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def probe_case_on_the_way_in(base, token, portfolio, timeout) -> Dict[str, Any]:
    """A. Does the case a reader types change what is matched?"""
    out: Dict[str, Any] = {}
    for region in REGIONS:
        forms = {"as_written": region, "upper": region.upper(),
                 "lower": region.lower()}
        runs = {name: _strip_local(_ask(
            base, token, f"What is the balance in {value}?",
            lens=None, portfolio=portfolio, timeout=timeout))
            for name, value in forms.items()}
        outcomes = {r["ok"] for r in runs.values()}
        applied = {tuple(r["applied_filters"]) for r in runs.values()}
        out[region] = {
            "runs": runs,
            "all_forms_agree": len(outcomes) == 1 and len(applied) == 1,
            "any_answered": any(r["ok"] for r in runs.values()),
        }
    rows = [v for k, v in out.items() if k != "verdict"]
    agree = all(v["all_forms_agree"] for v in rows)
    answered = any(v["any_answered"] for v in rows)
    if not answered:
        # NOT A MEASUREMENT. Three forms that all REFUSE agree perfectly and
        # prove nothing about case: the question failed for some other reason
        # and took the experiment with it. The first run of this probe reported
        # "case-insensitive" off exactly that, which is a verdict dressed up
        # from an absence of evidence.
        out["verdict"] = ("not established — no case form answered, so the "
                          "forms agreeing says nothing about case. The region "
                          "question is failing for another reason; see the "
                          "error codes below.")
    elif agree:
        out["verdict"] = "case-insensitive"
    else:
        out["verdict"] = ("CASE-SENSITIVE — the case a reader types changes "
                          "what is matched")
    return out


def probe_case_on_the_way_out(base, token, portfolio, timeout) -> Dict[str, Any]:
    """B. Does one region come back as two groups?"""
    out: Dict[str, Any] = {}
    for lens in LENSES:
        res = _ask(base, token, "Show balance by region.", lens=lens,
                   portfolio=portfolio, timeout=timeout)
        name = lens or "total"
        out[name] = {**_strip_local(res),
                     "grouping": worst_collision(res["_labels"])}
    measured = {n: v for n, v in out.items()
                if isinstance(v, dict) and v.get("grouping", {}).get("measured")}
    split = [n for n, v in measured.items() if not v["grouping"]["clean"]]
    if not measured:
        # NOT A MEASUREMENT. No lens returned a grouped artefact, so nothing
        # was compared. Reporting that as "clean" is how a probe launders a
        # failed run into a pass.
        out["verdict"] = ("not established — no lens returned a grouped "
                          "answer, so no grouping was examined")
    elif split:
        out["verdict"] = ("ONE REGION REPORTED TWICE within a single artefact "
                          "under: " + ", ".join(split)
                          + " — every share, rank and 'largest region' over "
                            "that grouping is computed on split rows")
    else:
        out["verdict"] = ("no region is reported twice within any single "
                          "artefact (%d lens(es) examined)" % len(measured))
    return out


def probe_cut_off_alignment(base, token, portfolio, timeout) -> Dict[str, Any]:
    """C. Do the two books declare the same as-of date?"""
    out: Dict[str, Any] = {}
    for lens in LENSES:
        res = _ask(base, token, "Summarise the funded portfolio.", lens=lens,
                   portfolio=portfolio, timeout=timeout)
        out[lens or "total"] = _strip_local(res)
    dates = {name: tuple(v.get("reporting_dates") or ())
             for name, v in out.items() if isinstance(v, dict)}
    direct = set(dates.get("direct", ()))
    acquired = set(dates.get("acquired", ()))
    out["dates_by_lens"] = {k: list(v) for k, v in dates.items()}
    if direct and acquired and direct != acquired:
        out["verdict"] = ("THE BOOKS DECLARE DIFFERENT AS-OF DATES — direct "
                          + ", ".join(sorted(direct)) + " vs acquired "
                          + ", ".join(sorted(acquired))
                          + ". A combined figure has no single point in time, "
                            "and a period comparison over it compares two.")
    elif not direct or not acquired:
        out["verdict"] = ("not established — a lens declared no reporting "
                          "date, so the two could not be compared")
    else:
        out["verdict"] = "both books declare the same as-of date"
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="https://app.traktinfra.io/api")
    ap.add_argument("--portfolio", default="ERE/2026-06-30")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default="alignment.json")
    args = ap.parse_args(argv)

    token = os.environ.get("MI_BEARER", "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token:
        print("MI_BEARER is not set:\n    export MI_BEARER='<token>'",
              file=sys.stderr)
        return 2

    print("target %s  portfolio %s" % (args.base, args.portfolio))
    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base": args.base, "portfolio": args.portfolio}

    for key, label, fn in (
            ("A_case_on_the_way_in", "A. case a reader types",
             probe_case_on_the_way_in),
            ("B_case_on_the_way_out", "B. one region, two groups?",
             probe_case_on_the_way_out),
            ("C_cut_off_alignment", "C. do the books share an as-of date?",
             probe_cut_off_alignment)):
        print("\n=== %s ===" % label)
        section = fn(args.base, token, args.portfolio, args.timeout)
        report[key] = section
        print("  %s" % section.get("verdict"))
        for name, value in section.items():
            if name in ("verdict", "dates_by_lens") or not isinstance(value, dict):
                continue
            if "grouping" in value:
                g = value["grouping"]
                print("    %-10s groups=%d distinct=%d colliding=%d widest=%d"
                      % (name, g["labels_returned"],
                         g["distinct_case_insensitive"], g["colliding_groups"],
                         g["widest_collision"]))
            elif "reporting_dates" in value and "runs" not in value:
                print("    %-10s ok=%s dates=%s"
                      % (name, value.get("ok"),
                         ", ".join(value.get("reporting_dates") or ["-"])))
            elif "runs" in value:
                print("    %-12s agree=%s answered=%s"
                      % (name, value["all_forms_agree"], value["any_answered"]))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
