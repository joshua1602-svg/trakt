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

#: NOT `\b...\b`. A trailing word boundary does not exist between the "0" of
#: a date and the "T" of a timestamp, so "2025-11-30T00:00:00" matched nothing
#: and every timestamped date in the platform's own metadata was invisible to
#: this probe. Digit lookarounds instead, which is what was meant.
_ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
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
         portfolio: Optional[str], timeout: float,
         keep_payload: bool = False) -> Dict[str, Any]:
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
        **({"_payload": payload} if keep_payload else {}),
    }


def _get(base: str, token: str, path: str, timeout: float) -> Any:
    """A platform metadata surface, or {} if it cannot be read."""
    req = urllib.request.Request(base.rstrip("/") + path, method="GET")
    req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except Exception:  # noqa: BLE001 - an unreadable surface is not a finding
        return {}


def _strip_local(result: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in result.items() if not k.startswith("_")}


#: Anything the platform might call the date the DATA was last updated, as
#: opposed to the period the figures are labelled as. Both concepts are
#: legitimate and they are not the same: a 30 June report can be produced from
#: data cut on 20 May, and the reader has to be told which they are looking at.
_CUT_OFF_KEYS = ("data_cut_off_date", "dataCutOffDate", "cut_off_date",
                 "cutOffDate", "dataCutOff")


def _find_cut_off(obj: Any, depth: int = 0) -> List[Tuple[str, str]]:
    """Every (key, ISO date) the platform exposes as a data cut-off.

    Searched rather than read from one place because the question is whether
    the value is surfaced ANYWHERE a reader or a channel could see it — not
    whether it sits in the field this probe expected.
    """
    found: List[Tuple[str, str]] = []
    if depth > 6:
        return found
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _CUT_OFF_KEYS and value:
                match = _ISO_DATE.search(str(value))
                if match:
                    found.append((str(key), match.group(0)))
            found.extend(_find_cut_off(value, depth + 1))
    elif isinstance(obj, list):
        for item in obj[:40]:
            found.extend(_find_cut_off(item, depth + 1))
    return found


#: The columns `engine.region_taxonomy` stamps when harmonisation RUNS. Their
#: absence from what MI binds to is the difference between "the books disagree
#: about case" and "nothing harmonised them".
HARMONISED_REGION_FIELDS = ("canonical_region_detail",
                            "canonical_region_reporting")


def _verdict(confirmed: bool, exonerated: bool, why: str) -> str:
    """CONFIRMED / EXONERATED / NOT ESTABLISHED, and never a fourth thing.

    A test that can only confirm is not evidence. Every comparison below is
    built so that the hypothesis can FAIL — and "not established" is a real
    outcome, not a polite way of saying yes.
    """
    if confirmed and not exonerated:
        return "CONFIRMED — " + why
    if exonerated and not confirmed:
        return "EXONERATED — " + why
    return "NOT ESTABLISHED — " + why


def hypothesis_region_vocabulary(base, token, portfolio, timeout,
                                 fetch=None) -> Dict[str, Any]:
    """H1. Are un-harmonised region names actually blocking the agent?

    FOUR STEPS, and the last two are the ones that can exonerate:

      1. does the platform expose a HARMONISED region field at all?
      2. which field does a region filter actually bind to?
      3. does the same region question behave DIFFERENTLY per book? If a
         question fails identically on direct, on acquired and on the two
         combined, then whatever is wrong is not a disagreement between them;
      4. does a region breakdown split WITHIN one artefact, and does it split
         on a single book or only on the two combined? A split that appears
         only when the books are combined is harmonisation; a split inside one
         book is that book's own data.
    """
    out: Dict[str, Any] = {}
    getter = fetch or _get

    # 1. What the platform says it carries.
    catalogue = getter(base, token, "/mi/catalogue", timeout)
    blob = json.dumps(catalogue)
    harmonised_present = [f for f in HARMONISED_REGION_FIELDS if f in blob]
    out["harmonised_fields_exposed"] = harmonised_present

    # 2 and 3. The same question, per book.
    per_book: Dict[str, Any] = {}
    for lens in LENSES:
        res = _ask(base, token, "What is the balance in Scotland?", lens=lens,
                   portfolio=portfolio, timeout=timeout)
        per_book[lens or "total"] = _strip_local(res)
    out["one_region_per_book"] = per_book
    bound = sorted({f for v in per_book.values()
                    for f in (v.get("applied_filters") or [])})
    out["region_bound_to"] = bound
    out["bound_to_harmonised_field"] = any(
        f in HARMONISED_REGION_FIELDS for f in bound)
    outcomes = {name: v.get("ok") for name, v in per_book.items()}
    out["outcome_per_book"] = outcomes
    differs_by_book = len(set(outcomes.values())) > 1

    # 4. The breakdown, per book and combined.
    splits: Dict[str, Any] = {}
    for lens in LENSES:
        res = _ask(base, token, "Show balance by region.", lens=lens,
                   portfolio=portfolio, timeout=timeout)
        splits[lens or "total"] = {**_strip_local(res),
                                   "grouping": worst_collision(res["_labels"])}
    out["breakdown_per_book"] = splits
    measured = {n: v for n, v in splits.items()
                if v["grouping"].get("measured")}
    dirty = {n for n, v in measured.items() if not v["grouping"]["clean"]}
    single_books_clean = not (dirty & {"direct", "acquired"})
    combined_dirty = "total" in dirty

    # FILTERING AND GROUPING ARE TWO CLAIMS, and the 2026-09-03 run showed
    # them coming apart: every book refused "the balance in Scotland" with the
    # SAME message — including `direct`, whose ten region values collide not at
    # all. A book with a clean vocabulary that still cannot be filtered proves
    # the filter failure is not a case failure, and a verdict that reports one
    # number for both hides exactly that.
    clean_books_that_cannot_filter = sorted(
        n for n, v in measured.items()
        if v["grouping"]["clean"] and per_book.get(n, {}).get("ok") is False)
    out["clean_books_that_cannot_filter"] = clean_books_that_cannot_filter
    if clean_books_that_cannot_filter:
        out["filter_finding"] = (
            "SEPARATE DEFECT — " + ", ".join(clean_books_that_cannot_filter)
            + " has a clean region vocabulary and still cannot be filtered to a "
              "region, so the filter failure is NOT caused by case. Grouping and "
              "filtering are binding to different things; compare the field the "
              "breakdown grouped on against the one the filter looked for.")
    else:
        out["filter_finding"] = "none — no clean book failed to filter"

    # The verdict, from the comparisons rather than from the expectation.
    if not measured and not any(outcomes.values()):
        out["verdict"] = _verdict(
            False, False,
            "no region question answered anywhere and no breakdown came back, "
            "so nothing could be compared between the books. The region path "
            "is failing for a reason this test has not reached.")
    elif combined_dirty and single_books_clean:
        out["verdict"] = _verdict(
            True, False,
            "a region splits into two rows only when the books are COMBINED "
            "and neither book splits alone. That is unharmonised vocabulary, "
            "and it is upstream: nothing in the query path should be casefolding "
            "a group key.")
    elif dirty:
        out["verdict"] = _verdict(
            True, False,
            "a region splits within a single book (" + ", ".join(sorted(dirty))
            + "), so that book's own region values are inconsistent before any "
              "combination happens.")
    elif differs_by_book:
        out["verdict"] = _verdict(
            True, False,
            "the same region question succeeds on one book and fails on "
            "another (" + json.dumps(outcomes) + "), which is a difference "
            "BETWEEN the books rather than a defect in the question.")
    elif not harmonised_present and not out["bound_to_harmonised_field"]:
        out["verdict"] = _verdict(
            False, False,
            "no region is reported twice and every book behaves the same, so "
            "case is NOT blocking anything measurable here — but MI binds "
            "region to " + (", ".join(bound) or "no field this test could see")
            + " and no harmonised column is exposed, so nothing guarantees it "
              "stays that way when a book changes.")
    else:
        out["verdict"] = _verdict(
            False, True,
            "no region splits, every book behaves the same, and MI binds to a "
            "harmonised column. Region vocabulary is not the problem.")
    return out


def hypothesis_data_cut_off(base, token, portfolio, timeout,
                            fetch=None) -> Dict[str, Any]:
    """H2. Are two data cut-off dates actually blocking the agent?

    THE HYPOTHESIS CAN FAIL HERE, and on the code as read it probably should.
    `_platform_reporting_date` picks the first column present from
    ("reporting_date", "data_cut_off_date", "cut_off_date"), so a tape carrying
    a `reporting_date` never has `data_cut_off_date` read at all — and a value
    nothing reads cannot block anything. That would make it a DISCLOSURE
    failure (every "as at" overstating freshness) rather than a blocking one,
    and the two need different remedies.

    So this asks both questions separately:
      * is a cut-off surfaced anywhere a reader could see it?   (disclosure)
      * does a period question behave differently per book?     (blocking)
    """
    out: Dict[str, Any] = {}
    getter = fetch or _get
    surfaced: List[Tuple[str, str]] = []

    for lens in LENSES:
        res = _ask(base, token, "Summarise the funded portfolio.", lens=lens,
                   portfolio=portfolio, timeout=timeout, keep_payload=True)
        payload = res.pop("_payload", {})
        hits = _find_cut_off(payload)
        surfaced.extend(hits)
        out[lens or "total"] = {**_strip_local(res),
                                "cut_off_keys_found": sorted({k for k, _ in hits}),
                                "cut_off_dates_found": sorted({d for _, d in hits})}
    for path in ("/health", "/mi/snapshots"):
        body = getter(base, token, path, timeout)
        hits = _find_cut_off(body)
        surfaced.extend(hits)
        out["GET " + path] = {
            "cut_off_keys_found": sorted({k for k, _ in hits}),
            "cut_off_dates_found": sorted({d for _, d in hits}),
            "reporting_dates": sorted(set(_ISO_DATE.findall(json.dumps(body))))[:8]}

    # THE BLOCKING TEST. A period comparison over each book alone, and over the
    # two combined. If the combined one fails where both singles succeed, the
    # misalignment is blocking. If all three behave the same, it is not.
    period: Dict[str, Any] = {}
    for lens in LENSES:
        res = _ask(base, token, "How has the funded balance moved since last "
                                "month?", lens=lens, portfolio=portfolio,
                   timeout=timeout)
        period[lens or "total"] = _strip_local(res)
    out["period_question_per_book"] = period
    ok = {name: v.get("ok") for name, v in period.items()}
    out["period_outcome_per_book"] = ok
    combined_only_fails = (ok.get("total") is False
                           and ok.get("direct") and ok.get("acquired"))
    # THE REVERSE, which the first draft never tested and then printed silence
    # over: a book that fails ALONE while the total containing it succeeds.
    # Benign when the book has one period and the refusal says so; serious if
    # the total is answering by leaving that book out. The reason distinguishes
    # them, so it is carried into the finding rather than reduced to a flag.
    single_fails_under_passing_total = sorted(
        n for n in ("direct", "acquired")
        if ok.get(n) is False and ok.get("total") is True)
    out["single_book_fails_under_passing_total"] = {
        n: {"error_code": period[n].get("error_code"),
            "reason": period[n].get("reason")}
        for n in single_fails_under_passing_total}

    dates = sorted({d for _, d in surfaced})
    labelled = sorted({d for v in out.values() if isinstance(v, dict)
                       for d in (v.get("reporting_dates") or ())})
    out["cut_off_dates_surfaced"] = dates
    out["reporting_dates_declared"] = labelled

    if single_fails_under_passing_total and not combined_only_fails:
        out["book_asymmetry"] = (
            "CHECK — " + ", ".join(single_fails_under_passing_total)
            + " cannot answer a period question alone while the total that "
              "contains it can. Read the reason: a book that exists at only one "
              "reporting date is a correct refusal, and the total then compares "
              "two dates across which that book did not exist throughout.")
    else:
        out["book_asymmetry"] = "none"
    if combined_only_fails:
        out["verdict"] = _verdict(
            True, False,
            "a period question answers on each book alone and fails on the two "
            "combined, which is the misalignment blocking the answer.")
    elif not dates:
        out["verdict"] = _verdict(
            False, False,
            "NOT BLOCKING, but not safe either: no data cut-off is surfaced "
            "anywhere — not on an answer, not on /health, not on the snapshot "
            "index — and a value nothing reads cannot block anything. It is a "
            "DISCLOSURE failure: every 'as at' a reader sees is the reporting "
            "label, so books cut at different times are presented under one "
            "date with nothing to tell them apart.")
    elif len(dates) > 1 or (labelled and any(d not in labelled for d in dates)):
        out["verdict"] = _verdict(
            False, False,
            "a cut-off IS surfaced (" + ", ".join(dates) + ") and differs from "
            "the reported date (" + (", ".join(labelled) or "none")
            + "), but every book answers the same, so it is degrading "
              "disclosure rather than blocking answers.")
    else:
        out["verdict"] = _verdict(
            False, True,
            "one cut-off, matching the reported date, and every book answers "
            "the same. Cut-off alignment is not the problem.")
    return out


def _wrap(text: Any, width: int = 76) -> str:
    """A verdict is a sentence, and a sentence has to be readable in a
    terminal to be acted on."""
    words, line, out = str(text or "").split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = "    " + word
        else:
            line = (line + " " + word) if line else word
    if line:
        out.append(line)
    return "\n  ".join(out)


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
            ("H1_region_vocabulary",
             "H1. are un-harmonised region names blocking the agent?",
             hypothesis_region_vocabulary),
            ("H2_data_cut_off",
             "H2. are two data cut-off dates blocking the agent?",
             hypothesis_data_cut_off)):
        print("\n=== %s ===" % label)
        section = fn(args.base, token, args.portfolio, args.timeout)
        report[key] = section
        print("  %s" % _wrap(section.get("verdict")))
        for extra in ("filter_finding", "book_asymmetry",
                      "clean_books_that_cannot_filter",
                      "single_book_fails_under_passing_total",
                      "harmonised_fields_exposed", "region_bound_to",
                      "outcome_per_book", "period_outcome_per_book",
                      "cut_off_dates_surfaced", "reporting_dates_declared"):
            if extra in section:
                print("    %-26s %s" % (extra, json.dumps(section[extra])))
        for name, value in section.items():
            if name in ("verdict", "cut_off_dates_surfaced", "filter_finding",
                        "book_asymmetry", "clean_books_that_cannot_filter",
                        "single_book_fails_under_passing_total",
                        "reporting_dates_declared", "region_bound_to",
                        "outcome_per_book", "period_outcome_per_book",
                        "harmonised_fields_exposed",
                        "bound_to_harmonised_field") \
                    or not isinstance(value, dict):
                continue
            if "grouping" in value:
                g = value["grouping"]
                print("    %-10s groups=%d distinct=%d colliding=%d widest=%d"
                      % (name, g["labels_returned"],
                         g["distinct_case_insensitive"], g["colliding_groups"],
                         g["widest_collision"]))
            elif "cut_off_keys_found" in value:
                print("    %-16s cutOffKeys=%s cutOffDates=%s"
                      % (name, value.get("cut_off_keys_found") or "-",
                         ", ".join(value.get("cut_off_dates_found") or ["none"])))
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
