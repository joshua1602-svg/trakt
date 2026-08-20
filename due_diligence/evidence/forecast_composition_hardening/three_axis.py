"""SEMANTIC / ARITHMETIC / PRESENTATION — three axes, scored separately.

Why this exists
---------------
``nl_score.py`` is the frozen control and is NOT touched by this sprint. It
grades an answer as CORRECT / HONEST_PARTIAL / SAFE_REFUSAL / … against the
behaviour of the shipped system. What it cannot do — what nothing before this
did — is say whether the row set the agent computed over was the row set the
question ASKED for. Reconciliation proves arithmetic over the executed
population; it is silent on whether that population was the right one.

This scorer answers three separable questions, and reports them separately so a
pass on one can never be read as a pass on another:

  SEMANTIC      did the agent read the question as the frozen expectation says
                it should be read — family, operation, and above all POPULATION?
  ARITHMETIC    is every numeric finding right, against a recomputation from
                the fixture CSVs that imports nothing from the agent?
  PRESENTATION  does the answer disclose what the expectation says a correct
                answer must disclose — rolling-cohort membership, the pipeline
                exclusion, a refusal that names what is missing — and does it
                print only figures the findings hold?

The ARITHMETIC axis is DELEGATED to ``nl_reconcile.py``, which already
recomputes every numeric finding from the fixture CSVs. It is not reimplemented
here: two matchers would be two chances to be wrong, and the reconciler is the
one that has been run against a baseline.

The PRESENTATION figure check is DIFFERENTIAL — it compares this run against a
comparator run and asks whether each CHANGED figure is held by a finding. An
absolute "every printed figure is held" check was written first and thrown away:
it flags bucket labels ("30-40%", "70-75", "0-12 months") and rounded renderings
("£2.3m" for 2,332,075.32) as unheld figures, i.e. it measures the extractor, not
the product. The differential form cancels every label artefact, because labels
are identical on both sides.

The frozen expectation file stands. Where agent and expectation disagree, the
disagreement is REPORTED. Nothing here is tuned until they agree.

Provenance of each axis
-----------------------
  RECORDED  population (metadata.populationApplied + finding population
            predicates), families/operations (metadata.analyticalIntent,
            stamped by chat_routing._stamp_analytical_intent on the answer it
            shaped), answer text, findings — all captured at run time by
            nl_harness.py.
  DERIVED   the canonical tokens below, which map both sides of the comparison
            onto one vocabulary. The mapping is in this file and auditable.

    python3 three_axis.py <run.json> [<run.json> …]
"""
from __future__ import annotations
import collections, json, re, sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXPECTATIONS = HERE / "frozen_expectations.yaml"

# --------------------------------------------------------------------------- #
# Figure extraction — shared with the A5 neutrality proof, same rules.
# --------------------------------------------------------------------------- #
TOKEN = re.compile(r"[-+−]?£?\d[\d,]*\.?\d*\s?(?:bn|m|k|pp|%)?")
DATE = re.compile(r"\b\d{4}-\d{2}(-\d{2})?\b")


def figures(text: str) -> list:
    text = DATE.sub(" ", text or "")
    return [t.strip().replace("−", "-").replace(" ", "")
            for t in TOKEN.findall(text) if any(c.isdigit() for c in t)]


def held(findings: list) -> set:
    vals = set()

    def rec(v):
        try:
            f = abs(float(v))
        except (TypeError, ValueError):
            return
        for x in (round(f, 2), round(f, 1), round(f / 1e3, 1), round(f / 1e6, 1),
                  round(f / 1e9, 2), round(f * 100, 1), round(f * 100, 2)):
            vals.add(x)
    def walk(o, depth=0):
        if depth > 4:
            return
        if isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, depth + 1)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            rec(o)
    for fd in findings:
        for k in ("value", "priorValue", "change", "relativeChange", "comparandValue",
                  "forecastValue", "share", "priorShare", "limit", "headroom", "rank"):
            rec(fd.get(k))
        for side in ("population", "comparand"):
            p = fd.get(side) or {}
            for k in ("rows", "rowsBefore", "rowsPrior", "exposure"):
                rec(p.get(k))
        # The disclosure figures a narrator may print live on the finding's
        # evidence (excludedCaseCount, excludedFromWeightingAmount, caseCount …).
        walk(fd.get("evidence"))
    return vals


def close_enough(tok: str, value: float, candidate: float) -> bool:
    """Is a printed token a rendering of a held value?

    The tolerance comes from the token's OWN precision, not from a fixed
    percentage. "£752k" is printed to whole thousands, so it stands for anything
    in [751.5k, 752.5k); "£2.3m" is printed to one decimal of a million, so it
    stands for [2.25m, 2.35m). A flat 0.5% band rejects the first (751,834 is
    0.02% away but 166 units away) and a flat unit band rejects the second.
    Unsuffixed figures keep the tight absolute tolerance.
    """
    body = tok.replace("£", "").replace(",", "").replace("+", "").replace("-", "")
    for suf, unit in (("bn", 1e9), ("m", 1e6), ("k", 1e3)):
        if tok.endswith(suf):
            digits = body[: -len(suf)]
            dp = len(digits.split(".")[1]) if "." in digits else 0
            half_ulp = 0.5 * (10 ** -dp) * unit
            return abs(value - candidate) <= half_ulp
    return abs(value - candidate) < max(0.02, abs(candidate) * 0.005)


def numeric(tok: str):
    t = tok.replace("£", "").replace(",", "").replace("+", "")
    mult = 1.0
    for suf, m in (("bn", 1e9), ("m", 1e6), ("k", 1e3), ("pp", 1), ("%", 1)):
        if t.endswith(suf):
            t, mult = t[: -len(suf)], m
            break
    try:
        return abs(float(t)) * mult
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# The canonical population vocabulary. Both sides of every comparison are mapped
# onto these tokens; nothing else is compared.
# --------------------------------------------------------------------------- #
NEW, RECENT, FRONT, BACK = "NEW_L1M", "RECENT_L3M", "FRONT_BOOK", "BACK_BOOK"
WHOLE, DIRECT, ACQUIRED, GEO = "WHOLE_BOOK", "DIRECT", "ACQUIRED", "GEOGRAPHY"
OPEN_PIPE, PIPE_OFFER, LIMITS, FLOW = "OPEN_PIPELINE", "PIPELINE_OFFER", "LIMIT_SCHEDULE", "FLOW"

#: Expectations whose "population" is not a funded-book row set. For these the
#: population axis is not checkable from a PopulationRef and is scored on the
#: governed owner instead — recorded as NOT_APPLICABLE with the reason, never
#: silently as a pass.
NON_ROW_POPULATIONS = {PIPE_OFFER, LIMITS, FLOW}


def expected_token(text) -> str:
    t = (text or "").lower()
    if "months_on_book <= 1" in t:
        return NEW
    if "months_on_book <= 3" in t:
        return RECENT
    if "front book" in t:
        return FRONT
    if "back book" in t:
        return BACK
    if "kfi, application, offer" in t or "open pipeline" in t:
        return OPEN_PIPE
    if "pipeline_stage == offer" in t:
        return PIPE_OFFER
    if "limit schedule" in t:
        return LIMITS
    if "flow" in t or "per unit time" in t:
        return FLOW
    if "source_portfolio_type == direct" in t:
        return DIRECT
    if "source_portfolio_type == acquired" in t:
        return ACQUIRED
    if "collateral_geography" in t:
        return GEO
    if "whole funded book" in t:
        return WHOLE
    return "UNMAPPED:" + t[:40]


def recorded_tokens(run: dict) -> set:
    """Every population the RUN says it computed over, as canonical tokens."""
    found = set()
    seen = []
    pop_applied = (run.get("populationApplied") or {}).get("applied") or []
    seen.extend(pop_applied)
    for fd in run.get("findings") or []:
        for side in ("population", "comparand"):
            p = fd.get(side) or {}
            if p.get("predicate"):
                seen.append(p["predicate"])
            if p.get("label"):
                seen.append(p["label"])
    for raw in seen:
        t = str(raw).lower()
        if "months_on_book le 1" in t or "last 1 month" in t:
            found.add(NEW)
        elif "months_on_book le 3" in t or "last 3 month" in t:
            found.add(RECENT)
        elif "front" in t:
            found.add(FRONT)
        elif "back" in t:
            found.add(BACK)
        elif "direct" in t:
            found.add(DIRECT)
        elif "acquired" in t:
            found.add(ACQUIRED)
        elif "open pipeline" in t or "pipeline_stage" in t:
            found.add(OPEN_PIPE)
        elif ("region" in t or "geograph" in t or "collateral_geography" in t):
            found.add(GEO)
        elif "whole" in t or "entire" in t or "all loans" in t:
            found.add(WHOLE)
    if not found and not (run.get("findings") or []):
        return set()
    if not found:
        found.add(WHOLE)
    return found


# --------------------------------------------------------------------------- #
# Required disclosures, per expectation.
# --------------------------------------------------------------------------- #
def required_disclosures(exp: dict) -> list:
    out = []
    if exp.get("rolling_population"):
        out.append(("rolling cohort membership", ("rolling cohort",)))
    if expected_token(exp.get("population")) == OPEN_PIPE:
        out.append(("the pipeline population and its exclusion",
                    ("open pipeline",)))
    return out


def load_expectations() -> dict:
    return yaml.safe_load(EXPECTATIONS.read_text())


def key_for(entry: dict) -> str:
    tag, var, pair = entry["intent"], entry["variation"], entry.get("pair") or ""
    return f"{tag}.{var}/{pair}" if pair else f"{tag}.{var}"



def headline_check(exp: dict, findings: list):
    """Was the HEADLINE figure built from the population the question asked for?

    The population axis above asks whether the expected population appears
    anywhere in the answer. That is not the same question, and the difference is
    exactly the V1 audit's finding: an answer can name "the open pipeline" in a
    secondary finding while its headline is composed over the whole extract.

    Where the frozen expectation names a ``headline_finding`` and a
    ``headline_addend_population``, this decomposes the headline —
    ``headline = base + addend`` — and asks which finding's population the addend
    actually came from. It compares VALUES, so a relabelling cannot satisfy it.

    Returns (state, note). State is MATCH / DIVERGE / NOT_APPLICABLE.
    """
    metric = exp.get("headline_finding")
    want = exp.get("headline_addend_population")
    if not metric or not want:
        return "NOT_APPLICABLE", None
    head = next((f for f in findings if f.get("metric") == metric), None)
    if head is None or head.get("value") is None:
        return "NOT_APPLICABLE", f"no finding carries metric {metric}"
    base = next((f for f in findings
                 if (f.get("population") or {}).get("key") == "total"
                 and f.get("kind") == "measure" and f.get("value") is not None), None)
    if base is None:
        return "NOT_APPLICABLE", "no base measure to decompose the headline against"
    addend = float(head["value"]) - float(base["value"])
    want_tok = expected_token(want)
    for f in findings:
        if f.get("value") is None or abs(float(f["value"]) - addend) > 0.01:
            continue
        pop = f.get("population") or {}
        tok = recorded_tokens({"findings": [f]})
        if want_tok in tok:
            return "MATCH", None
        return "DIVERGE", (f"headline addend {addend:,.2f} came from "
                           f"{pop.get('label')!r} ({pop.get('key')}), expected {want}")
    return "DIVERGE", (f"headline addend {addend:,.2f} matches no finding, so the "
                       f"population it was computed over cannot be identified")


def score_run(exp: dict, entry: dict, run: dict, prior: dict = None) -> dict:
    answer = run.get("answer") or ""
    findings = run.get("findings") or []
    refused = bool(run.get("controlledRefusal")) or run.get("ok") is False
    ai = run.get("analyticalIntent") or {}

    # ---------------- SEMANTIC ------------------------------------------- #
    notes = []
    fam_exp, fam_got = set(exp.get("family") or []), set(ai.get("families") or [])
    op_exp, op_got = set(exp.get("operation") or []), set(ai.get("operations") or [])
    fam_ok = (not fam_exp) or bool(fam_exp & fam_got)
    op_ok = (not op_exp) or bool(op_exp & op_got)
    if not fam_ok:
        notes.append(f"family expected {sorted(fam_exp)} got {sorted(fam_got) or 'none'}")
    if not op_ok:
        notes.append(f"operation expected {sorted(op_exp)} got {sorted(op_got) or 'none'}")

    want = expected_token(exp.get("population"))
    want_cmp = expected_token(exp.get("comparison_population")) if exp.get("comparison_population") else None
    got = recorded_tokens(run)
    if want in NON_ROW_POPULATIONS:
        pop_state = "NOT_APPLICABLE"
        notes.append(f"population axis n/a: expectation names {want}, not a funded row set")
    elif refused:
        pop_state = "NOT_APPLICABLE"
        notes.append("population axis n/a: the answer is a refusal, so no population was executed")
    elif not findings:
        pop_state = "NOT_APPLICABLE"
        notes.append("population axis n/a: this route publishes no structured "
                     "finding, so it carries no PopulationRef to check")
    elif want in got and (want_cmp is None or want_cmp in got):
        pop_state = "MATCH"
    else:
        pop_state = "DIVERGE"
        missing = [t for t in (want, want_cmp) if t and t not in got]
        notes.append(f"population expected {missing} got {sorted(got) or 'none recorded'}")

    outcome = (exp.get("expected_outcome") or "").lower()
    outcome_ok, outcome_note = True, None
    if outcome.startswith("refusal"):
        outcome_ok = refused
        outcome_note = ("refusal expected and delivered" if refused
                        else "refusal expected, a substantive answer was given")
    elif "not a balance" in outcome:
        # A milestone question must not be answered with a balance as its
        # headline. Three things satisfy it: a date or horizon; a statement that
        # the milestone is ALREADY PASSED (a milestone behind you has no future
        # date, and saying so is the correct answer, not an evasion); or a
        # refusal. The first version of this rule accepted only the first, and
        # graded "The book has already reached £100.0m" as a divergence — the
        # rule was wrong, not the product. Both readings are reported.
        already = bool(re.search(r"already (reached|passed|exceeded)", answer, re.I))
        dated = bool(re.search(r"\b20\d{2}-\d{2}\b|month|quarter|year", answer, re.I))
        outcome_ok = refused or already or dated
        outcome_strict = refused or dated
        outcome_note = None if outcome_ok else "expected a date, a horizon, or 'already reached'; none present"
        if outcome_ok and not outcome_strict:
            notes.append("milestone already passed — answered as such, not with a future date")
    elif "informational" in outcome:
        outcome_note = "informational response is the correct outcome (not a calculated answer)"
    if outcome_note:
        notes.append(outcome_note)

    head_state, head_note = headline_check(exp, findings)
    if head_note:
        notes.append(head_note)

    semantic = ("MATCH" if (fam_ok and op_ok and outcome_ok
                            and pop_state != "DIVERGE" and head_state != "DIVERGE")
                else "DIVERGE")

    # ---------------- ARITHMETIC ----------------------------------------- #
    # Delegated to nl_reconcile.py — see the module docstring. Recorded here
    # only as whether this run produced numeric findings for it to check.
    arithmetic = "DELEGATED" if findings else "NO_FINDINGS"

    # ---------------- PRESENTATION --------------------------------------- #
    missing_disclosure = []
    if not refused:
        for label, needles in required_disclosures(exp):
            if not any(n in answer.lower() for n in needles):
                # A disclosure is only required where the situation arose.
                # A disclosure ABOUT THE PIPELINE is required only where the
                # answer actually presents a pipeline-derived figure. Q2.4 names
                # the pipeline in the question and is correctly answered "already
                # reached" without reaching for it; demanding the disclosure
                # there measured the rule, not the answer.
                if label.startswith("the pipeline") and not any(
                        (fd.get("capability") or "").startswith(
                            ("funded_balance_forecast", "pipeline_"))
                        for fd in findings):
                    continue
                # A rolling population that did not in fact change membership
                # has nothing to disclose.
                if label.startswith("rolling"):
                    changed = any((fd.get("population") or {}).get("membershipChanged")
                                  for fd in findings)
                    if not changed:
                        continue
                missing_disclosure.append(label)
    unheld = []
    if prior is not None:
        # DIFFERENTIAL: only figures that this revision prints and the
        # comparator did not. Labels are identical on both sides, so they cancel.
        vals = held(findings) | held(prior.get("findings") or [])
        was = set(figures(prior.get("answer") or ""))
        for tok in figures(answer):
            if tok in was:
                continue
            v = numeric(tok)
            if v is None:
                continue
            if not any(close_enough(tok, v, h) for h in vals):
                unheld.append(tok)
    presentation = ("MISSING_DISCLOSURE" if missing_disclosure
                    else "UNHELD_NEW_FIGURE" if unheld else "COMPLETE")

    return {"semantic": semantic, "population": pop_state, "headline": head_state,
            "arithmetic": arithmetic,
            "presentation": presentation, "unheld": unheld,
            "missing": missing_disclosure, "notes": notes, "refused": refused}


def main(paths, prior_dir=None):
    exps = load_expectations()
    priors = {}
    if prior_dir:
        for path in paths:
            q = Path(prior_dir) / Path(path).name
            if not q.exists():
                raise SystemExit(f"comparator run missing: {q}")
            for entry in json.load(open(q)):
                for i, run in enumerate(entry.get("runs") or []):
                    priors[(entry["book"], entry["arm"], key_for(entry), i)] = run
    per_key = collections.defaultdict(list)
    unknown = set()
    for path in paths:
        for entry in json.load(open(path)):
            key = key_for(entry)
            exp = exps.get(key)
            if exp is None:
                unknown.add(key)
                continue
            for i, run in enumerate(entry.get("runs") or []):
                if run.get("hardFailure"):
                    per_key[key].append({"semantic": "HARD_FAILURE", "population": "n/a",
                                         "arithmetic": "n/a", "presentation": "n/a",
                                         "headline": "n/a", "unheld": [], "missing": [],
                                         "notes": [], "refused": False})
                    continue
                prior = priors.get((entry["book"], entry["arm"], key, i)) if priors else None
                per_key[key].append(score_run(exp, entry, run, prior))

    print(f"{'variation':22s} {'SEMANTIC':10s} {'population':16s} {'headline':16s} "
          f"{'ARITHMETIC':11s} PRESENTATION")
    print("-" * 108)
    axes = {a: collections.Counter() for a in ("semantic", "population", "headline",
                                               "arithmetic", "presentation")}
    divergences = []
    for key in sorted(per_key, key=lambda k: (k.split(".")[0], k)):
        rows = per_key[key]
        agg = {a: sorted({r[a] for r in rows}) for a in axes}
        for a in axes:
            for r in rows:
                axes[a][r[a]] += 1
        print(f"{key:22s} {'/'.join(agg['semantic']):10s} {'/'.join(agg['population']):16s} "
              f"{'/'.join(agg['headline']):16s} {'/'.join(agg['arithmetic']):11s} "
              f"{'/'.join(agg['presentation'])}")
        seen = set()
        for r in rows:
            if r["semantic"] == "DIVERGE" or r["presentation"] != "COMPLETE":
                for n in r["notes"] + [f"new figure no finding holds: {t}" for t in r["unheld"]] \
                        + [f"missing disclosure: {m}" for m in r["missing"]]:
                    if n not in seen:
                        seen.add(n)
                        print(f"{'':22s}   ! {n}")
                divergences.append(key)
    print()
    total = sum(axes["semantic"].values())
    print(f"runs scored: {total}   variations: {len(per_key)}"
          + (f"   (no expectation for: {sorted(unknown)})" if unknown else ""))
    for a in ("semantic", "population", "headline", "arithmetic", "presentation"):
        line = "  ".join(f"{k}={v}" for k, v in sorted(axes[a].items()))
        print(f"  {a.upper():13s} {line}")
    print()
    sem_ok = axes["semantic"]["MATCH"]
    pres_bad = (axes["presentation"]["MISSING_DISCLOSURE"]
                + axes["presentation"]["UNHELD_NEW_FIGURE"])
    print(f"SEMANTIC agreement with the frozen expectation : {sem_ok}/{total} "
          f"({100.0 * sem_ok / total:.1f}%)")
    print(f"HEADLINE built from the expected population    : "
          f"{axes['headline']['MATCH']} match, {axes['headline']['DIVERGE']} diverge, "
          f"{axes['headline']['NOT_APPLICABLE']} n/a")
    print( "ARITHMETIC                                      : delegated to nl_reconcile.py")
    print(f"PRESENTATION defects (missing disclosure / unheld new figure) : {pres_bad}")
    print()
    print(f"DIVERGING VARIATIONS ({len(set(divergences))}): {sorted(set(divergences))}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    prior = None
    if "--against" in argv:
        i = argv.index("--against")
        prior = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    main(argv, prior)
