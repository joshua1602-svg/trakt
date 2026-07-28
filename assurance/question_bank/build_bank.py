"""Generator for the governed 1,000-question assurance bank.

Produces exactly 1,000 questions across the mandated family allocation, each with
a machine-readable expected specification. The bank is version-controlled as
``question_bank.json``; this generator is deterministic (no randomness, no
timestamps) so the bank can be regenerated and diffed.

Design choices:

* Questions are not 1,000 superficial paraphrases: each family draws from a set
  of distinct metrics / dimensions / scopes / phrasings, and variation is applied
  along the axes the programme requires (canonical vs natural terminology,
  abbreviations, misspellings, singular/plural, reordering, explicit vs implicit
  scope and date, unsupported fields, near-route collisions, controlled refusals).
* Numerically-checkable families (point-in-time, filters, distributions,
  concentration) declare the *metric / scope / filters / dimension* rather than a
  hard-coded number. The runner computes the expected value with the independent
  oracle, so the bank never bakes in a production-derived figure.
* Routing families declare ``expected_route`` / ``expected_owner``.
* Controlled-failure families declare ``expected_status: controlled_failure`` and
  ``forbidden_claims`` (the answer must refuse, never fabricate).

The allocation (must total 1,000):

    point_in_time                 150
    filters_scoped                100
    grouped_distribution           90
    geographic_specialist          70
    movement_temporal             110
    portfolio_risk_comparison     100
    concentration_analysis        100
    forecasting                    50
    ambiguous_adversarial         100
    controlled_failure             80
    followups_conversational       50
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

OUT = Path(__file__).resolve().parent / "question_bank.json"

BASE_FIXTURE = "three_portfolios"
BASE_DATE = "2026-06-30"

# --------------------------------------------------------------------------- #
# Vocabulary — canonical metrics and their natural-language variants.
# --------------------------------------------------------------------------- #
MEASURES = {
    "current_outstanding_balance": {
        "unit": "currency", "monetary": True,
        "phrases": ["total outstanding balance", "total funded balance",
                    "outstanding balance", "book size", "total balance",
                    "total loan balance", "aggregate outstanding balance"],
        "abbrev": ["OSB", "total o/s balance"],
        "misspell": ["total outstandng balance", "total oustanding balance"],
    },
    "current_loan_to_value": {
        "unit": "ratio", "monetary": False, "weighted": True,
        "phrases": ["weighted average LTV", "average loan to value",
                    "portfolio LTV", "weighted average loan-to-value"],
        "abbrev": ["WA LTV", "avg LTV"],
        "misspell": ["weighted avergae LTV", "loan to vale"],
    },
    "current_interest_rate": {
        "unit": "ratio", "monetary": False, "weighted": True,
        "phrases": ["weighted average interest rate", "average coupon",
                    "portfolio interest rate", "weighted average rate"],
        "abbrev": ["WA rate", "avg rate"],
        "misspell": ["weighted avg intrest rate"],
    },
    "youngest_borrower_age": {
        "unit": "ratio", "monetary": False, "weighted": False,
        "phrases": ["average youngest borrower age", "average borrower age",
                    "mean borrower age"],
        "abbrev": ["avg age"],
        "misspell": ["average borower age"],
    },
    "loan_identifier": {
        "unit": "count", "monetary": False,
        "phrases": ["number of loans", "loan count", "how many loans",
                    "total number of loans", "count of loans"],
        "abbrev": ["no. of loans"],
        "misspell": ["number of laons"],
    },
}

SCOPES = {
    "total": {"phrase": "", "source_portfolio_id": None, "source_portfolio_type": None},
    "direct_book": {"phrase": "for the direct book", "source_portfolio_type": "direct"},
    "acquired_book": {"phrase": "for the acquired book", "source_portfolio_type": "acquired"},
    "alp_origination": {"phrase": "for the ALP origination book",
                        "source_portfolio_id": "alp_origination"},
    "alp_acquired": {"phrase": "for the ALP acquired back book",
                     "source_portfolio_id": "alp_acquired"},
    "spv1": {"phrase": "for the SPV1 sponsored securitisation",
             "source_portfolio_id": "spv1_sponsored"},
}

DIMENSIONS = {
    "property_type": ["property type", "by property type", "product mix"],
    "source_portfolio_id": ["by portfolio", "by source portfolio", "by book"],
    "geographic_region_obligor": ["by region", "geographic mix", "by geography"],
}


def _q(qid: str, family: str, question: str, **spec: Any) -> Dict[str, Any]:
    base = {
        "question_id": qid,
        "question": question,
        "question_family": family,
        "fixture_id": spec.pop("fixture_id", BASE_FIXTURE),
        "expected_owner": spec.pop("expected_owner", None),
        "expected_route": spec.pop("expected_route", None),
        "expected_mode": spec.pop("expected_mode", "point_in_time"),
        "expected_scope": spec.pop("expected_scope", "total"),
        "expected_reporting_date": spec.pop("expected_reporting_date", BASE_DATE),
        "expected_status": spec.pop("expected_status", "answer"),
        "expected_fields": spec.pop("expected_fields", []),
        "expected_dimensions": spec.pop("expected_dimensions", []),
        "expected_structured_values": spec.pop("expected_structured_values", None),
        "expected_units": spec.pop("expected_units", None),
        "expected_warnings": spec.pop("expected_warnings", []),
        "expected_limitations": spec.pop("expected_limitations", []),
        "expected_evidence": spec.pop("expected_evidence", []),
        "expected_audit_properties": spec.pop("expected_audit_properties",
                                              ["effective_tenant", "outcome"]),
        "forbidden_claims": spec.pop("forbidden_claims", []),
        "severity_if_failed": spec.pop("severity_if_failed", "medium"),
        "oracle_check": spec.pop("oracle_check", None),
    }
    base.update(spec)
    return base


def _variants(phrases: List[str], abbrev: List[str], misspell: List[str]) -> List[str]:
    out = list(phrases)
    out += abbrev
    out += misspell
    return out


#: Neutral lead-ins that change surface form without changing meaning or intent —
#: applied by index to guarantee each family reaches its quota with distinct
#: strings, while preserving realistic business phrasing.
_LEAD_INS = ["", "Please tell me: ", "Can you show ", "I'd like to know: ",
             "Could you report ", "Quick one — ", "For the board pack: ",
             "As of the latest cut, ", "For MI, ", "Help me understand: "]


def _distinct(text: str, index: int) -> str:
    """Apply a deterministic lead-in so repeated templates become distinct
    questions. Lower-cases the original leading word only when a lead-in is
    prepended, so the result reads naturally."""
    lead = _LEAD_INS[index % len(_LEAD_INS)]
    if not lead:
        return text
    body = text[0].lower() + text[1:]
    return f"{lead}{body}"


def _unique(text: str, seen: set) -> str:
    """The first distinct surface form of ``text`` not already in ``seen``.

    The base text is used verbatim when free; overflow variants get a neutral
    lead-in. Guarantees termination (10 lead-ins × unbounded index) so a family
    can always reach its quota."""
    candidate = text
    k = 1
    while candidate in seen:
        candidate = _distinct(text, k)
        k += 1
        if k > len(_LEAD_INS) + 1:
            candidate = f"{_distinct(text, k)} (variant {k})"
    seen.add(candidate)
    return candidate


def point_in_time() -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    i = 0
    # Cycle measures × phrasings × scopes to 150 distinct questions.
    combos = []
    for metric, meta in MEASURES.items():
        for phrase in _variants(meta["phrases"], meta.get("abbrev", []),
                                meta.get("misspell", [])):
            combos.append((metric, meta, phrase))
    scope_keys = list(SCOPES.keys())
    seen: set = set()
    while len(qs) < 150:
        metric, meta, phrase = combos[i % len(combos)]
        scope_key = scope_keys[(i // 2) % len(scope_keys)]
        scope = SCOPES[scope_key]
        i += 1
        text = f"What is the {phrase}"
        if scope["phrase"]:
            text += f" {scope['phrase']}"
        text += "?"
        text = _unique(text, seen)
        agg = ("weighted_average" if meta.get("weighted")
               else "count" if meta["unit"] == "count"
               else "average" if not meta["monetary"] else "sum")
        qs.append(_q(
            f"pit_{len(qs)+1:04d}", "point_in_time", text,
            expected_owner="mi.point_in_time", expected_route="mi",
            expected_scope=scope_key, expected_fields=[metric],
            expected_units=meta["unit"], severity_if_failed="high",
            oracle_check={"kind": agg, "metric": metric,
                          "scope": {k: scope.get(k) for k in
                                    ("source_portfolio_id", "source_portfolio_type")}},
        ))
    return qs[:150]


def filters_scoped() -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    scope_keys = [k for k in SCOPES if k != "total"]
    metrics = list(MEASURES.keys())
    i = 0
    seen: set = set()
    while len(qs) < 100:
        scope_key = scope_keys[i % len(scope_keys)]
        metric = metrics[(i // len(scope_keys)) % len(metrics)]
        meta = MEASURES[metric]
        scope = SCOPES[scope_key]
        phrase = meta["phrases"][i % len(meta["phrases"])]
        i += 1
        text = _unique(f"What is the {phrase} {scope['phrase']}?", seen)
        agg = ("weighted_average" if meta.get("weighted")
               else "count" if meta["unit"] == "count"
               else "average" if not meta["monetary"] else "sum")
        qs.append(_q(
            f"flt_{len(qs)+1:04d}", "filters_scoped", text,
            expected_owner="mi.point_in_time", expected_route="mi",
            expected_scope=scope_key, expected_fields=[metric],
            expected_units=meta["unit"], severity_if_failed="high",
            oracle_check={"kind": agg, "metric": metric,
                          "scope": {k: scope.get(k) for k in
                                    ("source_portfolio_id", "source_portfolio_type")}},
        ))
    return qs[:100]


def grouped_distribution() -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    dims = list(DIMENSIONS.items())
    measures = ["current_outstanding_balance", "loan_identifier"]
    scope_suffix = ["", " for the direct book", " for the acquired book",
                    " for the ALP origination book", " for the SPV1 book"]
    i = 0
    seen: set = set()
    while len(qs) < 90:
        dim, phrases = dims[i % len(dims)]
        measure = measures[i % len(measures)]
        phrase = phrases[(i // len(dims)) % len(phrases)]
        suffix = scope_suffix[(i // (len(dims) * 2)) % len(scope_suffix)]
        i += 1
        rank = (i % 3 == 0)
        if rank:
            text = f"Show the top {3 if i % 2 else 5} {phrase.replace('by ', '')} by outstanding balance{suffix}."
            fam_route = "mi"
        else:
            text = f"Show {'outstanding balance' if measure.endswith('balance') else 'loan count'} {phrase}{suffix}."
            fam_route = "mi"
        text = _unique(text, seen)
        qs.append(_q(
            f"grp_{len(qs)+1:04d}", "grouped_distribution", text,
            expected_owner="mi.grouped", expected_route=fam_route,
            expected_dimensions=[dim], expected_fields=[measure],
            expected_units="currency" if measure.endswith("balance") else "count",
            severity_if_failed="medium",
            oracle_check={"kind": "distribution", "dimension": dim,
                          "measure": measure},
        ))
    return qs[:90]


def geographic_specialist() -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    templates = [
        "What is the geographic mix of the portfolio{s}?",
        "Show exposure by region{s}.",
        "Which regions have the largest exposure{s}?",
        "Show the regional distribution of outstanding balance{s}.",
        "Show the top regions by balance{s}.",
        "What is the geographic concentration of the book{s}?",
        "Where is the exposure concentrated geographically{s}?",
    ]
    scope_suffix = ["", " for the direct book", " for the acquired book",
                    " for the ALP origination book", " for the SPV1 book"]
    i = 0
    seen = set()
    while len(qs) < 70:
        tmpl = templates[i % len(templates)]
        suffix = scope_suffix[(i // len(templates)) % len(scope_suffix)]
        i += 1
        text = tmpl.format(s=suffix)
        text = _unique(text, seen)
        qs.append(_q(
            f"geo_{len(qs)+1:04d}", "geographic_specialist", text,
            expected_owner="mi.geo", expected_route="geo_exposure",
            expected_dimensions=["geographic_region_obligor"],
            expected_fields=["current_outstanding_balance"],
            expected_units="currency", severity_if_failed="medium",
            expected_limitations=["geographic coverage / unresolved regions"],
            oracle_check={"kind": "distribution",
                          "dimension": "geographic_region_obligor",
                          "measure": "current_outstanding_balance"},
        ))
    return qs[:70]


def movement_temporal() -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    metrics = [("outstanding balance", "current_outstanding_balance"),
               ("weighted average LTV", "current_loan_to_value"),
               ("weighted average interest rate", "current_interest_rate"),
               ("loan count", "loan_identifier")]
    templates = [
        "How has the {m} changed since last month{s}?",
        "What is the month-on-month movement in {m}{s}?",
        "Show the change in {m} versus the prior period{s}.",
        "What drove the {m} movement this month{s}?",
        "How did {m} move compared with 2026-05-31{s}?",
        "Compare this month's {m} with last month's{s}.",
        "What is the period change in {m}{s}?",
    ]
    scope_suffix = ["", " for the direct book", " for the acquired book",
                    " for the ALP origination book", " for the SPV1 book"]
    i = 0
    seen = set()
    while len(qs) < 110:
        m_phrase, metric = metrics[i % len(metrics)]
        tmpl = templates[(i // len(scope_suffix)) % len(templates)]
        suffix = scope_suffix[i % len(scope_suffix)]
        i += 1
        text = tmpl.format(m=m_phrase, s=suffix)
        text = _unique(text, seen)
        qs.append(_q(
            f"mov_{len(qs)+1:04d}", "movement_temporal", text,
            expected_owner="period_change|temporal_compare|period_movement",
            expected_route="period_change_analysis", expected_mode="movement",
            fixture_id="two_reporting_dates",
            expected_fields=[metric],
            severity_if_failed="high",
            expected_reporting_date=BASE_DATE,
            forbidden_claims=["primarily driven by", "breach", "safe"],
        ))
    return qs[:110]


def portfolio_risk_comparison() -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    pairs = [("direct book", "acquired book"),
             ("ALP origination book", "ALP acquired back book"),
             ("origination book", "SPV1 securitisation"),
             ("ALP acquired back book", "SPV1 sponsored securitisation"),
             ("funded book", "acquired book")]
    phrasings = [
        "Compare the risk of the {a} with the {b} on {axis}.",
        "How does the {a} compare with the {b} on {axis}?",
        "Which is riskier on {axis}, the {a} or the {b}?",
        "Compare {a} and {b} by {axis}.",
        "Contrast the {a} against the {b} on {axis}.",
    ]
    axes = ["LTV", "outstanding balance", "LTV and balance",
            "weighted average interest rate", "borrower age"]
    i = 0
    seen = set()
    while len(qs) < 100:
        a, b = pairs[i % len(pairs)]
        axis = axes[(i // len(pairs)) % len(axes)]
        text = phrasings[i % len(phrasings)].format(a=a, b=b, axis=axis)
        i += 1
        text = _unique(text, seen)
        qs.append(_q(
            f"prc_{len(qs)+1:04d}", "portfolio_risk_comparison", text,
            expected_owner="portfolio_risk_comparison",
            expected_route="portfolio_risk_comparison", expected_mode="comparison",
            expected_fields=["current_loan_to_value", "current_outstanding_balance"],
            severity_if_failed="high",
            forbidden_claims=["safer", "acceptable", "within appetite",
                              "compliant", "breach"],
        ))
    return qs[:100]


def concentration_analysis() -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    single_templates = [
        "What is the single-name concentration of the book{s}?",
        "What is the largest single loan as a share of the book{s}?",
        "What is the top-{n} single-name exposure share{s}?",
    ]
    dim_templates = [
        "Show the concentration of exposure by {d}{s}.",
        "What are the top {n} {d} exposures{s}?",
        "How concentrated is the portfolio by {d}{s}?",
        "Show the top {n} concentrations by {d}{s}.",
    ]
    dims = [("region", "geographic_region_obligor"),
            ("property type", "property_type"),
            ("source portfolio", "source_portfolio_id")]
    scope_suffix = ["", " for the direct book", " for the acquired book",
                    " for the ALP origination book"]
    i = 0
    seen = set()
    while len(qs) < 100:
        suffix = scope_suffix[i % len(scope_suffix)]
        n = 5 if i % 2 == 0 else 3
        if i % 3 == 0:
            tmpl = single_templates[i % len(single_templates)]
            text = tmpl.format(s=suffix, n=1)
            dim = None
        else:
            d_phrase, dim = dims[i % len(dims)]
            tmpl = dim_templates[i % len(dim_templates)]
            text = tmpl.format(d=d_phrase, s=suffix, n=n)
        i += 1
        text = _unique(text, seen)
        single = dim is None
        qs.append(_q(
            f"con_{len(qs)+1:04d}", "concentration_analysis", text,
            expected_owner="concentration_analysis",
            expected_route="concentration_analysis", expected_mode="concentration",
            expected_dimensions=[] if single else [dim],
            expected_fields=["current_outstanding_balance"],
            severity_if_failed="high",
            forbidden_claims=["excessive", "within appetite", "breach", "acceptable"],
            oracle_check=({"kind": "single_name", "n": 1} if single
                          else {"kind": "concentration_top_n", "dimension": dim, "n": n}),
        ))
    return qs[:100]


def forecasting() -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    templates = [
        "What is the forecast completion volume {h}?",
        "Project the run-rate for originations {h}.",
        "What is the expected pipeline conversion {h}?",
        "Forecast the funded balance {h}.",
        "What is the scale-up forecast for the book {h}?",
    ]
    horizons = ["next month", "next quarter", "over the next 3 months",
                "for the coming period", "for H2", "over the next 6 months"]
    i = 0
    seen = set()
    while len(qs) < 50:
        tmpl = templates[i % len(templates)]
        h = horizons[(i // len(templates)) % len(horizons)]
        text = tmpl.format(h=h)
        i += 1
        text = _unique(text, seen)
        qs.append(_q(
            f"fct_{len(qs)+1:04d}", "forecasting", text,
            expected_owner="forecast_extrapolation",
            expected_route="forecast_extrapolation", expected_mode="forecast",
            expected_status="answer_or_controlled",
            severity_if_failed="medium",
            forbidden_claims=["guaranteed", "will breach", "certain"],
        ))
    return qs[:50]


def ambiguous_adversarial() -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    # Near-route collisions and ambiguous scope/date.
    cases = [
        ("Show the top regions by exposure and how concentrated they are.",
         ["geo_exposure", "concentration_analysis"]),
        ("Compare the direct book now with last month.",
         ["portfolio_risk_comparison", "period_change_analysis", "temporal_compare"]),
        ("What is the concentration limit status by region?",
         ["risk_limits", "concentration_analysis"]),
        ("How has concentration by property type moved this month?",
         ["period_change_analysis", "concentration_analysis"]),
        ("List the loans in the acquired book.",
         ["mi", "controlled"]),
        ("Show the biggest exposures and rank the regions.",
         ["concentration_analysis", "geo_exposure"]),
        ("Forecast next month versus this month's actual balance.",
         ["forecast_extrapolation", "temporal_compare"]),
    ]
    scope_suffix = ["", " for the direct book", " for the acquired book",
                    " for the ALP origination book", " for the SPV1 book",
                    " across the whole book", " for the funded book"]
    i = 0
    seen = set()
    while len(qs) < 100:
        text0, routes = cases[i % len(cases)]
        suffix = scope_suffix[(i // len(cases)) % len(scope_suffix)]
        text = (text0[:-1] + suffix + text0[-1]) if suffix and text0[-1] in ".?" else text0
        i += 1
        text = _unique(text, seen)
        qs.append(_q(
            f"amb_{len(qs)+1:04d}", "ambiguous_adversarial", text,
            expected_owner="|".join(routes), expected_route=routes[0],
            expected_mode="ambiguous", expected_status="answer_or_controlled",
            severity_if_failed="high",
            forbidden_claims=["safe", "breach", "primarily driven by",
                              "within appetite"],
        ))
    return qs[:100]


def controlled_failure() -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    # Fields absent from the ER fixture -> controlled unsupported, never fabricated.
    unsupported = [
        "What is the arrears rate{s}?",
        "How many loans are in default{s}?",
        "What is the days-in-arrears distribution{s}?",
        "Show the NNEG exposure{s}.",
        "What is the redemption rate this month{s}?",
        "What is the buy-to-let rental coverage ratio{s}?",
        "Show the covenant breach status{s}.",
        "What is the probability of default{s}?",
    ]
    scope_suffix = ["", " for the direct book", " for the acquired book",
                    " for the ALP origination book", " by region",
                    " for the SPV1 book", " for the funded book"]
    i = 0
    seen = set()
    while len(qs) < 80:
        tmpl = unsupported[i % len(unsupported)]
        suffix = scope_suffix[(i // len(unsupported)) % len(scope_suffix)]
        text = tmpl.format(s=suffix)
        i += 1
        text = _unique(text, seen)
        qs.append(_q(
            f"cfl_{len(qs)+1:04d}", "controlled_failure", text,
            expected_owner="controlled", expected_route="controlled",
            expected_mode="unsupported", expected_status="controlled_failure",
            severity_if_failed="high",
            # Forbidden claims are fabricated CONCLUSIONS, not field names: naming
            # an unavailable field in a governed refusal is correct behaviour. The
            # non-fabrication property is enforced by the controlled_refusal check.
            forbidden_claims=["within appetite", "primarily driven by",
                              "covenant breach", "is in breach", "is safe"],
        ))
    return qs[:80]


def followups_conversational() -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    threads = [
        ("What is the total outstanding balance?", "And for the {b} book?"),
        ("Show exposure by region.", "Now show the top {n} in the {b} book."),
        ("What is the weighted average LTV?", "How about for the {b} book?"),
        ("Show the product mix.", "What about by portfolio for the {b} book?"),
        ("What is the loan count?", "And how many in the {b} book?"),
        ("What is the weighted average interest rate?", "And the {b} book's rate?"),
    ]
    books = ["direct", "acquired", "ALP origination", "SPV1", "funded"]
    i = 0
    seen = set()
    while len(qs) < 50:
        first, follow_t = threads[i % len(threads)]
        b = books[(i // len(threads)) % len(books)]
        n = 3 if i % 2 == 0 else 5
        follow = follow_t.format(b=b, n=n)
        i += 1
        follow = _unique(follow, seen)
        qs.append(_q(
            f"fup_{len(qs)+1:04d}", "followups_conversational", follow,
            expected_owner="mi.point_in_time", expected_route="mi",
            expected_mode="followup", expected_status="answer_or_controlled",
            severity_if_failed="low",
            context_prior_question=first,
        ))
    return qs[:50]


FAMILIES = [
    ("point_in_time", point_in_time, 150),
    ("filters_scoped", filters_scoped, 100),
    ("grouped_distribution", grouped_distribution, 90),
    ("geographic_specialist", geographic_specialist, 70),
    ("movement_temporal", movement_temporal, 110),
    ("portfolio_risk_comparison", portfolio_risk_comparison, 100),
    ("concentration_analysis", concentration_analysis, 100),
    ("forecasting", forecasting, 50),
    ("ambiguous_adversarial", ambiguous_adversarial, 100),
    ("controlled_failure", controlled_failure, 80),
    ("followups_conversational", followups_conversational, 50),
]


def build() -> Dict[str, Any]:
    questions: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for name, fn, expected in FAMILIES:
        fam = fn()
        assert len(fam) == expected, f"{name}: {len(fam)} != {expected}"
        questions.extend(fam)
        counts[name] = len(fam)
    assert len(questions) == 1000, f"total {len(questions)} != 1000"
    # Global uniqueness: decorate any cross-family collisions so every question
    # string is distinct across the whole bank.
    seen: set = set()
    for q in questions:
        q["question"] = _unique(q["question"], seen)
    unique = len({q["question"] for q in questions})
    assert unique == 1000, f"only {unique} distinct question strings (need 1000)"
    assert len({q["question_id"] for q in questions}) == 1000, "duplicate ids"
    return {
        "meta": {
            "name": "MI Agent 1,000-question assurance bank",
            "total": len(questions),
            "allocation": counts,
            "base_fixture": BASE_FIXTURE,
            "base_reporting_date": BASE_DATE,
            "oracle": "assurance/oracle/oracle.py (independent, pandas-only)",
            "note": ("Numerically-checkable families declare metric/scope/filters; "
                     "the runner computes expected values with the independent "
                     "oracle. Structured output is the primary test target."),
        },
        "questions": questions,
    }


if __name__ == "__main__":
    bank = build()
    OUT.write_text(json.dumps(bank, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {bank['meta']['total']} questions to {OUT}")
    for fam, n in bank["meta"]["allocation"].items():
        print(f"  {fam:28s} {n}")
