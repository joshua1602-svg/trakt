"""The NL robustness bank: 9 analytical intents, 44 paraphrases.

Transcribed verbatim from the brief. The ONE departure is Q8, which the brief
names as single vs joint borrowers: neither test book carries a borrower
structure field (the only borrower columns are identifiers, age and an
impairment flag), and the brief forbids inventing one. Per the operator's
ruling, Q8 runs against ALL THREE governed X/Y pairs — provenance, seasoning and
dimension values — using the SAME four sentence shapes for each, so the
paraphrase axis is held constant and only the population pair varies.

Region names differ between the books, so the dimension-value pair is
parameterised per book rather than hard-coded.
"""

CANONICAL = {
    "Q1": "How has the profile of new originations changed in the last few months?",
    "Q2": "When are we likely to reach £100m of loans?",
    "Q3": ("What is the current value of outstanding offers in the pipeline, "
           "how much do we expect to complete, and when?"),
    "Q4": "What is the current completion run rate?",
    "Q5": "Are we likely to breach any concentration limits in the next three months?",
    "Q6": "Which limits are closest to breaching?",
    "Q7": "How does the risk profile of older vintages compare with the front book?",
    "Q8": "How has the balance from dimension X changed relative to dimension Y?",
    "Q9": "Based on the current pipeline, what is the forecast funded balance?",
}

#: Intents the brief flags as highest risk — 5 genuine LLM runs each, 3 elsewhere.
HIGH_RISK = ("Q1", "Q3", "Q5", "Q7", "Q8")

VARIATIONS = {
    "Q1": [
        "How has the profile of our new lending changed over the last few months?",
        "Are we originating different types of loans now compared with a few months ago?",
        "How does recent lending compare with what we were originating earlier in the year?",
        "Has the risk and borrower profile of new business changed recently?",
    ],
    "Q2": [
        "When do we expect the funded book to reach £100m?",
        "At the current run rate, when should we get to £100m of loans?",
        "When are we forecast to cross £100m of funded balances?",
        "Based on the current book and pipeline, how long until we reach £100m?",
    ],
    "Q3": [
        "How much do we currently have at offer and how much of it is likely to complete?",
        "What's the value of outstanding offers, and when do we expect them to fund?",
        "Of the current offer pipeline, how much should convert and over what timeframe?",
        "How much is sitting at offer today and what do we expect to complete from it and when?",
    ],
    "Q4": [
        "What completion rate are we currently running at?",
        "How many loans are we completing at the moment?",
        "What's our recent run rate for completions?",
        "Based on the last few weeks, what level of completions are we currently achieving?",
    ],
    "Q5": [
        "Are any of our concentration limits likely to breach over the next three months?",
        "Do we expect to hit any concentration limits in the next quarter?",
        "Which concentration limits are at risk of being breached over the next three months?",
        "Based on the forecast book, are any of our limits likely to come under pressure?",
    ],
    "Q6": [
        "Which concentration limits have the least headroom?",
        "Where are we closest to our limits?",
        "Which of our limits are currently most at risk?",
        "Show me the concentration limits we're closest to hitting.",
    ],
    "Q7": [
        "How does the front book compare with our older lending from a risk perspective?",
        "Are older loans riskier than the loans we've originated recently?",
        "How different is the risk profile of recent originations versus the back book?",
        "Compare the credit profile of the front book with our seasoned loans.",
    ],
    "Q9": [
        "What do we expect the funded book to grow to based on the current pipeline?",
        "Where is the funded balance forecast to get to from today's pipeline?",
        "If the current pipeline converts as expected, what will our funded balance be?",
        "What funded balance should we expect once the current pipeline flows through?",
    ],
}

#: The four Q8 sentence shapes, with the two population names substituted. Held
#: identical across the three pairs so a difference in outcome is attributable to
#: the POPULATION RESOLVER, not to the phrasing.
Q8_SHAPES = [
    "How has lending to {a} changed compared with {b}?",
    "How has the balance of {a} loans moved relative to {b} loans?",
    "Are {a} and {b} balances developing differently over time?",
    "Compare how the {a} and {b} books have changed over the last few months.",
]

#: (pair id, resolver under test, A, B). Regions are per-book.
Q8_PAIRS_COMMON = [
    ("provenance", "portfolio_lens (governed registry)", "direct", "acquired"),
    ("seasoning", "seasoning binary partition", "the front book", "the back book"),
]

Q8_REGIONS = {
    "alderbridge": ("South East", "London"),
    "kestrelmoor": ("North West", "Scotland"),
}


def q8_variations(book: str):
    """The twelve Q8 phrasings for one book."""
    pairs = list(Q8_PAIRS_COMMON)
    a, b = Q8_REGIONS[book]
    pairs.append(("dimension_value", "literal category match on a governed "
                                     "dimension", a, b))
    out = []
    for pair_id, resolver, a, b in pairs:
        for index, shape in enumerate(Q8_SHAPES, start=1):
            out.append({"pair": pair_id, "resolver": resolver,
                        "variation": index, "question": shape.format(a=a, b=b)})
    return out


def bank(book: str):
    """Every (intent, variation) pair for one book, in a stable order."""
    rows = []
    for intent in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"):
        for index, question in enumerate(VARIATIONS[intent], start=1):
            rows.append({"intent": intent, "variation": index,
                         "question": question, "pair": None, "resolver": None})
    for entry in q8_variations(book):
        rows.append({"intent": "Q8", "variation": entry["variation"],
                     "question": entry["question"], "pair": entry["pair"],
                     "resolver": entry["resolver"]})
    for index, question in enumerate(VARIATIONS["Q9"], start=1):
        rows.append({"intent": "Q9", "variation": index, "question": question,
                     "pair": None, "resolver": None})
    return rows


def runs_for(intent: str) -> int:
    return 5 if intent in HIGH_RISK else 3


if __name__ == "__main__":
    for book in ("alderbridge", "kestrelmoor"):
        rows = bank(book)
        calls = sum(runs_for(r["intent"]) for r in rows)
        print(f"{book}: {len(rows)} variations, {calls} genuine LLM runs required")
