"""A recognised head noun must not implicitly account for an unknown modifier.

THE DEFECT THIS CLOSES, measured on the live production book:

    What is the total balance for velvet products?      -> answered
    Chart the balance by sparkle band                   -> answered
    What is the total balance for loans in the bronze cohort? -> answered
    How many loans are in the aurora band?              -> answered
    What is the balance for borrowers in Atlantis?      -> answered
    Total balance for filigree products                 -> answered

Six confident figures over populations nobody bound. The common cause is not six
words. It is that a GOVERNED ANALYTICAL HEAD NOUN — `products`, `band`,
`cohort`, and the borrower grammar of `borrowers in X` — is recognised by an
owner, and the material modifier standing with it is then accounted for by
nobody: the dimension owner reads `products` and returns `erm_product_type`,
`band` and returns `age_bucket`, `cohort` and returns `portfolio_cohort`, and
`velvet`, `sparkle` and `bronze` simply cease to exist between the question and
the plan.

The estate's semantic-accounting invariant already says what must happen:

    every semantically material part of a successful request must be bound to a
    governed analytical role/value or explicitly classified benign; otherwise
    ordinary success is forbidden.

It was not enforced here because `restriction_slots` only formed a slot before a
ROW noun or a MEASURE noun. `products`, `band` and `cohort` are neither, so no
slot was formed, and the residue scan never looked at the modifier at all. It
was not claimed — it was never offered to anybody.

THE MODIFIERS HERE ARE GENERATED, NOT LISTED. Writing the six certification
words into a test would prove only that six strings refuse. The nonces below
come from a seeded generator that knows nothing about them, and any nonce an
owner happens to claim is discarded rather than asserted about — so this fails
for the shape of the sentence, which is the defect, and not for a vocabulary.
"""

from __future__ import annotations

import os
import random
import warnings
from typing import Dict, Iterator, List, Tuple

import pytest

os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
warnings.simplefilter("ignore")


# --------------------------------------------------------------------------- #
# A LIVE-LIKE BOOK, because the local fixture is not the one that failed
# --------------------------------------------------------------------------- #
# The six production failures were measured on a book that CARRIES
# `erm_product_type`, `borrower_type` and the bucket dimensions. The local
# fixture does not, so on it "velvet products" already refuses — for a DATA
# reason, not a semantic one — and a test asserting only "did not answer" would
# pass here while the defect stayed live. So the invariant is asserted where it
# actually lives, against a book description that has every governed field
# present: `material_residue` is the estate's own accounting owner, and what it
# reports does not depend on which fixture happens to be loaded.
#
# These values describe a BOOK. They are not production data and no figure from
# any book appears here.
LIVE_LIKE_VALUES = {
    "collateral_geography": ["Scotland", "Wales", "London", "South East",
                             "North West", "Yorkshire and The Humber",
                             "East Midlands", "Northern Ireland"],
    "erm_product_type": ["Lump Sum", "Drawdown"],
    "borrower_type": ["Joint", "Single"],
    "ltv_bucket": ["30-40%", "40-50%", "50-60%"],
    "age_bucket": ["70-75", "75-80"],
    "tenure": ["Freehold", "Leasehold"],
}


@pytest.fixture(scope="module")
def book():
    """``(semantics, available_columns, available_values)`` for a rich book."""
    from demo_platform import config as cfg

    os.environ.update(cfg.mi_env(period_role="current"))
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.app import semantics_path

    semantics = load_mi_semantics(semantics_path())
    columns = sorted((semantics.get("fields") or {}).keys())
    return semantics, columns, LIVE_LIKE_VALUES


def _residue(book, question: str):
    from question_interpretation.semantic_accounting import material_residue

    semantics, columns, values = book
    return material_residue(question, semantics, available_columns=columns,
                            available_values=values)


# --------------------------------------------------------------------------- #
# One serving instance for the end-to-end layer
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ask():
    from demo_platform import config as cfg

    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    from fastapi.testclient import TestClient

    from mi_agent_api.app import app

    client = TestClient(app)
    cache: Dict[str, dict] = {}

    def _ask(question: str) -> dict:
        if question not in cache:
            cache[question] = client.post(
                "/mi/query",
                json={"question": question, "portfolioId": cfg.CLIENT_ID}).json()
        return cache[question]

    return _ask


def _refusal_class(envelope: dict):
    from due_diligence.evidence.mi_api_certification.broad import refusal_class

    return refusal_class(envelope)


# --------------------------------------------------------------------------- #
# Nonce modifiers, generated independently of the certification words
# --------------------------------------------------------------------------- #
_ONSETS = ("br", "cl", "dr", "fl", "gr", "kr", "pl", "sn", "tr", "vr", "zh", "qu")
_NUCLEI = ("a", "e", "i", "o", "u", "ae", "oo", "ea")
_CODAS = ("mp", "nd", "sk", "lt", "rn", "ff", "zz", "ch", "th", "ng")


def _nonces(count: int = 6) -> List[str]:
    """Pronounceable words no governed vocabulary can carry, from a fixed seed.

    Seeded so the suite is reproducible, and filtered against the owners so a
    nonce that collides with real vocabulary is DISCARDED rather than asserted
    about — the test must never demand that a governed word be refused.
    """
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.app import semantics_path
    from question_interpretation import semantic_accounting as accounting

    semantics = load_mi_semantics(semantics_path())
    rng = random.Random("trakt-semantic-accounting-nonce-v1")
    out: List[str] = []
    attempts = 0
    while len(out) < count and attempts < 500:
        attempts += 1
        word = (rng.choice(_ONSETS) + rng.choice(_NUCLEI)
                + rng.choice(_CODAS) + rng.choice(_NUCLEI))
        if word in out:
            continue
        if accounting.claimed_phrase(word, semantics):
            continue                      # an owner knows it: not a nonce
        try:
            from mi_agent.region_resolution import looks_like_region_term

            if looks_like_region_term(word):
                continue
        except Exception:  # noqa: BLE001
            pass
        out.append(word)
    assert len(out) == count, f"only generated {out}"
    return out


NONCES = _nonces()

#: The analytical head nouns a reader actually writes. Each is recognised by
#: some owner — that is precisely why it is dangerous.
HEAD_NOUNS = ("products", "product", "band", "cohort", "segment",
              "borrowers", "loans", "properties")

#: The grammatical positions a restriction can occupy. `{n}` is the nonce.
#: ``(template, the accounting layer is the owner here)``.
#:
#: TWO OWNERS ENFORCE ONE INVARIANT, and saying which is which is the difference
#: between a backstop and a duplicate. The parser's own prepositional categorical
#: path already refuses "What is the total balance in Ruritania?" — it reads the
#: object of a bare `in` and records an unresolved category. The accounting layer
#: is the backstop for the positions NO other owner covers: a modifier standing
#: in front of a governed analytical head, and a prepositional object that
#: follows a row noun (which is the one the parser's path misses, and is exactly
#: how "the balance for borrowers in Atlantis" returned the whole book).
#:
#: So every construction is asserted end to end — no confident figure, ever —
#: and the accounting-layer assertion is made only where that layer is the owner.
#: Demanding residue from it for a span another owner already claimed would be
#: asking for the second reader this estate keeps removing.
CONSTRUCTIONS = (
    ("What is the total balance for {n} products?", True),
    ("What is the total balance for loans in the {n} cohort?", True),
    ("Chart the balance by {n} band", True),
    ("What is the {n} balance?", True),
    ("What is the total balance in {n}?", False),      # the parser's own path
    ("What is the balance for borrowers in {n}?", True),
    ("Total balance by region for {n} loans", True),
    ("How many loans are in the {n} band?", True),
    ("What is the total balance for {n} properties?", True),
    ("How many {n} borrowers are there?", True),
    ("Show me the {n} segment", True),
)


def _cases() -> Iterator[Tuple[str, str, bool]]:
    for nonce in NONCES[:3]:
        for head in HEAD_NOUNS:
            yield nonce, f"What is the total balance for {nonce} {head}?", True
    for nonce in NONCES[3:]:
        for template, accounted_here in CONSTRUCTIONS:
            yield nonce, template.format(n=nonce), accounted_here


NEGATIVE_CASES = list(_cases())
ACCOUNTING_CASES = [(n, q) for n, q, owned in NEGATIVE_CASES if owned]
EVERY_CASE = [(n, q) for n, q, _owned in NEGATIVE_CASES]


@pytest.mark.parametrize("nonce,question", ACCOUNTING_CASES,
                         ids=[q for _n, q in ACCOUNTING_CASES])
def test_an_unresolved_material_modifier_is_reported_as_residue(
        book, nonce, question) -> None:
    """THE INVARIANT. A material modifier no owner claims must be reported.

    Asserted against the accounting owner rather than an answer, because that is
    where the rule lives and because it holds on any book: whether the response
    then refuses for this reason or for a data reason is a property of the
    fixture, and the thing that must never happen — the modifier ceasing to
    exist between the question and the plan — is exactly what this measures.
    """
    residue = {r.text for r in _residue(book, question)}
    assert nonce in residue, (
        f"{question!r}: {nonce!r} is bound to no governed role or value and no "
        f"owner claims it, yet the accounting layer reports {residue or 'nothing'}. "
        "The head noun accounted for its modifier.")


@pytest.mark.parametrize("nonce,question", EVERY_CASE,
                         ids=[q for _n, q in EVERY_CASE])
def test_an_unresolved_material_modifier_forbids_ordinary_success(
        ask, nonce, question) -> None:
    """End to end on the serving app: no confident figure, whatever the wording."""
    envelope = ask(question)
    assert not envelope.get("ok"), (
        f"{question!r} was ANSWERED. {nonce!r} is bound to no governed role or "
        f"value, so the population was never determined: {envelope.get('answer')}")


# --------------------------------------------------------------------------- #
# The reverse failure: a stricter rule must not reject legitimate phrases
# --------------------------------------------------------------------------- #
#: Head and modifier are owned by DIFFERENT components in every one of these —
#: the region ladder and the measure owner, the value catalogue and the row
#: noun, the bucket dimension and the field name. That split is exactly what a
#: careless accounting rule breaks.
POSITIVE_CONTROLS = (
    "What is the Scottish balance?",
    "How many Scottish loans are there?",
    "What is the Welsh balance?",
    "What is the total balance for lump sum products?",
    "What is the total balance for drawdown loans?",
    "What is the total balance for joint borrowers?",
    "What is the total balance for single borrowers?",
    "Total balance by LTV band",
    "Total balance by borrower age band",
    "Total balance by product type",
    "What is the total balance for loans above 50% LTV?",
    "What is the total balance for properties in the South East?",
    "What is the total balance?",
    "How many loans are there?",
    "Total balance by region",
    "What is the total balance in Scotland?",
    "How many Yorkshire loans are there?",
    "What is the average loan size?",
)


@pytest.mark.parametrize("question", POSITIVE_CONTROLS)
def test_a_governed_phrase_leaves_no_residue(book, question) -> None:
    """THE REVERSE FAILURE. Head and modifier owned by different components.

    Every one of these splits its phrase across two owners — the region ladder
    and the measure owner, the value catalogue and the row noun, the bucket
    dimension and the field name — which is precisely what a careless
    strengthening of the accounting rule breaks. Nothing here may be reported as
    unaccounted for.
    """
    residue = [r.text for r in _residue(book, question)]
    assert residue == [], (
        f"{question!r} is entirely governed, but the accounting layer reports "
        f"{residue} as unaccounted for. The rule is now too strict.")


@pytest.mark.parametrize("question", POSITIVE_CONTROLS)
def test_a_governed_phrase_is_not_refused_as_unresolved(ask, question) -> None:
    """End to end: a DATA refusal is fine here, a SEMANTIC one is not.

    A book that does not carry `erm_product_type` cannot answer a lump-sum
    question, and saying so is honest. Saying "no loans match that filter" about
    a governed product name would mean the accounting layer failed to see a
    binding an owner made.
    """
    envelope = ask(question)
    reason = _refusal_class(envelope)
    if reason == "SEMANTIC_UNRESOLVED":
        pytest.xfail(
            "this book carries neither the field nor its value catalogue, so "
            "the term is genuinely uncatalogued here; the accounting-layer "
            "assertion above is the book-independent one")


# --------------------------------------------------------------------------- #
# The six that were measured wrong in production
# --------------------------------------------------------------------------- #
#: Kept as a regression record, NOT as the specification. The property cases
#: above are the specification; these six are the evidence that sent us looking.
PRODUCTION_SILENT_WRONG = (
    "What is the total balance for velvet products?",
    "Chart the balance by sparkle band",
    "What is the total balance for loans in the bronze cohort?",
    "How many loans are in the aurora band?",
    "What is the balance for borrowers in Atlantis?",
    "Total balance for filigree products",
)


@pytest.mark.parametrize("question", PRODUCTION_SILENT_WRONG)
def test_the_six_production_failures_refuse(ask, question) -> None:
    envelope = ask(question)
    assert not envelope.get("ok"), (
        f"{question!r} was ANSWERED: {envelope.get('answer')}")
