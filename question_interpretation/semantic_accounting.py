#!/usr/bin/env python3
"""Was every material part of the question accounted for by some owner?

THE HOLE THIS CLOSES. The estate fails closed on a requested population it has
first NOTICED. Every existing guard compares something the question stated with
something the execution did — and a qualifier no owner recognised states nothing,
so there is nothing to compare and the answer succeeds over a broader population:

    user states a material restriction
        -> no owner recognises it
        -> no canonical facet exists
        -> the requested-vs-executed guards have nothing to reconcile
        -> a whole-book figure is returned, confidently, in silence

Measured: "Give me the Scottish balance." returned £115,450,800.70 where
£25,405,654.23 was asked for, `ok`, with no disclosure of any kind. Its sibling
"How many Scottish loans are there?" refuses correctly — the only difference is
that the second has a row noun for the residue scan to anchor on.

THIS IS NOT A SECOND PARSER, and that distinction is the whole design. It
decides nothing about what the question means. It asks the owners that already
ship — the value catalogue, the region ladder, the measure, statistic, dimension,
period, scope and framing vocabularies — *which spans did you claim?*, and then
looks at what is left. It never resolves a population of its own: a residue it
cannot explain is reported as unresolved, never bound.

    ONE semantic interpretation  +  ONE completeness check

The alternative — a second reader that decides what the first one missed — is
the duplicate-owner defect this programme has spent every sprint removing.

WHY POSITION, NOT PLAUSIBILITY. Materiality is decided by GRAMMAR, not by
guessing whether a word looks like a category. A word standing attributively
before a row noun ("Scottish loans") or before a measure noun ("Scottish
balance") is in a position where a governed restriction goes; a word elsewhere in
the sentence is not. That is why this can be strict without being fuzzy — and it
matters, because the region ladder answers TRUE for "me" and "so" (ME is Medway,
SO is Southampton). Any rule that asked "could this be a place?" would refuse
"give me the balance". This one never asks.

WHAT IT DOES NOT DO. It does not require every token to map to a field. Ordinary
English — articles, auxiliaries, presentation verbs, conjunctions, framing — is
claimed by owners that already exist, so the benign vocabulary here is small and
deliberately not a dumping ground.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from . import lexical as _lexical

__all__ = ["Residue", "material_residue", "claimed_phrase", "consumed_spans"]

#: A word, WITHOUT trailing sentence punctuation: "balance." must be read as the
#: measure "balance", or the owner that knows it declines and the head noun of
#: "Give me the Scottish balance." is a token nothing recognises.
_WORD = re.compile(r"[a-z0-9£][a-z0-9'%£-]*[a-z0-9%£]|[a-z0-9£]")

#: The longest phrase an owner is asked about. The governed multi-word values
#: this estate carries ("north west england", "inner london east") sit inside it,
#: and asking about longer spans costs time for nothing.
_MAX_PHRASE = 4


class Residue(tuple):
    """``(text, head_noun, position)`` — a material span nobody claimed."""

    __slots__ = ()

    def __new__(cls, text: str, head: str, position: str):
        return super().__new__(cls, (text, head, position))

    @property
    def text(self) -> str:
        return self[0]

    @property
    def head(self) -> str:
        return self[1]

    @property
    def position(self) -> str:
        return self[2]


def _owners():
    """The owners, imported lazily so this module can be read from either side.

    Returns a bundle of callables; a missing owner simply claims nothing, which
    can only make this layer quieter, never louder about something real.
    """
    bundle: Dict[str, Any] = {}
    try:
        from mi_agent import llm_query_parser as parser

        bundle["parser"] = parser
    except Exception:  # noqa: BLE001
        bundle["parser"] = None
    try:
        from mi_agent import statistic

        bundle["statistic"] = statistic
    except Exception:  # noqa: BLE001
        bundle["statistic"] = None
    try:
        from mi_agent import region_resolution

        bundle["region"] = region_resolution
    except Exception:  # noqa: BLE001
        bundle["region"] = None
    return bundle


def _norm(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


#: Memo for `claimed_phrase`. The slot scan asks about every sub-phrase of every
#: slot, and the same short phrases recur across slots and across the several
#: `_resolve_population` calls one parse makes. The expensive consultation is
#: `_explicit_dimensions`, which rebuilds the registry's term regexes each time
#: it is asked — 10 rebuilds per parse at ~18ms, which was the whole of the cost
#: this layer added.
_CLAIM_MEMO: Dict[Tuple[Any, ...], Tuple[Any, Any, Any, bool]] = {}
_CLAIM_MEMO_CAP = 8192


def claimed_phrase(phrase: str, semantics: dict, *,
                   available_columns=None, available_values=None) -> bool:
    """Does ANY existing owner claim this phrase? Ask them; never guess.

    Multi-word first, because a governed value, region or measure may span
    several words and asking word-by-word would report the halves of a name it
    knows perfectly well ("north west", "weighted average", "borrower age").
    """
    text = re.sub(r"\s+", " ", str(phrase or "")).strip().lower()
    if not text:
        return True
    # THE OBJECTS ARE HELD AND RE-CHECKED, not trusted by id. CPython reuses an
    # id once its object is collected, so an id-keyed memo can hand back another
    # book's answer. The entry keeps the objects it was computed from and the
    # hit is only taken when they are the SAME objects.
    memo_key = (text, id(semantics), id(available_columns), id(available_values))
    cached = _CLAIM_MEMO.get(memo_key)
    if cached is not None:
        held_semantics, held_columns, held_values, answer = cached
        if (held_semantics is semantics and held_columns is available_columns
                and held_values is available_values):
            return answer
    answer = _claimed_phrase_uncached(text, semantics, available_columns,
                                      available_values)
    if len(_CLAIM_MEMO) >= _CLAIM_MEMO_CAP:
        _CLAIM_MEMO.clear()
    _CLAIM_MEMO[memo_key] = (semantics, available_columns, available_values,
                             answer)
    return answer


def _claimed_phrase_uncached(text: str, semantics: dict, available_columns,
                             available_values) -> bool:
    own = _owners()
    parser, stat, region = own["parser"], own["statistic"], own["region"]

    if region is not None:
        try:
            # Multi-word only. The ladder resolves two-letter POSTCODE AREAS —
            # "me" is Medway, "so" is Southampton — so a single token is never
            # claimed as a place here. The value catalogue and the parser's own
            # owner still claim a real one-word region below.
            if " " in text and region.looks_like_region_term(text):
                return True
        except Exception:  # noqa: BLE001
            pass
    if stat is not None and " " not in text:
        # ONE TOKEN ONLY. `statistic_named` SEARCHES its argument, so asking it
        # about "wa ltv" would claim the measure along with the statistic. The
        # multi-word statistic phrases are consumed at question level by that
        # owner's own offset-preserving mask.
        try:
            if stat.statistic_named(text):
                return True
        except Exception:  # noqa: BLE001
            pass
    if parser is None:
        return False
    try:
        if parser._categorical_value_field(text, available_values, semantics):
            return True
        # EXACTLY THIS PHRASE, not a phrase containing something they know.
        #
        # Both owners answer about a span INSIDE the text they are given, so a
        # bare truthiness test let any phrase containing a measure claim itself
        # whole: `_detect_metric("scottish balance")` matches "balance", and
        # reading that as "the owner claims 'scottish balance'" is how the
        # residue this layer exists to find disappeared again. Over-claiming
        # here is silent; under-claiming is merely noisy.
        _terms = parser._explicit_dimensions(
            text, semantics, available_columns=available_columns)[1] or ()
        if any(_norm(term) == text for term in _terms):
            return True
        _matched = parser._detect_metric(text, semantics)[2] or ()
        if any(_norm(match) == text for match in _matched):
            return True
    except Exception:  # noqa: BLE001
        pass
    if text in _BENIGN or _is_row_noun(text):
        return True
    # A CONTRACTION IS THE WORD IT CONTRACTS. The framing vocabularies spell
    # these without the apostrophe — "whats", "id" — so "What's our average
    # LTV?" reported `what's` as an unrecognised category and refused a question
    # that answered. Both readings are offered: the apostrophe removed
    # ("what's" -> "whats") and the stem before it ("what's" -> "what"), because
    # the estate's lists carry each form somewhere.
    # A POSSESSIVE IS A DETERMINER, NOT A CATEGORY NAME. "today's pipeline",
    # "last month's balance" — the possessive says WHOSE, and the thing it
    # qualifies is owned by the period and scope owners. No governed value in
    # this estate is spelled possessively, so a word ending in "'s" standing in
    # a restriction slot is grammar, not an unrecognised category. Without this
    # the analytical-head rule reported `today's` as a category the book does
    # not carry.
    if " " not in text and text.endswith("'s"):
        return True
    if "'" in text:
        for variant in (text.replace("'", ""), text.split("'", 1)[0]):
            if variant and variant != text and claimed_phrase(
                    variant, semantics, available_columns=available_columns,
                    available_values=available_values):
                return True
    # A NUMBER IS A VALUE, NOT A CATEGORY NAME. "60-70%", "£25m" and "80+" are
    # owned by the threshold and bucket machinery, which reads them as bounds;
    # reporting one as an unrecognised category refused "how many loans are in
    # the 60-70% LTV bucket?", a question the estate answers.
    if re.fullmatch(r"[£$€]?[\d][\d,._%+/-]*[%+]?", text):
        return True
    if _in_registry_vocabulary(text, semantics):
        return True
    # THE COMPARATOR OWNER. "above", "over", "below" and their spellings are how
    # a bound is written, and `lexical` owns that vocabulary. `condition_span`
    # claims a NUMERIC bound at question level, but a field-to-field comparison
    # ("balance above current valuation") carries no number for it to find, so
    # the comparator word arrived here as an unrecognised category.
    try:
        if re.fullmatch(_lexical.comparator_alternation(), text, re.I):
            return True
    except Exception:  # noqa: BLE001
        pass
    if " " in text:
        return False
    # A single token: the shared owner test the unknown-category paths already
    # use. It consults the value catalogue, the dimension, measure and scope
    # owners, the period owner, and the estate's framing, aggregation, chart and
    # analytical vocabularies — which is where ordinary English is claimed.
    try:
        return bool(parser._claimed_by_an_owner(
            text, semantics, available_columns, available_values))
    except Exception:  # noqa: BLE001
        return False


#: Pure function words that carry no analytical meaning in any position: the
#: personal pronouns and the bare auxiliaries. Everything else ordinary — the
#: articles, prepositions, conjunctions, presentation verbs, aggregation and
#: chart vocabulary — is claimed by an OWNER, not listed here.
#:
#: This stays this small on purpose. A large benign list is the same hole as a
#: large ignore list: another route by which a meaningful term can vanish. Every
#: word here is one no governed reading could ever attach to.
_BENIGN = frozenset("""
i me my we us our you your they them their it its
do does did done be is are was were been being am
will would shall should can could may might must
""".split())


def _in_registry_vocabulary(word: str, semantics: dict) -> bool:
    """Does this word appear in ANY governed field's own name or synonyms?

    THE REGISTRY IS THE EVIDENCE. `_detect_metric` reports the synonym it
    MATCHED, which may be shorter than the phrase a reader wrote: "property
    valuation" and "collateral valuation" both match the synonym "valuation",
    while the registry's synonym list for that same field contains "property
    value" and "collateral value". So the modifier is the registry's own
    vocabulary and was being reported as a category the book does not carry.

    Asking the registry directly is narrower than it looks: a word qualifies
    only by appearing in a governed field's name or synonyms, which is why
    "scottish", "platinum", "risky" and "good" do not.
    """
    key = id(semantics)
    cached = _VOCAB_CACHE.get(key)
    if cached is None:
        words = set()
        for entry in ((semantics or {}).get("fields") or {}).values():
            if not isinstance(entry, dict):
                continue
            names = [entry.get("business_name"), entry.get("display_name")]
            names.extend(entry.get("synonyms") or ())
            for name in names:
                for token in re.findall(r"[a-z0-9]+", str(name or "").lower()):
                    words.add(token)
        cached = frozenset(words)
        _VOCAB_CACHE[key] = cached
    return word in cached


_VOCAB_CACHE: Dict[int, frozenset] = {}


def _is_row_noun(word: str) -> bool:
    try:
        return re.fullmatch(_lexical.row_noun_alternation(), word) is not None
    except Exception:  # noqa: BLE001
        return False


def _is_measure_noun(word: str, semantics: dict) -> bool:
    own = _owners()
    parser = own["parser"]
    if parser is None:
        return False
    try:
        return bool(parser._detect_metric(word, semantics)[2])
    except Exception:  # noqa: BLE001
        return False


_HEAD_TOKEN_CACHE: Dict[int, frozenset] = {}


def _analytical_head_tokens(semantics: dict) -> frozenset:
    """The words a governed DIMENSION's own name ends in.

    THE HEAD OF A GOVERNED NAME, not every word in one. `ltv band`, `product
    type`, `seasoning cohort` and `borrower structure` end in `band`, `type`,
    `cohort` and `structure`, and those are the nouns a reader puts a modifier
    in front of. Taking every word instead made a head out of `months` (from
    "months on book band") and `time`, and a slot in front of a period noun
    turns "over the next twelve months" into two unresolved categories — which
    is the reverse failure, measured on the census, at 23 movements.

    DIMENSIONS ONLY, because the defect is a recognised CATEGORY head absorbing
    its modifier. Measures and rows already anchor slots of their own through
    `is_measure_noun` and the row-noun vocabulary, so `balance`, `loans` and
    `properties` are covered without widening this set — and widening it to
    metric names is what put `pipeline` beside `plus` and `today's`.

    Read from the registry, never written here: the estate adds a dimension and
    this set follows, with no list to maintain and no vocabulary invented.
    """
    key = id(semantics)
    cached = _HEAD_TOKEN_CACHE.get(key)
    if cached is not None:
        return cached
    heads = set()
    for entry in ((semantics or {}).get("fields") or {}).values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("role") or "").lower() != "dimension":
            continue
        names = [entry.get("business_name"), entry.get("display_name")]
        names.extend(entry.get("synonyms") or ())
        for name in names:
            tokens = re.findall(r"[a-z0-9]+", str(name or "").lower())
            if tokens:
                heads.add(tokens[-1])
    cached = frozenset(heads)
    _HEAD_TOKEN_CACHE[key] = cached
    return cached


def _is_analytical_noun(word: str, semantics: dict) -> bool:
    """Does this word NAME a governed analytical concept?

    THE REGISTRY IS THE WHOLE ANSWER, and deliberately: `band`, `bucket`,
    `cohort`, `segment`, `product` and `type` are the nouns a governed dimension
    name ends in, and `sparkle`, `velvet`, `bronze` and `aurora` are not. So the
    head-noun vocabulary is not written here — it is READ from the same registry
    the dimension owner resolves against, which is what stops this from becoming
    a second parser with a list of nouns to maintain.

    NOT "is it claimed". Ordinary framing words are claimed by their owners too,
    and a slot in front of `is`, `show` or `the` is not a restriction position —
    it is noise, and noise here becomes a refusal of a working question. A
    boundary word is by definition not a head, whatever the registry spells.

    The singular is tried because a reader writes `products` where the registry
    spells `product`. That fold only ever CREATES a slot, so it can make this
    layer stricter and never laxer.
    """
    if not word or word in _lexical.RESTRICTION_SLOT_BOUNDARY or word in _BENIGN:
        return False
    if word.isdigit():
        return False
    heads = _analytical_head_tokens(semantics)
    if word in heads:
        return True
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1] in heads
    return False


def _slots(question: str, semantics: dict):
    """THE SHARED GRAMMAR, in both positions a restriction stands in.

    `lexical` is the single owner of where a restriction slot is; the parser
    reads it to RESOLVE what stands in one, and this layer reads it to check
    that something did. A second copy of that boundary is the duplicate-owner
    defect, and it would drift.

    Two positions, tagged, because they differ in one respect that matters. In
    an ATTRIBUTIVE slot the head is the thing being restricted — `loans`,
    `balance`, `band` — and the restriction is the run in front of it. In a
    PREPOSITIONAL object there is no separate thing: "in Atlantis" restricts the
    population to Atlantis, so the head is itself material and must be
    accounted for like any modifier.
    """
    measure = lambda word: _is_measure_noun(word, semantics)          # noqa: E731
    analytical = lambda word: _is_analytical_noun(word, semantics)    # noqa: E731
    out = [(head, slot, offset, "attributive") for head, slot, offset
           in _lexical.restriction_slots(question, measure, analytical)]
    try:
        out += [(head, slot, offset, "prepositional") for head, slot, offset
                in _lexical.prepositional_restriction_objects(
                    question, _is_row_noun)]
    except AttributeError:  # an older lexical module claims only the one position
        pass
    return out


def consumed_spans(question: str, semantics: dict, *,
                   available_columns=None, available_values=None
                   ) -> Tuple[Tuple[int, int], ...]:
    """The character ranges the owners report having claimed, before any slot is
    examined.

    THE OFFSET-PRESERVING MASKS ARE THE ESTATE'S OWN. `mask_statistic_phrases`,
    `mask_scope_phrases` and `mask_segment_phrases` exist precisely so one
    owner's span can be taken out of a sentence without moving any other owner's
    offsets, and `condition_span` reports its range directly. Reading them here
    is what keeps this layer from re-deriving anything: a numeric condition, a
    governed scope and a seasoning phrase are claimed by their owners, not
    re-recognised.
    """
    text = str(question or "").lower()
    spans: List[Tuple[int, int]] = []
    if not text:
        return ()
    own = _owners()
    parser = own["parser"]

    def mask_delta(masked: Optional[str]) -> None:
        """Whatever an offset-preserving mask blanked, its owner claimed."""
        if not masked or len(masked) != len(text):
            return
        start = None
        for i, (before, after) in enumerate(zip(text, masked)):
            if before != after and not before.isspace():
                if start is None:
                    start = i
            elif start is not None:
                spans.append((start, i))
                start = None
        if start is not None:
            spans.append((start, len(text)))

    stat = own["statistic"]
    if stat is not None:
        try:
            mask_delta(stat.mask_statistic_phrases(text))
        except Exception:  # noqa: BLE001
            pass
    try:
        from mi_agent.portfolio_lens import mask_scope_phrases

        mask_delta(mask_scope_phrases(text))
    except Exception:  # noqa: BLE001
        pass
    try:
        from mi_agent.seasoning import mask_segment_phrases

        mask_delta(mask_segment_phrases(text))
    except Exception:  # noqa: BLE001
        pass
    try:
        condition = _lexical.condition_span(text)
        if condition:
            spans.append(condition)
    except Exception:  # noqa: BLE001
        pass
    # THE COUNT REQUEST'S SPAN IS DELIBERATELY NOT CONSUMED. A count phrase
    # includes the modifiers of the thing counted — `count_request_spans` on
    # "how many Scottish loans are there?" returns the whole of "how many
    # scottish loans" — so treating it as claimed throws the POPULATION away.
    # That exact mistake was made once already, in the parser, and it turned a
    # narrowed count into a whole-book one. The count phrase's own words are
    # claimed word-by-word by the framing owner instead.
    try:
        from mi_agent.categorical_spans import value_spans

        spans.extend(value_spans(text, available_values) or ())
    except Exception:  # noqa: BLE001
        pass
    # THE PERIOD-GRAIN OWNER. "monthly" names a time grain, not a category.
    try:
        from mi_agent import period_request

        for match in re.finditer(r"\b[a-z]+\b", text):
            if period_request.requested_unit(match.group(0)):
                spans.append((match.start(), match.end()))
    except Exception:  # noqa: BLE001
        pass
    # THE FORECAST OWNERS: the milestone VALUE and the forward-looking framing,
    # each read by the owner that already acts on it.
    if parser is not None:
        try:
            if parser._forecast_target_value(text) is not None:
                for match in re.finditer(
                        r"£?\d[\d,.]*\s*(?:k|m|bn|billion|million|thousand)?\b",
                        text):
                    spans.append((match.start(), match.end()))
        except Exception:  # noqa: BLE001
            pass
        try:
            if parser._forecast_question_kind(text):
                for match in re.finditer(
                        r"\b(?:forecast|forecasts|forecasting|projected|projection|"
                        r"expected|expect|predicted)\b", text):
                    spans.append((match.start(), match.end()))
        except Exception:  # noqa: BLE001
            pass
        # The measure owner's own report of the phrases it matched.
        try:
            for phrase in parser._detect_metric(text, semantics)[2] or ():
                for match in re.finditer(
                        r"\b" + re.escape(str(phrase).lower()) + r"\b", text):
                    spans.append((match.start(), match.end()))
        except Exception:  # noqa: BLE001
            pass
    return tuple(spans)


#: Memo for `material_residue`, keyed on everything that decides its answer.
#: `_resolve_population` is called several times while one question is parsed —
#: once for the ungrouped reading, again through `_grouped_value_filters` — and
#: the accounting answer cannot differ between those calls. Measured: the layer
#: added 87ms to the median request before this, and 6ms after.
_RESIDUE_MEMO: Dict[Tuple[Any, ...], Tuple[Any, Any, Any, Tuple["Residue", ...]]] = {}
_RESIDUE_MEMO_CAP = 2048


def material_residue(question: str, semantics: dict, *,
                     available_columns=None, available_values=None
                     ) -> Tuple[Residue, ...]:
    """Every span in a restriction position that no owner claims.

    Two passes, in this order and for this reason:

      1. the owners claim their spans over the WHOLE question, so a numeric
         condition, a count request, a governed value or a governed scope is
         gone before any slot is looked at;
      2. what remains in each restriction slot is offered to the owners as a
         PHRASE, longest first, INCLUDING the head noun — because a governed
         name may end in one ("original balance", "borrower age", "loan count")
         and asking about the modifiers alone reports the halves of a name the
         estate knows perfectly well.

    Longest-first within the slot is the same discipline
    `_attributive_categorical` uses for exactly the same reason.
    """
    text = str(question or "").lower()
    if not text.strip():
        return ()
    # Same discipline as `claimed_phrase`: the objects are held and re-checked.
    memo_key = (text, id(semantics), id(available_columns), id(available_values))
    cached = _RESIDUE_MEMO.get(memo_key)
    if cached is not None:
        held_semantics, held_columns, held_values, result = cached
        if (held_semantics is semantics and held_columns is available_columns
                and held_values is available_values):
            return result
    consumed = consumed_spans(question, semantics,
                              available_columns=available_columns,
                              available_values=available_values)
    covered: Set[int] = set()
    for start, end in consumed:
        covered.update(range(start, end))

    out: List[Residue] = []
    seen: Set[str] = set()
    for head, slot, head_offset, kind in _slots(question, semantics):
        words = [(w, o) for w, o in slot if o not in covered]
        if kind == "attributive":
            if not words:
                continue
            # The head is the thing being restricted, not a restriction: it is
            # offered to the owners as part of the phrase and never reported.
            entries = words + [(head, head_offset)]
            head_index = len(words)
            position = "row" if _is_row_noun(head) else "measure"
        else:
            # THE HEAD OF A PREPOSITIONAL OBJECT IS ITSELF THE RESTRICTION.
            # "in Atlantis" names no thing being narrowed — the object IS the
            # narrowing — so excluding the head here would exclude the only
            # material word in it, which is exactly how that question returned
            # the whole book.
            entries = words + ([] if head_offset in covered
                               else [(head, head_offset)])
            if not entries:
                continue
            head_index = -1
            position = "prepositional"
        unclaimed = _unclaimed_in_slot(entries, semantics, available_columns,
                                       available_values, head_index=head_index)
        for word in unclaimed:
            if word in seen:
                continue
            seen.add(word)
            out.append(Residue(word, head, position))
    result = tuple(out)
    if len(_RESIDUE_MEMO) >= _RESIDUE_MEMO_CAP:
        _RESIDUE_MEMO.clear()
    _RESIDUE_MEMO[memo_key] = (semantics, available_columns, available_values,
                               result)
    return result


def _unclaimed_in_slot(words: Sequence[Tuple[str, int]], semantics: dict,
                       available_columns, available_values,
                       head_index: int) -> List[str]:
    """The words of one slot that survive every owner, longest phrase first.

    ``head_index`` marks the head noun, which is offered to the owners as part
    of the phrase but is never itself reported as residue: it is the thing being
    restricted, not a restriction.
    """
    tokens = [w for w, _o in words]
    claimed = [False] * len(tokens)
    for length in range(min(_MAX_PHRASE, len(tokens)), 0, -1):
        for start in range(0, len(tokens) - length + 1):
            if any(claimed[start:start + length]):
                continue
            phrase = " ".join(tokens[start:start + length])
            if claimed_phrase(phrase, semantics,
                              available_columns=available_columns,
                              available_values=available_values):
                for i in range(start, start + length):
                    claimed[i] = True
    return [w for i, (w, done) in enumerate(zip(tokens, claimed))
            if not done and i != head_index]
