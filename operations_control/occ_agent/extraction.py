"""operations_control.occ_agent.extraction — reading a sentence against the catalogue.

The agent must be able to consume every field its own onboarding pack asks for.
Maintaining a hand-written pattern per field would guarantee it eventually
cannot: a field added to ``config/onboarding/field_catalogue.yaml`` would be
asked for and then not understood, which is exactly the defect this replaces.

So the patterns are **derived from the catalogue**. For every collected field,
two things are built:

* **cues** — the phrases an operator would use to name it, taken from the
  field's own ``label`` and ``key``, plus the acronym its label implies
  ("Legal Entity Identifier" → "LEI"). A short prose map adds only the words a
  human types that no declaration contains ("their email", "book").
* **a value pattern** — taken from the field's declared ``validation`` or
  ``type``. An ``lei`` field accepts an ISO 17442 string and nothing else; an
  ``enum`` accepts its own declared options; a ``date`` accepts the three date
  shapes an operator writes.

Binding is cue *and* value together, which is what disambiguates a shared cue:
"the reporting contact is dana@example.com" and "the reporting contact is Dana
Fox" name the same thing, and the value decides which field is meant.

The text is read clause by clause, and a clause from which nothing could be
read is reported as **unrecognised** rather than dropped. That is what lets the
service refuse to apply a half-understood instruction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from ..onboarding.catalogue import Catalogue, Field, Section, catalogue

#: Confidence for a value bound by an explicit cue, versus one recognised by
#: shape alone (an LEI in a sentence that never says "LEI").
CUED = 1.0
SHAPE_ONLY = 0.6


@dataclass(frozen=True)
class FieldRef:
    """One catalogue field, flattened for matching."""

    section: str
    section_label: str
    key: str
    label: str
    type: str
    validation: str
    options: Tuple[str, ...]
    option_labels: Tuple[str, ...]
    repeatable: bool
    product: str = ""

    @property
    def path(self) -> str:
        return f"{self.section}.{self.key}"


@dataclass
class Extracted:
    ref: FieldRef
    value: Any
    confidence: float = CUED
    cue: str = ""
    span: Tuple[int, int] = (0, 0)


@dataclass
class Reading:
    """What one piece of text yielded."""

    found: List[Extracted] = field(default_factory=list)
    #: Clauses nothing could be read from, verbatim, for disclosure.
    unrecognised: List[str] = field(default_factory=list)
    #: The delivery period. The catalogue deliberately does NOT hold this —
    #: ``not_collected`` calls it "supplied with each delivery" — so it is read
    #: here and lives on the run.
    reporting_period: str = ""


# --------------------------------------------------------------------------- #
# Value patterns, by declared validation / type
# --------------------------------------------------------------------------- #

_LEI = r"[A-Z0-9]{18}[0-9]{2}"
_EMAIL = r"[^@\s,;]+@[^@\s,;]+\.[A-Za-z]{2,}"
_CCY = r"[A-Z]{3}"
_IDENT = r"[A-Za-z0-9][A-Za-z0-9_.\-]{1,63}"
_ISO_DATE = r"\d{4}-\d{2}-\d{2}"
_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")
_LONG_DATE = (r"\d{1,2}(?:st|nd|rd|th)?\s+(?:" + "|".join(_MONTHS) +
              r")\s+\d{4}")
_MONTH_YEAR = r"(?:" + "|".join(_MONTHS) + r")\s+\d{4}"
_DATE = f"(?:{_ISO_DATE}|{_LONG_DATE}|{_MONTH_YEAR})"
#: Six digits before three: alternation is first-match, and "#112233"
#: read as "#112" is worse than not reading it at all.
_COLOUR = r"#(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})"
_NAME_RUN = r"[A-Z][\w&'’-]*(?:\s+[A-Z][\w&'’-]*){0,4}"
_COUNTRY_CODE = r"[A-Z]{2}"

#: Patterns that match almost any word. Fine when the operator named the field
#: outright; never used to guess a neighbouring one.
_TOO_LOOSE_FOR_SIBLINGS = frozenset({_IDENT})

#: Country names an operator writes, to the two-letter code the catalogue wants.
#: Names only — a bare two-letter token in free text is too ambiguous to take
#: without a cue.
COUNTRY_NAMES: Dict[str, str] = {
    "united kingdom": "GB", "uk": "GB", "britain": "GB", "british": "GB",
    "england": "GB", "scotland": "GB", "wales": "GB",
    "northern ireland": "GB",
    "ireland": "IE", "irish": "IE", "republic of ireland": "IE",
    "netherlands": "NL", "dutch": "NL", "holland": "NL",
    "spain": "ES", "spanish": "ES", "france": "FR", "french": "FR",
    "germany": "DE", "german": "DE", "italy": "IT", "italian": "IT",
    "portugal": "PT", "belgium": "BE", "luxembourg": "LU",
    "united states": "US", "usa": "US", "us": "US",
}

CURRENCY_NAMES: Dict[str, str] = {
    "sterling": "GBP", "pounds": "GBP", "pound": "GBP", "gbp": "GBP",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "dollar": "USD", "dollars": "USD", "usd": "USD",
}

#: The words an operator types for a product. The PRODUCTS come from the regime
#: declaration the catalogue reads; only the phrasing is here.
PRODUCT_PROSE: Dict[str, Tuple[str, ...]] = {
    "mi": ("management information", "portfolio mi", "monthly mi", "mi pack",
           "mi reporting", "portfolio reporting", "mi"),
    "esma_annex2": ("annex 2", "annex2", "esma annex 2",
                    "regulatory reporting", "securitisation reporting",
                    "underlying exposures"),
    "investor_reporting": ("investor reporting", "investor report",
                           "annex 12", "annex12", "noteholder reporting"),
    "static_pools": ("static pool", "static pools", "vintage analysis"),
}

#: Words an operator types that no declaration contains. Keyed by catalogue
#: path, so this stays a *phrasing* map and never a second field list.
PROSE_CUES: Dict[str, Tuple[str, ...]] = {
    "client.client_name": ("client", "lender", "they are called", "called"),
    "client.client_id": ("client code", "client ref", "client reference"),
    "client.jurisdiction": ("jurisdiction", "based in", "domiciled",
                            "country"),
    "client.reporting_currency": ("currency", "report in", "reporting in"),
    "entities.legal_name": ("legal entity", "entity"),
    "entities.lei": ("lei",),
    "entities.country_of_establishment": ("established in",
                                          "incorporated in"),
    "contacts.reporting_contact_name": ("reporting contact",),
    "contacts.reporting_contact_email": ("reporting contact email",
                                         "reporting email",
                                         "report to", "reports go to"),
    "contacts.operational_contact_name": ("operational contact",
                                          "ops contact"),
    "contacts.operational_contact_email": ("operational contact email",
                                           "ops email"),
    "contacts.authorised_approver": ("approver", "approves"),
    "contacts.investor_report_recipients": ("investor report recipients",
                                            "investor reports go to"),
    "portfolios.portfolio_id": ("portfolio id", "portfolio identifier",
                                "portfolio ref", "portfolio reference",
                                "book id", "book reference"),
    "portfolios.display_name": ("portfolio called", "portfolio named",
                                "book called"),
    "portfolios.asset_class": ("asset class", "asset type", "book of",
                               "portfolio of"),
    "portfolios.portfolio_type": ("acquired", "originated", "bought",
                                  "book was"),
    "portfolios.structure": ("held in", "structure", "warehouse", "spv"),
    "sources.cadence": ("cadence", "frequency", "how often", "deliver",
                        "delivered", "send", "sends"),
    "sources.delivery_channel": ("delivery", "arrive", "arrives", "send via",
                                 "deliver by"),
    "sources.file_format": ("file format", "files are", "format"),
    "sources.source_party": ("sent by", "provided by", "supplier"),
    "sources.expected_files": ("file names", "filenames", "files called"),
    "reporting.products": ("products", "require", "requires", "need",
                           "needs", "reporting"),
    "presentation.report_title": ("report title", "report called"),
    "presentation.brand_colour": ("brand colour", "brand color", "colour",
                                  "color"),
    "presentation.disclaimer": ("disclaimer",),
}


def _value_pattern(f: Field) -> Optional[str]:
    """The shape a value for this field takes, from its own declaration."""
    rule = f.validation or f.type
    return {
        "lei": _LEI,
        "email": _EMAIL,
        "currency": _CCY,
        "country": _COUNTRY_CODE,
        "identifier": _IDENT,
        "colour": _COLOUR,
        "date": _DATE,
        "date_or_nd": f"(?:{_DATE}|ND[A-Z]*)",
        "yes_no": r"(?:yes|no|y|n)",
        "boolean": r"(?:yes|no|true|false)",
    }.get(rule)


def acronym(label: str) -> str:
    """"Legal Entity Identifier" -> "LEI". Derived, never listed."""
    words = [w for w in re.split(r"[^A-Za-z]+", label) if w]
    if len(words) < 3:
        return ""
    letters = "".join(w[0] for w in words if w[0].isupper() or len(w) > 3)
    return letters.upper() if len(letters) >= 3 else ""


def cues_for(section: Section, f: Field) -> List[str]:
    """Every phrase that names this field, longest first."""
    out = {section.label.lower() + " " + f.label.lower(), f.label.lower(),
           f.key.replace("_", " ").lower()}
    a = acronym(f.label)
    if a:
        out.add(a.lower())
    out.update(PROSE_CUES.get(f"{section.key}.{f.key}", ()))
    return sorted({c for c in out if len(c) >= 2}, key=len, reverse=True)


# --------------------------------------------------------------------------- #
# The catalogue, flattened
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Candidate:
    ref: FieldRef
    cue: str
    pattern: Optional[str]
    #: Fields in the same section whose key shares this one's first token —
    #: ``reporting_contact_name`` and ``reporting_contact_email`` are the same
    #: question asked two ways. A cue that names the group binds to whichever
    #: sibling the VALUE fits.
    siblings: Tuple[Tuple[FieldRef, str], ...] = ()

    @property
    def typed(self) -> bool:
        """Whether the field constrains its value's shape.

        A typed field is tried before a free-text one sharing the same cue:
        "the reporting contact is dana@example.com" names the email, and only
        the email field's pattern can prove it.
        """
        return bool(self.pattern) or bool(self.ref.options)


def _group(key: str) -> str:
    """The question a field belongs to: everything up to its last token."""
    parts = key.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else key


@lru_cache(maxsize=4)
def _candidates(version: int) -> Tuple[Candidate, ...]:
    cat = catalogue()
    refs: Dict[str, Tuple[FieldRef, Section, Field]] = {}
    pairs: List[Tuple[FieldRef, str, Optional[str]]] = []
    for section in cat.sections:
        for f in section.fields:
            if not f.collected:
                continue                    # derived / system: never asked
            options = [(str(o.get("value")), str(o.get("label")
                                                  or o.get("value")))
                       for o in (f.options or [])]
            if f.type == "product_selection":
                # The product list is the regime declaration's, read through
                # the catalogue rather than restated here.
                options = [(key, str((spec or {}).get("label") or key))
                           for key, spec in (cat.regime_products or {}).items()]
            ref = FieldRef(
                section=section.key, section_label=section.label, key=f.key,
                label=f.label, type=f.type, validation=f.validation,
                options=tuple(v for v, _l in options),
                option_labels=tuple(l for _v, l in options),
                repeatable=section.repeatable, product=f.product)
            refs[ref.path] = (ref, section, f)
            for cue in cues_for(section, f):
                pairs.append((ref, cue, _value_pattern(f)))
    by_group: Dict[Tuple[str, str], List[Tuple[FieldRef, str]]] = {}
    for ref, _section, f in refs.values():
        pattern = _value_pattern(f)
        if pattern:
            by_group.setdefault((ref.section, _group(ref.key)), []).append(
                (ref, pattern))

    out = [Candidate(ref=ref, cue=cue, pattern=pattern,
                     siblings=tuple(
                         s for s in by_group.get(
                             (ref.section, _group(ref.key)), [])
                         if s[0].path != ref.path))
           for ref, cue, pattern in pairs]
    # Longest cue first, so "reporting contact email" beats "reporting
    # contact"; then typed before free text, so a shape-constrained field
    # claims a value a text field would otherwise swallow whole.
    return tuple(sorted(out, key=lambda c: (-len(c.cue), not c.typed,
                                            c.ref.path)))


def candidates(cat: Optional[Catalogue] = None) -> Tuple[Candidate, ...]:
    cat = cat or catalogue()
    return _candidates(cat.version)


def collected_fields(cat: Optional[Catalogue] = None) -> List[FieldRef]:
    """Every field the catalogue says is collected, in ITS declaration order.

    The order matters wherever a value could belong to more than one field:
    "monthly" is both a delivery cadence and a payment frequency, and the one
    that wins is the one the catalogue declares first. Deriving that from the
    catalogue rather than from cue length means the answer is a governed
    decision — reorder the catalogue and the precedence follows — instead of an
    artefact of how the candidate list happens to sort.
    """
    cat = cat or catalogue()
    order = {f"{s.key}.{f.key}": i
             for i, (s, f) in enumerate((s, f) for s in cat.sections
                                        for f in s.fields)}
    seen: Dict[str, FieldRef] = {}
    for c in candidates(cat):
        seen.setdefault(c.ref.path, c.ref)
    return sorted(seen.values(),
                  key=lambda r: order.get(r.path, len(order)))


def reset_cache() -> None:
    _candidates.cache_clear()
    _ambiguous_bare.cache_clear()
    _joined_cues.cache_clear()
    asset_signal_tokens.cache_clear()


def match_asset(text: str, cat: Optional[Catalogue] = None
                ) -> Tuple[str, float]:
    """The asset class a phrase names, and how confidently. ``("", 0.0)`` if none.

    A thin read over :func:`read`, kept as its own entry point because
    recognising an asset from prose is asked for on its own — by the tests, and
    by anything that wants to know what a sentence is about before deciding
    what to do with it.
    """
    cat = cat or catalogue()
    for found in read(text, cat).found:
        if found.ref.key == "asset_class":
            return str(found.value), float(found.confidence)
    return "", 0.0


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

#: A clause. Sentence-ish: full stops, semicolons and newlines end one — but
#: the dot inside ``dana@northstar.example`` is not a sentence boundary, so
#: addresses are masked before splitting and restored afterwards.
_CLAUSE = re.compile(r"[^.;\n]+")
_DOT = "\u0001"
#: Stands in for a space inside a phrase that must not be split.
_SPACE = "\u0002"

#: A conjunction also ends a clause, because an operator writes two statements
#: in one sentence far more often than one long statement. Without this, "the
#: LEI is X and sort out the other thing" reads the LEI and drops the rest
#: **silently** — which is exactly the failure this must not have.
_CONJUNCTION = re.compile(r"\s+(?:and|but|then|also|plus)\s+", re.I)

#: A fragment shorter than this is not a statement; it is the tail of the one
#: before it, and is re-joined rather than reported as missed.
_MIN_FRAGMENT_WORDS = 3

#: Clauses that carry no information and are not worth reporting as missed.
_FILLER = re.compile(
    r"^\s*(please|thanks|thank you|ok|okay|yes|no|and|also|as well|"
    r"that'?s it|nothing else)\s*$", re.I)


@lru_cache(maxsize=4)
def _joined_cues(version: int) -> Tuple[str, ...]:
    """Cues that contain a conjunction, so a split cannot cut one in half.

    "Units and currency" is one question, not two clauses. Which phrases those
    are is the catalogue's to say, so they are read from it.
    """
    cat = catalogue()
    out: set = set()
    for section in cat.sections:
        for f in section.fields:
            if not f.collected:
                continue
            for cue in cues_for(section, f):
                if _CONJUNCTION.search(cue):
                    out.add(cue.lower())
    return tuple(sorted(out, key=len, reverse=True))


def _fragments(clause: str, cat: Optional[Catalogue] = None) -> List[str]:
    """One clause, split at conjunctions that join two real statements."""
    if not _CONJUNCTION.search(clause):
        return [clause]
    # Protect any conjunction that sits inside a catalogue cue before splitting.
    protected = clause
    cat = cat or catalogue()
    for cue in _joined_cues(cat.version):
        protected = re.sub(re.escape(cue),
                           lambda m: m.group(0).replace(" ", _SPACE),
                           protected, flags=re.I)
    parts = [p.replace(_SPACE, " ") for p in _CONJUNCTION.split(protected)]
    if len(parts) < 2:
        return [clause]
    out: List[str] = []
    for part in parts:
        if out and len(part.split()) < _MIN_FRAGMENT_WORDS:
            out[-1] = f"{out[-1]} and {part}"       # a tail, not a statement
        else:
            out.append(part)
    return out


def read(text: str, cat: Optional[Catalogue] = None) -> Reading:
    """Read every catalogue field the text names. Clause by clause."""
    cat = cat or catalogue()
    reading = Reading()
    masked = re.sub(_EMAIL, lambda m: m.group(0).replace(".", _DOT),
                    str(text or ""))
    for clause in (f for whole in _CLAUSE.findall(masked)
                   for f in _fragments(whole, cat)):
        stripped = clause.replace(_DOT, ".").strip()
        if len(stripped.split()) < 2 or _FILLER.match(stripped):
            continue
        found = _read_clause(stripped, cat)
        period = _run_fact_date(stripped)
        if period and not reading.reporting_period:
            reading.reporting_period = period
        if found:
            reading.found.extend(found)
        elif not period:
            reading.unrecognised.append(stripped)
    return reading


def _read_clause(clause: str, cat: Catalogue) -> List[Extracted]:
    lower = clause.lower()
    out: List[Extracted] = []
    taken: List[Tuple[int, int]] = []      # consumed spans, so one value binds once
    claimed: set = set()                   # one value per field path per clause

    for candidate in candidates(cat):
        if candidate.ref.path in claimed:
            continue
        for cue_start in _cue_positions(lower, candidate.cue):
            after = clause[cue_start + len(candidate.cue):]
            hit = _value_after(after, candidate)
            if hit is None:
                continue
            ref, value, (rel_start, rel_end) = hit
            if ref.path in claimed:
                continue
            start = cue_start + len(candidate.cue) + rel_start
            end = cue_start + len(candidate.cue) + rel_end
            if _overlaps(taken, (start, end)):
                continue
            taken.append((start, end))
            claimed.add(ref.path)
            out.append(Extracted(ref=ref, value=value, confidence=CUED,
                                 cue=candidate.cue, span=(start, end)))
            break

    # An option is its own name. "It is a UK equity-release lender" names the
    # asset class without ever saying "asset class", and an operator writing an
    # opening instruction always writes like that.
    out.extend(_by_option(clause, cat, taken, claimed))

    # A place or a currency is its own name too. "UK equity release" states the
    # jurisdiction without the word, and an opening instruction is written that
    # way far more often than not.
    out.extend(_by_name(clause, cat, taken, claimed))

    # Values whose shape is unmistakable and which nothing has claimed — an LEI
    # or an email in a sentence that never names the field. Taken at reduced
    # confidence, so the human confirms.
    out.extend(_by_shape(clause, cat, taken, claimed))
    return out


#: An option token shorter than this, or in this list, is too ordinary to
#: recognise on its own. "no" is an option of several yes/no fields.
_MIN_BARE_OPTION = 3
_NEVER_BARE = frozenset({"yes", "no", "y", "n", "true", "false", "other",
                         "othr", "ncom", "text", "mixd"})


@lru_cache(maxsize=1)
def asset_signal_tokens() -> Dict[str, Tuple[str, ...]]:
    """Each asset class's own vocabulary, from the product profiles.

    "a lifetime mortgage book" names equity release, and the platform already
    says so: ``config/asset/product_profiles.yaml`` lists the signal tokens the
    Onboarding Agent's own profile resolver scores against. They are read from
    there rather than restated, so an asset added to the profiles is understood
    in conversation without a change here.
    """
    try:
        from engine.onboarding_agent.product_profile import load_product_profiles
    except Exception:                       # pragma: no cover — optional import
        return {}
    out: Dict[str, Tuple[str, ...]] = {}
    for profile in (load_product_profiles() or {}).values():
        match = (getattr(profile, "spec", {}) or {}).get("match", {}) or {}
        tokens = tuple(str(t).strip().lower()
                       for t in (match.get("signal_tokens") or []) if t)
        for asset in (match.get("asset_class") or []):
            key = str(asset).strip().lower()
            if key and tokens:
                out[key] = tuple(sorted(set(out.get(key, ()) + tokens)))
    return out


def _bare_tokens(ref: FieldRef,
                 minimum: int = _MIN_BARE_OPTION) -> List[Tuple[str, str]]:
    """``(token, value)`` for options distinctive enough to stand alone."""
    out: List[Tuple[str, str]] = []
    signals = asset_signal_tokens() if ref.key == "asset_class" else {}
    for value, label in zip(ref.options, ref.option_labels):
        tokens = {value, label}
        if ref.type == "product_selection":
            tokens |= set(PRODUCT_PROSE.get(value, ()))
        tokens |= set(signals.get(value.strip().lower(), ()))
        for token in tokens:
            t = token.strip().lower()
            if len(t) >= minimum and t not in _NEVER_BARE:
                out.append((t, value))
    return sorted(set(out), key=lambda p: -len(p[0]))


@lru_cache(maxsize=4)
def _ambiguous_bare(version: int) -> frozenset:
    """Option tokens that are also part of some field's NAME.

    "email" is a delivery channel and also a word in four field labels. Read
    bare it is a coin toss, so it is only ever accepted after a cue.
    """
    cat = catalogue()
    words: set = set()
    for ref in collected_fields(cat):
        for source in (ref.label, ref.key.replace("_", " ")):
            words.update(w for w in re.split(r"[^a-z0-9]+", source.lower())
                         if w)
    ambiguous: set = set()
    for ref in collected_fields(cat):
        # A token that appears in the field's OWN name is not ambiguous for
        # that field — "acquired" is a value of "How the book was acquired",
        # and a question whose own wording contains its answer is the clearest
        # case there is, not the murkiest.
        own = set(re.split(r"[^a-z0-9]+",
                           f"{ref.key} {ref.label}".lower())) - {""}
        for token, _value in _bare_tokens(ref, minimum=1):
            if token in words - own:
                ambiguous.add(token)
    return frozenset(ambiguous)


def _by_option(clause: str, cat: Catalogue, taken: List[Tuple[int, int]],
               claimed: set) -> List[Extracted]:
    out: List[Extracted] = []
    for ref in collected_fields(cat):
        if not ref.options or ref.path in claimed:
            continue
        picked: List[str] = []
        spans: List[Tuple[int, int]] = []
        loose = _loose(clause)
        ambiguous = _ambiguous_bare(cat.version)
        for token, value in _bare_tokens(ref):
            if value in picked or token in ambiguous:
                continue
            m = re.search(rf"(?<![a-z0-9]){re.escape(_loose(token))}"
                          rf"(?![a-z0-9])", loose)
            if not m or _overlaps(taken, (m.start(), m.end())):
                continue
            picked.append(value)
            spans.append((m.start(), m.end()))
            if ref.type not in MULTI_VALUED:
                break
        if not picked:
            continue
        taken.extend(spans)
        claimed.add(ref.path)
        value: Any = picked if ref.type in MULTI_VALUED else picked[0]
        out.append(Extracted(ref=ref, value=value, confidence=CUED,
                             cue="", span=spans[0]))
    return out


def _cue_positions(lower: str, cue: str) -> List[int]:
    return [m.start() for m in
            re.finditer(rf"(?<![a-z0-9]){re.escape(cue)}(?![a-z0-9])", lower)]


#: What may sit between a cue and its value.
_JOIN = r"\s*(?:is|are|=|:|to|of|will be|should be|for|as)?\s*[\"'“]?"

#: Where a free-text value ends. A clause often carries a second answer after a
#: conjunction, and swallowing it would lose that answer and corrupt this one.
_TEXT_TAIL = re.compile(r"\s+(?:and|with|plus|but|who|whose|which)\s+", re.I)


def _value_after(text: str, candidate: Candidate
                 ) -> Optional[Tuple[FieldRef, Any, Tuple[int, int]]]:
    """The value following a cue, and the field it actually belongs to."""
    ref = candidate.ref

    if ref.options:
        hit = _match_option(text, ref)
        if hit is not None:
            return (ref,) + hit

    if ref.type == "product_selection":
        return None                        # products are matched by the caller

    rule = ref.validation or ref.type
    if rule in ("country", "currency"):
        # The written name first. A bare two-letter code is matched only
        # case-sensitively, or "established in Ireland" yields "IR".
        named = _match_named(text, COUNTRY_NAMES if rule == "country"
                             else CURRENCY_NAMES)
        if named is not None:
            return (ref,) + named
        m = re.match(_JOIN + f"({candidate.pattern})", text)
        return ((ref, _coerce(ref, m.group(1)), (m.start(1), m.end(1)))
                if m else None)

    if candidate.pattern:
        m = re.match(_JOIN + f"({candidate.pattern})", text, re.I)
        return ((ref, _coerce(ref, m.group(1)), (m.start(1), m.end(1)))
                if m else None)

    if ref.type in ("text", "multiline", "entity_reference"):
        m = re.match(_JOIN + r"([^,.;]{2,200})", text)
        if not m:
            return None
        value = _TEXT_TAIL.split(m.group(1))[0].strip().strip("\"'“”")
        if not value or value.lower() in ("is", "are"):
            return None
        # An address is never a person's name. A value carrying the marks of a
        # typed field belongs to that field, and a text field must not eat it.
        if "@" in value or any(re.fullmatch(pattern, value)
                               for _rule, pattern in _SHAPES):
            return _sibling_value(text, candidate)
        return ref, value, (m.start(1), m.start(1) + len(value))
    return _sibling_value(text, candidate)


def _sibling_value(text: str, candidate: Candidate
                   ) -> Optional[Tuple[FieldRef, Any, Tuple[int, int]]]:
    """The same question answered by a neighbouring field.

    Reached only when the cued field cannot take the value: "the reporting
    contact is dana@example.com" answers the email, not the name. A pattern
    loose enough to match any word is never used this way — it would turn every
    near miss into a wrong answer.
    """
    for sibling, pattern in candidate.siblings:
        if pattern in _TOO_LOOSE_FOR_SIBLINGS:
            continue
        m = re.match(_JOIN + f"({pattern})", text, re.I)
        if m:
            return sibling, _coerce(sibling, m.group(1)), (m.start(1),
                                                           m.end(1))
    return None


#: Field types that hold several answers at once.
MULTI_VALUED = ("multi_enum", "product_selection")


def _match_option(text: str, ref: FieldRef
                  ) -> Optional[Tuple[Any, Tuple[int, int]]]:
    """The field's OWN declared options, by value, label or phrasing.

    A multi-valued field collects every option the text names, not the best
    one: "portfolio MI and investor reporting" is two products, and taking
    only the longest match would silently drop the other.
    """
    head = _loose(text[:160])
    picked: List[str] = []
    spans: List[Tuple[int, int]] = []
    for token, value in _bare_tokens(ref, minimum=2):
        if value in picked:
            continue
        m = re.search(rf"(?<![a-z0-9]){re.escape(_loose(token))}(?![a-z0-9])",
                      head)
        if not m:
            continue
        picked.append(value)
        spans.append((m.start(), m.end()))
        if ref.type not in MULTI_VALUED:
            break
    if not picked:
        return None
    span = (min(s for s, _e in spans), max(e for _s, e in spans))
    return (picked if ref.type in MULTI_VALUED else picked[0]), span


def _loose(text: str) -> str:
    """Lower-cased with separators flattened, so "equity-release",
    "equity_release" and "equity release" are one phrase."""
    return re.sub(r"[\s_\-]+", " ", str(text or "").lower())


def _match_named(text: str, names: Dict[str, str]
                 ) -> Optional[Tuple[Any, Tuple[int, int]]]:
    head = text[:80]
    best: Optional[Tuple[str, Tuple[int, int], int]] = None
    for name, code in names.items():
        m = re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", head, re.I)
        if m and (best is None or len(name) > best[2]):
            best = (code, (m.start(), m.end()), len(name))
    return (best[0], best[1]) if best else None


def _coerce(ref: FieldRef, raw: str) -> Any:
    rule = ref.validation or ref.type
    text = raw.strip()
    if rule in ("lei", "currency", "country"):
        return text.upper()
    if rule in ("date", "date_or_nd"):
        return normalise_date(text)
    if rule == "yes_no":
        return "Y" if text.lower().startswith("y") else "N"
    if rule == "boolean":
        return text.lower() in ("yes", "true", "y")
    return text


def normalise_date(text: str) -> str:
    """The three shapes an operator writes, to the ISO one the catalogue wants."""
    text = text.strip()
    if re.fullmatch(_ISO_DATE, text):
        return text
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})",
                     text)
    if m and m.group(2).lower() in _MONTHS:
        month = _MONTHS.index(m.group(2).lower()) + 1
        return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(1)):02d}"
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", text)
    if m and m.group(1).lower() in _MONTHS:
        month = _MONTHS.index(m.group(1).lower()) + 1
        last = [31, 29 if int(m.group(2)) % 4 == 0 else 28, 31, 30, 31, 30,
                31, 31, 30, 31, 30, 31][month - 1]
        return f"{int(m.group(2)):04d}-{month:02d}-{last:02d}"
    return text.upper() if text.upper().startswith("ND") else text


#: The reporting period is a property of a delivery, not of a client, so the
#: catalogue does not declare it. It is still something an operator says in the
#: opening instruction, so it is read here and carried on the run.
_RUN_DATE_CUES = ("latest reporting date", "first reporting date",
                  "reporting date", "reporting period", "data cut-off",
                  "data cut off", "cut-off date", "as at", "as of")


def _run_fact_date(text: str) -> str:
    for cue in _RUN_DATE_CUES:
        m = re.search(rf"(?<![a-z]){re.escape(cue)}(?![a-z])" + _JOIN +
                      f"({_DATE})", text, re.I)
        if m:
            return normalise_date(m.group(1))
    return ""


#: Value shapes distinctive enough to recognise with no cue at all, and the
#: field type each belongs to. Deliberately short: a false positive here costs
#: an operator a confirmation they did not expect.
_SHAPES: Tuple[Tuple[str, str], ...] = (("lei", _LEI), ("email", _EMAIL))


def _by_shape(clause: str, cat: Catalogue, taken: List[Tuple[int, int]],
              claimed: set) -> List[Extracted]:
    out: List[Extracted] = []
    for rule, pattern in _SHAPES:
        for m in re.finditer(pattern, clause):
            if _overlaps(taken, (m.start(), m.end())):
                continue
            ref = _sole_field_of(cat, rule, claimed)
            if ref is None:
                continue                   # ambiguous or already answered
            taken.append((m.start(), m.end()))
            claimed.add(ref.path)
            out.append(Extracted(ref=ref, value=_coerce(ref, m.group(0)),
                                 confidence=SHAPE_ONLY, cue="",
                                 span=(m.start(), m.end())))
    return out


def _by_name(clause: str, cat: Catalogue, taken: List[Tuple[int, int]],
             claimed: set) -> List[Extracted]:
    """A country or currency written by name, with no cue in front of it.

    Which field it answers is the catalogue's decision, not this module's: the
    first collected field with that validation rule, in declaration order. So
    "UK" answers the client's jurisdiction and not an entity's country of
    establishment — because the catalogue asks the first question first — and an
    entity-level statement says "established in", which the cued pass has
    already taken by the time this runs.

    Read at reduced confidence, so it is proposed rather than assumed.
    """
    out: List[Extracted] = []
    for rule, names in (("country", COUNTRY_NAMES), ("currency",
                                                     CURRENCY_NAMES)):
        ref = next((r for r in collected_fields(cat)
                    if (r.validation or r.type) == rule
                    and r.path not in claimed), None)
        if ref is None:
            continue
        for name, code in sorted(names.items(), key=lambda p: -len(p[0])):
            m = re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", clause,
                          re.I)
            if not m or _overlaps(taken, (m.start(), m.end())):
                continue
            taken.append((m.start(), m.end()))
            claimed.add(ref.path)
            out.append(Extracted(ref=ref, value=code, confidence=SHAPE_ONLY,
                                 cue="", span=(m.start(), m.end())))
            break
    return out


def _sole_field_of(cat: Catalogue, rule: str, claimed: set) -> Optional[FieldRef]:
    """The one collected field with this validation, if there is exactly one.

    More than one and the shape alone cannot say which was meant, so nothing is
    taken and the clause stands or falls on its cues.
    """
    hits = [r for r in collected_fields(cat)
            if (r.validation or r.type) == rule and r.path not in claimed]
    return hits[0] if len(hits) == 1 else None


def _overlaps(taken: List[Tuple[int, int]], span: Tuple[int, int]) -> bool:
    return any(not (span[1] <= s or span[0] >= e) for s, e in taken)
