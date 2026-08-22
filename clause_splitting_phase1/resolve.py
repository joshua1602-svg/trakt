#!/usr/bin/env python3
"""clause_splitting_phase1/resolve.py

Concept resolution — the business-semantics stand-in.

It runs strictly INSIDE a span the splitter produced, and it is the ONLY place
in this directory that reads the registry. The separation is the whole point:
the splitter decides where a clause starts and stops knowing nothing about
fields, and this pass decides what the words in that clause mean knowing
nothing about sentence structure.

It resolves a span's text to registry keys by longest synonym match. Where the
text names nothing the registry knows, the span is reported UNRESOLVABLE rather
than left empty — that distinction is what turns "I did not follow part of
that" into a question back to the user instead of a confident narrower answer.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .span_model import FILLED, Span, Spans, filled, unresolvable

_PHRASES: Optional[List[Tuple[str, str, str]]] = None   # (phrase, key, role)


def _index(semantics: dict) -> List[Tuple[str, str, str]]:
    global _PHRASES
    if _PHRASES is not None:
        return _PHRASES
    out: List[Tuple[str, str, str]] = []
    for key, meta in (semantics.get("fields") or {}).items():
        role = meta.get("role") or ""
        names = [key.replace("_", " ")]
        for n in (meta.get("business_name"), meta.get("display_name")):
            if n:
                names.append(str(n))
        names += [str(s) for s in (meta.get("synonyms") or []) if s]
        for n in names:
            n = n.strip().lower()
            if n:
                out.append((n, key, role))
    out.sort(key=lambda t: -len(t[0]))
    _PHRASES = out
    return out


def _match(text: str, semantics: dict, roles: Tuple[str, ...]) -> List[str]:
    """Registry keys named in `text`, longest phrase first, no overlaps."""
    if not text:
        return []
    hay = " %s " % re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    hay = re.sub(r"\s+", " ", hay)
    taken: List[Tuple[int, int]] = []
    found: List[str] = []
    for phrase, key, role in _index(semantics):
        if role not in roles:
            continue
        for m in re.finditer(r"(?<= )%s(?= )" % re.escape(phrase), hay):
            if any(not (m.end() <= a or m.start() >= b) for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            if key not in found:
                found.append(key)
    return found


_COUNT_NOUNS = re.compile(r"\b(loans?|cases?|accounts?|borrowers?|deals?|"
                          r"facilit(?:y|ies)|records?)\b", re.I)


def resolve(split, semantics: dict) -> Spans:
    """Turn a text split into concept-bearing spans."""
    s = Spans()

    if split.operation:
        s.operation = filled([split.operation], text=split.operation_text)

    # -- grouping -------------------------------------------------------
    g_concepts: List[str] = []
    g_unres: List[str] = []
    for text, grain in zip(split.grouping_texts, split.grouping_grains):
        if grain is not None:
            g_concepts.append("time:%s" % grain)
            continue
        keys = _match(text, semantics, ("dimension",))
        if keys:
            g_concepts.extend(keys)
        else:
            g_unres.append(text)
    if g_unres and not g_concepts:
        s.grouping = unresolvable("; ".join(g_unres), text="; ".join(split.grouping_texts))
    elif g_unres:
        s.grouping = unresolvable("; ".join(g_unres), text="; ".join(split.grouping_texts))
        s.grouping.concepts = sorted(set(g_concepts))
    else:
        s.grouping = filled(sorted(set(g_concepts)))

    # -- filter ---------------------------------------------------------
    f_concepts: List[str] = []
    f_unres: List[str] = []
    for text in split.filter_texts:
        keys = _match(text, semantics, ("metric", "dimension", "date", "flag"))
        if keys:
            f_concepts.extend(keys)
        else:
            f_unres.append(text)
    if f_unres:
        s.filter = unresolvable("; ".join(f_unres), text="; ".join(split.filter_texts))
        s.filter.concepts = sorted(set(f_concepts))
    else:
        s.filter = filled(sorted(set(f_concepts)))

    # -- period ---------------------------------------------------------
    s.period = filled(sorted(set(split.period_kinds)),
                      text="; ".join(split.period_texts))

    # -- target ---------------------------------------------------------
    if split.target_texts:
        if split.target_kind == "configured":
            s.target = filled(["configured"], text="; ".join(split.target_texts))
        else:
            s.target = filled(_target_values(split.target_texts),
                              text="; ".join(split.target_texts))

    # -- subject, the residue (rule 34) ---------------------------------
    sub = split.subject_text or ""
    keys = _match(sub, semantics, ("metric",))
    if keys:
        s.subject = filled(keys[:1], text=sub)
    elif split.operation == "count" or _COUNT_NOUNS.search(sub):
        s.subject = filled(["loan_count"], text=sub)
    elif sub.strip():
        s.subject = filled([], text=sub)

    # Rule 35 residue that the splitter could not place at all.
    s.unresolved_residue = list(split.unresolved)
    return s


_MULT = {"k": 1e3, "m": 1e6, "mm": 1e6, "bn": 1e9, "b": 1e9}


def _target_values(texts: List[str]) -> List[str]:
    out: List[str] = []
    for t in texts:
        m = re.search(r"(?:£|\$|€)?\s*(\d[\d,.]*)\s*(k|mm|m|bn|b)?", t, re.I)
        if not m:
            continue
        try:
            n = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        n *= _MULT.get((m.group(2) or "").lower(), 1)
        out.append("value:%d" % int(n))
    return out
