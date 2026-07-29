"""
engine.annex_delivery_agent.instrumentation
===========================================

Making a proven builder's behaviour visible without changing it.

The Gate 5 Annex 2 builder reaches schema validity partly by inserting no-data
sentinels, coercing values into the shape a mapping branch demands, and skipping
optional fields it cannot place. Those behaviours are deliberate and this module
does not alter any of them. It answers a different question: *what exactly did
it do this time?*

Three techniques, in order of preference:

1. **Interception** — a transparent wrapper is installed over a builder helper
   for the duration of one build. The wrapper delegates to the original,
   forwards its return value and lets its exceptions propagate; it only records
   what it saw. Installed and removed by a context manager, so a failed build
   cannot leave a patched module behind.

2. **Structural derivation** — the finished artefact is censused for no-data
   leaves by element path, and each path is attributed to a source field or, if
   no source field maps there, to an automatic insertion. This reads the
   evidence out of the artefact itself rather than out of the log.

3. **Reconciliation** — no-data values counted in the input are compared with
   no-data values counted in the output. The difference is the number the
   builder added, independently of (1) and (2). When it disagrees with the
   structural attribution, the residual is reported as unattributed rather than
   quietly dropped.

Console output is never parsed.
"""

from __future__ import annotations

import re
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from .evidence import CoercionCollector

#: How many records to walk when discovering the distinct no-data element paths.
#: The artefact is structurally regular — every record is built by the same code
#: path — so a spread of records finds the shapes; the totals are then counted
#: exactly across the whole tree, and any shortfall is surfaced as unattributed.
PATH_DISCOVERY_SAMPLE = 40

_ND_RE = re.compile(r"^ND[1-5]$")


# --------------------------------------------------------------------------- #
# 1. Interception
# --------------------------------------------------------------------------- #


@contextmanager
def observe_coercions(module: Any, function_name: str,
                      collector: CoercionCollector) -> Iterator[CoercionCollector]:
    """Record every value a builder coercion function changed.

    ``module.function_name`` must have the signature ``(code, value) -> value``.
    The wrapper is exactly transparent: same arguments, same return value, same
    exceptions. It is removed on exit even if the build raises.

    Patching the module global (rather than editing the builder) is what keeps
    the business logic untouched — the builder resolves the helper through its
    module namespace at call time, so it runs the original function through our
    observer and its behaviour is bit-identical.
    """
    original: Callable[[str, str], str] = getattr(module, function_name)

    def observer(code: str, value: str) -> str:
        produced = original(code, value)
        if produced != value:
            collector.record(code, value, produced)
        return produced

    setattr(module, function_name, observer)
    try:
        yield collector
    finally:
        setattr(module, function_name, original)


@contextmanager
def observe_branch_drops(module: Any, function_name: str,
                         sink: Counter) -> Iterator[Counter]:
    """Count the times branch selection returned nothing for a non-blank value.

    ``module.function_name`` must have the signature ``(specs, value) -> [spec]``.
    An empty selection for a non-blank, non-no-data value means the value was
    present in the prepared data and did not reach the artefact. The function
    does not receive the field code, so this yields a *total*; per-field
    attribution comes from :func:`analyse_branch_coverage`, which is exact.
    """
    original = getattr(module, function_name)

    def observer(specs: Sequence[Any], value: str) -> Sequence[Any]:
        chosen = original(specs, value)
        if not chosen:
            text = str(value or "").strip()
            if text and not _ND_RE.match(text.upper()):
                sink["dropped_values"] += 1
        return chosen

    setattr(module, function_name, observer)
    try:
        yield sink
    finally:
        setattr(module, function_name, original)


# --------------------------------------------------------------------------- #
# 2. Structural derivation
# --------------------------------------------------------------------------- #


def _localname(tag: Any) -> str:
    text = str(tag)
    return text.rsplit("}", 1)[-1] if "}" in text else text


def _walk_paths(element: Any, prefix: Tuple[str, ...], leaf_name: str,
                stop_at: Optional[str], out: Set[Tuple[str, ...]]) -> None:
    """Depth-first collection of the distinct paths to ``leaf_name`` leaves."""
    for child in element:
        name = _localname(child.tag)
        if not isinstance(name, str) or name.startswith("<"):  # comments / PIs
            continue
        path = (*prefix, name)
        if name == leaf_name:
            out.add(path)
            continue
        if stop_at is not None and name == stop_at:
            continue
        _walk_paths(child, path, leaf_name, stop_at, out)


def discover_nodata_paths(root: Any, *, ns: str, record_anchor: str,
                          leaf_name: str = "NoData",
                          sample: int = PATH_DISCOVERY_SAMPLE) -> List[Tuple[str, ...]]:
    """The distinct element paths at which no-data leaves appear.

    Paths are absolute from ``Document`` so they can be compared directly with
    the mapping workbook's ``PATH`` column. Record-level paths are discovered
    from a spread of records; header-level paths are discovered by walking the
    tree with record subtrees pruned.
    """
    found: Set[Tuple[str, ...]] = set()
    root_name = _localname(root.tag)

    # Header / pool-level leaves: walk everything except inside records.
    _walk_paths(root, (root_name,), leaf_name, record_anchor, found)

    records = root.findall(f".//{{{ns}}}{record_anchor}")
    if records:
        picked = _spread(records, sample)
        # The prefix above a record is a singleton chain, so any record gives it.
        prefix = _absolute_prefix(records[0], root, root_name)
        for record in picked:
            _walk_paths(record, prefix, leaf_name, None, found)
        # The anchor itself, when a NoData sits directly beneath it.
        if _localname(records[0].tag) == leaf_name:  # pragma: no cover - defensive
            found.add(prefix)

    return sorted(found)


def _absolute_prefix(node: Any, root: Any, root_name: str) -> Tuple[str, ...]:
    """Path from the document root down to and including ``node``."""
    parts: List[str] = []
    cur = node
    while cur is not None:
        parts.append(_localname(cur.tag))
        if cur is root:
            break
        cur = cur.getparent()
    parts.reverse()
    if parts and parts[0] != root_name:  # pragma: no cover - defensive
        parts.insert(0, root_name)
    return tuple(parts)


def _spread(items: Sequence[Any], sample: int) -> List[Any]:
    """A deterministic spread across ``items`` — first, last and evenly between."""
    n = len(items)
    if n <= sample:
        return list(items)
    step = max(1, n // sample)
    picked = [items[i] for i in range(0, n, step)][:sample - 1]
    picked.append(items[-1])
    return picked


def count_paths(root: Any, paths: Iterable[Tuple[str, ...]], *, ns: str) -> Dict[Tuple[str, ...], int]:
    """Exact occurrence count for each absolute path, using lxml's own matcher."""
    counts: Dict[Tuple[str, ...], int] = {}
    for path in paths:
        # path[0] is the document root element itself.
        relative = path[1:]
        if not relative:
            continue
        expr = "/".join(f"{{{ns}}}{part}" for part in relative)
        counts[path] = len(root.findall(expr))
    return counts


def total_leaves(root: Any, *, ns: str, leaf_name: str = "NoData") -> int:
    """Total no-data leaves in the artefact (C-speed iteration)."""
    return sum(1 for _ in root.iter(f"{{{ns}}}{leaf_name}"))


def mapped_nodata_paths(specs_by_code: Dict[str, List[Any]],
                        available_codes: Iterable[str],
                        *, nodata_tags: Iterable[str] = ("NODATA", "NODATA4")) -> Dict[Tuple[str, ...], List[str]]:
    """``path -> [codes]`` for every no-data branch reachable from the input data.

    A no-data leaf standing at a path in this map came from a value in the
    prepared dataset. A no-data leaf standing anywhere else was inserted by the
    builder — that is the whole attribution rule, and it needs no knowledge of
    what any individual code means.
    """
    wanted = {str(t).upper() for t in nodata_tags}
    codes = {str(c) for c in available_codes}
    out: Dict[Tuple[str, ...], List[str]] = {}
    for code, specs in specs_by_code.items():
        if code not in codes:
            continue
        for spec in specs:
            if str(getattr(spec, "tag", "")).upper() not in wanted:
                continue
            path = tuple(p for p in str(getattr(spec, "path", "")).strip("/").split("/") if p)
            if path:
                out.setdefault(path, []).append(code)
    return out


def describe_path(path: Tuple[str, ...], *, record_anchor: str) -> str:
    """A short, readable location for an unmapped no-data leaf.

    Relative to the repeating record element, so the operator sees
    ``FinDtls/ScndryOblgrIncm/IncmVal/NoDataOptn/NoData`` rather than the full
    document path repeated on every line.
    """
    if record_anchor in path:
        idx = path.index(record_anchor)
        tail = path[idx + 1:]
    else:
        tail = path[1:]
    return "/".join(tail) or "/".join(path)


def owning_element(path: Tuple[str, ...],
                   *, choice_tags: Iterable[str] = ("NoDataOptn", "NoData")) -> str:
    """The element a no-data leaf stands *for*.

    ``.../ScndryOblgrIncm/IncmVal/NoDataOptn/NoData`` describes a missing
    ``IncmVal``, not a missing ``NoData``. Stripping the choice wrapper is what
    turns two identical-looking evidence lines into "secondary obligor income"
    and "secondary obligor income verification".
    """
    skip = {str(t) for t in choice_tags}
    meaningful = [p for p in path if p not in skip]
    return "/".join(meaningful[-2:]) if len(meaningful) >= 2 else (
        meaningful[-1] if meaningful else "(unknown)")


# --------------------------------------------------------------------------- #
# 3. Reconciliation and analytic coverage
# --------------------------------------------------------------------------- #


def count_nodata_in_frame(frame: Any) -> Tuple[int, Dict[str, int]]:
    """No-data sentinels present in the prepared dataset, in total and per field."""
    per_field: Dict[str, int] = {}
    total = 0
    for column in frame.columns:
        series = frame[column].astype(str).str.strip().str.upper()
        hits = int(series.str.fullmatch(r"ND[1-5]").fillna(False).sum())
        if hits:
            per_field[str(column)] = hits
            total += hits
    return total, per_field


def analyse_branch_coverage(frame: Any, specs_by_code: Dict[str, List[Any]],
                            select_specs: Callable[[Sequence[Any], str], Sequence[Any]],
                            *, record_anchor: str) -> Dict[str, Dict[str, Any]]:
    """Which prepared values the mapping cannot place, per field — exactly.

    Evaluates branch selection once per *distinct value* per field rather than
    once per row: the builder's selection depends only on the value, so this is
    exact and costs a few thousand evaluations instead of a million.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for code in frame.columns:
        specs = specs_by_code.get(str(code))
        if not specs:
            continue
        record_specs = [s for s in specs if record_anchor in str(getattr(s, "path", ""))]
        if not record_specs:
            continue
        series = frame[code].astype(str)
        value_counts = series.value_counts()
        dropped = 0
        samples: List[str] = []
        for raw_value, occurrences in value_counts.items():
            value = "" if raw_value is None else str(raw_value)
            text = value.strip()
            if text == "" or text.lower() == "nan":
                continue
            if not select_specs(record_specs, text):
                dropped += int(occurrences)
                if len(samples) < 5:
                    samples.append(text)
        if dropped:
            out[str(code)] = {"records_affected": dropped, "samples": samples}
    return out


__all__ = [
    "PATH_DISCOVERY_SAMPLE",
    "observe_coercions", "observe_branch_drops",
    "discover_nodata_paths", "count_paths", "total_leaves",
    "mapped_nodata_paths", "describe_path", "owning_element",
    "count_nodata_in_frame", "analyse_branch_coverage",
]
