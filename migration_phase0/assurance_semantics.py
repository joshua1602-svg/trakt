#!/usr/bin/env python3
"""migration_phase0/assurance_semantics.py — the ONE way assurance gets semantics.

Why this exists
---------------
A C6 trace was written against three plausible loader names on `mi_service` —
`load_semantics`, `_load_semantics`, `semantics_for`. None of them exists. The
probe loop caught `AttributeError`/`TypeError`, fell through to `{}`, and every
downstream measurement ran against an EMPTY registry while exiting 0 and printing
well-formed numbers. It was caught by hand, and only because one figure looked
implausible.

Two instruments carried that same `_env()`, and both had been reporting clean
results on no semantics at all:

    dependency_verification_temporal_compare   0 dataset disagreements,
                                               0 measure disagreements,
                                               "48 of an EXPECTED 48 structural"
    pipeline_stage_census                      the C6 stage owner-agreement figure

An assurance instrument that cannot tell a loaded registry from an empty one is
not assurance. This module is the single place that answers the question, it
delegates to production rather than reimplementing it, and it FAILS rather than
degrades.

Deliberately NOT provided: any default, any fallback, any `or {}`. A caller that
cannot get semantics must stop, because the alternative is what this exists to
prevent.

This module is also the home of the assurance failure vocabulary generally. The
loader defect above has a sibling: a broad `except` that turns a crashed
MEASUREMENT into an empty result, so the instrument continues and reports
zero. `AssuranceMeasurementError` and `measurement_failed` are for that, and the
distinction they enforce is the point —

    measurement ran, found nothing   -> an empty result, which is evidence
    measurement could not run        -> an exception, which is not

Those two must never share a representation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class AssuranceError(RuntimeError):
    """Base: this assurance run cannot be trusted and must not report a result."""


class AssuranceSemanticsError(AssuranceError):
    """Raised when governed semantics are absent, empty or materially incomplete."""


class AssuranceMeasurementError(AssuranceError):
    """Raised when a measurement could not be completed.

    NOT for a measurement that legitimately found nothing — that is an empty
    result and it is evidence. This is for the case where the instrument was
    unable to measure at all, which no count can represent.
    """


def measurement_failed(instrument: str, case: Optional[str],
                       exc: BaseException) -> "AssuranceMeasurementError":
    """The one fail-loud constructor, carrying enough to identify the failure.

    Chained with `raise ... from exc` at the call site so the original traceback
    and root cause survive; an assurance failure that hides its cause just moves
    the opacity somewhere else.
    """
    where = " on %r" % (case,) if case else ""
    return AssuranceMeasurementError(
        "ASSURANCE INVALID - measurement failed in %s%s: %s: %s"
        % (instrument, where, type(exc).__name__, exc))


#: Governed fields every semantics-dependent instrument in this directory relies
#: on, directly or through the parser. Presence is checked by NAME, not by count:
#: a partially built registry can satisfy `len(fields) > 0` and still be missing
#: the exact field an instrument measures, which is the failure this guards.
REQUIRED_FIELDS: tuple = (
    "current_outstanding_balance",   # the funded measure every conversion uses
    "current_loan_to_value",         # the numeric-bound filter family
    "collateral_geography",          # the categorical dimension family
    "youngest_borrower_age",         # the second numeric family
    "pipeline_stage",                # the governed Pipeline Stage dimension
)


def load_assurance_semantics(*, required_fields: Optional[Iterable[str]] = None
                             ) -> Dict[str, Any]:
    """The governed MI semantics, loaded exactly as the serving path loads them.

    Delegates to production twice over — `data_source.semantics_path()` resolves
    the file (honouring the `MI_AGENT_SEMANTICS` override, which the instruments
    that hardcode the registry path silently bypass), and
    `mi_query_validator.load_mi_semantics` parses it. No migration-specific
    notion of "loaded semantics" is defined here.

    Raises `AssuranceSemanticsError` — never returns a degraded dict — if the
    file is missing, the registry is empty, or a required governed field is
    absent. The message names what is missing.
    """
    try:
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.data_source import semantics_path
    except Exception as exc:  # noqa: BLE001 - an import failure is a hard stop
        raise AssuranceSemanticsError(
            "ASSURANCE INVALID - the production semantics loader could not be "
            "imported: %s: %s" % (type(exc).__name__, exc)) from exc

    path = semantics_path()
    if not Path(path).exists():
        raise AssuranceSemanticsError(
            "ASSURANCE INVALID - governed MI semantics file does not exist: %s "
            "(set MI_AGENT_SEMANTICS to override)" % path)

    try:
        semantics = load_mi_semantics(path)
    except Exception as exc:  # noqa: BLE001 - deliberately not swallowed
        raise AssuranceSemanticsError(
            "ASSURANCE INVALID - governed MI semantics at %s failed to load: "
            "%s: %s" % (path, type(exc).__name__, exc)) from exc

    if not isinstance(semantics, dict) or not semantics:
        raise AssuranceSemanticsError(
            "ASSURANCE INVALID - governed MI semantics loaded EMPTY from %s" % path)

    fields = semantics.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise AssuranceSemanticsError(
            "ASSURANCE INVALID - governed MI semantics from %s carry no 'fields' "
            "registry (top-level keys: %s)" % (path, sorted(semantics)))

    wanted = tuple(required_fields) if required_fields is not None else REQUIRED_FIELDS
    missing = [f for f in wanted if f not in fields]
    if missing:
        raise AssuranceSemanticsError(
            "ASSURANCE INVALID - governed MI semantics from %s are materially "
            "incomplete; %d field(s) present but these required governed fields "
            "are absent: %s" % (path, len(fields), ", ".join(missing)))
    return semantics


def describe(semantics: Dict[str, Any]) -> str:
    """One line an instrument can print to show WHAT it measured against."""
    fields = semantics.get("fields") or {}
    meta = semantics.get("metadata") or {}
    return ("governed semantics: %d fields, generated_at=%s"
            % (len(fields), meta.get("generated_at", "unknown")))


def main(argv=None) -> int:
    """Self-check: prove the loader works and says what it loaded."""
    try:
        semantics = load_assurance_semantics()
    except AssuranceSemanticsError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(describe(semantics))
    fields = semantics["fields"]
    for name in REQUIRED_FIELDS:
        print("   %-32s present" % name)
    print("   %-32s %d" % ("total governed fields", len(fields)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
