"""Several figures, one question: the governed answer composer (P1, D4).

"Is any of the pipeline overdue to complete, and how much?" asks for TWO
figures of ONE population — the number of overdue cases and their amount. The
plan says so exactly: one output, two measures, one filter. Every runtime
serves one figure per plan, so until this the question was declined (and,
before D18, answered by the legacy path with the whole pipeline).

The P0 design settled where this belongs (§22.3): "Two figures on one axis is
answer composition (P1, D4/D5), not a pipeline branch." So it is here, above
every runtime, and it is the same rule for the funded book, the pipeline, the
forecast and stage movement:

    SPLIT      the plan into one plan per figure — the same population, the
               same filters, the same grouping, the same period; only the
               measure differs. Structural: nothing is re-read or re-decided.
    SERVE      each through the runtime that owns it, with every gate that
               runtime applies to a one-figure question — perimeter,
               population proof, execution, reconciliation, rendering.
    ALL OR     every figure is answered or none is: a composed answer that
    NOTHING    drops a figure it was asked for is the substitution D18 ended.
    ONE DATA   the figures must come from the same data (the identity each
               receipt declares); figures from two extracts are not put side by
               side as if they described one population.
    COMPOSE    one answer that states every figure by name with its as-at (D4),
               one table with a column per figure where they share an axis,
               and the evidence of every part — the coverage owner proves the
               composed answer as the conjunction of its parts.

A runtime that serves a set of figures natively (stage movement's
reconciliation reports opening, arrivals, departures and closing together) is
not split: `natively_served` asks the runtime.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Why a composed answer was withheld although every part was answered: the
#: parts did not declare the same data.
FIGURES_NOT_ALIGNED = "COMPOSED_FIGURES_NOT_ALIGNED"

#: The receipt keys that say WHICH DATA a figure was read from. Whichever of
#: them a runtime declares must agree across the parts.
_IDENTITY_KEYS = ("dataset", "snapshot_id", "snapshot", "as_of", "as_of_date",
                  "period_resolution", "selected_periods", "input_vintages",
                  "population_base")


def _as_mapping(plan: Any) -> Mapping[str, Any]:
    if isinstance(plan, Mapping):
        return plan
    to_dict = getattr(plan, "to_dict", None)
    return to_dict() if callable(to_dict) else {}


def figures(plan: Any) -> List[str]:
    """The measures the plan's one output names, in the order asked."""
    outputs = list(_as_mapping(plan).get("outputs") or ())
    if len(outputs) != 1:
        return []
    return [str(m.get("concept") or "") for m in (outputs[0].get("measures") or ())
            if isinstance(m, Mapping)]


def natively_served(plan: Any) -> bool:
    """Does the plan's own runtime serve this SET of figures as one answer?

    Asked of the runtime, not decided here. Stage movement's reconciliation and
    "what moved" summary report several figures from one owner payload.
    """
    from mi_agent import plan_stage_movement_runtime as stage_rt
    return stage_rt.claims(plan) and stage_rt.serves_figures_together(plan)


def needs_composition(plan: Any) -> bool:
    return len(figures(plan)) > 1 and not natively_served(plan)


def split(plan: Any) -> List[Dict[str, Any]]:
    """One plan per figure. Everything but the measure is the plan's own.

    Each part carries `composition` — the figures of the whole and its place
    among them — so a renderer that states companion figures alongside its
    headline (D16) does not restate a figure a sibling part states.
    """
    body = _as_mapping(plan)
    names = figures(body)
    output = (body.get("outputs") or [{}])[0]
    parts: List[Dict[str, Any]] = []
    for position, measure in enumerate(output.get("measures") or ()):
        part = copy.deepcopy(dict(body))
        part_output = copy.deepcopy(dict(output))
        part_output["measures"] = [copy.deepcopy(dict(measure))]
        part["outputs"] = [part_output]
        part["plan_id"] = f"{body.get('plan_id') or 'plan'}.{position + 1}"
        part["composition"] = {"of_plan": body.get("plan_id"),
                               "figures": list(names), "position": position}
        parts.append(part)
    return parts


def siblings(plan: Any) -> List[str]:
    """The other figures of the composed answer this plan is a part of."""
    composition = _as_mapping(plan).get("composition") or {}
    wanted = list(composition.get("figures") or ())
    mine = figures(plan)
    return [f for f in wanted if f not in mine]


# --------------------------------------------------------------------------- #
# one data
# --------------------------------------------------------------------------- #

def _executed(envelope: Mapping[str, Any]) -> Mapping[str, Any]:
    block = ((envelope.get("metadata") or {}).get("governedPlan") or {})
    executed = block.get("executed") if isinstance(block, Mapping) else None
    return executed if isinstance(executed, Mapping) else {}


def data_identity(executed: Mapping[str, Any]) -> str:
    """What the figure was read from, as the receipt declares it."""
    declared = {k: executed.get(k) for k in _IDENTITY_KEYS if k in executed}
    return json.dumps(declared, sort_keys=True, default=str)


def aligned(envelopes: Sequence[Mapping[str, Any]]) -> bool:
    """Do all the parts declare the same data?"""
    identities = {data_identity(_executed(e)) for e in envelopes}
    return len(identities) == 1


# --------------------------------------------------------------------------- #
# compose
# --------------------------------------------------------------------------- #

#: A sentence boundary in a governed answer: a full stop (or ? !) followed by
#: space and the start of the next sentence. "£1.1m" and "2026-09-24" contain
#: no such boundary.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z£(])")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _BOUNDARY.split(str(text or "").strip()) if s.strip()]


def compose_answer(answers: Sequence[str]) -> str:
    """Every figure's own sentence first, in the order asked, then what they
    say about the population once: each part is already a governed answer
    naming its measure and as-at (D4), so nothing is reworded."""
    parts = [_sentences(a) for a in answers]
    leads = [p[0] for p in parts if p]
    seen = set(leads)
    rest: List[str] = []
    for p in parts:
        for sentence in p[1:]:
            if sentence not in seen:
                seen.add(sentence)
                rest.append(sentence)
    return " ".join(leads + rest)


def _merge_kpis(kpis: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """One KPI card: every part's figures, each field once (a part that states
    a companion figure — the loan count beside a balance — does not repeat
    one another part states)."""
    merged = copy.deepcopy(dict(kpis[0]))
    seen, cards = set(), []
    for artefact in kpis:
        for kpi in artefact.get("kpis") or ():
            key = str(kpi.get("field") or kpi.get("label"))
            if key not in seen:
                seen.add(key)
                cards.append(copy.deepcopy(dict(kpi)))
    merged["kpis"] = cards
    return merged


def _merge_tables(tables: Sequence[Mapping[str, Any]],
                  figure_names: Sequence[str]) -> Optional[Dict[str, Any]]:
    """One table, when every part's table is keyed by the same axis (its first
    column). Rows are matched by the axis value, never by position. A column
    two parts both carry with the same values is the same figure and is kept
    once; the same column with different values (each figure's own share of
    the total) is kept for each part, labelled with its figure."""
    axes = {str((list(t.get("columns") or ()) or [{}])[0].get("key")) for t in tables}
    if len(axes) != 1 or None in axes or "None" in axes:
        return None
    axis = axes.pop()
    by_member: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    columns: List[Dict[str, Any]] = [dict(tables[0]["columns"][0])]
    for position, table in enumerate(tables):
        rows = {str(r.get(axis)): r for r in table.get("rows") or ()}
        for member, row in rows.items():
            if member not in by_member:
                by_member[member] = {axis: row.get(axis)}
                order.append(member)
        for column in list(table.get("columns") or ())[1:]:
            key = str(column.get("key"))
            taken = next((c for c in columns if c["key"] == key), None)
            if taken is not None and all(
                    by_member[m].get(key) == (rows.get(m) or {}).get(key)
                    for m in rows):
                continue                                  # the same figure
            out_key = key if taken is None else f"{key}__{position + 1}"
            label = str(column.get("label") or key)
            if any(c.get("label") == label for c in columns):
                label = f"{label} ({figure_names[position]})"
            columns.append(dict(column, key=out_key, label=label))
            for member, row in rows.items():
                by_member[member][out_key] = row.get(key)
    merged = copy.deepcopy(dict(tables[0]))
    merged["columns"] = columns
    merged["rows"] = [by_member[m] for m in order]
    merged["description"] = f"{len(order)} rows."
    return merged


def compose_artefacts(parts: Sequence[Sequence[Mapping[str, Any]]],
                      figure_names: Sequence[str] = ()
                      ) -> List[Dict[str, Any]]:
    """The parts' artefacts, composed by kind: every KPI in one card, one
    table per shared axis, and every chart (one per figure), in the order
    asked."""
    names = list(figure_names) or [f"figure {i + 1}" for i in range(len(parts))]
    kpis = [a for p in parts for a in p if a.get("type") == "kpi"]
    tables = [[a for a in p if a.get("type") == "table"] for p in parts]
    out: List[Dict[str, Any]] = []
    if kpis:
        out.append(_merge_kpis(kpis))
    merged_table = None
    if all(len(t) == 1 for t in tables):
        merged_table = _merge_tables([t[0] for t in tables], names)
    for p in parts:
        for artefact in p:
            if artefact.get("type") == "kpi":
                continue
            if artefact.get("type") == "table" and merged_table is not None:
                continue
            out.append(copy.deepcopy(dict(artefact)))
    if merged_table is not None:
        out.append(merged_table)
    return out


def _union(lists: Sequence[Sequence[Any]]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for items in lists:
        for item in items or ():
            key = json.dumps(item, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def _figure_name(concept: str) -> str:
    """A figure's governed display name, for a column two figures share."""
    try:
        from mi_agent.interpretation_v2.vocabulary import load_governed_vocabulary
        found = load_governed_vocabulary().concepts.get(concept)
        if found is not None and found.label:
            return str(found.label)
    except Exception:                                                # noqa: BLE001
        pass
    return concept.replace("_", " ")


def compose(plan: Any, envelopes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The composed envelope: every part's figure, one answer, every part's
    evidence. The caller has already proved every part answered and aligned."""
    first = envelopes[0]
    composed = copy.deepcopy(dict(first))
    composed["answer"] = compose_answer([e.get("answer") or "" for e in envelopes])
    composed["artifacts"] = compose_artefacts(
        [list(e.get("artifacts") or ()) for e in envelopes],
        [_figure_name(f) for f in figures(plan)])
    for key in ("warnings", "assumptions", "diagnostics", "sourceNotes"):
        composed[key] = _union([list(e.get(key) or ()) for e in envelopes])
    meta = dict(composed.get("metadata") or {})
    requested = dict(((first.get("metadata") or {}).get("governedPlan") or {})
                     .get("requested") or {})
    requested["measure_concepts"] = figures(plan)
    meta["governedPlan"] = {
        "requested": requested,
        # EVERY PART'S OWN GOVERNED OBJECTS, so the coverage owner proves the
        # composed answer as the conjunction of the parts' proofs.
        "composed": [dict((e.get("metadata") or {}).get("governedPlan") or {})
                     for e in envelopes],
    }
    meta["composition"] = {"figures": figures(plan),
                           "plan_id": _as_mapping(plan).get("plan_id"),
                           "rule": "several figures of one population, each "
                                   "served by its own runtime, all or none"}
    composed["metadata"] = meta
    return composed
