#!/usr/bin/env python3
"""migration_phase0/filter_ownership_trace.py — who owns an evolution filter.

READ-ONLY. The last C6 prerequisite is stated as "evolution applies filters
through route-local machinery instead of consuming the governed FilterClaim
through analytical_plan.lens_filters".

Three things are called "filters" in that sentence and they are not one concept.
This measures what each can actually express, for the same questions, so the
remediation is designed against evidence rather than against the shared word.

    python -m migration_phase0.filter_ownership_trace
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")

#: Filtered-evolution probes: a numeric bound, a categorical dimension value, a
#: governed portfolio lens, and a pipeline stage.
PROBES: Tuple[Tuple[str, str], ...] = (
    ("F-NUM", "Show funded balance evolution by month for loans above 50% LTV."),
    ("F-CAT", "Show funded balance evolution by month for London."),
    ("F-CHAN", "Show the trend in funded balance over time for the Alpha channel."),
    ("F-LENS", "Show funded balance evolution by month for the acquired book."),
    ("F-STAGE", "How has pipeline balance for offer-stage cases changed over the last five weeks?"),
    ("F-NONE", "Show funded balance evolution by month."),
)


def _env() -> Tuple[str, Dict[str, Any]]:
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    # THE loader the serving path uses. An earlier cut of this instrument probed
    # three plausible names on `mi_service`, none of which exist, and silently
    # returned {} — which made every `spec.filters` empty and the whole census
    # vacuous in the direction that would have flattered the conclusion.
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    sem = load_mi_semantics(semantics_path())
    if not sem or "fields" not in sem:
        raise SystemExit("semantics did not load - refusing to measure vacuously")
    return cfg.CLIENT_ID, sem


def _corpus() -> List[str]:
    out, seen = [], set()
    for f in CORPORA:
        p = _REPO / f
        if not p.exists():
            continue
        for row in json.loads(p.read_text(encoding="utf-8"))["rows"]:
            q = row.get("question") or ""
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def main() -> int:
    client_id, semantics = _env()
    from mi_agent.parsed_question import ParsedQuestion
    from question_interpretation import projection as proj

    print("=" * 92)
    print("EVOLUTION FILTER OWNERSHIP — three things called 'filters'")
    print("=" * 92)

    # ---- 1. What each mechanism can EXPRESS ------------------------------- #
    print("\n1. WHAT EACH MECHANISM CAN EXPRESS")
    from question_interpretation.schema import FilterClaim
    import dataclasses
    fc_fields = [f.name for f in dataclasses.fields(FilterClaim)]
    print(f"   FilterClaim fields          : {fc_fields}")
    print(f"   ...carries a FIELD key?     : "
          f"{'YES' if any(f in fc_fields for f in ('field', 'field_key', 'candidate_concept')) else 'NO'}")

    import inspect
    from mi_agent_api import analytical_plan as ap
    lf = inspect.getsource(ap.lens_filters)
    print(f"   lens_filters returns        : "
          f"{'source_portfolio_id only' if 'source_portfolio_id' in lf else 'general'}")
    print(f"   lens_filters keys on step   : "
          f"{'kind == source_portfolio_lens' if 'source_portfolio_lens' in lf else '?'}")

    from mi_agent.mi_query_executor import _apply_filters
    af = inspect.getsource(_apply_filters)
    print(f"   _apply_filters handles      : "
          f"semantic-key resolution={'resolve_semantic_field' in af}, "
          f"numeric ops={'_OP_ALIASES' in af}, "
          f"percent scale={'percent_storage_scale' in af}")
    from mi_agent_api.evolution import _scope_frame_lens
    sfl = inspect.getsource(_scope_frame_lens)
    print(f"   _scope_frame_lens on absent column: "
          f"{'NO-OP (continue)' if 'continue' in sfl else 'raises'}")
    print(f"   _apply_filters  on absent column: "
          f"{'RAISES (_require_column)' if '_require_column' in af else 'no-op'}")

    # ---- 2. Per-question, what each says ---------------------------------- #
    print("\n2. PER-QUESTION: SPEC vs CONTRACT vs LENS")
    print(f"\n{'id':<9}{'spec.filters':<44}{'FilterClaim(op,val,cat)':<34}lens_filters")
    print("-" * 92)
    rows: List[Dict[str, Any]] = []
    for cid, q in PROBES:
        spec = ParsedQuestion.parse(q, semantics).spec
        sf = dict(getattr(spec, "filters", None) or {})
        qi = proj.project(q, semantics=semantics, frame=None)
        claims = [(f.operator, f.value, f.categorical_value)
                  for f in (getattr(qi, "filters", None) or [])]
        # An evolution plan has no source_portfolio_lens step unless the question
        # named a book, so lens_filters is asked directly for what it would give.
        try:
            plan = ap.build_temporal_compare_plan(qi)
            lens = ap.lens_filters(plan)
        except Exception as exc:  # noqa: BLE001
            lens = f"<{type(exc).__name__}>"
        print(f"{cid:<9}{str(sf)[:42]:<44}{str(claims)[:32]:<34}{lens}")
        rows.append({"id": cid, "question": q, "spec_filters": sf,
                     "claims": claims, "lens": str(lens)})

    # ---- 3. Corpus: which filter FIELDS ever reach evolution -------------- #
    print("\n3. WHICH FILTER FIELDS EVER REACH A FILTERED EVOLUTION QUESTION (882 corpus)")
    from mi_agent_api.workspace import resolve_dataset
    fields: Dict[str, int] = {}
    filtered_q = 0
    for q in _corpus():
        try:
            spec = ParsedQuestion.parse(q, semantics).spec
        except Exception:  # noqa: BLE001
            continue
        sf = dict(getattr(spec, "filters", None) or {})
        if not sf:
            continue
        filtered_q += 1
        for k in sf:
            fields[k] = fields.get(k, 0) + 1
    print(f"   corpus questions carrying spec.filters: {filtered_q}")
    for k, n in sorted(fields.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<34} {n}")
    print(f"   of these, expressible by lens_filters (source_portfolio_id only): "
          f"{fields.get('source_portfolio_id', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
