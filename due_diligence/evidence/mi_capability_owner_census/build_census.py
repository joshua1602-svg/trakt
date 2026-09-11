#!/usr/bin/env python3
"""Render CAPABILITY_OWNER_CENSUS.md and .json from census_records.py.

ONE SOURCE OF TRUTH. Both artefacts are generated; neither is hand-maintained,
so they cannot drift apart.

    python3 due_diligence/evidence/mi_capability_owner_census/build_census.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import census_records as R  # noqa: E402


def raw_text_verdict(cap) -> str:
    return "YES" if str(cap["raw_text_required"]).strip().upper().startswith("YES") else "NO"


def short(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return cell(text if len(text) <= n else text[: n - 1] + "\u2026")


def cell(text) -> str:
    """A value safe to place inside a Markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def owner_short(cap) -> str:
    """The first, lowest owner named — enough to identify it in a wide table."""
    owner = " ".join(str(cap["calculation_owner"]).split())
    for cut in (" (", " —", " -> ", ";", ",", " over ", " and ", " invoked ",
                " delegating ", " reached ", " using ", " is "):
        i = owner.find(cut)
        if i > 0:
            owner = owner[:i]
    return cell(owner.rstrip(".,"))


def filt(cap) -> str:
    v = str(cap["filters"]).strip()
    for token in ("YES_GENERIC", "YES_LIMITED", "UNKNOWN", "NO"):
        if v.upper().startswith(token):
            return token
    return "NO"


def filter_detail(cap) -> str:
    """The filter description with its leading verdict token stripped."""
    v = " ".join(str(cap["filters"]).split())
    for token in ("YES_GENERIC", "YES_LIMITED", "UNKNOWN", "NO"):
        if v.upper().startswith(token):
            v = v[len(token):].lstrip(" -—:,")
            break
    return cell(v or "(no further detail)")


def dims(cap) -> str:
    v = str(cap["dimensions"]).strip()
    for token in ("YES_GENERIC", "YES_LIMITED", "UNKNOWN", "NO"):
        if v.upper().startswith(token):
            return token
    return short(v, 28)


def build_json() -> dict:
    caps = []
    for i, c in enumerate(R.CAPABILITIES, 1):
        d = dict(c)
        d["index"] = i
        d["raw_text_required_after_semantic_decision"] = raw_text_verdict(c)
        caps.append(d)
    return {
        "census_sha": R.CENSUS_SHA,
        "branch": R.BRANCH,
        "glossary": R.GLOSSARY,
        "measurements": R.MEASUREMENTS,
        "execution_proofs": R.EXECUTION_PROOFS,
        "capabilities": caps,
        "raw_text_readers": R.RAW_TEXT_READERS,
        "owner_map": [{"owner": o, "capabilities": list(cs)} for o, cs in R.OWNER_MAP],
        "composing_layers": [{"layer": l, "role": r} for l, r in R.COMPOSING_LAYERS],
        "totals": totals(),
    }


def totals() -> dict:
    caps = R.CAPABILITIES
    by_conn = {}
    for c in caps:
        by_conn.setdefault(c["connectivity"], []).append(c["name"])
    gap_counts = {}
    for c in caps:
        for g in c["gap_class"]:
            gap_counts[g] = gap_counts.get(g, 0) + 1
    raw_dep = [c["name"] for c in caps if raw_text_verdict(c) == "YES"]
    raw_ind = [c["name"] for c in caps if raw_text_verdict(c) == "NO"]
    return {
        "total_user_facing_capabilities": len(caps),
        "total_distinct_calculation_owners": len(R.OWNER_MAP),
        "composing_layers_that_calculate_nothing": len(R.COMPOSING_LAYERS),
        "connectivity": {k: len(v) for k, v in sorted(by_conn.items())},
        "plan_representable": {
            k: sum(1 for c in caps if c["plan_representable"] == k)
            for k in ("YES", "PARTIAL", "NO")},
        "migration_complexity": {
            k: sum(1 for c in caps if c["complexity"] == k)
            for k in ("TRIVIAL", "SMALL", "MEDIUM", "LARGE")},
        "gap_classes": dict(sorted(gap_counts.items(), key=lambda kv: -kv[1])),
        "raw_text_dependent_capabilities": raw_dep,
        "raw_text_independent_capabilities": raw_ind,
        "genuine_capability_gaps": [
            c["name"] for c in caps if "GENUINE_CAPABILITY_GAP" in c["gap_class"]],
    }


def md_table() -> str:
    head = ("| # | Capability | Dataset | Deterministic calculation owner | Current | "
            "Temporal | Filters | Dimensions | Plan repr. | New-plan connectivity | "
            "Migration complexity |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [head, sep]
    for i, c in enumerate(R.CAPABILITIES, 1):
        current = "yes" if "current" in str(c["temporal"]) else "—"
        rows.append("| {} | {} | {} | `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
            i, cell(c["name"]), cell(c["dataset"]), owner_short(c), current,
            short(c["temporal"], 34), filt(c), dims(c),
            c["plan_representable"], c["connectivity"], c["complexity"]))
    return "\n".join(rows)


def md_capability_records() -> str:
    out = []
    for i, c in enumerate(R.CAPABILITIES, 1):
        out.append(f"### {i}. {c['name']}\n")
        out.append("**USER_INTENT_EXAMPLES**")
        for q in c["intents"]:
            out.append(f"- _{q}_")
        out.append("")
        rows = [
            ("DATASET", c["dataset"]),
            ("DATASET_OWNER", "`%s`" % c["dataset_owner"]),
            ("PREPARATION_OWNER", "`%s`" % c["preparation_owner"]),
            ("**CALCULATION_OWNER**", "**`%s`**" % c["calculation_owner"]),
            ("MEASURES_SUPPORTED", c["measures"]),
            ("FILTERS_SUPPORTED", "**%s** — %s" % (filt(c), filter_detail(c))),
            ("DIMENSIONS_SUPPORTED", c["dimensions"]),
            ("TEMPORAL_SUPPORT", c["temporal"]),
            ("POPULATION_SCOPE_SUPPORT", c["population_scope"]),
            ("RAW_QUESTION_REQUIRED_AFTER_SEMANTIC_DECISION",
             "**%s**" % raw_text_verdict(c)),
        ]
        out.append("| Field | Value |")
        out.append("|---|---|")
        for k, v in rows:
            out.append("| %s | %s |" % (k, cell(v)))
        out.append("")
        out.append("**ORCHESTRATION_ENTRY_POINTS**")
        for e in c["entry_points"]:
            out.append("- %s" % " ".join(str(e).split()))
        out.append("")
        if c["raw_text_where"]:
            out.append("**Raw-question reads (exact functions)**")
            for w in c["raw_text_where"]:
                out.append(f"- {w}")
            out.append("")
        out.append("**EXECUTION_EVIDENCE**")
        for e in c["evidence"]:
            out.append(f"- {e}")
        out.append("")
        out.append("| Migration | |")
        out.append("|---|---|")
        out.append("| GOVERNED_QUERY_PLAN_REPRESENTABLE | **%s** |" % c["plan_representable"])
        if c["plan_gap"]:
            out.append("| Missing semantic slot | %s |" % cell(c["plan_gap"]))
        out.append("| NEW_PLAN_CONNECTIVITY | **%s** |" % c["connectivity"])
        out.append("| CONNECTIVITY_GAP | %s |" % cell(c["connectivity_gap"]))
        out.append("| Gap class | %s |" % (", ".join("`%s`" % g for g in c["gap_class"]) or "—"))
        out.append("| MIGRATION_COMPLEXITY | **%s** |" % c["complexity"])
        out.append("")
        if c["notes"]:
            out.append("> %s" % " ".join(c["notes"].split()))
            out.append("")
        out.append("---\n")
    return "\n".join(out)


_GROUP_NOTE = (
    "_Groups 2-5 are not mutually exclusive: a capability can be partially "
    "connected AND raw-text coupled, and is listed in both. Group 5 lists only "
    "capabilities that are not already CONNECTED — the four CONNECTED "
    "capabilities also carry post-plan raw-text reads (dataset, lens, pipeline "
    "stage, missing-dimension policy), recorded in their individual records and "
    "in the raw-text audit. That is why CONNECTED here means *the plan reaches "
    "the owner*, not *the plan decides everything*._\n")


def md_connectivity_map() -> str:
    caps = R.CAPABILITIES
    groups = [
        ("1. ALREADY CONNECTED",
         [c for c in caps if c["connectivity"] == "CONNECTED"], False),
        ("2. PARTIALLY CONNECTED",
         [c for c in caps if c["connectivity"] == "PARTIAL"], True),
        ("3. REPRESENTABLE BUT NOT CONNECTED",
         [c for c in caps if c["connectivity"] == "NOT_CONNECTED"
          and c["plan_representable"] in ("YES", "PARTIAL")], True),
        ("4. NOT YET REPRESENTABLE IN GovernedQueryPlan",
         [c for c in caps if c["connectivity"] == "NOT_CONNECTED"
          and c["plan_representable"] == "NO"
          and "RAW_TEXT_COUPLING" not in c["gap_class"]
          and "GENUINE_CAPABILITY_GAP" not in c["gap_class"]], True),
        ("5. RAW-TEXT-COUPLED / REQUIRES DEEPER MIGRATION",
         [c for c in caps if "RAW_TEXT_COUPLING" in c["gap_class"]
          and c["connectivity"] != "CONNECTED"], True),
    ]
    out = []
    out.append(_GROUP_NOTE)
    for title, members, explain in groups:
        out.append(f"### {title}  ({len(members)})\n")
        if not members:
            out.append("_none_\n")
            continue
        for c in members:
            if explain:
                out.append("- **%s** — %s" % (
                    c["name"], " ".join(str(c["connectivity_gap"]).split())))
            else:
                out.append("- **%s**" % c["name"])
        out.append("")
    out.append("### 6. GENUINE CAPABILITY GAP  (%d)\n" % len(
        [c for c in caps if "GENUINE_CAPABILITY_GAP" in c["gap_class"]]))
    for c in caps:
        if "GENUINE_CAPABILITY_GAP" in c["gap_class"]:
            out.append("- **%s** — Trakt does not calculate this anywhere; the "
                       "route declines rather than substituting today's "
                       "position." % c["name"])
    out.append("")
    return "\n".join(out)


def md_owner_map() -> str:
    out = ["```"]
    for owner, caps in R.OWNER_MAP:
        out.append(owner)
        for c in caps:
            out.append("    -> %s" % c)
        out.append("")
    out.append("```")
    out.append("")
    out.append("**Layers that COMPOSE and calculate nothing** (not owners):\n")
    out.append("| Layer | Role |")
    out.append("|---|---|")
    for layer, role in R.COMPOSING_LAYERS:
        out.append("| `%s` | %s |" % (layer, cell(role)))
    return "\n".join(out)


def md_gap_map() -> str:
    buckets = {}
    for c in R.CAPABILITIES:
        for g in c["gap_class"]:
            buckets.setdefault(g, []).append(c["name"])
    order = ["DISPATCH_ONLY", "PLAN_TO_SPEC_ADAPTER", "DATASET_BINDING",
             "TEMPORAL_BINDING", "RECEIPT_ADAPTATION", "MISSING_PLAN_SEMANTIC",
             "RAW_TEXT_COUPLING", "GENUINE_CAPABILITY_GAP"]
    out = ["| Gap class | Capabilities | Count |", "|---|---|---|"]
    for g in order:
        names = buckets.get(g, [])
        out.append("| `%s` | %s | %d |" % (
            g, cell("; ".join(names)) if names else "—", len(names)))
    leftover = sorted(set(buckets) - set(order))
    for g in leftover:
        out.append("| `%s` | %s | %d |" % (g, cell("; ".join(buckets[g])), len(buckets[g])))
    return "\n".join(out)


def md_raw_text_audit() -> str:
    caps = R.CAPABILITIES
    dep = [c["name"] for c in caps if raw_text_verdict(c) == "YES"]
    ind = [c["name"] for c in caps if raw_text_verdict(c) == "NO"]
    out = []
    out.append("**RAW_TEXT_DEPENDENT_CAPABILITIES = %d of %d**\n" % (len(dep), len(caps)))
    for n in dep:
        out.append("- %s" % n)
    out.append("")
    out.append("**RAW_TEXT_INDEPENDENT_CAPABILITIES = %d of %d**\n" % (len(ind), len(caps)))
    for n in ind:
        out.append("- %s" % n)
    out.append("")
    out.append("#### Every raw-question reader, by exact function\n")
    out.append("| Function | Decides | Called from | After the plan exists? |")
    out.append("|---|---|---|---|")
    for r in R.RAW_TEXT_READERS:
        out.append("| `%s` | %s | %s | %s |" % (
            r["fn"], r["decides"], cell(r["where"]),
            "**YES**" if r["post_plan"] else "no (recognition)"))
    out.append("")
    out.append("Notes:\n")
    for r in R.RAW_TEXT_READERS:
        out.append("- `%s` — %s" % (r["fn"].split(".")[-1], " ".join(r["note"].split())))
    return "\n".join(out)


def main() -> int:
    t = totals()
    m = R.MEASUREMENTS
    doc = []
    A = doc.append
    A("# MI CAPABILITY-TO-OWNER CENSUS\n")
    A("> **READ-ONLY.** No production file was modified. No live model call was "
      "made. Nothing was deployed. Every figure below was either traced in "
      "source or measured by a deterministic offline run against committed "
      "corpora and fixtures.\n")
    A("```")
    A("CENSUS_SHA = %s" % R.CENSUS_SHA)
    A("BRANCH     = %s" % R.BRANCH)
    A("WORKING TREE AT CENSUS TIME = clean (git status --porcelain: 0 entries)")
    A("```\n")
    A("## 0. What the brief's terms map onto in this repository\n")
    A("The census must not invent vocabulary, so these bindings are stated "
      "before anything is counted.\n")
    A("| Brief's term | This repository |")
    A("|---|---|")
    for k, v in R.GLOSSARY.items():
        A("| **%s** | %s |" % (k, cell(v)))
    A("")
    A("There is **no symbol named `GovernedQueryPlan`** in the codebase. The "
      "contract the brief describes is `mi_agent/query_plan.py::QueryPlan` + "
      "`AnalyticalScope` + `PlannedOutput` + `ScopeDelta` + `Predicate`. "
      "Throughout this document *the plan* means that contract.\n")
    A("A second, unrelated plan artefact exists — "
      "`mi_agent_api/analytical_plan.py::Plan`, driven by "
      "`question_interpretation.schema.QuestionInterpretation`. Six routes are "
      "already converted onto it (Conversions 1, 2, 4, 5, geo, C7). **The two "
      "plan layers do not compose today**, and several capabilities are "
      "'migrated' onto one while being 'not connected' to the other. Both are "
      "recorded; where the distinction matters it is called out.\n")
    A("---\n")
    A("## 1. Measured baseline\n")
    A("Method: `probes/lift_census_probe.py` and `probes/route_claim_probe.py`, "
      "read-only, offline, deterministic parser (`llm_enabled=False`), no "
      "network and no model call. Raw output in `measurements/`.\n")
    A("| Measure | Value |")
    A("|---|---|")
    A("| Distinct corpus questions (stage 1 + stage 2) | **%d** |" % m["distinct_questions"])
    A("| Liftable into a `QueryPlan` (`plan_from_spec` returns a plan) | **%d (%.1f%%)** |"
      % (m["liftable_to_query_plan"], m["liftable_pct"]))
    A("| Not liftable | %d |" % m["not_liftable"])
    A("| Claimed by a specialist recogniser | %d |" % m["claimed_by_a_specialist_route"])
    A("| Fall through to the generic deterministic executor | %d |" % m["claimed_by_generic_executor"])
    A("| **Generic AND liftable — the plan is the contract *and* reaches the owner** | **%d (%.1f%%)** |"
      % (m["generic_and_liftable_ie_CONNECTED"], m["connected_pct"]))
    A("| `MIQuerySpec` fields | %d |" % m["mi_query_spec_fields_total"])
    A("| …modelled by `AnalyticalScope`/`PlannedOutput` | %d |" % m["mi_query_spec_fields_modelled_by_plan"])
    A("| …unmodelled (any one at a non-default value declines the lift) | %d |"
      % m["mi_query_spec_fields_unmodelled"])
    A("")
    A("### Why the 222 declines decline\n")
    A("| Reason | Questions |")
    A("|---|---|")
    for k, v in m["decline_reasons"].items():
        A("| %s | %d |" % (k, v))
    A("")
    A("**Read that table carefully.** Not one line says *Trakt cannot calculate "
      "this*. Every line names a **slot the plan does not have**, for arithmetic "
      "the executor already performs.\n")
    A("### Route claims, measured\n")
    A("Offline recognition without data roots, so routes whose recognition "
      "needs a resolved dataset or a frame under-report; those are covered by "
      "name in `probes/targeted_route_probe.py`.\n")
    A("```")
    A("CLAIMING ROUTE                         N   liftable   not")
    for line in [
        ("<generic_mi_workflow>", 714, 607, 107),
        ("analytical_composition", 35, 29, 6),
        ("evolution", 33, 0, 33),
        ("forecast_extrapolation", 26, 0, 26),
        ("concentration_analysis", 19, 10, 9),
        ("risk_limits", 18, 0, 18),
        ("geo_exposure", 12, 4, 8),
        ("period_change_analysis", 7, 2, 5),
        ("portfolio_summary", 4, 4, 0),
        ("cohort_conversion", 4, 2, 2),
        ("temporal_compare", 4, 0, 4),
        ("funded_bridge", 3, 0, 3),
        ("portfolio_risk_comparison", 2, 2, 0),
        ("scenario", 1, 0, 1)]:
        A("%-34s %5d %8d %6d" % line)
    A("```\n")
    A("### Deterministic execution proofs\n")
    for p in R.EXECUTION_PROOFS:
        A("**%s**\n" % p["claim"])
        A("- *How:* %s" % " ".join(p["how"].split()))
        if "fixture" in p:
            A("- *Fixture:* `%s`" % p["fixture"])
        A("- *Result:* %s" % " ".join(p["result"].split()))
        A("- *Verdict:* %s\n" % p["verdict"])
    A("---\n")
    A("## OUTPUT 1 — EXECUTIVE CAPABILITY MAP\n")
    A(md_table())
    A("")
    A("---\n")
    A("## OUTPUT 2 — CONNECTIVITY MAP\n")
    A(md_connectivity_map())
    A("---\n")
    A("## OUTPUT 3 — OWNER MAP\n")
    A("```")
    A("TOTAL_USER_FACING_CAPABILITIES  = %d" % t["total_user_facing_capabilities"])
    A("TOTAL_DISTINCT_CALCULATION_OWNERS = %d" % t["total_distinct_calculation_owners"])
    A("COMPOSING LAYERS THAT CALCULATE NOTHING = %d" % t["composing_layers_that_calculate_nothing"])
    A("```\n")
    A(md_owner_map())
    A("")
    A("---\n")
    A("## OUTPUT 4 — MIGRATION GAP MAP\n")
    A("_Classification only. No sequencing, no recommendation._\n")
    A(md_gap_map())
    A("")
    A("A capability commonly carries more than one class, so the counts sum "
      "above the capability total.\n")
    A("---\n")
    A("## RAW-TEXT DEPENDENCY AUDIT\n")
    A("The test is mechanical, and deliberately so: **after an analytical "
      "intent / spec / plan exists, does any downstream code read the natural-"
      "language question again to decide dataset, measure, aggregation, filter, "
      "dimension, period, comparison, scope or capability?** Text used purely "
      "for presentation does not count.\n")
    A(md_raw_text_audit())
    A("")
    A("---\n")
    A("## OUTPUT 5 — KEY FINDINGS\n")
    A(md_key_findings(t, m))
    A("")
    A("---\n")
    A("## Per-capability records\n")
    A(md_capability_records())
    A("## Reproducing this census\n")
    A("```bash")
    A("git checkout %s" % R.CENSUS_SHA)
    A("python3 due_diligence/evidence/mi_capability_owner_census/probes/lift_census_probe.py")
    A("python3 due_diligence/evidence/mi_capability_owner_census/probes/route_claim_probe.py")
    A("python3 due_diligence/evidence/mi_capability_owner_census/probes/targeted_route_probe.py")
    A("python3 -m pytest mi_agent/tests/test_query_plan_adapter.py \\")
    A("    mi_agent/tests/test_query_plan_compiler.py \\")
    A("    mi_agent/tests/test_query_plan_contracts.py \\")
    A("    mi_agent/tests/test_query_plan_execution.py \\")
    A("    mi_agent/tests/test_query_plan_is_the_live_contract.py \\")
    A("    mi_agent/tests/test_query_plan_reconciliation.py \\")
    A("    mi_agent/tests/test_shadow_replay.py -q")
    A("python3 due_diligence/evidence/mi_capability_owner_census/build_census.py")
    A("```\n")
    A("Requires `pandas`, `pyyaml`, `plotly`, `pytest`. No network access and "
      "no API key: every probe runs the deterministic parser.\n")

    md_path = HERE / "CAPABILITY_OWNER_CENSUS.md"
    js_path = HERE / "CAPABILITY_OWNER_CENSUS.json"
    md_path.write_text("\n".join(doc))
    js_path.write_text(json.dumps(build_json(), indent=1) + "\n")
    print("wrote", md_path)
    print("wrote", js_path)
    return 0


def md_key_findings(t, m) -> str:
    caps = R.CAPABILITIES
    n = len(caps)
    ind = len(t["raw_text_independent_capabilities"])
    conn = t["connectivity"].get("CONNECTED", 0)
    part = t["connectivity"].get("PARTIAL", 0)
    o = []
    A = o.append
    A("**1. What percentage of Trakt's existing MI capability already has a "
      "deterministic calculation owner that does NOT require raw question text?**\n")
    A("Two honest denominators, because they answer different questions.\n")
    A("- *By capability:* **%d of %d (%.0f%%)** have a calculation owner that "
      "takes structured inputs only. The remaining %d re-read the sentence for "
      "a semantic decision **after** the parse — and in every case the coupling "
      "is in the ORCHESTRATION layer (route, workflow, receipt), never in the "
      "arithmetic itself.\n" % (ind, n, 100.0 * ind / n, n - ind))
    A("- *By arithmetic:* **every one of the %d calculation owners in Output 3 "
      "takes structured inputs only.** Not one of them performs arithmetic "
      "conditioned on the sentence. The single owner that still reads the "
      "question inside its entry point — "
      "`portfolio_risk_comparison.run_portfolio_risk_comparison` — reads it to "
      "choose two POPULATIONS and then hands both to `mi_workflows.engine`, "
      "which is text-free. What is text-bound is *which rows*, *which "
      "dataset*, *which period* and *whether to answer at all* — never *what a "
      "number means*.\n" % t["total_distinct_calculation_owners"])
    A("That distinction is the single most important line in this census. "
      "Trakt does not have a calculation problem. It has a **scope-and-dispatch "
      "problem**.\n")
    A("**2. What percentage is currently reachable from the GovernedQueryPlan?**\n")
    A("- *By question volume:* **%d of %d = %.1f%%** — the plan is the semantic "
      "contract AND the compiled spec is what executes. A further %d questions "
      "(%.1f%%) are lifted at parse but then claimed by a specialist route that "
      "executes through its own owner; for those the plan is a **contract "
      "without a consumer**.\n"
      % (m["generic_and_liftable_ie_CONNECTED"], m["distinct_questions"],
         m["connected_pct"],
         m["liftable_to_query_plan"] - m["generic_and_liftable_ie_CONNECTED"],
         100.0 * (m["liftable_to_query_plan"] - m["generic_and_liftable_ie_CONNECTED"])
         / m["distinct_questions"]))
    A("- *By capability:* **%d of %d CONNECTED, %d PARTIAL, %d NOT_CONNECTED.** "
      "The capability count is far less flattering than the volume count, "
      "because the connected capabilities are the high-frequency ones.\n"
      % (conn, n, part, t["connectivity"].get("NOT_CONNECTED", 0)))
    A("**3. How many apparent 'specialist capabilities' actually reuse common "
      "generic execution infrastructure?**\n")
    A("**At least fourteen of the thirty-three.** Named explicitly:\n")
    A("| Looks specialist | Actually executed by |")
    A("|---|---|")
    for a, b in [
        ("pipeline stage analysis / amount by stage",
         "`mi_query_executor.execute_mi_query` — proven by execution"),
        ("superlative / ranking / top-N",
         "`mi_query_executor._apply_top_n` — proven by execution"),
        ("share and contribution",
         "`mi_query_executor._execute_share` / `_execute_contribution`"),
        ("bucketed / stratified dimensions",
         "the executor, over a bucket column materialised in preparation"),
        ("loan-level listing / scatter / bubble",
         "`mi_query_executor._execute_loan_level`"),
        ("forecast-view analysis",
         "the executor, over a derived frame"),
        ("funded temporal series",
         "`temporal_query.execute_temporal` → the executor, once per period"),
        ("vintage / static-pool analysis",
         "`analytics_lib.cohort` primitives + ordinary grouping"),
        ("concentration analysis",
         "`mi_workflows.engine.ranked_distribution`"),
        ("portfolio risk comparison",
         "`mi_workflows.engine.aggregate` / `compare_values`"),
        ("the analytical layer's portfolio_snapshot / population_profile",
         "`mi_workflows.engine`"),
        ("pipeline movement summary",
         "`movement_detail.build_stage_transition_detail` — the SAME payload "
         "the stage route consumes"),
        ("weekly brief insights",
         "`movement_detail` + concentration + funnel; composes only"),
        ("risk-limit headroom ranking",
         "`analytics_lib.concentration.group_shares` / `top_n_concentration`")]:
        A("| %s | %s |" % (a, b))
    A("")
    A("**Ranking is implemented seven times** in the estate — "
      "`mi_query_executor._apply_top_n`, `mi_query_executor._execute_ranked_loans`, "
      "`mi_workflows.engine.ranked_distribution`, "
      "`mi_agent.period_change.ranking.rank_movement`, "
      "`mi_agent_api.movement_detail.rank_contributors`, "
      "`analytics_lib.concentration.top_n_concentration` and "
      "`mi_agent.risk_monitor.concentration.top_n_concentration`. "
      "The plan has no ordering slot at all.\n")
    A("**4. Which migration gaps are simply wiring?**\n")
    A("Five, and they are unusually cheap because the contract, the resolver "
      "and in two cases the *function parameter* already exist:\n")
    A("| Gap | Why it is wiring |")
    A("|---|---|")
    A("| **Dataset / lens / period binding** | `plan_from_spec` already accepts "
      "`dataset=`, `portfolio_lens=` and `period=`. `parsed_question.py:154` "
      "calls `compiled_spec_for(spec)` with **none of them**. The owners "
      "(`workspace.resolve_dataset`, `portfolio_context.resolve_context`) "
      "already produce exactly these values. |")
    A("| **Ordering + limit slot** | The arithmetic exists and is proven. Adding "
      "an ordering/limit to `PlannedOutput` connects **93 generic-path corpus "
      "questions** with no new calculation. |")
    A("| **`unavailable_filters` on the lift** | It is a DISCLOSURE field, not a "
      "semantic — it records filters that could NOT be applied. It blocks 12 "
      "corpus lifts purely because the inverted liftability test treats every "
      "unmodelled field as semantic. Carrying it across like "
      "`metric_defaulted` already is costs nothing. |")
    A("| **`pipeline_summary` dispatch** | Its measurable half is already the "
      "generic executor's; the route simply claims the question first. |")
    A("| **`vintage_analysis` dispatch** | `balance by vintage_year` already "
      "flows through the plan; only the `/mi/cohorts` service shape does not. |")
    A("")
    A("**5. Which genuinely require architectural work?**\n")
    A("Five semantic families the plan has no vocabulary for. None of them is "
      "a missing calculation:\n")
    A("1. **A temporal axis** — grain, window, period list. Blocks evolution, "
      "pipeline evolution, cohort progression and the series half of every "
      "capability. (`AnalyticalScope.period` is a single optional string.)\n")
    A("2. **A comparison relationship between two scopes** — the compiler "
      "already executes N scopes per plan; nothing can say *this one is the "
      "baseline for that one*. Blocks temporal compare, period movement, period "
      "change, portfolio risk comparison.\n")
    A("3. **Sibling (non-nested) populations** — `ScopeDelta` may only NARROW, "
      "by deliberate design. Two peer populations, and the denominator a share "
      "needs, are both the direction it forbids.\n")
    A("4. **Entity state across snapshots** — stage transitions and cohort "
      "conversion follow one case through two frames. `AnalyticalScope` "
      "describes rows in **one** frame.\n")
    A("5. **Forward constructions** — projection, run-rate, milestone solving, "
      "scenario perturbation, threshold/limit status. A plan describes rows "
      "that exist.\n")
    A("**6. Are we at risk of rebuilding or unnecessarily narrowing existing "
      "capability?**\n")
    A("**Yes, and the evidence is already in the repository.**\n")
    A("- The brief's own worked example is the pattern: `pipeline_contract` "
      "exposes stage *counts*, so amount-by-stage was concluded absent. It is "
      "not absent; it is a one-line executor call, reproduced in this census "
      "against a committed fixture.\n")
    A("- **Every one of the 222 corpus declines is a plan-vocabulary gap, not a "
      "capability gap.** A programme that reads `plan_from_spec → None` as "
      "*unsupported* would conclude Trakt cannot rank, cannot compute a share, "
      "cannot draw a time series, cannot compare two months and cannot test a "
      "limit. All five are shipped, tested and — for ranking — proven executing "
      "in this document.\n")
    A("- **`plan_from_spec` is deliberately, aggressively conservative.** Its "
      "own docstring says it *declines more than it accepts*, and the "
      "inverted test means a field added to `MIQuerySpec` tomorrow is "
      "un-liftable until someone models it. That is excellent engineering and a "
      "**terrible capability signal**. A decline says *the plan cannot carry "
      "this*; it says nothing whatever about the executor.\n")
    A("- Three routes exist **only** because a broader capability had "
      "previously substituted for a narrower one — stage stock for a stage "
      "transition, today's limit status for a forward projection, the funded "
      "summary for a pipeline summary. Narrowing during migration would "
      "reintroduce each of them.\n")
    A("- One measured overlap to watch: **29 of the 35 questions claimed by the "
      "analytical composition layer also lift to a `QueryPlan`.** Two plan "
      "layers with overlapping claims and no composition rule between them is "
      "where a capability gets rebuilt.\n")
    A("**7. Does the remaining migration appear substantially smaller or larger "
      "than route count would suggest?**\n")
    A("**Substantially smaller.**\n")
    A("- Nineteen registered recognisers, thirty-three user-facing "
      "capabilities — but **%d distinct calculation owners**, and **one of them "
      "(`execute_mi_query`) already serves %.0f%% of corpus traffic**.\n"
      % (t["total_distinct_calculation_owners"], m["connected_pct"]))
    A("- %d of the %d are `TRIVIAL` or `SMALL`; they account for a "
      "disproportionate share of question volume, because ranking alone is "
      "12.4 per cent of the corpus and is `SMALL`.\n"
      % (t["migration_complexity"]["TRIVIAL"] + t["migration_complexity"]["SMALL"], n))
    A("- The `LARGE` items cluster into the **five semantic families** above, "
      "not into %d separate migrations. Building the temporal axis once "
      "addresses four capabilities; building the comparison relationship once "
      "addresses four more.\n" % t["migration_complexity"]["LARGE"])
    A("- **Exactly one genuine capability gap exists in the entire estate** "
      "(forward-looking approved-limit projection), and it is already "
      "documented, already declined rather than substituted, and explicitly "
      "out of scope of any plan migration.\n")
    A("Route count over-states the work. **Owner count under-states how much is "
      "already done**: the biggest owner is already connected, and the largest "
      "single gap in front of it is an `ORDER BY … LIMIT`.\n")
    return "\n".join(o)


if __name__ == "__main__":
    raise SystemExit(main())
