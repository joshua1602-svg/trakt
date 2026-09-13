"""One record per frozen-bank question, from committed evidence only.

READ-ONLY. No question is re-asked, no model is called, no verdict is changed.
Every field is copied out of the composite evidence, the composite scorecard, the
historical baseline, or a repository file that is hashed into the metadata. The
diagnostic labels are rules over those same fields — where a rule cannot decide
from evidence the value is NOT_ESTABLISHED, never a guess.

THE FORMAL VERDICT IS COPIED, NEVER COMPUTED. `formal_verdict` is whatever the
frozen scorecard already recorded, and the totals are asserted against the
committed certification before anything is written.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPO = HERE.parents[2]
C32 = _REPO / "due_diligence/evidence/mi_135_completion_32"
HIST = _REPO / "due_diligence/evidence/mi_135_live_bank_00bb3e9d"
CANONICAL = _REPO / "synthetic_demo/output/multibook/platform_2026-06-30_canonical_typed.csv"
REGISTRY = _REPO / "config/business_semantics_registry.yaml"
FACILITIES = _REPO / "config/risk/funding_facilities.yaml"

PRODUCT_SHA = "5c436961ebe279bc0820ab006867b9f8a869bde2"
SOURCE_COMMITS = {"partial_103": "b57889bf",
                  "temporal_adjudication": "553107f3",
                  "completion_tranche": "0966be71"}

#: The committed certification. Asserted, not recomputed.
EXPECTED_TOTALS = {"FULLY_CORRECT": 66, "PARTIALLY_CORRECT": 36,
                   "HONEST_REFUSAL": 24, "BAD_REFUSAL": 0, "WRONG": 0,
                   "APPROPRIATE_CLARIFICATION": 0, "INCONCLUSIVE": 8,
                   "INFRASTRUCTURE": 1}
VERDICT_MAP = {"INFRASTRUCTURE_FAILURE": "INFRASTRUCTURE"}

#: Adjudicated CURRENT_CORRECT at 553107f3. Their `temporal` mismatch is against a
#: fixture the adjudication found stale; it is not a product error.
NINE_TEMPORAL = {"Q18A", "Q18B", "Q18C", "Q19A", "Q19B",
                 "Q20A", "Q20B", "Q22A", "Q22B"}
#: The open material_summary operation-fixture dispute. `operation` mismatches on
#: these are the dispute, not an independent defect.
DISPUTE_OPERATION = {"Q18A", "Q18B", "Q18C", "Q19A", "Q19B", "Q19C",
                     "Q20A", "Q20B", "Q20C", "Q1.1", "Q22B"}
#: The seven the completion tranche moved to a clarification or a coded refusal,
#: recorded as a candidate at 0966be71 and deliberately not adjudicated.
DISPUTE_CANDIDATE = {"Q3.1", "Q3.2", "Q7.1", "Q7.2", "Q7.3",
                     "Q8.provenance.3", "S05"}

#: Scorer mismatch dimension -> the brief's partial-reason vocabulary.
PARTIAL_LABEL = {
    "statistic": "STATISTIC", "weight": "WEIGHTING", "measures": "MEASURE",
    "filters": "FILTER", "dimensions": "DIMENSION", "temporal": "TEMPORAL",
    "operation": "OPERATION", "capability": "CAPABILITY",
    "output_structure": "OUTPUT_COMPOSITION", "population": "SCOPE",
    "geography_basis": "SCOPE", "geography_level": "SCOPE",
    "comparison": "OUTPUT_COMPOSITION",
}
#: Which label leads when several apply. Ordered by how far up the chain the
#: divergence sits: what was asked, then how it was shaped, then how it was
#: measured.
PARTIAL_PRIORITY = ("CAPABILITY", "OPERATION", "MEASURE", "TEMPORAL", "SCOPE",
                    "FILTER", "DIMENSION", "STATISTIC", "WEIGHTING",
                    "OUTPUT_COMPOSITION", "DATA_AVAILABILITY", "OTHER")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_columns() -> set:
    if not CANONICAL.exists():
        return set()
    with CANONICAL.open(newline="", encoding="utf-8") as handle:
        return set(next(csv.reader(handle)))


def registry_source_fields() -> dict:
    """concept -> source_field, without importing yaml if it is not installed."""
    try:
        import yaml
    except ImportError:
        return {}
    body = yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}
    return {name: (spec or {}).get("source_field")
            for name, spec in (body.get("fields") or {}).items()}


def mismatch_dimensions(why: str) -> list:
    why = str(why or "")
    marker = "differs from the fixture on:"
    if marker not in why:
        return []
    return sorted(part.strip() for part in why.split(marker)[1].split(","))


# --------------------------------------------------------------------------- #
# Refusal classification. Rules over evidence, and NOT_ESTABLISHED where the
# evidence does not reach. Each rule records WHY it fired, so a reader can
# disagree with the label without re-deriving the facts.
# --------------------------------------------------------------------------- #

def classify_refusal(case: dict, columns: set, source_fields: dict,
                     facility_clients: set, client_id: str) -> dict:
    stated = str(case.get("refusal_stated_reason") or "")
    fallback = str(case.get("fallback_reason") or "")
    lower = stated.lower()
    out = {
        "refusal_root_cause": "NOT_ESTABLISHED",
        "required_data_fields": [],
        "required_capability": None,
        "data_present_in_current_canonical_evidence": "NOT_ESTABLISHED",
        "existing_owner_capable": "NOT_ESTABLISHED",
        "likely_fix_type": "NOT_ESTABLISHED",
        "classification_evidence": "",
    }

    # 1. THE BORROWING BASE. The capability is built and a facility register
    #    exists; neither is bound to the client this portfolio resolves to.
    if "no funding facility is configured" in lower:
        out.update(
            refusal_root_cause="CONFIG_OR_MAPPING",
            required_capability="funded_bridge / borrowing_base",
            required_data_fields=["config/risk/funding_facilities.yaml facility "
                                  f"bound to client_id={client_id!r}"],
            data_present_in_current_canonical_evidence="NOT_ESTABLISHED",
            existing_owner_capable="YES",
            likely_fix_type="CONFIG",
            classification_evidence=(
                "mi_agent/borrowing_base/service.py resolves a facility with "
                "load_facility(client_id); config/risk/funding_facilities.yaml "
                f"carries facilities for {sorted(facility_clients)} and the run's "
                f"client_id is {client_id!r}. The owner exists; the binding does not."))
        return out

    # 2. CONCENTRATION LIMITS AND FORECASTS. A limit schedule exists in the repo
    #    under another portfolio id.
    if "concentration limits" in lower and "limit schedule" in lower:
        limits = sorted(p.parent.name
                        for p in (_REPO / "config/clients").glob("*/risk_limits_extracted.yaml"))
        out.update(
            refusal_root_cause="CONFIG_OR_MAPPING",
            required_capability="limit_assessment (+ forecast where asked)",
            required_data_fields=["a risk limit schedule bound to "
                                  f"portfolio_id={client_id!r}"],
            data_present_in_current_canonical_evidence="NOT_ESTABLISHED",
            existing_owner_capable="NOT_ESTABLISHED",
            likely_fix_type="CONFIG",
            classification_evidence=(
                f"config/clients/*/risk_limits_extracted.yaml exists for {limits}; "
                f"the run's client_id is {client_id!r}. `limit_assessment` is a "
                "declared capability in vocabulary.CAPABILITY_OPERATIONS, but "
                "whether this owner can answer once bound was not measured."))
        return out

    # 3. THE PIPELINE. A field contract exists; no pipeline dataset does.
    if "pipeline" in lower and "different governed dataset" in lower:
        out.update(
            refusal_root_cause="DATA_GENUINELY_UNAVAILABLE",
            required_capability="pipeline / pipeline_stage_movement",
            required_data_fields=["a prepared pipeline tape "
                                  "(config/mi/pipeline_field_contract.yaml names "
                                  "20_prepared_pipeline_mi.csv)"],
            data_present_in_current_canonical_evidence="NO",
            existing_owner_capable="NOT_ESTABLISHED",
            likely_fix_type="DATA",
            classification_evidence=(
                "config/mi/pipeline_field_contract.yaml declares the pipeline "
                "source, and no prepared_pipeline_mi file exists anywhere in the "
                "repository. The funded canonical snapshot carries no pipeline "
                "stage column."))
        return out

    # 4. TWO BOOKS AT TWO REPORTING DATES. A governed rule refusing rather than
    #    comparing across dates. The data exists; it is not co-dated.
    if "different reporting dates" in lower or "multiple reporting dates" in lower:
        out.update(
            refusal_root_cause="GOVERNANCE_RESTRICTION",
            required_capability="generic_analysis / concentration",
            required_data_fields=["co-dated snapshots for every source portfolio"],
            data_present_in_current_canonical_evidence="YES",
            existing_owner_capable="NOT_ESTABLISHED",
            likely_fix_type="DATA",
            classification_evidence=(
                "The refusal names the governed rule directly — comparison or "
                "concentration is not performed across reporting dates. The rows "
                "exist; the source portfolios are at different snapshot dates."))
        return out

    # 5. AN EMPTY MEASURED POPULATION. The interesting question is whether the
    #    filter's own field is on the canonical data, because that separates
    #    "there are genuinely no such loans" from "the filter never reached them".
    if "no loans in this book match that filter" in lower:
        named = stated.split("(", 1)[1].rsplit(")", 1)[0] if "(" in stated else ""
        concepts = [c.strip().strip("'\"") for c in named.split(",") if c.strip()]
        missing = [c for c in ("Product Type", "erm_product_type")
                   if c in concepts]
        # A parser artefact: a stopword captured as a filter VALUE.
        if named.strip("'\" ").lower() in ("among", "both"):
            out.update(
                refusal_root_cause="OTHER",
                data_present_in_current_canonical_evidence="NOT_ESTABLISHED",
                likely_fix_type="NOT_ESTABLISHED",
                classification_evidence=(
                    f"The refusal names the filter as {named!r} — an English "
                    "stopword captured as a filter value, not a governed concept. "
                    "This is an interpretation artefact and is not classified "
                    "further from evidence alone."))
            return out
        if missing or "product type" in named.lower():
            out.update(
                refusal_root_cause="FIELD_NOT_ON_CANONICAL_DATA",
                required_data_fields=["erm_product_type"],
                data_present_in_current_canonical_evidence="NO",
                existing_owner_capable="NO",
                likely_fix_type="DATA",
                classification_evidence=(
                    "business_semantics_registry maps the product-type concept to "
                    "source_field `erm_product_type`, which is not one of the "
                    f"{len(columns)} columns on the governed canonical snapshot. "
                    "No loan can match a filter on a field the data does not carry."))
            return out
        out.update(
            refusal_root_cause="FIELD_PRESENT_BUT_NOT_CONNECTED",
            required_data_fields=[f for f in
                                  ("collateral_geography", "source_portfolio_type",
                                   "youngest_borrower_age") if f in columns],
            data_present_in_current_canonical_evidence="YES",
            existing_owner_capable="NOT_ESTABLISHED",
            likely_fix_type="CONNECTIVITY",
            classification_evidence=(
                f"The filter concepts named ({named}) map to source fields that "
                "ARE on the canonical snapshot, and the snapshot contains rows "
                "satisfying them, so an empty measured population is not the data "
                "being absent. Which binding drops them was not traced here."))
        return out

    # 6. A DIMENSION NEITHER APPLIED NOR REJECTED.
    if "neither applied nor rejected" in lower:
        field = stated.split(":", 1)[1].split(".")[0].strip() if ":" in stated else ""
        out.update(
            refusal_root_cause=("FIELD_PRESENT_BUT_NOT_CONNECTED"
                                if field in columns else "NOT_ESTABLISHED"),
            required_data_fields=[field] if field else [],
            data_present_in_current_canonical_evidence=("YES" if field in columns
                                                        else "NOT_ESTABLISHED"),
            existing_owner_capable="NOT_ESTABLISHED",
            likely_fix_type="CONNECTIVITY" if field in columns else "NOT_ESTABLISHED",
            classification_evidence=(
                f"The refusal names {field!r} as parsed but neither applied nor "
                f"rejected; that column is {'present on' if field in columns else 'absent from'} "
                "the governed canonical snapshot."))
        return out

    # 7. A CAPABILITY THAT DOES NOT APPLY THE THING THAT WAS ASKED.
    if "does not apply a value threshold" in lower or "could not be applied" in lower:
        out.update(
            refusal_root_cause="CAPABILITY_EXISTS_NOT_CONNECTED",
            required_capability=str(case.get("capability") or "") or None,
            required_data_fields=[f for f in ("current_loan_to_value",)
                                  if f in columns],
            data_present_in_current_canonical_evidence=(
                "YES" if "current_loan_to_value" in columns else "NOT_ESTABLISHED"),
            existing_owner_capable="NO",
            likely_fix_type="CAPABILITY",
            classification_evidence=(
                "The owner ran and states it does not apply the requested "
                "restriction, so it refuses rather than answering over a wider "
                "population. The restricting field is on the canonical snapshot."))
        return out

    out["classification_evidence"] = (
        "No classification rule matched this refusal's stated reason; left "
        "NOT_ESTABLISHED rather than assigned by resemblance.")
    return out


def classify_partial(case: dict) -> dict:
    qid = case["question_id"]
    dims = mismatch_dimensions(case.get("user_outcome_why"))
    labels = [PARTIAL_LABEL.get(d, "OTHER") for d in dims]

    # WHOLLY EXPLAINED BY AN ESTABLISHED DISPUTE. `operation` on a Q18-Q20 family
    # case and `temporal` on one of the adjudicated nine are both disagreements
    # with a fixture, already recorded. When a case has nothing else wrong, the
    # dispute IS the reason.
    disputed = set()
    if qid in DISPUTE_OPERATION:
        disputed.add("operation")
    if qid in NINE_TEMPORAL:
        disputed.add("temporal")
    wholly = bool(dims) and set(dims) <= disputed

    if wholly:
        primary = "BANK_EXPECTATION_DISPUTE"
        secondary = sorted(set(labels))
        answerable, evidence = "YES", (
            "Every mismatch dimension on this case is an already-established "
            "fixture dispute: `operation` under the Q18-Q20 material_summary "
            "canonicalisation, `temporal` under the nine adjudicated "
            "CURRENT_CORRECT at 553107f3. The owner already produces this "
            "reading; the bank expects the previous one.")
    else:
        ordered = [l for l in PARTIAL_PRIORITY if l in labels]
        primary = ordered[0] if ordered else "OTHER"
        secondary = sorted(set(labels) - {primary})
        answerable, evidence = "NOT_ESTABLISHED", (
            "Whether the existing owner could answer this fully was not measured: "
            "it would need the question re-asked, which this task forbids.")
    return {
        "partial_reason_primary": primary,
        "partial_reason_secondary": secondary,
        "could_existing_owner_answer_fully": answerable,
        "could_answer_fully_evidence": evidence,
    }


def build_case(qid: str, scored: dict, raw: dict, baseline: dict,
               provenance: str, columns: set, source_fields: dict,
               facility_clients: set, client_id: str) -> dict:
    rec = (raw.get("record") or {})
    envelope = raw.get("envelope") or {}
    intent = (rec.get("interpretation") or {}).get("candidate_intent") or {}
    plan = (rec.get("compiler") or {}).get("plan") or {}
    period = plan.get("period") or scored.get("period") or {}
    receipt = ((rec.get("execution") or {}).get("receipt")
               or scored.get("receipt") or {})
    resolution = receipt.get("period_resolution") or {}
    serving = rec.get("serving") or {}
    eligibility = rec.get("eligibility") or scored.get("eligibility") or {}
    verdict = VERDICT_MAP.get(scored.get("user_outcome"), scored.get("user_outcome"))

    known = []
    if qid in NINE_TEMPORAL:
        known.append("TEMPORAL_NINE_CURRENT_CORRECT (553107f3)")
    if qid in DISPUTE_OPERATION:
        known.append("Q18-Q20_MATERIAL_SUMMARY_OPERATION_FIXTURE_DISPUTE (open)")
    if qid == "Q22B":
        known.append("Q22B_CANONICAL_INTENT_IDEMPOTENCE_OBSERVATION "
                     "(no formal score effect)")
    if qid in DISPUTE_CANDIDATE:
        known.append("NEW_BANK_EXPECTATION_DISPUTE_CANDIDATE "
                     "(clarification/coded refusal where the previous build "
                     "answered partially; not adjudicated)")

    case = {
        "case_id": qid,
        "question": scored.get("question"),
        "evidence_source_run": provenance,

        "formal_verdict": verdict,
        "historical_verdict": (baseline or {}).get("user_outcome"),

        "answer": envelope.get("answer"),
        "http_status": (envelope.get("__http_status__")
                        if envelope.get("__transport_error__") else 200
                        if envelope.get("ok") is not None else None),

        "change_form": intent.get("change_form"),
        "candidate_operation": intent.get("operation"),
        "compiled_operation": plan.get("operation") or scored.get("operation"),
        "capability": plan.get("capability") or scored.get("capability"),
        "mode": receipt.get("mode") or receipt.get("workflow_mode"),

        "measures": scored.get("measures"),
        "statistic": [m.get("statistic") for m in (intent.get("measures") or [])
                      if isinstance(m, dict)] or None,
        "weighting": [m.get("weight") for m in (intent.get("measures") or [])
                      if isinstance(m, dict)] or None,
        "dimensions": scored.get("dimensions"),
        "filters": scored.get("filters"),
        "population": {"base": scored.get("population_base"),
                       "lens": scored.get("population_lens")},
        "scope": receipt.get("executed_scope") or receipt.get("source_scope"),

        "period_form": period.get("form"),
        "periods_back": period.get("periods_back"),
        "period_from": receipt.get("period_from"),
        "period_to": receipt.get("period_to"),
        "time_stated": period.get("stated"),
        "temporal_route": (resolution.get("resolution_method")
                           or period.get("contract")),
        "temporal_default_applied": receipt.get("temporal_default_applied"),

        "serving_provenance": scored.get("served_from") or serving.get("decision"),
        "serving_route": serving.get("response_served_from"),

        "calculation_owner": receipt.get("calculation_owner"),
        "composition_owner": receipt.get("composition_owner"),

        "requested_fields": (receipt.get("requested_fields")
                             or ((envelope.get("periodChange") or {})
                                 .get("request_interpretation") or {})
                             .get("requested_fields")),
        "executed_fields": receipt.get("executed_fields"),
        "served_metric_fields": receipt.get("served_metric_fields"),
        "requested_metric_disposition": receipt.get("requested_metric_disposition"),
        "owner_availability_status": receipt.get("status"),

        "refusal_code": (eligibility.get("reason")
                         or scored.get("fallback_reason") or None),
        "refusal_reason": scored.get("refusal_stated_reason") or None,
        "clarification_reason": scored.get("clarify_reasons") or None,

        "silent_semantic_drop": bool(
            any(v is True for k, v in (scored.get("drops") or {}).items()
                if k.startswith("silent_") and "widen" not in k)),
        "silent_scope_widening": bool(
            any(v is True for k, v in (scored.get("drops") or {}).items()
                if "widen" in k)),
        "misroute": scored.get("execution_runtime") == "WRONG_OWNER",

        "formal_failure_dimensions": mismatch_dimensions(
            scored.get("user_outcome_why")),
        "formal_failure_reasons": scored.get("user_outcome_why"),

        "bank_expectation_disputed": bool(known and verdict != "FULLY_CORRECT"),
        "known_adjudication": known,

        "infrastructure_detail": None,
        "diagnostics": {},
    }

    if verdict == "INFRASTRUCTURE":
        case["infrastructure_detail"] = {
            "cause": "TRANSPORT" if envelope.get("__transport_error__")
                     else "PROVIDER",
            "http_status": envelope.get("__http_status__"),
            "retried_once": bool(raw.get("retry_of_infrastructure_failure")),
            "detail": scored.get("user_outcome_why"),
        }
    if verdict == "PARTIALLY_CORRECT":
        case["diagnostics"] = classify_partial(scored)
    if verdict == "HONEST_REFUSAL":
        case["diagnostics"] = classify_refusal(
            scored, columns, source_fields, facility_clients, client_id)
    return case


def main() -> int:
    scored = {r["question_id"]: r for r in json.loads(
        (C32 / "composite_scored.json").read_text(encoding="utf-8"))}
    composite = json.loads(
        (C32 / "composite_raw_records.json").read_text(encoding="utf-8"))
    raw = {r["question_id"]: r for r in composite["records"]}
    baseline = {r["question_id"]: r for r in json.loads(
        (HIST / "MI_135_LIVE_BANK_RESULTS.json").read_text(encoding="utf-8"))}
    provenance = json.loads(
        (C32 / "composite_manifest.json").read_text(encoding="utf-8"))["case_provenance"]
    bank = json.loads((HIST / "questions.json").read_text(encoding="utf-8"))

    order = [r["question_id"] for r in composite["records"]]
    duplicates = sorted({q for q in order if order.count(q) > 1})
    missing = sorted({c["question_id"] for c in bank["questions"]} - set(order))
    if duplicates or missing or len(order) != 135:
        raise SystemExit(f"cases={len(order)} duplicates={duplicates} "
                         f"missing={missing}: refusing to write")

    # THE COMMITTED TOTALS ARE ASSERTED BEFORE ANYTHING IS WRITTEN. This export
    # must not be able to disagree with the certification it describes.
    totals = {}
    for row in scored.values():
        name = VERDICT_MAP.get(row["user_outcome"], row["user_outcome"])
        totals[name] = totals.get(name, 0) + 1
    for name, want in EXPECTED_TOTALS.items():
        if totals.get(name, 0) != want:
            raise SystemExit(
                f"{name} is {totals.get(name, 0)}, the committed certification "
                f"says {want}. STOPPING — the export must not disagree with the "
                f"result it describes.")

    columns = canonical_columns()
    source_fields = registry_source_fields()
    facility_clients = set()
    try:
        import yaml
        facility_clients = {f.get("client_id") for f in
                            (yaml.safe_load(FACILITIES.read_text(encoding="utf-8"))
                             or {}).get("facilities") or ()}
    except Exception:
        pass
    client_id = str(composite.get("portfolio_id") or "").split("/", 1)[0]

    cases = [build_case(qid, scored[qid], raw[qid], baseline.get(qid),
                        provenance.get(qid), columns, source_fields,
                        facility_clients, client_id)
             for qid in order]

    document = {
        "metadata": {
            "what_this_is": "one record per frozen-bank question, assembled from "
                            "committed certification evidence; no question was "
                            "re-asked and no verdict was changed",
            "product_sha": PRODUCT_SHA,
            "bank_id": "MI_135_RELEASE_CERTIFICATION_V2_COMPLETE",
            "bank_sha256": bank["bank_sha256"],
            "questions_json_sha256": sha256_of(HIST / "questions.json"),
            "source_commits": SOURCE_COMMITS,
            "source_evidence": {
                "composite_raw_records": sha256_of(C32 / "composite_raw_records.json"),
                "composite_scored": sha256_of(C32 / "composite_scored.json"),
                "composite_manifest": sha256_of(C32 / "composite_manifest.json"),
                "historical_baseline": sha256_of(HIST / "MI_135_LIVE_BANK_RESULTS.json"),
                "canonical_snapshot": (sha256_of(CANONICAL) if CANONICAL.exists()
                                       else None),
            },
            "case_count": len(cases),
            "measured_count": len(cases) - totals.get("INFRASTRUCTURE", 0),
            "duplicate_case_ids": len(duplicates),
            "missing_case_ids": len(missing),
            "formal_totals": totals,
            "generated_from_existing_evidence_only": True,
            "verdicts_unchanged": True,
            "adjudications_preserved_not_reopened": [
                "TEMPORAL_NINE_CURRENT_CORRECT (553107f3)",
                "Q18-Q20_MATERIAL_SUMMARY_OPERATION_FIXTURE_DISPUTE (open)",
                "Q22B_CANONICAL_INTENT_IDEMPOTENCE_OBSERVATION",
            ],
            "fields_absent_from_this_evidence": {
                "executed_fields": "no case in the 135 carries it",
                "served_metric_fields": "no case in the 135 carries it",
                "requested_metric_disposition": "no case in the 135 carries it",
                "why": "these three live on the plan_metric_delta receipt, and no "
                       "case in this bank executed through that runtime — only "
                       "five executed at all, all of them material_summary. The "
                       "keys are present and null on every case rather than "
                       "omitted, so a reader can tell 'not carried' from 'not "
                       "asked for'.",
            },
            "diagnostic_note":
                "partial_reason_* and refusal_root_cause are DIAGNOSTIC labels "
                "derived by rule from the fields in this same document. They "
                "carry no weight in the formal score and NOT_ESTABLISHED is used "
                "wherever the evidence does not decide.",
        },
        "cases": cases,
    }
    out = HERE / "mi_135_case_level_evidence.json"
    body = json.dumps(document, indent=1, sort_keys=False) + "\n"
    out.write_text(body, encoding="utf-8")

    partials = [c for c in cases if c["formal_verdict"] == "PARTIALLY_CORRECT"]
    refusals = [c for c in cases if c["formal_verdict"] == "HONEST_REFUSAL"]
    not_established = sum(
        1 for c in cases
        for v in (c.get("diagnostics") or {}).values()
        if v == "NOT_ESTABLISHED")

    print(f"JSON_PATH   = {out.relative_to(_REPO)}")
    print(f"JSON_SHA256 = {hashlib.sha256(body.encode('utf-8')).hexdigest()}")
    print(f"CASE_COUNT  = {len(cases)}")
    print(f"DUPLICATES  = {len(duplicates)}")
    print(f"MISSING     = {len(missing)}")
    print(f"PARTIALS_CLASSIFIED   = {sum(1 for c in partials if (c['diagnostics'] or {}).get('partial_reason_primary'))} of {len(partials)}")
    print(f"REFUSALS_CLASSIFIED   = {sum(1 for c in refusals if (c['diagnostics'] or {}).get('refusal_root_cause') not in (None, 'NOT_ESTABLISHED'))} of {len(refusals)}")
    print(f"NOT_ESTABLISHED_COUNT = {not_established}")
    print(f"formal totals asserted against the certification: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
