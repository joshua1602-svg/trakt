# 07 — Governed Agent Result contract

Every stage of every workflow returns the same operational contract to the
Control Centre. It is produced by the governed adapters in
`operations_control/adapters/` — **agents are not modified**; the adapter
derives the contract from each agent's existing `StepResult` and numbered
manifest.

It is deliberately distinct from `trakt_core.envelope.GovernedResult` (the API
transport envelope, statuses `success/partial_success/blocked/error`). A
Governed Agent Result is an *operational* object; when served over HTTP it
travels as the `result` payload inside a `GovernedResult` envelope.

## 1. Shape

```jsonc
{
  "schema_version": "1.0.0",
  "result_id": "gar_...",                 // stable id, persisted
  "workflow_id": "wf_...",
  "stage": "mapping",                     // operator-facing stage key (doc 04)
  "agent": "onboarding",                  // internal; never rendered
  "status": "needs_review",               // see §2
  "summary": "Most fields were matched automatically. Three mappings require your confirmation.",
  "why_it_matters": "These fields feed the regulatory report. Confirming them ensures the figures are attributed correctly.",
  "decisions_required": [                 // exactly what the operator must decide; may be empty
    {
      "decision_id": "dec_...",
      "kind": "field_mapping",            // field_mapping | alias | enum | transformation |
                                          // validation_exception | reconciliation | publication | question
      "title": "Confirm where 'Prop_Val_Idx' belongs",
      "question": "The file column 'Prop_Val_Idx' looks like the indexed property valuation. Is that right?",
      "recommendation": {                 // optional; provenance always shown
        "source": "llm | deterministic | memory",
        "value": "current_valuation_indexed",
        "confidence": 0.93
      },
      "options": [ {"value": "...", "label": "..."} ],
      "allowed_scopes": ["file", "portfolio", "client", "global"],
      "default_scope": "portfolio",
      "blocking": true
    }
  ],
  "evidence": [                           // collapsed by default in the UI
    {
      "label": "Sample values from your file",
      "kind": "table | text | metric",
      "data": { }                         // pre-rendered, plain-language content only
    }
  ],
  "counts": { "automatic": 118, "needs_review": 3, "blocked": 0 },
  "provenance": {                         // for audit/drill-down; never rendered raw
    "manifest_path": "…/30_transformation_manifest.json",
    "run_state_path": "…/run_state.json",
    "artefacts": ["…"]
  },
  "audit": { "created_at": "…", "created_by": "system|operator-id", "correlation_id": "…" }
}
```

## 2. Status vocabulary

| Status | Meaning for the operator | Derived from |
|---|---|---|
| `ready` | Stage can run / is running; nothing needed from you | Step pending/running; upstream readiness flags true |
| `needs_review` | The stage finished but needs your decision(s) | `StepResult.blocking=false` with open decisions, or onboarding `NEEDS_CONFIRMATION / NEEDS_CONFIGURATION`; manifest `review` counts > 0 |
| `blocked` | Trakt cannot continue until something is resolved | `StepResult.blocking=true`, onboarding `BLOCKED/FAILED`, `HandoffValidationError`-class refusals, orchestrator exit 3 |
| `approved` | Your decision has been recorded | Operator decision persisted; rerun not yet complete |
| `rejected` | You declined; the stage will not proceed as proposed | Operator rejection persisted |
| `completed` | Stage finished; nothing outstanding | Readiness flag for the next stage true, zero open decisions |

## 3. Derivation per agent (adapter mapping)

| Stage (operator name) | Agent + manifest | Status derivation |
|---|---|---|
| Understanding data / Mapping | `onboarding_agent` → `24_*`, `33_*`, `34_*`, `40_operator_workflow_summary.json` | `derive_status()` result mapped: `READY→completed`, `NEEDS_CONFIRMATION/NEEDS_CONFIGURATION→needs_review`, `BLOCKED/FAILED→blocked`; decisions from `34_target_first_decisions.yaml` + mapping review queue groups |
| Transformation | `transformation_agent` → `30_transformation_manifest.json`, `33_transformation_readiness.*` | `ready_for_validation` true & no blocking issues → `completed`; blocking issue count > 0 → `blocked`; advisory issues → `needs_review` |
| Validation | `validation_agent` → `40_validation_manifest.json`, `42_validation_readiness.json` | `ready_for_projection` → `completed`; blocking validation count → `blocked`; new/material warnings → `needs_review` |
| Projection | `projection_agent` → `50_projection_manifest.json`, `53_projection_readiness.json` | `ready_for_delivery_normalisation` → `completed`; blocker resolution rows open → `blocked` |
| Assembly | `assembler_agent` / `platform_assembler` manifest | assembly manifest complete → `completed` |
| Publication | delivery agent `60_*` + publication approval record | XML/XSD valid + assembly done → `needs_review` (publication decision); operator approval → `approved` then `completed` on promote |

## 4. Plain-language rules

The adapter layer owns translation. Enforced by contract tests:

- `summary` ≤ 2 sentences; `why_it_matters` is business language only.
- Forbidden in any rendered field: file paths, blob URIs, container names,
  Python exception text, schema IDs, regime codes (e.g. `RREL69`), JSON dumps.
  Technical detail lives only under `provenance` (not rendered) and inside
  `evidence` entries that have been explicitly humanised.
- Every `decisions_required` item must be answerable without opening any other
  screen: the question, the recommendation with its provenance, and the options
  are self-contained.
- Counts follow the review philosophy: items already approved and unchanged are
  counted under `automatic`, never re-surfaced.
