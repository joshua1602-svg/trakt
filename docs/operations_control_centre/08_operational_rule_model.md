# 08 — Operational rule model

Every approval becomes a governed rule: scoped, versioned, auditable, and
automatically applied to future work. Rules live in
`operations-control/…/rules/` (doc 06) and are **projected** into the
artefacts existing agents already read — agents are never modified.

## 1. Rule record

```jsonc
{
  "schema_version": "1.0.0",
  "rule_id": "rule_…",                 // stable across versions
  "version": 3,                        // int, monotonic; prior versions immutable
  "kind": "field_mapping | alias | enum | transformation | portfolio_rule |
           client_rule | exception | reporting_assumption",
  "scope": {
    "level": "file | portfolio | client | global",
    "client_id": "…",                  // null for global
    "portfolio_id": "…",               // null unless portfolio/file
    "file_ref": "sha256:…"             // only for file scope
  },
  "status": "active | superseded | retired",
  "payload": { },                      // kind-specific, see §3
  "provenance": {
    "suggested_by": "llm | deterministic | memory | operator",
    "confidence": 0.93,                // if suggested
    "decision_id": "dec_…",            // the approval that created this version
    "workflow_id": "wf_…"
  },
  "approval": { "approved_by": "…", "approved_at": "…", "reason": "…" },
  "effective_from": "…",               // approval time
  "superseded_by": null,               // set on the old version when a new one lands
  "description": "Map the file column 'Prop_Val_Idx' to the indexed property valuation."
}
```

## 2. Scope semantics and precedence

Operator chooses scope at approval time (doc 02 §2.5). Resolution order when
applying rules to a run — most specific wins:

```
file  >  portfolio  >  client  >  global
```

- **Current file** — one-off; applies to this delivery only (e.g. a known
  bad value in one month's tape). Never auto-applied to future files.
- **Current portfolio** — applies to every future delivery of this portfolio.
- **Current client** — applies to all portfolios of the client, including
  future acquired books (drives the "new portfolio reuses client rules" flow).
- **Global Trakt Registry** — applies to every client. Reserved for genuinely
  universal facts (e.g. a common alias). UI warns before selecting.

Conflicts at the same level are impossible by construction: approving a new
value for an existing (kind, scope, subject) creates **version n+1** of the
same rule, superseding n.

## 3. Kind-specific payloads and projection targets

The **rule projector** (`operations_control/rules/projector.py`) translates an
approved rule into the sink the relevant agent already reads, using existing
persistence functions only:

| Kind | Payload (essence) | Projection target (existing, unchanged) |
|---|---|---|
| `field_mapping` | `{source_column, canonical_field}` | Client memory `mapping_memory.yaml` (client/portfolio scope); `34_*_approved.yaml` for the live run; global → controlled `aliases_pipeline.yaml` via `mapping_persistence` sink B |
| `alias` | `{alias, canonical_field}` | Same sinks as field_mapping |
| `enum` | `{field, source_value, canonical_value}` | Client `enum_memory.yaml`; global → `enum_synonyms_confirmed.yaml` |
| `transformation` | `{field, treatment, parameters}` | Onboarding decision artefacts (`34_*`) consumed on rerun |
| `portfolio_rule` / `client_rule` | `{subject, setting, value}` (e.g. day-first dates, product default) | Client memory / client config overlay artefacts consumed by the orchestrator context |
| `exception` | `{rule_id_ref, condition, disposition, justification}` | Validation exception artefacts / remediation ledger, re-read on rerun |
| `reporting_assumption` | `{assumption, value, applies_to}` | Assembly/reporting context artefacts |

Invariants:
- New **regulatory canonical fields** are never created by rule projection —
  that remains the human-governed `fields_registry_pipeline.yaml` path.
- The core `config/system/fields_registry.yaml` is never written.
- Projection is deterministic and re-runnable: re-projecting the active rule
  set is idempotent.

## 4. Rule application at run time

1. Workflow start → engine resolves the applicable rule set (scope precedence
   above) and snapshots the resolved set as `{workflow_id}/rule_set.json`.
2. Projection materialises that set into the run's working artefacts before
   agents execute.
3. Publication records the exact `rule_versions` used (doc 06 §3), so any
   published report can name the rules it was built with, and any rule change
   is traceable to which publications preceded/followed it.

## 5. Review philosophy enforcement

- A delivery item matching an **active rule** is auto-applied and counted as
  "automatic" — never re-surfaced.
- An item that **conflicts** with an active rule (value changed at source)
  raises a review item flagged "changed since your approval", offering:
  keep the rule (treat file as exception) / update the rule (new version) /
  file-scope override.
- Genuinely **new** items (no rule at any scope) surface as normal decisions.

## 6. Rules Library queries (backing doc 02 §2.6)

Supported: by kind, scope, client, status, free text over `description`;
per-rule version history with the approving decision and operator; reverse
lookup "which publications used this rule version" via publication records.
