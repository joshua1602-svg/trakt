# Gap analysis — the first implementation against the corrected requirement

Traced against the code on `claude/client-onboarding-occ-oe3zzt` at `8ad667a`,
not against the implementation report.

---

## Summary

The core is less broken than the framing suggests, and the framing is more wrong
than the tests suggested. A brand-new client **could** already be onboarded end
to end with no legacy configuration — I verified it empirically against a fresh
store with a client Trakt has never seen:

```
blank draft origin: new | client_id: ''
ready with no legacy config: True
approve: 200 version 1
   wrote: client_config create
   wrote: portfolio_registry create
   wrote: source_registry create
ERE contamination: NONE
generated keys: ['client', 'default_regime', 'pipeline', 'portfolio', 'regime', 'supported_regimes']
```

What is genuinely wrong is everything around that core: the product is presented
as an adoption tool, the information model is far too thin for a live client
conversation, and three distinct workflows are collapsed into one.

---

## Point by point

### 1. Which parts correctly support blank onboarding

| Component | Verdict |
|---|---|
| `service.start_draft(client_id="", adopt=False)` | **Correct.** Falls to the `else` branch, `origin="new"`, empty profile. |
| `generation.plan(..., base_documents=None)` | **Correct.** With no base documents the client configuration is built from answers alone. A regression test already asserts a new client inherits no enrichment or transformations. |
| `generation.apply` → source registry | **Correct.** `SourceRegistry.upsert` is called for portfolios that have never delivered. |
| `derive_sources` / `derive_reporting` | **Correct and client-agnostic.** Neither reads a client file. |
| `EffectiveConfigResolver.client_config_for` | **Correct.** Falls back to the repository file only when a client has no generated artefact. |
| Backend test `test_a_new_client_can_be_created_end_to_end` | **Correct.** Starts from `start_draft(by=...)` with no client id. |

### 2. Which parts assume an existing client configuration

| Location | Problem |
|---|---|
| `generation.py:55-56` | `SEED_CLIENT_CONFIG` / `SEED_ANNEX12_CONFIG` still point at `config_client_ERM_UK.yaml` and `config_client_annex12.yaml`. Now dead after the `base_documents` refactor — `_seed()` has no callers — but their presence invites re-coupling. **Delete.** |
| `migration.DEFAULT_CLIENT_CONFIG` | Defaults to the ERE file for *any* client id. Correct for migration; wrong as a default anywhere else. |
| `model.PORTFOLIO_STRUCTURES`, `SELECTABLE_PRODUCTS` | Hard-coded Python constants, not configuration. Not ERE-specific, but not extensible without a deploy. |
| `service.list_clients` | Lists every client in the store and source registry as an onboarding subject. A client that has never onboarded is presented as work to do. |

### 3. Which screens are framed around adoption or import

**All three.** This is the substantive failure.

| Screen | Framing |
|---|---|
| `Home.tsx` | Single list, "Existing clients". Every non-onboarded row's action is `Adopt current configuration`. There is no queue model, no draft list, no case concept. Blank start exists (`startDraft(undefined, false)`) but is one button among adoption rows. |
| `Wizard.tsx:783-800` | `const adopted = current.origin !== "new"` drives the page title and the gap banner. The adopted path is the one with content; the blank path renders empty boxes with no guidance. |
| `ClientEditor.tsx:298` | `adopt: data?.status !== "onboarded"` — the editor's primary action doubles as the adoption entry. |
| `copy.ts:27` | `adopt: "Adopt current configuration"` is in the primary vocabulary. |
| Screenshots | Nine of eleven show an adopted ERE client. The blank journey was never captured, because it was never really the product. |

### 4. Which backend services require a legacy client file

**None require one.** `migration.adopt()` is the only reader, it is only reached
via `start_draft(adopt=True)`, and it degrades to an empty profile when the files
are absent. The dependency is optional throughout. The failure is one of
prominence, not of coupling.

### 5. Does approval create configuration for a brand-new client?

**Yes** — verified above. Client configuration, portfolio metadata and source
registrations are all written for a client with no prior existence.

### 6. Can source registry and portfolio records be created before any delivery?

**Yes.** `generation.apply` upserts source records at approval with
`status: pending_review` and no fingerprint, which is the correct state for a
source that has never delivered. Portfolio metadata is written at the same time.
This satisfies "portfolios and source records exist before the first delivery".

### 7. Can the UI complete onboarding without selecting an existing client?

**Yes, but only just, and untested.** The primary button starts a blank draft.
No frontend test completes a blank onboarding — every wizard test seeds ERE via
`adopt: true`. The blank path is reachable and unproven.

### 8. Is migration behaviour mixed into the core domain model?

**Partly.**

| Clean | Mixed |
|---|---|
| `migration.py` is a separate module and writes nothing. | `service.start_draft` takes `adopt` and branches three ways, so one entry point serves new, adopted and continued cases. |
| `model.py` has no migration concepts. | The draft document carries `provenance`, `gaps`, `sources_read` and `base_documents` — all migration-only — for every case including blank ones. |
| Generation is driven by `base_documents`, which a new client simply omits. | `Wizard.tsx` branches on `origin` for its title and banner. |

---

## What is missing entirely

These are absent, not merely mis-framed. They are the bulk of the corrected
requirement.

| Requirement | Status |
|---|---|
| Legal and reporting entities as first-class objects with reusable roles (originator, sponsor, SSPE, servicer, trustee, risk-retention holder …) | **Absent.** The model has one `legal_entity_name` + `lei` on the client. `grep -c` for sponsor/sspe/servicer/trustee in `model.py` returns 1 — a comment. |
| Onboarding case with a system-generated case ID | **Absent.** There is a `draft_id` (`onb_…`) but no case, and it is discarded at approval. |
| Full status lifecycle (Draft / Information requested / Awaiting client / In review / Changes required / Ready for approval / Approved / Activated / Withdrawn) | **Absent.** Two states: `draft` and `active`. |
| Information requests, responsible party, evidence, unresolved questions | **Absent.** Zero occurrences in the package. |
| Client-request checklist derived from conditional requirements | **Absent.** |
| Field catalogue as configuration | **Partial.** Regime fields are catalogue-driven (`onboarding_standing_fields.yaml`); client, portfolio and static-reporting fields are hard-coded in `Wizard.tsx` (lines 126-141, 476-487) and in Python dataclasses. |
| Amendment workflow distinct from onboarding | **Absent.** `grep` for `amend` returns nothing. "Edit configuration" reopens the onboarding wizard. |
| Identifier collision control | **Absent.** `IDENT_RE` checks shape only. Nothing prevents reusing an existing client or portfolio identifier. |
| Withdrawal that preserves the record and writes nothing | **Absent.** `discard_draft` exists but is not a governed status. |
| Idempotent regeneration | **Untested.** Generation is deterministic in practice but nothing asserts that re-approving an unchanged case is a no-op. |
| Per-portfolio reporting currency, period convention, source naming convention, delivery channel, file format, mandatory file set, sample-provided, mapping-complete | **Absent.** |
| Entities beyond the client: registration number, registered address, authorised approver | **Absent.** |
| Regime static data beyond RREL82/83/84 + the Annex 12 deal block — risk-retention percentage, sponsor/SSPE LEI, STS status, transaction identifiers and dates | **Absent**, and correctly reported as unrepresentable in the current artefacts. The corrected requirement asks for them to be *collected* regardless, with their consumer stated. |
| Mobile 390px | **Never checked.** |

---

## Verdict

Keep: the versioned immutable store, the audit integration, the resolver seam,
the artefact-merge discipline (never clearing what a delivery earned), the
governed-vocabulary sourcing, and `migration.py` itself.

Replace: the entry point, the information model, the status model, the
generation inputs, and every screen.

The first build answered "how do we stop hand-editing ERE's YAML". The product
is "how does a client Trakt has never met become a configured client".
