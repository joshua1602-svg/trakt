# Phase 1 — How client configuration is assembled today

Traced from the code, not inferred. Every claim below carries a file and, where
it matters, a function or line reference.

---

## 1. Client configuration

### Where it lives

| File | Role |
|---|---|
| `config/client/config_client_ERM_UK.yaml` | The master client configuration. Comment at line 2 calls it "LAYER 3: MASTER CLIENT CONFIG". |
| `config/client/config_client_ERM_UK_demo.yaml` | Demo variant. |
| `config/client/config_client_annex12.yaml` | Annex 12 investor-report overlay. Header says it "Inherits LEI/Identity from config_ERM_UK.yaml". |
| `demo_platform/config/config_client_ALDERBRIDGE_DEMO.yaml`, `synthetic_demo/config/config_client_SYNTHETIC_MULTIBOOK.yaml` | Demo/synthetic clients. |

There is **one file per client, hand-maintained**, and the production path
resolves a single default: `config/client/config_client_ERM_UK.yaml`.

### How it is loaded

Four independent readers, all pointing at the same default path:

1. **`operations_control/engine.py:135`** — `OpsEngine.__init__` reads
   `TRAKT_OPS_CLIENT_CONFIG`, defaulting to
   `config/client/config_client_ERM_UK.yaml`, into `self.client_config_path`.
2. **`operations_control/configuration/resolver.py:60`** —
   `EffectiveConfigResolver.__init__` takes the same path and uses it as the
   `client` layer of the precedence merge (`resolve_layers(..., ("client",
   self.client_config_path))`, line 164).
3. **`operations_control/annex2/preflight.py:23`** — `DEFAULT_CLIENT_CONFIG`
   points at the same file; `run_preflight()` (line 59) reads it with
   `yaml.safe_load` and a tree-wide `_deep_find` (line 43) that ignores the
   block structure entirely.
4. **`engine/orchestrator/trakt_run.py:1031`** — the pipeline CLI takes it as
   `--master-config`.

Additional readers of the same document for narrower purposes:
`analytics/streamlit_app_erm.py:85,99` (MI branding),
`engine/gate_4_projection/annex12_projector.py:331,338` (falls back to the
master config for the Annex 12 LEI and entity name),
`agents/config_bootstrap_agent.py:73` (declares which keys a bootstrap must
collect).

The merge engine itself is `config/system/config_resolver.py`, loaded by file
path (`resolver.py:132-142`), which records per-key provenance.

### Who consumes it

`EffectiveConfigResolver.resolve()` produces an immutable
`EffectiveConfiguration` (`operations_control/configuration/contract.py`),
persists it per workflow run, and `materialise_snapshot()` (`resolver.py:272`)
writes a hash-verified snapshot that the narrowed agent consumes. So the client
YAML is read once per run and pinned; a mid-run edit cannot change a running
delivery.

### Mandatory vs optional fields

Mandatory is defined in exactly one place — `annex2/preflight.py:97-131` — and
only for the regulatory route:

| Key | Blocking | Validator |
|---|---|---|
| `originator_legal_entity_identifier` | **Yes** | `^[A-Z0-9]{18}[0-9]{2}$` |
| `originator_establishment_country` | No (review) | `^[A-Z]{2}$` |
| `country` | No (review) | `^[A-Z]{2}$` |
| `base_currency` | No (review) | `^[A-Z]{3}$` |
| `reporting_period` | **Yes** | supplied by the workflow, not the file |
| `nd_defaults` | No (review) | presence only |

Everything else in the file is optional in the sense that nothing refuses a run
without it.

> **Finding worth acting on.** The committed value at
> `config/client/config_client_ERM_UK.yaml:67` is
> `213800ABCDE123456701N202501` — 27 characters. `LEI_RE` in
> `preflight.py:25` requires exactly 20. On today's code an ERE Annex 2
> delivery is **blocked** by its own committed configuration until an operator
> supplies a valid LEI through a governed decision. Onboarding surfaces this at
> the point of entry rather than at delivery time.

### Operational vs reporting metadata

| Block | Classification |
|---|---|
| `client.client_id`, `client.environment` | Operational — identity and routing |
| `client.display_name` | Reporting — comment at line 6 says "Used for UI/Reports only" |
| `portfolio.asset_class`, `country`, `base_currency` | Both — drives normalisation *and* is reported |
| `portfolio.static_reporting_date` | **Delivery-specific**, sitting in a standing file |
| `default_regime`, `supported_regimes`, `regime` | Operational — capability and routing |
| `regime_overrides.ESMA_Annex2.uk_geography` | Reporting — changes reported values |
| `pipeline.*` | Operational — which stages run |
| `pipeline_persistence.*` | Operational — where outputs go |
| `loan_engine.*` | Operational (calculation conventions), partly reporting |
| `defaults.originator_*` | **Reporting** — injected into every row as RREL82/83/84 |
| `transformations`, `enrichment` | Operational — data preparation |
| `nuts_classification_year` | Reporting |
| `mi.branding` | Reporting — display only |

---

## 2. Source registry

### How a client is represented

There is no client record. A client exists **because a source record names it**:
`SourceRegistry.seen_client()` (`apps/blob_trigger_app/source_registry.py:137`)
is `any(r.client_id == client_id for r in self._records.values())`.

The OCC's client list (`operations_control/api/app.py:210-222`) is the union of
the operations-control client index and every distinct `client_id` in the source
registry.

`config/tenancy.example.yaml` optionally adds a tenant record with a display
name and an authorised portfolio list, consumed by
`trakt_core.tenancy.authorise_portfolio_access`. It is optional and absent in the
current ERE shape.

### How portfolios are represented

Three separate places, none authoritative on its own:

1. **Source registry** — `SourceRecord.source_portfolio_id` +
   `source_portfolio_type` (`source_registry.py:26-51`). This is the operational
   truth: a portfolio exists once a record names it.
2. **Portfolio registry** — `config/client/portfolio_registry.example.yaml`,
   loaded by `mi_agent/portfolio_metadata.py:110` (`load_portfolio_metadata`).
   Optional. Carries `source_portfolio_label`, `originates`,
   `pipeline_data_available`, `forecast_treatment`, `runoff_curve`. Absent by
   default; behaviour then falls back to
   `trakt_core.portfolio.DEFAULT_ORIGINATION_BY_TYPE`.
3. **Tenancy config** — an authorisation allow-list only.

### How registrations are created

- **Automatically, on publication.** `OpsEngine._promote_source()`
  (`engine.py:1641`) calls `apps.blob_trigger_app.approvals.write_pending` →
  `approve` → `promote`, which upserts a `SourceRecord` and stamps
  `last_successful_run_id` / `last_successful_reporting_period`.
- **By hand.** `python -m apps.blob_trigger_app.repin` pins a schema
  fingerprint from a representative pack (documented at
  `config/source_registry.example.yaml:14-21`).
- **By backfill.** `python -m apps.blob_trigger_app.backfill` auto-pins
  recurring sources.

### How they are updated

`SourceRegistry.upsert()` (line 167) replaces the whole record and saves. There
is no field-level merge and no history: **the previous record is gone.** The
durable location is `TRAKT_SOURCE_REGISTRY_URI`, default
`blob://trakt-state/registry/source_registry.yaml` (`layout.py:34`).

---

## 3. Annex / regime configuration

### ESMA Annex 2

The field universe is `config/regime/annex2_field_universe.yaml` (107 fields,
workbook-derived). Delivery behaviour is `config/regime/annex2_delivery_rules.yaml`.

Fields whose value comes from **standing client configuration**, not the tape:

| Code | Field | Where it lives today |
|---|---|---|
| RREL82 | Originator Name | `defaults.originator_name` |
| RREL83 | Originator LEI | `defaults.originator_legal_entity_identifier` |
| RREL84 | Originator Establishment Country | `defaults.originator_establishment_country` |
| RREL80 | Original Lender LEI | delivery rules default to ND5 (line 1026-1035) |
| RREL81 | Original Lender Establishment Country | delivery rules default to ND5 |

Every other Annex 2 field is loan-level and comes from the tape.

### ESMA Annex 12 (Investor Reporting)

Template: `config/regime/annex12_template.yaml`. The `annex12.deal` block
(lines 57-70) is explicitly "supplied by client/deal instance config". Client
instance: `config/client/config_client_annex12.yaml`. Projector:
`engine/gate_4_projection/annex12_projector.py`.

| Code | Field | Standing? |
|---|---|---|
| IVSS1 | Securitisation unique identifier | Standing client (resolved from master LEI at line 331 if absent) |
| IVSS3 / IVSS4 | Securitisation name / reporting entity name | Standing client (defaulted from master name, lines 338-341) |
| IVSS5 / IVSS6 / IVSS7 | Contact person / telephone / email | Standing client |
| IVSS8 | Risk retention method | Standing client (enum at template line 20) |
| IVSS9 | Risk retention holder | Standing client (enum at template line 27) |
| IVSS10 | Underlying exposure type | Standing portfolio (enum at template line 34) |
| IVSS11 / IVSS12 / IVSS13 / IVSS20 / IVSS30 | Structural features | Standing client |
| IVSS2 | Data cut-off date | **Delivery-specific** |
| IVSS14-IVSS29, IVSS38-40 | Balances, collections, arrears | **Delivery-specific** (computed) |
| IVSR1-IVSR10 | Trigger definitions | Deal structure — currently hard-coded per deal |
| IVSF1-IVSF6 | Cashflow items | **Delivery-specific** |

### MI

No standing configuration beyond `mi.branding` in the client file. MI applies to
every delivery: the outcome vocabulary
(`operations_control/contracts.py:OUTCOMES`) is `mi` and `mi_annex2` — there is
no MI-less option.

### Static Pools

`config/asset/static_pools_config_erm.yaml` — a chart list only, held at the
**asset** layer, not per client.

### Future regimes

`ASSET_MODEL` (`operations_control/configuration/packages.py:58`) is the single
table mapping an asset class to the regimes it supports. Adding a regime today
means editing that dict **in Python**.

### Classification summary

| Classification | Fields |
|---|---|
| Standing client | client id, display name, legal entity name, LEI, environment, originator name/LEI/country, Annex 12 IVSS1/3/4/5/6/7/8/9/11/12/13/20/30, branding, calculation conventions |
| Standing portfolio | asset class, portfolio type, jurisdiction, base currency, cadences, NUTS year, UK geography override, IVSS10, origination and pipeline availability |
| Standing transaction | *(none found)* — the platform holds no per-transaction standing configuration; transaction attributes arrive on the tape |
| Delivery-specific | `portfolio.static_reporting_date`, `loan_engine.reporting_date`, IVSS2, all IVSS balance/collection/arrears fields, IVSF cashflows, reporting period, input paths |

---

## 4. Other manually maintained configuration

| Configuration | Where | Should it become onboarding data? |
|---|---|---|
| Portfolio metadata | `config/client/portfolio_registry.yaml` | **Yes** — labels, type, origination, pipeline availability are standing. Runoff curves are a supplied analytical input; onboarding should carry them but never generate them. |
| Client display information | `client.display_name` | **Yes** |
| Branding | `mi.branding.app_title`, `theme.primary_color` | **Yes** |
| Report defaults | `defaults.originator_*`, `nuts_classification_year` | **Yes** — these are the Annex 2 standing fields |
| Notification contacts | *Nowhere.* No contact field exists in any client configuration; Annex 12 IVSS5/6/7 is the only contact in the system | **Yes** — a genuine gap |
| Legal entity information | `defaults.originator_name` (doubles as RREL82) | **Yes**, and it should stop doubling |
| Issuer information | *Nowhere.* No issuer/SSPE field exists | Capture, but **cannot be written** — no artefact holds it |
| Warehouse information | *Nowhere* | Capture as static reporting information |
| Trustee information | *Nowhere* | Capture as static reporting information |
| Servicer information | `SourceRecord.source_system` records who supplies data; no regulatory servicer field | Partially — the data supplier is already standing portfolio information |
| Tenancy | `config/tenancy.yaml` | **No** — authorisation, not business configuration. Left where it is deliberately. |
| System / regime / asset packages | `config/system/`, `config/regime/`, `config/asset/` | **No** — already governed through the administrator area with draft → validate → activate → rollback. |

---

## 5. What this means

Three things stand out.

1. **Configuration is authored by editing YAML in a repository.** Every value in
   the tables above is maintained by hand. There is no attribution, no reason,
   no before/after, and no way to answer "what was this client's configuration
   in March".
2. **The same fact is stored more than once.** The legal entity name is
   `defaults.originator_name` because Annex 2 needs it there; the Annex 12
   projector then falls back to it for IVSS3/IVSS4. A client's identity is
   expressed through a regulatory field code.
3. **Some standing information has nowhere to live at all.** Reporting and
   operational contacts, trustee, warehouse lender, issuer and SSPE simply do
   not exist in any configuration file.
