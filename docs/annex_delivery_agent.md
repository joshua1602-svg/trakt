# Annex Delivery Agent

The governed path from projected regulatory data to a submission-ready artefact.

```
projected data → delivery normalisation → artefact generation → schema validation
              → governed result → explicit approval → publication
```

One lifecycle, one governance model, one persistence layer, one evidence
contract. Everything report-specific lives behind a provider, so adding Annex 3
or Annex 9 means writing a provider — not cloning the agent.

Package: `engine/annex_delivery_agent/`.

---

## 1. Why it exists

The repository already had a reproducible Annex 2 route:

```
platform_canonical_typed.csv
  → engine/gate_4_projection/regime_projector.py
  → engine/gate_4b_delivery/annex2_delivery_normalizer.py
  → engine/gate_5_delivery/xml_builder_annex2.py
  → config/system/DRAFT1auth.099.001.04_1.3.0.xsd
  → lxml XMLSchema validation
```

What it did not have was governance around that route: no tenant binding, no
deterministic run identity, no restart safety, no approval gate before
publication, and — most importantly — **no disclosure of what the builder did to
the data on the way to schema validity**.

The agent supplies exactly those things. It reimplements none of the route's
business logic.

## 2. Architecture

| Module | Responsibility |
|---|---|
| `contracts.py` | `AnnexDefinition`, `DeliveryRequest`, `DeliveryResult`, `DeliveryStatus` — annex-neutral vocabulary |
| `registry.py` | `(annex_id, version) → provider`. `default_registry()` is the production allow-list |
| `agent.py` | Tenant binding, run identity, idempotency, governed envelope |
| `lifecycle.py` | The shared state machine: preflight → normalise → build → validate |
| `preflight.py` | Blocking checks, in the operator's language |
| `evidence.py` | Fills, coercions, omissions, unsupported fields, structural assumptions |
| `instrumentation.py` | How that evidence is obtained without changing the builder |
| `persistence.py` | Atomic writes, checkpoints, restart, operational store |
| `publication.py` | Approval gate and published package |
| `occ_adapter.py` | Operations Control Centre view and service facade |
| `providers/base.py` | The annex plug-in interface |
| `providers/annex2.py` | ESMA Annex 2, wrapping the proven components |
| `validators/xsd.py` | `lxml` XML-Schema validation |
| `cli.py` | Command-line access |

### The provider interface

```python
class AnnexProvider(ABC):
    definition: AnnexDefinition

    def check_configuration(request, report) -> {label: fingerprint}
    def check_applicability(request, report) -> bool
    def resolve_input(request, report)      -> ResolvedInput | None
    def preflight(request, resolved, report) -> None
    def normalise(request, resolved, work_dir) -> NormalisationOutcome
    def build(request, normalised, work_dir)   -> BuildOutcome
    def validate(request, built)               -> SchemaValidationResult
    def inspect(request, resolved, normalised, built) -> TransformationEvidence
    def classify(evidence, validation) -> (blocking, warnings)   # sensible default
    def publication_metadata(...) -> dict
    def component_versions() -> {name: sha256}                   # default: {}
```

The shared agent never sees a field code, a schema or a mapping workbook.
`tests/test_annex_delivery_agent_core.py` proves this: all 53 core tests run
against a stub provider and never import Annex 2.

## 3. Status model

```
RECEIVED → PREFLIGHT_BLOCKED
         → READY → NORMALISING → NORMALISATION_FAILED
                 → BUILDING    → BUILD_FAILED
                 → VALIDATING  → VALIDATION_FAILED
                 → PREPARED | PREPARED_WITH_WARNINGS
                             → AWAITING_APPROVAL → APPROVED → PUBLISHED
                                                 → REJECTED
```

`PREPARED` is not `PUBLISHED`. Nothing in the lifecycle can cross that line —
publication lives in a different module and requires an explicit human decision.

Statuses map onto the repository's existing envelope statuses
(`trakt_core.envelope`): `PREPARED`/`APPROVED`/`PUBLISHED` → `success`,
`PREPARED_WITH_WARNINGS`/`AWAITING_APPROVAL` → `partial_success`, the blocked
states → `blocked`, `BUILD_FAILED` → `error`.

## 4. Annex 2: what is wrapped

| Concern | Component | How it is called |
|---|---|---|
| Delivery normalisation | `annex2_delivery_normalizer.normalize_delivery` | Directly, as a library function |
| Artefact build | `xml_builder_annex2.build_annex2_tree` | Directly, under observation |
| Mapping | `xml_builder_annex2.load_mapping_specs` on the 1.3.1 workbook, sheet `DRAFT1auth.099.001.04` | Directly |
| Field order | `xml_builder_annex2.load_code_order` on `config/system/esma_code_order.yaml` | Directly |
| Schema | `config/system/DRAFT1auth.099.001.04_1.3.0.xsd` | `lxml.etree.XMLSchema` |
| Projection | `engine/gate_4_projection/regime_projector.py` | Referenced, not re-run — its output is the agent's input |

The CLI `main()` functions are *not* used: they add argument parsing and a
`sys.exit(2)` hard gate. The functions beneath are exactly what those CLIs run,
so the calculations are byte-for-byte the proven ones while the agent holds the
blocking decision, the timings and the evidence.

Configuration paths are the repository's own — the same defaults
`engine/orchestrator/trakt_run.py` uses. No alternative copies were created.

## 5. Instrumentation: how the evidence is obtained

Three techniques, in order of preference. Console output is never parsed.

**1. Interception.** A transparent wrapper is installed over a builder helper
for the duration of one build, delegating to the original and recording what it
saw. Used for `_coerce_record_value_for_branch` (the `RREL12` non-year → `2026`
rewrite) and `select_specs_for_value` (dropped values). Installed and removed by
a context manager, so a failed build cannot leave a patched module behind — and
a test asserts the observer returns exactly what the original returns.

**2. Structural derivation.** The finished artefact is censused for no-data
leaves by element path. A leaf standing at a path that some input column maps to
came from the data; a leaf standing anywhere else was inserted by the builder.
That rule needs no knowledge of what any code means, so it works for any annex.

**3. Reconciliation.** No-data values in the input are counted and compared with
no-data values in the output. The difference is what the builder added,
established independently of (1) and (2). Where the structural attribution does
not account for the whole total, the residual is reported as `unattributed`
rather than quietly dropped.

Currently surfaced for Annex 2:

- `_ensure_scndry_oblgr_incm_defaults` — `ND5` into `ScndryOblgrIncm/{IncmVal,Vrfctn}` on every record;
- `_ensure_hstrcl_colltn_nd_defaults` — `ND5` across 4 blocks × 36 months (NPRF);
- `_ensure_nprf_nonprfrmgdata_defaults` — `ND5` for NPE fallback codes;
- `_coerce_record_value_for_branch` — `RREL12` year coercion;
- optional fields dropped because no non-no-data branch accepts the value;
- codes carrying data with no mapping in the workbook for the selected template;
- universe fields this route cannot emit;
- the one-record-per-row and single-collateral-per-exposure shape assumptions.

None of these behaviours was changed.

## 6. Idempotency, restart and atomicity

**Run identity is derived, not allocated.** `run_id` is a digest over tenant,
portfolio, annex, version, reporting date, input path + size + mtime, the
configuration fingerprints and the provider options. Same governed inputs → same
run directory → the completed artefact is returned instead of rebuilt. Change a
rules file and the id moves, so a stale artefact can never be served as current.

**Atomic finalisation.** Every write lands on a temporary sibling and is moved
into place with `os.replace`. The artefact path therefore holds either the
complete previous file or the complete new one — never a partial 200 MB XML.

**Checkpoints.** Each stage records its outputs with size and SHA-256. A stage is
reusable only if every output still exists *and* still hashes to the recorded
value, so an interrupted run resumes from the last intact stage and a tampered
or truncated artefact is rebuilt.

**Failure invalidates downstream.** A failure at any stage drops that stage and
everything after it from the checkpoint.

## 7. Persistence

Operational state, under its own root, never mixed into production/latest:

```
<store_root>/<tenant>/<annex_id>/<portfolio>/<reporting_date>/<run_id>/
    00_request.json                  execution request + trusted context + annex definition
    01_preflight.json                every finding, blocking and advisory
    02_checkpoint.json               stage completion, output hashes, attempts
    03_validation.json               schema validation report
    04_transformation_evidence.json  fills, coercions, omissions, reconciliation
    05_result.json                   the governed result
    06_publication.json              package + receipt (after publication)
    07_failure.json                  failure evidence
    *_delivery_ready.csv             the normaliser's output
    *_delivery_issues.csv            the normaliser's issues
    *_delivery_report.json           the normaliser's summary
    <annex>_<portfolio>_<date>.xml   the artefact
```

Default root `out_annex_delivery`; override with `TRAKT_ANNEX_DELIVERY_ROOT`.

## 8. Publication

```
PREPARED / PREPARED_WITH_WARNINGS
    → request_approval()   → AWAITING_APPROVAL
    → approve() / reject() → APPROVED / REJECTED
    → publish()            → PUBLISHED
```

`publish()` refuses anything that is not `APPROVED`, and refuses a second time
for a run already published. Approval vocabulary matches
`apps/blob_trigger_app/approvals.py` (`pending`/`approved`/`rejected`/
`promoted`) so the console speaks one language.

The published package references the source data, the projected dataset, the
normalised dataset, the artefact, the schema and its hash, the validation
report, the automatic-value evidence, the operator approval, the configuration
and component versions, hashes and timestamps — plus an explicit disclaimer that
schema validity is not regulatory correctness.

`PublicationSink` is the destination seam. `FilesystemPublicationSink` ships;
a blob sink is a subclass with one method.

## 9. Tenancy

- The tenant always comes from the trusted `ExecutionContext`, never from a request body.
- A request naming a different `client_id` is refused with `TENANT_MISMATCH`.
- Portfolio selectors are narrowed within the tenant by `authorise_portfolio_access`.
- Selectors are validated tokens, so one cannot traverse the store.
- Run directories are tenant-first, and `list_runs` never crosses tenants.
- The publication gate re-checks the tenant recorded in the stored result, so holding a path is not enough.
- The operator console resolves the browser-supplied `client_id` against the tenancy configuration and an optional `TRAKT_OPERATOR_TENANTS` allowlist before building a context.

## 10. Running it

```bash
# What this deployment can prepare
python -m engine.annex_delivery_agent.cli list-annexes

# Prepare (never publishes)
python -m engine.annex_delivery_agent.cli prepare \
    --tenant ERE --annex ESMA_Annex2 \
    --reporting-date 2025-11-30 \
    --projected-input out/ere_ESMA_Annex2_projected.csv \
    --store-root out_annex_delivery

# Approve, then publish — two deliberate steps
python -m engine.annex_delivery_agent.cli approve \
    --tenant ERE --reporting-date 2025-11-30 --run-id run_… \
    --note "ND5 fills reviewed"
python -m engine.annex_delivery_agent.cli publish \
    --tenant ERE --reporting-date 2025-11-30 --run-id run_… \
    --publish-root out_annex_delivery/published
```

Tests:

```bash
python -m pytest tests/test_annex_delivery_agent_core.py \
                 tests/test_annex_delivery_agent_annex2.py \
                 tests/test_annex_delivery_agent_golden.py -q

# Full-scale golden reproduction (minutes, ~200 MB of output)
python scripts/run_annex2_golden.py
# or
TRAKT_ANNEX2_GOLDEN=1 python -m pytest tests/test_annex_delivery_agent_golden.py -s
```

## 10a. Measured results

Full-scale golden reproduction, run twice on the development container
(`python scripts/run_annex2_golden.py`):

| | Historical run | Golden reproduction | Note |
|---|---|---|---|
| Records | 11,035 | **11,035** | matches |
| Projected fields | 105 | 104 | different source data (synthetic projection) |
| Delivered fields | — | 101 | 3 projected codes have no workbook mapping for this template |
| Artefact | 208.8 MB | **186.6 MB** | narrower source rows |
| XSD validation | PASSED | **PASSED**, 0 errors | same XSD, same validator |
| Wall clock | ~171.5 s | **122.7 s / 124.0 s** | 28% faster |
| Peak RSS | ~772.7 MB | **738.5 MB / 737.0 MB** | 4% lower |

Stage breakdown of the golden run: preflight 0.4 s, normalise 24.6 s,
build 90.2 s, validate 0.8 s. The build dominates, as expected — it is the
unchanged Gate 5 builder.

Evidence from the same run, all reconciling exactly:

```
no_data_values_in_prepared_data   441,094
no_data_values_in_report          463,164
difference_added_by_builder        22,070   = 11,035 records × 2
attributed_as_automatic            22,070   ← _ensure_scndry_oblgr_incm_defaults
attributed_as_sourced             441,094
unattributed                            0
```

CI tier (5-record fixture): the same route, `XSD PASSED`, in about 2 seconds.
It proves the route is intact; it proves nothing about performance at scale, and
no assertion in `tests/test_annex_delivery_agent_golden.py` claims otherwise.

---

## 11. Adding Annex 3

Annex 3 (ABCP underlying exposures) is **not implemented and not registered** —
there is no authoritative normaliser, builder or schema for it in this
repository, and a placeholder registration would let an operator start a run
that cannot finish.

To add it:

1. **Obtain the authoritative inputs.** The Annex 3 XSD into `config/system/`,
   the ESMA mapping workbook, and an Annex 3 field universe (generate it the way
   `scripts/build_annex2_universe.py` generates Annex 2's).
2. **Add configuration.** `config/regime/annex3_delivery_rules.yaml` (the field
   rules, ND allowances, validators, precision), and an Annex 3 code order —
   either a new key in `config/system/esma_code_order.yaml` or its own file.
3. **Confirm the projector emits Annex 3.** `regime_projector.py` is already
   regime-parameterised (`--regime ESMA_Annex3`); verify its output columns
   against the new field universe. If it does not, that is a projector change
   and needs its own approval checkpoint — it is outside this agent.
4. **Write the normaliser** — or reuse Gate 4b if the Annex 3 rules fit its rule
   grammar, which is regime-agnostic apart from its configuration.
5. **Write the builder.** If Annex 3's XML is workbook-path driven like Annex 2's,
   `xml_builder_annex2.py`'s approach generalises; whether to parameterise it or
   write `xml_builder_annex3.py` is a judgement about how much its record anchor
   and choice-branch rules differ.
6. **Write `providers/annex3.py`** — an `AnnexProvider` subclass with an
   `ANNEX3_DEFINITION` and the ten methods above. Reuse `LxmlXsdValidator`
   unchanged. Reuse `instrumentation.py` unchanged: the no-data attribution rule
   is structural, not Annex 2-specific.
7. **Register it** — one line in `registry._register_builtin`.
8. **Write `tests/test_annex_delivery_agent_annex3.py`**, mirroring the Annex 2
   suite: component resolution, preflight refusals, a real build, XSD pass and
   fail, evidence, publication gating.

**Not required:** the agent, lifecycle, registry, contracts, evidence model,
instrumentation, persistence, publication, OCC adapter, CLI, API or the core
test suite.

## 12. Adding Annex 9

Identical to the above with Annex 9's own schema, workbook, rules, field
universe and code order. Annex 9 is investor-report shaped rather than
exposure-record shaped, so two things differ in the provider and nowhere else:

- **`record_count` is not row count.** `BuildOutcome.record_count` should carry
  whatever "one reported item" means for Annex 9; the lifecycle only reports it.
- **The record anchor changes.** `providers/annex2.py` hard-codes
  `RECORD_ANCHOR = "UndrlygXpsrRcrd"` for its own evidence derivation. Annex 9's
  provider declares its own; `instrumentation.py` takes it as a parameter and
  needs no change.

Note that `engine/gate_5_delivery/xml_builder_investor.py` exists and may be the
starting point for an Annex 9 builder — but its authority for Annex 9 has not
been established, and "it is in the repository" is not evidence that it is
correct. Establish that first, with a real artefact and a real schema pass,
before registering anything.

## 13. Known limitations

1. **The golden fixture is generated, not the original tape.** The 11,035-record
   client dataset behind the historical run is not in this repository. The golden
   tier reproduces the route at that scale from the committed 36-record synthetic
   projected data (`scripts/generate_annex2_scale_fixture.py`), with per-record
   identifiers made unique and every other value left exactly as the projector
   emitted it. It exercises the real components at real size; it is not a re-run
   of the original data.

2. **Memory is peak process RSS**, not a per-stage measurement — `getrusage` only
   reports a high-water mark. Stage timings are exact; stage memory is not.

3. **The builder holds the whole artefact in memory.** A 200 MB artefact means a
   correspondingly large `lxml` tree. Streaming the build would fix this and is
   the obvious next optimisation, but it changes the builder, which is out of
   scope here. The validator already avoids a second copy by validating the tree
   the builder produced rather than re-parsing the file.

4. **No-data path discovery samples records.** The distinct element paths are
   discovered from a spread of records and then counted exactly across the whole
   tree. The totals are always exact; if a path appears only in an unsampled
   record its count lands in `unattributed`, which is reported rather than
   hidden. Observed `unattributed` is 0 for every run to date.

5. **Publication reuses the approval vocabulary, not `ops.promote_pack`.** The
   existing promotion path promotes *sources* into the source registry; it has no
   notion of a delivery artefact. The gate here mirrors its semantics and states
   rather than forcing an unrelated model onto it.

6. **The projector is referenced, not orchestrated.** The agent consumes
   projected data; it does not run `regime_projector.py`. Wiring projection into
   the same governed run is a reasonable next step and needs no architectural
   change — the provider already declares the projector it expects.

7. **A run is not resumable across machines** unless the store root is shared
   storage. Checkpoints are filesystem paths.
