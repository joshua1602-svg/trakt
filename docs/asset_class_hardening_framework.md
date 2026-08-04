# Asset-class hardening framework (`simulation/`)

Deterministic, seeded simulation and regression framework that generates realistic
funded portfolio histories for **equity release**, **bridge lending** and **asset
finance** (with *equipment finance* as an asset-finance subtype), renders them into
materially different lender source dialects, and drives them through the **real**
Trakt production pathway.

---

## 1. Production architecture discovered

The investigation below was done before any code was written. Every stage named
here is exercised by the framework; nothing is re-implemented.

| Stage | Production module | Notes |
|---|---|---|
| Funded ingestion entry point | `engine/orchestrator/trakt_run.py --mode mi` | Gate 1 → Gate 3b. The blob trigger (`function_app.py`) and `demo_platform.orchestration` both drive this same script. |
| Field registry | `config/system/fields_registry.yaml` | `portfolio_type` selects the canonical field superset (`common` + the requested type). |
| Alias / synonym libraries | `config/system/aliases_*.yaml`, `--extra-aliases-dir` overlay | Client-contract overlays are the governed way to add a lender vocabulary without touching the global mapper. |
| Messy → canonical mapper | `engine/gate_1_alignment/semantic_alignment.py` | Tiers 1–6 deterministic (exact / normalised / alias / token-set / RapidFuzz). |
| Canonical transform | `engine/gate_2_transform/canonical_transform.py` | Typing, date ladder, percentage scale, geography enrichment, LTV derivation. |
| Lineage | `engine/gate_2_transform/lineage_tracker.py` | Gate 2.5. |
| Canonical validation | `engine/gate_3_validation/validate_canonical.py` | Gate 2. |
| Business-rule validation | `engine/gate_3_validation/validate_business_rules.py` | Gate 3. |
| Validation aggregation | `engine/gate_3_validation/aggregate_validation_results.py` | Gate 3b. |
| Provenance | `engine/provenance.py` | `source_portfolio_id` / `_type` / `_label` / `portfolio_cohort` stamped on every row. |
| Platform consolidation | `engine/platform_assembler.py` | Latest canonical per `source_portfolio_id` → `platform_canonical_typed.csv`. |
| Historical funded snapshots | `snapshot/model.py`, `snapshot/store.py`, `snapshot/adapters/local_fs.py` | `SnapshotHeader` + `SnapshotStore.resolve_latest/as_of/range/compare`. Reporting date is never defaulted from `upload_timestamp`. |
| Governed portfolio contract | `trakt_core/portfolio.py` | Registry, scope resolution, capabilities, coverage disclosure. |
| Deterministic MI states | `mi_agent/states/assembler.py`, `mi_agent/states/temporal.py` | `total_funded`, cohort states, `compare` / `trend`. |
| MI runtime boundary | `mi_agent/mi_runtime.py` | flat / state / temporal / risk dispatch behind one entry point. |
| Governed MI Agent capability | `mi_agent_api/mi_service.py::execute_governed_mi_query` | `capability id: mi.question.answer`. React **and** M365 Copilot are adapters over this one function. |
| Risk monitor (store-backed) | `mi_agent/risk_monitor/monitor.py` | concentration / migration / trajectory over the snapshot store. |
| Governed limit library | `config/risk/concentration_test_library.yaml` + `mi_agent/concentration_tests/` | Declares *what a test is* (field roles, evaluator, parameter schema). Client thresholds arrive as an operator-approved `ActiveConfiguration`. |
| Limit evaluation | `mi_agent/concentration_tests/evaluation.py::evaluate_active_tests` | `pass` / `warning` / `breach` / `unavailable`, fail-closed. |
| Regime projection | `engine/gate_4_projection/regime_projector.py` | Canonical → ESMA Annex 2/3/4/8/9. ND codes inserted **only** here. |
| Regulatory artefact | `engine/gate_5_delivery/xml_builder_annex2.py`, `xml_builder.py` | XML + XSD validation for Annex 2; generic builder for the other annexes. |

### Findings that shaped the design

1. **The canonical registry already supports bridge and asset finance.** Bridge
   needs `charge_type`, `lien`, `maturity_date`, `original_term`, both LTVs,
   `collateral_geography`, `purpose`, `property_type` — all `portfolio_type: common`.
   Asset finance needs `manufacturer`, `model`, `original_/current_residual_value_of_asset`,
   `date_of_lease_expiration`, `number_of_leased_objects`,
   `year_of_manufacture_construction` (all `portfolio_type: equipment`) plus
   `balloon_amount`, `collateral_type`, `product_type`, `nace_industry_code`,
   `seller_name`, `scheduled_principal_payment_frequency` (all `common`).
   **No new canonical fields were added.**
2. **`--portfolio-type` is the existing asset-class extension point.** Bridge runs
   as `cre`, asset finance as `equipment`, equity release as `equity_release`.
   Equipment finance is a *subtype* of asset finance, expressed through contract
   and collateral configuration, not a separate architecture.
3. **`demo_platform/` is the precedent** for driving the real pipeline from
   synthetic data. This framework follows the same wiring (subprocess `trakt_run.py`,
   `platform_assembler.py`, `blob://` filesystem backend) but adds seeded manifests,
   multiple dialects per economic case, independent reference truth and layered
   assertions.
4. **Four production defects were found and fixed** — see
   `tests/test_simulation_platform_extensions.py`, which pins each fix:
   * `trakt_run.run_regulatory` invoked the generic XML builder with no
     `--template`, so it fell through to a default file that does not exist and
     died inside a subprocess with a Jinja `TemplateNotFound`. Every regime
     except Annex 2 was therefore un-deliverable *and* reported as a crash.
     Now a missing template is a governed Gate 5 refusal naming the regime.
   * `config/system/enum_mapping.yaml` mapped `collateral_type` onto codes
     (`R1`/`R2`/`C1`/`C2`) that are not members of the `CollTp` enumeration in
     `DRAFT1auth.099.001.04_1.3.0.xsd`, so every Annex 2 submission carrying a
     mapped collateral type failed XSD validation at Gate 5.
   * `engine/provenance.stamp_dataframe` blanked every *optional* provenance
     field whose portfolio-level value was absent, destroying loan-level
     `seller_name` — so a vendor concentration silently resolved to the
     originator instead. It now leaves an optional field alone when it has no
     portfolio-level value to stamp.
   * Gate 1 reported `OK  0 fields mapped` for a source whose every header was
     unmapped, and the run continued to produce an empty canonical and exit 0.
     A zero-mapping source is now refused as an unsupported schema.
5. **The MI semantics registry did not reach the new asset classes.** Ten
   canonical fields the registry already carried — `charge_type`,
   `collateral_type`, `seller_name`, `nace_industry_code`, `manufacturer`,
   `model`, `balloon_amount`, `current_residual_value_of_asset`,
   `date_of_lease_expiration`, `number_of_leased_objects` — had no curated MI
   business vocabulary, so the governed MI Agent could not answer "balance by
   charge rank" or "exposure by vendor". They are now curated
   (`mi_agent/build_mi_semantics_registry.py`, 106 → 116 fields).
6. **`spv_id` is a gap.** It is a reserved snapshot column (`snapshot/model.py`)
   and a governed risk field role (`spv`), but it is **not** in the canonical field
   registry, so a funded SPV segmentation cannot round-trip through canonical
   today. The framework therefore carries the funding structure on the production
   segmentation that *does* round-trip — `source_portfolio_id` / `portfolio_cohort` —
   and records the gap.

---

## 2. Package layout

```
simulation/
  models.py            typed manifest + loan-state + movement + failure vocabulary
  manifests.py         the catalogue (18 economic + 2 guard), validation, JSON round-trip
  generator.py         seeded economic generator (asset-class agnostic core)
  history.py           month-by-month roll-forward + movement reconciliation
  reference_truth.py   INDEPENDENT expected truth (small, transparent arithmetic)
  assets/
    equity_release.py  roll-up interest, NNEG-relevant LTV, redemptions
    bridge.py          charge rank, maturity wall, extensions, enforcement
    asset_finance.py   HP / lease / loan, balloon, residual, repossession
  dialects/
    clean_csv.py       ISO dates, decimal numerics, canonical-like enums
    lender_excel.py    lender header aliases, reordered columns, extra columns
    locale_csv.py      day-first dates, comma decimals, enum synonyms, Y/N
  pipeline.py          drives the REAL production pathway
  runner.py            CLI: generate | run | run-all | reproduce | list
  assertions/          canonical / history / mi / risk / agent / regime
  aliases/<asset>/     approved onboarding-contract alias overlays (per asset class)
```

---

## 3. Adding a case, dialect or asset subtype

### A new economic case

1. Add a `CaseManifest` to the catalogue in `simulation/manifests.py`.
   Every field is explicit: `case_id`, `seed`, `asset_class`, `asset_subtype`,
   `client_id`, `portfolio_id`, `spv_id`, `currency`, `jurisdiction`,
   `start_reporting_date`, `months`, `opening_loan_count`, `dialects`,
   `economic_intent`, `risk_intent`, `regime`, `expectation`.
2. Give it a *hardening purpose* — the `purpose` field is mandatory and is
   asserted to be non-empty by `tests/test_simulation_framework.py`.
3. Run `python -m simulation.runner run --case <case_id>`.

### A new source dialect

1. Implement `render(frame, case, period) -> Path` in a new module under
   `simulation/dialects/` and register it in `simulation/dialects/__init__.py`.
2. Declare the dialect id in the manifests that should use it.
3. Canonical equivalence is asserted automatically by
   `simulation/assertions/canonical.py` — a dialect that loses economic
   information fails the run rather than silently degrading.

### A new asset subtype

1. Add the subtype name to `ASSET_SUBTYPES[<asset_class>]` in
   `simulation/models.py` — the vocabulary is closed, so an undeclared subtype
   fails manifest validation rather than being quietly accepted.
2. Give it its configuration in the owning asset-class module. For asset finance
   that is `SUBTYPE_CATEGORIES` (which equipment categories it draws on) plus
   `EQUIPMENT_CATEGORIES` (manufacturers, typical value, depreciation,
   collateral class, units financed) — data, not code. Bridge uses
   `_SUBTYPE_COLLATERAL`; equity release uses `_SUBTYPE_PRODUCTS`.
3. If the subtype needs a canonical field the asset class does not already emit,
   add it to that module's field tuple **and** to
   `simulation/dialects/vocabulary.py`, then regenerate the alias contract:
   `python -m simulation.tools.build_alias_contracts`.
   `tests/test_simulation_framework.py` fails if the two drift apart, and
   `simulation._registry.unsupported_fields` fails if the field is one
   production would discard for that `--portfolio-type`.
4. Do **not** create a new top-level asset class unless the production
   `--portfolio-type` selection genuinely differs. Equipment finance did not:
   it is a subtype of asset finance, expressed entirely through the
   configuration above.

### Refreshing the regression fixtures

`tests/fixtures/simulation/` pins six cases' manifests and expected truth.
Regenerating is a deliberate act — it re-baselines the economics:

```bash
python -m simulation.tools.build_regression_fixtures
```

---

## 4. Running

```bash
python -m simulation.runner list
python -m simulation.runner generate --case bridge_maturity_wall_v1
python -m simulation.runner run      --case bridge_maturity_wall_v1
python -m simulation.runner run-all  --profile smoke
python -m simulation.runner run-all  --profile standard
python -m simulation.runner run-all  --profile performance
python -m simulation.runner run-all  --profile performance --timeout 1800
python -m simulation.runner run-all  --profile performance --stages regime
python -m simulation.runner reproduce --case bridge_maturity_wall_v1 --seed 41204
```

Every run writes a structured evidence directory:

```
out_simulation/<case_id>/
  manifest.json                  the reproducibility contract, verbatim
  generated_sources/             <dialect>/<reporting_date>/<file>
  canonical_outputs/             <dialect>/<reporting_date>/out|validation
  integrated_snapshot_evidence/  the governed snapshot store + registrations.json
  expected_truth/                the independent expected-truth package
  mi_results/    risk_results/    agent_results/    regime_outputs/
  assertion_results.json         every assertion, passed and failed
  run_summary.json               stages, timings, seeds, production modules
```

`run-all` additionally writes `run_summary_<profile>.json` at the run root, with
the cases run, the assertion totals, the expected and unexpected failures, the
per-case timings and the exact `reproduce` command for each case.

---

## 5. Profiles

| Profile | Cases | Periods | Dialects | Scale | Intended use |
|---|---|---|---|---|---|
| `smoke` | one per asset family (3) | 3 | clean CSV only | small (~110 loans) | ordinary CI |
| `standard` | the whole catalogue (20) | 6 | all three | small | the full regression sweep |
| `performance` | 4 runs: one per family at standard scale, then the equity-release case AGAIN at large scale | 6, and **1** for the large run | clean CSV only | standard (5 000 loans) / large (100 000 loans) | timing and memory, run deliberately |

The performance schedule is explicit (`manifests.PERFORMANCE_RUNS`) rather than
inferred from case ids. The large run repeats one of the standard runs at a
bigger row count on purpose: holding the economics constant and changing only
the scale is what makes the two timings comparable. Its evidence lands in
`<case_id>__large/` so the two runs do not overwrite each other.

**The large run is deliberately narrow** — one asset class, one 100 000-loan
funded snapshot, one dialect, stopping at risk. The first full-width run
measured why:

| Stage | 100 000 loans × 6 periods | Share |
|---|---|---|
| regime (Gate 4 + Gate 5 XML) | 3 108 s | 61.6% |
| generate | 651 s | 12.9% |
| history_integration | 597 s | 11.8% |
| agent | 235 s | 4.6% |
| ingest | 177 s | 3.5% |
| everything else | 277 s | 5.6% |

Regime and Agent are 66% of the run and measure neither funded ingestion nor
canonical throughput. Gate 5 is also run **twice** there — once for the artefact
and once to prove the artefact reproduces — producing two 2.6 GB Annex 2
submissions. Narrowing to a single snapshot took the benchmark from **5 045 s /
1 704 MB to 884 s / 759 MB** while still covering ingestion, canonical
transformation, validation, snapshot integration, MI and risk at full row count.

Regime and Agent throughput at scale are a **separate** diagnostic, asked for
explicitly and measured on their own:

```bash
python -m simulation.runner run-all --profile performance --stages regime
python -m simulation.runner run-all --profile performance --stages agent
```

### Bounded execution

Long runs are observable and interruptible, never open-ended:

| Control | Behaviour |
|---|---|
| live progress | every stage prints on start and on finish with its elapsed time and assertion count |
| `--timeout <seconds>` | wall-clock budget for the whole profile, checked **between** stages (never mid-stage — a half-written canonical is not evidence) |
| `PARTIAL_PERFORMANCE` | on expiry the run stops cleanly, keeps every completed timing and artefact, lists what was truncated and what was never started, and exits **2** (`0` complete, `1` real failure) |
| `--stages a,b,c` | restricts the run; prerequisites are added automatically, and excluded stages are recorded as skipped **with the reason**, so a narrowed run never reads as a run that verified everything |

`performance` is excluded from the ordinary unit-test suite and from the smoke
and standard profiles: the repository has no timing-threshold convention, so the
profile MEASURES and reports rather than asserting a wall-clock bound that would
flap on shared CI hardware.

---

## 6. What runs in the ordinary test suite

| Test file | Covers |
|---|---|
| `tests/test_simulation_framework.py` | manifests, deterministic seeding, loan-state transitions, reconciliation, dialect formatting, reference-truth independence, failure classification |
| `tests/test_simulation_pipeline.py` | source → canonical through the real gates, cross-dialect equivalence, one end-to-end case, both guard cases failing at the right boundary |
| `tests/test_simulation_platform_extensions.py` | the governed-library extensions, the three defect fixes, the KPI evidence enhancement, and the recorded limitations |
| `tests/test_simulation_regression_fixtures.py` | the pinned economics still reproduce byte-for-byte from their committed manifests |
