# Phase 1D — report

# STOP — IDENTITY MODEL GAP

**No production module changed.** `portfolio_summary` remains unconverted.

React and MI share one governed identity **model**. They do not share an
identity **vocabulary**, and the break is on the natural-language side: MI's
portfolio-id recognition is hardcoded to the **storage naming convention** and
does not recognise the governed ids this client actually has.

Commit: `dd6bd4a` (identity map instrument + 15 tests).

---

## 1. Portfolio identity map

`python -m migration_phase0.portfolio_identity_map` ·
`migration_phase0/PORTFOLIO_IDENTITY_MAP.json`

**React's selector is built from the governed registry, not a separate table** —
`frontend/mi-agent-ui/src/state/useWorkspace.ts:300` maps
`portfolio_context.context_index()` straight onto its lens list, with the comment
*"derived from the governed hierarchy … exactly one source of portfolio truth in
the app."* So there is no duplicate React mapping to remove, and §7's target
architecture is already half-built.

| storage / seed name | governed portfolio id | source type | React label | rows |
|---|---|---|---|---:|
| — | `total` (group) | direct + acquired | **Total** | 11,035 |
| — | `direct` (group) | direct | **Direct** | 7,126 |
| `raw-v2/…/direct_001/…` * | `alp_origination` | direct | **ALP Origination Book** | 7,126 |
| — | `acquired` (group) | acquired | **Acquired** | 3,909 |
| `raw-v2/…/acquired_001/…` * | `alp_acquired` | acquired | **ALP Acquired Back Book** | 3,909 |
| — | `spv1_sponsored` | *(none)* | **spv1_sponsored** | 0 |

\* the `direct_001` / `acquired_001` shape is the **seed/example** convention
(`config/source_registry.example.yaml`, blob prefixes, `config/dev/*.yaml`). This
client's governed ids are `alp_*` / `spv1_*`.

---

## 2. React label inventory — and whether MI understands it

Asking `"Summarise the <React label>"`, resolved through MI's lens and the
governed registry:

| React label | MI resolves to | population | correct? |
|---|---|---:|---|
| Total | `total` | 11,035 | **yes** |
| Direct | `direct` | 7,126 | **yes** (3/3 phrasings) |
| Acquired | `acquired` | 3,909 | **yes** (3/3 phrasings) |
| **ALP Origination Book** | `total` | **11,035** | **no — silently the whole book** |
| **ALP Acquired Back Book** | `acquired` | 3,909 | **coincidence** — see §5 |
| **spv1_sponsored** | `total` | **11,035** | **no — silently the whole book** |

**The two category labels work. No named portfolio label does.**

---

## 3. The cause — one line

```python
_COHORT_ID_RE = re.compile(r"\b((?:direct|acquired)_\d+)\b", re.IGNORECASE)
```

MI's natural-language portfolio-id recognition matches the
`direct_NNN` / `acquired_NNN` **storage-and-seed convention** and nothing else.
Measured:

| written in the question | recognised? |
|---|---|
| `acquired_001`, `direct_001` — the seed/storage shape | **yes** |
| `alp_acquired`, `alp_origination`, `spv1_sponsored` — the governed ids in use | **no** |

The general slug pattern (`_SELECTABLE_COHORT_ID_RE`) exists but is used **only
for an explicit UI selection**, never for text. `resolve_lens` has **no registry
access at all** — it is a pure-text function.

**So MI knows the infrastructure names and not the governed ones.** That is the
inversion of what the client needs, and it is exactly the thing this task's final
constraint warns against.

---

## 4. Business semantics, as governed

| concept | governed meaning | evidence |
|---|---|---|
| **Acquired Book** | a **category** — every registry portfolio typed `acquired`, computed per call | `resolve_scope`: *"`direct` is whatever is currently typed `direct`"* |
| **Funded / Direct Book** | `direct` is the type group. **"Funded" is NOT a synonym for it** — `funded` is the *dataset* (funded vs pipeline), orthogonal to source type. Measured: *"Summarise the funded book"* resolves to `total`, not `direct`, and an acquired source also carries `dataset: funded` | `source_registry.example.yaml`; lens probe |
| **Named portfolio** | one `source_portfolio_id` with a governed `display_label` | `PortfolioRegistry` |
| **SPV** | a named portfolio like any other. `spv1_sponsored` is typed `None`, so it is in **no** category group | measured, §1 |
| **Vintage** | a **cohort/time filter within** a population, not a portfolio identity. Measured: the source lens is resolved independently and is unchanged by a vintage — *"the 2023 vintage of the acquired book"* still resolves scope `acquired`. **But the year only survives on the progression path** — see §5b | `mi_query_spec.py`; parser + service probe |

**Worth flagging:** `spv1_sponsored` is untyped, so `Total` includes it while
neither `Direct` nor `Acquired` does. "Direct + Acquired" ≠ "Total" on this
book — already true, today, before any migration.

---

## 5. Multi-acquired-book test — the coincidence broken

`"Summarise the ALP Acquired Back Book"` **looks correct on the shipped book**:
it returns the acquired population, 3,909 loans. It is not correct. The label
contains the word *"Acquired"*, so the **type** lens matches, and this book has
exactly one acquired portfolio — so type == portfolio.

With a second acquired book in the registry:

| | selects |
|---|---|
| MI resolves `"Summarise the ALP Acquired Back Book"` → type `acquired` | `{alp_acquired, nbs_acquired}` |
| the client named | `{alp_acquired}` |

**The question answers for a portfolio the client did not name.**

This is the **third** fixture coincidence this programme has found — after
raw-vs-governed filters and Phase 1A's economics. All three have the same
shape: *one member per group makes a wrong mapping produce a right number.*

---

## 5b. Vintage — validated, and a fourth silent drop

§10 asked for vintage validation. Running it found that the vintage **year** is
only extracted on one path.

`_cohort_vintage` is reachable **only** from `_cohort_progression_recognizer`,
which requires a progression marker first
(`evolve|progress|season|over time|trend|…`). So:

| question | `cohort_vintage` | outcome |
|---|---|---|
| *"How has the 2023 vintage evolved over time?"* | `'2023'` | routes to `cohort_progression`, then **refuses honestly** — *"I understood that you asked for vintage, but that could not be applied"* |
| *"Summarise the 2023 vintage"* | **`None`** | `ok=True` — a bar **grouped by vintage across 13 cohorts** |

**The client asks for one vintage and is shown all of them.** The only facet
raised is the grouping *dimension* (`grouping_dimension · vintage · applied`);
the requested **year** — a narrowing — is not represented anywhere and nothing
says it was dropped. No facet or warning mentions `2023`.

The contrast is the point: where the vintage **is** carried, the product refuses
and names what it lost. Where it is dropped at parse, the answer ships.

A third phrasing, *"Show the 2023 vintage of the acquired book"*, refuses for the
**wrong reason** — *"'acquired' is not a governed measure in this dataset"* — the
provenance word read as a measure.

Pinned in `TestVintageIsNotPortfolioIdentity` (4 tests).

---

## 6. `acquired_001` classified

**Internal governed portfolio id that doubles as a storage folder name, and is
not client-visible for this client.**

* It **is** a `source_portfolio_id` in `config/source_registry.example.yaml`, and a resource id in `config/dev/{tenancy,resources}.yaml` — so not *purely* storage.
* It **is** a blob path segment: `raw-v2/ERE/direct/funded/monthly/direct_001/2025-11-30`.
* It is **not** in this client's governed registry, and React would never render it — React renders `label`.

### Correction to Phase 1C

My Case D used `acquired_001` and framed it as a client typing an unknown
portfolio id. **That example was wrong** — you are right that no client would
type it.

**The defect is real and I understated it.** The same silent widening is reached
by typing a *legitimate label the client is shown*: `"Summarise the ALP
Origination Book"` → the whole book, `ok=True`, nothing disclosed. And the
ranking is inverted — MI recognises the storage-shaped name and fails on the
governed ids in use.

Per §6 I have **not** added `acquired_001` to MI vocabulary, and the rule that
an unresolved scope must not widen to Total still stands unmet.

---

## 7. Semantic ownership

| concept | owner today | authoritative? |
|---|---|---|
| acquired / direct **category** | `trakt_core.portfolio.resolve_scope` | **yes** — computed from the registry per call |
| **specific portfolio** | `trakt_core.portfolio` (registry) | **yes** for selection; **absent** for text |
| **React label** | `PortfolioRegistry.display_label` via `context_index()` | **yes** — React renders it, MI cannot read it |
| **storage folder** | blob prefix / source registry | separate, correctly |
| **vintage** | `spec.cohort_vintage` + `cohorts` | separate concern, not identity |
| **text → portfolio** | `mi_agent.portfolio_lens`, **hardcoded convention, no registry** | **NO — this is the gap** |

**What must become authoritative:** the governed registry, for text-side
identity as it already is for selection-side identity.

---

## 8. Contract requirements

The interpretation contract must be able to carry, distinctly:

unrestricted · acquired category · direct category · **specific portfolio (by governed id)** · SPV (a specific portfolio) · **unresolved** · and — from Phase 1C — **provenance** (stated by the question vs supplied by the caller).

Phase 1A's `SourceScopeClaim` already has `scope ∈ {total, direct, acquired, cohort}` and `portfolio_ids`, and keeps `EMPTY`/`UNRESOLVABLE` distinct from any `FILLED` scope. **The contract shape is adequate. What cannot be produced is a correct `cohort` claim**, because nothing upstream can turn *"ALP Origination Book"* into `alp_origination`.

Vintage is **not** an identity and should not enter this claim.

---

## 9. Changes made — none to production

A bounded fix is **not available**. Making text-side identity registry-aware
means giving `resolve_lens` the registry, changing the single owner's signature
and every caller — 11 recognisers, the route handlers, `mi_service`, and the
projection. §9 calls that a broader redesign, so this stops.

And the corrective action is itself a behaviour change: removing or replacing
`_COHORT_ID_RE` changes what `acquired_001` resolves to today, which is
user-visible and needs its own authorisation.

---

## 10. Regression

**No production module changed** — `git diff 27f5ecc HEAD` touches only
`migration_phase0/`, `tests/` and `docs/`, so no shipped behaviour can have
moved. Verified rather than asserted:

| gate | result |
|---|---|
| frozen conversion baseline, 11 cases | **0 differences** |
| calibration bank | **267 passed** |
| shipped shapes | 15 correct, **0 wrong** |
| routed surface | 31 passed, `rt_004` (known) |
| robustness 44 | **32 / 6 / 4 / 2**; seasoning **Q1 4 · Q7 4 · Q8 12** |
| recognition (61) | **15 / 7 / 10 / 29**, 13 no-route; by-shape row for row |
| time-series surface | T1 PROVEN … T8 ABSENT, **silent drops 0** |
| interpretation + portfolio-scope + 1C/1D tests | **591 passed, 7 xfailed**, 1 pre-existing failure |
| introduced failing names | **0** |

The single failure is `test_p0_time_axis_request` · *"balance by each month"*,
established in Phase 1A as pre-existing at `42cef00` and recorded in the
baseline.

---

## 11. Final status

# STOP — IDENTITY MODEL GAP

**Which layers disagree:** React and the governed registry agree. MI's
**text-side** lens layer disagrees with both — it recognises a storage
convention (`direct_NNN`/`acquired_NNN`) and neither the governed ids
(`alp_acquired`) nor the governed labels (*"ALP Acquired Back Book"*).

**What must become authoritative:** the governed registry, for text-side
portfolio identity. Concretely, `mi_agent.portfolio_lens` needs to resolve
against `PortfolioRegistry` — matching governed ids and `display_label` — rather
than against a hardcoded id shape, with the registry still owning what a scope
*contains*.

**Not GO-READY**, and `portfolio_summary` should **not** be retried next: it
owns questions naming portfolios it cannot resolve, and every such question
silently answers for the whole book.

### The order I would put these in

1. **Text-side identity becomes registry-aware** — this task's gap. Unblocks named portfolios and SPVs, and removes the storage convention from MI semantics.
2. **The unresolved-scope widening** (Phase 1C) — once labels resolve, what remains genuinely unresolvable must refuse or disclose, not widen.
3. **Provenance in the contract** (Phase 1C §4) — small, designed, blocked only on 2.
4. **Then** retry the `portfolio_summary` conversion.

Doing 4 before 1–3 converts a route onto a compositional path while it is still
silently answering the wrong population for named portfolios.

---

## 12. Measured effort

| | |
|---|---|
| production lines changed | **0** |
| tests added | 15 (all passing; they pin measured behaviour) |
| instruments added | 1 |
| commits | 2 |
| baselines updated | 0 |

### Does this change estimated migration cost?

**Yes — it moves work out of the migration and into the product.**

The last four phases found, in order: `lens_from_selection` defaults to Total;
`resolve_lens` returns provenance type where the route uses portfolio ids;
`resolve_lens` omits the precedence bit; `resolve_scope` widens silently while
the route drops the flag; and now `resolve_lens` recognises the wrong id family
entirely.

**None of these is a migration problem.** Every one is a pre-existing defect or
gap in how the product resolves portfolio identity, surfaced because the
compositional work required the semantics to be written down exactly. The
migration is not costing more than the study estimated — **it is uncovering
product debt that was invisible while thirteen routes each resolved identity
their own way.**

That is worth something on its own, and it is a better reason to continue than
the compositional architecture was.
