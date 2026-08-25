# P0 assurance audit — migration instruments running on empty semantics

Base `16586cf`. **No production code changed.** Assurance tooling only.

---

## 1. The authoritative loader, from production

| | |
|---|---|
| loader | `mi_agent.mi_query_validator.load_mi_semantics` |
| path resolver | `mi_agent_api.data_source.semantics_path()` → `DEFAULT_SEMANTICS`, overridable by `MI_AGENT_SEMANTICS` |
| resolved file | `mi_agent/mi_semantics_field_registry.yaml` |
| returns | `dict` with top-level `fields` (118 governed fields) and `metadata` |
| serving caller | `mi_service.execute_governed_mi_query` → `load_mi_semantics(semantics_path())` |

**Not ambiguous.** The hardcoded path used by several `question_interpretation/`
scripts resolves to the same file today — but it **bypasses the
`MI_AGENT_SEMANTICS` override**, so under a client-specific registry they would
silently diverge from the serving path. Recorded as a qualification (§7), not
changed here.

## 2. Inventory

23 files in `migration_phase0/` mention semantics. Classified by how they obtain
them:

| how | count | verdict |
|---|---|---|
| `load_mi_semantics` (authoritative) | 17 | healthy |
| guessed loader names on `mi_service` | **2** | **silently degraded** |
| mention semantics only in prose | 4 | not dependent |

The two degraded:

```
migration_phase0/dependency_verification_temporal_compare.py
migration_phase0/pipeline_stage_census.py
```

Both carried the same `_env()`:

```python
for name in ("load_semantics", "_load_semantics", "semantics_for"):
    fn = getattr(mi_service, name, None)
    ...
    except Exception:  # noqa: BLE001
        pass
return cfg.CLIENT_ID, sem          # sem is still {}
```

None of those three names exists on `mi_service`. Verified:

```
dependency_verification_temporal_compare  client=alderbridge  top_keys=[]  fields=0
    balance=False  ltv=False  stage=False
pipeline_stage_census                     client=alderbridge  top_keys=[]  fields=0
    balance=False  ltv=False  stage=False
```

Both **exited 0** and printed well-formed results.

## 3. What the degraded run concealed

`dependency_verification_temporal_compare` — the instrument behind C5's
dependency proof — reported on empty semantics:

```
readings                                    : 48
dataset disagreements, contract AS BUILT    : 0
measure disagreements (at the same dataset) : 0
readings whose periods are STRUCTURAL       : 48 of an EXPECTED 48
```

With semantics actually loaded, **12 of 48 readings resolve a different
measure**:

| | distinct measures | distribution |
|---|---|---|
| degraded | 4 | `funded_balance` **30**, `loan_count` 8, `pipeline_amount` 6, `pipeline_case_count` 4 |
| real | **7** | `funded_balance` **18**, `loan_count` 8, `wa_ltv` 6, `wa_interest_rate` 4, `avg_borrower_age` 2, `pipeline_amount` 6, `pipeline_case_count` 4 |

Every LTV, interest-rate and borrower-age question had collapsed onto the default
`funded_balance`. The measure axis — one of the two axes that instrument exists
to verify — was **not being exercised at all** on a quarter of its readings.

## 4. Historical impact

| instrument | claims supported | could empty semantics change the conclusion? |
|---|---|---|
| `dependency_verification_temporal_compare` | C5 §4 dependency verification: dataset owner agreement, measure agreement, structural comparison periods | **yes in principle** — the measure axis was vacuous on 12 of 48 readings |
| `pipeline_stage_census` | C6 Pipeline Stage owner agreement (894/904), blast census, five-stage coverage | **no** — its measurement path is semantics-independent |

## 5. Revalidation on corrected semantics

| claim | before | after | classification |
|---|---|---|---|
| C5 dataset disagreements (as built) | 0 / 48 | **0 / 48** | **NUMERICALLY CHANGED, CONCLUSION UNCHANGED** |
| C5 dataset disagreements (view wired) | 0 / 48 | **0 / 48** | as above |
| C5 measure disagreements | 0 / 48 | **0 / 48** | as above — but now over 7 measures rather than 4 |
| C5 structural comparison periods | 48 of 48 | **48 of 48** | UNCHANGED |
| C6 stage owner agreement | 894 / 904 | **894 / 904** | **UNCHANGED** |
| C6 stage disagreement identities | 10 IDs | **same 10 IDs** | UNCHANGED |

**No material prior conclusion is invalidated.** The C5 headline numbers are
identical; what changed is that the evidence behind them is now non-vacuous where
a quarter of it previously was not. The C6 stage figures are bit-identical.

No `STOP — MIGRATION ASSURANCE BASELINE INVALID`.

## 6. Remediation

**`migration_phase0/assurance_semantics.py`** — the one way assurance gets
semantics. Delegates to production twice over (`semantics_path()` then
`load_mi_semantics`), defines no migration-specific notion of "loaded", and
**raises rather than returning a degraded dict**. No default, no fallback, no
`or {}`.

Validation is **by field name, not by count**, because a partially built registry
satisfies `len(fields) > 0` while missing exactly the field an instrument
measures. Required: `current_outstanding_balance`, `current_loan_to_value`,
`collateral_geography`, `youngest_borrower_age`, `pipeline_stage`.

Both degraded instruments now call it. Their `_env()` shrinks from 16 lines of
name-guessing to two.

## 7. Mutation tests

| mutation | result |
|---|---|
| semantics file missing | `ASSURANCE INVALID - ... does not exist` — exit **2**; the dependent instrument exits **1** |
| empty registry | `ASSURANCE INVALID - ... carry no 'fields' registry` — exit **2** |
| **partial registry** (117 of 118 fields, `pipeline_stage` removed) | `ASSURANCE INVALID - ... materially incomplete ... pipeline_stage` — exit **2** |
| loader raises (the original defect) | `ASSURANCE INVALID - ... failed to load` — exit **2** |

The partial-registry case is the one a `len() > 0` check cannot catch, and it
fails.

## 8. Durable estate-wide control

`tests/test_assurance_semantics_loading.py`, 9 tests, including three that bind
the whole directory rather than the two files fixed today:

- no instrument may name a guessed loader;
- every instrument that **calls** a semantics consumer must load authoritatively;
- no instrument may swallow a loader failure into an empty registry.

**That control took three iterations, and the failures are worth recording.** A
substring match flagged an instrument that names the consumers in a label map.
Matching the attribute `parse` then flagged three instruments calling
`ast.parse` — Python's own parser. It is now receiver-aware
(`ParsedQuestion.parse`), passes clean on all 23 instruments, and still fails
when a genuine offender is planted. A grep-shaped guard that fires on the wrong
thing is not a stricter guard.

## 9. ASSURANCE QUALIFICATIONS

Recorded, not fixed — none is required to make this audit trustworthy.

1. **Hardcoded registry paths bypass the override.** Eight
   `question_interpretation/` scripts call
   `load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")`
   directly. Same file today; divergent the moment `MI_AGENT_SEMANTICS` is set.
2. **Four instruments convert a broad `except` into an empty result set** —
   `contract_role_census.py:99`, `equivalence_portfolio_summary.py:123`,
   `filter_ownership_trace.py:146`, `route_ownership_evolution.py:124`. Same
   shape as the defect audited here, on a different axis.
3. **Most instruments print counts without asserting an expected denominator.**
   Only a minority state an EXPECTED figure the way
   `dependency_verification_temporal_compare` does ("48 of an EXPECTED 48"). A
   surface that silently shrinks to zero would still print a clean report.
4. **The re-keyed dataset guard remains a module-path whitelist** (carried
   forward from the previous task's audit).

## 10. Cost

| bucket | raw lines |
|---|---|
| `migration_phase0/assurance_semantics.py` | 137 added, 0 deleted = **137** |
| `dependency_verification_temporal_compare.py` `_env()` | 7 added, 16 deleted = **23** |
| `pipeline_stage_census.py` `_env()` | 7 added, 16 deleted = **23** |
| **assurance tooling total** | **183** |
| tests (`test_assurance_semantics_loading.py`) | 178 |
| docs (this file) | 200 |
| regenerated evidence artefact `DEPENDENCY_TEMPORAL_COMPARE.json` | 246 added, 60 deleted = 306 (output, not code) |
| **production** | **0** |

Both `_env()` rewrites are net *reductions* in executable lines: 16 lines of
loader name-guessing replaced by two lines that call the one loader.

Methodology work. **Not** to be folded into C6 migration thresholds.

## 11. Regression

Full suite, same tree both sides (the change stashed for the baseline, not a
worktree), over `tests/`, `mi_agent_api/tests/`, `mi_agent/tests/`,
`question_interpretation/tests/`:

```
baseline: 159 failed, 10313 passed, 36 skipped, 16 xfailed, 28 errors
after   : 159 failed, 10322 passed, 36 skipped, 16 xfailed, 28 errors

INTRODUCED: none
FIXED     : none
UNCHANGED : 159 pre-existing failures
```

The +9 passes are exactly the nine tests added by this task. **No production
behaviour changed** — zero production files are touched by the diff, so no answer
or refusal could move.

## 12. Status

# P0 ASSURANCE SEMANTICS LOADING CLOSED

Authoritative loader identified and unambiguous; all 23 instruments inventoried;
both degraded instruments fixed; missing, empty, partial and unloadable semantics
now fail loudly and non-zero before any analysis; mutation-proven in four ways;
both affected historical claims re-run; **no material prior migration conclusion
invalidated**.

**Recommended next task:** close qualification (2) — the four instruments that
turn a broad `except` into an empty result set. It is the same failure shape this
audit just closed on the loader axis (an instrument that cannot distinguish "no
findings" from "the measurement did not run"), it is bounded to four known sites,
and it should be settled before C6 conversion evidence is gathered.
