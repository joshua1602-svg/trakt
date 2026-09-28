# Slice 3 production canary — result

Run `34629201862`, six questions, `ERE/2026-06-30`, 17:42–17:45Z.

```
EXPECTED_SHA = 853231e3
SERVED_SHA   = 853231e3601b30578e7b146a9f7d6ad389e0d2e2   CONFIRMED (2 reads, >= 15s apart)
```

---

## 1. Provenance, and the first read that looked like a mismatch

The first `/health` read returned `a2c36327` — the Slice 2 build — and the run
stopped, asking nothing. That read was taken **35 seconds** after OneDeploy
completed (deploy 17:28:33, read 17:29:08). App Service runs a remote build and
restarts after deployment, so the previous build answering that early does not
distinguish a failed deploy from an unfinished restart. Re-read after a settle
window: `853231e3` twice, ≥15s apart, and again after the canary restart. The
live catalogue matched what the bank assumes.

**The deployed ref is `853231e3`, and the product under test is `e6e16c63` byte
for byte.** The tag could not be created (push credentials are branch-scoped;
403 on the tag ref), so the branch head was deployed. All 17 paths in
`deploy/trakt-mi-api/package_contents.txt` hash identically at both commits
(`43ede9541c7489bb915856ad15a0341b390c1aeb6a15e759ebcb663d5b7d5434`), and
`deploy/trakt-mi-api/requirements.txt` is the same blob. The only two files that
differ are the acceptance harness and its workflow, neither of which is staged
into the artefact.

---

## 2. What the six questions did

| | question | population | decision | verdict |
|---|---|---|---|---|
| S3-P1 | the back book balance | `base=funded`, no seasoning | **NEW** | INCONCLUSIVE — harness |
| S3-P2 | acquired drawdown, LTV > 50% | `base=funded`, `lens=acquired`, all three predicates proved | **NEW** | INCONCLUSIVE — harness |
| S3-P3 | funded balance for the ALP back book | no plan (CLARIFY) | LEGACY_FALLBACK | FAIL as specified |
| S3-P4 | ALP Acquired Back Book each month | no plan (CLARIFY) | LEGACY_FALLBACK | FAIL as specified |
| S3-N1 | what is in the front book | `base=pipeline` | LEGACY_FALLBACK | **PASS** |
| S3-N2 | how much of the increase came from the acquired back book | `base=funded`, `lens=acquired` | LEGACY_FALLBACK | **PASS** |

```
silent_scope_drops 2   silent_scope_widenings 2   silent_scope_narrowings 0
model_authored_physical_source_bindings 0   cross_client_scope_selections 0
semantic_coverage_refusals 0   execution_errors 0   exception_escapes 0
```

The two drops and two widenings are the harness's accounting of P3/P4's absent
named scope — the same single cause counted on two axes, not four events.

---

## 3. P1 and P2 were not measured, and that is my harness's fault

Both **served NEW with the correct population**, and for P2 the only reported
problem was the reconciliation — meaning `lens=acquired`, the scope predicate,
and all three proved fields (`erm_product_type`, `current_loan_to_value`,
`source_portfolio_type`) passed. The Slice 3 assertion held.

What failed is my `scalar_of`, which required *exactly one* numeric KPI in the
response. A governed envelope carries several — a balance and a loan count — so
it extracted none. The estate already owns the correct reader, and its docstring
names this exact trap:

> `corresponds()` — *"scalar plans on the KPI whose field name the bound spec
> implies — not the first KPI, which is the loan count and read 130 for a
> question about £37.9MM of balance."*

I wrote a naive extractor instead of reusing it. The product's behaviour on P1
and P2 is unmeasured, not wrong, and the honest verdict for both is
INCONCLUSIVE rather than FAIL.

---

## 4. P3 and P4 are a configuration gap, not a product defect

Both clarified and fell back. The cause is upstream of the code:

```
mi_agent.portfolio_metadata.registry_path()          -> None
load_portfolio_metadata("ERE")                       -> {}
load_portfolio_metadata("ere_funding_uk")            -> {}
config/client/  holds only  portfolio_registry.example.yaml
```

`load_portfolio_metadata`'s own docstring says it: *"Returns `{}` when no
registry is configured — **the supported and current state for ERE**."*

So this client declares **no governed source portfolio labels or aliases**.
`get_source_portfolios` had no books to show, the model was shown none, and it
raised a blocking ambiguity rather than guessing at "the ALP back book" — which
is precisely the fail-closed behaviour Slice 3 was built to produce, and the
same behaviour the model-boundary run measured before the affordance existed.

The named-source axis is proven offline and against a fixture registry. It
cannot be exercised in production until ERE's portfolio registry declares names.
**No change to product code would make P3/P4 pass; onboarding data would.**

---

## 5. The two negatives passed, and they are the ones that matter most

**S3-N1** — `"What is in the front book?"` read as `population.base = pipeline`,
was **not executed**, produced no receipt, and was not served. This is the defect
the population-base gate was built for, refusing in production exactly as the
offline controls said it would.

**S3-N2** — the attribution question kept `base=funded` and `lens=acquired` and
did not degrade into a plain scoped balance. Scope did not cost the capability.

---

## 6. Verdict

```
SLICE_3_PRODUCTION_CANARY = INCONCLUSIVE
SLICE_3 = NOT CLOSED
```

Not PASS: the brief requires P1–P4 to serve NEW and reconcile, and P3/P4 did
not serve NEW. Not a flat FAIL either: nothing in the run shows the product
behaving wrongly. Two positives served NEW with correct scope and went
unmeasured through my error; two could not be exercised without governed
portfolio names; both negatives passed.

**The canary is still on.** `MI_AGENT_PLAN_SERVE` must be set back to `off` on
`trakt-mi-api` — ARM access this automation does not hold.
