# Open investigations

Things that went wrong and were not immediately explicable. Each entry records
what was tried, what was ruled out, what the diagnostics will print if it
recurs, and — most importantly — the trigger at which it stops being a curiosity
and becomes a defect. Closed entries are kept, not deleted: the reasoning that
turned a red build into a diagnosis is worth more than the conclusion.

If you are reading this because one of them fired again: go straight to the
escalation section of that entry. Do not re-run the ruled-out list.

---

## OI-1 — Demo pack reported STALE in CI, reproducible everywhere else

| | |
|---|---|
| **Raised** | 28 July 2026 |
| **Surface** | `deploy-landing-page.yml` → `landing-page/tests/demo_pack_reproducible_test.py::test_committed_pack_matches_a_fresh_build` |
| **Status** | **CLOSED, 28 July 2026.** Recurred; the diagnostics named the cause on the first line. |
| **Root cause** | `mi_agent/portfolio_lens.py` read the word "origination" as a portfolio-scope selector. |
| **Outcome** | A correctness defect in the product, exactly as the escalation predicted — not a CI flake. |

### What happened

A manually dispatched deploy run reported:

```
STALE: .../landing-page/data/demo-pack.json differs from a fresh build.
Re-run without --check.
```

The same commit reproduced the committed pack byte-for-byte in every environment
available for testing. Five other failures in the same run were genuine and
unrelated (test assertions still pinned to the pre-multi-book dataset); those
were fixed. This entry concerns only the STALE result.

### What was tried, and ruled out

| Hypothesis | Test | Result |
|---|---|---|
| pandas version | Built under 2.2.3, 2.3.3 and 3.0.5 | Identical output in all three. Ruled out. |
| Python version | CI is 3.11; reproduced on 3.11.15 | Same version. Ruled out. |
| A file the build reads was not committed | `git ls-files` against every path the generator opens | All present. Ruled out. |
| The push did not carry everything | Fresh `git clone` **from the remote**, not the local repo | Reproduces. Ruled out. |
| Dependency set differs from local | Fresh venv, `pip install -r requirements.txt` only | Reproduces. Ruled out. |
| `anthropic` present changes the MI parser path | Installed `anthropic>=0.40.0`, rebuilt | Reproduces. Ruled out. |
| CI ran an older commit | `--check` at `af7471e`, `489a7e6`, `850d9b1` | All reproduce. Ruled out. |
| Non-determinism within one environment | Built twice consecutively, `cmp` the outputs | Byte-identical. Ruled out. |

### What was left standing

Nothing conclusive. The most likely remaining explanations are a transitive
dependency resolving differently on the runner than in a same-day fresh venv, or
a checkout-layer difference (line endings, filter attributes) that changes a
canonical CSV's bytes and therefore its embedded SHA-256. Neither was
demonstrated.

### What the diagnostics will print next time

`scripts/build_demo_pack.py --check` no longer prints a bare `STALE`. On a
mismatch it emits, to stderr, beside the failure in the CI log:

* every JSON leaf path that differs, with the committed and freshly built value
  (first 12, then a count of the remainder)
* `no leaf differs — the difference is key ordering or whitespace` when the
  documents are semantically equal
* the interpreter and pandas versions of the environment that produced the
  fresh build

That is enough to distinguish a data drift from an environment drift without
reproducing anything. Verified by corrupting a value and confirming the report.

### Escalation — read this before dismissing a recurrence

**If this fires once more, it is a determinism defect in the product, not a CI
flake.**

The reason is not process pedantry. The page's central claim, and the one the
deterministic engine exists to support, is that *the same question returns the
same number, every time and in every channel*. A demo pack that cannot be
rebuilt identically from committed inputs is that claim failing in the one place
it is directly observable. It does not matter that the difference might be
cosmetic; the guarantee is byte-level or it is not a guarantee.

On recurrence:

1. Capture the full `--check` stderr from the CI log. It names the differing
   leaves and the environment.
2. Do **not** regenerate and commit the pack to make the job green. That
   destroys the evidence and converts a determinism defect into a silent one.
3. Treat it as a defect against the engine's determinism, not against the
   landing page, unless the diff is confined to landing-page presentation
   values.

### Resolution — it recurred, and the escalation was right

The second occurrence printed, on the first diff line:

```
intents[8].answer
  committed: Direct is the largest origination channel at £15,432,544 (61.0%) of the funded book.
  fresh:     Direct is the largest origination channel at £15,432,544 (41.4%) of the funded book.
intents[8].artifacts[0].coverage
  committed: 67.9
  fresh:     100.0
```

`_DIRECT_TERMS` in `mi_agent/portfolio_lens.py` contained the bare token
`"origination"`. The question *"Show the portfolio by origination channel"*
names a **dimension**, and it was being matched as a **portfolio scope** — so a
Direct-only filter was applied to a question that had asked for a breakdown.

The consequence was not cosmetic. On the three-book platform the answer covered
81 of 118 exposures and £25.3m of £37.3m — 67.9% — while describing itself as a
share *"of the funded book"*. The acquired book's 37 exposures and £11.97m were
absent from a breakdown that claimed to be complete, and nothing in the answer
said so.

The module already forbids exactly this. Its own note on the total terms reads:
a bare word that is really an aggregation is not a scope, and treating it as one
is "a silent scope mutation, and exactly the class of defect the governed
context exists to stop." `"origination"` and `"originated"` had been added to
the direct terms without that rule being applied to them.

**Fix.** Both bare tokens removed from `_DIRECT_TERMS`. The portfolio-qualified
forms are kept — `"directly originated"`, `"new origination"`, `"newly
originated"` name the book rather than the dimension — as are `"direct"`,
`"organic"` and the book phrases, which comparisons depend on. Regression test:
`tests/test_source_portfolio_provenance.py::TestPortfolioLensResolver::test_a_dimension_named_origination_is_not_a_portfolio_scope`.

Every intent in the pack now reports 100% coverage, and the channel rows sum to
£37,270,061.47 — the sponsor total to the penny.

**Why it looked environment-dependent.** It was not a build-environment
difference at all. The committed pack had been generated while the defect was
firing; a fresh build in another environment did not reproduce that particular
scoping. Chasing pandas and Python versions was the wrong tree, and the earlier
"could not reproduce" conclusion was wrong — it was reproducible, in the pack
that had been committed. The diagnostics existed precisely because that
conclusion could not be trusted, and they are what closed it.

**Lesson recorded:** a `--check` that reports only pass/fail on a large
generated artefact is not a control, because the first instinct on a red build
is to regenerate. The instruction not to regenerate, plus a diff, is what turned
a second red build into a ten-minute diagnosis.

### Related guards added at the same time

* The source extracts beneath the pack must regenerate byte-identically
  (`synthetic_demo/build_multibook_input.py --check`, on the deploy gate).
* Every canonical row must carry book identity
  (`test_the_canonical_carries_book_identity_on_every_row`).

Guarding the pack alone left the layer that actually carries the figures
unchecked.
