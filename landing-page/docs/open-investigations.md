# Open investigations

Things that went wrong once, could not be reproduced, and are therefore **parked
rather than closed**. Each entry records what was tried, what was ruled out, what
the diagnostics will print if it recurs, and — most importantly — the trigger at
which it stops being a curiosity and becomes a defect.

If you are reading this because one of them fired again: go straight to the
escalation section of that entry. Do not re-run the ruled-out list.

---

## OI-1 — Demo pack reported STALE in CI, reproducible everywhere else

| | |
|---|---|
| **Raised** | 28 July 2026 |
| **Surface** | `deploy-landing-page.yml` → `landing-page/tests/demo_pack_reproducible_test.py::test_committed_pack_matches_a_fresh_build` |
| **Status** | Parked. Occurred once. Not reproduced. |
| **Severity if it recurs** | High — see *Escalation*. |

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

### Related guards added at the same time

* The source extracts beneath the pack must regenerate byte-identically
  (`synthetic_demo/build_multibook_input.py --check`, on the deploy gate).
* Every canonical row must carry book identity
  (`test_the_canonical_carries_book_identity_on_every_row`).

Guarding the pack alone left the layer that actually carries the figures
unchecked.
