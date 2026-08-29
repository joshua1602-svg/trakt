# Production freeze tag — recreation record

The annotated tag below was created locally at the production SHA and **could not
be pushed**: this environment's git proxy rejects tag pushes. The remote carries
zero tags — the earlier `mi-launch-baseline-7of7` from a previous programme is
local-only for the same reason — and this container is ephemeral, so the tag will
not survive it.

This file exists so the tag can be recreated **verbatim, at the same SHA**, by
someone with push rights. Nothing about the freeze depends on the tag existing;
the evidence is in the commits.

    tag     mi-query-agent-go-live-freeze
    commit  23804de   (the production SHA intended for deployment)

To recreate and push:

```sh
git tag -a mi-query-agent-go-live-freeze 23804de -F migration_phase0/MI_FREEZE_TAG.md.msg
git push origin refs/tags/mi-query-agent-go-live-freeze
```

Do not move the tag once it exists.

---

## The annotation, verbatim

```
MI Query Agent — production freeze

This tag marks the FROZEN QUERY ARCHITECTURE at the SHA intended for
deployment. It does not assert that go-live was cleared.

Production SHA   23804de
Acceptance SHA   6d31fe9 (oracle and evidence only; production tree identical)

Shipping configuration, 166-question acceptance bank:

    CORRECT                     136
    CORRECTLY DECLINED           16
    NO CHECKABLE TRUTH            0
    DECLINED BUT ANSWERABLE      12
    WRONG                         2

    correct or correctly declined   152 of 166   91.6%

Governed engine alone: 127 correct, 4 wrong.
Reach recovery: 8 of the original 16 existing-capability cases — 50%.
Frozen regression manifest: 85 failing/erroring names, name for name.
Provider unavailable: 0 answered, 0 wrong, 0 whole-book fallback.
Semantic coverage census: 1,612 questions, 0 answering questions carry an
unaccounted concept.

Accepted residual defects, both deterministic rather than intermittent:
    Q04C  correct 24-loan population, loan-level groups where a scalar total
          was asked for
    Q19A  five-period progression where a last-month delta was asked for

VERDICT AT THIS SHA

    GO-LIVE DATA CONTRACT BLOCKER — QUERY ARCHITECTURE REMAINS FROZEN

The query architecture is sound and frozen. The live-data half of go-live could
not be demonstrated: no curated ERE MI tape exists in the environment this was
closed out in. The only ERE artefact present is an ESMA Annex 2 regulatory
OUTPUT whose current_principal_balance is ND-coded on every row, and which the
MI contract does not read.

To clear it: curate the live ERE source extract into a governed tape, load the
ERE client configuration, and re-run sections 5A-5C of MI_GO_LIVE_CLOSE_OUT.md
unchanged. The Query Agent needs no modification for that.

Do not move this tag.
```
