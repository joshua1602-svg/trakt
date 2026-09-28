# Live data-source provenance — and a correction to my own earlier finding

    READ-ONLY. No code, Azure configuration or data changed. No bank re-run, no
    model call. Everything below comes from the committed certification evidence
    and from code at the SHA the preflight verified as deployed.

## The correction, first

At `4ebb02a4` I wrote:

> "the tape the certification queried was produced by
> `synthetic_demo/run_multibook_pipeline.sh` under client_id `alderbridge_demo`"

**That statement is FALSE.** It was `INFERRED_FROM_LOCAL_FIXTURE` — I matched a local
file whose name and shape resembled the dataset label the service reported, and did
not check the identifiers the service itself published. The live runtime states its
own source in every envelope, and it does not match that file:

    deployed service reported        local synthetic multibook tape
      row_count      958               rows              118
      content_hash   2c26d0e65340…     sha256            ddea1c505459…
      source_portfolios                source_portfolio_id values
        acquired_001, direct_001         alp_origination, alp_acquired,
                                         spv1_sponsored

Different row count, different content hash, different source portfolios. The two
are not the same dataset and never were.

**A consequence I must also flag:** the finding in `2243eafe` that "8 rows satisfy
London + Direct + age over 75" was computed against that same local 118-row file. It
says nothing about the 958-row dataset the service actually queried, and the
`FIELD_CONNECTIVITY` classification it supported for Q04A/B/C is therefore
**unproven**, not disproven. It needs re-deriving against the live dataset.

---

# A. What the live certification actually called

    CERTIFICATION_BASE_URL       https://app.traktinfra.io/api
    CERTIFICATION_ENDPOINT       /mi/query
    CERTIFICATION_PORTFOLIO_ID   ERE/2026-06-30
    DEPLOYED_SHA                 5c436961ebe279bc0820ab006867b9f8a869bde2
    AUTHENTICATED_LIVE_SERVICE   YES

Evidence: `preflight_result.json` records `base_url`, a build read of `/` returning
HTTP 200 with `commit = 5c436961…`, an authenticated preflight at HTTP 200, and two
successive liveness reads. Every one of the 135 records carries a service-issued
`governance.requestId` (e.g. `req_12c0dcc8b20e4f67`), a `tenantId` and a
`portfolioId` — values a local execution would not mint. The evidence sink read over
Kudu showed 409 pre-existing records written by the deployed app.

**The 135 were sent over HTTP to the deployed production API. They were not executed
locally.**

---

# B. The funded data source the deployed API actually loaded

    FUNDED_SOURCE_KIND        platform_canonical   (KIND_PLATFORM_CANONICAL)
    FUNDED_SOURCE_URI_OR_PATH NOT_ESTABLISHED — the envelope publishes a label and a
                              fingerprint, not a URI, and the App Service
                              application settings are not readable from here
    FUNDED_CLIENT_ID          tenantId "ERE"   (no separate client_id is published
                              on the funded snapshot descriptor)
    FUNDED_PORTFOLIO_ID       ERE/2026-06-30
    FUNDED_REPORTING_DATE     2026-06-30
    FUNDED_SOURCE_FINGERPRINT snapshot_id   platform_canonical_typed@2c26d0e65340
                              content_hash  sha256:2c26d0e65340ef515407a382f6350311
                              row_count     958
                              approval_state approved
    FUNDED_LINEAGE            source_portfolios = ["acquired_001", "direct_001"]
                              source_base = platform_canonical
                              policy.runtime_mode = production
                              policy.fixture_source = false
                              policy.data_approved = true

**All 134 measured cases report one identical funded snapshot.** No case saw a
different dataset.

## Which resolver branch was active

`data_source.resolve_data_source()` selects in this order:

    1. MI_AGENT_ANALYTICS_DATASET      -> prepared_explicit
    2. _resolve_platform_canonical()   -> platform_canonical   <== ACTIVE
    3. MI_AGENT_CENTRAL_TAPE           -> central_tape
    4. MI_AGENT_DATA_CSV               -> explicit_csv
    5. synthetic_demo fallback         -> synthetic_demo       (last resort)

The runtime emits a **distinct** kind per branch — `KIND_SYNTHETIC_DEMO =
"synthetic_demo"` is its own value — and the service reported
`platform_canonical`, with `policy.fixture_source: false` and
`runtime_mode: "production"`.

**ACTIVE BRANCH = MI_AGENT_PLATFORM_URI / platform canonical (branch 2).**
The synthetic-demo fallback is the last resort and did not fire; had it fired the
envelope would say so in its own vocabulary.

This is runtime metadata proving branch selection, not an inference from a local
file — which is what the earlier claim lacked.

## Context: the data was refreshed between baseline and release candidate

    historical run @ 00bb3e9d   platform_canonical, 958 rows,
                                content_hash 56015daf6d32…
    certification  @ 5c436961   platform_canonical, 958 rows,
                                content_hash 2c26d0e65340…

Same kind, same shape, same portfolios, **different content**. Worth knowing when
reading any baseline comparison, and not something either run claimed.

---

# C. Dashboard parity

    DASHBOARD_FUNDED_SOURCE   NOT_ESTABLISHED
    QUERY_FUNDED_SOURCE       platform_canonical, 958 rows,
                              sha256:2c26d0e65340…, [acquired_001, direct_001]
    SAME_FUNDED_SOURCE        NOT_ESTABLISHED

The certification called `/mi/query` only, plus `/` for liveness and the Kudu
evidence sink. **No dashboard GET endpoint was called during it**, and the funded
fingerprint `2c26d0e65340` appears nowhere in the repository except the
`/mi/query` evidence files themselves. There is no dashboard-side capture to compare
against.

This is establishable cheaply and was not: one authenticated GET of the dashboard's
own summary endpoint, compared on `content_hash` / `row_count` / `source_portfolios`,
would settle it outright. **It is not claimed either way here.**

---

# D. The pipeline source, traced separately

    PIPELINE_SOURCE_KIND       NOT_ESTABLISHED
    PIPELINE_SOURCE_URI        NOT_ESTABLISHED
    PIPELINE_CLIENT_ID         NOT_ESTABLISHED
    PIPELINE_REPORTING_DATE    2026-01-12   (ESTABLISHED — stated in the answers:
                               "At the weekly extract of 12 January 2026…";
                               stage movements between 2026-01-05 and 2026-01-12)

The envelope's `governance.snapshot` describes the **funded** source only; it is
identical on pipeline questions and carries no pipeline descriptor. What is
established is that the pipeline route ran (`metadata.route = "pipeline_summary"`,
`requirements: ["pipeline_dataset"]`) and returned a full stage table — KFI 2,120
cases / £455.2m, Offer 179 / £29.9m, Application 66 / £11.9m, Completed 135 / £18.3m,
Withdrawn 279 / £47.4m.

    MIXED_SOURCE_DEPLOYMENT_POSSIBLE          YES
      funded and pipeline resolve through entirely separate chains —
      resolve_data_source() vs _pipeline_root()/MI_AGENT_PIPELINE_URI — so nothing
      in the design forces them to the same client or the same vintage.
    MIXED_SOURCE_DEPLOYMENT_ACTUALLY_OBSERVED NOT_ESTABLISHED
      the funded side is a production-mode approved platform canonical, not a
      demo source, so the specific "funded = synthetic / pipeline = real ERE" mix
      is NOT what happened. Whether the two belong to the same client cannot be
      answered without the pipeline descriptor.

What IS observable is a **vintage gap**: funded at 2026-06-30, pipeline at
2026-01-12 — five and a half months apart, on the same portfolio, in one deployment.

---

# E. The alderbridge claim

    CLASSIFICATION = FALSE
    (and the route by which it was reached: INFERRED_FROM_LOCAL_FIXTURE)

Decisive evidence, all from the service's own published identifiers:

    1. row_count 958 vs the local file's 118
    2. content_hash sha256:2c26d0e65340… vs the local file's ddea1c505459…
    3. source_portfolios [acquired_001, direct_001] vs the local file's
       [alp_origination, alp_acquired, spv1_sponsored]
    4. source_kind "platform_canonical", while the runtime has a distinct
       "synthetic_demo" kind it did not report
    5. policy.fixture_source = false, runtime_mode = "production"

Any one of 1–3 alone falsifies it.

---

# FINAL ANSWERS

    LIVE_135_HIT_PRODUCTION_API                YES
    LIVE_135_USED_DASHBOARD_BACKEND            NOT_ESTABLISHED
    LIVE_135_FUNDED_DATA_WAS_OCC_ERE           NOT_ESTABLISHED
      It is a production-mode, approved platform canonical for tenant ERE with ERE
      portfolio identifiers — but nothing in the evidence states whether the tape
      was produced by the current OCC path, and the absent canonical_region_* and
      product columns remain unexplained by this check.
    LIVE_135_FUNDED_DATA_WAS_ALDERBRIDGE_DEMO  NO
    LIVE_135_PIPELINE_DATA_WAS_REAL_ERE        NOT_ESTABLISHED
      Real pipeline figures at a stated extract date; no client identifier published.
    DASHBOARD_AND_QUERY_USED_SAME_FUNDED_DATA  NOT_ESTABLISHED

# What happened, in ten lines

1. The 135 were genuinely live: HTTPS to app.traktinfra.io/api/mi/query, bearer
   authenticated, service-issued request ids, deployed SHA verified at 5c436961.
2. The deployed API loaded a production platform canonical: 958 rows, approved,
   fingerprint 2c26d0e65340, source portfolios acquired_001 and direct_001.
3. That is branch 2 of the resolver — the governed platform canonical — not the
   synthetic-demo fallback, which is last in precedence and has its own kind.
4. My earlier claim named a local 118-row demo file with different portfolios and a
   different hash. It was pattern-matched on filename shape and it is false.
5. I did not check the identifiers the service published in every envelope, which
   is the one place that could settle it, and which I had already read for other
   fields in this same evidence.
6. The product and region findings that rest on live evidence still stand.
7. The specific row-level finding computed against the local file does not, and is
   downgraded to unproven pending re-derivation against the live dataset.
8. Pipeline ran on a separate resolver chain at a 2026-01-12 extract against funded
   at 2026-06-30 — a real vintage gap, separately worth attention.
9. Dashboard parity was never measured; one authenticated GET would settle it.
10. Nothing here changes the frozen certification result.
