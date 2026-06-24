# Pipeline MI — Phase 3 Refinement Review (watchlist, top-10, historical rates)

> Part 0 review completed before patching. Scope: clearer watchlist classification,
> top-10 broker/channel capping, and an initial deterministic historical
> completion-rate model. No NL-parser/scenario expansion, funded MI unchanged,
> SSoT preserved (pipeline separate, forecast backend-derived, probabilities
> governed or empirically derived).

## Where things currently live

| Concern | Location |
| --- | --- |
| Watchlist items | `forecast_bridge.build_pipeline_watchlist` |
| Completion probability assignment | `pipeline_prep._derive_probabilities_and_amounts` |
| Expected completion date derivation | `pipeline_prep._derive_expected_completion` |
| Stage / status normalisation | `pipeline_prep._normalise_stage` (`_STAGE_CANON`) |
| Broker/channel breakdowns | `pipeline_contract._dimension_breakdown` → `compute_pipeline_snapshot` |
| Stage probability config | `config/client/pipeline_expected_funding.yaml` via `_stage_probabilities()` |
| Cross-week case identity | contract aliases `pipeline_case_identifier` (Account Number) / `application_identifier` (KFI Number) |
| Weekly history availability | discovery groups weekly files per scope (`weekly_files`); governed sources materialised under `output/pipeline/<folder>/` |
| Forecast consumes weighted amount | `forecast_bridge.compute_forecast_bridge` |

## What currently drives the symptoms

- **"X cases without a completion probability"** — `completion_probability` is NaN.
  It is NaN for any stage absent from the config map: that means **WITHDRAWN**
  (intentionally excluded) *and* **UNKNOWN** stages, lumped together with any
  genuinely-missing active case. Withdrawn cases are already excluded from the
  weighted sum (NaN × amount → dropped), but the watchlist words it as a generic
  WARNING.
- **"X cases without an expected completion date"** — `expected_completion_date`
  is NaT. NaT for WITHDRAWN/UNKNOWN (no `stage_days_to_fund` entry) and for cases
  with no base date — again conflated.
- **Weighted expected funded by completion month** — `_expected_completion_breakdown`
  groups by `expected_completion_month`; withdrawn/no-date cases simply have no
  bucket.
- **Broker/channel bars** — `_dimension_breakdown` returns **all** brokers
  uncapped; the frontend `BarList` renders every row → the section runs down the page.

## Plan

1. **Watchlist** — classify probability-missing and completion-date-missing rows by
   stage: WITHDRAWN/inactive → INFO ("excluded from weighted forecast"); active
   (KFI/APPLICATION/OFFER) missing → WARNING; UNKNOWN stage → separate warning with
   the offending stage/status values. Backend metadata carries counts, affected
   stages, excluded-vs-missing, and whether weighted.
2. **Top-10** — cap broker/channel (and any >10-cardinality categorical) at 9 + an
   aggregated **Other** row (amount, count, share, brokers-included) in the API;
   keep the full breakdown in `*BreakdownFull`. Frontend renders the capped list.
3. **Historical model** — `pipeline_history.build_historical_completion_model`
   tracks cases across weekly snapshots (by KFI/account id), derives empirical
   completion rate + timing per stage, and gates on a minimum observation count.
   Probability hierarchy: row-level → historical (if sufficient) → config →
   excluded/withdrawn → missing. Row-level `completion_probability_source` and an
   overall `completionProbabilityBasis` (`historical_observed` / `stage_config` /
   `mixed_historical_and_config` / `unavailable`).
4. **Forecast disclosure** — gross pipeline, excluded amount, amount weighted by
   historical vs config, blended weighted conversion (= weighted / gross active),
   probability basis.

Funded MI, the funded tape, and the NL parser are untouched.
