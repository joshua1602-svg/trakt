# Pipeline MI — Historical completion-model evidence & lineage

Explainability refinement: surface how many weekly pipeline files the completion
estimate relies on, and clarify the three distinct dates.

## historicalModelEvidence

Exposed on `/mi/pipeline/snapshot`, `/mi/forecast/snapshot`, and
`/mi/workspace/view?view=pipeline|forecast` (under each view's `lineage`):

```
historicalModelEvidence:
  weeklyFilesUsed            # weekly pipeline files that contributed
  weeklyFileNames            # their basenames
  observationWindowStart     # earliest weekly extract date
  observationWindowEnd       # latest weekly extract date
  historicalRowsUsed         # case-rows scanned across all weekly files
  trackedCaseCount           # distinct cases tracked across snapshots
  observedCompletionCount    # distinct cases ever seen COMPLETED
  stableIdentifierUsed       # e.g. "pipeline_case_identifier (Account Number)"
  stagesUsingHistoricalRates # stages with a trusted empirical rate
  stagesUsingConfigFallback  # active stages observed but below MIN_OBSERVATIONS
  excludedStageCounts        # {WITHDRAWN: n, UNKNOWN: m}
  completionProbabilityBasis # historical_observed | stage_config | mixed_* | unavailable
```

Top-level `completionProbabilityBasis` and `historicalModelEvidence` are now
populated on the forecast envelope (previously null).

## Three distinct dates (do not conflate)

| Date | Field | Meaning |
| --- | --- | --- |
| Funded reporting date | `fundedReportingDate` | funded book cut-off (e.g. `2025-11-30`) |
| Pipeline snapshot as-of | `pipelineAsOfDate` | the SELECTED (latest) weekly extract (e.g. `2025-12-01`) |
| Historical observation window | `observationWindowStart` → `observationWindowEnd` | span of ALL weekly files used for the rates (e.g. `2025-10-06` → `2025-12-01`) |

## Source scope clarification (issue #5)

`pipelineSourceFolderDate` is the **folder that contains the materialised weekly
files** (the source scope folder). When a run's weekly extracts are all
materialised into one folder named for the earliest cut (e.g. `2025-10-01`), that
folder date is `2025-10-01` even though the latest weekly file is `2025-12-01` —
so it is **not** the current snapshot date.

- **Current pipeline snapshot** = the latest weekly extract → `pipelineExtractDate`
  / `pipelineAsOfDate`.
- **Historical completion modelling** = ALL weekly files in scope →
  `observationWindow{Start,End}` + `weeklyFileNames`.
- `pipelineSourceFolderDate` = the scope folder (earliest/source folder), kept for
  provenance and shown distinctly in the lineage.

## Source filtering (issue #6)

The historical model consumes only governed pipeline/KFI files. Discovery globs
already require `M2L*KFI*` / `*KFI*Pipeline*`; on top of that,
`pipeline_contract._is_governed_pipeline_file` excludes funded/funder files by
name (`funder`, `principal and interest`, `central lender`, `loan extract`,
`funded tape`) so a `Funder Principal And Interest_test.csv` sitting under
`output/pipeline/` is never counted as weekly pipeline model evidence.

## UI

The Pipeline and Forecast "How calculated" lineage panel shows a concise
**Completion model evidence** line (weekly files · historical rows · tracked
cases · observed completions), the observation window, the basis, and the
identifier — with the three dates listed separately in the expandable detail.
