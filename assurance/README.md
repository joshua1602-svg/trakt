# MI Agent System Assurance Programme

Pre-production go-live assessment of the MI Agent, executed ahead of first
client onboarding. This directory is the version-controlled record of the
assurance work: what was verified, how it was verified, what failed, and the
resulting go-live recommendation.

This is an assurance artefact set, not a feature. Nothing in this directory is
imported by production code.

## Layout

| Path | Contents |
|------|----------|
| `architecture/` | Verified runtime architecture map, runtime dependency map, duplicate calculation register, registry consumption findings |
| `question_bank/` | The governed 1,000-question bank (version-controlled, machine-readable) |
| `expected_results/` | Expected-result specifications for the question bank |
| `fixtures/` | Deterministic edge-case mutation fixtures and base-fixture manifest |
| `oracle/` | Independent numerical oracle (must not import production calculation code) |
| `runners/` | Automated assurance runners (parser / recogniser / workflow / service / API layers) |
| `reports/` | Machine-readable results, human-readable reports, performance report, deployment checklist, go-live scorecard |
| `defect_register/` | Live defect register with severity, root cause, disposition |

## Ground rules applied

* Production calculation code is never used as its own oracle.
* Expected values come from hand calculation, transparent reference functions,
  or independent dataframe computation in `oracle/`.
* Structured output is the primary test target; rendered prose is secondary.
* Failures are recorded, classified and either fixed (minimal, isolated,
  regression-tested) or explicitly dispositioned — never hidden by weakening
  expectations.
* The base synthetic fixture is never edited to make tests pass; edge cases are
  covered by explicit mutation fixtures in `fixtures/`.

## How to run

```bash
# Full assurance run (all layers) against the base synthetic portfolio
python -m pytest assurance/runners -q

# Question-bank execution with machine-readable results
python assurance/runners/run_question_bank.py --out assurance/reports/results.json
```

See `reports/go_live_scorecard.md` for the final gate assessment and
recommendation.
