# accept-target-advice — apply LLM target-first advice into an approved 34 file

## Problem

The target-first LLM advisor reports `advised=N` and writes
`36_target_first_llm_recommendations.*`, but `34_target_first_decisions.yaml`
stays pending/null by design (advisory only). Operators had no simple way to turn
that advice into the approved decision file the workflow consumes on rerun, so
the product felt broken ("the LLM advised but nothing changed").

## Feature

A new CLI command turns *advised* recommendations into an approved 34 file:

```bash
python -m engine.onboarding_agent.cli accept-target-advice \
  --project-dir onboarding_output/client_001/mi_2025_10 \
  --recommendations onboarding_output/client_001/mi_2025_10/36_target_first_llm_recommendations.json \
  --out onboarding_output/client_001/mi_2025_10/34_target_first_decisions_approved.yaml \
  --approved-by Joshua
```

Then rerun onboarding with `--target-first-decisions <approved file>`.

### Behaviour (engine/onboarding_agent/accept_target_advice.py)

1. Reads `34_target_first_decisions.yaml` and `36_target_first_llm_recommendations.json`
   from the project dir (override with `--decisions` / `--recommendations`).
2. Applies a recommendation only when `llm_advice_status == advised`. The
   statuses `invalid_response`, `parse_failed`, `skipped_budget`, `no_advice`,
   `requires_operator_review` are skipped unless explicitly opted in with
   `--allow-status`.
3. `requires_operator_review` / `merge_or_reconcile` / `reject_recommendation` /
   `defer` actions are never auto-approved (opt in with `--allow-action`).
4. Maps the advisor action to the `selected_action` vocabulary that
   `target_first_decisions.apply_decisions` consumes:
   `provide_source_mapping` / `choose_alternative` (map a source column),
   `confirm_selected` (confirm the source), `configure_static_value`
   (set a derivation/default), `confirm_default_or_nd` (confirm an ND/default),
   `mark_not_applicable`. It populates `selected_source_file`/`column`,
   `configured_value`, `default_confirmed`, `not_applicable_confirmed`,
   `operator_note`, and sets `status: approved`, `approved_by`, `approved_at`.
5. **Validates** that a recommended source column is within the field's allowed
   candidates (from `28a_target_coverage_matrix.json`); an invented column is
   skipped, never approved.
6. Decisions without usable advice stay `pending`.
7. Emits a summary: approved / pending / skipped (with reasons).
8. Never touches 28a/28c — only the new approved 34 file is written, and that
   approved file remains the single thing the workflow consumes on rerun.

A `--min-confidence` flag can skip low-confidence advice.

## Advisory wording

The advisory note is now explicit in the artefacts an operator sees — the 36
summary, the HTML review pack, and the workflow console / `next_operator_action`:

> LLM recommendations are advisory. To apply them, run `accept-target-advice`
> (writes `34_target_first_decisions_approved.yaml`) or approve manually, then
> rerun with `--target-first-decisions`.

## Files

* `engine/onboarding_agent/accept_target_advice.py` *(new)* — accept logic.
* `engine/onboarding_agent/cli.py` — `accept-target-advice` subcommand.
* `engine/onboarding_agent/target_first_llm_advisor.py`,
  `review_pack_builder.py`, `workflow.py` — advisory wording + command hint.
* `tests/test_onboarding_accept_target_advice.py` *(new)* — 11 tests.

## Tests

`tests/test_onboarding_accept_target_advice.py` — **11 passed**, proving: valid
source-mapping / `mark_not_applicable` / `configure_static_value` advice → approved
decisions; `invalid_response` not applied; `requires_operator_review` not
auto-approved; pending stays pending; out-of-candidate source skipped; the
generated approved file is applied by `apply_decisions` on rerun; 28a/28c are
not mutated. Review-pack / workflow / advisor suites (49 tests) remain green.
