# Live change_form interpretation gate — PROVIDER_BLOCKED

```
BASELINE_SHA          665176864d2f5dbceedc277e351c5d2cd978777f
TEST_BRANCH           claude/live-change-form-gate
PRODUCT_CODE_CHANGED  NO
BANK_ID               live_change_form_gate_v1
BANK_SHA256           4e6fca5920ea07b8c5e56321657ae47de9f2953c3c639184ab910570afd87c59
MODEL                 claude-opus-5   (requested; never reached)

LIVE_MODEL_CALLS_ATTEMPTED    0
LIVE_MODEL_CALLS_SUCCESSFUL   0
PROVIDER_FAILURES             1   (pre-call credential check)
```

## What was completed

STEP 0 and STEP 1 both hold, and the harness is proved at zero cost.

| | |
|---|---|
| branch rooted exactly at the baseline | `HEAD == 66517686`, `is-ancestor` true, `diff 66517686..HEAD` empty |
| dependencies | `requirements.txt` unmodified, installed into a clean 3.11 venv — the equivalent of the workflow's fresh `setup-python`, because Debian-managed packages block an in-place install |
| harness imports | `compiler`, `intent`, `opus_interpreter`, `vocabulary` all import; vocabulary 2.2.0, compiler 1.0.0, `CHANGE_FORMS` all four present |
| existing slice 3 harness dry-run | `model_boundary_run.py --dry-run` → 4/4 adjudicated, 3/3 compile, manifest hash verified, zero model calls |
| this gate's harness dry-run | 6/6 adjudicated, 6/6 compiled |

The dry-run also records, incidentally, that the Sprint A routing reaches four
distinct execution contracts from the four forms:

```
material_summary  -> period_movement / summary      (canonicalised)
metric_delta      -> period_movement / movement     (NOT canonicalised)
attribution       -> funded_bridge   / bridge
level_comparison  -> generic_analysis / compare
```

That is a deterministic observation about the compiler, made from authored
stand-in payloads. It is **not** evidence about live interpretation, which is
what this gate exists to measure and what did not run.

## Why it stopped

No Anthropic credential exists in this environment, so the gate could not ask
the model anything.

```
ANTHROPIC_API_KEY      unset
ANTHROPIC_KEY          unset
ANTHROPIC_API_TOKEN    unset
ANTHROPIC_AUTH_TOKEN   unset
CLAUDE_API_KEY         unset
no .env, no ~/.anthropic
```

Confirmed against the provider rather than inferred from the environment: an
unauthenticated `POST https://api.anthropic.com/v1/messages` returns **401**.
`api.anthropic.com` is in the session proxy's `noProxy` list, so the proxy
neither relays nor injects credentials for it.

`AnthropicInterpreterClient.available` is false, and the runner therefore refused
**before** the first call. No result file was written: a harness that records
eighteen fabricated failures produces an artefact that looks like a measurement
and is not one.

## What was deliberately not done

A session-ingress token is present in this container. It is the Claude Code
harness's own credential for serving this conversation — not the product's API
credential, and not the configuration this gate specifies. The slice 3 workflow
sources the real one from `secrets.ANTHROPIC_API_KEY`. Borrowing the harness's
auth channel to run a paid eighteen-call benchmark would produce a number
obtained from the wrong configuration, so it was not used.

No question was asked a second way, no cheaper model was substituted, no gate was
relaxed, and no product code was touched.

## How to complete the gate

Either supply `ANTHROPIC_API_KEY` to this environment and run:

```
python due_diligence/evidence/live_change_form_gate/change_form_run.py
```

or run the same file on the CI path that already holds the secret — the
dependency and credential shape of `.github/workflows/slice3-model-boundary.yml`,
which installs `requirements.txt` and exports `secrets.ANTHROPIC_API_KEY`.

The bank is pinned. The runner verifies `change_form_bank_manifest.sha256` on
every start and refuses to run if the manifest has moved, so the expectations
committed here are the expectations any later run is scored against.

## One discrepancy in the gate brief, recorded and not resolved

The brief states `SCOPE_PRESERVED = 3/3`. The question set contains **four**
scope-bearing questions: CF05 direct, CF10 acquired, CF14 direct, CF18 acquired.
All four are pinned and will be scored, and the result will be reported as x/4.
No case was dropped to make the count agree; a gate is not relaxed to fit an
arithmetic slip.

```
LIVE_INTERPRETATION_GATE = PROVIDER_BLOCKED
```
