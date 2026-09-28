# V3 release certification — frozen, and stopped before Q1

    BANK_ID                    MI_135_RELEASE_CERTIFICATION_V3
    QUESTION_COUNT             135
    BANK_SHA256                9624a0a8f2e7d1c3e59ccf88ad2698be01950fce6e03c111bc428f8272b52122
    EXPECTATION_OVERLAY_SHA256 a842479642ec2202a61e303df44eeb952fccac5c0fed0d8fbd54f472111cc072
    SCORING_CONTRACT_SHA256    bdec83f9f7ad86de3343de72677e672a33b3a40c825b8c7f0afe9360ff3e7be2
    COLLECTOR_SHA256           ebb17776db847bb0402a930bf9fbf994a5d9090beb59200eea7b0ece2712f143
    CERT_MANIFEST_SHA256       058fdcd9df49f4c972be44d7a90c152916289c1d5c6ec248d03133ccebad36b0
    PRODUCT_SHA                5c436961ebe279bc0820ab006867b9f8a869bde2

    OLD_V2_EVIDENCE_MODIFIED   NO

The questions and the overlay are byte-identical to V2 and to the historical run.
Only the two test instruments changed, and both new hashes are pinned above.

## The replay gate, before anything is spent

    historical evidence through the repaired scorer
    FULLY_CORRECT 52   PARTIALLY_CORRECT 56   HONEST_REFUSAL 25   INCONCLUSIVE 2
    PASSED

## Preflight, run 34773935403 at 18:13:42Z

| check | result |
| --- | --- |
| certification manifest | PASS `058fdcd9…` |
| unmodified: questions | PASS `9624a0a8…` |
| unmodified: expectation_overlay | PASS `a8424796…` |
| unmodified: scoring_contract | PASS `bdec83f9…` (the repaired scorer) |
| unmodified: collector | PASS `ebb17776…` (the repaired collector) |
| question count | PASS 135 |
| historical results present | PASS, unmodified |
| deployed SHA | PASS `5c436961…` |
| authenticated preflight | PASS HTTP 200 |
| evidence sink readable | PASS, 271 existing records |
| liveness twice in succession | PASS, 2 healthy of 2 |
| **bearer lifetime >= 60 minutes** | **FAIL — 44.6 minutes remaining of 82.5 total** |

    TOKEN_REMAINING_MINUTES_AT_Q1 = 44.6   (required >= 60)
    STOPPED BEFORE Q1. Zero live questions. Zero model calls.

## This is the gate doing its job, not a new problem

The bearer **authenticates** — HTTP 200, cleanly. Under V2's preflight this would
have passed every check and the run would have started. The bank takes thirty to
forty-five minutes and this token has forty-four minutes left, so it would very
likely have died part way through for the second time.

That is exactly the case the brief named: a cached token that authenticates
successfully but is already close to expiry. The token was issued 38 minutes ago
with an 82.5-minute lifetime; it is not fresh.

## What is needed before the run

Two operator actions, and the order matters:

    1. MI_AGENT_PLAN_SERVE:  off -> canary
       (this restarts App Service; the liveness-twice gate absorbs that)

    2. THEN mint a genuinely fresh MI_BEARER, last, so it is at its
       longest when Q1 is asked

Doing the mint last matters: the restart takes about a minute and the
certification then wants the token's full life ahead of it, not behind it.

Nothing else is outstanding. Everything the preflight can check without the
operator already passes.

---

# Second stop, 18:27:48Z — the secret in CI is the SAME token

    TOKEN_REMAINING_MINUTES_AT_Q1 = 30.9   (required >= 60)
    STOPPED BEFORE Q1. Zero live questions. Zero model calls.

Two independent readings, fourteen minutes apart:

| read at | remaining | total lifetime | secret length |
| --- | --- | --- | --- |
| 18:13:42Z | 44.6 min | 82.5 min | 1831 chars |
| 18:27:48Z | 30.9 min | 82.5 min | 1831 chars |

    elapsed wall clock      14.1 min
    decrease in remaining   13.7 min
    change in total life     0.0 min
    change in secret length    0 chars

The remaining time fell by exactly the wall clock that passed, the total lifetime
is identical and so is the secret's length. **This is one token ageing, not two
tokens.** Its claims imply it was issued around 17:36Z and expires around 18:58Z.

A genuinely new token would show `remaining` close to `total`, and would almost
certainly differ in length.

## The most likely cause, and it is not carelessness

**Entra and MSAL return a CACHED access token.** `az account get-access-token`
and the MSAL silent-acquire path both hand back the token already in the cache
until it is within a few minutes of expiry. Asking for a token is therefore not
the same as being issued one: the command succeeds, prints a token, and it is the
same JWT as last time. That is precisely the trap this gate exists to catch, and
it is why "MI_BEARER has been freshly minted" and "MI_BEARER is a fresh token"
can both be said in good faith and still be different things.

Two other possibilities, neither excluded by this evidence: the secret was
updated somewhere other than this repository's Actions secrets, or the paste did
not save.

## To get a genuinely new token

Force the cache to be bypassed rather than asking again:

    az account get-access-token --scope <the API scope> --force-refresh
      (or `az account clear` / `az logout` then sign in again)

then confirm before pasting: decode the JWT at jwt.ms or with

    python -c "import base64,json,sys,time; p=sys.argv[1].split('.')[1]; \
      p+='='*(-len(p)%4); c=json.loads(base64.urlsafe_b64decode(p)); \
      print('minutes left', round((c['exp']-time.time())/60,1))" "<token>"

and check it reads close to the full lifetime, not thirty minutes.

The current token expires around 18:58Z. After that the cache will have no valid
token to return and the next acquire will genuinely mint one.

## Everything else still passes

Deployed SHA `5c436961…`, auth HTTP 200, sink readable (271 records), liveness
healthy twice in succession, question count 135, all four hash pins including
both repaired instruments, and the historical replay at 52 / 56 / 25 / 2.

The bank remains unspent and V2's evidence remains untouched.
