# Governed Trakt identity mapping — implementation report

Replaces Static Web Apps invitation-role authorisation with an identity→access
mapping Trakt owns. Static Web Apps remains the authentication boundary and
nothing else.

---

## 1. What changed, and what deliberately did not

**Unchanged.** Static Web Apps still performs the Entra sign-in and still
forwards a verified `X-MS-CLIENT-PRINCIPAL`. The API still refuses a request
carrying no principal with **401**. Both principal shapes (SWA `userRoles`, App
Service Easy Auth `claims`) are parsed exactly as before. Copilot's bearer-token
authentication is untouched — asserted by a test that greps `copilot_auth` for
any reference to the directory.

**Changed.** Only the authorisation half. A verified identity is now looked up in
`config/access.yaml`, and that lookup decides tenant, platform role, allowed
portfolio contexts and whether the account is live.

**Not a second identity system.** There is still exactly one identity — the one
the platform verified — and one trusted context type, `ExecutionContext`. The
directory supplies the facts that context is built from, which previously came
from a deployment-wide environment variable and an unreliable role claim. No
`rolesSource` function: it would have put the decision back on the SWA round
trip that was the problem.

---

## 2. Files

| File | Change |
|---|---|
| `config/access.yaml` | **New.** The directory. Seeded with the two ERE operators. |
| `trakt_core/access.py` | **New.** `AccessGrant`, loader, resolution, context authorisation. Pure — no web framework, so every channel can use it. |
| `trakt_core/errors.py` | `ACCESS_NOT_PROVISIONED`, `ACCESS_DISABLED` — both 403, category `authorisation`, non-retryable. |
| `mi_agent_api/auth.py` | `auth_guard` resolves the grant, enforces tenant and context, stashes both on `request.state`. |
| `mi_agent_api/identity.py` | `context_from_principal` takes tenant and scopes from the grant. |
| `mi_agent_api/app.py` | `_execution_context` passes the grant; `/me` reports it; `/health` reports the directory's posture; two body-borne selectors gated. |
| `frontend/.../staticwebapp.config.json` | `/api/*` → `["authenticated"]`; invitation route removed. |
| `frontend/.../domain/accessErrors.ts` | **New.** Refusal → title/detail/retryability. |
| `frontend/.../api/HttpAgentClient.ts` | GET path reads the governed envelope. |
| `frontend/.../components/states/States.tsx` | Renders a refusal as an explanation, not a fault. |
| `docs/auth_setup_runbook.md` | Step 7 rewritten; the two superseded mechanisms recorded. |

---

## 3. The mapping

```yaml
principals:
  - identities: [admin@ratecheck.uk]
    tenant_id: ERE
    role: operator
```

* **`identities`** — one or more verified identifiers for the same person.
  Matched case-insensitively against both the email/UPN and the Entra object id,
  so a deployment can pin either. Seeded with emails only: no environment-specific
  object ids in source, per the constraint.
* **`tenant_id`** — populates `ExecutionContext.tenant_id`. Never from the request.
* **`role`** — `client` | `operator`. Same read scopes today; the split exists so
  an operator-only capability can be gated later.
* **`portfolio_contexts`** — optional allow-list. **Omitted means unrestricted
  within the tenant**, which is the normal case and the default. Making the empty
  list mean "all" rather than "none" is what stops a new context from silently
  locking out every existing user.
* **`enabled`** — optional, default true. `false` revokes while keeping the record.

Nothing here is caller-supplied. The lookup key is the platform-verified subject.

---

## 4. Fail-closed behaviour

| Condition | Status | Code |
|---|---|---|
| No principal | 401 | — |
| Authenticated, not in the directory | 403 | `ACCESS_NOT_PROVISIONED` |
| In the directory, `enabled: false` | 403 | `ACCESS_DISABLED` |
| Grant names another tenant | 403 | `TENANT_MISMATCH` |
| Context outside the grant's allow-list | 403 | `PERMISSION_DENIED` |

A missing, unparseable or empty `access.yaml` provisions **nobody**. That is the
opposite of `insight_config` and `tenancy`, which fall back to permissive
defaults — deliberately, because those govern presentation and this governs
access. An authorisation source that degrades to "allow" is not one.

### Why cross-tenant is refused rather than scoped

The dataset layer resolves data from deployment configuration
(`dependencies.default_tenant_id`), **not** from the context. So a grant naming
another tenant would be served *this* deployment's book. It has to be refused at
the door.

**This means `MI_AGENT_CLIENT_ID` must equal the `tenant_id` in the directory**
(case-insensitively). If it does not, every user is refused — correctly, but
totally. See §8.

---

## 5. Where authorisation is enforced

**Tenant, role and provisioning** — in `auth_guard`, the global dependency on
every route. One place, cannot be forgotten by a new route.

**Portfolio contexts** — also in `auth_guard`, over the `portfolioContext` and
`lens` query parameters.

Not in `_resolve_portfolio_context`, though that is the single resolution choke
point, because it is documented never to raise and returns `None` on failure —
which every route treats as *unscoped*. Refusing there would silently **widen** a
denied request to the full book instead of blocking it.

Two selectors travel in a request body, where the guard cannot see them (reading
the body in a dependency consumes the stream). Both are authorised in their
handlers:

* `/mi/query` — `sourcePortfolioLens`
* `/mi/decks/generate` — `portfolioContext`

The second matters most: generating an investor pack over a context the account
cannot view would export exactly what the restriction exists to bound. It was
found by the coverage test, not by reading the routes — that test walks endpoint
signatures *and* request-model fields, so a third selector fails a test rather
than quietly becoming an unchecked way in.

---

## 6. Legacy SWA roles

Parsed and echoed exactly as before, surfaced on `/me` as `platformRoles` for
migration visibility. They decide nothing.

Where the compatibility requirement met the fail-closed requirement, **fail-closed
won**: a principal claiming `operator` does not admit an identity the directory
omits. Supporting that would have reintroduced the forgeable-header problem —
anyone who can reach the App Service directly can present any roles they like.
`test_a_forged_identity_gains_nothing` pins it.

`/me` reports the *directory's* role, not the platform's. Reporting the platform's
would tell the UI a provisioned operator was not one.

---

## 7. User experience

The 403 body is the governed envelope: `{"error": "Your account is signed in but
has not been provisioned for Trakt access.", "errorCode": "ACCESS_NOT_PROVISIONED",
"retryable": false}`.

`HttpAgentClient`'s GET path now reads that envelope — previously only `/mi/query`
did, which is why a provisioning gap surfaced as *"MI Agent API returned 403 for
/mi/snapshots"*: technically true, useless to the reader.

`ErrorState` renders a refusal as an explanation with **no Retry button**, since
retrying cannot provision an account. The message is pinned against the backend
constant by `TestFrontendContract` so the two cannot drift.

---

## 8. Deployment steps

**Order matters.** Step 1 before step 3, or every user is locked out between them.

1. **Set `MI_AGENT_CLIENT_ID=ERE`** on the `trakt-mi-api` App Service — it must
   match `tenant_id` in `config/access.yaml`. Verify the current value first:

   ```
   curl -s https://trakt-mi-api.azurewebsites.net/health | jq .governance
   ```

   If it already reads `ERE`, nothing to do. If it reads something else, either
   change it or change `tenant_id` in the directory to match — but they must agree.

2. **Merge and deploy the API.** `config/access.yaml` ships with it.

3. **Deploy the site.** `staticwebapp.config.json` drops `/api/*` to
   `["authenticated"]`.

4. **Verify** — the same `/health` call now reports:

   ```json
   "accessDirectory": {"configured": true, "provisioned_identities": 2}
   ```

   `configured: false` means the file did not deploy and **everyone is locked
   out**. This is on the open health route on purpose: if the directory fails to
   deploy, the one signal that diagnoses it must not sit behind the directory.
   Counts only — never the roster.

5. **Sign in** as `joshuahall@digifinsolutions.co.uk`. `/me` should report
   `"role": "operator"`, `"tenantId": "ERE"`.

**No portal action is required, and no invitation is issued.** Existing SWA
invitations and role records can be left alone; they are inert. To add someone
later, add an entry to `config/access.yaml` and deploy — it takes effect on the
next request without a restart (the loader keys its cache on the file's mtime).

**Rollback**: revert the two commits. `MI_AGENT_AUTH_ENABLED=false` remains the
emergency switch, but it disables authentication entirely and must not be left on
with the site reachable.

---

## 9. Verification

| Suite | Result |
|---|---|
| `mi_agent_api/tests/test_governed_access.py` | 37 passed (new) |
| `mi_agent_api/tests/test_auth.py` | 19 passed |
| `mi_agent_api/tests/test_linked_backend_auth.py` | 24 passed |
| `mi_agent_api/tests` | failures identical to `origin/main` |
| Frontend `vitest` | 460 passed, 62 files |
| `tsc -b` + `vite build` | clean |

Required scenarios, all covered in `test_governed_access.py`:

* mapped operator, mapped client — admitted, roles distinguished
* unmapped — 403 `ACCESS_NOT_PROVISIONED`, non-retryable, exact message
* disabled — 403 `ACCESS_DISABLED`, distinguishable from never-provisioned
* cross-tenant — 403 `TENANT_MISMATCH`
* forged identity — three forged role sets, plus forged tenant claims, all refused
* **invitations not required** — `test_a_user_with_no_swa_role_at_all_is_admitted_when_provisioned`
  sends `["anonymous", "authenticated"]`, verbatim what SWA forwards for a user
  who never accepted an invitation, and asserts 200

Also covered: object-id matching, case-insensitivity, missing/unparseable/partial
config, mtime-based reload, the shipped directory's contents, the absence of
secrets in it, health disclosure without a roster, and the `ExecutionContext`
tenant coming from the grant.

---

## 10. Known limits

1. **`MI_AGENT_CLIENT_ID` and the directory must agree.** A mismatch locks
   everyone out. Diagnosable in one `/health` call; step 1 above exists for this.
2. **The directory is deployment-scoped configuration.** Adding a user is a
   commit and a deploy. That is the intent — access changes should be reviewable —
   but it is slower than a portal click.
3. **Cross-tenant is refused, not served.** Multi-tenant serving from one
   deployment would need the dataset layer to resolve from `context.tenant_id`
   rather than deployment config. Out of scope here and deliberately not started.
4. **The forgeable-header exposure is unchanged by this work.** If the App
   Service answers requests directly, a caller can still present any principal —
   they now need a *provisioned* identity rather than any identity, which is
   narrower but not closed. Closing it is step 5b of the auth runbook (restrict
   inbound traffic to the Static Web App), and it remains outstanding.

5. **Teams notification recipients are still authorised separately, and I did
   not unify them.** Worth stating plainly because it is the one place the
   "single mapping" claim needs qualifying.

   Requirement 4 is met for the four channels named: React, the MI API, PPTX/deck
   generation and Teams **deep links** all resolve identity through
   `ExecutionContext`, and that context's tenant now comes from this directory.

   But `trakt_notifications.recipients` (pre-existing, added with the Teams bot)
   holds a per-recipient `portfolio_contexts` list with its own operator
   approval step. It governs a different direction — whether Trakt may **push**
   MI into a Teams chat — and it is deliberately *stricter*: there, an empty list
   means *nothing* is authorised, whereas an empty list in the access directory
   means *unrestricted*, because it is a restriction layered on top of being
   listed at all.

   Those two defaults are opposite, which is correct for their directions
   (proactively sending data is a higher bar than answering someone who asked)
   but is exactly the sort of thing that drifts. Collapsing them would change
   push semantics and require migrating existing recipient records, so it is out
   of scope for a narrow change. **Recommendation:** treat the directory as the
   source of *who exists and for which tenant*, and keep the recipient store as
   the *delivery opt-in*, with a follow-up that makes a recipient record
   unusable once its identity is absent or disabled here. `test_governed_access.py`
   asserts the bot path has not been silently re-plumbed in the meantime.
