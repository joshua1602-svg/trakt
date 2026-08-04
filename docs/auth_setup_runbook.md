# Turning on the login for the MI Agent — step-by-step guide

This switches on "sign in to see the reports" for the MI Agent. After this, only
people you allow can open it. **This is setup only — nobody is emailed.** You put
just yourself on the list now; the client's people are added later (a 2-minute
step) once they've signed.

The app code is already done. Everything below is clicking through your Microsoft
(Azure) account.

**Your details, already filled in for you:**

| Thing | Value |
|---|---|
| Your login (operator) | your `@digifinsolutions.co.uk` Microsoft account |
| Client users (added later, not now) | `@equityreleaseeurope.com` and their NED at `@becquerelventures.com` |
| Reports service (API) | the `trakt-mi-api` app |
| Website plan | Paid (Standard) — confirmed |
| Who gets access now | **only you** (`@digifinsolutions.co.uk`) |
| Emails sent to client now | **none** |

Time needed: ~20–30 minutes, one time.

---

## Part 1 — the one-time setup (just you)

Do these in order in the Azure portal (portal.azure.com), signed in with your
`@digifinsolutions.co.uk` account.

### 1. Upgrade the website to the paid plan
- Open your **Static Web App** (the website that serves the MI Agent).
- Find **Hosting plan** (under Settings) and switch it from **Free** to
  **Standard**. This is the ~$9/month upgrade. It shows the exact price before you
  confirm.

### 2. Create the "sign-in registration"
This tells Microsoft "these people are allowed to sign in to this app."
- Go to **Microsoft Entra ID** → **App registrations** → **New registration**.
- Name it `trakt-mi-agent`.
- Under "Supported account types", choose the option that includes your
  organisation **and guests** (accounts in this directory + invited guests).
- Under **Redirect URI**, pick **Web** and paste:
  `https://<your-website-address>/.auth/login/aad/callback`
  (replace `<your-website-address>` with your Static Web App's URL).
- Click **Register**.
- On the next screen, **copy the "Application (client) ID"** — you'll paste it in
  step 3.
- Go to **Certificates & secrets** → **New client secret** → create one →
  **copy its "Value" immediately** (you can't see it again).

### 3. Give the website those two values
- Back in your **Static Web App** → **Configuration** (Application settings) →
  add two settings:
  - `AAD_CLIENT_ID` = the Application (client) ID from step 2
  - `AAD_CLIENT_SECRET` = the secret Value from step 2
- Save.

*(You don't need to touch the tenant ID — I've already set your organisation's
domain, `digifinsolutions.co.uk`, in the app's config file.)*

### 4. Connect the reports service to the website
- In your **Static Web App** → **APIs** → **Link** → choose **`trakt-mi-api`**.
- This lets the website hand the logged-in person's identity to the reports
  service automatically.

### 5. Turn the lock on, on the reports side
- Open the **`trakt-mi-api`** app → **Configuration** (Application settings) → add:
  - `MI_AGENT_AUTH_ENABLED` = `true`
  - `MI_AGENT_CLIENT_ID` = a short label for this client, e.g. `ERE`
- Save and restart the app.

`MI_AGENT_CORS_ORIGINS` and `MI_AGENT_ALLOWED_ORIGIN` are **no longer needed**:
the browser now calls the reports service same-origin through the website, so no
cross-origin call is made. Leaving them set is harmless.

### 5b. Close the side door (important)
Linking the backend does **not** stop the reports service being reachable
directly at `https://trakt-mi-api.azurewebsites.net`. The service trusts the
identity the platform passes it — which is correct behind the website, but means
anyone who can reach the service directly could send that identity header
themselves and be treated as an operator.

Do **one** of these on the `trakt-mi-api` app:

- **Authentication** → add the Microsoft identity provider and set unauthenticated
  requests to **HTTP 401** (not "allow"). Easy Auth then strips any identity
  header a caller sent and injects only one it verified itself; or
- **Networking** → **Access restrictions** → deny public access, allowing only
  the Static Web App.

Check it worked — from any machine, this must **not** return data:
```
curl -s -o /dev/null -w "%{http_code}\n" https://trakt-mi-api.azurewebsites.net/me
```
`401` or `403` is correct. `200` means the service is still answering strangers.

### 6. Point the website at the connected service — **already done in code**
- The website now sends report questions to `/api` (same origin), so the sign-in
  identity is passed through to the reports service. This is set in
  `.github/workflows/azure-static-web-apps-nice-smoke-067ac7603.yml`
  (`VITE_AGENT_API_URL: /api`) and takes effect on the next deploy of the site.
- **It only works once step 4 is done.** Until the backend is linked, `/api/*`
  has nothing behind it and every report call returns 404. Do step 4 first, then
  redeploy the site.

> **Note (fixed).** The reports service now answers on **both** `/api/mi/…` and
> `/mi/…`, so this step no longer depends on getting the two sides to agree. It
> previously did: the website asked for `/api/mi/query`, the service only
> answered `/mi/query`, and every question came back as "not found" (HTTP 404).
> To check which form a deployment answers on:
> `curl -s https://trakt-mi-api.azurewebsites.net/health | jq .routing`
>
> If you keep the absolute address instead of `/api`, also set
> `MI_AGENT_ALLOWED_ORIGIN` on the reports service to your website's address —
> otherwise the browser blocks the call (and the sign-in identity is not passed
> through, so you would be asked to sign in again).

### 7. Put only yourself on the list — **in the website, not in Entra**
- Go to your **Static Web App** → **Role management** → **Invite**.
- Invite **your own** `@digifinsolutions.co.uk` account and give it the role
  **`operator`** (type it exactly). Open the generated invitation link once,
  signed in as that account, to accept it.
- That's it. No one else is added. Nothing is sent to the client.

> **Important — this changed.** An earlier version of this guide said to assign
> the role under **Enterprise applications → Users and groups**. That assigns an
> *Entra app role*, which appears in the Entra token but **not** in the identity
> Static Web Apps forwards to the reports service. The reports service checks the
> role in the forwarded identity, so assigning it in Entra alone means you sign in
> successfully and are then refused with "No MI access role assigned" (HTTP 403).
> Roles must be granted in the Static Web App's own **Role management**.

---

## Part 2 — done. Test it
- Open your website address in a private/incognito browser window.
- It should now ask you to **sign in with Microsoft** before showing anything.
- Sign in with your `@digifinsolutions.co.uk` account → you see the MI Agent.
- Try it signed out (or a different account) → you're kept out. 

If sign-in gives an "issuer"/"tenant" error, tell me — it's a one-line fix (I'll
swap your domain for your tenant's ID in the config).

---

## Later — when the client signs (NOT now)
A 2-minute step, done by you when you're ready:
1. **Static Web App** → **Role management** → **Invite** each of their people —
   the `@equityreleaseeurope.com` staff **and** their NED at
   `@becquerelventures.com` — and give each the role **`client`**. (Not
   Enterprise applications — see the note in step 7.)
2. They receive a Microsoft invitation and can sign in. Client users see the
   reports; only you (operator) have full access.

Until you do this, the client is not contacted and cannot get in.

---

## If you ever need to turn it off
Set `MI_AGENT_AUTH_ENABLED` = `false` on the `trakt-mi-api` app. (Don't leave the
website reachable by the client with the lock off.)

---

## For reference — what the code already handles (no action needed)
- `mi_agent_api/auth.py` — reads the signed-in person's identity that the website
  passes to the reports service; requires an allowed role (client or operator);
  refuses everyone else. Switched by `MI_AGENT_AUTH_ENABLED`.
- `mi_agent_api/app.py` — applies that check to every report request; tightened
  the cross-site rules; stopped the health check from revealing a server file
  path; hides internal error details from users; adds a `/me` "who am I" check.
- `frontend/mi-agent-ui/staticwebapp.config.json` — the website's own rules:
  sign-in with Microsoft (Entra), the whole site requires a signed-in user, and
  `/api/*` (the reports service) additionally requires the `client` or
  `operator` role. Unused sign-in providers are closed off. Your domain is set in
  `openIdIssuer`.

  > **Correction.** An earlier version of this guide said this file already
  > contained those rules. It did not — the file had only the security headers
  > and the single-page-app fallback, and no auth block had ever been committed.
  > The rules above are now actually in it.

- Tested in `mi_agent_api/tests/test_auth.py` (the identity parser) and
  `mi_agent_api/tests/test_linked_backend_auth.py` (the end-to-end contract: no
  anonymous access to any report route, the forwarded identity is honoured under
  `/api`, and the website config and the site build agree on the topology).
