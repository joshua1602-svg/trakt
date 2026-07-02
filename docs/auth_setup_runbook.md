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

### 4. Connect the reports service to the website ("link the backend")

**Exact location in the portal:**
- Open your **Static Web App** resource (this is the *website*, not the
  `trakt-mi-api` app — make sure you're on the right one).
- In the **left-hand menu**, under the **Settings** group, click **APIs**.
- You'll see an environment row (usually **Production**). Select it, then click
  **Link** at the top.
- **Backend resource type:** choose **App Service**. Pick your subscription, then
  select **`trakt-mi-api`**. Click **Link** / OK.

**If you can't see "APIs", or the Link button is greyed out, or `trakt-mi-api`
isn't in the list** — it's almost always one of these:
1. **The website is still on the Free plan.** Linking needs **Standard** (Step 1).
   Double-check the upgrade actually applied to *this* Static Web App. This is the
   #1 reason the option is missing.
2. **Region.** Linked backends are only supported when the App Service is in a
   supported region. If `trakt-mi-api` is in an unsupported region it won't appear
   in the list — use the command-line method below (it gives a clearer error), or
   tell me the region and I'll confirm.
3. **You're looking at the App Service, not the Static Web App.** The APIs/Link
   blade only exists on the Static Web App resource.

**Command-line alternative (more reliable than hunting in the portal).** Open
**Azure Cloud Shell** (the `>_` icon at the top of the portal, choose **Bash**) and
run — replacing the two `<...>` values:
```
az staticwebapp backends link \
  --name <your-static-web-app-name> \
  --resource-group <your-resource-group> \
  --backend-resource-id $(az webapp show -n trakt-mi-api -g <your-resource-group> --query id -o tsv) \
  --backend-region <region-of-trakt-mi-api>
```
If it errors, copy the message to me — it usually says exactly what's wrong (tier
or region).

Linking is what lets the website hand the logged-in person's identity to the
reports service automatically.

### 4b. Lock the reports app so it can't be bypassed  ⚠️ important
The reports service trusts the identity the website passes it. That means the
service must **not** be openly reachable on its own web address — otherwise
someone could skip the login by calling it directly. This needs one extra
lock-down (either restricting the app's network access to the website, or turning
on the platform's own sign-in check on the app as a second layer).

There are two ways to do this and the exact clicks differ by your setup, so
rather than guess: **once you've completed the linking in step 4, tell me and I'll
give you the precise steps for your case (and verify it's actually closed).** Do
**not** give the client the link until this is done.

### 5. Turn the lock on, on the reports side
- Open the **`trakt-mi-api`** app → **Configuration** (Application settings) → add:
  - `MI_AGENT_AUTH_ENABLED` = `true`
  - `MI_AGENT_CORS_ORIGINS` = your website's address (e.g. `https://<your-website-address>`)
  - `MI_AGENT_CLIENT_ID` = a short label for this client, e.g. `ERE`
- Save and restart the app.

### 6. Point the website at the connected service
- The website needs to send report questions to `/api`. This is one line in the
  deployment settings (`VITE_AGENT_API_URL` = `/api`). **Tell me when you reach
  this step and I'll make that change in the code for you** (it only takes effect
  after step 4, so timing matters — that's why I've held it).

### 7. Put only yourself on the list
- Go to **Enterprise applications** → open **`trakt-mi-agent`** → **Users and
  groups** → **Add user** → add **your own** `@digifinsolutions.co.uk` account and
  give it the **operator** role. (It's your own tenant, so no invitation is
  needed — you're already in the directory.)
- That's it. No one else is added. Nothing is sent to the client.

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
1. **Enterprise applications** → `trakt-mi-agent` → **Users and groups** → invite
   each of their people — the `@equityreleaseeurope.com` staff **and** their NED at
   `@becquerelventures.com` — and give each the **client** role.
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
  sign-in with Microsoft, only client/operator roles allowed, login/logout,
  security headers. Your domain is already set here.
- Tested in `mi_agent_api/tests/test_auth.py`; the rest of the app is unaffected.
