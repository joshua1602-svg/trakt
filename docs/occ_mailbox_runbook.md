# OCC onboarding mailbox — provisioning and verification

The OCC Agent sends a client's onboarding pack from one Microsoft 365 mailbox
and reads their reply back from the same one. This is what has to exist in the
tenant before either works, and how to prove it does.

None of it can be done from the repository: creating a mailbox and consenting a
Graph permission are tenant administration. Everything below is run by a
Microsoft 365 administrator.

---

## 1. Why a dedicated mailbox

Use `onboarding@traktinfra.io`. Do not point this at `admin@traktinfra.io`.

The control that stops an app-only Graph identity reading the whole tenant's
mail is the Exchange **Application Access Policy**, and it scopes an
application to **a mailbox**. There is no folder-level equivalent. So the
moment `Mail.Read` is consented and the policy names `admin@`, the application
can read every message in `admin@` — including everything unrelated to any
client.

Three consequences, in the order they will matter to you:

1. **Data minimisation.** The application holds more than its purpose needs,
   which is the thing a client's security questionnaire asks about directly.
2. **The audit claim gets weaker.** "Every file in this sandbox arrived as
   onboarding evidence" is provable when the mailbox only ever receives
   onboarding mail. It is an assertion about operator behaviour when the
   mailbox receives everything else too.
3. **Blast radius on a leaked secret.** The client secret in
   `TRAKT_MAIL_CLIENT_SECRET` is one credential. What it reaches should be one
   mailbox that contains only what this integration is for.

If a dedicated mailbox genuinely cannot be provisioned in time, the fallback is
an Outlook rule filing client replies into a dedicated folder and
`TRAKT_MAIL_INBOUND_FOLDER` pointing at it. Be clear about what that buys: it
narrows what the **reader looks at**, not what the **application may read**.
It is a process control. Record it as one, and move to a dedicated mailbox
after the first onboarding.

---

## 2. Create the mailbox

Microsoft 365 admin centre, or PowerShell. Either a licensed user mailbox or a
shared mailbox works; a shared mailbox needs no licence and cannot be signed
into interactively, which is the better default here.

```powershell
Connect-ExchangeOnline -UserPrincipalName <admin-upn>

New-Mailbox -Shared -Name "Trakt Onboarding" `
            -DisplayName "Trakt Onboarding" `
            -PrimarySmtpAddress onboarding@traktinfra.io

# Keep replies threaded and readable, and stop Exchange from converting
# messages in ways that lose the reply headers correlation depends on.
Set-Mailbox onboarding@traktinfra.io -MessageCopyForSentAsEnabled $true
```

Send a message to it from an external address and confirm it arrives before
going any further. A mailbox that does not receive mail will otherwise look
identical to a permissions problem later.

---

## 3. Point the app registration at it

The `Trakt OCC Mail` app registration already exists with `Mail.Send` and
`Mail.Read` as **application** permissions, admin-consented. Nothing about the
registration changes — only the mailbox the policy scopes it to.

```powershell
# Replace an existing policy rather than adding a second one: policies are
# additive, so leaving the admin@ policy in place would keep granting it.
Get-ApplicationAccessPolicy |
  Where-Object { $_.AppId -eq "<app-registration-client-id>" } |
  Remove-ApplicationAccessPolicy

New-DistributionGroup -Name "Trakt OCC Mail Scope" `
                      -Type Security `
                      -PrimarySmtpAddress trakt-occ-mail-scope@traktinfra.io `
                      -Members onboarding@traktinfra.io

New-ApplicationAccessPolicy `
  -AppId "<app-registration-client-id>" `
  -PolicyScopeGroupId trakt-occ-mail-scope@traktinfra.io `
  -AccessRight RestrictAccess `
  -Description "Trakt OCC Agent — onboarding mailbox only"
```

A mail-enabled security group as the scope, rather than the mailbox directly,
so a second mailbox can be added later without recreating the policy.

---

## 4. Verify before switching anything on

This is the gate. Do not set `TRAKT_MAIL_INBOUND_ENABLED` until all three pass.

```powershell
# 1. The app CAN reach the onboarding mailbox.
Test-ApplicationAccessPolicy -Identity onboarding@traktinfra.io `
                             -AppId "<app-registration-client-id>"
#    AccessCheckResult must be: Granted

# 2. The app CANNOT reach anything else. Test a mailbox that holds real
#    business mail — this is the assertion the whole design rests on.
Test-ApplicationAccessPolicy -Identity admin@traktinfra.io `
                             -AppId "<app-registration-client-id>"
#    AccessCheckResult must be: Denied
```

Policy changes take time to propagate — allow up to 30 minutes and re-test
rather than assuming a `Denied` on step 1 is a misconfiguration.

Third check, from anywhere with `curl`, using a token for the app registration:

```bash
# Reads the folder the OCC Agent will read. 200 with a "value" array is a pass;
# 403 means the policy has not propagated or does not cover this mailbox.
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://graph.microsoft.com/v1.0/users/onboarding@traktinfra.io/mailFolders/inbox/messages?\$top=1"
```

---

## 5. Application settings

Set on the `trakt-ops-api` App Service. The full annotated list is
`deploy/trakt-ops-api/app_settings.example.json`; the ones that change here:

| Setting | Value |
| --- | --- |
| `TRAKT_MAIL_MAILBOX` | `onboarding@traktinfra.io` |
| `TRAKT_MAIL_INBOUND_ENABLED` | `true` |
| `TRAKT_MAIL_INBOUND_FOLDER` | `inbox` (or the dedicated folder, if sharing) |
| `TRAKT_MAIL_SENDER_ALLOWLIST` | the client's domain, for the first onboarding |

The client secret stays a Key Vault reference. It is never a literal in app
settings and never in this repository.

`TRAKT_MAIL_INBOUND_ENABLED` is deliberately a **separate switch** from
`TRAKT_MAIL_OUTBOUND_ENABLED`. Sending a pack and reading a mailbox are
different capabilities with different blast radii, and either can be turned off
without the other — which is also the fastest way to stop the reader if
something is wrong, with no redeploy.

---

## 6. What the reader does, and what it deliberately does not

Once on, an operator working a case can ask what has arrived
(`GET /ops/agent/cases/{case_ref}/mail`) and take a reply in
(`POST /ops/agent/cases/{case_ref}/mail/ingest`).

**It is pull, not push.** Nothing polls. A person asks; the reader answers.

**A reply is matched to a case on evidence the mail system carries** — the
Graph conversation, the `In-Reply-To` header naming the message Trakt sent, or
the case reference in the subject line. A message whose only connection is that
the *sender is a known contact* is reported as unmatched, because one contact
can be on several onboardings and a shared address is one mailbox for a whole
firm. There is no override: if evidence cannot establish which case a message
belongs to, a person looks at the mailbox.

**Files are registered; words are not applied.** An attached file goes through
the same `register_synthetic_artefact` an operator's own upload uses — same
state check, same sandbox, same audit. The client's message text is recorded on
the case for the operator to read, and changes no answer. Applying it is an
instruction a human gives, through the existing path where the interpreter
shows its reading and asks for confirmation first.

**Nothing is deleted or moved.** An ingested message is marked read, and that
is all. Read state is treated as a hint rather than a memory — a person opening
the mailbox in Outlook marks mail read too — so the run's own record of what it
has ingested is what stops the same reply being taken in twice.

---

## 7. Rollback

1. Set `TRAKT_MAIL_INBOUND_ENABLED` to `false`. Reading stops immediately; no
   redeploy. Sending is unaffected.
2. To stop both, also set `TRAKT_MAIL_OUTBOUND_ENABLED` to `false`. The OCC
   falls back to its `RecordOnlyAdapter`, which states plainly that nothing was
   sent rather than pretending otherwise.
3. To revoke entirely, remove the Application Access Policy and the app
   registration's client secret. Anything already ingested stays on its case —
   it is evidence, and the audit chain references it.
