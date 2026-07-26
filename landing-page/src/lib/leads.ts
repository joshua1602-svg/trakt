import "server-only";

import { createHmac, randomUUID } from "node:crypto";
import { appendFile, mkdir } from "node:fs/promises";
import path from "node:path";

import {
  CRM_WEBHOOK_SECRET,
  CRM_WEBHOOK_URL,
  EMAIL_API_KEY,
  EMAIL_API_URL,
  LEAD_DELIVERY_PROVIDER,
  LEAD_FROM_EMAIL,
  LEAD_NOTIFICATION_EMAIL,
  LEAD_STORE_DIR,
  type LeadProvider,
} from "@/lib/env";

/**
 * Lead delivery.
 *
 * Exactly one provider is active, chosen by `LEAD_DELIVERY_PROVIDER`. Every
 * adapter either delivers or throws — a submission is never silently dropped,
 * and the route surfaces a real failure to the visitor rather than a false
 * success. The `file` adapter is the development-safe default and is documented
 * as such in the README.
 */

export interface Lead {
  id: string;
  receivedAt: string;
  name: string;
  email: string;
  company: string;
  role: string;
  message: string;
  consent: true;
  source: string;
}

export interface DeliveryOutcome {
  provider: LeadProvider;
  delivered: true;
  /** True when the lead was stored locally rather than sent onward. */
  local: boolean;
}

const DELIVERY_TIMEOUT_MS = 8_000;

async function postJson(url: string, body: unknown, headers: Record<string, string>) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), DELIVERY_TIMEOUT_MS);
  try {
    return await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", ...headers },
      body: JSON.stringify(body),
      signal: controller.signal,
      cache: "no-store",
    });
  } finally {
    clearTimeout(timer);
  }
}

function textBody(lead: Lead): string {
  return [
    `Name:    ${lead.name}`,
    `Email:   ${lead.email}`,
    `Company: ${lead.company}`,
    `Role:    ${lead.role}`,
    `Source:  ${lead.source}`,
    `Time:    ${lead.receivedAt}`,
    "",
    lead.message || "(no message)",
  ].join("\n");
}

/* --------------------------- provider adapters ---------------------------- */

async function deliverByEmail(lead: Lead): Promise<DeliveryOutcome> {
  if (!EMAIL_API_KEY || !LEAD_NOTIFICATION_EMAIL || !LEAD_FROM_EMAIL) {
    throw new Error(
      "LEAD_DELIVERY_PROVIDER=email requires EMAIL_API_KEY, LEAD_FROM_EMAIL and " +
        "LEAD_NOTIFICATION_EMAIL.",
    );
  }

  const response = await postJson(
    EMAIL_API_URL,
    {
      from: LEAD_FROM_EMAIL,
      to: [LEAD_NOTIFICATION_EMAIL],
      reply_to: lead.email,
      subject: `Trakt demo enquiry — ${lead.company}`,
      text: textBody(lead),
    },
    { authorization: `Bearer ${EMAIL_API_KEY}` },
  );

  if (!response.ok) {
    throw new Error(`email provider responded ${response.status}`);
  }
  return { provider: "email", delivered: true, local: false };
}

async function deliverByWebhook(lead: Lead): Promise<DeliveryOutcome> {
  if (!CRM_WEBHOOK_URL) {
    throw new Error("LEAD_DELIVERY_PROVIDER=webhook requires CRM_WEBHOOK_URL.");
  }

  const payload = { type: "trakt.landing.lead", lead };
  const headers: Record<string, string> = {};
  if (CRM_WEBHOOK_SECRET) {
    // Lets the receiver verify the payload came from this deployment.
    headers["x-trakt-signature"] = createHmac("sha256", CRM_WEBHOOK_SECRET)
      .update(JSON.stringify(payload))
      .digest("hex");
  }

  const response = await postJson(CRM_WEBHOOK_URL, payload, headers);
  if (!response.ok) {
    throw new Error(`CRM webhook responded ${response.status}`);
  }
  return { provider: "webhook", delivered: true, local: false };
}

async function deliverToFile(lead: Lead): Promise<DeliveryOutcome> {
  // turbopackIgnore keeps the bundler from tracing the whole project because of
  // this runtime path join; the directory is configuration, not an import.
  const dir = path.resolve(/*turbopackIgnore: true*/ process.cwd(), LEAD_STORE_DIR);
  await mkdir(dir, { recursive: true });
  await appendFile(path.join(dir, "leads.jsonl"), `${JSON.stringify(lead)}\n`, "utf8");
  return { provider: "file", delivered: true, local: true };
}

async function deliverToConsole(lead: Lead): Promise<DeliveryOutcome> {
  console.info(`[lead] ${JSON.stringify(lead)}`);
  return { provider: "console", delivered: true, local: true };
}

export async function deliverLead(lead: Lead): Promise<DeliveryOutcome> {
  switch (LEAD_DELIVERY_PROVIDER) {
    case "email":
      return deliverByEmail(lead);
    case "webhook":
      return deliverByWebhook(lead);
    case "console":
      return deliverToConsole(lead);
    case "file":
      return deliverToFile(lead);
    default:
      throw new Error(`Unknown LEAD_DELIVERY_PROVIDER: ${LEAD_DELIVERY_PROVIDER}`);
  }
}

export function newLeadId(): string {
  return randomUUID();
}
