import { NextResponse } from "next/server";

import { MIN_FORM_FILL_MS, RATE_LIMITS } from "@/lib/env";
import {
  isSameOrigin,
  jsonError,
  logRequest,
  rateLimitedResponse,
  readJsonBody,
} from "@/lib/http";
import { botSignals, validateLead, type LeadInput } from "@/lib/lead-validation";
import { deliverLead, newLeadId, type Lead } from "@/lib/leads";
import { checkRateLimit, clientIp } from "@/lib/rate-limit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Leads carry a message field, so the body ceiling is larger than the demo's. */
const MAX_LEAD_BYTES = 8_192;

/**
 * POST /api/leads — the lead capture endpoint behind the final CTA.
 *
 * Controls: same-origin only, per-IP hourly rate limit, hard body ceiling,
 * server-side validation and sanitisation, honeypot field, minimum
 * form-completion time, and exactly one configured delivery provider. A
 * delivery failure is reported as a failure — submissions are never discarded.
 */
export async function POST(request: Request): Promise<NextResponse> {
  if (!isSameOrigin(request)) {
    logRequest({ route: "leads", outcome: "cross_origin" });
    return jsonError(403, "Request rejected.");
  }

  const ip = clientIp(request.headers);
  const limit = checkRateLimit(`lead:${ip}`, RATE_LIMITS.leadPerIp);
  if (!limit.allowed) {
    logRequest({ route: "leads", outcome: "rate_limited" });
    return rateLimitedResponse(limit.retryAfterSeconds);
  }

  const body = await readJsonBody<LeadInput>(request, MAX_LEAD_BYTES);
  if (!body.ok) return body.response;

  // Bots get the same success shape a human sees, so failure teaches nothing.
  const bot = botSignals(body.value, MIN_FORM_FILL_MS);
  if (bot.bot) {
    logRequest({ route: "leads", outcome: `rejected_${bot.reason}` });
    return NextResponse.json({ status: "received" }, { status: 202 });
  }

  const { ok, errors, value } = validateLead(body.value);
  if (!ok) {
    logRequest({ route: "leads", outcome: "invalid" });
    return NextResponse.json({ status: "invalid", errors }, { status: 400 });
  }

  const lead: Lead = {
    id: newLeadId(),
    receivedAt: new Date().toISOString(),
    ...value,
    consent: true,
    source: "landing-page",
  };

  try {
    const outcome = await deliverLead(lead);
    logRequest({ route: "leads", outcome: `delivered_${outcome.provider}` });
    return NextResponse.json({ status: "received" }, { status: 201 });
  } catch (error) {
    // The reason is logged for operators; the visitor sees a generic message
    // and is given a way to reach us that does not depend on this endpoint.
    console.error(
      `[leads] delivery failed: ${error instanceof Error ? error.message : "unknown"}`,
    );
    logRequest({ route: "leads", outcome: "delivery_failed" });
    return NextResponse.json(
      {
        status: "error",
        message:
          "We could not record your request just now. Please email us directly " +
          "and we will pick it up.",
      },
      { status: 502 },
    );
  }
}
