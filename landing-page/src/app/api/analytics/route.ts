import { NextResponse } from "next/server";

import {
  MAX_ANALYTICS_BODY_BYTES,
  isAnalyticsEvent,
  sanitiseProperties,
} from "@/lib/analytics-events";
import { ANALYTICS_PROVIDER, ANALYTICS_RATE_LIMIT_PER_MINUTE } from "@/lib/env";
import { isSameOrigin, readJsonBody } from "@/lib/http";
import { checkRateLimit, clientIp } from "@/lib/rate-limit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * POST /api/analytics — the first-party analytics collector.
 *
 * Design decisions worth stating:
 *
 *   • **Always 204.** Analytics must never shape the visitor's experience, so
 *     a rejected event, a rate-limited caller and a malformed body all get the
 *     same empty success. The browser has nothing useful to do with a failure,
 *     and an error would only tell an abuser what got through.
 *   • **Allow-list, not a sink.** An unrecognised event name is dropped. This
 *     endpoint cannot be used as general-purpose internet-writable logging.
 *   • **No free text can reach it.** The property allow-list carries only
 *     identifiers, each stripped to `[A-Za-z0-9_.-]` and capped — so a question
 *     string, an answer or an email address has no field to travel in.
 */
export async function POST(request: Request): Promise<NextResponse> {
  // Beacons from the page carry an Origin; anything else is not our page.
  if (!isSameOrigin(request)) return noContent();

  const ip = clientIp(request.headers);
  const limit = await checkRateLimit(
    `analytics:${ip}`,
    { limit: ANALYTICS_RATE_LIMIT_PER_MINUTE, windowMs: 60_000 },
  );
  if (!limit.allowed) return noContent();

  const body = await readJsonBody<{ event?: unknown; properties?: unknown }>(
    request,
    MAX_ANALYTICS_BODY_BYTES,
  );
  if (!body.ok) return noContent();

  const { event, properties } = body.value;
  if (!isAnalyticsEvent(event)) return noContent();

  const clean = sanitiseProperties(properties);

  // The record. No IP, no user agent, no identifier that outlives the request —
  // this is a counter, not a profile.
  const record = {
    at: new Date().toISOString(),
    kind: "analytics",
    event,
    ...clean,
  };

  if (ANALYTICS_PROVIDER === "appinsights") {
    // App Insights' Node SDK auto-collects console output when the agent is
    // attached, so a structured line reaches it without adding a dependency
    // or a second network hop from the request path.
    console.info(JSON.stringify(record));
  } else if (ANALYTICS_PROVIDER === "firstparty") {
    console.info(JSON.stringify(record));
  }
  // provider "none": accepted and discarded, so the page behaves identically.

  return noContent();
}

function noContent(): NextResponse {
  return new NextResponse(null, { status: 204 });
}
