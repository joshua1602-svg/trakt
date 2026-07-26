import { NextResponse } from "next/server";

import { buildMeta } from "@/lib/demo-pack";
import { RATE_LIMITS } from "@/lib/env";
import { rateLimitedResponse } from "@/lib/http";
import { checkRateLimit, clientIp } from "@/lib/rate-limit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * GET /api/demo/meta — everything the demo UI needs to render before a question
 * is asked: the synthetic scope, the suggested questions, the report actions,
 * the session limits and two examples of what the demo will decline.
 */
export function GET(request: Request): NextResponse {
  const ip = clientIp(request.headers);
  const limit = checkRateLimit(`meta:${ip}`, RATE_LIMITS.demoPerIp);
  if (!limit.allowed) return rateLimitedResponse(limit.retryAfterSeconds);

  return NextResponse.json(buildMeta());
}
