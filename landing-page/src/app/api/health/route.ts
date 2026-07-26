import { NextResponse } from "next/server";

import { demoPack } from "@/lib/demo-pack";
import { APPLICATION_ENV } from "@/lib/env";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * GET /api/health — liveness/readiness for Azure health checks.
 *
 * Reports only what an operator needs to see that the app booted with a valid
 * demo pack. No secrets, no connection strings, no provider names, no paths.
 */
export function GET(): NextResponse {
  const healthy =
    demoPack.intents.length > 0 &&
    demoPack.reports.length > 0 &&
    Boolean(demoPack.portfolio.asOfDate);

  return NextResponse.json(
    {
      status: healthy ? "ok" : "degraded",
      environment: APPLICATION_ENV,
      demoPack: {
        version: demoPack.packVersion,
        intents: demoPack.intents.length,
        reports: demoPack.reports.length,
        asOfDate: demoPack.portfolio.asOfDate,
      },
    },
    { status: healthy ? 200 : 503 },
  );
}
