import { NextResponse } from "next/server";

import { componentStates } from "@/lib/config";
import { demoPack } from "@/lib/demo-pack";
import { APPLICATION_ENV } from "@/lib/env";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * GET /api/health — liveness.
 *
 * Answers "is this process alive and did it load a demo pack". Deliberately
 * cheap and deliberately tolerant: it does not fail because lead delivery is
 * misconfigured, because that is a readiness concern (`/api/ready`) and
 * restarting the process would not fix it.
 *
 * Reports only what an operator needs. No secrets, no connection strings, no
 * provider names, no addresses, no paths.
 */
export function GET(): NextResponse {
  const alive =
    demoPack.intents.length > 0 &&
    demoPack.reports.length > 0 &&
    Boolean(demoPack.portfolio.asOfDate);

  const states = componentStates();

  return NextResponse.json(
    {
      status: alive ? "ok" : "degraded",
      environment: APPLICATION_ENV,
      demoPack: states.demoPack,
      leadDelivery: states.leadDelivery,
      rateLimit: states.rateLimit,
      analytics: states.analytics,
    },
    { status: alive ? 200 : 503 },
  );
}
