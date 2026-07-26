import { NextResponse } from "next/server";

import { componentStates, validateConfig } from "@/lib/config";
import { APPLICATION_ENV } from "@/lib/env";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * GET /api/ready — readiness.
 *
 * Distinct from `/api/health`, which answers "is this process alive". This
 * answers "is this process correctly configured to serve", and returns 503
 * when it is not, so a bad deployment is caught by a smoke test rather than by
 * a visitor.
 *
 * Every check is in-process and bounded: reading already-loaded configuration
 * and the already-parsed demo pack. It performs no network call — a readiness
 * probe that reaches a mail provider would turn that provider's outage into
 * this site's outage.
 *
 * The response carries coarse component states only. No file path, no email
 * address, no service URL, no provider name, no credential, no stack trace and
 * no internal engine name.
 */
export function GET(): NextResponse {
  const states = componentStates();
  const issues = validateConfig();
  const ready = issues.length === 0;

  return NextResponse.json(
    {
      status: ready ? "ready" : "not_ready",
      environment: APPLICATION_ENV,
      components: states,
      // Codes, never messages — the messages name variables, which is useful
      // in a log and needless on a public endpoint.
      issues: issues.map((issue) => issue.code),
    },
    { status: ready ? 200 : 503 },
  );
}
