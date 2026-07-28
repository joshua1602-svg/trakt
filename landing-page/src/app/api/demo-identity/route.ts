import { NextResponse } from "next/server";

import { buildDemoIdentity } from "@/lib/demo-identity";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * GET /api/demo-identity — which demo pack is this deployment serving?
 *
 * A deployment control, not a product feature. `/api/health` answers "alive"
 * and `/api/ready` answers "correctly configured"; both would return 200 for a
 * perfectly healthy App Service still running last week's bundle. This answers
 * the remaining question — *which* dataset is behind the figures on the page —
 * and it is the only surface that answers it as a published fact rather than as
 * an accident of framework serialisation.
 *
 * Read-only, unauthenticated and free of side effects. Every field is a direct
 * projection of the committed `data/demo-pack.json`, is synthetic, and is
 * already visible to any visitor — see `lib/demo-identity.ts`, which also
 * states the rule for what may be added.
 *
 * Fails closed. If the pack cannot yield a complete identity the response is
 * 503 with no partial fields, because a partial document would satisfy a
 * field-by-field comparison of whatever happened to survive.
 *
 * `no-store`, because a cached copy of this document would confirm the identity
 * of a bundle that is no longer running — which is precisely the failure the
 * endpoint exists to detect.
 */
export function GET(): NextResponse {
  const identity = buildDemoIdentity();

  if (!identity) {
    return NextResponse.json(
      { status: "unavailable" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }

  return NextResponse.json(identity, {
    status: 200,
    headers: { "Cache-Control": "no-store" },
  });
}
