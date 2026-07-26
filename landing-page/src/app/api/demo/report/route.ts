import { NextResponse } from "next/server";

import { ALLOWED_REPORT_IDS, followUpsFor, getReport } from "@/lib/demo-pack";
import { LIMITS, RATE_LIMITS } from "@/lib/env";
import {
  jsonError,
  logRequest,
  rateLimitedResponse,
  readJsonBody,
  sanitiseText,
} from "@/lib/http";
import { checkRateLimit, clientIp } from "@/lib/rate-limit";
import {
  SESSION_COOKIE,
  encodeSession,
  loadSession,
  sessionCookieOptions,
} from "@/lib/session";
import type { DemoReportResponse } from "@/types/demo";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

interface ReportBody {
  reportId?: unknown;
}

function usage(used: number): DemoReportResponse["usage"] {
  return {
    reportsUsed: used,
    reportsRemaining: Math.max(0, LIMITS.reportsPerSession - used),
  };
}

/**
 * POST /api/demo/report — synthetic report preview.
 *
 * The response is a set of rendered preview *pages* built at demo-pack
 * generation time from the deterministic engine. No document is produced, no
 * file is read at request time, no path is accepted from the caller, and no
 * storage URL (signed or otherwise) is ever returned. `reportId` is checked
 * against a fixed allow-list before anything else happens, so there is no
 * parameter through which a caller could reach the filesystem.
 */
export async function POST(request: Request): Promise<NextResponse> {
  const ip = clientIp(request.headers);
  const limit = checkRateLimit(`report:${ip}`, RATE_LIMITS.reportPerIp);
  if (!limit.allowed) {
    logRequest({ route: "demo/report", outcome: "rate_limited" });
    return rateLimitedResponse(limit.retryAfterSeconds);
  }

  const body = await readJsonBody<ReportBody>(request);
  if (!body.ok) return body.response;

  const reportId = sanitiseText(body.value.reportId ?? "", 64);
  if (!reportId || !ALLOWED_REPORT_IDS.includes(reportId)) {
    logRequest({ route: "demo/report", outcome: "invalid_report" });
    return jsonError(400, "Unknown report.");
  }

  const session = loadSession(
    request.headers.get("cookie")?.match(/(?:^|;\s*)trakt_demo_session=([^;]*)/)?.[1],
  );

  if (session.r >= LIMITS.reportsPerSession) {
    const payload: DemoReportResponse = {
      status: "limit_reached",
      reportId: null,
      answer:
        `You have previewed the ${LIMITS.reportsPerSession} reports available in ` +
        "this demonstration session. A walkthrough with our team covers the full " +
        "reporting suite against your own portfolio.",
      synthetic: true,
      usage: usage(session.r),
    };
    logRequest({ route: "demo/report", outcome: "session_limit", sessionId: session.sid });
    return NextResponse.json(payload);
  }

  const report = getReport(reportId);
  if (!report) return jsonError(400, "Unknown report.");

  session.r += 1;

  const payload: DemoReportResponse = {
    status: "ready",
    reportId: report.id,
    documentTitle: report.documentTitle,
    documentSubtitle: report.documentSubtitle,
    answer: report.answer,
    pages: report.pages,
    followUps: followUpsFor(report.followUps),
    synthetic: true,
    usage: usage(session.r),
  };

  const response = NextResponse.json(payload);
  response.cookies.set(SESSION_COOKIE, encodeSession(session), sessionCookieOptions);
  logRequest({
    route: "demo/report",
    outcome: "ready",
    intentId: report.id,
    sessionId: session.sid,
  });
  return response;
}
