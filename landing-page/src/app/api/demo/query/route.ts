import { NextResponse } from "next/server";

import {
  ALLOWED_PORTFOLIO_IDS,
  demoPack,
  followUpsFor,
  getIntent,
  getUnsupported,
} from "@/lib/demo-pack";
import { LIMITS, RATE_LIMITS } from "@/lib/env";
import {
  GENERIC_ERROR,
  jsonError,
  logRequest,
  rateLimitedResponse,
  readJsonBody,
  sanitiseText,
} from "@/lib/http";
import { resolveId, resolveQuestion } from "@/lib/intents";
import { checkRateLimit, clientIp } from "@/lib/rate-limit";
import {
  SESSION_COOKIE,
  encodeSession,
  loadSession,
  sessionCookieOptions,
} from "@/lib/session";
import type { DemoAnswerResponse } from "@/types/demo";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

interface QueryBody {
  question?: unknown;
  questionId?: unknown;
  portfolioId?: unknown;
}

const PORTFOLIO_SCOPE = `${demoPack.client.name} · ${demoPack.portfolio.id}`;

function usage(used: number): DemoAnswerResponse["usage"] {
  return {
    questionsUsed: used,
    questionsRemaining: Math.max(0, LIMITS.questionsPerSession - used),
  };
}

/**
 * POST /api/demo/query — the constrained public demo endpoint.
 *
 * The visitor's text never reaches the Trakt MI engine. It is normalised,
 * length-capped and matched against a fixed allow-list of intents
 * (`lib/intents.ts`); a match selects one pre-computed governed answer from the
 * demo pack, and anything else returns a controlled "not supported" response.
 * Because the engine is not part of this deployment at all, there is no code
 * path from this route to portfolio computation, client data or Azure storage.
 */
export async function POST(request: Request): Promise<NextResponse> {
  const started = Date.now();
  const ip = clientIp(request.headers);

  const limit = checkRateLimit(`query:${ip}`, RATE_LIMITS.demoPerIp);
  if (!limit.allowed) {
    logRequest({ route: "demo/query", outcome: "rate_limited" });
    return rateLimitedResponse(limit.retryAfterSeconds);
  }

  const body = await readJsonBody<QueryBody>(request);
  if (!body.ok) return body.response;

  // A portfolio may be named, but only the one this demo publishes.
  const portfolioId = sanitiseText(body.value.portfolioId ?? "", 80);
  if (portfolioId && !ALLOWED_PORTFOLIO_IDS.includes(portfolioId)) {
    logRequest({ route: "demo/query", outcome: "invalid_portfolio" });
    return jsonError(400, "Unknown portfolio.");
  }

  const questionId = sanitiseText(body.value.questionId ?? "", 64);
  const rawQuestion = body.value.question;

  // Oversized input is rejected outright rather than silently truncated, so a
  // visitor is never shown an answer to a question they did not ask.
  if (typeof rawQuestion === "string" && rawQuestion.length > LIMITS.maxQuestionLength) {
    logRequest({ route: "demo/query", outcome: "question_too_long" });
    return jsonError(
      400,
      `Please keep questions under ${LIMITS.maxQuestionLength} characters.`,
    );
  }

  const question = sanitiseText(rawQuestion ?? "", LIMITS.maxQuestionLength);
  if (!question && !questionId) {
    return jsonError(400, "Please enter a question.");
  }

  const session = loadSession(
    request.headers.get("cookie")?.match(/(?:^|;\s*)trakt_demo_session=([^;]*)/)?.[1],
  );

  if (session.q >= LIMITS.questionsPerSession) {
    const payload: DemoAnswerResponse = {
      status: "limit_reached",
      intentId: null,
      question: questionId || question,
      answer:
        `You have reached the ${LIMITS.questionsPerSession}-question limit for this ` +
        "demonstration session. Thank you for exploring Trakt — a walkthrough with " +
        "our team covers your own portfolio, reporting requirements and channels.",
      synthetic: true,
      usage: usage(session.q),
    };
    logRequest({ route: "demo/query", outcome: "session_limit", sessionId: session.sid });
    return NextResponse.json(payload, { status: 200 });
  }

  const resolution = questionId ? resolveId(questionId) : resolveQuestion(question);

  // A chip that names a report is handled by the report endpoint; asking for
  // one in free text is answered with a pointer rather than a refusal.
  if (resolution.kind === "report") {
    const report = demoPack.reports.find((r) => r.id === resolution.id);
    if (report) {
      session.q += 1;
      const payload: DemoAnswerResponse = {
        status: "answered",
        intentId: report.id,
        question: question || report.label,
        answer: report.answer,
        interpreted: `Report action · ${report.documentTitle}`,
        artifacts: [],
        asOfDate: demoPack.portfolio.asOfDate,
        asOfDisplay: demoPack.portfolio.asOfDisplay,
        portfolioScope: PORTFOLIO_SCOPE,
        followUps: followUpsFor(report.followUps),
        synthetic: true,
        usage: usage(session.q),
      };
      const response = NextResponse.json(payload);
      response.cookies.set(SESSION_COOKIE, encodeSession(session), sessionCookieOptions);
      logRequest({
        route: "demo/query",
        outcome: "report_pointer",
        intentId: report.id,
        sessionId: session.sid,
        durationMs: Date.now() - started,
      });
      return response;
    }
  }

  let payload: DemoAnswerResponse;
  let outcome: string;

  if (resolution.kind === "intent") {
    const intent = getIntent(resolution.id);
    if (!intent) return jsonError(500, GENERIC_ERROR);
    session.q += 1;
    outcome = "answered";
    payload = {
      status: "answered",
      intentId: intent.id,
      question: question || intent.label,
      answer: intent.answer,
      interpreted: intent.interpreted,
      artifacts: intent.artifacts,
      asOfDate: demoPack.portfolio.asOfDate,
      asOfDisplay: demoPack.portfolio.asOfDisplay,
      portfolioScope: PORTFOLIO_SCOPE,
      followUps: followUpsFor(intent.followUps),
      synthetic: true,
      usage: usage(session.q),
    };
  } else if (resolution.kind === "unsupported") {
    const topic = getUnsupported(resolution.id);
    if (!topic) return jsonError(500, GENERIC_ERROR);
    session.q += 1;
    outcome = "unsupported_known";
    payload = {
      status: "unsupported",
      intentId: topic.id,
      question: question || topic.label,
      answer: topic.reason,
      productionNote: topic.productionNote,
      asOfDate: demoPack.portfolio.asOfDate,
      asOfDisplay: demoPack.portfolio.asOfDisplay,
      portfolioScope: PORTFOLIO_SCOPE,
      followUps: followUpsFor(["funded_balance", "region_exposure", "management_summary"]),
      synthetic: true,
      usage: usage(session.q),
    };
  } else {
    session.q += 1;
    outcome = "unmatched";
    payload = {
      status: "unsupported",
      intentId: null,
      question,
      answer:
        "This demonstration answers a fixed set of governed questions about the " +
        "synthetic portfolio, and will not guess at anything outside it. Try one " +
        "of the suggested questions below.",
      productionNote:
        "In a client environment, Trakt's MI Agent answers open portfolio " +
        "questions across the full governed dataset, and still refuses to " +
        "produce a figure it cannot derive deterministically.",
      asOfDate: demoPack.portfolio.asOfDate,
      asOfDisplay: demoPack.portfolio.asOfDisplay,
      portfolioScope: PORTFOLIO_SCOPE,
      followUps: followUpsFor(["funded_balance", "region_exposure", "ltv_band"]),
      synthetic: true,
      usage: usage(session.q),
    };
  }

  const response = NextResponse.json(payload);
  response.cookies.set(SESSION_COOKIE, encodeSession(session), sessionCookieOptions);
  logRequest({
    route: "demo/query",
    outcome,
    intentId: payload.intentId,
    sessionId: session.sid,
    durationMs: Date.now() - started,
  });
  return response;
}
