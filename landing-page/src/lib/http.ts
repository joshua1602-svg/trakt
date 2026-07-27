import "server-only";

import { NextResponse } from "next/server";

import { LIMITS } from "@/lib/env";

/**
 * Shared request handling for the public API routes: body-size enforcement,
 * input sanitisation, generic errors and the same-origin check that stands in
 * for CORS (these routes are same-origin by design and send no CORS headers, so
 * a cross-origin browser call is blocked by the absence of a permit).
 */

export const GENERIC_ERROR = "Something went wrong. Please try again.";

export function jsonError(status: number, message: string = GENERIC_ERROR): NextResponse {
  return NextResponse.json({ error: message }, { status });
}

/** Strip control characters and collapse whitespace. Never HTML-escape here —
 *  values are rendered as text by React, which escapes on output. */
export function sanitiseText(value: unknown, maxLength: number): string {
  if (typeof value !== "string") return "";
  return value
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, maxLength);
}

export type BodyResult<T> =
  | { ok: true; value: T }
  | { ok: false; response: NextResponse };

/**
 * Read a JSON body with a hard byte ceiling applied before parsing, so an
 * oversized payload is rejected without ever being buffered into an object.
 */
export async function readJsonBody<T = unknown>(
  request: Request,
  maxBytes: number = LIMITS.maxBodyBytes,
): Promise<BodyResult<T>> {
  const declared = request.headers.get("content-length");
  if (declared && Number.parseInt(declared, 10) > maxBytes) {
    return { ok: false, response: jsonError(413, "Request too large.") };
  }

  let text: string;
  try {
    text = await request.text();
  } catch {
    return { ok: false, response: jsonError(400, "Malformed request.") };
  }

  if (Buffer.byteLength(text, "utf8") > maxBytes) {
    return { ok: false, response: jsonError(413, "Request too large.") };
  }

  try {
    return { ok: true, value: JSON.parse(text || "{}") as T };
  } catch {
    return { ok: false, response: jsonError(400, "Malformed request.") };
  }
}

/**
 * Reject cross-site form posts. Same-origin fetches from the page always carry
 * an Origin (or at least a same-host Referer); anything else is refused.
 */
export function isSameOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  const host = request.headers.get("host");
  if (!host) return false;

  if (origin) {
    try {
      return new URL(origin).host === host;
    } catch {
      return false;
    }
  }

  const referer = request.headers.get("referer");
  if (referer) {
    try {
      return new URL(referer).host === host;
    } catch {
      return false;
    }
  }
  // No Origin and no Referer: a non-browser client. Allowed for GETs, and the
  // POST routes are rate-limited and side-effect-light by design.
  return true;
}

export function rateLimitedResponse(retryAfterSeconds: number): NextResponse {
  const response = NextResponse.json(
    { error: "Too many requests. Please wait a moment and try again." },
    { status: 429 },
  );
  response.headers.set("Retry-After", String(retryAfterSeconds));
  return response;
}

/**
 * Structured request logging. Deliberately records the resolved intent id and
 * outcome, never the visitor's question text.
 */
export function logRequest(event: {
  route: string;
  outcome: string;
  intentId?: string | null;
  sessionId?: string;
  durationMs?: number;
}): void {
  const line = {
    at: new Date().toISOString(),
    ...event,
  };
  console.info(JSON.stringify(line));
}
