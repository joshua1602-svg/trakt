import "server-only";

import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";

import { DEMO_SESSION_SECRET, IS_PRODUCTION, LIMITS } from "@/lib/env";

/**
 * Demo session state, carried in a signed HttpOnly cookie.
 *
 * The cookie holds counters only — no question text, no visitor identity.
 * Because the payload is signed rather than encrypted, tampering is detectable
 * (the counters cannot be reset by editing the cookie) while nothing sensitive
 * is stored client-side in the first place. Session state deliberately lives in
 * the cookie so the demo needs no server-side session store to scale.
 */

export interface DemoSession {
  /** Opaque session id, for correlating analytics events only. */
  sid: string;
  /** Issued-at, epoch seconds. */
  iat: number;
  /** Questions consumed. */
  q: number;
  /** Report previews consumed. */
  r: number;
}

export const SESSION_COOKIE = "trakt_demo_session";

let ephemeralSecret: string | null = null;

function secret(): string {
  if (DEMO_SESSION_SECRET) return DEMO_SESSION_SECRET;
  if (IS_PRODUCTION) {
    // Fail closed rather than silently issuing forgeable sessions.
    throw new Error(
      "DEMO_SESSION_SECRET must be set when APPLICATION_ENV=production.",
    );
  }
  ephemeralSecret ??= randomBytes(32).toString("hex");
  return ephemeralSecret;
}

function sign(payload: string): string {
  return createHmac("sha256", secret()).update(payload).digest("base64url");
}

function b64(value: string): string {
  return Buffer.from(value, "utf8").toString("base64url");
}

function unb64(value: string): string {
  return Buffer.from(value, "base64url").toString("utf8");
}

export function newSession(now: number = Date.now()): DemoSession {
  return { sid: randomBytes(9).toString("base64url"), iat: Math.floor(now / 1000), q: 0, r: 0 };
}

export function encodeSession(session: DemoSession): string {
  const payload = b64(JSON.stringify(session));
  return `${payload}.${sign(payload)}`;
}

export function decodeSession(
  raw: string | undefined | null,
  now: number = Date.now(),
): DemoSession | null {
  if (!raw) return null;
  const dot = raw.lastIndexOf(".");
  if (dot <= 0) return null;

  const payload = raw.slice(0, dot);
  const signature = raw.slice(dot + 1);

  const expected = Buffer.from(sign(payload));
  const provided = Buffer.from(signature);
  if (expected.length !== provided.length || !timingSafeEqual(expected, provided)) {
    return null;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(unb64(payload));
  } catch {
    return null;
  }

  if (!parsed || typeof parsed !== "object") return null;
  const candidate = parsed as Partial<DemoSession>;
  if (
    typeof candidate.sid !== "string" ||
    typeof candidate.iat !== "number" ||
    typeof candidate.q !== "number" ||
    typeof candidate.r !== "number"
  ) {
    return null;
  }

  const ageSeconds = Math.floor(now / 1000) - candidate.iat;
  if (ageSeconds < 0 || ageSeconds > LIMITS.sessionTtlSeconds) return null;

  return {
    sid: candidate.sid.slice(0, 32),
    iat: candidate.iat,
    q: Math.max(0, Math.min(candidate.q, LIMITS.questionsPerSession)),
    r: Math.max(0, Math.min(candidate.r, LIMITS.reportsPerSession)),
  };
}

/** Read the session from a request, minting a fresh one when absent or stale. */
export function loadSession(cookieValue: string | undefined, now: number = Date.now()): DemoSession {
  return decodeSession(cookieValue, now) ?? newSession(now);
}

export const sessionCookieOptions = {
  httpOnly: true,
  sameSite: "lax" as const,
  secure: IS_PRODUCTION,
  path: "/",
  maxAge: LIMITS.sessionTtlSeconds,
};

export function questionsRemaining(session: DemoSession): number {
  return Math.max(0, LIMITS.questionsPerSession - session.q);
}

export function reportsRemaining(session: DemoSession): number {
  return Math.max(0, LIMITS.reportsPerSession - session.r);
}
