import "server-only";

import { RATE_LIMIT_STORE_URL } from "@/lib/env";

/**
 * Fixed-window rate limiting.
 *
 * The default store is in-process, which is correct and sufficient for a single
 * App Service instance — the documented deployment. `RATE_LIMIT_STORE_URL` is
 * read here so that a multi-instance deployment fails loudly at review time
 * rather than silently multiplying every limit by the instance count; wiring a
 * shared store is a documented follow-up, not a silent gap.
 */

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetAt: number;
  retryAfterSeconds: number;
}

interface Bucket {
  count: number;
  resetAt: number;
}

const buckets = new Map<string, Bucket>();

/** Keeps the map from growing without bound under sustained traffic. */
const MAX_TRACKED_KEYS = 20_000;

function sweep(now: number): void {
  if (buckets.size < MAX_TRACKED_KEYS) return;
  for (const [key, bucket] of buckets) {
    if (bucket.resetAt <= now) buckets.delete(key);
  }
  if (buckets.size >= MAX_TRACKED_KEYS) buckets.clear();
}

export function checkRateLimit(
  key: string,
  { limit, windowMs }: { limit: number; windowMs: number },
  now: number = Date.now(),
): RateLimitResult {
  sweep(now);
  const existing = buckets.get(key);

  if (!existing || existing.resetAt <= now) {
    const resetAt = now + windowMs;
    buckets.set(key, { count: 1, resetAt });
    return { allowed: true, remaining: limit - 1, resetAt, retryAfterSeconds: 0 };
  }

  if (existing.count >= limit) {
    return {
      allowed: false,
      remaining: 0,
      resetAt: existing.resetAt,
      retryAfterSeconds: Math.max(1, Math.ceil((existing.resetAt - now) / 1000)),
    };
  }

  existing.count += 1;
  return {
    allowed: true,
    remaining: limit - existing.count,
    resetAt: existing.resetAt,
    retryAfterSeconds: 0,
  };
}

/** Test seam. */
export function resetRateLimits(): void {
  buckets.clear();
}

export const usesSharedStore = Boolean(RATE_LIMIT_STORE_URL);

/**
 * Best-effort client address.
 *
 * Azure App Service and Container Apps both terminate at a front end that sets
 * `x-forwarded-for`; the left-most entry is the client. When no proxy header is
 * present every caller collapses to one bucket, which fails safe (stricter),
 * not open.
 */
export function clientIp(headers: Headers): string {
  const forwarded = headers.get("x-forwarded-for");
  if (forwarded) {
    const first = forwarded.split(",")[0]?.trim();
    // Azure appends :port to forwarded addresses.
    if (first) return first.replace(/:\d+$/, "");
  }
  return headers.get("x-client-ip")?.trim() || "unknown";
}
