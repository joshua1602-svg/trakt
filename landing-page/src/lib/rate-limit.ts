import "server-only";

import {
  ALLOW_IN_MEMORY_RATE_LIMIT,
  IS_PRODUCTION,
  RATE_LIMIT_STORE_TIMEOUT_MS,
  RATE_LIMIT_STORE_URL,
} from "@/lib/env";

/**
 * Fixed-window rate limiting over a pluggable store.
 *
 * Two adapters:
 *   • in-memory — correct on exactly one instance; the documented default for
 *     a single App Service instance with `ALLOW_IN_MEMORY_RATE_LIMIT=true`.
 *   • shared    — an HTTP counter service named by `RATE_LIMIT_STORE_URL`,
 *     required whenever the app runs more than one replica.
 *
 * **Behaviour when the shared store is unavailable** (a decision, documented
 * here and in the README rather than left implicit):
 *
 *   • *Lead submission fails closed.* A lead is a state change with a cost to
 *     get wrong, and the endpoint is already the tightest limit on the site.
 *     If we cannot count, we refuse.
 *   • *Demo queries degrade to the stricter local limit.* Marketing pages are
 *     read-mostly and the demo is served from a static pack; making the page
 *     unusable because a counter service blinked would trade a small abuse
 *     risk for a total loss of function. Each instance then enforces the full
 *     per-IP limit locally, so the worst case is N× the intended ceiling for
 *     the duration of the outage — bounded, and strictly better than open.
 */

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetAt: number;
  retryAfterSeconds: number;
  /** True when the shared store was unreachable and the local policy applied. */
  degraded: boolean;
}

export interface RateLimitStore {
  readonly kind: "memory" | "shared";
  increment(
    key: string,
    windowSeconds: number,
  ): Promise<{ count: number; resetAt: number }>;
}

/* -------------------------------------------------------------------------- */
/* In-memory adapter                                                          */
/* -------------------------------------------------------------------------- */

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

export const memoryStore: RateLimitStore = {
  kind: "memory",
  async increment(key, windowSeconds) {
    const now = Date.now();
    sweep(now);
    const existing = buckets.get(key);
    if (!existing || existing.resetAt <= now) {
      const resetAt = now + windowSeconds * 1000;
      buckets.set(key, { count: 1, resetAt });
      return { count: 1, resetAt };
    }
    existing.count += 1;
    return { count: existing.count, resetAt: existing.resetAt };
  },
};

/* -------------------------------------------------------------------------- */
/* Shared adapter                                                             */
/* -------------------------------------------------------------------------- */

/**
 * A deliberately minimal HTTP contract, so this needs no client library and
 * adds no dependency:
 *
 *   POST {RATE_LIMIT_STORE_URL}
 *   { "key": "...", "windowSeconds": 60 }
 *   → 200 { "count": 3, "resetAt": 1730000000000 }
 *
 * Any Redis-backed counter, Azure Function or container can satisfy it. The
 * store is never trusted for anything but counting: it receives an opaque
 * hashed key and returns two numbers.
 */
export function createSharedStore(url: string): RateLimitStore {
  return {
    kind: "shared",
    async increment(key, windowSeconds) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), RATE_LIMIT_STORE_TIMEOUT_MS);
      try {
        const response = await fetch(url, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ key, windowSeconds }),
          signal: controller.signal,
          cache: "no-store",
        });
        if (!response.ok) throw new Error(`store responded ${response.status}`);
        const data = (await response.json()) as { count?: number; resetAt?: number };
        if (typeof data.count !== "number" || typeof data.resetAt !== "number") {
          throw new Error("store returned a malformed counter");
        }
        return { count: data.count, resetAt: data.resetAt };
      } finally {
        clearTimeout(timer);
      }
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Selection                                                                  */
/* -------------------------------------------------------------------------- */

let active: RateLimitStore | null = null;

export function rateLimitStore(): RateLimitStore {
  if (active) return active;
  active = RATE_LIMIT_STORE_URL ? createSharedStore(RATE_LIMIT_STORE_URL) : memoryStore;
  return active;
}

/** Startup validation. Throws when production would silently under-limit. */
export function assertRateLimitStore(production: boolean = IS_PRODUCTION): void {
  if (!production) return;
  if (RATE_LIMIT_STORE_URL) return;
  if (ALLOW_IN_MEMORY_RATE_LIMIT) return;
  throw new Error(
    "CONFIG_RATE_LIMIT_STORE_REQUIRED: no RATE_LIMIT_STORE_URL configured and " +
      "ALLOW_IN_MEMORY_RATE_LIMIT is not set. In-process counters are only " +
      "correct on a single instance.",
  );
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

export type FailureMode = "closed" | "local";

export async function checkRateLimit(
  key: string,
  { limit, windowMs }: { limit: number; windowMs: number },
  { failureMode = "local" }: { failureMode?: FailureMode } = {},
): Promise<RateLimitResult> {
  const windowSeconds = Math.max(1, Math.round(windowMs / 1000));
  const store = rateLimitStore();

  let count: number;
  let resetAt: number;
  let degraded = false;

  try {
    ({ count, resetAt } = await store.increment(key, windowSeconds));
  } catch (error) {
    // The reason is operator-facing only; the caller sees a limit decision.
    console.error(
      `[rate-limit] shared store unavailable: ${
        error instanceof Error ? error.message : "unknown"
      }`,
    );
    degraded = true;

    if (failureMode === "closed") {
      return {
        allowed: false,
        remaining: 0,
        resetAt: Date.now() + windowMs,
        retryAfterSeconds: Math.ceil(windowMs / 1000),
        degraded: true,
      };
    }
    // Degrade to the local counter — stricter than open, never unbounded.
    ({ count, resetAt } = await memoryStore.increment(key, windowSeconds));
  }

  if (count > limit) {
    return {
      allowed: false,
      remaining: 0,
      resetAt,
      retryAfterSeconds: Math.max(1, Math.ceil((resetAt - Date.now()) / 1000)),
      degraded,
    };
  }

  return {
    allowed: true,
    remaining: Math.max(0, limit - count),
    resetAt,
    retryAfterSeconds: 0,
    degraded,
  };
}

/** Test seam. */
export function resetRateLimits(): void {
  buckets.clear();
  active = null;
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
