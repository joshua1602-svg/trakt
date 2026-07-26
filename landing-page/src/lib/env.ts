/**
 * Server-side configuration. Nothing in this module may be imported from a
 * client component — the only values the browser receives are the
 * `NEXT_PUBLIC_*` ones re-exported through `publicConfig`.
 */

function str(name: string, fallback: string): string {
  const value = process.env[name];
  return value && value.trim() ? value.trim() : fallback;
}

function bool(name: string, fallback: boolean): boolean {
  const raw = process.env[name]?.trim().toLowerCase();
  if (raw === undefined || raw === "") return fallback;
  return raw === "true" || raw === "1" || raw === "yes";
}

function int(name: string, fallback: number, min: number, max: number): number {
  const raw = process.env[name];
  const parsed = raw ? Number.parseInt(raw, 10) : Number.NaN;
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

export const APPLICATION_ENV = str("APPLICATION_ENV", process.env.NODE_ENV ?? "development");
export const IS_PRODUCTION = APPLICATION_ENV === "production";

/** Canonical public origin. Never hard-coded to a specific Trakt domain. */
export const SITE_URL = str("NEXT_PUBLIC_SITE_URL", "http://localhost:3000").replace(/\/$/, "");

export const DEMO_NAME = str("NEXT_PUBLIC_DEMO_NAME", "Synthetic Demo Lender");

/** Reserved for a future deployment that proxies a private demo service. */
export const DEMO_API_BASE_URL = process.env.DEMO_API_BASE_URL?.trim() || null;

/** HMAC key for demo session cookies. Ephemeral in dev; required in production. */
export const DEMO_SESSION_SECRET = process.env.DEMO_SESSION_SECRET?.trim() || null;

/* ----------------------------- Demo limits ------------------------------- */
export const LIMITS = {
  questionsPerSession: int("DEMO_MAX_QUESTIONS_PER_SESSION", 12, 1, 100),
  reportsPerSession: int("DEMO_MAX_REPORTS_PER_SESSION", 3, 1, 20),
  sessionTtlSeconds: int("DEMO_SESSION_TTL_SECONDS", 60 * 60 * 2, 60, 60 * 60 * 24),
  maxQuestionLength: int("DEMO_MAX_QUESTION_LENGTH", 240, 40, 1000),
  maxBodyBytes: int("DEMO_MAX_BODY_BYTES", 4096, 512, 65536),
} as const;

/* ---------------------------- Rate limiting ------------------------------ */
export const RATE_LIMITS = {
  /** Per-IP demo queries. */
  demoPerIp: { limit: int("RATE_LIMIT_DEMO_PER_MINUTE", 30, 1, 600), windowMs: 60_000 },
  /** Per-IP report previews. */
  reportPerIp: { limit: int("RATE_LIMIT_REPORT_PER_MINUTE", 10, 1, 200), windowMs: 60_000 },
  /** Per-IP lead submissions — deliberately tight. */
  leadPerIp: { limit: int("RATE_LIMIT_LEAD_PER_HOUR", 5, 1, 100), windowMs: 60 * 60_000 },
} as const;

/**
 * Optional external store for rate-limit counters. Unset means the in-process
 * store, which is correct for a single instance and documented as such.
 */
export const RATE_LIMIT_STORE_URL = process.env.RATE_LIMIT_STORE_URL?.trim() || null;

/**
 * Deliberate acceptance of in-process rate limiting in production.
 *
 * Only correct on a single instance. Left false so that scaling out without a
 * shared store fails configuration validation rather than silently multiplying
 * every limit by the replica count.
 */
export const ALLOW_IN_MEMORY_RATE_LIMIT = bool("ALLOW_IN_MEMORY_RATE_LIMIT", false);

/** Bounded wait for the shared store before falling back to the local policy. */
export const RATE_LIMIT_STORE_TIMEOUT_MS = int("RATE_LIMIT_STORE_TIMEOUT_MS", 250, 25, 5000);

/* ---------------------------- Lead delivery ------------------------------ */
export type LeadProvider = "file" | "email" | "webhook" | "console";

export const LEAD_DELIVERY_PROVIDER = str("LEAD_DELIVERY_PROVIDER", "file") as LeadProvider;
export const LEAD_NOTIFICATION_EMAIL = process.env.LEAD_NOTIFICATION_EMAIL?.trim() || null;
export const LEAD_FROM_EMAIL = process.env.LEAD_FROM_EMAIL?.trim() || null;
export const EMAIL_API_KEY = process.env.EMAIL_API_KEY?.trim() || null;
export const EMAIL_API_URL = str("EMAIL_API_URL", "https://api.resend.com/emails");
/** Informational only — never logged, never returned to a caller. */
export const EMAIL_PROVIDER = str("EMAIL_PROVIDER", "resend");
export const LEAD_DELIVERY_TIMEOUT_MS = int("LEAD_DELIVERY_TIMEOUT_MS", 8000, 1000, 30000);
export const CRM_WEBHOOK_URL = process.env.CRM_WEBHOOK_URL?.trim() || null;
export const CRM_WEBHOOK_SECRET = process.env.CRM_WEBHOOK_SECRET?.trim() || null;
export const LEAD_STORE_DIR = str("LEAD_STORE_DIR", ".leads");

/* ------------------------------ Analytics -------------------------------- */
export type AnalyticsProvider = "none" | "appinsights" | "firstparty";
export const ANALYTICS_PROVIDER = str(
  "NEXT_PUBLIC_ANALYTICS_PROVIDER",
  "none",
) as AnalyticsProvider;
export const APPINSIGHTS_CONNECTION_STRING =
  process.env.APPLICATIONINSIGHTS_CONNECTION_STRING?.trim() || null;

/** Per-IP cap on analytics events. Generous — this must never shape the page. */
export const ANALYTICS_RATE_LIMIT_PER_MINUTE = int("ANALYTICS_RATE_LIMIT_PER_MINUTE", 120, 1, 6000);

/** Bot protection: a form completed faster than this is rejected. */
export const MIN_FORM_FILL_MS = int("LEAD_MIN_FILL_MS", 2500, 0, 60_000);

/** Everything the browser is allowed to know. */
export const publicConfig = {
  siteUrl: SITE_URL,
  demoName: DEMO_NAME,
  analyticsProvider: ANALYTICS_PROVIDER,
  applicationEnv: APPLICATION_ENV,
} as const;
