import "server-only";

import packJson from "@data/demo-pack.json";

import {
  ALLOW_IN_MEMORY_RATE_LIMIT,
  ANALYTICS_PROVIDER,
  APPLICATION_ENV,
  APPINSIGHTS_CONNECTION_STRING,
  CRM_WEBHOOK_URL,
  DEMO_SESSION_SECRET,
  EMAIL_API_KEY,
  IS_PRODUCTION,
  LEAD_DELIVERY_PROVIDER,
  LEAD_FROM_EMAIL,
  LEAD_NOTIFICATION_EMAIL,
  RATE_LIMIT_STORE_URL,
  SITE_URL,
} from "@/lib/env";

/**
 * The one place production configuration is judged.
 *
 * Every unsafe production setting is an **error**, not a warning: a landing
 * page that boots with file-based lead capture, a forgeable session cookie or
 * a localhost canonical URL is worse than one that refuses to boot, because
 * the failure is silent and the loss (a lead, a leaked origin) is invisible.
 *
 * Messages never contain a secret, a credential, an address or a file path —
 * they name the variable and what is wrong with it.
 */

export type ConfigCode =
  | "CONFIG_SITE_URL_INVALID"
  | "CONFIG_DEMO_PACK_MISSING"
  | "CONFIG_DEMO_PACK_VERSION"
  | "CONFIG_DEMO_SOURCE_MISMATCH"
  | "CONFIG_SESSION_SECRET_WEAK"
  | "CONFIG_LEAD_PROVIDER_UNSAFE"
  | "CONFIG_LEAD_PROVIDER_INCOMPLETE"
  | "CONFIG_RATE_LIMIT_STORE_REQUIRED"
  | "CONFIG_ANALYTICS_INVALID"
  | "CONFIG_ENV_INVALID";

export interface ConfigIssue {
  code: ConfigCode;
  message: string;
}

/* -------------------------------------------------------------------------- */
/* Pinned demo-source identity                                                */
/* -------------------------------------------------------------------------- */

/**
 * What the runtime requires of `data/demo-pack.json`.
 *
 * This mirrors `DEMO_SOURCE` in `scripts/build_demo_pack.py`. The generator
 * refuses to *build* a pack from the wrong portfolio; this refuses to *serve*
 * one. Repointing the landing page at a different governed portfolio means
 * changing both, deliberately — which is the intent.
 */
export const EXPECTED_DEMO_SOURCE = {
  clientId: "synthetic_demo",
  portfolioId: "SYNTHETIC_ERE_Portfolio_012026",
  currency: "GBP",
  reportingDate: "2025-11-30",
  minBalance: 5_000_000,
  maxBalance: 6_000_000,
  minExposures: 30,
  packVersion: 1,
} as const;

interface PackSource {
  clientId?: string;
  portfolioId?: string;
  currency?: string;
  reportingDate?: string;
  totalBalance?: number;
  exposures?: number;
}

interface PackShape {
  packVersion?: number;
  source?: PackSource;
  intents?: unknown[];
  reports?: unknown[];
}

const pack = packJson as PackShape;

/** Demo-pack presence, schema version and portfolio identity. */
export function checkDemoPack(): ConfigIssue[] {
  const issues: ConfigIssue[] = [];

  if (!pack || !Array.isArray(pack.intents) || pack.intents.length === 0) {
    issues.push({
      code: "CONFIG_DEMO_PACK_MISSING",
      message: "demo-pack.json is absent or contains no intents.",
    });
    return issues;
  }

  if (pack.packVersion !== EXPECTED_DEMO_SOURCE.packVersion) {
    issues.push({
      code: "CONFIG_DEMO_PACK_VERSION",
      message:
        `demo-pack.json is version ${pack.packVersion}; this build expects ` +
        `${EXPECTED_DEMO_SOURCE.packVersion}. Re-run scripts/build_demo_pack.py.`,
    });
  }

  const source = pack.source;
  if (!source) {
    issues.push({
      code: "CONFIG_DEMO_SOURCE_MISMATCH",
      message:
        "demo-pack.json carries no source identity block. It predates the " +
        "pinned-source contract; regenerate it.",
    });
    return issues;
  }

  const mismatches: string[] = [];
  if (source.clientId !== EXPECTED_DEMO_SOURCE.clientId) {
    mismatches.push(`client ${source.clientId} != ${EXPECTED_DEMO_SOURCE.clientId}`);
  }
  if (source.portfolioId !== EXPECTED_DEMO_SOURCE.portfolioId) {
    mismatches.push(
      `portfolio ${source.portfolioId} != ${EXPECTED_DEMO_SOURCE.portfolioId}`,
    );
  }
  if (source.currency !== EXPECTED_DEMO_SOURCE.currency) {
    mismatches.push(`currency ${source.currency} != ${EXPECTED_DEMO_SOURCE.currency}`);
  }
  if (source.reportingDate !== EXPECTED_DEMO_SOURCE.reportingDate) {
    mismatches.push(
      `reporting date ${source.reportingDate} != ${EXPECTED_DEMO_SOURCE.reportingDate}`,
    );
  }
  const balance = typeof source.totalBalance === "number" ? source.totalBalance : -1;
  if (balance < EXPECTED_DEMO_SOURCE.minBalance || balance > EXPECTED_DEMO_SOURCE.maxBalance) {
    mismatches.push(
      `total balance ${balance} outside ${EXPECTED_DEMO_SOURCE.minBalance}–` +
        `${EXPECTED_DEMO_SOURCE.maxBalance}`,
    );
  }
  if ((source.exposures ?? 0) < EXPECTED_DEMO_SOURCE.minExposures) {
    mismatches.push(
      `exposures ${source.exposures} below ${EXPECTED_DEMO_SOURCE.minExposures}`,
    );
  }

  if (mismatches.length > 0) {
    issues.push({
      code: "CONFIG_DEMO_SOURCE_MISMATCH",
      message:
        "demo-pack.json was built from a different portfolio than this build " +
        `is pinned to: ${mismatches.join("; ")}.`,
    });
  }

  return issues;
}

/* -------------------------------------------------------------------------- */
/* Canonical origin                                                           */
/* -------------------------------------------------------------------------- */

export interface SiteUrlResult {
  ok: boolean;
  origin: string | null;
  problem: string | null;
}

/**
 * Validate the canonical origin.
 *
 * Requirements in production: present, https, no path/query/fragment (an
 * origin, not a URL), and not a loopback or private host. Outside production a
 * localhost origin is expected and allowed.
 */
export function validateSiteUrl(
  raw: string,
  { requireProduction }: { requireProduction: boolean },
): SiteUrlResult {
  const value = raw.trim();
  if (!value) {
    return { ok: false, origin: null, problem: "NEXT_PUBLIC_SITE_URL is not set." };
  }

  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return {
      ok: false,
      origin: null,
      problem: "NEXT_PUBLIC_SITE_URL is not a valid absolute URL.",
    };
  }

  if (url.pathname !== "/" && url.pathname !== "") {
    return {
      ok: false,
      origin: null,
      problem: "NEXT_PUBLIC_SITE_URL must be a bare origin — it carries a path.",
    };
  }
  if (url.search || url.hash) {
    return {
      ok: false,
      origin: null,
      problem:
        "NEXT_PUBLIC_SITE_URL must be a bare origin — it carries a query or fragment.",
    };
  }

  const host = url.hostname.toLowerCase();
  const isLoopback =
    host === "localhost" ||
    host === "127.0.0.1" ||
    host === "::1" ||
    host.endsWith(".localhost");

  if (requireProduction) {
    if (url.protocol !== "https:") {
      return {
        ok: false,
        origin: null,
        problem: "NEXT_PUBLIC_SITE_URL must use https in production.",
      };
    }
    if (isLoopback) {
      return {
        ok: false,
        origin: null,
        problem: "NEXT_PUBLIC_SITE_URL must not be a loopback host in production.",
      };
    }
  } else if (url.protocol !== "https:" && url.protocol !== "http:") {
    return {
      ok: false,
      origin: null,
      problem: "NEXT_PUBLIC_SITE_URL must use http or https.",
    };
  }

  return { ok: true, origin: url.origin, problem: null };
}

/* -------------------------------------------------------------------------- */
/* Full validation                                                            */
/* -------------------------------------------------------------------------- */

/** Minimum entropy for the session-signing secret. */
const MIN_SECRET_LENGTH = 32;

export function validateConfig(
  { production = IS_PRODUCTION }: { production?: boolean } = {},
): ConfigIssue[] {
  const issues: ConfigIssue[] = [...checkDemoPack()];

  // --- environment name ---
  if (!["development", "test", "staging", "production"].includes(APPLICATION_ENV)) {
    issues.push({
      code: "CONFIG_ENV_INVALID",
      message:
        `APPLICATION_ENV is "${APPLICATION_ENV}"; expected development, test, ` +
        "staging or production.",
    });
  }

  // --- canonical origin ---
  const site = validateSiteUrl(SITE_URL, { requireProduction: production });
  if (!site.ok) {
    issues.push({ code: "CONFIG_SITE_URL_INVALID", message: site.problem ?? "invalid" });
  }

  if (!production) return issues;

  // ------------------------------------------------------------------ //
  // Production-only requirements                                        //
  // ------------------------------------------------------------------ //

  // --- session secret ---
  if (!DEMO_SESSION_SECRET) {
    issues.push({
      code: "CONFIG_SESSION_SECRET_WEAK",
      message: "DEMO_SESSION_SECRET is not set; demo sessions would be forgeable.",
    });
  } else if (DEMO_SESSION_SECRET.length < MIN_SECRET_LENGTH) {
    issues.push({
      code: "CONFIG_SESSION_SECRET_WEAK",
      message: `DEMO_SESSION_SECRET is shorter than ${MIN_SECRET_LENGTH} characters.`,
    });
  } else if (new Set(DEMO_SESSION_SECRET).size < 8) {
    issues.push({
      code: "CONFIG_SESSION_SECRET_WEAK",
      message: "DEMO_SESSION_SECRET has too little variation to be a real secret.",
    });
  }

  // --- lead delivery ---
  if (LEAD_DELIVERY_PROVIDER === "file" || LEAD_DELIVERY_PROVIDER === "console") {
    issues.push({
      code: "CONFIG_LEAD_PROVIDER_UNSAFE",
      message:
        `LEAD_DELIVERY_PROVIDER is "${LEAD_DELIVERY_PROVIDER}", which is a ` +
        "development-only adapter. Production leads would never reach anyone. " +
        "Set it to email or webhook.",
    });
  } else if (LEAD_DELIVERY_PROVIDER === "email") {
    const missing = [
      !EMAIL_API_KEY && "EMAIL_API_KEY",
      !LEAD_FROM_EMAIL && "LEAD_FROM_EMAIL",
      !LEAD_NOTIFICATION_EMAIL && "LEAD_NOTIFICATION_EMAIL",
    ].filter(Boolean);
    if (missing.length > 0) {
      issues.push({
        code: "CONFIG_LEAD_PROVIDER_INCOMPLETE",
        message: `LEAD_DELIVERY_PROVIDER=email requires ${missing.join(", ")}.`,
      });
    }
  } else if (LEAD_DELIVERY_PROVIDER === "webhook" && !CRM_WEBHOOK_URL) {
    issues.push({
      code: "CONFIG_LEAD_PROVIDER_INCOMPLETE",
      message: "LEAD_DELIVERY_PROVIDER=webhook requires CRM_WEBHOOK_URL.",
    });
  }

  // --- rate limiting ---
  if (!RATE_LIMIT_STORE_URL && !ALLOW_IN_MEMORY_RATE_LIMIT) {
    issues.push({
      code: "CONFIG_RATE_LIMIT_STORE_REQUIRED",
      message:
        "No RATE_LIMIT_STORE_URL is configured. In-process counters are only " +
        "correct on a single instance; set RATE_LIMIT_STORE_URL, or set " +
        "ALLOW_IN_MEMORY_RATE_LIMIT=true to accept single-instance operation " +
        "deliberately.",
    });
  }

  // --- analytics ---
  if (ANALYTICS_PROVIDER === "appinsights" && !APPINSIGHTS_CONNECTION_STRING) {
    issues.push({
      code: "CONFIG_ANALYTICS_INVALID",
      message:
        "NEXT_PUBLIC_ANALYTICS_PROVIDER=appinsights requires " +
        "APPLICATIONINSIGHTS_CONNECTION_STRING.",
    });
  }
  if (!["none", "appinsights", "firstparty"].includes(ANALYTICS_PROVIDER)) {
    issues.push({
      code: "CONFIG_ANALYTICS_INVALID",
      message: `NEXT_PUBLIC_ANALYTICS_PROVIDER "${ANALYTICS_PROVIDER}" is not a known adapter.`,
    });
  }

  return issues;
}

/** Throw on any unsafe production configuration. Used at startup. */
export function assertConfig(options?: { production?: boolean }): void {
  const issues = validateConfig(options);
  if (issues.length === 0) return;
  const lines = issues.map((i) => `  [${i.code}] ${i.message}`).join("\n");
  throw new Error(
    `Trakt landing page: refusing to start with unsafe configuration.\n${lines}`,
  );
}

/** Coarse component states for the readiness endpoint. Never detail. */
export function componentStates(): Record<string, string> {
  const issues = validateConfig();
  const has = (...codes: ConfigCode[]) => issues.some((i) => codes.includes(i.code));

  return {
    demoPack: has(
      "CONFIG_DEMO_PACK_MISSING",
      "CONFIG_DEMO_PACK_VERSION",
      "CONFIG_DEMO_SOURCE_MISMATCH",
    )
      ? "invalid"
      : "ready",
    siteUrl: has("CONFIG_SITE_URL_INVALID") ? "invalid" : "configured",
    session: has("CONFIG_SESSION_SECRET_WEAK") ? "insecure" : "configured",
    leadDelivery: has("CONFIG_LEAD_PROVIDER_UNSAFE", "CONFIG_LEAD_PROVIDER_INCOMPLETE")
      ? "unconfigured"
      : LEAD_DELIVERY_PROVIDER === "file" || LEAD_DELIVERY_PROVIDER === "console"
        ? "development"
        : "configured",
    rateLimit: has("CONFIG_RATE_LIMIT_STORE_REQUIRED")
      ? "unconfigured"
      : RATE_LIMIT_STORE_URL
        ? "shared"
        : "in-memory",
    analytics: has("CONFIG_ANALYTICS_INVALID")
      ? "invalid"
      : ANALYTICS_PROVIDER === "none"
        ? "disabled"
        : "configured",
  };
}
