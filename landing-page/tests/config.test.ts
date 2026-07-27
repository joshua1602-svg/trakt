import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { validateSiteUrl } from "@/lib/config";

/**
 * Production configuration validation.
 *
 * `lib/env.ts` reads `process.env` at module load, so the suite re-imports the
 * config module under a mutated environment rather than trying to mutate
 * already-frozen constants. `vi.resetModules()` between cases keeps them
 * independent.
 */

const BASE_PRODUCTION_ENV: Record<string, string> = {
  APPLICATION_ENV: "production",
  NEXT_PUBLIC_SITE_URL: "https://trakt.example.com",
  DEMO_SESSION_SECRET: "a-long-and-varied-production-secret-value-9f2c",
  LEAD_DELIVERY_PROVIDER: "email",
  EMAIL_API_KEY: "test-key",
  LEAD_FROM_EMAIL: "site@example.com",
  LEAD_NOTIFICATION_EMAIL: "sales@example.com",
  ALLOW_IN_MEMORY_RATE_LIMIT: "true",
  NEXT_PUBLIC_ANALYTICS_PROVIDER: "none",
};

const ORIGINAL = { ...process.env };

async function loadConfig(overrides: Record<string, string | undefined> = {}) {
  vi.resetModules();
  for (const key of Object.keys(BASE_PRODUCTION_ENV)) delete process.env[key];
  Object.assign(process.env, BASE_PRODUCTION_ENV);
  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  return import("@/lib/config");
}

beforeEach(() => {
  vi.spyOn(console, "warn").mockImplementation(() => undefined);
});

afterEach(() => {
  process.env = { ...ORIGINAL };
  vi.resetModules();
});

describe("validateSiteUrl", () => {
  it("accepts a bare https origin in production", () => {
    const result = validateSiteUrl("https://trakt.example.com", { requireProduction: true });
    expect(result.ok).toBe(true);
    expect(result.origin).toBe("https://trakt.example.com");
  });

  it.each([
    ["", "not set"],
    ["not-a-url", "valid absolute URL"],
    ["https://trakt.example.com/landing", "path"],
    ["https://trakt.example.com?utm=1", "query or fragment"],
    ["https://trakt.example.com#top", "query or fragment"],
    ["http://trakt.example.com", "https in production"],
    ["https://localhost:3000", "loopback"],
    ["https://127.0.0.1", "loopback"],
  ])("rejects %j in production", (value, expected) => {
    const result = validateSiteUrl(value, { requireProduction: true });
    expect(result.ok).toBe(false);
    expect(result.problem).toContain(expected);
  });

  it("allows localhost outside production", () => {
    expect(validateSiteUrl("http://localhost:3000", { requireProduction: false }).ok).toBe(true);
  });

  it("hard-codes no Trakt hostname", () => {
    for (const host of ["https://trakt.acme.co.uk", "https://www.acme.co.uk", "https://demo.acme.co.uk"]) {
      expect(validateSiteUrl(host, { requireProduction: true }).ok).toBe(true);
    }
  });
});

describe("validateConfig — a well-formed production environment", () => {
  it("passes", async () => {
    const { validateConfig } = await loadConfig();
    expect(validateConfig({ production: true })).toEqual([]);
  });

  it("asserts without throwing", async () => {
    const { assertConfig } = await loadConfig();
    expect(() => assertConfig({ production: true })).not.toThrow();
  });
});

describe("validateConfig — unsafe production settings are errors", () => {
  async function codesFor(overrides: Record<string, string | undefined>) {
    const { validateConfig } = await loadConfig(overrides);
    return validateConfig({ production: true }).map((i) => i.code);
  }

  it("rejects the file lead adapter", async () => {
    expect(await codesFor({ LEAD_DELIVERY_PROVIDER: "file" })).toContain(
      "CONFIG_LEAD_PROVIDER_UNSAFE",
    );
  });

  it("rejects the console lead adapter", async () => {
    expect(await codesFor({ LEAD_DELIVERY_PROVIDER: "console" })).toContain(
      "CONFIG_LEAD_PROVIDER_UNSAFE",
    );
  });

  it("rejects an email adapter missing credentials", async () => {
    expect(await codesFor({ EMAIL_API_KEY: undefined })).toContain(
      "CONFIG_LEAD_PROVIDER_INCOMPLETE",
    );
  });

  it("rejects a webhook adapter with no URL", async () => {
    expect(await codesFor({ LEAD_DELIVERY_PROVIDER: "webhook" })).toContain(
      "CONFIG_LEAD_PROVIDER_INCOMPLETE",
    );
  });

  it("rejects a missing session secret", async () => {
    expect(await codesFor({ DEMO_SESSION_SECRET: undefined })).toContain(
      "CONFIG_SESSION_SECRET_WEAK",
    );
  });

  it("rejects a short session secret", async () => {
    expect(await codesFor({ DEMO_SESSION_SECRET: "short" })).toContain(
      "CONFIG_SESSION_SECRET_WEAK",
    );
  });

  it("rejects a low-entropy session secret", async () => {
    expect(await codesFor({ DEMO_SESSION_SECRET: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" })).toContain(
      "CONFIG_SESSION_SECRET_WEAK",
    );
  });

  it("rejects in-memory rate limiting unless explicitly allowed", async () => {
    expect(
      await codesFor({ ALLOW_IN_MEMORY_RATE_LIMIT: "false", RATE_LIMIT_STORE_URL: undefined }),
    ).toContain("CONFIG_RATE_LIMIT_STORE_REQUIRED");
  });

  it("accepts in-memory rate limiting when consciously overridden", async () => {
    expect(await codesFor({ ALLOW_IN_MEMORY_RATE_LIMIT: "true" })).not.toContain(
      "CONFIG_RATE_LIMIT_STORE_REQUIRED",
    );
  });

  it("accepts a shared store without the override", async () => {
    expect(
      await codesFor({
        ALLOW_IN_MEMORY_RATE_LIMIT: "false",
        RATE_LIMIT_STORE_URL: "https://counter.internal/increment",
      }),
    ).not.toContain("CONFIG_RATE_LIMIT_STORE_REQUIRED");
  });

  it("rejects an invalid canonical origin", async () => {
    expect(await codesFor({ NEXT_PUBLIC_SITE_URL: "http://localhost:3000" })).toContain(
      "CONFIG_SITE_URL_INVALID",
    );
  });

  it("rejects App Insights without a connection string", async () => {
    expect(await codesFor({ NEXT_PUBLIC_ANALYTICS_PROVIDER: "appinsights" })).toContain(
      "CONFIG_ANALYTICS_INVALID",
    );
  });

  it("rejects an unknown environment name", async () => {
    const { validateConfig } = await loadConfig({ APPLICATION_ENV: "prod" });
    expect(validateConfig({ production: true }).map((i) => i.code)).toContain(
      "CONFIG_ENV_INVALID",
    );
  });

  it("throws from assertConfig, naming codes but never secrets", async () => {
    // A distinctive value, so "must not appear" is a real assertion rather
    // than an accidental substring of the message text.
    const secret = "zq7-secret-value-that-must-never-be-echoed";
    const { assertConfig } = await loadConfig({
      LEAD_DELIVERY_PROVIDER: "file",
      DEMO_SESSION_SECRET: secret.slice(0, 12),
      EMAIL_API_KEY: "zq7-api-key-that-must-never-be-echoed",
    });
    try {
      assertConfig({ production: true });
      expect.unreachable("assertConfig should have thrown");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      expect(message).toContain("CONFIG_LEAD_PROVIDER_UNSAFE");
      expect(message).toContain("CONFIG_SESSION_SECRET_WEAK");
      // Names variables, never values.
      expect(message).not.toContain("zq7-");
    }
  });
});

describe("demo-pack identity", () => {
  it("matches the pinned source", async () => {
    const { checkDemoPack } = await loadConfig();
    expect(checkDemoPack()).toEqual([]);
  });

  it("pins the same portfolio the generator pins", async () => {
    const { EXPECTED_DEMO_SOURCE } = await loadConfig();
    const pack = (await import("@data/demo-pack.json")) as unknown as {
      source: { clientId: string; portfolioId: string; totalBalance: number };
    };
    expect(pack.source.clientId).toBe(EXPECTED_DEMO_SOURCE.clientId);
    expect(pack.source.portfolioId).toBe(EXPECTED_DEMO_SOURCE.portfolioId);
    expect(pack.source.totalBalance).toBeGreaterThanOrEqual(EXPECTED_DEMO_SOURCE.minBalance);
    expect(pack.source.totalBalance).toBeLessThanOrEqual(EXPECTED_DEMO_SOURCE.maxBalance);
  });
});

describe("componentStates", () => {
  it("reports coarse states without leaking configuration", async () => {
    const { componentStates } = await loadConfig();
    const states = componentStates();
    expect(states.demoPack).toBe("ready");
    expect(states.leadDelivery).toBe("configured");

    const serialised = JSON.stringify(states);
    expect(serialised).not.toContain("example.com");
    expect(serialised).not.toContain("test-key");
    expect(serialised).not.toContain("resend");
  });

  it("flags an unconfigured lead adapter", async () => {
    const { componentStates } = await loadConfig({ EMAIL_API_KEY: undefined });
    expect(componentStates().leadDelivery).toBe("unconfigured");
  });
});
