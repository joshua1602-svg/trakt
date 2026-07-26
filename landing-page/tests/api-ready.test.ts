import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * Readiness.
 *
 * `/api/health` answers "alive"; `/api/ready` answers "correctly configured".
 * The distinction matters operationally: restarting a process fixes the first
 * and never the second, so they must be able to disagree.
 */

const ORIGINAL = { ...process.env };

const HEALTHY_PRODUCTION = {
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

async function loadRoutes(env: Record<string, string | undefined>) {
  vi.resetModules();
  for (const [key, value] of Object.entries(env)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  const ready = await import("@/app/api/ready/route");
  const health = await import("@/app/api/health/route");
  return { readyGet: ready.GET, healthGet: health.GET };
}

beforeEach(() => {
  vi.spyOn(console, "warn").mockImplementation(() => undefined);
});

afterEach(() => {
  process.env = { ...ORIGINAL };
  vi.resetModules();
});

describe("GET /api/ready", () => {
  it("is ready when production configuration is complete", async () => {
    const { readyGet } = await loadRoutes(HEALTHY_PRODUCTION);
    const response = readyGet();
    expect(response.status).toBe(200);

    const body = await response.json();
    expect(body.status).toBe("ready");
    expect(body.issues).toEqual([]);
    expect(body.components.demoPack).toBe("ready");
    expect(body.components.leadDelivery).toBe("configured");
  });

  it("returns 503 when lead delivery would drop leads", async () => {
    const { readyGet } = await loadRoutes({
      ...HEALTHY_PRODUCTION,
      LEAD_DELIVERY_PROVIDER: "file",
    });
    const response = readyGet();
    expect(response.status).toBe(503);

    const body = await response.json();
    expect(body.status).toBe("not_ready");
    expect(body.issues).toContain("CONFIG_LEAD_PROVIDER_UNSAFE");
    expect(body.components.leadDelivery).toBe("unconfigured");
  });

  it("returns 503 when rate limiting would silently under-count", async () => {
    const { readyGet } = await loadRoutes({
      ...HEALTHY_PRODUCTION,
      ALLOW_IN_MEMORY_RATE_LIMIT: "false",
      RATE_LIMIT_STORE_URL: undefined,
    });
    const body = await readyGet().json();
    expect(body.issues).toContain("CONFIG_RATE_LIMIT_STORE_REQUIRED");
  });

  it("reports codes, never messages, paths, addresses or providers", async () => {
    const { readyGet } = await loadRoutes({
      ...HEALTHY_PRODUCTION,
      LEAD_DELIVERY_PROVIDER: "file",
      NEXT_PUBLIC_SITE_URL: "http://localhost:3000",
    });
    const serialised = JSON.stringify(await readyGet().json());

    expect(serialised).toContain("CONFIG_");
    // No detail of any kind.
    expect(serialised).not.toContain("example.com");
    expect(serialised).not.toContain("test-key");
    expect(serialised).not.toContain("localhost");
    expect(serialised).not.toContain("resend");
    expect(serialised).not.toContain("/home/");
    expect(serialised).not.toMatch(/development-only adapter/);
    expect(serialised).not.toMatch(/\.csv|\.json|synthetic_demo/);
  });

  it("names no internal engine or module", async () => {
    const { readyGet } = await loadRoutes(HEALTHY_PRODUCTION);
    const serialised = JSON.stringify(await readyGet().json());
    for (const internal of ["mi_agent", "mi_service", "funded_prep", "adapters", "blob"]) {
      expect(serialised).not.toContain(internal);
    }
  });
});

describe("liveness versus readiness", () => {
  it("stays alive while not being ready", async () => {
    // A misconfigured lead provider is a readiness problem, not a liveness one:
    // restarting the process would not fix it, so health must not fail.
    const { readyGet, healthGet } = await loadRoutes({
      ...HEALTHY_PRODUCTION,
      LEAD_DELIVERY_PROVIDER: "file",
    });
    expect(healthGet().status).toBe(200);
    expect(readyGet().status).toBe(503);
  });

  it("reports the same component states from both endpoints", async () => {
    const { readyGet, healthGet } = await loadRoutes(HEALTHY_PRODUCTION);
    const health = await healthGet().json();
    const ready = await readyGet().json();
    expect(health.leadDelivery).toBe(ready.components.leadDelivery);
    expect(health.demoPack).toBe(ready.components.demoPack);
  });
});
