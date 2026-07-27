import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { resetRateLimits } from "@/lib/rate-limit";

/**
 * Lead delivery behaviour: provider safety, idempotency, retry, timeout and
 * the guarantee that a failure is never reported as a success.
 *
 * Each case re-imports the module under a mutated environment, because
 * `lib/env.ts` reads `process.env` once at load.
 */

const ORIGINAL = { ...process.env };

const LEAD = {
  id: "11111111-2222-3333-4444-555555555555",
  receivedAt: "2026-01-01T00:00:00.000Z",
  name: "Alex Fenn",
  email: "alex@northbridge-credit.co.uk",
  company: "Northbridge Credit",
  role: "Head of Portfolio",
  message: "Equity release book.",
  consent: true as const,
  source: "landing-page",
};

async function loadLeads(env: Record<string, string | undefined>) {
  vi.resetModules();
  for (const [key, value] of Object.entries(env)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  return import("@/lib/leads");
}

const EMAIL_ENV = {
  APPLICATION_ENV: "production",
  LEAD_DELIVERY_PROVIDER: "email",
  EMAIL_API_KEY: "test-key",
  LEAD_FROM_EMAIL: "site@example.com",
  LEAD_NOTIFICATION_EMAIL: "sales@example.com",
};

beforeEach(() => {
  resetRateLimits();
  vi.spyOn(console, "info").mockImplementation(() => undefined);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  process.env = { ...ORIGINAL };
  vi.resetModules();
  vi.unstubAllGlobals();
});

describe("provider safety", () => {
  it.each(["file", "console"])(
    "refuses the %s adapter in production",
    async (provider) => {
      const { deliverLead } = await loadLeads({
        APPLICATION_ENV: "production",
        LEAD_DELIVERY_PROVIDER: provider,
      });
      await expect(deliverLead(LEAD)).rejects.toThrow(/CONFIG_LEAD_PROVIDER_UNSAFE/);
    },
  );

  it("allows the file adapter outside production", async () => {
    const { deliverLead } = await loadLeads({
      APPLICATION_ENV: "development",
      LEAD_DELIVERY_PROVIDER: "console",
    });
    const outcome = await deliverLead(LEAD);
    expect(outcome.delivered).toBe(true);
    expect(outcome.local).toBe(true);
  });
});

describe("email adapter", () => {
  it("sends HTML and plain text, with the enquirer as reply-to", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const { deliverLead } = await loadLeads(EMAIL_ENV);
    const outcome = await deliverLead(LEAD);

    expect(outcome.delivered).toBe(true);
    expect(outcome.duplicate).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(String(init.body));
    expect(body.to).toEqual(["sales@example.com"]);
    expect(body.from).toBe("site@example.com");
    expect(body.reply_to).toBe("alex@northbridge-credit.co.uk");
    expect(body.text).toContain("Northbridge Credit");
    expect(body.html).toContain("<table");
    // The key travels in the header, never in the body.
    expect(String(init.body)).not.toContain("test-key");
    expect((init.headers as Record<string, string>).authorization).toBe("Bearer test-key");
  });

  it("escapes HTML so a lead cannot inject markup into the notification", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const { deliverLead } = await loadLeads(EMAIL_ENV);
    await deliverLead({ ...LEAD, company: "<script>alert(1)</script>" });

    const body = JSON.parse(String((fetchMock.mock.calls[0] as [string, RequestInit])[1].body));
    expect(body.html).not.toContain("<script>");
    expect(body.html).toContain("&lt;script&gt;");
  });

  it("retries once on a transient failure, then succeeds", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 503 })
      .mockResolvedValueOnce({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const { deliverLead } = await loadLeads(EMAIL_ENV);
    const outcome = await deliverLead(LEAD);

    expect(outcome.attempts).toBe(2);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not retry a permanent failure", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 422 });
    vi.stubGlobal("fetch", fetchMock);

    const { deliverLead } = await loadLeads(EMAIL_ENV);
    await expect(deliverLead(LEAD)).rejects.toThrow();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("throws rather than reporting a false success when the provider is down", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("ECONNREFUSED")));

    const { deliverLead } = await loadLeads(EMAIL_ENV);
    await expect(deliverLead(LEAD)).rejects.toThrow(/lead delivery failed/);
  });

  it("never puts a credential in the thrown message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500 }));

    const { deliverLead } = await loadLeads({ ...EMAIL_ENV, EMAIL_API_KEY: "zq7-secret-key" });
    await expect(deliverLead(LEAD)).rejects.toThrow(
      expect.objectContaining({ message: expect.not.stringContaining("zq7-") }),
    );
  });
});

describe("idempotency", () => {
  it("suppresses an identical resubmission inside the window", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const { deliverLead, resetIdempotency } = await loadLeads(EMAIL_ENV);
    resetIdempotency();

    const first = await deliverLead(LEAD, 1_000);
    const second = await deliverLead({ ...LEAD, id: "different-id" }, 2_000);

    expect(first.duplicate).toBe(false);
    expect(second.duplicate).toBe(true);
    // The provider was called exactly once.
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("delivers a genuinely different enquiry", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const { deliverLead, resetIdempotency } = await loadLeads(EMAIL_ENV);
    resetIdempotency();

    await deliverLead(LEAD, 1_000);
    const second = await deliverLead({ ...LEAD, message: "different question" }, 2_000);

    expect(second.duplicate).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("delivers again once the window has passed", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const { deliverLead, resetIdempotency } = await loadLeads(EMAIL_ENV);
    resetIdempotency();

    await deliverLead(LEAD, 1_000);
    const later = await deliverLead(LEAD, 1_000 + 11 * 60_000);

    expect(later.duplicate).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not suppress after a failure, so a retry can succeed", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 400 })
      .mockResolvedValue({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const { deliverLead, resetIdempotency } = await loadLeads(EMAIL_ENV);
    resetIdempotency();

    await expect(deliverLead(LEAD, 1_000)).rejects.toThrow();
    const retry = await deliverLead(LEAD, 2_000);
    expect(retry.duplicate).toBe(false);
    expect(retry.delivered).toBe(true);
  });
});

describe("webhook adapter", () => {
  it("signs the payload and carries an idempotency key", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const { deliverLead, resetIdempotency } = await loadLeads({
      APPLICATION_ENV: "production",
      LEAD_DELIVERY_PROVIDER: "webhook",
      CRM_WEBHOOK_URL: "https://crm.example.com/hook",
      CRM_WEBHOOK_SECRET: "hook-secret",
      EMAIL_API_KEY: undefined,
    });
    resetIdempotency();
    await deliverLead(LEAD);

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(url).toBe("https://crm.example.com/hook");
    expect(headers["x-trakt-signature"]).toMatch(/^[0-9a-f]{64}$/);
    expect(headers["x-trakt-idempotency-key"]).toBe(LEAD.id);
    expect(String(init.body)).not.toContain("hook-secret");
  });
});

describe("attribution in the notification", () => {
  it("is included in the email but never in the response", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const { deliverLead, resetIdempotency } = await loadLeads(EMAIL_ENV);
    resetIdempotency();
    await deliverLead({
      ...LEAD,
      attribution: { utm_source: "linkedin", utm_campaign: "q3-erm", persona: "coo" },
    });

    const body = JSON.parse(String((fetchMock.mock.calls[0] as [string, RequestInit])[1].body));
    expect(body.text).toContain("linkedin");
    expect(body.text).toContain("q3-erm");
    expect(body.html).toContain("coo");
  });
});
